"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def create_monotonic_attention_mask(input_ids, pad_token_id):
    """
    Fastest implementation of a monotonic attention mask.
    
    1. Checks for non-padding tokens.
    2. Uses cumsum to 'latch' the attention to 1 once a non-pad is found.
    3. Optimized with dtype=torch.int32 to reduce GPU memory bandwidth usage 
       (saves 50% memory write overhead compared to default int64).
    """
    # .ne() creates a BoolTensor (very small)
    # .cumsum(dtype=torch.int32) accumulates using 32-bit ints instead of 64-bit
    # .gt(0) converts back to boolean/int mask
    
    return input_ids.ne(pad_token_id).cumsum(dim=1, dtype=torch.int32) > 0

def pad_sequence_left(sequences, batch_first=True, padding_value=0.0):
    """
    Pads a list of variable length Tensors with left padding.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in B x T x * if True, or in
            T x B x * otherwise. Default: True.
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor: The padded tensor.
    """
    
    # 1. Get the shape of the sequences
    # Assuming all sequences have the same trailing dimensions
    trailing_dims = sequences[0].size()[1:]
    max_len = max([s.size(0) for s in sequences])
    
    # 2. Allocate the output tensor filled with the padding value
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)

    # 3. Fill in the actual data
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        
        # Calculate the offset for left padding
        # We want the data to end at the last index, so it starts at (max_len - length)
        if batch_first:
            out_tensor[i, max_len - length : ] = tensor
        else:
            out_tensor[max_len - length : , i] = tensor

    return out_tensor

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    sentence_encoder_pad_token_id: int
    num_chunks_for_text: int
    sentence_encoder_padding_side: str = "left"
    sentence_encoder_max_length: int = 2048
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    use_text_world_state_model: bool = False
    use_instruction_sentence_encoder: bool = False
    

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels")) # zip like operation to get input_ids and labels
        # the input ids shall already splited in the previous step when generating the input_ids. 
        if self.use_text_world_state_model:
            # we have the text_world_state key
            text_world_state_input_ids = [instance["text_world_state"] for instance in instances]
            # assume text_world_state_input_ids structure is list of list of token ids 
            # flatten the list of list into list of token ids
            flattened_text_world_state_input_ids = []
            for batch in text_world_state_input_ids:
                assert isinstance(batch, list), f"Invalid text_world_state_input_ids batch type {type(batch)}"
                assert len(batch) == self.num_chunks_for_text, f"Invalid text_world_state_input_ids batch length {len(batch)} != {self.num_chunks_for_text}"
                flattened_text_world_state_input_ids.extend(batch)
                
            text_world_state_input_ids = flattened_text_world_state_input_ids
                
            if self.sentence_encoder_padding_side == "left":
                text_world_state_input_ids = pad_sequence_left(
                    text_world_state_input_ids, batch_first=True, padding_value=self.sentence_encoder_pad_token_id
                )
            else:
                text_world_state_input_ids = pad_sequence(
                    text_world_state_input_ids, batch_first=True, padding_value=self.sentence_encoder_pad_token_id
                )
                
            text_world_state_attention_mask = create_monotonic_attention_mask(
                text_world_state_input_ids, pad_token_id=self.sentence_encoder_pad_token_id
            )
            assert text_world_state_input_ids.shape[0] == len(instances) * self.num_chunks_for_text, f"Invalid text_world_state_input_ids shape {text_world_state_input_ids.shape}, expected first dim {len(instances) * self.num_chunks_for_text}"
            # Note that now the shape of text_world_state_input_ids is (batch_size * num_chunks_for_text, seq_len)
            
        else:
            text_world_state_input_ids = None   
            text_world_state_attention_mask = None
        
        if self.use_instruction_sentence_encoder:
            compressed_input_ids = [instance["compressed_input_ids"] for instance in instances]
            # also assume compressed_input_ids structure is list of list of token ids
            # flatten the list of list into list of token ids
            flattened_compressed_input_ids = []
            for batch in compressed_input_ids:
                assert isinstance(batch, list), f"Invalid compressed_input_ids batch type {type(batch)}"
                assert len(batch) == self.num_chunks_for_text, f"Invalid compressed_input_ids batch length {len(batch)} != {self.num_chunks_for_text}"
                flattened_compressed_input_ids.extend(batch)
            compressed_input_ids = flattened_compressed_input_ids
            if self.sentence_encoder_padding_side == "left":
                compressed_input_ids = pad_sequence_left(
                    compressed_input_ids, batch_first=True, padding_value=self.sentence_encoder_pad_token_id
                )
            else:
                compressed_input_ids = pad_sequence(
                    compressed_input_ids, batch_first=True, padding_value=self.sentence_encoder_pad_token_id
                )

            compressed_attention_mask = create_monotonic_attention_mask(
                compressed_input_ids, pad_token_id=self.sentence_encoder_pad_token_id
            )
            assert compressed_input_ids.shape[0] == len(instances) * self.num_chunks_for_text, f"Invalid compressed_input_ids shape {compressed_input_ids.shape}, expected first dim {len(instances) * self.num_chunks_for_text}"
        else:
            compressed_input_ids = None
            compressed_attention_mask = None
            
        if self.use_text_world_state_model:
            pixel_values = None
        else:
            pixel_values = [instance["pixel_values"] for instance in instances]
            
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        assert self.sentence_encoder_padding_side == "left", f"Invalid Sentence Encoder Tokenizer `{self.sentence_encoder_padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        # assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if pixel_values is not None:
            if isinstance(pixel_values[0], torch.Tensor):
                if "pixel_values_wrist" in instances[0]:
                    pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                    pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
                else:
                    pixel_values = torch.stack(pixel_values)
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        output = dict(
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if pixel_values is not None:
            output["pixel_values"] = pixel_values
        if text_world_state_input_ids is not None:
            output["text_world_state_input_ids"] = text_world_state_input_ids
            output["text_world_state_attention_mask"] = text_world_state_attention_mask
            # note the shape of text_world_state_input_ids is (batch_size * num_chunks_for_text, seq_len)
        if compressed_input_ids is not None:
            output["compressed_input_ids"] = compressed_input_ids
            output["compressed_attention_mask"] = compressed_attention_mask
            # note the shape of compressed_input_ids is (batch_size * num_chunks_for_text, seq_len)
        return output
