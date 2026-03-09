"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


def group_string_into_n_strings(raw_string: str, raw_split_func, N: int,
                                join_keyword: str = "\n") -> list[str]:
    """
    Groups a list of K string elements into N chunks.

    Each chunk is returned as a single string where elements are joined by '\n'.
    If N > K, the extra groups are represented by an empty string ("").

    Args:
        elements: The list of K string elements (e.g., words, sentences).
        N: The desired number of groups.

    Returns:
        A list of N strings, where each string is a chunk of elements.
    """
    elements = raw_split_func(raw_string)

    K = len(elements)
    
    if N <= 0:
        raise ValueError("N (the number of groups) must be a positive integer.")

    chunks = []

    if N <= K:
        # Case 1: N <= K (Normal, even distribution)
        k, m = divmod(K, N)
        current_index = 0
        
        for i in range(N):
            chunk_size = k + 1 if i < m else k
            
            # Slice the list of elements
            element_slice = elements[current_index : current_index + chunk_size]

            # Join the elements with the specified join keyword to form the final chunk string
            chunk_string = join_keyword.join(element_slice)
            # strip whitespace
            chunk_string = chunk_string.strip()
            chunks.append(chunk_string)
            
            current_index += chunk_size
            
    else:
        # Case 2: N > K (More groups than elements)
        # The first K groups get one element each.
        for i in range(K):
            # Join the single element into a string
            chunk_string = elements[i] 
            chunks.append(chunk_string)

        # The remaining N - K groups get an empty string
        empty_chunks_needed = N - K
        for _ in range(empty_chunks_needed):
            chunks.append("")
            
    return chunks


# ! this is used for text world state 
def split_by_double_newline(raw_string: str) -> list[str]:
    return raw_string.split("\n\n")

# ! this is used for language instruction
def split_by_please_token(raw_string: str) -> list[str]:
    return raw_string.split(", please")

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    vision_backbone_sentence_encoder_tokenizer: PreTrainedTokenizerBase
    language_instruction_sentence_encoder_tokenizer: PreTrainedTokenizerBase
    num_chunks_for_text: int
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # ! the rlds batch is set in prismatic/vla/datasets/rlds/dataset.py line 454 make_interleaved_dataset function
        # ! and it goes to line 39 make_dataset_from_rlds
        
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        if self.vision_backbone_sentence_encoder_tokenizer is not None:
            # need to let the RLDS dataset generation process to consider the text world state key
            # TODO xxxname backlog: text_world_state this should be handled in the RLDS dataset generation process
            img = rlds_batch['task']['text_world_state'].decode().lower()  # no image for text-based world state model
            state_collection = img.split('\n\n')
            img = '\n\n'.join(state_collection[:-1])
            position_info_str = state_collection[-1]
            # consider the chunk thing
            img = group_string_into_n_strings(
                raw_string = img, 
                raw_split_func = split_by_double_newline, 
                N = self.num_chunks_for_text, 
            )
        else:
            img = Image.fromarray(rlds_batch["observation"]["image_primary"][0]) 
            position_info_str = None
            
        if self.language_instruction_sentence_encoder_tokenizer is not None:
            lang = "achieve the mission?"
            compressed_lang = rlds_batch["task"]["language_instruction"].decode().lower()
            # consider the chunk thing
            compressed_lang = group_string_into_n_strings(
                raw_string = compressed_lang, 
                raw_split_func = split_by_please_token, 
                N = self.num_chunks_for_text, 
            )
            
        else:
            lang = rlds_batch["task"]["language_instruction"].decode().lower() + "?"
            compressed_lang = None
            
        if position_info_str is not None:
            lang = f"{lang}\n{position_info_str}"
        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        
        compressed_input_ids = None
        if compressed_lang is not None:
            compressed_input_dict = self.language_instruction_sentence_encoder_tokenizer(
                compressed_lang,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            compressed_input_ids = [compressed_input_dict.input_ids[i] for i in range(compressed_input_dict.input_ids.shape[0])]
            # shape is (num_chunks_for_text, seq_len)
            

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # print(f'Input IDs shape: {input_ids.shape}') # Input IDs shape: torch.Size([168])
        # print(f'ndim of input IDs: {input_ids.ndim}') # ndim of input IDs: 1
        
        if self.vision_backbone_sentence_encoder_tokenizer is None:
            pixel_values = self.image_transform(img)
            text_world_state = None
        else:
            pixel_values = None  # no image for text-based world state model
            text_world_state = img  # here img is actually text for text-based world state model
            # Tokenize text world state
            text_world_state_dict = self.vision_backbone_sentence_encoder_tokenizer(
                text_world_state,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            text_world_state = [text_world_state_dict.input_ids[i] for i in range(text_world_state_dict.input_ids.shape[0])]
            # shape is (num_chunks_for_text, seq_len)
            

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
            
        # ! xxxname CHECK: please check the transfrom from original RLDS batch to final return dict

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, 
                           actions=actions,
                           compressed_input_ids=compressed_input_ids,
                           text_world_state=text_world_state,
                           )

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        load_text_based_world_state: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist") # Apply to overcooked and mini_behavior too

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            load_text_based_world_state=load_text_based_world_state,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
