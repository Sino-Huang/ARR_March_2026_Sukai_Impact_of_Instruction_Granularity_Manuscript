import torch
from torch import nn as nn
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding

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

# Text-based World State Model Setup

class TextWorldStatePrismaticVisionBackbone(nn.Module):
    # try to copy FiLMedPrismaticVisionBackbone structure in `prismatic/models/film_vit_wrapper.py`
    def __init__(self, num_chunks: int = 12, backbone_id = "Qwen/Qwen3-Embedding-0.6B"):
        super().__init__()
        self.backbone_id = backbone_id
        self.vision_backbone = AutoModel.from_pretrained(backbone_id)
        if "Qwen3-Embedding-4B" in backbone_id:
            self.output_dim = 2560
        elif "Qwen3-Embedding-7B" in backbone_id:
            self.output_dim = 4096
        elif "Qwen3-Embedding-0.6B" in backbone_id:
            self.output_dim = 1024
            
        self.num_patches = num_chunks

    def forward(self, x):
        # assert the x type is BatchEncoding
        if not isinstance(x, BatchEncoding):
            assert isinstance(x, dict), f'Expected input type BatchEncoding or dict but got {type(x)}'
            assert 'input_ids' in x, f'Expected input to have key "input_ids" but got {x.keys()}'
            assert 'attention_mask' in x, f'Expected input to have key "attention_mask" but got {x.keys()}'
        # also assert the rank of input ids is 2
        assert x['input_ids'].ndim == 2, f'Expected input_ids to have rank 2 but got {x["input_ids"].ndim}'
        outputs = self.vision_backbone(**x)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, x['attention_mask']
        )
        return embeddings # Shape  (batch_size, hid)
    
    def get_num_patches(self) -> int:
        return self.num_patches
    
    def get_num_images_in_input(self) -> int:
        return 1 # since this is a text-based model, we consider it as 1 "image" input
    
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    def debug(self):
        # A debug method to inspect intermediate outputs
        tokenizer = AutoTokenizer.from_pretrained(self.backbone_id, padding_side='left')
        # put both model and tokenizer to the cuda 
        sample_text = "This is a sample instruction."
        batch_dict = tokenizer(
            [sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        batch_dict.to(self.vision_backbone.device)
        outputs = self.vision_backbone(**batch_dict)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, batch_dict.attention_mask
        )
        print(f'Mask look like: {batch_dict.attention_mask}')
        print("Embeddings shape:", embeddings.shape)
        
        
        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery:{query}'

        # Each query must come with a one-sentence instruction that describes the task
        task = 'Given a web search query, retrieve relevant passages that answer the query'

        queries = [
            get_detailed_instruct(task, 'What is the capital of China?'),
            get_detailed_instruct(task, 'Explain gravity')
        ]
        
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        ]
        batch_dict = tokenizer(
            queries + documents,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )


        batch_dict.to(self.vision_backbone.device)
        # print batch_dict shape
        print(f"Type of batch_dict: {type(batch_dict)}") # Type of batch_dict: <class 'transformers.tokenization_utils_base.BatchEncoding'>
        print(f'Input IDs shape: {batch_dict["input_ids"].shape}') # Input IDs shape: torch.Size([4, 31])
        print(f'ndim of input IDs: {batch_dict["input_ids"].ndim}') # ndim of input IDs: 2
        outputs = self.vision_backbone(**batch_dict)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, batch_dict.attention_mask
        )
        # print embeddings shape
        print("Embeddings shape for queries and documents:", embeddings.shape) # torch.Size([4, 2560])
        # normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:len(queries)] @ embeddings[len(queries):].T) * 100.0
        print("Similarity scores shape:", scores.shape)
        print("Similarity scores:", scores)
        
        
        # iterate and check the way in the training loop
        print("==== Now testing with padding sequence ====")
        total_sentences = queries + documents
        input_ids_list = []
        for i in range(len(total_sentences)):
            single_input = tokenizer(
                total_sentences[i],
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            input_ids = single_input.input_ids[0]
            input_ids_list.append(input_ids)
            
        print(f"Pad value is {tokenizer.pad_token_id}")
        input_ids_padded = pad_sequence_left(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        input_ids_padded = input_ids_padded.to(self.vision_backbone.device)
        
        attention_mask = create_monotonic_attention_mask(input_ids_padded, pad_token_id=tokenizer.pad_token_id)
        original_attention_mask = batch_dict.attention_mask
        
        
        print(f'\n\n input_ids_padded: {input_ids_padded}')
        print(f'\n\n original input_ids: {batch_dict["input_ids"]}')
        
        # decode to check correctness
        decoded_texts = tokenizer.batch_decode(input_ids_padded, skip_special_tokens=True)
        print(f"\n\n Decoded texts after padding: {decoded_texts}")
        
        original_decoded_texts = tokenizer.batch_decode(batch_dict["input_ids"], skip_special_tokens=True)
        print(f"\n\n Original decoded texts: {original_decoded_texts}")
        
        print(f"Attention mask look like after padding: {attention_mask}") # 1 means valid token, 0 means padding token
        print(f"Original Attention mask look like: {original_attention_mask}")
        outputs = self.vision_backbone(input_ids=input_ids_padded, attention_mask=attention_mask)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, attention_mask
        )
        print("Embeddings shape after padding:", embeddings.shape) 
        # normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:len(queries)] @ embeddings[len(queries):].T) * 100.0
        print("Similarity scores shape:", scores.shape)
        print("Similarity scores:", scores)
        
    
class InstructionSentenceEncoder(nn.Module):
    def __init__(self, num_chunks: int = 1, backbone_id = "Qwen/Qwen3-Embedding-0.6B", projector_output_dim: int = 4096):
        super().__init__()
        self.backbone_id = backbone_id
        self.sentence_encoder = AutoModel.from_pretrained(backbone_id)
        if "Qwen3-Embedding-4B" in backbone_id:
            self.output_dim = 2560
        elif "Qwen3-Embedding-7B" in backbone_id:
            self.output_dim = 4096
        elif "Qwen3-Embedding-0.6B" in backbone_id:
            self.output_dim = 1024
            
        # set up a light projector inside
        self.light_projector = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, bias=True),
            nn.Linear(self.output_dim, projector_output_dim, bias=True),
            nn.GELU(),
        )
            
        self.num_patches = num_chunks

    def forward(self, x):
        # assert the x type is BatchEncoding
        if not isinstance(x, BatchEncoding):
            assert isinstance(x, dict), f'Expected input type BatchEncoding or dict but got {type(x)}'
            assert 'input_ids' in x, f'Expected input to have key "input_ids" but got {x.keys()}'
            assert 'attention_mask' in x, f'Expected input to have key "attention_mask" but got {x.keys()}'
        # also assert the rank of input ids is 2
        assert x['input_ids'].ndim == 2, f'Expected input_ids to have rank 2 but got {x["input_ids"].ndim}'
        outputs = self.sentence_encoder(**x)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, x['attention_mask']
        )
        embeddings = self.light_projector(embeddings)
        return embeddings # Shape  (batch_size, hid)
    
    def get_num_patches(self) -> int:
        return self.num_patches
    
    def get_num_images_in_input(self) -> int:
        return 1 # since this is a text-based model, we consider it as 1 "image" input
    
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    
    
# This model shall mimic the PrismaticForConditionalGeneration class in /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_modules/openvla-oft/prismatic/extern/hf/modeling_prismatic.py


if __name__ == "__main__":
    model = TextWorldStatePrismaticVisionBackbone()
    model.to('cuda:0')
    model.debug()
    
    # RUN COMMAND
    # CUDA_VISIBLE_DEVICES=0 python prismatic/models/backbones/llm/text_world_state_encoder.py