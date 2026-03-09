import argparse
import itertools
import pathlib 
import os 
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from copy import deepcopy
import draccus
import numpy as np
import tqdm
import wandb
import cv2
import re
import tensorflow_datasets as tfds
import json 
import pickle 
from glob import glob
from loguru import logger
import torch 
from time import sleep, time
from loguru import logger
from dataclasses import dataclass
import draccus
import subprocess
from eval_results_dataclass import EvalResult, DatabaseHelper
REMOVE_DIRS_AFTER_EVAL = True
EVAL_ROUND_ID = "DEC-RUSH-B"
USE_WANDB = True
SAVE_VIDEO = 0.01 # save video for 1% of the episodes
DATABASE_HELPER = DatabaseHelper()
DATABASE_HELPER.create_table()

MODULE_BASE_DIR = str(Path(__file__).parent.parent)
sys.path.insert(0, MODULE_BASE_DIR)

MODEL_SAVE_DIR = "/data/scratch/projects/xxxplace2219/xxxname_storage/openvla-oft" # ! UPDATE it if storage is limited

IMAGE_RESOLUTION = 256
RLDS_DATA_DIR_xxxorgan="/data/gpfs/projects/xxxplace2219/xxxname_temp_storage/single_sentence_modified_nsai_rlds"
RLDS_DATA_DIR_xxxorganx="/nfsdata/data/xxxnameh/SimpleVLA-RL/data/01_raw/single_sentence_modified_nsai_rlds"  # Change this if needed

if os.path.exists(RLDS_DATA_DIR_xxxorgan):
    RLDS_DATA_DIR = RLDS_DATA_DIR_xxxorgan
elif os.path.exists(RLDS_DATA_DIR_xxxorganx):
    RLDS_DATA_DIR = RLDS_DATA_DIR_xxxorganx
else:
    raise ValueError("RLDS data directory not found. Please check the RLDS_DATA_DIR paths.")


from experiments.robot.mini_behavior.mini_behavior_utils import (
    get_mini_behavior_env,
    save_rollout_video,
    ACTION_PARSING_DICT,
)

from mini_behavior.utils.policy_sketch.save_n_load_env import (
    get_mini_behavior_image,
)

from mini_behavior.utils.policy_sketch.dynamic_instruction_selector import InstructorSelector

from experiments.robot.openvla_utils import (
    get_action_head,
    get_instruction_sentence_encoder,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class TaskSuite(str, Enum): # ! these info was stored in rlds_dataset_builder
    TEST_SINGLE_INSTR_ID_GRANULARITY_L = 'test_single_instr_id_granularity_L'
    TEST_SINGLE_INSTR_ID_GRANULARITY_M = 'test_single_instr_id_granularity_M'
    TEST_SINGLE_INSTR_ID_GRANULARITY_H = 'test_single_instr_id_granularity_H'
    TEST_SINGLE_INSTR_TASK_OOD_GRANULARITY_L = 'test_single_instr_task_ood_granularity_L'
    TEST_SINGLE_INSTR_TASK_OOD_GRANULARITY_M = 'test_single_instr_task_ood_granularity_M'
    TEST_SINGLE_INSTR_TASK_OOD_GRANULARITY_H = 'test_single_instr_task_ood_granularity_H'
    TEST_SINGLE_INSTR_LANG_OOD = 'test_single_instr_lang_ood'
    TEST_SINGLE_INSTR_NO_LANG_OOD = 'test_single_instr_no_lang_ood'
    
# longest episode length in the testing env is 261, we relax the max steps to 300 / 400 to allow some buffer

TASK_MAX_STEPS = {
    TaskSuite.TEST_SINGLE_INSTR_ID_GRANULARITY_L: 330,
    TaskSuite.TEST_SINGLE_INSTR_ID_GRANULARITY_M: 330,
    TaskSuite.TEST_SINGLE_INSTR_ID_GRANULARITY_H: 330,
    TaskSuite.TEST_SINGLE_INSTR_TASK_OOD_GRANULARITY_L: 330,
    TaskSuite.TEST_SINGLE_INSTR_TASK_OOD_GRANULARITY_M: 330,
    TaskSuite.TEST_SINGLE_INSTR_TASK_OOD_GRANULARITY_H: 330,
    TaskSuite.TEST_SINGLE_INSTR_LANG_OOD: 330,
    TaskSuite.TEST_SINGLE_INSTR_NO_LANG_OOD: 330,
}

# Task laying_wood_floors: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task preparing_salad: [0, 1, 2, 3, 4] -> L:[0] M:[1, 2] H:[3, 4]
# Task cleaning_up_the_kitchen_only: [0, 1, 2, 3, 4, 5] -> L:[0, 1] M:[2, 3] H:[4, 5]
# Task organizing_file_cabinet: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task thawing_frozen_food: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task making_tea: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task opening_packages: [0, 1] -> L:[0] M:[0] H:[1]
# Task boxing_books_up_for_storage: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task collect_misplaced_items: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task putting_away_dishes_after_cleaning: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task washing_pots_and_pans: [0, 1, 2, 3, 4] -> L:[0] M:[1, 2] H:[3, 4]
# Task cleaning_shoes: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task installing_a_printer: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task setting_up_candles: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task watering_houseplants: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task cleaning_a_car: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task storing_food: [0, 1, 2, 3] -> L:[0] M:[1] H:[2, 3]
# Task throwing_away_leftovers: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task moving_boxes_to_storage: [0, 1, 2] -> L:[0] M:[1] H:[2]
# Task sorting_books: [0, 1, 2] -> L:[0] M:[1] H:[2]

UNORMED_KEY_DOMAIN_WIDTH_MAPPING = {
    "laying_wood_floors" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "preparing_salad" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_M',
        3: 'mini_behavior_train_single_instr_id_granularity_H',
        4: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "cleaning_up_the_kitchen_only" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_L',
        2: 'mini_behavior_train_single_instr_id_granularity_M',
        3: 'mini_behavior_train_single_instr_id_granularity_M',
        4: 'mini_behavior_train_single_instr_id_granularity_H',
        5: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "organizing_file_cabinet" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
        3: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "thawing_frozen_food" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "making_tea" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
        3: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "opening_packages" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "boxing_books_up_for_storage" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "collect_misplaced_items" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "putting_away_dishes_after_cleaning" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
        3: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "washing_pots_and_pans" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_M',
        3: 'mini_behavior_train_single_instr_id_granularity_H',
        4: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "cleaning_shoes" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
        3: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "installing_a_printer" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "setting_up_candles" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "watering_houseplants" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
        3: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "cleaning_a_car" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
        3: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "storing_food" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
        3: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "throwing_away_leftovers" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "moving_boxes_to_storage" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
    },
    "sorting_books" : {
        0: 'mini_behavior_train_single_instr_id_granularity_L',
        1: 'mini_behavior_train_single_instr_id_granularity_M',
        2: 'mini_behavior_train_single_instr_id_granularity_H',
    }
}


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    use_discrete_diffusion: bool = False             # If True, uses discrete diffusion model for action generation

    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    use_text_world_state_model: bool = False        # If True, uses Text-based World State Model to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                         # Whether to include proprio state in input
    use_instruction_sentence_encoder: bool = False  # Whether to use instruction sentence encoder for instruction encoding
    
    num_chunks_for_text: int = 1                     # Number of text chunks to use when instruction sentence encoder or text world state model is used
    center_crop: bool = False                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 6                     # Number of actions to execute open-loop before requerying policy # check modules/openvla-oft/prismatic/vla/constants.py

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Mini_behavior environment-specific parameters
    #################################################################################################################
    task_suite_name: str = None
    num_trials_per_task: int = 2                    # Number of rollouts per task
    env_img_res: int = IMAGE_RESOLUTION                           # Resolution for environment images (not policy input resolution)
    dataset_name: str = "mini_behavior"               # Name of the RLDS dataset to use (default: "mini_behavior")

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = "parallel_dec--6_acts_chunk--L1_regression"                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = True                          # Whether to also log results in Weights & Biases
    wandb_entity: str = ""          # Name of WandB entity
    wandb_project: str = "OpenVLA-OFT-Mini_behavior-Eval"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    
def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    if cfg.use_discrete_diffusion:
        assert not cfg.use_l1_regression, "Cannot use L1 regression with discrete diffusion!"
        assert not cfg.use_diffusion, "Cannot use continuous diffusion with discrete diffusion!"
    model = get_model(cfg) 
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for Mini behavior 
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)
        
    # load instruction_sentence_encoder 
    if cfg.use_instruction_sentence_encoder:
        instruction_sentence_encoder = get_instruction_sentence_encoder(cfg, model.llm_dim)
        # to cuda 
        instruction_sentence_encoder = instruction_sentence_encoder.to(torch.bfloat16).to(DEVICE)
    else:
        instruction_sentence_encoder = None

    # Get OpenVLA processor if needed
    processor = None
    sentence_encoder_pad_token_id = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        if cfg.use_instruction_sentence_encoder or cfg.use_text_world_state_model:
            sentence_encoder_pad_token_id = processor.sentence_encoder_tokenizer.pad_token_id
            
    # sentence_encoder_tokenizer
    vision_backbone_sentence_encoder_tokenizer = None
    language_instruction_sentence_encoder_tokenizer = None
    if cfg.use_instruction_sentence_encoder:
        language_instruction_sentence_encoder_tokenizer = processor.sentence_encoder_tokenizer
    
    if cfg.use_text_world_state_model:
        vision_backbone_sentence_encoder_tokenizer = processor.sentence_encoder_tokenizer

    return model, action_head, proprio_projector, noisy_action_projector, processor, sentence_encoder_pad_token_id, instruction_sentence_encoder, vision_backbone_sentence_encoder_tokenizer, language_instruction_sentence_encoder_tokenizer

def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = cfg.run_id_note
    
    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id

def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

def prepare_observation(obs, env, resize_size): 
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_mini_behavior_image(obs, env, add_noise=True, image_size=IMAGE_RESOLUTION)

    # rescale using opencv2
    if img.shape[0] != IMAGE_RESOLUTION:
        img = cv2.resize(img, (IMAGE_RESOLUTION, IMAGE_RESOLUTION), interpolation=cv2.INTER_AREA)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": None,
        "state": None,
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, env, model_family, deterministic=False): 
    """Process action before sending to environment. The model outputs action in one-hot format for OpenVLA.
    The action from the VLA model will be converted to the actual action object used in the environment.
    This function handles that conversion.
    """
 
    # assume action type is np.ndarray
    if model_family == "openvla":
        # action is one-hot, get the idx with max value
        if deterministic:
            action_id = np.argmax(action, axis=-1)
        else:
            action_probs = np.exp(action) / np.sum(np.exp(action))
            action_id = np.random.choice(len(action_probs), p=action_probs)
        action_id = int(action_id)
        # get the key from ACTION_PARSING_DICT whose value is equal to action_id
        action_key = None 
        for key, value in ACTION_PARSING_DICT.items():
            if value == action_id:
                action_key = key
                break
        assert action_key is not None, f"Action ID {action_id} not found in ACTION_PARSING_DICT!"
        
        # get the actual action object from env 
        action_objs = env.actions
        for action_obj in action_objs:
            if action_obj.name == action_key:
                return action_obj
            
    raise ValueError(f"Action ID {action_id} not found in environment actions!")


def run_episode( 
    cfg: GenerateConfig,
    env,
    instruction_selector,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    problem_id=None,
    log_file=None,
    deterministic=False,
    sentence_encoder_pad_token_id = None,
    instruction_sentence_encoder = None,
    vision_backbone_sentence_encoder_tokenizer = None,
    language_instruction_sentence_encoder_tokenizer = None,
    
):
    """Run a single episode in the environment.
    The environment is obtained outside this function
    """
    # reset the environment
    # make the selector as the input here
    env.seed(int(problem_id))
    obs = env.reset() # ! some envs may require reset seed
    instruction_selector.reset(env)

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    replay_instructions = []
    final_instruction = ""
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    dynamic_instruction = ""
    # Run episode
    success = False
    try:
        while t < max_steps:

            # Prepare observation
            observation, img = prepare_observation(obs, env, resize_size)
            replay_images.append(img)

            if cfg.task_suite_name == TaskSuite.TEST_SINGLE_INSTR_NO_LANG_OOD:
                dynamic_instruction = "Empty instruction."
            elif cfg.task_suite_name == TaskSuite.TEST_SINGLE_INSTR_LANG_OOD:
                dynamic_instruction = task_description
            else:
                dynamic_instruction = instruction_selector.select_instruction()
                final_instruction = dynamic_instruction
            replay_instructions.append(dynamic_instruction)
            
            if cfg.use_text_world_state_model:
                text_world_state_label = instruction_selector.generate_text_world_state_str()
            else:
                text_world_state_label = None


            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    dynamic_instruction,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_text_world_state_model=cfg.use_text_world_state_model,
                    use_instruction_sentence_encoder=cfg.use_instruction_sentence_encoder,
                    num_chunks_for_text=cfg.num_chunks_for_text,
                    sentence_encoder_pad_token_id=sentence_encoder_pad_token_id,
                    instruction_sentence_encoder=instruction_sentence_encoder,
                    vision_backbone_sentence_encoder_tokenizer=vision_backbone_sentence_encoder_tokenizer,
                    language_instruction_sentence_encoder_tokenizer=language_instruction_sentence_encoder_tokenizer,
                    text_world_state_label=text_world_state_label,
                    use_discrete_diffusion =cfg.use_discrete_diffusion,
                )
                for i in range(action_queue.maxlen):
                    action_queue.append(actions[i])
                    

            # Get action from queue
            action = action_queue.popleft()
            

            # Process action
            action = process_action(action, env, cfg.model_family, deterministic)

            # Execute action in environment
            obs, reward, done, info = env.step(action)
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        raise e
        instruction_selector.close()

    instruction_selector.close()
    # get the num step from final_instruction
    match = re.search(r'At step (?P<step_num>\d+):', final_instruction)
    if match:
        instruction_steps = int(match.group('step_num'))
    else:
        instruction_steps = -1
    return success, replay_images, replay_instructions, instruction_steps

def run_task(
    cfg: GenerateConfig,
    domain_name,
    problem_id,
    instruction_width,
    instruction_domain_name,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    sentence_encoder_pad_token_id = None,
    instruction_sentence_encoder = None,
    vision_backbone_sentence_encoder_tokenizer = None,
    language_instruction_sentence_encoder_tokenizer = None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    domain_success_metrics=None,
    eval_result_skeleton: EvalResult = None,
    train_data_setup: str = "",
    save_video=SAVE_VIDEO,
):
    """Run evaluation for a single task."""
    
    assert domain_success_metrics is not None, "domain_success_metrics must not be None!"

    if domain_name not in domain_success_metrics:
        domain_success_metrics[domain_name] = {
            "total_episodes": 0,
            "total_successes": 0,
            "success_rate": 0.0,
        }
    if 'overall' not in domain_success_metrics:
        domain_success_metrics['overall'] = {
            "total_episodes": 0,
            "total_successes": 0,
            "success_rate": 0.0,
        }

    # Initialize environment and get task description
    env, task_description, _, _, _, _ = get_mini_behavior_env(
        domain_name=domain_name,
        problem_id=problem_id,
    )
    
    # Get unnormalized key and update cfg 
    if "pure" not in train_data_setup:
        unnorm_key = UNORMED_KEY_DOMAIN_WIDTH_MAPPING[domain_name][instruction_width]
    else:
        last_char = train_data_setup[-1]
        unnorm_key = f'mini_behavior_train_single_instr_id_granularity_{last_char}'
    cfg.unnorm_key = unnorm_key
    
    
    # init the selector 
    instruction_selector = InstructorSelector(instruction_width, instruction_domain_name)

    # Start episodes
    task_episodes, task_successes = 0, 0
    final_success = False
    instruction_steps = -1
    assert eval_result_skeleton is not None, "eval_result_skeleton must not be None!"
    
    eval_result_for_episode = deepcopy(eval_result_skeleton)
    eval_result_for_episode.problem_id = problem_id
    eval_result_for_episode.instruction_width = instruction_width
    eval_result_for_episode.domain_name = domain_name
    eval_result_for_episode.eval_id = f'{int(time() * 1000000)}'
    
    # check if this eval_result already exists in the database
    # ! interrupting the eval and rerunning will not duplicate the results
    if DATABASE_HELPER.check_if_exists(eval_result_for_episode):
        log_message(f"Eval result for problem_id {problem_id}, instruction_width {instruction_width}, domain_name {domain_name} already exists in database. Skipping...", log_file)
        return total_episodes, total_successes
    
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        log_message(f"Starting episode {episode_idx + 1} / {cfg.num_trials_per_task}...", log_file)
        
        if env is None:
            env, _, _, _, _, _ = get_mini_behavior_env(
                domain_name=domain_name,
                problem_id=problem_id,
            )
        if episode_idx == cfg.num_trials_per_task -1: # last episode 
            deterministic = True 
        else:
            deterministic = False
            
        # Run episode
        success, replay_images, replay_instructions, instruction_steps = run_episode(
            cfg,
            env,
            instruction_selector,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            problem_id,
            log_file,
            deterministic,
            sentence_encoder_pad_token_id,
            instruction_sentence_encoder,
            vision_backbone_sentence_encoder_tokenizer,
            language_instruction_sentence_encoder_tokenizer,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1
            final_success = True

        # Save replay video
        if save_video:
            if isinstance(save_video, float):
                import random

                if random.random() > save_video:
                    continue  # Skip saving video based on probability
            try:
                save_rollout_video(
                    replay_images, total_episodes, success=success, task_description=replay_instructions, log_file=log_file, notes='single_sent_eval', domain_name=domain_name,
                )
            except Exception as e:
                log_message(f"Error saving rollout video: {e}", log_file)
        

        # Log results
        log_message(f"Success: {success}", log_file)
        logger.success(f"Success: {success}")
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        logger.info(f"# episodes tried so far: {episode_idx + 1} / {cfg.num_trials_per_task}")
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
        
        env = None 
        
        if final_success:
            break

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # task success rate and task_successes and task_episodes to domain_success_metrics
    domain_success_metrics[domain_name]["total_episodes"] += 1
    domain_success_metrics['overall']["total_episodes"] += 1
    if final_success:
        domain_success_metrics[domain_name]["total_successes"] += 1
        domain_success_metrics['overall']["total_successes"] += 1

    domain_success_metrics[domain_name]["success_rate"] = float(domain_success_metrics[domain_name]["total_successes"]) / float(domain_success_metrics[domain_name]["total_episodes"]) if domain_success_metrics[domain_name]["total_episodes"] > 0 else 0.0
    domain_success_metrics['overall']["success_rate"] = float(domain_success_metrics['overall']["total_successes"]) / float(domain_success_metrics['overall']["total_episodes"]) if domain_success_metrics['overall']["total_episodes"] > 0 else 0.0

    # wandb_logs = {
    #             f"success_rate/{task_description}": task_success_rate,
    #             f"num_episodes/{task_description}": task_episodes,
    #         }
    wandb_logs = dict()
    for d_name in domain_success_metrics.keys():
        wandb_logs[f"num_episodes/{d_name}"] = domain_success_metrics[d_name]["total_episodes"]
        wandb_logs[f"num_successes/{d_name}"] = domain_success_metrics[d_name]["total_successes"]
        wandb_logs[f"success_rate/{d_name}"] = domain_success_metrics[d_name]["success_rate"]
        wandb_logs[f"overall_success_rate"] = domain_success_metrics['overall']["success_rate"]
        wandb_logs[f"overall_successes"] = domain_success_metrics['overall']["total_successes"]
        wandb_logs[f"overall_episodes"] = domain_success_metrics['overall']["total_episodes"]

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            wandb_logs
        )
        
    # ! Save to EvalResult 
    eval_result_for_episode.success = final_success
    eval_result_for_episode.actual_instruction_steps = instruction_steps
    
    # check again if this eval_result already exists in the database
    if not DATABASE_HELPER.check_if_exists(eval_result_for_episode):
        DATABASE_HELPER.insert_eval_result(eval_result_for_episode)
       

    return total_episodes, total_successes

def eval_mini_behavior(cfg: GenerateConfig, eval_result_skeleton: EvalResult, train_data_setup: str = "") -> float:
    """Main function to evaluate a trained policy on Mini_behavior benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor,\
        sentence_encoder_pad_token_id, instruction_sentence_encoder,\
            vision_backbone_sentence_encoder_tokenizer,\
                language_instruction_sentence_encoder_tokenizer = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    
    # Load RLDS dataset 
    dataset_name = cfg.dataset_name
    data_dir = RLDS_DATA_DIR
    dataset = tfds.load(dataset_name, data_dir=data_dir)
    if cfg.task_suite_name == TaskSuite.TEST_SINGLE_INSTR_NO_LANG_OOD:
        dataset = dataset[TaskSuite.TEST_SINGLE_INSTR_LANG_OOD.value] # re-use the lang_ood split rather than its own split
    else:
        dataset = dataset[cfg.task_suite_name]


    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    domain_success_metrics = dict()
    total_episodes, total_successes = 0, 0
    for episode in tqdm.tqdm(dataset):
        episode_metadata = episode['episode_metadata']
        file_path_info = episode_metadata['domain_name'].numpy().decode() # same as instruction_domain_name
        problem_id = episode_metadata['problem_id'].numpy().decode()
        # ! this is where we get additional info for dynamic single instruction selection
        instruction_width = int(episode_metadata['width'].numpy().decode())
        instruction_domain_name = episode_metadata['domain_name'].numpy().decode() # moving_boxes_to_storage
        
        # get last language instruction
        last_lang_instr = None
        for i in episode['steps']:
            last_lang_instr = i['language_instruction'].numpy().decode()
            
        match = re.search(r'At step (?P<step_num>\d+):', last_lang_instr)
        if match:
            step_num = int(match.group('step_num'))
        else:
            step_num = -1
        # run the task
        eval_result_skeleton.reference_instruction_steps = step_num
        total_episodes, total_successes = run_task(
            cfg,
            file_path_info,
            problem_id,
            instruction_width,
            instruction_domain_name,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            sentence_encoder_pad_token_id,
            instruction_sentence_encoder,
            vision_backbone_sentence_encoder_tokenizer,
            language_instruction_sentence_encoder_tokenizer,
            total_episodes,
            total_successes,
            log_file,
            domain_success_metrics,
            eval_result_skeleton,
            train_data_setup,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("\n================ Final Evaluation Results ================\n", log_file)
    # domain name 
    log_message(f"Domain: {instruction_domain_name}", log_file)
    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.save(local_log_filepath)
        wandb.finish()

    # Close log file
    if log_file:
        log_file.close()

    # remove model to free vram 
    del model 
    torch.cuda.empty_cache()
    return domain_success_metrics


def eval_model_meta_process(model_dir, checkpoint_step: int , args):
    
    # model_dir update to local path 
    model_dir = os.path.join(MODEL_SAVE_DIR, os.path.basename(model_dir))
    
    final_results = dict()
    logger.info(f"Evaluating model dir: {model_dir}")
    # DEBUG, we only for loop the 2 OOD Lang tasks 
    # FOCUSED_TASKS = [TaskSuite.TEST_SINGLE_INSTR_NO_LANG_OOD, TaskSuite.TEST_SINGLE_INSTR_LANG_OOD]
    # for task in FOCUSED_TASKS:
    
    # for task in TaskSuite: # loop all tasks
    # do it reverse order 
    for task in reversed(TaskSuite): # ! DEBUG
        # check if this task has been evaluated in existing_results
        
        if args.model_type == 'regressive':
            use_discrete_diffusion = False 
            use_l1_regression = False 
        elif args.model_type == 'action_head':
            use_discrete_diffusion = False 
            use_l1_regression = True
        elif args.model_type == 'discrete_diffusion':
            use_discrete_diffusion = True 
            use_l1_regression = False
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")
        
        # Construct run_id_note with all relevant parameters
        text_wsm_flag = "text_wsm" if args.text_world_state_model else "no_text_wsm"
        instr_enc_flag = "instr_enc" if args.use_instruction_encoding else "no_instr_enc"
        run_id_note = f"{args.train_data_setup}--{args.model_type}--{text_wsm_flag}--{instr_enc_flag}--chunks{args.num_chunks_for_text}--lr{args.lr}--{task.value}--step{checkpoint_step}--{EVAL_ROUND_ID}"
        
        cfg = GenerateConfig(
            model_family="openvla",
            pretrained_checkpoint=model_dir,
            use_l1_regression=use_l1_regression,
            use_diffusion=False,
            num_diffusion_steps_train=50,
            num_diffusion_steps_inference=50,
            use_film=False,
            use_text_world_state_model=args.text_world_state_model,
            use_instruction_sentence_encoder=args.use_instruction_encoding,
            num_chunks_for_text=args.num_chunks_for_text,
            use_discrete_diffusion=use_discrete_diffusion,
            num_images_in_input=1,
            use_proprio=False,
            center_crop=False,
            num_open_loop_steps=6,
            lora_rank=32,
            load_in_8bit=False,
            load_in_4bit=False,
            task_suite_name=task.value,
            num_trials_per_task=2, # ! one stochastic trial + one deterministic trial 
            env_img_res=IMAGE_RESOLUTION,
            dataset_name="mini_behavior",
            run_id_note=run_id_note,
            local_log_dir="./experiments/logs",
            use_wandb=USE_WANDB,
            wandb_entity="",
            wandb_project="OpenVLA-OFT-Single-Sent-Mini_behavior-Eval",
            seed=7,
        )
        if cfg.use_discrete_diffusion:
            cfg.use_l1_regression = False
            cfg.use_diffusion = False
        
        eval_result_skeleton = EvalResult(
            eval_round_id = EVAL_ROUND_ID,
            train_data_setup = args.train_data_setup,
            model_type = args.model_type,
            if_text_world_state_model = args.text_world_state_model,
            if_instruction_sentence_encoder = args.use_instruction_encoding,
            checkpoint_step = checkpoint_step,
            task_suite_name = task.value,
            num_chunks_for_text = args.num_chunks_for_text,
        )
        domain_success_metrics = eval_mini_behavior(cfg, eval_result_skeleton, train_data_setup=args.train_data_setup)
        final_results[task.value] = domain_success_metrics 
        
    return final_results
         
def main(args, ordered_model_dirs, ordered_checkpoint_steps):
    for ind, model_dir in enumerate(ordered_model_dirs):
        checkpoint_step = int(ordered_checkpoint_steps[ind])
        # ! ==== Download model dirs from storage server ====
        dirname = os.path.basename(model_dir)
        local_dir_path = os.path.join(MODEL_SAVE_DIR, dirname)
        if not os.path.exists(local_dir_path):
            print(f"Downloading model dir {model_dir} to local path {local_dir_path}...")
            rsync_cmd = f'rsync -chavP -e "ssh -p {STORAGE_SERVER_PORT}" {STORAGE_SERVER_USERNAME}@{STORAGE_SERVER_IP}:{model_dir} {MODEL_SAVE_DIR}/'
            subprocess.run(rsync_cmd, shell=True)
        else:
            print(f"Model dir {model_dir} already exists locally at {local_dir_path}. Skipping download.")
   
        
        eval_model_meta_process(model_dir, checkpoint_step, args)
        
        # ! ==== Remove local model dirs after eval to save space ====
        if REMOVE_DIRS_AFTER_EVAL:
            if os.path.exists(local_dir_path):
                print(f"Removing local model dir {local_dir_path} to free up space...")
                shutil.rmtree(local_dir_path)
                sleep(5)  # wait for a few seconds to ensure deletion
            else:
                print(f"Local model dir {local_dir_path} does not exist. Skipping removal.")
    


TRAIN_DATA_SETUP_ALLOWED = [
    "single_instr_pure_H",
    "single_instr_pure_M",
    "single_instr_pure_L",
    "single_instr_axial_H",
    "single_instr_axial_M",
    "single_instr_axial_L",
    "single_instr_centroid",
]

LR_ALLOWED = [
    "2.5e-4",
    "2.5e-5",
]

MODEL_TYPE_ALLOWED = [
    "regressive",
    "action_head",
    "discrete_diffusion",
]

STORAGE_SERVER_PORT = 13333
STORAGE_SERVER_IP = "115.146.82.102"
STORAGE_SERVER_USERNAME = "xxx"

STORAGE_DIRS = [f'/home/{STORAGE_SERVER_USERNAME}/Project/xxxx',
                f'/home/{STORAGE_SERVER_USERNAME}/Project/xxxx']
               
def obtain_matched_model_dir(train_data_setup, lr_str, model_type, text_world_state_model_flag, use_instruction_encoding_flag, num_chunks_for_text):
    
    # Example: openvla-7b+mini_behavior_single_instr_pure_M+b8+lr-0.001+lora-r32+dropout-0.0--finetune-no_instruction_se-chunks15-use_text_wsm--100000_chkpt
    
    # Example 2: openvla-7b+mini_behavior_single_instr_axial_M+b8+lr-0.00025+lora-r32+dropout-0.0--finetune--100000_chkpt
    lr_str_lst = [str(float(lr_str))]  # normalize lr str
    num_chunks_for_text = int(num_chunks_for_text)
    if model_type == 'action_head':
        model_type_str_lst = [""]
    elif model_type == 'regressive':
        model_type_str_lst = [r"\-regressive"]
    elif model_type == 'discrete_diffusion':
        model_type_str_lst = [r"\-with_discrete_diffusion"]
        
    train_data_setup_str_lst = [train_data_setup]
    if text_world_state_model_flag:
        text_world_state_model_str_lst = [r"\-use_text_wsm"]
    else:
        text_world_state_model_str_lst = [r"\-no_text_wsm", ""]
        
    if use_instruction_encoding_flag:
        instruction_encoding_str_lst = [r"\-use_instruction_se"]
    else:
        instruction_encoding_str_lst = [r"\-no_instruction_se", ""]
        
    if text_world_state_model_flag or use_instruction_encoding_flag:
        if text_world_state_model_flag:
            num_chunks_for_text_str_lst = [rf"\-chunks{num_chunks_for_text}"]
        elif use_instruction_encoding_flag:
            num_chunks_for_text_str_lst = [rf"\-chunks{num_chunks_for_text}"]
    else:
        num_chunks_for_text_str_lst = ["", r'\-chunks1']

    POSSIBLE_COMBINATIONS = list(itertools.product(train_data_setup_str_lst, lr_str_lst, model_type_str_lst, text_world_state_model_str_lst, instruction_encoding_str_lst, num_chunks_for_text_str_lst))
    
    patterns_for_re = []
    for combo in POSSIBLE_COMBINATIONS:
        train_data_setup_choice, lr_choice, model_type_choice, text_world_state_model_choice, instruction_encoding_choice, num_chunks_for_text_choice = combo
        patterns = rf'openvla-7b\+mini_behavior_{train_data_setup_choice}\+b\d+\+lr\-{lr_choice}\+lora\-r32\+dropout\-0\.0\-\-finetune{instruction_encoding_choice}{num_chunks_for_text_choice}{text_world_state_model_choice}{model_type_choice}\-\-(?P<checkpoint_step>\d+)_chkpt'
        patterns_for_re.append(patterns)
    
    full_remote_dirs = []
    selected_model_dirs = []
    checkpoint_steps = []
    
    for storage_dir in STORAGE_DIRS:
       
        ssh_cmd = f'ssh -p {STORAGE_SERVER_PORT} {STORAGE_SERVER_USERNAME}@{STORAGE_SERVER_IP} "ls {storage_dir}"'
        try:
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"SSH command failed with error: {result.stderr}")
                continue
            remote_dirs = result.stdout.strip().split('\n')
            for i in range(len(remote_dirs)):
                # add the storage_dir path
                remote_dirs[i] = os.path.join(storage_dir, remote_dirs[i])
            full_remote_dirs.extend(remote_dirs)
        except subprocess.TimeoutExpired:
            print(f"SSH command to {STORAGE_SERVER_IP} timed out.")
            continue
    for remote_dir in full_remote_dirs:
        dir_basename = os.path.basename(remote_dir)
        for pattern in patterns_for_re:
            match = re.search(pattern, dir_basename)
            if match:
                print(f"Matched model dir: {remote_dir} with pattern: {pattern}")
                selected_model_dirs.append(remote_dir)
                checkpoint_steps.append(int(match.group('checkpoint_step')))
        
    if len(selected_model_dirs) > 0:
        # order the selected_model_dirs by checkpoint_steps descending
        ordered_indices = np.argsort(checkpoint_steps)[::-1]
        # do it reverse order
        ordered_indices = ordered_indices[::-1] # ! DEBUG
        ordered_model_dirs = [selected_model_dirs[i] for i in ordered_indices]
        ordered_checkpoint_steps = [checkpoint_steps[i] for i in ordered_indices]
        return ordered_model_dirs, ordered_checkpoint_steps
    else:
        raise ValueError("No matched model dirs found.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_setup", type=str, required=True, choices=TRAIN_DATA_SETUP_ALLOWED, help="Evaluation setting to use. E.g., single_instr_pure_H")
    parser.add_argument("--lr", type=str, default="2.5e-4",choices=LR_ALLOWED, help="Learning rate to use. E.g., 2e-5")
    # text world state model flag 
    parser.add_argument("--model_type", type=str, required=True, choices=MODEL_TYPE_ALLOWED, help="Model type to use. E.g., regressive, action_head, discrete_diffusion")
    parser.add_argument("--text_world_state_model", action="store_true", help="Whether to use the text world state model.")
    parser.add_argument("--use_instruction_encoding", action="store_true", help="Whether to use instruction encoding.")
    parser.add_argument("--num_chunks_for_text", type=int, default=1, help="Number of chunks for text input.")

    # add use_discrete_diffusion
    # parser.add_argument("--use_discrete_diffusion", action="store_true", help="Whether to use discrete diffusion model.")
    
    args = parser.parse_args()
    ordered_model_dirs, ordered_checkpoint_steps = obtain_matched_model_dir(args.train_data_setup, args.lr, args.model_type, args.text_world_state_model, args.use_instruction_encoding, args.num_chunks_for_text)

    main(args, ordered_model_dirs, ordered_checkpoint_steps)
    
    
    # source env_xxxorgan.sh && CUDA_VISIBLE_DEVICES=2 python /data/projects/xxxplace0478/xxxnameh/xxxname_Project/vla/SimpleVLA-RL/modules/openvla-oft/eval/get_eval_result_instruction_split.py --train_data_setup single_instr_pure_H --model_type discrete_diffusion
    
 
    