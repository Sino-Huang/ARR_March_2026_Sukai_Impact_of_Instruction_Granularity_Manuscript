"""
Generate Mini Behavior dataset (HDF5 files) by replaying demonstrations in the environments

The images format is opencv format (height x width)

Notes:
    - save images at 256 x 256 px resolution 
    - multiple plans and also task description list (can do permutation)
    - all are successful demonstrations 
    
Usage:
    python experiments/robot/mini_behavior/regenerate_mini_behavior_dataset.py \
        --domain_name <DOMAIN_NAME> \
        --mini_behavior_target_dir <PATH TO TARGET DIR>
RLDS loads/exposes steps aligned as:
(o0, a0, r0, d0, m0) → (o1, a1, r1, d1, m1) → (o2, a2, r2, d2, m2) →

Later, need to modify modules/openvla-oft/prismatic/vla/datasets/rlds/oxe/configs.py content to have the mini_behavior config also load_proprio become false
"""

import argparse
import json
import os
import time
from glob import glob
import re
from tqdm.auto import tqdm
import h5py
import numpy as np
from pathlib import Path
from random import shuffle
import sys
import cv2
import random 
import pickle 
from sketch_width_analysis import test_all_tasks
from mini_behavior.utils.policy_sketch.save_n_load_env import load_env_from_pickled_sketch_data
import multiprocessing as mp
from multiprocessing import Pool, Manager
from multiprocessing.dummy import Pool as DummyPool

# Get the current directory of main_script.py
current_dir = os.path.dirname(os.path.abspath(__file__))
openvla_oft_dir = Path(current_dir).parent.parent.parent

# Add the current directory to sys.path
sys.path.insert(0, str(openvla_oft_dir))


from experiments.robot.mini_behavior.mini_behavior_utils import (
    get_mini_behavior_env,
    ACTION_PARSING_DICT,
    get_mini_behavior_image,
    DOMAIN_NAME_LIST,
    PDDL_PROBLEM_PATTERN,
)

IMAGE_RESOLUTION = 256
TEST_INSTANCE_NUM = 5 # number of test instances per domain
REPEAT_PER_PROBLEM_ID = 4 # repeat to cope with noise in visual observations, train the agent to be robust to noise

def one_hot_action(action, mapping=ACTION_PARSING_DICT, dtype=np.float32):
    """
    Convert an action (str or int) to a one-hot numpy array of shape (num_actions,)
    with the given dtype (default float32).
    """
    # action type is <enum 'Actions'>
    
    # Number of actions assuming 0..N-1 ids (matches your mapping)
    num_actions = len(ACTION_PARSING_DICT)
    if getattr(action, 'value', None) is not None:
        idx = int(action.value)
    elif isinstance(action, str):
        idx = int(mapping[action])
    # One-hot vector
    vec = np.zeros(num_actions, dtype=dtype)
    vec[idx] = 1.0
    return vec

# Examples:
# one_hot_action('dir-up')  -> array([0., 0., 0., 0., 1.], dtype=float32)
# one_hot_action(2)         -> array([0., 0., 1., 0., 0.], dtype=float32)


def process_image(image):
    # check if already IMAGE_RESOLUTION THEN RETURN ITSELF
    if image.shape[0] == IMAGE_RESOLUTION and image.shape[1] == IMAGE_RESOLUTION:
        return image
    else:
    # ! rescale opencv2 numpy image
        image = cv2.resize(image, (IMAGE_RESOLUTION, IMAGE_RESOLUTION), interpolation=cv2.INTER_AREA)
        return image

def mp_helper(arg):

    pickle_fp, split_type, width_val, granularity_level, domain_name, repeat_id, metainfo_json_dict, final_data_dict, mini_behavior_target_dir, map_id = arg

    with open(pickle_fp, 'rb') as f:
        gps_data = pickle.load(f)
    
    obs_collection, text_obs_collection, action_collection, reward_collection, done_collection, info_collection, env, gps_data, window = load_env_from_pickled_sketch_data(gps_data, want_render=False, image_for_vla=True)
    obs_collection = [process_image(image) for image in obs_collection]
    if len(obs_collection) == 0:
        print(f"No observations found in GPS data from {pickle_fp}")
        return False
    
    actions = []
    for action in action_collection:
        action_vec = one_hot_action(action)
        actions.append(action_vec)

    # for nl_length_type in ["whole_trajectory", "single_instruction", "single_rule"]: # ! LOOP #3: whole trajectory vs single instruction
    # for nl_length_type in ["single_instruction", "single_rule"]: # ! LOOP #3: whole trajectory vs single instruction
    for nl_length_type in ["single_instruction"]: # ! LOOP #3: whole trajectory vs single instruction
        # create create HDF5 file for generated demos
        new_data_path = os.path.join(mini_behavior_target_dir, f"gps-{split_type}-width_{width_val}-granularity_{granularity_level}-{nl_length_type}-domain_{domain_name}.hdf5")
       
        demo_str = f"demo_map_{map_id}_sub_{repeat_id}"
        dones = np.zeros(len(obs_collection), dtype=np.uint8)
        dones[-1] = 1
        rewards = np.asarray(reward_collection, dtype=np.float32)
        rewards[-1] = 1.0
        
        # instr_str 
        # we get info['instr_at_cur_step'] and info['overall_instr']
        if nl_length_type == "whole_trajectory":
            inst_str_lst = [info['overall_instr'] for info in info_collection]
        elif nl_length_type == "single_instruction":
            inst_str_lst = [info['instr_at_cur_step'] for info in info_collection]
        elif nl_length_type == "single_rule":
            inst_str_lst = [info['rule_at_cur_step'] for info in info_collection]
        else:
            print(f"Invalid nl_length_type: {nl_length_type}")
            raise ValueError(f"Invalid nl_length_type: {nl_length_type}")
        
        obs_images = np.stack(obs_collection, axis=0)
        obs_language_instructions = np.array(inst_str_lst, dtype='S')
        obs_text_based = np.array(text_obs_collection, dtype='S')
        
        # put all these into final_data_dict

        compose_key = (new_data_path, demo_str)
        local_dict = dict()
        local_dict['obs_images'] = obs_images
        local_dict['obs_language_instructions'] = obs_language_instructions
        local_dict['obs_text_world_state'] = obs_text_based
        local_dict['actions'] = actions
        local_dict['rewards'] = rewards
        local_dict['dones'] = dones
        
        final_data_dict[compose_key] = local_dict

        # create a key that is the basename of new_data_path without extension + demo_str
        task_key = os.path.basename(new_data_path).replace('.hdf5', '')
        episode_key = demo_str 
        # record gps_data into metainfo_json_dict
        
        compose_key_metainfo = (task_key, episode_key)
        local_metainfo_dict = dict()
        local_metainfo_dict['width'] = width_val
        local_metainfo_dict['granularity_level'] = granularity_level
        local_metainfo_dict['nl_length_type'] = nl_length_type
        local_metainfo_dict['repeat_id'] = repeat_id
        local_metainfo_dict['map_id'] = map_id
        local_metainfo_dict['split_type'] = split_type
        local_metainfo_dict['nl_instructions'] = gps_data['nl_instructions']
        
        metainfo_json_dict[compose_key_metainfo] = local_metainfo_dict
        
    return bool(done_collection[-1])  # return True if last step is done

def main(args):  
    domain_name = args.domain_name
    mini_behavior_target_dir = args.mini_behavior_target_dir
    assert domain_name in DOMAIN_NAME_LIST, f"Invalid domain name: {domain_name}. Must be one of {DOMAIN_NAME_LIST}"
        
    # process mini_behavior_target_dir
    mini_behavior_target_dir = os.path.join(os.environ["WORKING_DIR"], mini_behavior_target_dir)
    
    Path(mini_behavior_target_dir).mkdir(parents=True, exist_ok=True)
    
    # get width analysis 
    tasks, width_classification_list = test_all_tasks()
    # only get the current domain_name's width classification
    width_classification = None # dict with key 'Low', 'Middle', 'High' and values as list of widths
    for i, task_id in enumerate(tasks.keys()):
        if task_id == domain_name:
            width_classification = width_classification_list[i]
            break
    assert width_classification is not None, f"Width classification for domain {domain_name} not found."
    
    # obtain general policy sketch pickle data folder path 
    gps_dirpath = os.path.join(os.environ["WORKING_DIR"], f"data/02_intermediate/gps_data/{domain_name}")
    assert os.path.exists(gps_dirpath), f"GPS data directory {gps_dirpath} does not exist."
    map_id_width_pickle_fp_dict = dict()
    for width in width_classification['Low'] + width_classification['Middle'] + width_classification['High']:
        pattern_str = os.path.join(gps_dirpath, f"gps_data_map_*_width_{width}_env_MiniGrid-*_goal_True.pkl")
        matched_files = glob(pattern_str, recursive=True)
        assert len(matched_files) > 0, f"No GPS data files found for pattern: {pattern_str}"
        for fp in matched_files:
            map_id_width_pattern = r'gps_data_map_(\d+)_width_(\d+)_env_MiniGrid-.*\.pkl'
            re_match = re.search(map_id_width_pattern, os.path.basename(fp))
            if re_match:
                map_id = re_match.group(1)
                width_val = int(re_match.group(2))
                if map_id not in map_id_width_pickle_fp_dict:
                    map_id_width_pickle_fp_dict[map_id] = dict()
                map_id_width_pickle_fp_dict[map_id][width_val] = fp
                
    
    print(f"Generating Mini Behavior dataset HDF5 {domain_name}")
    
    # Setup 

    random_seed = 42 
    random.seed(random_seed)
    
    # random pick TEST_INSTANCE_NUM from map_id_width_pickle_fp_dict
    
    all_problem_ids = list(map_id_width_pickle_fp_dict.keys())
    shuffle(all_problem_ids)
    test_problem_ids = all_problem_ids[:TEST_INSTANCE_NUM]
    train_problem_ids = all_problem_ids[TEST_INSTANCE_NUM:]
    
    mp_manager = Manager()
    metainfo_json_dict = mp_manager.dict()

    metainfo_json_out_path = os.path.join(mini_behavior_target_dir, f"{domain_name}_metainfo.json")
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(dict(metainfo_json_dict), f)

    
    final_data_dict = mp_manager.dict()  # key: new_data_path, value: dict of demos
    
    args = [] 

    for map_id in tqdm(test_problem_ids + train_problem_ids, desc="Processing map IDs"): # ! LOOP #1: train/test split
        if map_id in test_problem_ids:
            split_type = "test"
        else:
            split_type = "train"
            
        further_data_dict = map_id_width_pickle_fp_dict[map_id] # dict with width as key, pickle fp as value
        
        for width_val, pickle_fp in further_data_dict.items(): # ! LOOP #2: different widths
            # check width belongs to which granularity level
            granularity_level = None
            for level_key, width_list in width_classification.items():
                if width_val in width_list:
                    granularity_level = level_key
                    break
                
            assert granularity_level is not None, f"Width value {width_val} not found in any granularity level for domain {domain_name}."
            
            if split_type == "train":
                repeat_per_task = REPEAT_PER_PROBLEM_ID
            elif split_type == "test":
                repeat_per_task = 1
            else:
                raise ValueError(f"Invalid split_type: {split_type}")
            for repeat_id in tqdm(range(repeat_per_task), desc="Processing repeats", leave=False): # ! LOOP #4: repeat per problem id to cope with noise
                
                arg = [
                    pickle_fp, 
                    split_type,
                    width_val, 
                    granularity_level,
                    domain_name,
                    repeat_id,
                    metainfo_json_dict,
                    final_data_dict,
                    mini_behavior_target_dir,
                    map_id,
                ]
                args.append(arg)
                
    with Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(mp_helper, args, chunksize=1), total=len(args), desc="Generating datasets in parallel"))
        
    # print results summary
    num_successful_demos = sum(results)
    print(f"Generated {num_successful_demos} successful demonstrations out of {len(results)} attempts.")
    
    # decompose keys of final_data_dict and metainfo_json_dict
    final_data_dict = dict(final_data_dict)
    metainfo_json_dict = dict(metainfo_json_dict)
    
    final_data_dict_decomposed = dict()
    metainfo_json_dict_decomposed = dict()
    
    for compose_key in final_data_dict:
        new_data_path, demo_str = compose_key
        if new_data_path not in final_data_dict_decomposed:
            final_data_dict_decomposed[new_data_path] = dict()
        final_data_dict_decomposed[new_data_path][demo_str] = final_data_dict[compose_key]
    final_data_dict = final_data_dict_decomposed
    for compose_key in metainfo_json_dict:
        task_key, episode_key = compose_key
        if task_key not in metainfo_json_dict_decomposed:
            metainfo_json_dict_decomposed[task_key] = dict()
        metainfo_json_dict_decomposed[task_key][episode_key] = metainfo_json_dict[compose_key]
    metainfo_json_dict = metainfo_json_dict_decomposed
    
    print("Data generation completed.")
    print(f"Size of final_data_dict: {len(final_data_dict)} HDF5 files to write.")
    print(f"Size of metainfo_json_dict: {len(metainfo_json_dict)} episodes to write.")
                    
    # After all data collected, write to HDF5 files
    print("Writing generated datasets to HDF5 files...")
    for hdf5_fp, demos_dict in tqdm(final_data_dict.items(), desc="Writing HDF5 files", total=len(final_data_dict)):
        new_data_file = h5py.File(hdf5_fp, 'w')
        grp = new_data_file.create_group("data")
        for demo_str, data_content in demos_dict.items():
            ep_data_grp = grp.create_group(demo_str)
            obs_grp = ep_data_grp.create_group("obs")
            obs_grp.create_dataset("images", data=data_content['obs_images'])
            obs_grp.create_dataset("language_instruction", data=data_content['obs_language_instructions'])
            # add text world state observation
            obs_grp.create_dataset("text_world_state", data=data_content['obs_text_world_state'])
            ep_data_grp.create_dataset("actions", data=data_content['actions'])
            ep_data_grp.create_dataset("rewards", data=data_content['rewards'])
            ep_data_grp.create_dataset("dones", data=data_content['dones'])
        new_data_file.close()
        print(f"Saved generated dataset at: {hdf5_fp}")
    # Write metainfo dict to JSON file
    with open(metainfo_json_out_path, "w") as f:
        json.dump(metainfo_json_dict, f, indent=2)
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")
    
    # overall analysis, how many episodes generated per task
    for task_key in metainfo_json_dict.keys():
        num_episodes = len(metainfo_json_dict[task_key])
        print(f"Task {task_key} has {num_episodes} episodes generated.")
    
if __name__ == "__main__":
    # parse command-line arugments
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str, help="mini behavior domain name")
    parser.add_argument("--mini_behavior_target_dir", type=str, help="Path to generate dataset directory, automatically concate with WORKING_DIR", required=True)
    
    # combinations of output 
    # 1. train vs test split
    # 2. the different widths' dataset, further labeled with L/M/H level of granularity
    # 3. whole trajectory vs single instruction trajectory
    
    # washing pots and pans, clean shoes, watering houseplants can be three OOD-hard tasks
    # 1 preparing salad, 2 cleaning up the kitchen only, 3 organizing file cabinet, 4 making tea, 5 putting away dishes after cleaning, 6 setting up candles, 7 cleaning a car
    

    
    
    args = parser.parse_args()
    
    print("Arguments:", args)
    # start data generation 
    main(args)
    
    
    # example 
    # python data/00_modules/openvla-oft/experiments/robot/mini_behavior/regenerate_mini_behavior_dataset_based_on_width.py --domain_name cleaning_up_the_kitchen_only --mini_behavior_target_dir data/01_raw/mini_behavior_gps_hdf5