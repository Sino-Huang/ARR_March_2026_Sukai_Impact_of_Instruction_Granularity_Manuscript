# %%
import re
from glob import glob
import os 
from tqdm.auto import tqdm
import json 
from pathlib import Path
from copy import deepcopy

# %%
# RAW_RLDS_DIRPATH_xxxorgan = "/data/projects/xxxplace0478/xxxnameh/xxxname_Project/vla/SimpleVLA-RL/data/01_raw/modified_nsai_rlds/mini_behavior"
# RAW_RLDS_DIRPATH_xxxorganx = "/nfsdata/data/xxxnameh/SimpleVLA-RL/data/01_raw/modified_nsai_rlds/mini_behavior"

# single sentence version
RAW_RLDS_DIRPATH_xxxorgan = "/data/gpfs/projects/xxxplace2219/xxxname_temp_storage/single_sentence_modified_nsai_rlds/mini_behavior"
RAW_RLDS_DIRPATH_xxxorganx = "/nfsdata/data/xxxnameh/SimpleVLA-RL/data/01_raw/single_sentence_modified_nsai_rlds/mini_behavior"

answer = input("xxxorganx or xxxorgan? (m/u): ")

if answer.lower() == 'm':
    RAW_RLDS_DIRPATH = RAW_RLDS_DIRPATH_xxxorganx
elif answer.lower() == 'u':
    RAW_RLDS_DIRPATH = RAW_RLDS_DIRPATH_xxxorgan
else:
    raise ValueError("Invalid input. Please enter 'm' for xxxorganx or 'u' for xxxorgan.")

# %%
feature_json_fp = os.path.join(RAW_RLDS_DIRPATH, "*/features.json")
dataset_info_json_fp = os.path.join(RAW_RLDS_DIRPATH, "*/dataset_info.json")

feature_json_fps = glob(feature_json_fp, recursive=True)
dataset_info_json_fps = glob(dataset_info_json_fp, recursive=True)

assert len(feature_json_fps) == len(dataset_info_json_fps)
assert len(feature_json_fps) == 1

# %%
feature_json_fps

# %%
feature_data = json.load(open(feature_json_fps[0], "r"))
dataset_info_data = json.load(open(dataset_info_json_fps[0], "r"))

# %%
dataset_info_data

# %%
# feature data is static
# the dataset info data shall be splitted
raw_dataset_name = dataset_info_data['name']

# %%
def parse_split_name(split_name):
    pattern = r'(train|test)_(full_instr|single_instr)_(id|task_ood|lang_ood)_granularity_(L|M|H)'
    pattern_2 = r'test_full_instr_lang_ood|test_single_instr_lang_ood'
    match = re.match(pattern, split_name)
    if match:
        split_type, instr_type, ood_type, granularity = match.groups()
        return split_type, instr_type, ood_type, granularity
    else:
        match_2 = re.match(pattern_2, split_name)
        if match_2:
            return 'test', 'full_instr', 'lang_ood', None
    return None

# %%

for split_data in tqdm(dataset_info_data['splits'], desc="Processing splits"):
    split_name = split_data['name']
    parse_output = parse_split_name(split_name)
    if parse_output is not None:
        split_type, instr_type, ood_type, granularity = parse_output
    else:
        raise ValueError(f"Split name {split_name} does not match expected pattern.")
    print(f"Processing split: {split_name}")
    associated_files = glob(os.path.join(RAW_RLDS_DIRPATH, f"*/{raw_dataset_name}-{split_name}.*"), recursive=True)
    version_number = Path(associated_files[0]).parent.name
    print(f"  Found {len(associated_files)} files for this split.")
    # now we only operate on the train split type 
    if split_type == 'train':
        # step 1: create new folder next to the old one
        new_dirpath = os.path.join(os.path.dirname(RAW_RLDS_DIRPATH), f"mini_behavior_{split_name}")
        # make new_dirpath lower() for its last part only
        
        new_dirpath = os.path.dirname(new_dirpath) + '/' + os.path.basename(new_dirpath).lower()
        Path(os.path.join(new_dirpath, version_number)).mkdir(parents=True, exist_ok=True)
        # step 2: copy feature_data and dataset_info_data
        with open(os.path.join(new_dirpath, version_number, "features.json"), "w") as f:
            json.dump(feature_data, f, indent=4)
        new_dataset_info_data = deepcopy(dataset_info_data)
        # update name to be f"mini_behavior_{split_name}"
        new_dataset_info_data['name'] = f"mini_behavior_{split_name}"
        # splits only contain the current split
        new_dataset_info_data['splits'] = [split_data]
        # change the current split name to 'train'
        new_dataset_info_data['splits'][0]['name'] = 'train'
        with open(os.path.join(new_dirpath, version_number, "dataset_info.json"), "w") as f:
            json.dump(new_dataset_info_data, f, indent=4)
        # step 3: copy associated files
        for filepath in associated_files:
            new_filename = os.path.basename(filepath).split('-')
            new_filename = '-'.join(new_filename[1:])  # remove the raw dataset name prefix
            new_filename_want_index = new_filename.find('.tfrecord')
            new_filename = new_filename[new_filename_want_index:]  # keep only the .tfrecord and beyond
            new_filename = f"mini_behavior_{split_name}" + '-' + 'train' + new_filename
            new_filepath = os.path.join(new_dirpath, version_number, new_filename)
            os.system(f"cp {filepath} {new_filepath}")
        print(f"  Created new dataset at {new_dirpath}")
        

# %%



