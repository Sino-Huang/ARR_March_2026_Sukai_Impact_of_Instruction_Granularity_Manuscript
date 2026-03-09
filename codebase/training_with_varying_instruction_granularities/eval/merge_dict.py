import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from time import sleep
from glob import glob 
from loguru import logger
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model checkpoints and eval results')
    parser.add_argument('--fuse_dict_path', type=str, required=True, help='Path to the dictionary file to be merged with eval results')
    
    args = parser.parse_args()
    model_dir = args.model_dir
    fuse_dict_path = args.fuse_dict_path
    
    base_eval_result_fp = os.path.join(model_dir, 'single_sentence_eval_results.json')
    assert os.path.exists(base_eval_result_fp), f"Eval result file not found: {base_eval_result_fp}"
    with open(base_eval_result_fp, 'r') as f:
        base_eval_results = json.load(f)
        
    # backup original eval results
    backup_eval_result_fp = os.path.join(model_dir, 'single_sentence_eval_results_backup.json')
    with open(backup_eval_result_fp, 'w') as f:
        json.dump(base_eval_results, f, indent=4)

    assert os.path.exists(fuse_dict_path), f"Fuse dict file not found: {fuse_dict_path}"
    with open(fuse_dict_path, 'r') as f:
        fuse_dict = json.load(f)
        
    # Merge fuse_dict into base_eval_results
    for setting_key in tqdm(base_eval_results.keys(), desc="Merging eval results", total=len(base_eval_results.keys())):
        for sub_key in base_eval_results[setting_key].keys():
            for checkpoint_key in base_eval_results[setting_key][sub_key].keys():
                fuse_data = fuse_dict[setting_key][sub_key][checkpoint_key]
                for task_suite_key in fuse_data.keys():
                    if task_suite_key in ['test_single_instr_lang_ood', 'test_single_instr_no_lang_ood']:
                        logger.critical(f"Merging data for {setting_key} | {sub_key} | {checkpoint_key} | {task_suite_key}")
                        base_eval_results[setting_key][sub_key][checkpoint_key][task_suite_key] = fuse_data[task_suite_key]

    # Save merged results
    with open(base_eval_result_fp, 'w') as f:
        json.dump(base_eval_results, f, indent=4)
    logger.info(f"Merged eval results saved to: {base_eval_result_fp}")
    
