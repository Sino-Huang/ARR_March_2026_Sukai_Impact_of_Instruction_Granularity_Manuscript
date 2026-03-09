import os 
import sys 
import json 
import random
import time
from pathlib import Path 
from subprocess import Popen, PIPE
import subprocess
from glob import glob
from multiprocessing import Pool  # multiprocessing for parallel execution
import tempfile
from tqdm.auto import tqdm
import argparse
import pickle
from time import sleep
from mini_behavior.utils.policy_sketch.save_n_load_env import load_env_from_pickled_sketch_data            


def process_helper(cmd, display_output=False):
    """execute a shell command and return its success status and output"""
    # print(f"Executing command: {cmd}")
    process = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        if display_output:
            print(f"Error message: {stderr.decode()}")
        return False, cmd
    else:
        if display_output:
            print(f"Command succeeded with output: {stdout.decode()}")
        return True, cmd

def generate_bash_script(width, working_path, domain_name, map_id):
    """generate a bash script to run the policy sketch generation"""
    os.makedirs(working_path, exist_ok=True)
    py_file_path = os.path.join(os.environ['WORKING_DIR'], f'data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/{domain_name}.py')
    
    bash_script = f"""
#!/bin/bash
cd {working_path}
source ~/anaconda3/bin/activate nsai_behavior
timeout 120m python {py_file_path} --map_id {map_id} --domain_name {domain_name} --width {width}
cd ~
rm -rf {working_path}
"""
    return bash_script

def run_in_parallel(commands, max_processes=None):
    """parallel execution of shell commands"""
    # If max_processes is not specified, use the number of CPU cores
    if max_processes is None:
        max_processes = int(os.cpu_count()* 0.93)
    
    print(f"Starting parallel execution with {max_processes} processes...")
    
    # use Pool to manage parallel processes
    results = []
    with Pool(processes=max_processes) as pool:
        # map the process_helper function to the commands
        for result in tqdm(
            pool.imap_unordered(process_helper, commands, chunksize=1),
            total=len(commands),
            desc="Processing commands",
            unit="command"
        ):
            results.append(result)

    # Count successful commands
    success_count = sum(1 for success, _ in results if success)
    print(f"Parallel execution complete. {success_count}/{len(commands)} commands succeeded.")
    return results

def check_plan_valid_helper(exist_file):
    """Check if a file leads to goal condition and return result"""
    try:
        with open(exist_file, 'rb') as f:
            sketch_data = pickle.load(f)
        obs_collection, text_obs_collection, action_collectoin, reward_collection, done_collection, info_collection, env, sketch_data, window = load_env_from_pickled_sketch_data(sketch_data, want_render=False)
        if done_collection[-1]:
            return (True, exist_file, None)
        else:
            return (False, exist_file, "Does not lead to goal condition")
    except Exception as e:
        return (False, exist_file, f"Error: {str(e)}")

def generate_diff_width_data(width_list, domain_list=None, max_processes=None, delete_existing_invalid=True):
    start_time = time.time()
    
    plan_dir = os.path.join(os.environ['WORKING_DIR'], 'data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/**/*.json')
    plan_files = glob(plan_dir, recursive=True)
    
    if domain_list is not None:
        # filter plan_files if any domain_name in the list is in the filename
        in_list = False 
        filtered_plan_files = []
        for plan_file in plan_files:
            for domain_name in domain_list:
                if domain_name in plan_file:
                    in_list = True 
                    break
            if in_list:
                filtered_plan_files.append(plan_file)
            in_list = False
        plan_files = filtered_plan_files
    plan_files = [f for f in plan_files if 'deprecated' not in f] # remove deprecated files
    
    print(f"Found {len(plan_files)} plan files.")
    sleep(5)  # wait for user to read the message
    commands = []
    # e.g., data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/boxing_books_up_for_storage/p33715827-boxing_books_up_for_storage_plans.json
    
    not_ok_list = []
    existing_files = glob(os.path.join(os.environ['WORKING_DIR'], f'data/02_intermediate/gps_data/**/gps_data_map_*goal_True.pkl'), recursive=True)
    
    # * === check if existing files can smoothly lead to goal condition ===

    
    ok_plan_count = 0 
    all_plan_count = len(existing_files)
    
    # Use multiprocessing to check files in parallel
    max_check_processes = max_processes if max_processes else int(os.cpu_count() * 0.93)
    print(f"Checking {all_plan_count} existing files with {max_check_processes} processes...")
    failed_files = []
    with Pool(processes=max_check_processes) as pool:
        for success, exist_file, error_msg in tqdm(
            pool.imap_unordered(check_plan_valid_helper, existing_files),
            total=all_plan_count,
            desc="Checking existing files"
        ):
            if success:
                ok_plan_count += 1
            else:
                print(f"File {exist_file} failed: {error_msg}")
                failed_files.append(exist_file)
                # delete the file
                if delete_existing_invalid:
                    try:
                        os.remove(exist_file)
                    except Exception as e:
                        print(f"Failed to delete {exist_file}: {e}")
    
    print(f"File check complete: {ok_plan_count}/{all_plan_count} successful")
    print("Iterating print failed files:")
    for f in failed_files:
        print(f)
    # input("Press Enter to continue with dataset generation...")
    for plan_file in plan_files:
        plan_file = Path(plan_file)
        plan_filename = plan_file.name
        map_id_str, domain_name_part = plan_filename.split('-')
        map_id = int(map_id_str[1:])  # remove leading 'p'
        domain_name = domain_name_part.replace('_plans.json', '')
        
        # * === Check if the corresponding output file already exists ===
        exist_flag = False
        for exist_file in existing_files:
            exist_filename = Path(exist_file).name
            if f'map_{map_id}_' in exist_filename:
                exist_flag = True
                break
        if not exist_flag and 'deprecated' not in str(plan_file):
            not_ok_list.append((map_id, domain_name, str(plan_file)))
        # * === end check ===
        
        for width in width_list:
            print(f"Processing plan file: {plan_file}, map_id: {map_id}, domain_name: {domain_name}")
            temp_dir = tempfile.mkdtemp(prefix=f"gps_{domain_name}_w{width}_m{map_id}_")
            commands.append(generate_bash_script(width, temp_dir, domain_name, map_id))
        
    # for map_id, domain_name, plan_file in not_ok_list:
    #     print(f"map_id: {map_id}, domain_name: {domain_name} is missing.")
                
    # print(f"Total missing files to process: {len(not_ok_list)}")
        
    # Run commands in parallel
    run_in_parallel(commands, max_processes)
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets with different widths.")
    parser.add_argument('--widths', nargs='+', type=int, required=True, help='List of widths to generate data for.')
    parser.add_argument('--domains', nargs='*', type=str, help='List of domain names to filter plan files.')
    parser.add_argument('--max_processes', type=int, help='Maximum number of parallel processes to use.')
    args = parser.parse_args()

    generate_diff_width_data(args.widths, args.domains, args.max_processes)

    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/generate_dataset.py --max_processes 10 --width 0 1 2 3 4 5 6 --domains laying_wood_floors