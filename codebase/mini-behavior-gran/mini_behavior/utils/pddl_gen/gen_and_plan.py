import time
import numpy as np
from mini_behavior.utils.pddl_gen.pddl_problem_gen import main as pddl_problem_gen_main, GOAL_CLAUSES
from pathlib import Path 
import subprocess 
import os 
import tempfile 
import random 
import numpy as np
from tqdm.auto import tqdm 
import json
from glob import glob
from termcolor import colored
from multiprocessing import Pool
import shutil

ACTION_PARSING_DICT = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "toggle": 3,
    "open": 4,
    "close": 5,
    "slice": 6,
    "cook": 7,
    "drop_in": 8,
    "pickup_0": 9,
    "pickup_1": 10,
    "pickup_2": 11,
    "drop_0": 12,
    "drop_1": 13,
    "drop_2": 14,
}


class FastDownward():
    """FastDownward planner. (multiple plans)
    """
    def __init__(self,fast_downward_path =None, quality_bound=0.9, number_of_plans_bound=10):
        super().__init__()
        if fast_downward_path is not None:
            self.fast_downward_path = fast_downward_path
        else:
            self.fast_downward_path = os.path.join(os.path.dirname(__file__), "fast-downward.py")
        self.quality_bound = quality_bound
        self.number_of_plans_bound = number_of_plans_bound
        self._statistics = {}
 
    def get_statistics(self):
        """Get statistics of the planner.
        """
        return self._statistics
            
    def plan_from_pddl(self, dom_file, prob_file, horizon=np.inf, timeout=10,
                       remove_files=False, optimal=False):
        """PDDL-specific planning method.
        """
        start_time = time.time()
        
        fast_downward_path = self.fast_downward_path
        if not os.path.exists(fast_downward_path):
            raise FileNotFoundError(f"Fast Downward script not found at {fast_downward_path}. Please ensure it is in the correct directory.")
        if not optimal:
            cmd_lst = [
                str(fast_downward_path),
                '--alias',
                'lama-first',
                '--search-time-limit',
                str(timeout),
                dom_file,
                prob_file,
            ]
            print(colored(f"Running command: {' '.join(cmd_lst)}", 'green'))
            output = subprocess.run(cmd_lst,
                                    capture_output=True, text=True, check=True)
            plan_raw_output = output.stdout
        else:
            try:
                cmd_lst = [
                    str(fast_downward_path),
                    '--alias',
                    'lama',
                    '--search-time-limit',
                    str(timeout),
                    '--overall-time-limit',
                    str(timeout),
                    dom_file,
                    prob_file,
                ]
                print(colored(f"Running command: {' '.join(cmd_lst)}", 'green'))
                output = subprocess.run(cmd_lst,
                                        capture_output=True, text=True, check=True)
                plan_raw_output = output.stdout
                
            except subprocess.CalledProcessError as e:
                stdout_text = e.stdout
                stderr_text = e.stderr
                plan_raw_output = stdout_text + stderr_text
        
        # pick lines after Actual search time and before Plan length
        plan_group = []
        plan_cost_list = [] 
        plan_lines = []
        start_collecting = False 
        print("Plan raw output:")
        print(plan_raw_output)
        split_lines = plan_raw_output.splitlines()
        for i, line in enumerate(split_lines):
            if "Actual search time" in line:
                # further check if next line start if [t=, if yes, then this is not a valid plan
                if i + 1 < len(split_lines):
                    next_line = split_lines[i + 1]
                    if next_line.strip().startswith('[t='):
                        continue
                start_collecting = True
                continue
            if "Plan length" in line:
                start_collecting = False
                plan_group.append(plan_lines)
                # i + 1 will be plan cost line
                if i + 1 < len(split_lines):
                    cost_line = split_lines[i + 1]
                    assert 'Plan cost' in cost_line, f"Expected 'Plan cost' in line: {cost_line}"
                    cost_of_plan = cost_line.split(':')[-1].strip()
                    cost_of_plan = int(cost_of_plan)
                    plan_cost_list.append(cost_of_plan)
                else:
                    plan_cost_list.append(1000)
                    
                
                plan_lines = []
            if start_collecting:
                plan_lines.append(line.strip())
            else:
                continue
        
        # sort the plan_group
        plan_group = [x for _, x in sorted(zip(plan_cost_list, plan_group), key=lambda pair: pair[0])]
        plan_cost_list = sorted(plan_cost_list)
        
        
        output = {'plans': []}
        for i, each_plan in enumerate(plan_group):
            output['plans'].append({
                'actions': each_plan,
                'cost': plan_cost_list[i]
            })

        pddl_plan_lst = self._output_to_plan_list(output)

        pddl_plan_lst = [x for x in pddl_plan_lst if len(x) <= horizon]
        if len(pddl_plan_lst) == 0:
            raise ValueError("No valid plans found within the horizon limit: {}".format(horizon))
        if remove_files:
            os.remove(dom_file)
            os.remove(prob_file)
        return pddl_plan_lst

    def _output_to_plan_list(self, output):
        if output.get('plans') is None:
            raise ValueError("No plans found in the output. Check if the planner was installed correctly or if the input PDDL files are valid.")
        else:
            self._statistics['num_plans'] = len(output['plans'])
            cost_dict = {}
            cost_id_dict = {}
            for i, plan in enumerate(output['plans']):
                cost = plan['cost']
                if cost not in cost_dict:
                    cost_dict[cost] = 0
                cost_dict[cost] += 1
                if cost not in cost_id_dict:
                    cost_id_dict[cost] = []
                cost_id_dict[cost].append(i)
            # calculate cost distribution
            total_plans = sum(cost_dict.values())
            cost_dict = {k: v / total_plans for k, v in cost_dict.items()}
            self._statistics['cost_distribution'] = cost_dict
            pddl_plan_lst = [] # sorted by lower cost first
            sorted_costs = sorted(cost_dict.keys())
            self._statistics['sorted_costs'] = sorted_costs
            for cost in sorted_costs:
                for plan_id in cost_id_dict[cost]:
                    pddl_plan = output['plans'][plan_id]['actions']
                    pddl_plan_lst.append(pddl_plan)

            return pddl_plan_lst
        
    def _parse_action(self, action_str):
        for action, action_id in ACTION_PARSING_DICT.items():
            if action_str.startswith(action):
                return action_id
        raise ValueError(f"Action string '{action_str}' does not match any known action.")
        
    def __call__(self, domain_file, problem_file, horizon=np.inf, timeout=30,
                 return_files=False, parse_actions=False, optimal=False):

        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        # copy the domain and problem files to temporary files
        with open(domain_file, 'r') as f:
            with open(dom_file, 'w') as temp_f:
                temp_f.write(f.read())
        with open(problem_file, 'r') as f:
            with open(prob_file, 'w') as temp_f:
                temp_f.write(f.read())
        
        pddl_plan = self.plan_from_pddl(
            dom_file, prob_file, horizon=horizon,
            timeout=timeout, remove_files=(not return_files), optimal=optimal)

        plan = []
        
        if parse_actions:
            for pddl_single_plan in pddl_plan:
                parsed_plan = [
                    self._parse_action(plan_step)
                    for plan_step in pddl_single_plan
                ]
                plan.append(parsed_plan)
        else:
            plan = pddl_plan

        if return_files:
            return plan, dom_file, prob_file
        return plan


def generate_single_instance(args_tuple):
    """Generate a single PDDL instance and plan in a separate working directory.
    
    Args:
        args_tuple: (map_id, domain_name, timeout, optimal, keep_instance_file, working_dir)
    
    Returns:
        tuple: (success, map_id, domain_name, error_message)
    """
    map_id, domain_name, timeout, optimal, keep_instance_file, working_dir = args_tuple
    
    # Set seeds for this specific instance
    random.seed(map_id)
    np.random.seed(map_id)
    
    # Change to working directory
    original_dir = os.getcwd()
    os.chdir(working_dir)
    
    pddl_problem_file_path = None
    plan_file_path = None
    
    try:
        # Step 1: Generate PDDL problem file
        pddl_problem_file_path, register_env_choice = pddl_problem_gen_main(
            path=None,
            map_id=map_id,
            domain_name=domain_name,
            if_display=False,
        )
        
        if not pddl_problem_file_path.exists():
            raise FileNotFoundError(f"PDDL problem file {pddl_problem_file_path} does not exist.")
        
        # Step 2: Generate PDDL plan
        my_planner = FastDownward()
        
        domain_file = Path(__file__).parent / "minibehavior.pddl"
        
        if not domain_file.exists():
            raise FileNotFoundError(f"PDDL domain file {domain_file} does not exist.")
        
        if domain_name == "setting_up_candles":
            special_domain_file = tempfile.NamedTemporaryFile(suffix=".pddl", delete=False)
            with open(domain_file, 'r') as f:
                with open(special_domain_file.name, 'w') as ff:
                    precon_edit_flag = False 
                    effect_edit_flag = False
                    for line in f:
                        if "(:predicates" in line:
                            special_content = """(:predicates
    (place-once-flag ?o - normal-item)
"""
                            ff.write(special_content)
                        elif '(:action pickup_' in line:
                            precon_edit_flag = True
                            effect_edit_flag = True
                            ff.write(line)
                        elif precon_edit_flag and ':precondition' in line:
                            updated_line = line + "\n" + "    (not (place-once-flag ?o))\n"
                            precon_edit_flag = False
                            ff.write(updated_line)
                        elif effect_edit_flag and ':effect' in line:
                            updated_line = line + "\n" + "    (place-once-flag ?o)\n"
                            effect_edit_flag = False
                            ff.write(updated_line)
                        else:
                            ff.write(line)
            
            domain_file = special_domain_file.name
        
        # Generate plans
        plans = my_planner(domain_file, pddl_problem_file_path, timeout=timeout, optimal=optimal)
        
        if len(plans) == 0:
            print(f"No plans found for domain {domain_name} and map_id {map_id}.")
            raise ValueError("No plans found.")
        
        # Save the plans to a JSON file
        plan_info_dict = {
            "env_id": register_env_choice,
            "map_id": map_id,
            "domain_name": domain_name,
            "plans": plans,
        }
        
        plan_filename = pddl_problem_file_path.name.replace('.pddl', '_plans.json')
        plan_file_path = Path(os.path.join(os.path.dirname(__file__), f"../pddl_plans/{domain_name}/{plan_filename}"))
        plan_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_file_path, 'w') as f:
            json.dump(plan_info_dict, f, indent=2)
        
        # Clean up working directory
        os.chdir(original_dir)
        shutil.rmtree(working_dir, ignore_errors=True)
        
        return (True, map_id, domain_name, None)
        
    except Exception as e:
        error_msg = f"Error generating plan for map_id={map_id}, domain={domain_name}: {e}"
        
        # Clean up files if needed
        if not keep_instance_file:
            if pddl_problem_file_path is not None and pddl_problem_file_path.exists():
                pddl_problem_file_path.unlink(missing_ok=True)
            if plan_file_path is not None and plan_file_path.exists():
                plan_file_path.unlink(missing_ok=True)
        
        # Clean up working directory
        os.chdir(original_dir)
        shutil.rmtree(working_dir, ignore_errors=True)
        
        return (False, map_id, domain_name, error_msg)

    
    
if __name__ == "__main__":
    
    # example python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/gen_and_plan.py --map_id 3368913918 --domain_name collect_misplaced_items --keep_instance_file --optimal --timeout 30m
    
    # then example run python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/single_agent_gameplay_test.py --map_id 3368913918 --domain_name collect_misplaced_items --no_display
    import argparse
    parser = argparse.ArgumentParser(description="Generate PDDL problem and plan.")
    parser.add_argument("--map_id", type=int, help="Map ID to generate.")
    parser.add_argument("--num_output", type=int, default=1, help="Number of problems instances to generate.")
    parser.add_argument("--timeout", type=str, default="60", help="Timeout for planner")
    parser.add_argument("--optimal", action='store_true', help="Use optimal planning (default: False).")
    parser.add_argument("--domain_name", type=str, help="Domain name for the PDDL problem.")
    parser.add_argument("--keep_instance_file", action='store_true', help="Keep the instance file after generation.")
    parser.add_argument("--max_processes", type=int, help="Maximum number of parallel processes to use.")
    
    args = parser.parse_args()
    raw_map_id = args.map_id
    num_output = args.num_output
    timeout = args.timeout
    optimal = args.optimal
    domain_name = args.domain_name
    keep_instance_file = args.keep_instance_file
    max_processes = args.max_processes
    
    if max_processes is None:
        max_processes = int(os.cpu_count() * 0.93)
    
    if raw_map_id is not None and num_output > 1:
        raise ValueError("If map_id is provided, num_output must be 1.")
    
    current_time = time.time()
    random.seed(int(current_time))
    np.random.seed(int(current_time))
    
    # Select domain names
    if domain_name is None:
        domain_name_lst = np.random.choice(list(GOAL_CLAUSES.keys()), size=num_output, replace=True)
        domain_name_lst = domain_name_lst.tolist()
    else:
        domain_name_lst = [domain_name] * num_output
    
    # Generate map IDs
    already_ids = set()
    task_args = []
    
    for i in range(num_output):
        if raw_map_id is None:
            map_id = random.randint(0, 2**32 - 1)
            while map_id in already_ids:
                map_id = random.randint(0, 2**32 - 1)
            already_ids.add(map_id)
        else:
            map_id = raw_map_id
        
        # Create a unique working directory for this task
        working_dir = tempfile.mkdtemp(prefix=f"pddl_{domain_name_lst[i]}_m{map_id}_")
        
        task_args.append((map_id, domain_name_lst[i], timeout, optimal, keep_instance_file, working_dir))
    
    print(f"Starting parallel generation of {num_output} instances with {max_processes} processes...")
    
    # Execute tasks in parallel
    success_count = 0
    failed_tasks = []
    
    with Pool(processes=max_processes) as pool:
        for success, map_id, domain, error_msg in tqdm(
            pool.imap_unordered(generate_single_instance, task_args),
            total=len(task_args),
            desc="Generating PDDL problems and plans",
            unit="instance"
        ):
            if success:
                success_count += 1
            else:
                print(f"\n{error_msg}")
                failed_tasks.append((map_id, domain))
    
    print(f"\nGeneration complete: {success_count}/{num_output} instances succeeded.")
    
    if failed_tasks:
        print("\nFailed instances:")
        for map_id, domain in failed_tasks:
            print(f"  - map_id: {map_id}, domain: {domain}")
    
    print("End of generation.")


            
            
            