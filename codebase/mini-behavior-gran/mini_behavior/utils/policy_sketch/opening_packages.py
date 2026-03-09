from pathlib import Path
from typing import List
import os
from mini_behavior.utils.policy_sketch.general_features import Feature, Count, DistanceToNearest, Holding, count_not_complete_first_layer_salad, isOpened, isToggled, isSoaked, isDustFree, isCleaned
from mini_behavior.utils.policy_sketch.general_policy_sketch import GeneralPolicySketch, Rule
from functools import partial
from mini_behavior.utils.policy_sketch.save_n_load_env import init_env_for_policy_sketch, save_gps_data
import argparse
from mini_behavior.utils.policy_sketch.general_features import (
    count_isolated,
    count_closed,
    count_not_soaked,
    count_not_cleaned,
    count_not_wiped,
    count_not_inside_target_object,
    count_incomplete_salad_place,
    count_not_inside_target_type,
    count_not_near_target_type,
    count_not_ontop,
    count_not_ontop_sometype,
    count_not_onfloor,
    count_not_samelocation_sometype,
    count_not_toggled,
    count_not_sliced)
"""
7
### Feature set (mirroring the plywood features)
- $p$: distance to the nearest package
- $N$: number of unopened packages

### Sketch with W = 0 (approximate, including distance features)
$\{N > 0, p > 0\} \mapsto \{p \downarrow\}$  ; move toward the nearest unopened package
$\{N > 0, p = 0\} \mapsto \{N \downarrow\}$  ; open the package when reachable
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "open packages": []
    }
    # 1. move toward the nearest unopened package
    # * precons 
    N_larger_0 = Count(
        type_tag='package',
        wanted_value='>0',
        count_func=count_closed,
    )
    
    p_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='package',
    )
    
    # * effects 
    p_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='package',
    )
    
    rule_move_to_package = Rule(
        preconditions=[N_larger_0, p_larger_0],
        effects=[p_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {package})       
"""
    )
    sketch_dict["open packages"].append(rule_move_to_package)
    
    # 2. open the package when reachable
    N_larger_0 = Count(
        type_tag='package',
        wanted_value='>0',
        count_func=count_closed,
    )
    p_zero = DistanceToNearest(
        wanted_value='=0',
        type_tag='package',
    )
    # * effects
    N_decrease = Count(
        type_tag='package',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_package = Rule(
        preconditions=[N_larger_0, p_zero],
        effects=[N_decrease],
        goal_clause_pattern="""
(is-opened {package})
"""
    )
    sketch_dict["open packages"].append(rule_open_package)
    
    goal_count_feature = Count(
        type_tag='package',
        wanted_value='=0',
        count_func=count_closed,
    )
    
    domain_filepath = os.path.join(os.environ['WORKING_DIR'], 'data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/minibehavior.pddl')
    
    general_policy_sketch = GeneralPolicySketch(
        env=env,
        domain_filepath = domain_filepath,
        width = width,
        sketch_dict = sketch_dict,
        goal_count_feature= goal_count_feature,
        map_id= map_id,
        domain_name = domain_name,
        window=window,
    )
    
    return general_policy_sketch

### Sketch with W = 1
# $\{N > 0\} \mapsto \{N \downarrow\}$  ; open a package (width 1)

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "open packages": []
    }
    
    # 1. open a package
    # * precons
    N_larger_0 = Count(
        type_tag='package',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    N_decrease = Count(
        type_tag='package',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_package = Rule(
        preconditions=[N_larger_0],
        effects=[N_decrease],
        goal_clause_pattern="""
(is-opened {package})
"""
    )
    sketch_dict["open packages"].append(rule_open_package)
    
    goal_count_feature = Count(
        type_tag='package',
        wanted_value='=0',
        count_func=count_closed,
    )
    
    domain_filepath = os.path.join(os.environ['WORKING_DIR'], 'data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/minibehavior.pddl')
    
    general_policy_sketch = GeneralPolicySketch(
        env=env,
        domain_filepath = domain_filepath,
        width = width,
        sketch_dict = sketch_dict,
        goal_count_feature= goal_count_feature,
        map_id= map_id,
        domain_name = domain_name,
        window=window,
    )
    
    return general_policy_sketch

def main():
    # create argument parser
    parser = argparse.ArgumentParser(description='Collect data for a specific map and domain using a policy sketch.')
    
    # add arguments
    parser.add_argument('--map_id', type=int, help='map id')
    parser.add_argument('--domain_name', type=str, help='domain name')
    parser.add_argument('--width', type=int, help='width for policy sketch')
    parser.add_argument('--rewrite', action='store_true', help='whether to overwrite existing data', default=False)
    parser.add_argument('--render', action='store_true', help='whether to render the environment', default=False)

    args = parser.parse_args()
    map_id = args.map_id
    domain_name = args.domain_name
    width = args.width
    rewrite = args.rewrite
    render = args.render
    
    env, map_id, domain_name, window, env_id = init_env_for_policy_sketch(f'data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/{domain_name}/p{map_id}-{domain_name}_plans.json', want_render=render)
    
    
    output_dir = os.path.join(os.environ['WORKING_DIR'], "data/02_intermediate/gps_data", domain_name)
    
    output_pickle_filename = os.path.join(output_dir, f'gps_data_map_{map_id}_width_{width}_env_{env_id}_goal_True.pkl')
    
    if os.path.exists(output_pickle_filename):
        if not rewrite:
            print(f"Data file {output_pickle_filename} already exists. Skipping...")
            return
        else:
            print(f"Data file {output_pickle_filename} already exists. Overwriting as --rewrite is set.")
    create_sketch_func_name = f"create_width_{width}_sketch"
    if create_sketch_func_name not in globals():
        raise ValueError(f"Function {create_sketch_func_name} not found.")
    else:
        create_sketch_func = globals()[create_sketch_func_name]

    gps = create_sketch_func(env, map_id, domain_name, window) # type: GeneralPolicySketch

    goal_achieved, stored_info = gps.evaluate_full_sketch()
    
    save_gps_data(
        stored_info = stored_info,
        goal_achieved = goal_achieved,
        map_id = map_id,
        domain_name = domain_name,
        width = width,
        env_id = env_id,
        goal_count_feature = gps.goal_count_feature
    )
if __name__ == '__main__':
    main()
    # debug 
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/opening_packages.py --map_id 372273396 --domain_name opening_packages --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/opening_packages/p372273396-opening_packages_plans.json', want_render=False)
    
    # gps = create_width_0_sketch(env, map_id, domain_name, window)
    
    # goal_achieved, stored_info = gps.evaluate_full_sketch()
    # # .stored_info ={
    #     #     "performed_rules": [], # list of (count, subgoal, rule, actions)
    #     #     "executed_actions": [], # list of executed action strings
    #     #     "map_id": self.map_id,
    #     #     "domain_name": self.domain_name,
    #     # }

    # if not goal_achieved: # debug
    #     breakpoint()
    #     a = 1