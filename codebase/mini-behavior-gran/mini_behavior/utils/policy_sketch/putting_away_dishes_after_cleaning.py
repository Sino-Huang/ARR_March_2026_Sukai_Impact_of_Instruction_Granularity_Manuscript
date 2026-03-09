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
10
### Feature set (mirroring the plywood features)
- $H$: “holding a plate”
- $p$: distance to the nearest plate
- $b$: distance to a valid cabinet target
- $N$: number of unboxed plates
- $is\_{opened}$: whether the cabinet is opened

### Sketch with W = 0 (approximate)

$\{\neg is\_opened, b > 0\} \mapsto \{b \downarrow\}$  ; move toward the cabinet
$\{\neg is\_opened, b = 0\} \mapsto \{is\_opened\}$  ; open the cabinet when reachable
$\{\neg H, N > 0, p > 0, is\_opened\} \mapsto \{p \downarrow, b ?\}$  ; move toward the nearest unboxed plate
$\{\neg H, N > 0, p = 0, is\_opened\} \mapsto \{H, b ?\}$  ; pick up the plate when reachable
$\{H, N > 0, b > 0, is\_opened\} \mapsto \{b \downarrow\}$  ; move toward a chosen cabinet spot
$\{H, N > 0, b = 0, is\_opened\} \mapsto \{\neg H, N \downarrow, p ?\}$  ; place the plate into the cabinet when at the spot
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "put away dishes after cleaning": []
    }
    # 1. move towards closed cabinet
    # * precons
    n_closed_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    b_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='cabinet',
    )
    # * effects
    b_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='cabinet',
    )
    rule_move_to_closed_cabinet = Rule(
        preconditions=[n_closed_larger_0, b_larger_0],
        effects=[b_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {cabinet})
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_move_to_closed_cabinet)
    # 2. open the cabinet when reachable
    n_closed_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    b_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='cabinet',
    )
    # * effects
    n_opened_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet = Rule(
        preconditions=[n_closed_larger_0, b_equal_0],
        effects=[n_opened_decrease],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_open_cabinet)
    
    # 3. move toward the nearest unboxed plate
    n_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    neg_H = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    p_larger_0 = DistanceToNearest(
        type_tag='plate',
        wanted_value='>0',
    )
    # * effects
    p_decrease = DistanceToNearest(
        type_tag='plate',
        wanted_value='decrease',
    )
    
    rule_move_to_plate = Rule(
        preconditions=[n_close_equal_0, neg_H, n_larger_0, p_larger_0],
        effects=[p_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {plate})
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_move_to_plate)
    # 4. pick up the plate when reachable
    n_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    neg_H = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    p_equal_0 = DistanceToNearest(
        type_tag='plate',
        wanted_value='=0',
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    
    rule_pick_up_plate = Rule(
        preconditions=[n_close_equal_0, neg_H, n_larger_0, p_equal_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {plate})
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_pick_up_plate)
    # 5. move toward a chosen cabinet spot
    n_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    H = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    b_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='cabinet',
    )
    # * effects
    b_down = DistanceToNearest(
        wanted_value='decrease',
        type_tag='cabinet',
    )
    rule_move_to_cabinet = Rule(
        preconditions=[n_close_equal_0, H, n_larger_0, b_larger_0],
        effects=[b_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {cabinet})
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_move_to_cabinet)
    # 6. place the plate into the cabinet when at the spot
    n_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    H = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    b_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='cabinet',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    n_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    rule_place_in_cabinet = Rule(
        preconditions=[n_close_equal_0, H, n_larger_0, b_equal_0],
        effects=[neg_H, n_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
     (inside {plate} {cabinet} ?loc ?dim)
)
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_place_in_cabinet)
    
    goal_count_feature = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
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

### Sketch with W = 1 ; separate carrying and placing
# $\{\neg is\_opened\} \mapsto \{is\_opened\}$  ; open the cabinet (width 1) 
# $\{\neg H\} \mapsto \{H\}$  ; pick up an unboxed plate (width 1)
# $\{H, N > 0, is\_opened\} \mapsto \{H?, N \downarrow\}$  ; place the carried plate into some valid cabinet spot (width 1)

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "put away dishes after cleaning": []
    }
    # 1. open the cabinet (width 1)
    # * precons
    n_closed_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    n_opened_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet = Rule(
        preconditions=[n_closed_larger_0],
        effects=[n_opened_decrease],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_open_cabinet)
    # 2. pick up an unboxed plate (width 1)
    n_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    neg_H = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    rule_pick_up_plate = Rule(
        preconditions=[n_close_equal_0, neg_H, n_larger_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {plate})
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_pick_up_plate)
    # 3. place the carried plate into some valid cabinet spot (width 1)
    n_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    H = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    n_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    rule_place_in_cabinet = Rule(
        preconditions=[n_close_equal_0, H, n_larger_0],
        effects=[neg_H, n_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
     (inside {plate} {cabinet} ?loc ?dim)
)
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_place_in_cabinet)
    goal_count_feature = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
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

### Sketch with W = 2,  separate opening cabinet 
# $\{\neg is\_opened\} \mapsto \{is\_opened\}$  ; open the cabinet (width 1)
# $\{N > 0, is\_opened\} \mapsto \{N \downarrow\}$  ; move a plate into a cabinet (width 2)

def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "put away dishes after cleaning": []
    }
    # 1. open the cabinet (width 1)
    # * precons
    n_closed_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    n_opened_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet = Rule(
        preconditions=[n_closed_larger_0],
        effects=[n_opened_decrease],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_open_cabinet)
    # 2. move a plate into a cabinet (width 2)
    n_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    # * effects
    n_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    rule_place_in_cabinet = Rule(
        preconditions=[n_close_equal_0, n_larger_0],
        effects=[n_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
     (inside {plate} {cabinet} ?loc ?dim)
)
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_place_in_cabinet)
    goal_count_feature = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
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

### Sketch with W = 3
# $\{N > 0\} \mapsto \{N \downarrow\}$  ; move a plate into a cabinet

def create_width_3_sketch(env, map_id, domain_name, window=None):
    width = 3
    sketch_dict = {
        "put away dishes after cleaning": []
    }
    # 1. move a plate into a cabinet (width 3)
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    # * effects
    n_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
    )
    rule_place_in_cabinet = Rule(
        preconditions=[n_larger_0],
        effects=[n_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
     (inside {plate} {cabinet} ?loc ?dim)
)
"""
    )
    sketch_dict["put away dishes after cleaning"].append(rule_place_in_cabinet)
    goal_count_feature = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet')
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
if __name__ == "__main__":
    main()
    # debug 
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/putting_away_dishes_after_cleaning.py --map_id 45646795 --domain_name putting_away_dishes_after_cleaning --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/putting_away_dishes_after_cleaning/p45646795-putting_away_dishes_after_cleaning_plans.json', want_render=False)
    
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