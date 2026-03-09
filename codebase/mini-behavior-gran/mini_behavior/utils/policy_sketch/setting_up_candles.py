from pathlib import Path
from typing import List
import os
from mini_behavior.utils.policy_sketch.general_features import Feature, Count, DistanceToNearest, Holding, count_not_complete_first_layer_salad, count_not_complete_tables, isOpened, isToggled, isSoaked, isDustFree, isCleaned
from mini_behavior.utils.policy_sketch.general_policy_sketch import GeneralPolicySketch, Rule
from functools import partial
from mini_behavior.utils.policy_sketch.save_n_load_env import init_env_for_policy_sketch, save_gps_data
import argparse
from mini_behavior.utils.policy_sketch.general_features import (
    count_isolated,
    count_closed,
    count_not_soaked,
    count_not_cleaned,
    count_not_inside_target_object,
    count_incomplete_salad_place,
    count_not_inside_target_type,
    count_not_near_target_type,
    count_not_ontop,
    count_not_ontop_sometype,
    count_not_sliced)
"""
8
### Feature set (mirroring the plywood features)
- $H$: “holding a candle”
- $p$: distance to the nearest candle
- $b$: distance to a table
- $N$: number of unplaced candles
- $N_t$: number of unsettled tables (tables with less than 3 candles)


### Sketch with W = 0 (approximate)

$\{\neg H, N > 0, N_t > 0, p > 0\} \mapsto \{p \downarrow, b ?\}$  ; move toward the nearest candle
$\{\neg H, N > 0, N_t > 0, p = 0\} \mapsto \{H, b ?\}$  ; pick up the candle when reachable
$\{H, N > 0, N_t > 0, b > 0\} \mapsto \{b \downarrow\}$  ; move toward a table
$\{H, N > 0, N_t > 0, b = 0\} \mapsto \{\neg H, N \downarrow, N_t?, p=0\}$  ; place the candle on the table when at the spot
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "setting up candles": []
    }
    # 1. move toward the nearest candle
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='candle',
    )
    n_larger_0 = Count(
        type_tag='candle',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    
    n_t_larger_0 = Count(
        type_tag='table',
        wanted_value='>0',
        count_func=count_not_complete_tables,
    )
    
    p_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='candle',
    )
    # * effects
    p_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='candle',
    )
    
    rule_move_to_candle = Rule(
        preconditions=[neg_H, n_larger_0, n_t_larger_0, p_larger_0],
        effects=[p_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {candle})
"""
    )
    sketch_dict["setting up candles"].append(rule_move_to_candle)
    # 2. pick up the candle when reachable
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='candle',
    )
    n_larger_0 = Count(
        type_tag='candle',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    
    n_t_larger_0 = Count(
        type_tag='table',
        wanted_value='>0',
        count_func=count_not_complete_tables,
    )
    
    p_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='candle',
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='candle',
    )
    rule_pick_up_candle = Rule(
        preconditions=[neg_H, n_larger_0, n_t_larger_0, p_equal_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {candle})
"""
    )
    sketch_dict["setting up candles"].append(rule_pick_up_candle)
    # 3. move toward a table
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='candle',
    )
    n_larger_0 = Count(
        type_tag='candle',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    
    n_t_larger_0 = Count(
        type_tag='table',
        wanted_value='>0',
        count_func=count_not_complete_tables,
    )
    
    b_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='table',
    )
    # * effects 
    b_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='table',
    )
    rule_move_to_table = Rule(
        preconditions=[H, n_larger_0, n_t_larger_0, b_larger_0],
        effects=[b_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {table})
"""
    )
    sketch_dict["setting up candles"].append(rule_move_to_table)
    # 4. place the candle on the table when at the spot
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='candle',
    )
    n_larger_0 = Count(
        type_tag='candle',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    n_t_larger_0 = Count(
        type_tag='table',
        wanted_value='>0',
        count_func=count_not_complete_tables,
    )
    b_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='table',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='candle',
    )
    n_decrease = Count(
        type_tag='candle',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    
    rule_put_on_table = Rule(
        preconditions=[H, n_larger_0, n_t_larger_0, b_equal_0],
        effects=[neg_H, n_decrease],
        goal_clause_pattern="""
(onTop {candle} {table})
"""
    )
    sketch_dict["setting up candles"].append(rule_put_on_table)
    
    goal_count_feature = [
        Count(
            type_tag='candle',
            wanted_value="=0",
            count_func=partial(count_not_ontop_sometype, surface_type='table'),
        ),
        Count(
            type_tag='table',
            wanted_value="=0",
            count_func=count_not_complete_tables,
        )
    ]
    
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
# $\{\neg H, N > 0\} \mapsto \{H\}$  ; pick up an unboxed book
# $\{H, N > 0\} \mapsto \{H?, N \downarrow\}$  ; place the carried book into some valid box spot

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "setting up candles": []
    }
    # 1. pick up an unplaced candle
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='candle',
    )
    n_larger_0 = Count(
        type_tag='candle',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    
    n_t_larger_0 = Count(
        type_tag='table',
        wanted_value='>0',
        count_func=count_not_complete_tables,
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='candle',
    )
    rule_pick_up_candle = Rule(
        preconditions=[neg_H, n_larger_0, n_t_larger_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {candle})
"""
    )
    sketch_dict["setting up candles"].append(rule_pick_up_candle)
    # 2. place the carried candle into some valid table spot
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='candle',
    )
    n_larger_0 = Count(
        type_tag='candle',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    
    n_t_larger_0 = Count(
        type_tag='table',
        wanted_value='>0',
        count_func=count_not_complete_tables,
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='candle',
    )
    n_decrease = Count(
        type_tag='candle',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    
    rule_put_on_table = Rule(
        preconditions=[H, n_larger_0, n_t_larger_0],
        effects=[neg_H, n_decrease],
        goal_clause_pattern="""
(onTop {candle} {table})
"""
    )
    sketch_dict["setting up candles"].append(rule_put_on_table)
    goal_count_feature = [
        Count(
            type_tag='candle',
            wanted_value="=0",
            count_func=partial(count_not_ontop_sometype, surface_type='table'),
        ),
        Count(
            type_tag='table',
            wanted_value="=0",
            count_func=count_not_complete_tables,
        )
    ]
    
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

### Sketch with W = 2
# $\{N > 0\} \mapsto \{N \downarrow\}$  ; move a book into a box
    
def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "setting up candles": []
    }
    # 1. move a candle into a table
    # * precons
    n_larger_0 = Count(
        type_tag='candle',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    
    n_t_larger_0 = Count(
        type_tag='table',
        wanted_value='>0',
        count_func=count_not_complete_tables,
    )
    # * effects
    n_decrease = Count(
        type_tag='candle',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    
    rule_put_on_table = Rule(
        preconditions=[n_larger_0, n_t_larger_0],
        effects=[n_decrease],
        goal_clause_pattern="""
(onTop {candle} {table})
"""
    )
    sketch_dict["setting up candles"].append(rule_put_on_table)
    goal_count_feature = [
        Count(
            type_tag='candle',
            wanted_value="=0",
            count_func=partial(count_not_ontop_sometype, surface_type='table'),
        ),
        Count(
            type_tag='table',
            wanted_value="=0",
            count_func=count_not_complete_tables,
        )
    ]
    
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/setting_up_candles.py --map_id 89952986 --domain_name setting_up_candles --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/setting_up_candles/p89952986-setting_up_candles_plans.json', want_render=False)
    
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
        # a = 1
    