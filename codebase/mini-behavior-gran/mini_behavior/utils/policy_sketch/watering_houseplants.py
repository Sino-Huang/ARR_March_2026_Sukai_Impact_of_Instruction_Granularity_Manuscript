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
    count_not_inside_target_object,
    count_incomplete_salad_place,
    count_not_inside_target_type,
    count_not_near_target_type,
    count_not_toggled,
    count_not_ontop,
    count_not_sliced)
"""
15
### Feature set (mirroring the plywood features)
- $H$: “holding a pot plant”
- $p$: distance to the nearest pot plant
- $b$: distance to the sink
- $is\_{toggled}$: whether the sink is toggled on
- $N$: number of unsoaked pot plants


 ### Sketch with W = 0 (approximate)
 $\{\neg H, N > 0, p > 0\} \mapsto \{p \downarrow, b ?\}$  ; move toward the nearest pot plant
 $\{\neg H, N > 0, p = 0\} \mapsto \{H, b ?\}$  ; pick up the pot plant when reachable
 $\{\neg is\_toggled, b > 0\} \mapsto \{b \downarrow, p ?\}$  ; move toward the sink
 $\{\neg is\_toggled, b = 0\} \mapsto \{is\_toggled, p ?\}$  ; toggle the sink on when reachable
 $\{H, N > 0, is\_toggled, b > 0\} \mapsto \{b \downarrow\}$  ; move toward the sink
 $\{H, N > 0, is\_toggled, b = 0\} \mapsto \{\neg H, N \downarrow, p=0\}$  ; soak the pot plant in the sink when at the spot
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "soak pot plants": []
    }
    # 1. move toward the nearest pot plant
    # * precons 
    neg_H = Holding(
        wanted_value=False,
        type_tag='pot_plant',
    )
    
    N_larger_0 = Count(
        type_tag='pot_plant',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    
    is_toggled_sink = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    
    p_larger_0 = DistanceToNearest(
        type_tag = 'pot_plant',
        wanted_value = '>0',
    )
    
    # * effects
    p_decrease = DistanceToNearest(
        type_tag = 'pot_plant',
        wanted_value = 'decrease',
    )
    
    is_toggled_sink_2 = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    
    rule_move_towards_plant = Rule(
        preconditions = [neg_H, N_larger_0, p_larger_0, is_toggled_sink],
        effects = [p_decrease, is_toggled_sink_2],
        goal_clause_pattern="""
(inreachofrobot agent-01 {pot_plant})
"""
    )

    sketch_dict["soak pot plants"].append(rule_move_towards_plant)

    # 2. pick up the pot plant when reachable
    
    neg_H = Holding(
        wanted_value=False,
        type_tag='pot_plant',
    )
    
    N_larger_0 = Count(
        type_tag='pot_plant',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    
    p_equal_0 = DistanceToNearest(
        type_tag = 'pot_plant',
        wanted_value = '=0',
    )
    
    is_toggled_sink = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='pot_plant',
    )
    
    is_toggled_sink_2 = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    
    rule_pick_up_plant = Rule(
        preconditions = [neg_H, N_larger_0, p_equal_0, is_toggled_sink],
        effects = [H, is_toggled_sink_2],
        goal_clause_pattern="""
(inhandofrobot agent-01 {pot_plant})
"""
    )
    sketch_dict["soak pot plants"].append(rule_pick_up_plant)
    # 3. move toward the sink
    neg_is_toggled = isToggled(
        wanted_value=False,
        type_tag='sink',
    )
    
    b_larger_0 = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = '>0',
    )
    # * effects
    b_decrease = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = 'decrease',
    )
    rule_move_towards_sink = Rule(
        preconditions = [neg_is_toggled, b_larger_0],
        effects = [b_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    
    sketch_dict["soak pot plants"].append(rule_move_towards_sink)
    
    # 4. toggle the sink on when reachable
    neg_is_toggled = isToggled(
        wanted_value=False,
        type_tag='sink',
    )
    b_equal_0 = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = '=0',
    )
    
    # * effects
    is_toggled = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    rule_toggle_on_sink = Rule(
        preconditions = [neg_is_toggled, b_equal_0],
        effects = [is_toggled],
        goal_clause_pattern="""
(is-toggled {sink})
"""
    )
    
    sketch_dict["soak pot plants"].append(rule_toggle_on_sink)
    
    # 5. move toward the sink when holding a pot plant
    H = Holding(
        wanted_value=True,
        type_tag='pot_plant',
    )
    N_larger_0 = Count(
        type_tag='pot_plant',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    is_toggled = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    b_larger_0 = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = '>0',
    )
    # * effects
    b_decrease = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = 'decrease',
    )
    is_toggled_2 = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    rule_move_towards_sink_holding_plant = Rule(
        preconditions = [H, N_larger_0, is_toggled, b_larger_0],
        effects = [b_decrease, is_toggled_2],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    sketch_dict["soak pot plants"].append(rule_move_towards_sink_holding_plant)
    
    # 6. soak the pot plant in the sink when at the spot
    H = Holding(
        wanted_value=True,
        type_tag='pot_plant',
    )
    N_larger_0 = Count(
        type_tag='pot_plant',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    is_toggled = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    b_equal_0 = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = '=0',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='pot_plant',
    )
    N_decrease = Count(
        type_tag='pot_plant',
        wanted_value='decrease',
        count_func=count_not_soaked,
    )
    rule_soak_plant = Rule(
        preconditions = [H, N_larger_0, is_toggled, b_equal_0],
        effects = [neg_H, N_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {pot_plant}))
(is-soaked {pot_plant})
"""
    )
    sketch_dict["soak pot plants"].append(rule_soak_plant)
    
    goal_count_feature = Count(
        type_tag='pot_plant',
        wanted_value='=0',
        count_func=count_not_soaked,
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

### Sketch with W = 1 ; separate carrying and soaking
# $\{\neg is\_toggled\} \mapsto \{is\_toggled\}$  ; toggle the sink on (width 1)
# $\{\neg H\} \mapsto \{H\}$  ; pick up a pot plant (width 1)
# $\{H, N > 0, is\_toggled\} \mapsto \{H?, N \downarrow\}$  ; soak the carried pot plant in the sink (width 1)
def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "soak pot plants": []
    }
    # 1. toggle the sink on
    c_sink_not_toggled = Count(
        type_tag='sink',
        wanted_value='>0',
        count_func=count_not_toggled,
    )
    
    # * effects
    c_sink_toggled_decrease = Count(
        type_tag='sink',
        wanted_value='decrease',
        count_func=count_not_toggled,
    )
    rule_toggle_on_sink = Rule(
        preconditions = [c_sink_not_toggled],
        effects = [c_sink_toggled_decrease],
        goal_clause_pattern="""
(is-toggled {sink})
"""
    )
    sketch_dict["soak pot plants"].append(rule_toggle_on_sink)
    
    # 2. pick up a pot plant
    neg_H = Holding(
        wanted_value=False,
        type_tag='pot_plant',
    )
    is_toggled_sink = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    N_larger_0 = Count(
        type_tag='pot_plant',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='pot_plant',
    )
    rule_pick_up_plant = Rule(
        preconditions = [neg_H, is_toggled_sink, N_larger_0],
        effects = [H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {pot_plant})
"""
    )
    sketch_dict["soak pot plants"].append(rule_pick_up_plant)
    
    # 3. soak the carried pot plant in the sink
    H = Holding(
        wanted_value=True,
        type_tag='pot_plant',
    )
    N_larger_0 = Count(
        type_tag='pot_plant',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    is_toggled = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='pot_plant',
    )
    N_decrease = Count(
        type_tag='pot_plant',
        wanted_value='decrease',
        count_func=count_not_soaked,
    )
    rule_soak_plant = Rule(
        preconditions = [H, N_larger_0, is_toggled],
        effects = [neg_H, N_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {pot_plant}))
(is-soaked {pot_plant})
"""
    )
    sketch_dict["soak pot plants"].append(rule_soak_plant)
    
    goal_count_feature = Count(
        type_tag='pot_plant',
        wanted_value='=0',
        count_func=count_not_soaked,
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

### Sketch with W = 2 ; separate toggling sink
# $\{\neg is\_toggled\} \mapsto \{is\_toggled\}$  ; toggle the sink on (width 1)
# $\{N > 0, is\_toggled\} \mapsto \{N \downarrow, is\_toggled?\}$  ; soak a pot plant (width 2)

def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "soak pot plants": []
    }
    # 1. toggle the sink on
    c_sink_not_toggled = Count(
        type_tag='sink',
        wanted_value='>0',
        count_func=count_not_toggled,
    )
    
    # * effects
    c_sink_toggled_decrease = Count(
        type_tag='sink',
        wanted_value='decrease',
        count_func=count_not_toggled,
    )
    rule_toggle_on_sink = Rule(
        preconditions = [c_sink_not_toggled],
        effects = [c_sink_toggled_decrease],
        goal_clause_pattern="""
(is-toggled {sink})
"""
    )
    sketch_dict["soak pot plants"].append(rule_toggle_on_sink)
    # 2. soak a pot plant
    N_larger_0 = Count(
        type_tag='pot_plant',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    c_sink_toggle_equal_0 = Count(
        type_tag='sink',
        wanted_value='=0',
        count_func=count_not_toggled,
    )
    # * effects
    N_decrease = Count(
        type_tag='pot_plant',
        wanted_value='decrease',
        count_func=count_not_soaked,
    )
    rule_soak_plant = Rule(
        preconditions = [N_larger_0, c_sink_toggle_equal_0],
        effects = [N_decrease],
        goal_clause_pattern="""
(is-soaked {pot_plant})
"""
    )
    sketch_dict["soak pot plants"].append(rule_soak_plant)
    
    goal_count_feature = Count(
        type_tag='pot_plant',
        wanted_value='=0',
        count_func=count_not_soaked,
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
# $\{N > 0\} \mapsto \{N \downarrow\}$ ; soak a pot plant (width 3)
def create_width_3_sketch(env, map_id, domain_name, window=None):
    width = 3
    sketch_dict = {
        "soak pot plants": []
    }
    
    # 1. soak a pot plant
    N_larger_0 = Count(
        type_tag='pot_plant',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    # * effects
    N_decrease = Count(
        type_tag='pot_plant',
        wanted_value='decrease',
        count_func=count_not_soaked,
    )
    rule_soak_plant = Rule(
        preconditions = [N_larger_0],
        effects = [N_decrease],
        goal_clause_pattern="""
(is-soaked {pot_plant})
"""
    )
    sketch_dict["soak pot plants"].append(rule_soak_plant)
    
    goal_count_feature = Count(
        type_tag='pot_plant',
        wanted_value='=0',
        count_func=count_not_soaked,
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/watering_houseplants.py --map_id 122968710 --domain_name watering_houseplants --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/watering_houseplants/p122968710-watering_houseplants_plans.json', want_render=False)
    
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
    