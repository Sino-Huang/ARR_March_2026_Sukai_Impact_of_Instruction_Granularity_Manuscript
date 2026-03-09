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
    count_not_toggled,
    count_not_ontop_sometype,
    count_not_onfloor,
    count_not_sliced)
"""
13
### Feature set (mirroring the plywood features)
- $H$: “holding a printer”
- $p$: distance to the nearest printer
- $b$: distance to a table
- $is\_{toggled}$: whether the printer is toggled on
- $N$: number of uninstalled printers


### Sketch with W = 0 (approximate)

$\{\neg H, N > 0, p > 0\} \mapsto \{p \downarrow, b ?\}$  ; move toward the nearest printer
$\{\neg H, \neg is\_toggled = 0, N > 0, p=0\} \mapsto \{\neg H, is\_toggled\}$  ; turn on the printer when reachable and not toggled yet
$\{\neg H, is\_toggled, N > 0, p = 0\} \mapsto \{H, b ?\}$  ; pick up the printer when reachable
$\{H, N > 0, b > 0\} \mapsto \{b \downarrow\}$  ; move toward a table
$\{H, N > 0, b = 0\} \mapsto \{\neg H, N \downarrow, p=0\}$  ; place the printer on the table when at the spot

"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0 
    sketch_dict = {
        "install a printer": []
    }
    # 1. move toward the nearest printer
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='printer',
    )
    n_larger_0 = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
    )
    p_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='printer',
    )
    # * effects
    p_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='printer',
    )
    rule_move_to_printer = Rule(
        preconditions=[neg_H, n_larger_0, p_larger_0],
        effects=[p_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {printer})
"""
    ) 
    sketch_dict["install a printer"].append(rule_move_to_printer)
    # 2. turn on the printer when reachable and not toggled yet
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='printer',
    )
    neg_toggled = isToggled(
        wanted_value=False,
        type_tag='printer',
    )
    n_larger_0 = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
    )
    p_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='printer',
    )
    # * effects
    toggled = isToggled(
        wanted_value=True,
        type_tag='printer',
    )
    neg_H_2 = Holding(
        wanted_value=False,
        type_tag='printer',
    )
    rule_turn_on_printer = Rule(
        preconditions=[neg_H, neg_toggled, n_larger_0, p_equal_0],
        effects=[neg_H_2, toggled],
        goal_clause_pattern="""
(is-toggled {printer})
(not (inhandofrobot agent-01 {printer}))
"""
    )
    sketch_dict["install a printer"].append(rule_turn_on_printer)
    # 3. pick up the printer when reachable
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='printer',
    )
    toggled = isToggled(
        wanted_value=True,
        type_tag='printer',
    )
    n_larger_0 = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
    )
    p_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='printer',
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='printer',
    )
    rule_pick_up_printer = Rule(
        preconditions=[neg_H, toggled, n_larger_0, p_equal_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {printer})
"""
    )
    sketch_dict["install a printer"].append(rule_pick_up_printer)
    # 4. move toward a table
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='printer',
    )
    n_larger_0 = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
    )
    b_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='table',
    )
    # * effects
    b_down = DistanceToNearest(
        wanted_value='decrease',
        type_tag='table',
    )
    rule_move_to_table = Rule(
        preconditions=[H, n_larger_0, b_larger_0],
        effects=[b_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {table})
"""
    )
    sketch_dict["install a printer"].append(rule_move_to_table)
    # 5. place the printer on the table when at the spot
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='printer',
    )
    n_larger_0 = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
    )
    b_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='table',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='printer',
    )
    n_down = Count(
        type_tag='printer',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
    )
    
    rule_place_printer_on_table = Rule(
        preconditions=[H, n_larger_0, b_equal_0],
        effects=[neg_H, n_down],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {printer}))
(onTop {printer} {table})
"""
    )
    sketch_dict["install a printer"].append(rule_place_printer_on_table)
    
    goal_count_feature = Count(
        type_tag='printer',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
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
# $\{ \neg is\_toggled \} \mapsto \{ is\_toggled \}$  ; turn on a printer (width 1)
# $\{\neg H\} \mapsto \{H\}$  ; pick up a printer (width 1)
# $\{H, N > 0\} \mapsto \{H?, N \downarrow\}$  ; place the carried printer on a table (width 1)

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "install a printer": []
    }
    # 1. turn on a printer
    # * precons
    c_not_toggled = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=count_not_toggled,
    )
    # * effects
    c_not_toggled_down = Count(
        type_tag='printer',
        wanted_value='decrease',
        count_func=count_not_toggled,
    )
    rule_turn_on_printer = Rule(
        preconditions=[c_not_toggled],
        effects=[c_not_toggled_down],
        goal_clause_pattern="""
(is-toggled {printer})
"""
    )
    sketch_dict["install a printer"].append(rule_turn_on_printer)
    # 2. pick up a printer
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='printer',
    )
    c_not_toggled_equal_0 = Count(
        type_tag='printer',
        wanted_value='=0',
        count_func=count_not_toggled,
    )
    n_larger_0 = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='printer',
    )
    rule_pick_up_printer = Rule(
        preconditions=[neg_H, c_not_toggled_equal_0, n_larger_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {printer})
"""
    )
    sketch_dict["install a printer"].append(rule_pick_up_printer)
    # 3. place the carried printer on a table
    c_not_toggled_equal_0 = Count(
        type_tag='printer',
        wanted_value='=0',
        count_func=count_not_toggled,
    )
    n_larger_0 = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
    )
    H = Holding(
        wanted_value=True,
        type_tag='printer',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='printer',
    )
    n_down = Count(
        type_tag='printer',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
    )
    rule_place_printer_on_table = Rule(
        preconditions=[c_not_toggled_equal_0, n_larger_0, H],
        effects=[neg_H, n_down],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {printer}))
(onTop {printer} {table})
"""
    )
    sketch_dict["install a printer"].append(rule_place_printer_on_table)
    
    goal_count_feature = Count(
        type_tag='printer',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
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
    
    
### Sketch with W = 2
# $\{ \neg is\_toggled \} \mapsto \{ is\_toggled \}$  ; turn on a printer (width 1)
# $\{N > 0\} \mapsto \{N \downarrow\}$  ; place a printer on a table (width 2)
    
def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "install a printer": []
    }
    # 1. turn on a printer
    # * precons
    c_not_toggled = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=count_not_toggled,
    )
    # * effects
    c_not_toggled_down = Count(
        type_tag='printer',
        wanted_value='decrease',
        count_func=count_not_toggled,
    )
    rule_turn_on_printer = Rule(
        preconditions=[c_not_toggled],
        effects=[c_not_toggled_down],
        goal_clause_pattern="""
(is-toggled {printer})
"""
    )
    sketch_dict["install a printer"].append(rule_turn_on_printer)
    
    # 2. place a printer on a table
    n_larger_0 = Count(
        type_tag='printer',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table')
    )
    # * effects
    n_down = Count(
        type_tag='printer',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table')
    )
    rule_place_printer_on_table = Rule(
        preconditions=[n_larger_0],
        effects=[n_down],
        goal_clause_pattern="""
(onTop {printer} {table})
"""
    )
    sketch_dict["install a printer"].append(rule_place_printer_on_table)
    
    goal_count_feature = Count(
        type_tag='printer',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='table', additional_predicate_on_object_type= 'is-toggled')
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/installing_a_printer.py --map_id 10031077 --domain_name installing_a_printer --width 1 --rewrite
    # example 
    # python data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/installing_a_printer.py --map_id 10031077 --domain_name installing_a_printer --width 0
    
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/installing_a_printer/p10031077-installing_a_printer_plans.json', want_render=False)
    
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
    # else:
    #     print("Goal achieved!")
    #     print(gps.stored_info)
        