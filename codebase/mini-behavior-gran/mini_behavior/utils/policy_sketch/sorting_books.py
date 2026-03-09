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
20
### Feature set (mirroring the plywood features)
- $H$: “holding an unsorted book (book or hardback)”
- $p$: distance to the nearest unsorted book
- $s$: distance to a shelf
- $N$: number of unsorted books

### Sketch with W = 0 (approximate)
$\{\neg H, N > 0, p > 0\} \mapsto \{p \downarrow, s ?\}$  ; move toward the nearest unsorted book
$\{\neg H, N > 0, p = 0\} \mapsto \{H, s ?\}$  ; pick up the book when reachable
$\{H, N > 0, s > 0\} \mapsto \{s \downarrow\}$  ; move toward a shelf
$\{H, N > 0, s = 0\} \mapsto \{\neg H, N \downarrow, p=0\}$  ; place the book onto the shelf when at the spot
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "sort books": []
    }
    # 1. move toward the nearest unboxed book
    # * precons 
    neg_H = Holding(
        wanted_value=False,
        type_tag='book hardback',
    )
    
    p_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='book hardback',
    )
    n_larger_0 = Count(
        type_tag='book hardback',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    
    
    # * effects 
    p_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='book hardback',
    )
    
    rule_move_to_book = Rule(
        preconditions=[neg_H, p_larger_0, n_larger_0],
        effects=[p_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {book hardback})
"""
    )
    sketch_dict["sort books"].append(rule_move_to_book)
    # 2. pick up the book when reachable
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='book hardback',
    )
    p_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='book hardback',
    )
    n_larger_0 = Count(
        type_tag='book hardback',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='book hardback',
    )
    rule_pick_up_book = Rule(
        preconditions=[neg_H, p_equal_0, n_larger_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {book hardback})
"""
    )
    sketch_dict["sort books"].append(rule_pick_up_book)
    # 3. move toward a shelf
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='book hardback',
    )
    n_larger_0 = Count(
        type_tag='book hardback',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    s_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='shelf',
    )
    # * effects
    s_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='shelf',
    )
    rule_move_to_shelf = Rule(
        preconditions=[H, n_larger_0, s_larger_0],
        effects=[s_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {shelf})
"""
    )
    sketch_dict["sort books"].append(rule_move_to_shelf)
    # 4. place the book onto the shelf when at the spot
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='book hardback',
    )
    n_larger_0 = Count(
        type_tag='book hardback',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    s_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='shelf',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='book hardback',
    )
    n_decrease = Count(
        type_tag='book hardback',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    
    rule_place_book_on_shelf = Rule(
        preconditions=[H, n_larger_0, s_equal_0],
        effects=[neg_H, n_decrease],
        goal_clause_pattern="""
(ontop {book hardback} {shelf})
"""
    )
    sketch_dict["sort books"].append(rule_place_book_on_shelf)
    
    
    goal_count_feature = Count(
        type_tag='book hardback',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
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
# $\{\neg H\} \mapsto \{H\}$  ; pick up a book (width 1)
# $\{H, N > 0\} \mapsto \{H?, N \downarrow\}$  ; place the carried book onto some valid shelf (width 1)

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "sort books": []
    }
    # 1. pick up a book
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='book hardback',
    )
    n_larger_0 = Count(
        type_tag='book hardback',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='book hardback',
    )
    rule_pick_up_book = Rule(
        preconditions=[neg_H, n_larger_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {book hardback})
"""
    )
    sketch_dict["sort books"].append(rule_pick_up_book)
    # 2. place the carried book onto some valid shelf
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='book hardback',
    )
    n_larger_0 = Count(
        type_tag='book hardback',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='book hardback',
    )
    n_decrease = Count(
        type_tag='book hardback',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    rule_place_book_on_shelf = Rule(
        preconditions=[H, n_larger_0],
        effects=[neg_H, n_decrease],
        goal_clause_pattern="""
(ontop {book hardback} {shelf})
"""
    )
    sketch_dict["sort books"].append(rule_place_book_on_shelf)
    goal_count_feature = Count(
        type_tag='book hardback',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
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
# $\{N > 0\} \mapsto \{N \downarrow\}$  ; place a book onto a shelf (width 2)

def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "sort books": []
    }
    # 1. place a book onto a shelf
    # * precons
    n_larger_0 = Count(
        type_tag='book hardback',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    # * effects
    n_decrease = Count(
        type_tag='book hardback',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
    )
    rule_place_book_on_shelf = Rule(
        preconditions=[n_larger_0],
        effects=[n_decrease],
        goal_clause_pattern="""
(ontop {book hardback} {shelf})
"""
    )
    sketch_dict["sort books"].append(rule_place_book_on_shelf)
    goal_count_feature = Count(
        type_tag='book hardback',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='shelf'),
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/sorting_books.py --map_id 92062276 --domain_name sorting_books --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/sorting_books/p92062276-sorting_books_plans.json', want_render=False)
    
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
    
    