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
5
### Feature set (mirroring the plywood features)
- $H_{fish}$: “holding a fish”
- $H_{olive}$: “holding an olive”
- $H_{date}$: “holding a date”
- $p_{fish}$: distance to the nearest fish
- $p_{olive}$: distance to the nearest olive
- $p_{date}$: distance to the nearest date
- $p_{sink}$: distance to a sink
- $N_{fish}$: number of fish not next to a sink
- $N_{olive}$: number of olives not next to a sink
- $N_{date}$: number of dates not next to a fish

### Sketch with W = 0 (approximate, including distance features)

**Fish Task**
$\{\neg H_{fish}, p_{fish} > 0\} \mapsto \{p_{fish} \downarrow\}$  ; move toward the nearest fish
$\{\neg H_{fish}, p_{fish} = 0\} \mapsto \{H_{fish}\}$  ; pick up the fish when reachable
$\{H_{fish}, p_{sink} > 0\} \mapsto \{p_{sink} \downarrow\}$  ; move toward a chosen sink
$\{H_{fish}, p_{sink} = 0\} \mapsto \{N_{fish} \downarrow, \neg H_{fish}, p_{fish} ?\}$  ; put the fish next to the sink when reachable

**Olive Task**
$\{\neg H_{olive}, p_{olive} > 0\} \mapsto \{p_{olive} \downarrow\}$  ; move toward the nearest olive
$\{\neg H_{olive}, p_{olive} = 0\} \mapsto \{H_{olive}\}$  ; pick up the olive when reachable
$\{H_{olive}, p_{sink} > 0\} \mapsto \{p_{sink} \downarrow\}$  ; move toward a chosen sink
$\{H_{olive}, p_{sink} = 0\} \mapsto \{N_{olive} \downarrow, \neg H_{olive}, p_{olive} ?\}$  ; put the olive next to the sink when reachable

**Date Task**

$\{\neg H_{date}, N_{\text{fish olive}}= 0, p_{date} > 0\} \mapsto \{p_{date} \downarrow\}$  ; move toward the nearest date
$\{\neg H_{date}, N_{\text{fish olive}}= 0,p_{date} = 0\} \mapsto \{H_{date}\}$  ; pick up the date when reachable
$\{H_{date}, N_{\text{fish olive}}= 0,p_{fish} > 0\} \mapsto \{p_{fish} \downarrow\}$  ; move toward a chosen fish
$\{H_{date}, N_{\text{fish olive}}= 0, p_{fish} = 0\} \mapsto \{N_{date} \downarrow, \neg H_{date}, p_{date} ?\}$  ; put the date next to the fish when reachable
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0 
    sketch_dict = {
        "thawing fish and olives": [],
        "thawing dates": []
    }
    # 1 Thawing fish and olives task
    # 1.1. move toward the nearest fish
    # * precons
    neg_H_fish_olive = Holding(
        wanted_value=False,
        type_tag='fish olive',
    )
    p_fish_olive_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='fish olive',
    )
    n_fish_olive_larger_0 = Count(
        type_tag='fish olive',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    
    # * effects
    p_fish_olive_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='fish olive',
    )
    rule_move_to_fish_olive = Rule(
        preconditions=[neg_H_fish_olive, p_fish_olive_larger_0, n_fish_olive_larger_0],
        effects=[p_fish_olive_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {fish olive})
"""
    )
    sketch_dict["thawing fish and olives"].append(rule_move_to_fish_olive)
    # 1.2. pick up the fish/olive when reachable
    # * precons
    neg_H_fish_olive = Holding(
        wanted_value=False,
        type_tag='fish olive',
    )
    p_fish_olive_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='fish olive',
    )
    n_fish_olive_larger_0 = Count(
        type_tag='fish olive',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    # * effects
    H_fish_olive = Holding(
        wanted_value=True,
        type_tag='fish olive',
    )
    rule_pick_up_fish_olive = Rule(
        preconditions=[neg_H_fish_olive, p_fish_olive_equal_0, n_fish_olive_larger_0],
        effects=[H_fish_olive],
        goal_clause_pattern="""
(inhandofrobot agent-01 {fish olive})
"""
    )
    sketch_dict["thawing fish and olives"].append(rule_pick_up_fish_olive)
    # 1.3. move toward a chosen sink
    # * precons
    H_fish_olive = Holding(
        wanted_value=True,
        type_tag='fish olive',
    )
    p_sink_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='sink',
    )
    n_fish_olive_larger_0 = Count(
        type_tag='fish olive',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    # * effects
    p_sink_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='sink',
    )
    rule_move_to_sink_fish_olive = Rule(
        preconditions=[H_fish_olive, p_sink_larger_0, n_fish_olive_larger_0],
        effects=[p_sink_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    sketch_dict["thawing fish and olives"].append(rule_move_to_sink_fish_olive)
    # 1.4. put the fish/olive next to the sink when reachable
    # * precons
    H_fish_olive = Holding(
        wanted_value=True,
        type_tag='fish olive',
    )
    p_sink_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='sink',
    )
    n_fish_olive_larger_0 = Count(
        type_tag='fish olive',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    # * effects
    neg_H_fish_olive = Holding(
        wanted_value=False,
        type_tag='fish olive',
    )
    n_fish_olive_down = Count(
        type_tag='fish olive',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    
    rule_put_fish_olive_next_to_sink = Rule(
        preconditions=[H_fish_olive, p_sink_equal_0, n_fish_olive_larger_0],
        effects=[n_fish_olive_down, neg_H_fish_olive],
        goal_clause_pattern="""
(nextto {fish olive} {sink})
(not (inhandofrobot agent-01 {fish olive}))
"""
    )
    sketch_dict["thawing fish and olives"].append(rule_put_fish_olive_next_to_sink)
    
    # 2 Thawing dates task
    # 2.1. move toward the nearest date
    # * precons
    neg_H_date = Holding(
        wanted_value=False,
        type_tag='date',
    )
    p_date_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='date',
    )
    n_fish_olive_equal_0 = Count(
        type_tag='fish olive',
        wanted_value='=0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    n_date_larger_0 = Count(
        type_tag='date',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    # * effects
    p_date_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='date',
    )
    rule_move_to_date = Rule(
        preconditions=[neg_H_date, n_fish_olive_equal_0, p_date_larger_0, n_date_larger_0],
        effects=[p_date_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {date})
"""
    )
    sketch_dict["thawing dates"].append(rule_move_to_date)
    # 2.2. pick up the date when reachable
    # * precons
    neg_H_date = Holding(
        wanted_value=False,
        type_tag='date',
    )
    p_date_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='date',
    )
    n_fish_olive_equal_0 = Count(
        type_tag='fish olive',
        wanted_value='=0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    n_date_larger_0 = Count(
        type_tag='date',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    # * effects
    H_date = Holding(
        wanted_value=True,
        type_tag='date',
    )
    rule_pick_up_date = Rule(
        preconditions=[neg_H_date, p_date_equal_0, n_date_larger_0, n_fish_olive_equal_0],
        effects=[H_date],
        goal_clause_pattern="""
(inhandofrobot agent-01 {date})
"""
    )
    sketch_dict["thawing dates"].append(rule_pick_up_date)
    # 2.3. move toward a chosen fish
    # * precons
    H_date = Holding(
        wanted_value=True,
        type_tag='date',
    )
    p_fish_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='fish',
    )
    n_fish_olive_equal_0 = Count(
        type_tag='fish olive',
        wanted_value='=0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    n_date_larger_0 = Count(
        type_tag='date',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    # * effects
    p_fish_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='fish',
    ) 
    rule_move_to_fish = Rule(
        preconditions=[H_date, n_fish_olive_equal_0, p_fish_larger_0, n_date_larger_0],
        effects=[p_fish_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {fish})
"""
    )
    sketch_dict["thawing dates"].append(rule_move_to_fish)
    # 2.4. put the date next to the fish when reachable
    # * precons
    H_date = Holding(
        wanted_value=True,
        type_tag='date',
    )
    p_fish_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='fish',
    )
    n_fish_olive_equal_0 = Count(
        type_tag='fish olive',
        wanted_value='=0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    n_date_larger_0 = Count(
        type_tag='date',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    # * effects
    neg_H_date = Holding(
        wanted_value=False,
        type_tag='date',
    )
    n_date_down = Count(
        type_tag='date',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    rule_put_date_next_to_fish = Rule(
        preconditions=[H_date, p_fish_equal_0, n_date_larger_0, n_fish_olive_equal_0],
        effects=[n_date_down, neg_H_date],
        goal_clause_pattern="""
(nextto {date} {fish})
(not (inhandofrobot agent-01 {date}))
"""
    )
    sketch_dict["thawing dates"].append(rule_put_date_next_to_fish)
    
    goal_count_feature = [
        Count(
            type_tag='fish olive',
            wanted_value='=0',
            count_func=partial(count_not_near_target_type, target_type_name='sink'),
        ),
        Count(
            type_tag='date',
            wanted_value='=0',
            count_func=partial(count_not_near_target_type, target_type_name='fish'),
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

### Sketch with W = 1 ; separate carrying and placing
# $\{N_{fish} > 0, \neg H_{fish}\} \mapsto \{H_{fish}\}$  ; pick up a fish (width 1)
# $\{N_{fish} > 0, H_{fish}\} \mapsto \{N_{fish} \downarrow, \neg H_{fish}\}$  ; put the carried fish next to a sink (width 1)
# $\{N_{olive} > 0, \neg H_{olive}\} \mapsto \{H_{olive}\}$  ; pick up an olive (width 1)
# $\{N_{olive} > 0, H_{olive}\} \mapsto \{N_{olive} \downarrow, \neg H_{olive}\}$  ; put the carried olive next to a sink (width 1)
# $\{N_{date} > 0, \neg H_{date}\} \mapsto \{H_{date}\}$  ; pick up a date (width 1)
# $\{N_{date} > 0, H_{date}\} \mapsto \{N_{date} \downarrow, \neg H_{date}\}$  ; put the carried date next to a fish (width 1)

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "thawing fish and olives": [],
        "thawing dates": []
    } 
    # 1 Thawing fish and olives task
    # 1.1. pick up a fish/olive (width 1)
    # * precons
    neg_H_fish_olive = Holding(
        wanted_value=False,
        type_tag='fish olive',
    )
    n_fish_olive_larger_0 = Count(
        type_tag='fish olive',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    # * effects
    H_fish_olive = Holding(
        wanted_value=True,
        type_tag='fish olive',
    )
    rule_pick_up_fish_olive = Rule(
        preconditions=[neg_H_fish_olive, n_fish_olive_larger_0],
        effects=[H_fish_olive],
        goal_clause_pattern="""
(inhandofrobot agent-01 {fish olive})
"""
    )
    sketch_dict["thawing fish and olives"].append(rule_pick_up_fish_olive)
    # 1.2. put the carried fish/olive next to a sink (width 1)
    # * precons
    H_fish_olive = Holding(
        wanted_value=True,
        type_tag='fish olive',
    )
    n_fish_olive_larger_0 = Count(
        type_tag='fish olive',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    # * effects
    neg_H_fish_olive = Holding(
        wanted_value=False,
        type_tag='fish olive',
    )
    n_fish_olive_down = Count(
        type_tag='fish olive',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    rule_put_fish_olive_next_to_sink = Rule(
        preconditions=[H_fish_olive, n_fish_olive_larger_0],
        effects=[n_fish_olive_down, neg_H_fish_olive],
        goal_clause_pattern="""
(nextto {fish olive} {sink})
(not (inhandofrobot agent-01 {fish olive}))
"""
    )
    sketch_dict["thawing fish and olives"].append(rule_put_fish_olive_next_to_sink)
    # 2 Thawing dates task
    # 2.1. pick up a date (width 1)
    # * precons
    neg_H_date = Holding(
        wanted_value=False,
        type_tag='date',
    )
    n_fish_olive_equal_0 = Count(
        type_tag='fish olive',
        wanted_value='=0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    n_date_larger_0 = Count(
        type_tag='date',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    # * effects
    H_date = Holding(
        wanted_value=True,
        type_tag='date',
    )
    rule_pick_up_date = Rule(
        preconditions=[neg_H_date, n_date_larger_0, n_fish_olive_equal_0],
        effects=[H_date],
        goal_clause_pattern="""
(inhandofrobot agent-01 {date})
"""
    )
    sketch_dict["thawing dates"].append(rule_pick_up_date)
    # 2.2. put the carried date next to a fish (width 1)
    # * precons
    H_date = Holding(
        wanted_value=True,
        type_tag='date',
    )
    n_fish_olive_equal_0 = Count(
        type_tag='fish olive',
        wanted_value='=0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    n_date_larger_0 = Count(
        type_tag='date',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    # * effects
    neg_H_date = Holding(
        wanted_value=False,
        type_tag='date',
    )
    n_date_down = Count(
        type_tag='date',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    rule_put_date_next_to_fish = Rule(
        preconditions=[H_date, n_date_larger_0, n_fish_olive_equal_0],
        effects=[n_date_down, neg_H_date],
        goal_clause_pattern="""
(nextto {date} {fish})
(not (inhandofrobot agent-01 {date}))
"""
    )
    sketch_dict["thawing dates"].append(rule_put_date_next_to_fish)
    
    goal_count_feature = [
        Count(
            type_tag='fish olive',
            wanted_value='=0',
            count_func=partial(count_not_near_target_type, target_type_name='sink'),
        ),
        Count(
            type_tag='date',
            wanted_value='=0',
            count_func=partial(count_not_near_target_type, target_type_name='fish'),
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

### Sketch with W = 2 ; state each sub-goal (no mentioning of nextto)
# $\{N_{fish} > 0\} \mapsto \{N_{fish} \downarrow\}$  ; put fish next to a sink (width 2)
# $\{N_{olive} > 0\} \mapsto \{N_{olive} \downarrow\}$  ; put olives next to a sink (width 2)
# $\{N_{date} > 0\} \mapsto \{N_{date} \downarrow\}$  ; put dates next to a fish (width 2)

def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "thawing fish and olives": [],
        "thawing dates": []
    } 
    # 1 Thawing fish and olives task
    n_fish_olive_larger_0 = Count(
        type_tag='fish olive',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    # * effects
    n_fish_olive_down = Count(
        type_tag='fish olive',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    rule_put_fish_olive_next_to_sink = Rule(
        preconditions=[n_fish_olive_larger_0],
        effects=[n_fish_olive_down],
        goal_clause_pattern="""
(nextto {fish olive} {sink})
"""
    )
    sketch_dict["thawing fish and olives"].append(rule_put_fish_olive_next_to_sink)
    # 2 Thawing dates task
    n_fish_olive_equal_0 = Count(
        type_tag='fish olive',
        wanted_value='=0',
        count_func=partial(count_not_near_target_type, target_type_name='sink'),
    )
    n_date_larger_0 = Count(
        type_tag='date',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    # * effects
    n_date_down = Count(
        type_tag='date',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='fish'),
    )
    rule_put_date_next_to_fish = Rule(
        preconditions=[n_date_larger_0, n_fish_olive_equal_0],
        effects=[n_date_down],
        goal_clause_pattern="""
(nextto {date} {fish})
"""
    )
    sketch_dict["thawing dates"].append(rule_put_date_next_to_fish)

    goal_count_feature = [
        Count(
            type_tag='fish olive',
            wanted_value='=0',
            count_func=partial(count_not_near_target_type, target_type_name='sink'),
        ),
        Count(
            type_tag='date',
            wanted_value='=0',
            count_func=partial(count_not_near_target_type, target_type_name='fish'),
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
if __name__ == '__main__':
    main()
    # debug 
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/thawing_frozen_food.py --map_id 51026391 --domain_name thawing_frozen_food --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/thawing_frozen_food/p51026391-thawing_frozen_food_plans.json', want_render=False)
    
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