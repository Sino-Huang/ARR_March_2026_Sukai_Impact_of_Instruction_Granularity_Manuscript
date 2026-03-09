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
    count_not_sliced)
"""
9
### Feature set (mirroring the plywood features)
- $H_s$: “holding a gym shoe”
- $H_n$: “holding a necklace”
- $H_b$: “holding a notebook”
- $H_k$: “holding a sock”
- $p_s$: distance to the nearest gym shoe
- $p_n$: distance to the nearest necklace
- $p_b$: distance to the nearest notebook
- $p_k$: distance to the nearest sock
- $p_t$: distance to a table
- $N_s$: number of gym shoes not on a table
- $N_n$: number of necklaces not on a table
- $N_b$: number of notebooks not on a table
- $N_k$: number of socks not on a table


### Sketch with W = 0 (approximate, including distance features)

**Gym shoe task**
$\{\neg H_s, N_s > 0, p_s > 0\} \mapsto \{p_s \downarrow, p_t ?\}$  ; move toward the nearest gym shoe
$\{\neg H_s, N_s > 0, p_s = 0\} \mapsto \{H_s, p_t ?\}$  ; pick up the gym shoe when reachable
$\{H_s, N_s > 0, p_t > 0\} \mapsto \{p_t \downarrow\}$  ; move toward a table
$\{H_s, N_s > 0, p_t = 0\} \mapsto \{N_s \downarrow, \neg H_s, p_s ?\}$  ; put the gym shoe on the table when reachable

**Necklace task**

$\{\neg H_n, N_n > 0, p_n > 0\} \mapsto \{p_n \downarrow, p_t ?\}$  ; move toward the nearest necklace
$\{\neg H_n, N_n > 0, p_n = 0\} \mapsto \{H_n, p_t ?\}$  ; pick up the necklace when reachable
$\{H_n, N_n > 0, p_t > 0\} \mapsto \{p_t \downarrow\}$  ; move toward a table
$\{H_n, N_n > 0, p_t = 0\} \mapsto \{N_n \downarrow, \neg H_n, p_n ?\}$  ; put the necklace on the table when reachable

**Notebook task**
$\{\neg H_b, N_b > 0, p_b > 0\} \mapsto \{p_b \downarrow, p_t ?\}$  ; move toward the nearest notebook
$\{\neg H_b, N_b > 0, p_b = 0\} \mapsto \{H_b, p_t ?\}$  ; pick up the notebook when reachable
$\{H_b, N_b > 0, p_t > 0\} \mapsto \{p_t \downarrow\}$  ; move toward a table
$\{H_b, N_b > 0, p_t = 0\} \mapsto \{N_b \downarrow, \neg H_b, p_b ?\}$  ; put the notebook on the table when reachable

**Sock task**
$\{\neg H_k, N_k > 0, p_k > 0\} \mapsto \{p_k \downarrow, p_t ?\}$  ; move toward the nearest sock
$\{\neg H_k, N_k > 0, p_k = 0\} \mapsto \{H_k, p_t ?\}$  ; pick up the sock when reachable
$\{H_k, N_k > 0, p_t > 0\} \mapsto \{p_t \downarrow\}$  ; move toward a table
$\{H_k, N_k > 0, p_t = 0\} \mapsto \{N_k \downarrow, \neg H_k, p_k ?\}$  ; put the sock on the table when reachable
"""

def helper_func_width_0(sketch_dict, type_tag, table_name, subgoal_name):
    # 1.1 move toward the nearest gym shoe
    # * precons
    neg_H_s = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    N_s_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    p_s_larger_0 = DistanceToNearest(
        type_tag=type_tag,
        wanted_value='>0',
    )
    # * effects
    p_s_decrease = DistanceToNearest(
        type_tag=type_tag,
        wanted_value='decrease',
    )
    rule_move_to_gym_shoe = Rule(
        preconditions=[neg_H_s, N_s_larger_0, p_s_larger_0],
        effects=[p_s_decrease],
        goal_clause_pattern=f"""
(inreachofrobot agent-01 {{{type_tag}}})
"""
    )
    sketch_dict[subgoal_name].append(rule_move_to_gym_shoe)
    # 1.2 pick up the gym shoe when reachable
    # * precons
    neg_H_s = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    p_s_equal_0 = DistanceToNearest(
        type_tag=type_tag,
        wanted_value='=0',
    )
    N_s_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    # * effects
    H_s = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    rule_pick_up_gym_shoe = Rule(
        preconditions=[neg_H_s, p_s_equal_0, N_s_larger_0],
        effects=[H_s],
        goal_clause_pattern=f"""
(inhandofrobot agent-01 {{{type_tag}}})
"""
    )
    sketch_dict[subgoal_name].append(rule_pick_up_gym_shoe)
    # 1.3 move toward a table
    # * precons
    H_s = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    N_s_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    p_t_larger_0 = DistanceToNearest(
        type_tag='table',
        wanted_value='>0',
        obj_name=table_name,
    )
    # * effects
    p_t_decrease = DistanceToNearest(
        type_tag='table',
        wanted_value='decrease',
        obj_name=table_name,
    )
    rule_move_to_table = Rule(
        preconditions=[H_s, N_s_larger_0, p_t_larger_0],
        effects=[p_t_decrease],
        goal_clause_pattern=f"""
(inreachofrobot agent-01 {table_name})
"""
    )

    sketch_dict[subgoal_name].append(rule_move_to_table)

    # 1.4 put the gym shoe on the table when reachable
    # * precons
    H_s = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    N_s_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    p_t_equal_0 = DistanceToNearest(
        type_tag='table',
        wanted_value='=0',
        obj_name=table_name,
    )
    # * effects
    neg_H_s = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    N_s_decrease = Count(
        type_tag=type_tag,
        wanted_value='decrease',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    rule_put_on_table = Rule(
        preconditions=[H_s, N_s_larger_0, p_t_equal_0],
        effects=[neg_H_s, N_s_decrease],
        goal_clause_pattern=f"""
(onTop {{{type_tag}}} {table_name})
(not (inhandofrobot agent-01 {{{type_tag}}}))
"""
    )
    sketch_dict[subgoal_name].append(rule_put_on_table)
    


def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "gym shoe": [],
        "necklace": [],
        "notebook": [],
        "sock": [],
    }
    # 1. gym shoe task
    helper_func_width_0(sketch_dict, type_tag='gym_shoe', table_name='table_0', subgoal_name='gym shoe')
    
    # 2. necklace task
    helper_func_width_0(sketch_dict, type_tag='necklace', table_name='table_0', subgoal_name='necklace')
    
    # 3. notebook task
    helper_func_width_0(sketch_dict, type_tag='notebook', table_name='table_1', subgoal_name='notebook')

    # 4. sock task
    helper_func_width_0(sketch_dict, type_tag='sock', table_name='table_1', subgoal_name='sock')
    
    goal_count_feature = [
        Count(
            type_tag='gym_shoe',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_0')
        ),
        Count(
            type_tag='necklace',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_0')
        ),
        Count(
            type_tag='notebook',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_1')
        ),
        Count(
            type_tag='sock',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_1')
        ),
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

# **Gym shoe task**
# $\{N_s > 0, \neg H_s\} \mapsto \{H_s\}$  ; pick up a gym shoe (width 1)
# $\{N_s > 0, H_s\} \mapsto \{N_s \downarrow, \neg H_s\}$  ; put the carried gym shoe on a table (width 1)
# **Necklace task**
# $\{N_n > 0, \neg H_n\} \mapsto \{H_n\}$  ; pick up a necklace (width 1)
# $\{N_n > 0, H_n\} \mapsto \{N_n \downarrow, \neg H_n\}$  ; put the carried necklace on a table (width 1)
# **Notebook task**
# $\{N_b > 0, \neg H_b\} \mapsto \{H_b\}$  ; pick up a notebook (width 1)
# $\{N_b > 0, H_b\} \mapsto \{N_b \downarrow, \neg H_b\}$  ; put the carried notebook on a table (width 1)
# **Sock task**
# $\{N_k > 0, \neg H_k\} \mapsto \{H_k\}$  ; pick up a sock (width 1)
# $\{N_k > 0, H_k\} \mapsto \{N_k \downarrow, \neg H_k\}$  ; put the carried sock on a table (width 1)

def helper_func_width_1(sketch_dict, type_tag, table_name, subgoal_name):
    # 1.1 move toward the nearest gym shoe
    # * precons
    neg_H_s = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    N_s_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    # * effects
    H_s = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    rule_pick_up_gym_shoe = Rule(
        preconditions=[neg_H_s, N_s_larger_0],
        effects=[H_s],
        goal_clause_pattern=f"""
(inhandofrobot agent-01 {{{type_tag}}})
"""
    )
    sketch_dict[subgoal_name].append(rule_pick_up_gym_shoe)
    # 1.2 put item on the table 
    # * precons
    H_s = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    N_s_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    # * effects
    neg_H_s = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    N_s_decrease = Count(
        type_tag=type_tag,
        wanted_value='decrease',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    rule_put_on_table = Rule(
        preconditions=[H_s, N_s_larger_0],
        effects=[neg_H_s, N_s_decrease],
        goal_clause_pattern=f"""
(onTop {{{type_tag}}} {table_name})
(not (inhandofrobot agent-01 {{{type_tag}}}))
"""
    )
    sketch_dict[subgoal_name].append(rule_put_on_table)


def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1 
    sketch_dict = {
        "gym shoe": [],
        "necklace": [],
        "notebook": [],
        "sock": [],
    }
    # 1. gym shoe task
    helper_func_width_1(sketch_dict, type_tag='gym_shoe', table_name='table_0', subgoal_name='gym shoe')
    
    # 2. necklace task
    helper_func_width_1(sketch_dict, type_tag='necklace', table_name='table_0', subgoal_name='necklace')
    
    # 3. notebook task
    helper_func_width_1(sketch_dict, type_tag='notebook', table_name='table_1', subgoal_name='notebook')

    # 4. sock task
    helper_func_width_1(sketch_dict, type_tag='sock', table_name='table_1', subgoal_name='sock')
    
    goal_count_feature = [
        Count(
            type_tag='gym_shoe',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_0')
        ),
        Count(
            type_tag='necklace',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_0')
        ),
        Count(
            type_tag='notebook',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_1')
        ),
        Count(
            type_tag='sock',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_1')
        ),
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
    
### Sketch with W = 2 ; state each sub-goal (no mentioning of onTop)
# $\{N_s > 0\} \mapsto \{N_s \downarrow\}$  ; put gym shoes on a table (width 2)
# $\{N_n > 0\} \mapsto \{N_n \downarrow\}$  ; put necklaces on a table (width 2)
# $\{N_b > 0\} \mapsto \{N_b \downarrow\}$  ; put notebooks on a table (width 2)
# $\{N_k > 0\} \mapsto \{N_k \downarrow\}$  ; put socks on a table (width 2)

def helper_func_width_2(sketch_dict, type_tag, table_name, subgoal_name):
    # 1.1 move toward the nearest gym shoe
    # * precons
    N_s_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    # * effects
    N_s_decrease = Count(
        type_tag=type_tag,
        wanted_value='decrease',
        count_func=partial(count_not_ontop, surface_obj_name=table_name)
    )
    rule_put_on_table = Rule(
        preconditions=[N_s_larger_0],
        effects=[N_s_decrease],
        goal_clause_pattern=f"""
(onTop {{{type_tag}}} {table_name})
"""
    )
    sketch_dict[subgoal_name].append(rule_put_on_table)
    
def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2 
    sketch_dict = {
        "gym shoe": [],
        "necklace": [],
        "notebook": [],
        "sock": [],
    }
    # 1. gym shoe task
    helper_func_width_2(sketch_dict, type_tag='gym_shoe', table_name='table_0', subgoal_name='gym shoe')
    
    # 2. necklace task
    helper_func_width_2(sketch_dict, type_tag='necklace', table_name='table_0', subgoal_name='necklace')
    
    # 3. notebook task
    helper_func_width_2(sketch_dict, type_tag='notebook', table_name='table_1', subgoal_name='notebook')

    # 4. sock task
    helper_func_width_2(sketch_dict, type_tag='sock', table_name='table_1', subgoal_name='sock')
    
    goal_count_feature = [
        Count(
            type_tag='gym_shoe',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_0')
        ),
        Count(
            type_tag='necklace',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_0')
        ),
        Count(
            type_tag='notebook',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_1')
        ),
        Count(
            type_tag='sock',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='table_1')
        ),
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/collect_misplaced_items.py --map_id 65632709 --domain_name collect_misplaced_items --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/collect_misplaced_items/p65632709-collect_misplaced_items_plans.json', want_render=False)
    
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
        
        
        
# ! Observation: if not having action cost, when we split task into subgoals, it is possible that the completion of current subgoal undoes the previous subgoal.
# ! unless we make undo actions more expensive than other actions. (some heuristic is needed)
# ! but this heuristic is not possible for behavior cloning for OpenVLA model training -> thus this is a insightful observation. 