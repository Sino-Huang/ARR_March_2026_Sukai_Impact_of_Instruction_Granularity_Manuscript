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
4
### Feature set (mirroring the plywood features)
- $H_m$: “holding a marker”
- $H_f$: “holding a folder”
- $H_d$: “holding a document”
- $p_m$: distance to the nearest marker
- $p_f$: distance to the nearest folder
- $p_d$: distance to the nearest document
- $p_t$: distance to a table
- $p_c$: distance to a valid cabinet target (a cabinet/location/dimension where placement
- $N_m$: number of markers not on a table
- $N_f$: number of folders not in a cabinet
- $N_d$: number of documents not in a cabinet
- $N_o$: number of closed cabinets

### Sketch with W = 0 (approximate, including distance features)
**Opening Cabinet**
$\{N_o > 0, p_c > 0\} \mapsto \{p_c \downarrow\}$  ; move toward a closed cabinet
$\{N_o > 0, p_c = 0\} \mapsto \{N_o \downarrow\}$  ; open the cabinet when reachable

**Marker task**
$\{N_m > 0, \neg H_m, p_m > 0\} \mapsto \{p_m \downarrow, p_t ?\}$  ; move toward the nearest marker
$\{N_m > 0, \neg H_m, p_m = 0\} \mapsto \{H_m, p_t ?\}$  ; pick up the marker when reachable
$\{N_m > 0, H_m, p_t > 0\} \mapsto \{p_t \downarrow\}$  ; move toward a table
$\{N_m > 0, H_m, p_t = 0\} \mapsto \{N_m \downarrow, \neg H_m, p_m ?\}$  ; put the marker on the table when reachable

**Folder task**
$\{N_f > 0, N_o = 0, \neg H_f, p_f > 0\} \mapsto \{p_f \downarrow, p_c ?\}$  ; move toward the nearest folder
$\{N_f > 0, N_o = 0, \neg H_f, p_f = 0\} \mapsto \{H_f, p_c ?\}$  ; pick up the folder when reachable
$\{N_f > 0, N_o = 0, H_f, p_c > 0\} \mapsto \{p_c \downarrow\}$  ; move toward a valid cabinet spot
$\{N_f > 0, N_o = 0, H_f, p_c = 0\} \mapsto \{N_f \downarrow, \neg H_f, p_f ?\}$  ; put the folder into the cabinet when reachable

**Document task**
$\{N_d > 0, N_o = 0, \neg H_d, p_d > 0\} \mapsto \{p_d \downarrow, p_c ?\}$  ; move toward the nearest document
$\{N_d > 0, N_o = 0, \neg H_d, p_d = 0\} \mapsto \{H_d, p_c ?\}$  ; pick up the document when reachable
$\{N_d > 0, N_o = 0, H_d, p_c > 0\} \mapsto \{p_c \downarrow\}$  ; move toward a valid cabinet spot
$\{N_d > 0, N_o = 0, H_d, p_c = 0\} \mapsto \{N_d \downarrow, \neg H_d, p_d ?\}$  ; put the document into the cabinet when reachable
"""

def put_inside_helper_func_width_0(sketch_dict, type_tag, container_type_name, subgoal_name):
    # 1 move towards the nearest object of type_tag
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    N_all_open_0 = Count(
        type_tag=container_type_name,
        wanted_value='=0',
        count_func=count_closed,
    )
    
    N_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    p_s_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag=type_tag,
    )
    # * effects
    p_s_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag=type_tag,
    )
    rule_move_to_obj = Rule(
        preconditions=[neg_H, N_larger_0, p_s_larger_0, N_all_open_0],
        effects=[p_s_down],
        goal_clause_pattern=f"""
(inreachofrobot agent-01 {{{type_tag}}})
"""
    )
    sketch_dict[subgoal_name].append(rule_move_to_obj)
    # 2 pick up the object when reachable
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    N_all_open_0 = Count(
        type_tag=container_type_name,
        wanted_value='=0',
        count_func=count_closed,
    )
    N_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    p_s_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag=type_tag,
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    rule_pick_up_obj = Rule(
        preconditions=[neg_H, N_larger_0, p_s_equal_0],
        effects=[H],
        goal_clause_pattern=f"""
(inhandofrobot agent-01 {{{type_tag}}})
"""
    )
    sketch_dict[subgoal_name].append(rule_pick_up_obj)
    # 3 move toward a valid container spot
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    N_all_open_0 = Count(
        type_tag=container_type_name,
        wanted_value='=0',
        count_func=count_closed,
    )
    N_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    p_c_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag=container_type_name,
    )
    # * effects
    p_c_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag=container_type_name,
    )
    rule_move_to_container = Rule(
        preconditions=[H, N_larger_0,N_all_open_0, p_c_larger_0],
        effects=[p_c_down],
        goal_clause_pattern=f"""
(inreachofrobot agent-01 {{{container_type_name}}})
"""
    )
    sketch_dict[subgoal_name].append(rule_move_to_container)
    # 4 put the object into the container when reachable
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    N_all_open_0 = Count(
        type_tag=container_type_name,
        wanted_value='=0',
        count_func=count_closed,
    )
    N_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    p_c_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag=container_type_name,
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    N_decrease = Count(
        type_tag=type_tag,
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    rule_put_obj_in_container = Rule(
        preconditions=[H, N_larger_0,N_all_open_0, p_c_equal_0],
        effects=[neg_H, N_decrease],
        goal_clause_pattern=f"""
(exists (?loc - location ?dim - dimension)
    (inside {{{type_tag}}} {{{container_type_name}}} ?loc ?dim)
)
"""
    )
    sketch_dict[subgoal_name].append(rule_put_obj_in_container)


def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "opening cabinet": [],
        "organize markers": [],
        "organize folders": [],
        "organize documents": []
    }
    
    # 1. Opening cabinet
    # 1.1 move toward a closed cabinet
    # * precons
    N_o_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    p_c_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='cabinet',
    )
    # * effects
    p_c_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='cabinet',
    )
    rule_move_to_cabinet = Rule(
        preconditions=[N_o_larger_0, p_c_larger_0],
        effects=[p_c_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {cabinet})
"""
    )
    sketch_dict["opening cabinet"].append(rule_move_to_cabinet)
    
    # 1.2 open the cabinet when reachable
    # * precons
    N_o_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    p_c_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='cabinet',
    )
    # * effects
    N_o_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet = Rule(
        preconditions=[N_o_larger_0, p_c_equal_0],
        effects=[N_o_decrease],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["opening cabinet"].append(rule_open_cabinet)
    
    # 2. Organize markers
    # 2.1 move toward the nearest marker
    # * precons
    neg_H_m = Holding(
        wanted_value=False,
        type_tag='marker',
    )
    N_m_larger_0 = Count(
        type_tag='marker',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    p_m_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='marker',
    )
    # * effects
    p_m_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='marker',
    )
    rule_move_to_marker = Rule(
        preconditions=[neg_H_m, N_m_larger_0, p_m_larger_0],
        effects=[p_m_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {marker})
"""
    )
    sketch_dict["organize markers"].append(rule_move_to_marker)
    # 2.2 pick up the marker when reachable
    # * precons
    neg_H_m = Holding(
        wanted_value=False,
        type_tag='marker',
    )
    N_m_larger_0 = Count(
        type_tag='marker',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    p_m_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='marker',
    )
    # * effects
    H_m = Holding(
        wanted_value=True,
        type_tag='marker',
    )
    rule_pick_up_marker = Rule(
        preconditions=[neg_H_m, N_m_larger_0, p_m_equal_0],
        effects=[H_m],
        goal_clause_pattern="""
(inhandofrobot agent-01 {marker})
"""
    )
    sketch_dict["organize markers"].append(rule_pick_up_marker)
    # 2.3 move toward a table
    # * precons
    H_m = Holding(
        wanted_value=True,
        type_tag='marker',
    )
    N_m_larger_0 = Count(
        type_tag='marker',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    p_t_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='table',
    )
    # * effects
    p_t_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='table',
    )
    rule_move_to_table = Rule(
        preconditions=[H_m, N_m_larger_0, p_t_larger_0],
        effects=[p_t_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {table})
"""
    )
    sketch_dict["organize markers"].append(rule_move_to_table)
    # 2.4 put the marker on the table when reachable
    # * precons
    H_m = Holding(
        wanted_value=True,
        type_tag='marker',
    )
    N_m_larger_0 = Count(
        type_tag='marker',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    p_t_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='table',
    )
    # * effects
    neg_H_m = Holding(
        wanted_value=False,
        type_tag='marker',
    )
    N_m_decrease = Count(
        type_tag='marker',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    rule_put_marker_on_table = Rule(
        preconditions=[H_m, N_m_larger_0, p_t_equal_0],
        effects=[neg_H_m, N_m_decrease],
        goal_clause_pattern="""
(onTop {marker} {table})
"""
    )
    sketch_dict["organize markers"].append(rule_put_marker_on_table)
    # 3. Organize folders
    # we use helper func 
    put_inside_helper_func_width_0(sketch_dict, type_tag='folder', container_type_name='cabinet', subgoal_name="organize folders")
    # 4. Organize documents
    put_inside_helper_func_width_0(sketch_dict, type_tag='document', container_type_name='cabinet', subgoal_name="organize documents")
    
    goal_count_feature = [
        Count(
            type_tag='marker',
            wanted_value='=0',
            count_func=partial(count_not_ontop_sometype, surface_type='table'),
        ),
        Count(
            type_tag='folder',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
        ),
        Count(
            type_tag='document',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
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
# $\{N_o > 0\} \mapsto \{N_o \downarrow\}$  ; open cabinets (width 1)
# **Marker task**
# $\{N_m > 0, \neg H_m\} \mapsto \{H_m\}$  ; pick up a marker (width 1)
# $\{N_m > 0, H_m\} \mapsto \{N_m \downarrow, \neg H_m\}$  ; put the carried marker on a table (width 1)
# **Folder task**
# $\{N_f > 0, N_o = 0, \neg H_f\} \mapsto \{H_f\}$  ; pick up a folder (width 1)
# $\{N_f > 0, N_o = 0, H_f\} \mapsto \{N_f \downarrow, \neg H_f\}$  ; put the carried folder into some valid cabinet spot (width 1)
# **Document task**
# $\{N_d > 0, N_o = 0, \neg H_d\} \mapsto \{H_d\}$  ; pick up a document (width 1)
# $\{N_d > 0, N_o = 0, H_d\} \mapsto \{N_d \downarrow, \neg H_d\}$  ; put the carried document into some valid cabinet spot (width 1)

def put_inside_helper_func_width_1(sketch_dict, type_tag, container_type_name, subgoal_name):
    # 1 move towards the nearest object of type_tag
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    N_all_open_0 = Count(
        type_tag=container_type_name,
        wanted_value='=0',
        count_func=count_closed,
    )
    
    N_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    rule_pick_up_obj = Rule(
        preconditions=[neg_H, N_larger_0, N_all_open_0],
        effects=[H],
        goal_clause_pattern=f"""
(inhandofrobot agent-01 {{{type_tag}}})
"""
    )
    sketch_dict[subgoal_name].append(rule_pick_up_obj)
    # 2 put the object into the container 
    H = Holding(
        wanted_value=True,
        type_tag=type_tag,
    )
    N_all_open_0 = Count(
        type_tag=container_type_name,
        wanted_value='=0',
        count_func=count_closed,
    )
    N_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag=type_tag,
    )
    N_decrease = Count(
        type_tag=type_tag,
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    rule_put_obj_in_container = Rule(
        preconditions=[H, N_larger_0, N_all_open_0],
        effects=[neg_H, N_decrease],
        goal_clause_pattern=f"""
(exists (?loc - location ?dim - dimension)
    (inside {{{type_tag}}} {{{container_type_name}}} ?loc ?dim)
)
"""
    )
    sketch_dict[subgoal_name].append(rule_put_obj_in_container)
    
    

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "opening cabinet": [],
        "organize markers": [],
        "organize folders": [],
        "organize documents": []
    }
    # 1. Opening cabinet
    N_o_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    N_o_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet = Rule(
        preconditions=[N_o_larger_0],
        effects=[N_o_decrease],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["opening cabinet"].append(rule_open_cabinet)
    
    # 2. Organize markers
    # 2.1 pick up a marker
    # * precons
    neg_H_m = Holding(
        wanted_value=False,
        type_tag='marker',
    )
    N_m_larger_0 = Count(
        type_tag='marker',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    # * effects
    H_m = Holding(
        wanted_value=True,
        type_tag='marker',
    )
    rule_pick_up_marker = Rule(
        preconditions=[neg_H_m, N_m_larger_0],
        effects=[H_m],
        goal_clause_pattern="""
(inhandofrobot agent-01 {marker})
"""
    )
    sketch_dict["organize markers"].append(rule_pick_up_marker)
    # 2.2 put the carried marker on a table
    # * precons
    H_m = Holding(
        wanted_value=True,
        type_tag='marker',
    )
    N_m_larger_0 = Count(
        type_tag='marker',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    # * effects
    neg_H_m = Holding(
        wanted_value=False,
        type_tag='marker',
    )
    N_m_decrease = Count(
        type_tag='marker',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    rule_put_marker_on_table = Rule(
        preconditions=[H_m, N_m_larger_0],
        effects=[neg_H_m, N_m_decrease],
        goal_clause_pattern="""
(onTop {marker} {table})
"""
    )
    sketch_dict["organize markers"].append(rule_put_marker_on_table)
    # 3. Organize folders
    # we use helper func 
    put_inside_helper_func_width_1(sketch_dict, type_tag='folder', container_type_name='cabinet', subgoal_name="organize folders")
    # 4. Organize documents
    put_inside_helper_func_width_1(sketch_dict, type_tag='document', container_type_name='cabinet', subgoal_name="organize documents")
    goal_count_feature = [
        Count(
            type_tag='marker',
            wanted_value='=0',
            count_func=partial(count_not_ontop_sometype, surface_type='table'),
        ),
        Count(
            type_tag='folder',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
        ),
        Count(
            type_tag='document',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
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
    
### Sketch with W = 2 ; factor opening cabinets out
# $\{N_o > 0\} \mapsto \{N_o \downarrow\}$  ; open cabinets (width 1)
# $\{N_m > 0\} \mapsto \{N_m \downarrow\}$  ; put markers on a table (width 2)
# $\{N_f > 0, N_o = 0\} \mapsto \{N_f \downarrow\}$  ; put folders into a cabinet (width 2)
# $\{N_d > 0, N_o = 0\} \mapsto \{N_d \downarrow\}$  ; put documents into a cabinet (width 2)

def put_inside_helper_func_width_2(sketch_dict, type_tag, container_type_name, subgoal_name):
    N_all_open_0 = Count(
        type_tag=container_type_name,
        wanted_value='=0',
        count_func=count_closed,
    )
    
    N_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    # * effects
    N_decrease = Count(
        type_tag=type_tag,
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    rule_put_obj_in_container = Rule(
        preconditions=[N_larger_0, N_all_open_0],
        effects=[N_decrease],
        goal_clause_pattern=f"""
(exists (?loc - location ?dim - dimension)
    (inside {{{type_tag}}} {{{container_type_name}}} ?loc ?dim)
)
"""
    )
    sketch_dict[subgoal_name].append(rule_put_obj_in_container)
    

def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "opening cabinet": [],
        "organize markers": [],
        "organize folders": [],
        "organize documents": []
    }
    # 1. Opening cabinet
    N_o_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    N_o_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet = Rule(
        preconditions=[N_o_larger_0],
        effects=[N_o_decrease],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["opening cabinet"].append(rule_open_cabinet)
    
    # 2. Organize markers
    N_m_larger_0 = Count(
        type_tag='marker',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    # * effects
    N_m_decrease = Count(
        type_tag='marker',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    rule_put_marker_on_table = Rule(
        preconditions=[N_m_larger_0],
        effects=[N_m_decrease],
        goal_clause_pattern="""
(onTop {marker} {table})
"""
    )
    sketch_dict["organize markers"].append(rule_put_marker_on_table)
    # 3. Organize folders
    # we use helper func 
    put_inside_helper_func_width_2(sketch_dict, type_tag='folder', container_type_name='cabinet', subgoal_name="organize folders")
    # 4. Organize documents
    put_inside_helper_func_width_2(sketch_dict, type_tag='document', container_type_name='cabinet', subgoal_name="organize documents")
    goal_count_feature = [
        Count(
            type_tag='marker',
            wanted_value='=0',
            count_func=partial(count_not_ontop_sometype, surface_type='table'),
        ),
        Count(
            type_tag='folder',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
        ),
        Count(
            type_tag='document',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
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

### Sketch with W = 3 ; state each sub-goal (no mentioning of opening cabinets)
# $\{N_m > 0\} \mapsto \{N_m \downarrow\}$  ; put markers on a table (width 2)
# $\{N_f > 0\} \mapsto \{N_f \downarrow\}$  ; put folders into a cabinet (width 3)
# $\{N_d > 0\} \mapsto \{N_d \downarrow\}$  ; put documents into a cabinet (width 3)

def put_inside_helper_func_width_3(sketch_dict, type_tag, container_type_name, subgoal_name):
    N_larger_0 = Count(
        type_tag=type_tag,
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    # * effects
    N_decrease = Count(
        type_tag=type_tag,
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name=container_type_name),
    )
    rule_put_obj_in_container = Rule(
        preconditions=[N_larger_0],
        effects=[N_decrease],
        goal_clause_pattern=f"""
(exists (?loc - location ?dim - dimension)
    (inside {{{type_tag}}} {{{container_type_name}}} ?loc ?dim)
)
"""
    )
    sketch_dict[subgoal_name].append(rule_put_obj_in_container)
    
def create_width_3_sketch(env, map_id, domain_name, window=None):
    width = 3
    sketch_dict = {
        "organize markers": [],
        "organize folders": [],
        "organize documents": []
    }
    
    # 2. Organize markers
    N_m_larger_0 = Count(
        type_tag='marker',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    # * effects
    N_m_decrease = Count(
        type_tag='marker',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='table'),
    )
    rule_put_marker_on_table = Rule(
        preconditions=[N_m_larger_0],
        effects=[N_m_decrease],
        goal_clause_pattern="""
(onTop {marker} {table})
"""
    )
    sketch_dict["organize markers"].append(rule_put_marker_on_table)
    # 3. Organize folders
    # we use helper func 
    put_inside_helper_func_width_3(sketch_dict, type_tag='folder', container_type_name='cabinet', subgoal_name="organize folders")
    # 4. Organize documents
    put_inside_helper_func_width_3(sketch_dict, type_tag='document', container_type_name='cabinet', subgoal_name="organize documents")
    goal_count_feature = [
        Count(
            type_tag='marker',
            wanted_value='=0',
            count_func=partial(count_not_ontop_sometype, surface_type='table'),
        ),
        Count(
            type_tag='folder',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
        ),
        Count(
            type_tag='document',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/organizing_file_cabinet.py --map_id 3135782 --domain_name organizing_file_cabinet --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/organizing_file_cabinet/p3135782-organizing_file_cabinet_plans.json', want_render=False)
    
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