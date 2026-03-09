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
2
### Features 
- $H_n$: holding a not-sliceable food
- $H_s$: holding a sliceable food
- $H_k$: holding a slicer
- $U$: number of sliceable food that is not sliced yet
- $d_p$: distance to the selected incomplete plate p
- $d_n, d_s$: distances to nearest not-sliceable / sliceable
- $k$: distance to a slicer (knife)
- $B$: selected plate p complete bottom salad; 
- $n$: #incomplete plates


### Sketch width W=0 

**Acquire slicer**
$\{U>0 , \neg H_k, k>0\} \mapsto \{k \downarrow\}$  ; move to slicer
$\{U>0 , \neg H_k, k=0\} \mapsto \{H_k\}$ ; hold a slicer
$\{U>0 , H_k, d_s > 0\} \mapsto \{d_s \downarrow\}$ ; move to sliceable food 
$\{U>0 , H_k, d_s = 0\} \mapsto \{U \downarrow, \neg H_k \}$ ; cut sliceable food
$\{U=0 , H_k\} \mapsto \{\neg H_k, \}$ ; put down slicer

**Place bottom salad**

$\{\, n>0, U=0, nn>0, \neg B,\; \neg H_n,\; d_n>0 \,\}\ \mapsto\ \{\ d_n \downarrow \}$  (width =0)
$\{\, n>0, U=0,nn>0, \neg B,\; \neg H_n,\; d_n=0 \,\}\ \mapsto\ \{\, H_n \,\}$  (width =0)
$\{\, n>0, U=0,nn>0, \neg B,\; H_n,\; d_p>0 \,\}\ \mapsto\ \{\, d_p\downarrow \,\}$  (width =0)
$\{\, n>0, U=0, nn>0, \neg B,\; H_n,\; d_p=0 \,\}\ \mapsto\ \{\, B, nn \downarrow \; \neg H_n \,\}$  (width =0)

$nn$: number of plates that do not have bottom salad yet

**Handle sliceable salad**
$\{nn=0, n>0, U=0 , \neg H_s, d_s>0\} \mapsto \{d_s \downarrow\}$  ; move to sliceable food 
$\{nn=0, n>0, U=0 , \neg H_s, d_s=0\} \mapsto \{H_s\}$  ; pick up sliceable food 
$\{nn=0, n>0, U=0 , H_s, d_p>0\} \mapsto \{d_p \downarrow\}$  ; move to the plate
$\{nn=0, n>0, U=0 , H_s, d_p=0\} \mapsto \{n \downarrow, \neg H_s, \}$  ; put down sliced food
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "slice food": [],
        "place salad bottom": [],
        "place salad top": []
    }
    # 1. Acquire slicer
    # 1.1 move to slicer
    # * precons
    U_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    neg_H_k = Holding(
        wanted_value=False,
        type_tag='carving_knife',
    )
    k_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='carving_knife',
    )
    # * effects
    k_down = DistanceToNearest(
        wanted_value='decrease',
        type_tag='carving_knife',
    )
    rule_move_to_knife = Rule(
        preconditions=[U_larger_0, neg_H_k, k_larger_0],
        effects=[k_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {carving_knife})
"""
    )
    sketch_dict["slice food"].append(rule_move_to_knife)
    
    # 1.2 hold a slicer
    U_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    neg_H_k = Holding(
        wanted_value=False,
        type_tag='carving_knife',
    )
    k_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='carving_knife',
    )
    # * effects
    H_k = Holding(
        wanted_value=True,
        type_tag='carving_knife',
    )
    rule_pick_up_knife = Rule(
        preconditions=[U_larger_0, neg_H_k, k_equal_0],
        effects=[H_k],
        goal_clause_pattern="""
(inhandofrobot agent-01 {carving_knife})
"""
    )
    sketch_dict["slice food"].append(rule_pick_up_knife)   
    # 1.3 move to sliceable food
    U_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    H_k = Holding(
        wanted_value=True,
        type_tag='carving_knife',
    )
    d_s_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='apple tomato',
    )
    # * effects
    d_s_down = DistanceToNearest(
        wanted_value='decrease',
        type_tag='apple tomato',
    )
    rule_move_to_sliceable = Rule(
        preconditions=[U_larger_0, H_k, d_s_larger_0],
        effects=[d_s_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {apple tomato})
"""
    )
    sketch_dict["slice food"].append(rule_move_to_sliceable)
    # 1.4 cut sliceable food
    U_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    H_k = Holding(
        wanted_value=True,
        type_tag='carving_knife',
    )
    d_s_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='apple tomato',
    )
    # * effects
    U_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=count_not_sliced,
    )
    rule_slice_food = Rule(
        preconditions=[U_larger_0, H_k, d_s_equal_0],
        effects=[U_decrease],
        goal_clause_pattern="""
(is-sliced {apple tomato})
"""
    )
    sketch_dict["slice food"].append(rule_slice_food)
    
    # 2. Place salad bottom
    # 2.1 move to bottom salad 
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    
    n_n_larger_0 = Count(
        type_tag='lettuce radish',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    
    n_first_layer_plate = Count(
        wanted_value=">0",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    neg_H_n = Holding(
        wanted_value=False,
        type_tag='lettuce radish',
    )
    
    d_n_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='lettuce radish',
    )
    # * effects
    d_n_down = DistanceToNearest(
        wanted_value='decrease',
        type_tag='lettuce radish',
    )
    rule_move_to_bottom_salad = Rule(
        preconditions=[n_larger_0, U_equal_0, n_n_larger_0, n_first_layer_plate, neg_H_n, d_n_larger_0],
        effects=[d_n_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {lettuce radish})
"""
    )
    sketch_dict["place salad bottom"].append(rule_move_to_bottom_salad)
    # 2.2 hold a bottom salad
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    
    n_n_larger_0 = Count(
        type_tag='lettuce radish',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    
    n_first_layer_plate = Count(
        wanted_value=">0",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    neg_H_n = Holding(
        wanted_value=False,
        type_tag='lettuce radish',
    )
    
    d_n_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='lettuce radish',
    )
    
    # * effects
    H_n = Holding(
        wanted_value=True,
        type_tag='lettuce radish',
    )
    rule_pick_up_bottom_salad = Rule(
        preconditions=[n_larger_0, U_equal_0, n_n_larger_0, n_first_layer_plate, neg_H_n, d_n_equal_0],
        effects=[H_n],
        goal_clause_pattern="""
(inhandofrobot agent-01 {lettuce radish})
"""
    )
    sketch_dict["place salad bottom"].append(rule_pick_up_bottom_salad)
    # 2.3 move to the plate
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    
    n_n_larger_0 = Count(
        type_tag='lettuce radish',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )

    n_first_layer_plate = Count(
        wanted_value=">0",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    H_n = Holding(
        wanted_value=True,
        type_tag='lettuce radish',
    )
    
    d_p_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='plate',
    )
    # * effects
    d_p_down = DistanceToNearest(
        wanted_value='decrease',
        type_tag='plate',
    )
    rule_move_to_plate = Rule(
        preconditions=[n_larger_0, U_equal_0, n_n_larger_0, n_first_layer_plate, H_n, d_p_larger_0],
        effects=[d_p_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {plate})
"""
    )
    sketch_dict["place salad bottom"].append(rule_move_to_plate)
    # 2.4 put down bottom salad on the plate
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    
    n_n_larger_0 = Count(
        type_tag='lettuce radish',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )

    n_first_layer_plate = Count(
        wanted_value=">0",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    H_n = Holding(
        wanted_value=True,
        type_tag='lettuce radish',
    )
    
    d_p_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='plate',
    )
    # * effects
    n_first_layer_plate_decrease = Count(
        wanted_value="decrease",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    neg_H_n = Holding(
        wanted_value=False,
        type_tag='lettuce radish',
    )
    n_n_decrease = Count(
        type_tag='lettuce radish',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    rule_put_down_bottom_salad = Rule(
        preconditions=[n_larger_0, U_equal_0, n_n_larger_0, n_first_layer_plate, H_n, d_p_equal_0],
        effects=[n_first_layer_plate_decrease, neg_H_n, n_n_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {lettuce radish}))
(exists (?loc - location)
   (and 
        (at {plate} ?loc bottom)
        (at {lettuce radish} ?loc middle)
   )
)
"""
    )
    sketch_dict["place salad bottom"].append(rule_put_down_bottom_salad)
    
    # 3. Handle sliceable salad
    # 3.1 move to sliceable food 
    n_n_equal_0 = Count(
        type_tag='lettuce radish',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )

    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='apple tomato',
    )
    d_s_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='apple tomato',
    )
    # * effects
    d_s_down = DistanceToNearest(
        wanted_value='decrease',
        type_tag='apple tomato',
    )
    rule_move_to_sliceable = Rule(
        preconditions=[n_n_equal_0, n_n_top_larger_0, n_larger_0, U_equal_0, neg_H_s, d_s_larger_0],
        effects=[d_s_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {apple tomato})
"""
    )
    sketch_dict["place salad top"].append(rule_move_to_sliceable)
    # 3.2 pick up sliceable food 
    n_n_equal_0 = Count(
        type_tag='lettuce radish',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='apple tomato',
    )
    d_s_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='apple tomato',
    )
    # * effects
    H_s = Holding(
        wanted_value=True,
        type_tag='apple tomato',
    )
    rule_pick_up_sliceable = Rule(
        preconditions=[n_n_equal_0, n_n_top_larger_0, n_larger_0, U_equal_0, neg_H_s, d_s_equal_0],
        effects=[H_s],
        goal_clause_pattern="""
(inhandofrobot agent-01 {apple tomato})
"""
    )
    sketch_dict["place salad top"].append(rule_pick_up_sliceable)
    # 3.3 move to the plate
    n_n_equal_0 = Count(
        type_tag='lettuce radish',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    H_s = Holding(
        wanted_value=True,
        type_tag='apple tomato',
    )
    d_p_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='plate',
    )
    # * effects
    d_p_down = DistanceToNearest(
        wanted_value='decrease',
        type_tag='plate',
    )
    rule_move_to_plate = Rule(
        preconditions=[n_n_equal_0, n_larger_0,n_n_top_larger_0, U_equal_0, H_s, d_p_larger_0],
        effects=[d_p_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {plate})
"""
    )
    sketch_dict["place salad top"].append(rule_move_to_plate)
    # 3.4 put down sliced food on the plate
    n_n_equal_0 = Count(
        type_tag='lettuce radish',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    H_s = Holding(
        wanted_value=True,
        type_tag='apple tomato',
    )
    d_p_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='plate',
    )
    # * effects
    n_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=count_incomplete_salad_place,
    )
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='apple tomato',
    )
    n_n_top_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    rule_put_down_sliceable = Rule(
        preconditions=[n_n_equal_0,  n_n_top_larger_0, n_larger_0, U_equal_0, H_s, d_p_equal_0],
        effects=[n_decrease, neg_H_s, n_n_top_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {apple tomato}))
(exists (?loc - location)
    (and
        (at {plate} ?loc bottom)
        (at {apple tomato} ?loc top)
    )
)
"""
    )
    sketch_dict["place salad top"].append(rule_put_down_sliceable)
    
    
    goal_count_feature =[
        Count(
            type_tag='apple tomato',
            wanted_value='=0',
            count_func=count_not_sliced,
        ),
        Count(
            type_tag='plate',
            wanted_value='=0',
            count_func=count_incomplete_salad_place,
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

### Sketch width W=1 

# **Acquire a not-sliceable**
# $\{\, \neg B,\; \neg H_n\}\ \mapsto\ \{\, H_n \,\}$ width =1 (reachable and then carry)
# **Place to achieve the bottom**
# $\{\, \neg B,\; H_n \}\ \mapsto\ \{\, B,\; \neg H_n \,\}$ width = 1 (moving and then drop)
# **Slice food**
# $\{U>0 , \neg H_n, \neg H_s, \neg H_k\} \mapsto \{H_k\}$ ; hold a slicer  (move and hold)
# $\{U>0 , H_k,\} \mapsto \{U \downarrow\}$ ; sliceable food width = 1 (move and slice)
# $\{U=0 , H_k\} \mapsto \{\neg H_k\}$ ; put down slicer
# **Handle sliceable food**
# $\{B, U=0 , \neg H_s\} \mapsto \{H_s\}$  ; pick up sliceable food 
# $\{n>0, B, U=0 , H_s\} \mapsto \{\neg B, n \downarrow, \neg H_s\}$  ; put down sliced food


def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "slice food": [],
        "place salad bottom": [],
        "place salad top": []
    }
    
    # 1. Acquire slicer
    # 1.1 pick up slicer
    # * precons
    U_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    neg_H_k = Holding(
        wanted_value=False,
        type_tag='carving_knife',
    )
    # * effects
    H_k = Holding(
        wanted_value=True,
        type_tag='carving_knife',
        
    )
    rule_pick_up_knife = Rule(
        preconditions=[U_larger_0, neg_H_k],
        effects=[H_k],
        goal_clause_pattern="""
(inhandofrobot agent-01 {carving_knife})
"""
    )
    sketch_dict["slice food"].append(rule_pick_up_knife)   
    # 1.2 slice food
    U_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    H_k = Holding(
        wanted_value=True,
        type_tag='carving_knife',
    )
    # * effects
    U_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=count_not_sliced,
    )
    rule_slice_food = Rule(
        preconditions=[U_larger_0, H_k],
        effects=[U_decrease],
        goal_clause_pattern="""
(is-sliced {apple tomato})
"""
    )
    sketch_dict["slice food"].append(rule_slice_food)
    # 2. Place salad bottom
    # 2.1 pick up bottom salad
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    
    n_n_larger_0 = Count(
        type_tag='lettuce radish',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    
    n_first_layer_plate = Count(
        wanted_value=">0",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    neg_H_n = Holding(
        wanted_value=False,
        type_tag='lettuce radish',
    )
    # * effects
    H_n = Holding(
        wanted_value=True,
        type_tag='lettuce radish',
    )
    rule_pick_up_bottom_salad = Rule(
        preconditions=[n_larger_0, U_equal_0, n_n_larger_0, n_first_layer_plate, neg_H_n],
        effects=[H_n],
        goal_clause_pattern="""
(inhandofrobot agent-01 {lettuce radish})
"""
    )
    sketch_dict["place salad bottom"].append(rule_pick_up_bottom_salad)
    # 2.2 put down bottom salad on the plate
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    
    n_n_larger_0 = Count(
        type_tag='lettuce radish',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )

    n_first_layer_plate = Count(
        wanted_value=">0",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    H_n = Holding(
        wanted_value=True,
        type_tag='lettuce radish',
    )
    # * effects
    n_first_layer_plate_decrease = Count(
        wanted_value="decrease",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    neg_H_n = Holding(
        wanted_value=False,
        type_tag='lettuce radish',
    )
    n_n_decrease = Count(
        type_tag='lettuce radish',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    rule_put_down_bottom_salad = Rule(
        preconditions=[n_larger_0, U_equal_0, n_n_larger_0, n_first_layer_plate, H_n],
        effects=[n_first_layer_plate_decrease, neg_H_n, n_n_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {lettuce radish}))
(exists (?loc - location)
   (and 
        (at {plate} ?loc bottom)
        (at {lettuce radish} ?loc middle)
   )
)
"""
    )
    sketch_dict["place salad bottom"].append(rule_put_down_bottom_salad)
    
    # 3. Handle sliceable salad
    # 3.1 pick up sliceable food
    n_n_equal_0 = Count(
        type_tag='lettuce radish',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )

    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='apple tomato',
    )
    # * effects
    H_s = Holding(
        wanted_value=True,
        type_tag='apple tomato',
    )
    rule_pick_up_sliceable = Rule(
        preconditions=[n_n_equal_0, n_n_top_larger_0, n_larger_0, U_equal_0, neg_H_s],
        effects=[H_s],
        goal_clause_pattern="""
(inhandofrobot agent-01 {apple tomato})
"""
    )
    sketch_dict["place salad top"].append(rule_pick_up_sliceable)
    # 3.2 put down sliceable food on the plate
    n_n_equal_0 = Count(
        type_tag='lettuce radish',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    H_s = Holding(
        wanted_value=True,
        type_tag='apple tomato',
    )
    # * effects
    n_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=count_incomplete_salad_place,
    )
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='apple tomato',
    )
    n_n_top_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    rule_put_down_sliceable = Rule(
        preconditions=[n_n_equal_0,  n_n_top_larger_0, n_larger_0, U_equal_0, H_s],
        effects=[n_decrease, neg_H_s, n_n_top_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {apple tomato}))
(exists (?loc - location)
    (and
        (at {plate} ?loc bottom)
        (at {apple tomato} ?loc top)
    )
)
"""
    )
    sketch_dict["place salad top"].append(rule_put_down_sliceable)
    goal_count_feature =[
        Count(
            type_tag='apple tomato',
            wanted_value='=0',
            count_func=count_not_sliced,
        ),
        Count(
            type_tag='plate',
            wanted_value='=0',
            count_func=count_incomplete_salad_place,
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

### Sketch width W=2
# **Acquire a not-sliceable**
# $\{\, \neg B,\; \neg H_n\}\ \mapsto\ \{\, H_n \,\}$ width =1 (reachable and then carry)
# **Place to achieve the bottom**
# $\{\, \neg B,\; H_n \}\ \mapsto\ \{\, B,\; \neg H_n \,\}$ width = 1 (moving and then drop)
# **Slice food**
# $\{U>0 \} \mapsto \{U \downarrow\}$  ; slice food if they are sliceable 
# - slice food has width 2: [(carrying slicer), (reachable agent food)]
# **Complete the whole salad** 
# $\{n >0, U=0, B \} \mapsto \{n \downarrow, \neg B\}$  ; complete salad plate after the selected plate p complete bottom salad
# - width = 2 ["(carrying sliceable_food)", "(reachable agent plate)"]
    
def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "slice food": [],
        "place salad bottom": [],
        "place salad top": []
    }
    # 1. slice food
    U_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    # * effects
    U_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=count_not_sliced,
    )
    rule_slice_food = Rule(
        preconditions=[U_larger_0],
        effects=[U_decrease],
        goal_clause_pattern="""
(is-sliced {apple tomato})
"""
    )
    sketch_dict["slice food"].append(rule_slice_food)
    # 2. Place salad bottom
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    
    n_n_larger_0 = Count(
        type_tag='lettuce radish',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    
    n_first_layer_plate = Count(
        wanted_value=">0",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    
    # * effects
    n_first_layer_plate_decrease = Count(
        wanted_value="decrease",
        type_tag='plate',
        count_func=count_not_complete_first_layer_salad,
    )
    n_n_decrease = Count(
        type_tag='lettuce radish',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    rule_put_down_bottom_salad = Rule(
        preconditions=[n_larger_0, U_equal_0, n_n_larger_0, n_first_layer_plate],
        effects=[n_first_layer_plate_decrease, n_n_decrease],
        goal_clause_pattern="""
(exists (?loc - location)
   (and 
        (at {plate} ?loc bottom)
        (at {lettuce radish} ?loc middle)
   )
)
"""
    )
    sketch_dict["place salad bottom"].append(rule_put_down_bottom_salad)
    # 3. Handle sliceable salad
    n_n_equal_0 = Count(
        type_tag='lettuce radish',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )

    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    # * effects
    n_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    rule_put_down_sliceable = Rule(
        preconditions=[n_n_equal_0,  n_n_top_larger_0, n_larger_0, U_equal_0],
        effects=[n_decrease, n_n_top_decrease],
        goal_clause_pattern="""
(exists (?loc - location)
    (and
        (at {plate} ?loc bottom)
        (at {apple tomato} ?loc top)
    )
)
"""
    )
    sketch_dict["place salad top"].append(rule_put_down_sliceable)
    goal_count_feature =[
        Count(
            type_tag='apple tomato',
            wanted_value='=0',
            count_func=count_not_sliced,
        ),
        Count(
            type_tag='plate',
            wanted_value='=0',
            count_func=count_incomplete_salad_place,
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

### Sketch width W=3 
# $\{U>0 \} \mapsto \{U \downarrow\}$  ; slice food if they are sliceable 
# - slice food has width 2: [(carrying slicer), (reachable agent food)]
# $\{n >0, U=0 \} \mapsto \{n \downarrow\}$  ; complete salad plate
# - has width 3: ["(onTop not-sliceable-food plate)", "(carrying sliceable_food)", "(reachable agent plate)"]

def create_width_3_sketch(env, map_id, domain_name, window=None):
    width = 3
    sketch_dict = {
        "slice food": [],
        "place salad": [],
    }
    # 1. slice food
    U_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    # * effects
    U_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=count_not_sliced,
    )
    rule_slice_food = Rule(
        preconditions=[U_larger_0],
        effects=[U_decrease],
        goal_clause_pattern="""
(is-sliced {apple tomato})
"""
    )
    sketch_dict["slice food"].append(rule_slice_food)
    
    # 2. Place salad
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    U_equal_0 = Count(
        type_tag='apple tomato',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    
    n_n_larger_0 = Count(
        type_tag='lettuce radish',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_n_top_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    # * effects
    n_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_n_decrease = Count(
        type_tag='lettuce radish',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    rule_put_down_salad = Rule(
        preconditions=[n_larger_0, U_equal_0, n_n_larger_0, n_n_top_larger_0],
        effects=[n_decrease, n_n_top_decrease, n_n_decrease],
        goal_clause_pattern="""
(exists (?loc - location)
    (and
        (at {plate} ?loc bottom)
        (at {lettuce radish} ?loc middle)
        (at {apple tomato} ?loc top)
    )
)
"""
    )
    sketch_dict["place salad"].append(rule_put_down_salad)
    goal_count_feature =[
        Count(
            type_tag='apple tomato',
            wanted_value='=0',
            count_func=count_not_sliced,
        ),
        Count(
            type_tag='plate',
            wanted_value='=0',
            count_func=count_incomplete_salad_place,
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

### Sketch width W=4
# $\{n >0\} \mapsto \{n \downarrow\}$  ; complete salad plate

def create_width_4_sketch(env, map_id, domain_name, window=None):
    width = 4
    sketch_dict = {
        "place salad": [],
    }
    
    n_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_incomplete_salad_place,
    )
    
    n_n_larger_0 = Count(
        type_tag='lettuce radish',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_n_top_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    U_larger_0 = Count(
        type_tag='apple tomato',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    # * effects
    n_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=count_incomplete_salad_place,
    )
    n_n_top_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    n_n_decrease = Count(
        type_tag='lettuce radish',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='plate'),
    )
    U_decrease = Count(
        type_tag='apple tomato',
        wanted_value='decrease',
        count_func=count_not_sliced,
    )
    
    rule_put_down_salad = Rule(
        preconditions=[n_larger_0, n_n_larger_0, n_n_top_larger_0, U_larger_0],
        effects=[n_decrease, n_n_top_decrease, n_n_decrease, U_decrease],
        goal_clause_pattern="""
(exists (?loc - location)
    (and
        (at {plate} ?loc bottom)
        (at {lettuce radish} ?loc middle)
        (at {apple tomato} ?loc top)
        (is-sliced {apple tomato})
    )
)
"""
    )
    sketch_dict["place salad"].append(rule_put_down_salad)
    goal_count_feature =[
        Count(
            type_tag='apple tomato',
            wanted_value='=0',
            count_func=count_not_sliced,
        ),
        Count(
            type_tag='plate',
            wanted_value='=0',
            count_func=count_incomplete_salad_place,
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/preparing_salad.py --map_id 237740848 --domain_name preparing_salad --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/preparing_salad/p86578575-preparing_salad_plans.json', want_render=False)
    
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