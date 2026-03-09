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
    count_not_onfloor,
    count_not_toggled,
    count_not_sliced)
"""
3
### Features 

- **Holding flags**
  - $H_b$: holding a blender
  - $H_f$: holding a food (apple or casserole)
  - $H_s$: holding a soap
  - $H_r$: holding a rag (cleaning tool)
  - $H_p$: holding a plate
  - $H_o$: holding a vegetable_oil
- **Distances (shortest-path over `move-dir`)**
  - $d_{ct}$: to a target countertop
  - $d_{fr}$: to a target fridge
  - $d_{si}$: to a target sink
  - $d_{c1}$: to target cabinet $c_1$ (for plate)
  - $d_{c2}$: to target cabinet $c_2$ (for oil)
  - $d_x$: to a target object $x\in\{\text{blender, plate, rag, soap, apple, casserole, oil}\}$
- **State booleans**
  - $\text{Open}(c)$: `is-opened c` (for any cabinet or fridge)
  - $\text{Tog}(s)$: `is-toggled s` (sink tap on)
  - $\text{Soaked}$: the held rag is `is-soaked`
  - $\text{DustFree}(c)$: `is-not-dusted c`
  - $\text{Clean}(p)$: `is-not-stained p`
  - $\text{OilIn}(c_2)$: `inside vegetable_oil c_2 …`
  - $\text{Diff}$: the inequality $c_1 \neq c_2$ for the chosen plate/oil pair
- **Counters (sub-problem counters)**
  - $N_b$: # blenders **not** on a countertop
  - $N_a$: # apples **not** inside a fridge
  - $N_{cas}$: # casseroles **not** inside a fridge
  - $N_s$: # soaps **not** next to a sink
  - $N_{cab}$: # cabinets **not** dusted
  - $N_r$: # rags **not** next to/inside a sink
  - $N_{p\_clean}$: # plates **not** clean (stained)
  - $N_{oil}$: # vegetable_oil items **not** inside a cabinet
  - $N_{p\_store}$: # plates **not** stored in some cabinet $c_1$ **different** from the one holding oil $c_2$
  - $N_{close}$: # number of closed cabinet or fridge furnitures 



#### Sketch width **W=0** (distance decrements only)

- **Open Cabinet and Fridge** -- width 0
  $\{N_{close} > 0, d_c>0\}\mapsto\{d_c\downarrow\}$  ; move toward closed furniture
  $\{\ N_{close} > 0, d_c = 0\}\ \mapsto\ \{\ N_{close}\downarrow \,\}$

- **Oil placement — width 0**
  $\{N_{close} = 0,\neg H_o,\, d_{oil}>0\}\mapsto\{d_{oil}\downarrow\}$  ; approach oil
  $\{N_{close} = 0,\neg H_o,\, d_{oil}=0\}\mapsto\{H_o\}$  ; pick up oil
  $\{N_{close} = 0, H_o, N_{oil} > 0, d_{c2}>0\}\mapsto\{d_{c2}\downarrow\}$  ; go to cabinet $c_2$
  $\{N_{close} = 0, H_o, N_{oil} > 0, d_{c2} = 0\}\ \mapsto\ \{ N_{oil}\downarrow, \neg H_o\}$ ; place oil in $c_2$

- **Plate cleaning — width 0**

  $\{\neg \text{Tog}(s),\, d_{si}>0\}\mapsto\{d_{si}\downarrow\}$ ; toggle sink on 
  $\{\neg \text{Tog}(s),\, d_{si}=0\}\mapsto\{\text{Tog}(s)\}$
  $\{\neg H_r,\, d_{rag}>0\}\mapsto\{d_{rag}\downarrow\}$ ; get rag
  $\{\neg H_r,\, d_{rag}=0\}\mapsto\{H_r\}$
  $\{H_r,\, \neg \text{Soaked},\, d_{si}>0\}\mapsto\{d_{si}\downarrow\}$ ; soak rag
  $\{H_r,\, \neg \text{Soaked},\, d_{si}=0\}\mapsto\{\text{Soaked}\}$
  $\{H_r,\, \text{Soaked},\, N_{p\_clean}>0,\, d_{plate}>0\}\mapsto\{d_{plate}\downarrow\}$
  $\{H_r,\, \text{Soaked},\, N_{p\_clean}>0,\, d_{plate}=0\}\mapsto\{N_{p\_clean}\downarrow, d_{plate}?\}$

- **Plate placement distinct from oil — width 0 (dominant)**

  $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 ,\neg H_p,\, N_{p\_store}>0,\, d_{plate}>0\}\mapsto\{d_{plate}\downarrow\}$
  $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 , \neg H_p,\, N_{p\_store}>0,\, d_{plate}=0\}\mapsto\{H_p\}$
  $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 , H_p,\, N_{p\_store}>0,\, d_{c1}>0\}\mapsto\{d_{c1}\downarrow\}$
  $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 , H_p,\, N_{p\_store}>0,\, d_{c1}=0\}\mapsto\{N_{p\_store}\downarrow,\, \neg H_p\}$

- **Blender → countertop — width 0**

  $\{N_b > 0, \neg H_b,\, d_{blender}>0\}\mapsto\{d_{blender}\downarrow\}$
  $\{N_b > 0, \neg H_b,\, d_{blender}=0\}\mapsto\{H_b\}$
  $\{H_b,\, N_b>0,\, d_{ct}>0\}\mapsto\{d_{ct}\downarrow\}$
  $\{H_b,\, N_b>0,\, d_{ct}=0\}\mapsto\{N_b\downarrow,\, \neg H_b\}$

- **Apple → fridge — width 0**

  $\{N_{close} =0,\neg H_f,\, d_{apple}>0\}\mapsto\{d_{apple}\downarrow\}$
  $\{N_{close} =0,\neg H_f,\, d_{apple}=0\}\mapsto\{H_f\}$
  $\{N_{close} =0,H_f,\, N_a>0,\, d_{fr}>0\}\mapsto\{d_{fr}\downarrow\}$
  $\{N_{close} =0,H_f,\, N_a>0,\, d_{fr}=0\}\mapsto\{N_a\downarrow,\, \neg H_f\}$
- **Casserole → fridge — width 0**

  $\{N_{close} =0 ,\neg H_f,\, d_{casserole}>0\}\mapsto\{d_{casserole}\downarrow\}$
  $\{N_{close} =0 ,\neg H_f,\, d_{casserole}=0\}\mapsto\{H_f\}$
  $\{N_{close} =0 ,H_f,\, N_{cas}>0,\, d_{fr}>0\}\mapsto\{d_{fr}\downarrow\}$
  $\{N_{close} =0 ,H_f,\, N_{cas}>0,\, d_{fr}=0\}\mapsto\{N_{cas}\downarrow,\, \neg H_f\}$
- **Cabinet dusting — width 0**
  $\{\neg H_r,\, d_{rag}>0\}\mapsto\{d_{rag}\downarrow\}$ ; get rag
  $\{\neg H_r,\, d_{rag}=0\}\mapsto\{H_r\}$
  $\{H_r,\, N_{cab}>0,\, d_c>0\}\mapsto\{d_c\downarrow\}$ ; clean cabinet 
  $\{H_r,\, N_{cab}>0,\, d_c=0\}\mapsto\{N_{cab}\downarrow,\, \neg H_r\}$
- **Rag near/inside sink — width 0**

  $\{N_{cab} = 0, N_{p\_clean} = 0,\neg H_r,\, d_{rag}>0\}\mapsto\{d_{rag}\downarrow\}$
  $\{N_{cab} = 0, N_{p\_clean} = 0,\neg H_r,\, d_{rag}=0\}\mapsto\{H_r\}$
  $\{N_{cab} = 0, N_{p\_clean} = 0,H_r,\, N_r>0,\, d_{si}>0\}\mapsto\{d_{si}\downarrow\}$
  $\{N_{cab} = 0, N_{p\_clean} = 0,H_r,\, N_r>0,\, d_{si}=0\}\mapsto\{N_r\downarrow,\, \neg H_r\}$
- **Soap next to sink — width 0**
  $\{N_{cab} = 0, N_{p\_clean} = 0,\neg H_s,\, d_{soap}>0\}\mapsto\{d_{soap}\downarrow\}$
  $\{N_{cab} = 0, N_{p\_clean} = 0,\neg H_s,\, d_{soap}=0\}\mapsto\{H_s\}$
  $\{N_{cab} = 0, N_{p\_clean} = 0,H_s,\, N_s>0,\, d_{si}>0\}\mapsto\{d_{si}\downarrow\}$
  $\{N_{cab} = 0, N_{p\_clean} = 0,H_s,\, N_s>0,\, d_{si}=0\}\mapsto\{N_s\downarrow,\, \neg H_s\}$

"""

def _create_goal_count_feature():
    goal_count_feature = []
    goal_count_feature.append(
        Count(
            type_tag='cabinet',
            wanted_value='=0',
            count_func=count_closed,
        )
    )
    goal_count_feature.append(
        Count(
            type_tag='electric_refrigerator',
            wanted_value='=0',
            count_func=count_closed,
        )
    )
    goal_count_feature.append(
        Count(
            type_tag='vegetable_oil',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
        )
    )
    
    goal_count_feature.append(
        Count(
            type_tag='plate',
            wanted_value='=0',
            count_func=count_not_cleaned, 
        )
    )
    
    goal_count_feature.append(
        Count(
            type_tag='plate',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
        )
    )
    
    goal_count_feature.append(
        Count(
            type_tag='blender',
            wanted_value='=0',
            count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
        )
    )
    
    goal_count_feature.append(
        Count(
            type_tag='apple',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
        )
    )
    
    goal_count_feature.append(
        Count(
            type_tag='casserole',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
        )
    )
    
    goal_count_feature.append(
        Count(
            type_tag='cabinet',
            wanted_value='=0',
            count_func=count_not_wiped,
        )
    )
    
    goal_count_feature.append(
        Count(
            type_tag='rag',
            wanted_value='=0',
            count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
        )
    )
    
    goal_count_feature.append(
        Count(
            type_tag='soap',
            wanted_value='=0',
            count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
        )
    )
    return goal_count_feature

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "open cabinet and fridge": [],
        "place oil in cabinet": [],
        "clean plate": [],
        "place plate in cabinet distinct from oil": [],
        "place blender on countertop": [],
        "place apple in fridge": [],
        "place casserole in fridge": [],
        "wipe cabinet": [],
        "rag near/inside sink": [],
        "soap next to sink": [],
    }
    # 1. open cabinet and fridge
    # 1.1 move towards closed cabinet 
    N_close_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    d_larger_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='>0',
    )
    # * effects
    d_decrease = DistanceToNearest(
        type_tag = 'cabinet',
        wanted_value = 'decrease',
    )
    
    rule_move_towards_closed_cabinet = Rule(
        preconditions = [N_close_larger_0, d_larger_0],
        effects = [d_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {cabinet})
"""
    )
    
    sketch_dict["open cabinet and fridge"].append(rule_move_towards_closed_cabinet)

    # 1.2 open the closed cabinet when reachable
    N_close_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    
    d_equal_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='=0',
    )
    # * effects
    N_close_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_closed,
    )
    
    rule_open_closed_cabinet = Rule(
        preconditions = [N_close_larger_0, d_equal_0],
        effects = [N_close_decrease],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["open cabinet and fridge"].append(rule_open_closed_cabinet)
    
    # 1.3 move towards closed fridge
    # also add ordering we only do this after all cabinets are opened
    N_cabinet_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_fridge_larger_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='>0',
        count_func=count_closed,
    )
    d_larger_0 = DistanceToNearest(
        type_tag='electric_refrigerator',
        wanted_value='>0',
    )
    # * effects
    d_decrease = DistanceToNearest(
        type_tag = 'electric_refrigerator',
        wanted_value = 'decrease',
    )
    
    rule_move_towards_closed_fridge = Rule(
        preconditions = [N_cabinet_equal_0, N_fridge_larger_0, d_larger_0],
        effects = [d_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {electric_refrigerator})
"""
    )
    sketch_dict["open cabinet and fridge"].append(rule_move_towards_closed_fridge) 
    # 1.4 open the closed fridge when reachable
    N_cabinet_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_fridge_larger_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='>0',
        count_func=count_closed,
    )
    d_equal_0 = DistanceToNearest(
        type_tag='electric_refrigerator',
        wanted_value='=0',
    )
    # * effects
    N_fridge_decrease = Count(
        type_tag='electric_refrigerator',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_closed_fridge = Rule(
        preconditions = [N_cabinet_equal_0, N_fridge_larger_0, d_equal_0],
        effects = [N_fridge_decrease],
        goal_clause_pattern="""
(is-opened {electric_refrigerator})
"""
    )
    sketch_dict["open cabinet and fridge"].append(rule_open_closed_fridge) 
    # 2. place oil in cabinet
    # 2.1 move towards oil
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    neg_H_o = Holding(
        wanted_value=False,
        type_tag='vegetable_oil',
    )
    d_oil_larger_0 = DistanceToNearest(
        type_tag='vegetable_oil',
        wanted_value='>0',
    )
    # * effects
    d_oil_decrease = DistanceToNearest(
        type_tag = 'vegetable_oil',
        wanted_value = 'decrease',
    )
    
    rule_move_to_oil = Rule(
        preconditions = [N_close_equal_0, neg_H_o, d_oil_larger_0],
        effects = [d_oil_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {vegetable_oil})
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_move_to_oil)
    # 2.2 pick up oil when reachable
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    neg_H_o = Holding(
        wanted_value=False,
        type_tag='vegetable_oil',
    )
    d_oil_equal_0 = DistanceToNearest(
        type_tag='vegetable_oil',
        wanted_value='=0',
    )
    # * effects
    H_o = Holding(
        wanted_value=True,
        type_tag='vegetable_oil',
    )
    
    rule_pick_up_oil = Rule(
        preconditions = [N_close_equal_0, neg_H_o, d_oil_equal_0],
        effects = [H_o],
        goal_clause_pattern="""
(inhandofrobot agent-01 {vegetable_oil})
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_pick_up_oil)
    # 2.3 move towards cabinet to place oil
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    H_o = Holding(
        wanted_value=True,
        type_tag='vegetable_oil',
    )
    N_oil_larger_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    d_cabinet_larger_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='>0',
        obj_name='cabinet_1',
    )
    # * effects
    d_cabinet_decrease = DistanceToNearest(
        type_tag = 'cabinet',
        wanted_value = 'decrease',
        obj_name='cabinet_1',
    )
    
    rule_move_to_cabinet = Rule(
        preconditions = [N_close_equal_0, H_o, N_oil_larger_0, d_cabinet_larger_0],
        effects = [d_cabinet_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {cabinet})
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_move_to_cabinet)
    # 2.4 place oil in cabinet when reachable
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    H_o = Holding(
        wanted_value=True,
        type_tag='vegetable_oil',
    )
    N_oil_larger_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    d_cabinet_equal_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='=0',
        obj_name='cabinet_1',
    )
    # * effects
    N_oil_decrease = Count(
        type_tag='vegetable_oil',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    
    rule_place_oil_in_cabinet = Rule(
        preconditions = [N_close_equal_0, H_o, N_oil_larger_0, d_cabinet_equal_0],
        effects = [N_oil_decrease], # actually we can also write (inside {vegetable_oil} {cabinet} ?loc ?dim) as d_cabinet_equal_0 will update the ground_var_value_dict
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {vegetable_oil} cabinet_1 ?loc ?dim)
)
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_place_oil_in_cabinet)
    
    # 3. clean plate
    # 3.1 move towards sink to toggle on
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    neg_Tog = isToggled(
        wanted_value=False,
        type_tag='sink',
    )
    d_sink_larger_0 = DistanceToNearest(
        type_tag='sink',
        wanted_value='>0',
    )
    
    # * effects
    d_sink_decrease = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = 'decrease',
    )
    
    rule_move_to_sink = Rule(
        preconditions = [n_plate_clean_larger_0, neg_Tog, d_sink_larger_0],
        effects = [d_sink_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    sketch_dict["clean plate"].append(rule_move_to_sink)
    # 3.2 toggle on sink when reachable
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    neg_Tog = isToggled(
        wanted_value=False,
        type_tag='sink',
    )
    d_sink_equal_0 = DistanceToNearest(
        type_tag='sink',
        wanted_value='=0',
    )
    # * effects
    Tog = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    rule_toggle_on_sink = Rule(
        preconditions = [n_plate_clean_larger_0, neg_Tog, d_sink_equal_0],
        effects = [Tog],
        goal_clause_pattern="""
(is-toggled {sink})
"""
    )
    sketch_dict["clean plate"].append(rule_toggle_on_sink)
    # 3.3 move towards rag 
    tog = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    
    d_rag_larger_0 = DistanceToNearest(
        type_tag='rag',
        wanted_value='>0',
    )
    # * effects
    d_rag_decrease = DistanceToNearest(
        type_tag = 'rag',
        wanted_value = 'decrease',
    )
    rule_move_to_rag = Rule(
        preconditions = [tog, neg_H_r, d_rag_larger_0, n_plate_clean_larger_0],
        effects = [d_rag_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {rag})
"""
    )
    sketch_dict["clean plate"].append(rule_move_to_rag)
    # 3.4 pick up rag when reachable
    tog = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    d_rag_equal_0 = DistanceToNearest(
        type_tag='rag',
        wanted_value='=0',
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag = Rule(
        preconditions = [tog, neg_H_r, d_rag_equal_0, n_plate_clean_larger_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["clean plate"].append(rule_pick_up_rag)
    # 3.5 move towards sink to soak rag
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    neg_Soaked = isSoaked(
        wanted_value=False,
        type_tag='rag',
    )
    d_sink_larger_0 = DistanceToNearest(
        type_tag='sink',
        wanted_value='>0',
    )
    # * effects
    d_sink_decrease = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = 'decrease',
    )
    rule_move_to_sink_to_soak_rag = Rule(
        preconditions = [H_r, neg_Soaked, d_sink_larger_0, n_plate_clean_larger_0],
        effects = [d_sink_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    sketch_dict["clean plate"].append(rule_move_to_sink_to_soak_rag)
    # 3.6 soak rag when reachable to sink
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    neg_Soaked = isSoaked(
        wanted_value=False,
        type_tag='rag',
    )
    d_sink_equal_0 = DistanceToNearest(
        type_tag='sink',
        wanted_value='=0',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    # * effects
    Soaked = isSoaked(
        wanted_value=True,
        type_tag='rag',
    )
    rule_soak_rag = Rule(
        preconditions = [H_r, neg_Soaked, d_sink_equal_0, n_plate_clean_larger_0],
        effects = [Soaked],
        goal_clause_pattern="""
(is-soaked {rag})
"""
    )
    
    sketch_dict["clean plate"].append(rule_soak_rag)
    # 3.7 move towards dirty plate
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    Soaked = isSoaked(
        wanted_value=True,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    d_plate_larger_0 = DistanceToNearest(
        type_tag='plate',
        wanted_value='>0',
    )
    # * effects
    d_plate_decrease = DistanceToNearest(
        type_tag = 'plate',
        wanted_value = 'decrease',
    )
    rule_move_to_dirty_plate = Rule(
        preconditions = [H_r, Soaked, n_plate_clean_larger_0, d_plate_larger_0],
        effects = [d_plate_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {plate})
"""
    )
    sketch_dict["clean plate"].append(rule_move_to_dirty_plate)
    # 3.8 clean dirty plate when reachable
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    Soaked = isSoaked(
        wanted_value=True,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    d_plate_equal_0 = DistanceToNearest(
        type_tag='plate',
        wanted_value='=0',
    )
    # * effects
    n_plate_clean_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=count_not_cleaned, 
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_clean_dirty_plate = Rule(
        preconditions = [H_r, Soaked, n_plate_clean_larger_0, d_plate_equal_0],
        effects = [n_plate_clean_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-stained {plate})
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["clean plate"].append(rule_clean_dirty_plate)
    
    # 4. place plate in cabinet distinct from oil
    # 4.1 move towards clean plate
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    neg_H_p = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    d_plate_larger_0 = DistanceToNearest(
        type_tag='plate',
        wanted_value='>0',
    )
    # * effects
    d_plate_decrease = DistanceToNearest(
        type_tag = 'plate',
        wanted_value = 'decrease',
    )
    rule_move_to_clean_plate = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, neg_H_p, n_plate_store_larger_0, d_plate_larger_0],
        effects = [d_plate_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {plate})
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_move_to_clean_plate)
    # 4.2 pick up clean plate when reachable
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    neg_H_p = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    d_plate_equal_0 = DistanceToNearest(
        type_tag='plate',
        wanted_value='=0',
    )
    # * effects
    H_p = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    rule_pick_up_clean_plate = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, neg_H_p, n_plate_store_larger_0, d_plate_equal_0],
        effects = [H_p],
        goal_clause_pattern="""
(inhandofrobot agent-01 {plate})
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_pick_up_clean_plate)
    # 4.3 move towards cabinet to place plate
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    H_p = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    d_cabinet_larger_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='>0',
        obj_name='cabinet_0',
    )
    # * effects
    d_cabinet_decrease = DistanceToNearest(
        type_tag = 'cabinet',
        wanted_value = 'decrease',
        obj_name='cabinet_0',
    )
    rule_move_to_cabinet_to_place_plate = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, H_p, n_plate_store_larger_0, d_cabinet_larger_0],
        effects = [d_cabinet_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 cabinet_0)
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_move_to_cabinet_to_place_plate)
    # 4.4 place plate in cabinet when reachable
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    H_p = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    d_cabinet_equal_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='=0',
        obj_name='cabinet_0',
    )
    # * effects
    n_plate_store_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    neg_H_p = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    rule_place_plate_in_cabinet = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, H_p, n_plate_store_larger_0, d_cabinet_equal_0],
        effects = [n_plate_store_decrease, neg_H_p],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {plate} cabinet_0 ?loc ?dim)
)
(not (inhandofrobot agent-01 {plate}))
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_place_plate_in_cabinet)
    
    # 5. place blender on countertop
    # 5.1 move towards blender
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    neg_H_b = Holding(
        wanted_value=False,
        type_tag='blender',
    )
    d_blender_larger_0 = DistanceToNearest(
        type_tag='blender',
        wanted_value='>0',
    )
    # * effects
    d_blender_decrease = DistanceToNearest(
        type_tag = 'blender',
        wanted_value = 'decrease',
    )
    rule_move_to_blender = Rule(
        preconditions = [n_blender_larger_0, neg_H_b, d_blender_larger_0],
        effects = [d_blender_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {blender})
"""
    )
    sketch_dict["place blender on countertop"].append(rule_move_to_blender)
    # 5.2 pick up blender when reachable
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    neg_H_b = Holding(
        wanted_value=False,
        type_tag='blender',
    )
    d_blender_equal_0 = DistanceToNearest(
        type_tag='blender',
        wanted_value='=0',
    )
    # * effects
    H_b = Holding(
        wanted_value=True,
        type_tag='blender',
    )
    rule_pick_up_blender = Rule(
        preconditions = [n_blender_larger_0, neg_H_b, d_blender_equal_0],
        effects = [H_b],
        goal_clause_pattern="""
(inhandofrobot agent-01 {blender})
"""
    )
    sketch_dict["place blender on countertop"].append(rule_pick_up_blender)
    # 5.3 move towards countertop to place blender
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    H_b = Holding(
        wanted_value=True,
        type_tag='blender',
    )
    d_countertop_larger_0 = DistanceToNearest(
        type_tag='countertop',
        wanted_value='>0',
    )
    # * effects
    d_countertop_decrease = DistanceToNearest(
        type_tag = 'countertop',
        wanted_value = 'decrease',
    )
    rule_move_to_countertop = Rule(
        preconditions = [n_blender_larger_0, H_b, d_countertop_larger_0],
        effects = [d_countertop_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {countertop})
"""
    )
    sketch_dict["place blender on countertop"].append(rule_move_to_countertop)
    # 5.4 place blender on countertop when reachable
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    H_b = Holding(
        wanted_value=True,
        type_tag='blender',
    )
    d_countertop_equal_0 = DistanceToNearest(
        type_tag='countertop',
        wanted_value='=0',
    )
    # * effects
    n_blender_decrease = Count(
        type_tag='blender',
        wanted_value='decrease',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    neg_H_b = Holding(
        wanted_value=False,
        type_tag='blender',
    )
    rule_place_blender_on_countertop = Rule(
        preconditions = [n_blender_larger_0, H_b, d_countertop_equal_0],
        effects = [n_blender_decrease, neg_H_b],
        goal_clause_pattern="""
(onTop {blender} countertop_0)
(not (inhandofrobot agent-01 {blender}))
"""
    )
    sketch_dict["place blender on countertop"].append(rule_place_blender_on_countertop)
    
    # 6. place apple in fridge
    # 6.1 move towards apple
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    neg_H_a = Holding(
        wanted_value=False,
        type_tag='apple',
    )
    d_apple_larger_0 = DistanceToNearest(
        type_tag='apple',
        wanted_value='>0',
    )
    n_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    d_apple_decrease = DistanceToNearest(
        type_tag = 'apple',
        wanted_value = 'decrease',
    )
    rule_move_to_apple = Rule(
        preconditions = [N_close_equal_0, neg_H_a, d_apple_larger_0, n_apple_larger_0],
        effects = [d_apple_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {apple})
"""
    )
    sketch_dict["place apple in fridge"].append(rule_move_to_apple)
    # 6.2 pick up apple when reachable
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    neg_H_a = Holding(
        wanted_value=False,
        type_tag='apple',
    )
    n_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    d_apple_equal_0 = DistanceToNearest(
        type_tag='apple',
        wanted_value='=0',
    )
    # * effects
    H_a = Holding(
        wanted_value=True,
        type_tag='apple',
    )
    rule_pick_up_apple = Rule(
        preconditions = [N_close_equal_0, neg_H_a, d_apple_equal_0, n_apple_larger_0],
        effects = [H_a],
        goal_clause_pattern="""
(inhandofrobot agent-01 {apple})
"""
    )
    sketch_dict["place apple in fridge"].append(rule_pick_up_apple)
    # 6.3 move towards fridge to place apple
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    H_a = Holding(
        wanted_value=True,
        type_tag='apple',
    )
    N_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    d_fridge_larger_0 = DistanceToNearest(
        type_tag='electric_refrigerator',
        wanted_value='>0',
    )
    # * effects
    d_fridge_decrease = DistanceToNearest(
        type_tag = 'electric_refrigerator',
        wanted_value = 'decrease',
    )
    rule_move_to_fridge = Rule(
        preconditions = [N_close_equal_0, H_a, N_apple_larger_0, d_fridge_larger_0],
        effects = [d_fridge_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {electric_refrigerator})
"""
    )
    sketch_dict["place apple in fridge"].append(rule_move_to_fridge)
    # 6.4 place apple in fridge when reachable
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    H_a = Holding(
        wanted_value=True,
        type_tag='apple',
    )
    N_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    d_fridge_equal_0 = DistanceToNearest(
        type_tag='electric_refrigerator',
        wanted_value='=0',
    )
    # * effects
    N_apple_decrease = Count(
        type_tag='apple',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    neg_H_a = Holding(
        wanted_value=False,
        type_tag='apple',
    )
    rule_place_apple_in_fridge = Rule(
        preconditions = [N_close_equal_0, H_a, N_apple_larger_0, d_fridge_equal_0],
        effects = [N_apple_decrease, neg_H_a],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {apple} {electric_refrigerator} ?loc ?dim)
)
(not (inhandofrobot agent-01 {apple}))
"""
    )
    sketch_dict["place apple in fridge"].append(rule_place_apple_in_fridge)
    
    # 7. place casserole in fridge
    # 7.1 move towards casserole
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    neg_H_c = Holding(
        wanted_value=False,
        type_tag='casserole',
    )
    d_casserole_larger_0 = DistanceToNearest(
        type_tag='casserole',
        wanted_value='>0',
    )
    n_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    d_casserole_decrease = DistanceToNearest(
        type_tag = 'casserole',
        wanted_value = 'decrease',
    )
    rule_move_to_casserole = Rule(
        preconditions = [N_close_equal_0, neg_H_c, d_casserole_larger_0, n_casserole_larger_0],
        effects = [d_casserole_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {casserole})
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_move_to_casserole)
    # 7.2 pick up casserole when reachable
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    neg_H_c = Holding(
        wanted_value=False,
        type_tag='casserole',
    )
    d_casserole_equal_0 = DistanceToNearest(
        type_tag='casserole',
        wanted_value='=0',
    )
    n_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    H_c = Holding(
        wanted_value=True,
        type_tag='casserole',
    )
    rule_pick_up_casserole = Rule(
        preconditions = [N_close_equal_0, neg_H_c, d_casserole_equal_0, n_casserole_larger_0],
        effects = [H_c],
        goal_clause_pattern="""
(inhandofrobot agent-01 {casserole})
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_pick_up_casserole)
    # 7.3 move towards fridge to place casserole
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    H_c = Holding(
        wanted_value=True,
        type_tag='casserole',
    )
    N_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    d_fridge_larger_0 = DistanceToNearest(
        type_tag='electric_refrigerator',
        wanted_value='>0',
    )
    # * effects
    d_fridge_decrease = DistanceToNearest(
        type_tag = 'electric_refrigerator',
        wanted_value = 'decrease',
    )
    rule_move_to_fridge_casserole = Rule(
        preconditions = [N_close_equal_0, H_c, N_casserole_larger_0, d_fridge_larger_0],
        effects = [d_fridge_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {electric_refrigerator})
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_move_to_fridge_casserole)
    # 7.4 place casserole in fridge when reachable
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    H_c = Holding(
        wanted_value=True,
        type_tag='casserole',
    )
    N_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    d_fridge_equal_0 = DistanceToNearest(
        type_tag='electric_refrigerator',
        wanted_value='=0',
    )
    # * effects
    N_casserole_decrease = Count(
        type_tag='casserole',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    neg_H_c = Holding(
        wanted_value=False,
        type_tag='casserole',
    )
    rule_place_casserole_in_fridge = Rule(
        preconditions = [N_close_equal_0, H_c, N_casserole_larger_0, d_fridge_equal_0],
        effects = [N_casserole_decrease, neg_H_c],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {casserole} {electric_refrigerator} ?loc ?dim)
)
(not (inhandofrobot agent-01 {casserole}))
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_place_casserole_in_fridge)
    
    # 8. wipe cabinet
    # 8.1 move towards rag 
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    d_rag_larger_0 = DistanceToNearest(
        type_tag='rag',
        wanted_value='>0',
    )
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    # * effects
    d_rag_decrease = DistanceToNearest(
        type_tag = 'rag',
        wanted_value = 'decrease',
    )
    rule_move_to_rag_for_wipe = Rule(
        preconditions = [neg_H_r, d_rag_larger_0, n_cabinet_dust_larger_0],
        effects = [d_rag_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {rag})
"""
    )
    sketch_dict["wipe cabinet"].append(rule_move_to_rag_for_wipe)
    # 8.2 pick up rag when reachable
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    d_rag_equal_0 = DistanceToNearest(
        type_tag='rag',
        wanted_value='=0',
    )
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag_for_wipe = Rule(
        preconditions = [neg_H_r, d_rag_equal_0, n_cabinet_dust_larger_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["wipe cabinet"].append(rule_pick_up_rag_for_wipe)
    # 8.3 move towards cabinet to wipe
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    d_cabinet_larger_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='>0',
    )
    # * effects
    d_cabinet_decrease = DistanceToNearest(
        type_tag = 'cabinet',
        wanted_value = 'decrease',
    )
    rule_move_to_cabinet_for_wipe = Rule(
        preconditions = [H_r, n_cabinet_dust_larger_0, d_cabinet_larger_0],
        effects = [d_cabinet_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {cabinet})
"""
    )
    sketch_dict["wipe cabinet"].append(rule_move_to_cabinet_for_wipe)
    # 8.4 wipe cabinet when reachable
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    d_cabinet_equal_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='=0',
    )
    # * effects
    n_cabinet_dust_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_wipe_cabinet = Rule(
        preconditions = [H_r, n_cabinet_dust_larger_0, d_cabinet_equal_0],
        effects = [n_cabinet_dust_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-dusted {cabinet})
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["wipe cabinet"].append(rule_wipe_cabinet)
    
    # 9. rag near/inside sink
    # 9.1 move towards rag
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    d_rag_larger_0 = DistanceToNearest(
        type_tag='rag',
        wanted_value='>0',
    )
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    # * effects
    d_rag_decrease = DistanceToNearest(
        type_tag = 'rag',
        wanted_value = 'decrease',
    )
    rule_move_to_rag_for_putaway = Rule(
        preconditions = [neg_H_r, d_rag_larger_0, n_cab_wiped_equal_0, n_plate_clean, n_rag_not_stored],
        effects = [d_rag_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {rag})
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_move_to_rag_for_putaway)
    # 9.2 pick up rag when reachable
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    d_rag_equal_0 = DistanceToNearest(
        type_tag='rag',
        wanted_value='=0',
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag_for_putaway = Rule(
        preconditions = [neg_H_r, d_rag_equal_0, n_cab_wiped_equal_0, n_plate_clean, n_rag_not_stored],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_pick_up_rag_for_putaway)
    # 9.3 move towards sink to put away rag
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    d_sink_larger_0 = DistanceToNearest(
        type_tag='sink',
        wanted_value='>0',
    )
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    # * effects
    d_sink_decrease = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = 'decrease',
    )
    rule_move_to_sink_for_putaway = Rule(
        preconditions = [H_r, n_cab_wiped_equal_0, n_plate_clean, d_sink_larger_0, n_rag_not_stored],
        effects = [d_sink_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_move_to_sink_for_putaway)
    # 9.4 put away rag when reachable to sink
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    d_sink_equal_0 = DistanceToNearest(
        type_tag='sink',
        wanted_value='=0',
    )
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    # * effects
    n_rag_not_stored_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_put_away_rag = Rule(
        preconditions = [H_r, n_cab_wiped_equal_0, n_plate_clean, d_sink_equal_0, n_rag_not_stored],
        effects = [n_rag_not_stored_decrease, neg_H_r],
        goal_clause_pattern="""
(or
    (nextto {rag} {sink})
    (exists (?loc - location ?dim - dimension)
        (inside {rag} {sink} ?loc ?dim)
    )
)
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_put_away_rag)
    
    # 10. soap next to sink
    # 10.1 move towards soap
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    d_soap_larger_0 = DistanceToNearest(
        type_tag='soap',
        wanted_value='>0',
    )
    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    # * effects
    d_soap_decrease = DistanceToNearest(
        type_tag = 'soap',
        wanted_value = 'decrease',
    )
    rule_move_to_soap = Rule(
        preconditions = [neg_H_s, n_plate_clean, n_cab_wiped_equal_0, d_soap_larger_0, n_soap_not_nextto_sink],
        effects = [d_soap_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {soap})
"""
    )
    sketch_dict["soap next to sink"].append(rule_move_to_soap)
    # 10.2 pick up soap when reachable
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    d_soap_equal_0 = DistanceToNearest(
        type_tag='soap',
        wanted_value='=0',
    )
    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    # * effects
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    rule_pick_up_soap = Rule(
        preconditions = [neg_H_s, n_plate_clean, n_cab_wiped_equal_0, d_soap_equal_0, n_soap_not_nextto_sink],
        effects = [H_s],
        goal_clause_pattern="""
(inhandofrobot agent-01 {soap})
"""
    )
    sketch_dict["soap next to sink"].append(rule_pick_up_soap)
    # 10.3 move towards sink to put down soap
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    d_sink_larger_0 = DistanceToNearest(
        type_tag='sink',
        wanted_value='>0',
    )
    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    # * effects
    d_sink_decrease = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = 'decrease',
    )
    rule_move_to_sink_for_soap = Rule(
        preconditions = [H_s, n_plate_clean, n_cab_wiped_equal_0, d_sink_larger_0, n_soap_not_nextto_sink],
        effects = [d_sink_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    sketch_dict["soap next to sink"].append(rule_move_to_sink_for_soap)
    # 10.4 put down soap next to sink when reachable
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    d_sink_equal_0 = DistanceToNearest(
        type_tag='sink',
        wanted_value='=0',
    )
    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    # * effects
    n_soap_not_nextto_sink_decrease = Count(
        type_tag='soap',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    rule_put_down_soap = Rule(
        preconditions = [H_s, n_plate_clean, n_cab_wiped_equal_0, d_sink_equal_0, n_soap_not_nextto_sink],
        effects = [n_soap_not_nextto_sink_decrease, neg_H_s],
        goal_clause_pattern="""
(nextto {soap} {sink})
(not (inhandofrobot agent-01 {soap}))
"""
    )
    sketch_dict["soap next to sink"].append(rule_put_down_soap)
    
    goal_count_feature = _create_goal_count_feature()
    
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

#### Sketch width **W=1** (single-feature guards; order enforces preconditions)
# - **Open Cabinet and Fridge** -- width 1
#    $\{\ N_{close} > 0 \}\ \mapsto\ \{\ N_{close}\downarrow \,\}$
# - **Oil placement — width 1**
#   $\{N_{close} = 0, \neg H_o \}\ \mapsto\ \{ H_o\}$
#   $\{N_{close} = 0, H_o, N_{oil} > 0 \}\ \mapsto\ \{ N_{oil}\downarrow, \neg H_o\}$
# - **Plate cleaning — width 1**
#   $\{\ \neg isToggle(sink) \}\ \mapsto\ \{isToggle(sink)\}$
#   $\{\ \neg H_{rag} \}\ \mapsto\ \{H_{rag}\}$
#   $\{\ H_{rag}, isToggle(sink), \neg Soaked \}\ \mapsto\ \{Soaked\}$ (width 1)
#    $\{\  H_{rag}, Soaked, N_{p\_clean} > 0 \}\ \mapsto\ \{\ N_{p\_clean}\downarrow \,\}$ width = 1

# - **Plate placement distinct from oil — width 1 (dominant)**
#    $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 , \neg H_{plate} \}\ \mapsto\ \{\, H_{plate}\}$ (width = 1)
#    $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 , H_{plate}, N_{p\_store} > 0, d_{c1} > 0 \}\ \mapsto\ \{\, d_{c1}\downarrow\}$ width = 0
#    $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 , H_{plate}, N_{p\_store} > 0, d_{c1} = 0\}\ \mapsto\ \{\, N_{p\_store}\downarrow, \neg H_{plate}\}$ width = 0
# - **Blender → countertop — width 1**
#   $\{\ N_b > 0, \neg H_b \}\ \mapsto\ \{\, H_b \}$
#    $\{\ N_b > 0, H_b \}\ \mapsto\ \{\, N_b\downarrow, \neg H_b \}$
# - **Apple → fridge — width 1**
#   $\{\neg H_f\}\mapsto\{H_f\}$ carry food 
#    $\{N_{close} =0 ,N_a > 0, H_f \}\ \mapsto\ \{\, N_a\downarrow, \neg H_f\}$
# - **Casserole → fridge — width 1**
#   $\{\neg H_f\}\mapsto\{H_f\}$ carry food 
#    $\{N_{close} =0 , N_{cas} >0, H_f \}\ \mapsto\ \{\, N_{cas}\downarrow, \neg H_f\}$
# - **Cabinet dusting — width 1**
#   $\{\ \neg H_{rag} \}\ \mapsto\ \{H_{rag}\}$
#    $\{N_{cab} > 0, H_{rag} \}\ \mapsto\ \{ N_{cab}\downarrow, \neg H_{rag}\}$
# - **Rag near/inside sink — width 1**
#   $\{\ \neg H_{rag} \}\ \mapsto\ \{H_{rag}\}$
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_r > 0, H_{rag}\}\ \mapsto\ \{\, N_r\downarrow, \neg H_{rag}\}$
# - **Soap next to sink — width 1**
#   $\{\ \neg H_{soap} \}\ \mapsto\ \{H_{soap}\}$
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_s > 0, H_{soap}\}\ \mapsto\ \{\, N_s\downarrow, \neg H_{soap}\}$

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "open cabinet and fridge": [],
        "place oil in cabinet": [],
        "clean plate": [],
        "place plate in cabinet distinct from oil": [],
        "place blender on countertop": [],
        "place apple in fridge": [],
        "place casserole in fridge": [],
        "wipe cabinet": [],
        "rag near/inside sink": [],
        "soap next to sink": [],
    }
    # 1. open cabinet and fridge
    # 1.1 open them 
    N_close_larger_0 = Count(
        type_tag='cabinet electric_refrigerator',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    N_close_decrease = Count(
        type_tag='cabinet electric_refrigerator',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet_fridge = Rule(
        preconditions = [N_close_larger_0],
        effects = [N_close_decrease],
        goal_clause_pattern="""
(is-opened {cabinet electric_refrigerator})
"""
    )
    sketch_dict["open cabinet and fridge"].append(rule_open_cabinet_fridge)
    # 2. place oil in cabinet
    # 2.1 holding oil 
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    neg_H_o = Holding(
        wanted_value=False,
        type_tag='vegetable_oil',
    )
    N_oil_larger_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    # * effects
    H_o = Holding(
        wanted_value=True,
        type_tag='vegetable_oil',
    )
    rule_pick_up_oil = Rule(
        preconditions = [N_oil_larger_0, N_close_equal_0, neg_H_o],
        effects = [H_o],
        goal_clause_pattern="""
(inhandofrobot agent-01 {vegetable_oil})
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_pick_up_oil)
    
    # 2.2 place oil in cabinet
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    H_o = Holding(
        wanted_value=True,
        type_tag='vegetable_oil',
    )
    N_oil_larger_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    # * effects
    N_oil_decrease = Count(
        type_tag='vegetable_oil',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    neg_H_o = Holding(
        wanted_value=False,
        type_tag='vegetable_oil',
    )
    rule_place_oil_in_cabinet = Rule(
        preconditions = [N_close_equal_0, H_o, N_oil_larger_0],
        effects = [N_oil_decrease, neg_H_o],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {vegetable_oil} cabinet_1 ?loc ?dim)
)
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_place_oil_in_cabinet)
    
    # 3. clean plate
    # 3.1 toggle sink on 
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    c_sink_not_toggle_larger_zero = Count(
        type_tag='sink',
        wanted_value='>0',
        count_func=count_not_toggled,
    )
    # * effects
    c_sink_not_toggle_decrease = Count(
        type_tag='sink',
        wanted_value='decrease',
        count_func=count_not_toggled,
    )
    rule_toggle_sink_on = Rule(
        preconditions = [n_plate_clean_larger_0, c_sink_not_toggle_larger_zero],
        effects = [c_sink_not_toggle_decrease],
        goal_clause_pattern="""
(is-toggled {sink})
"""
    )
    sketch_dict["clean plate"].append(rule_toggle_sink_on)
    
    # 3.2 pick up rag
    
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    c_sink_toggle_equal_0 = Count(
        type_tag='sink',
        wanted_value='=0',
        count_func=count_not_toggled,
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag_for_clean = Rule(
        preconditions = [neg_H_r, n_plate_clean_larger_0, c_sink_toggle_equal_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["clean plate"].append(rule_pick_up_rag_for_clean)
    # 3.3 soak rag 
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    c_sink_toggle_equal_0 = Count(
        type_tag='sink',
        wanted_value='=0',
        count_func=count_not_toggled,
    )
    c_rag_not_soaked_larger_0 = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    # * effects
    c_rag_not_soaked_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=count_not_soaked,
    )
    rule_soak_rag = Rule(
        preconditions = [H_r, n_plate_clean_larger_0, c_sink_toggle_equal_0, c_rag_not_soaked_larger_0],
        effects = [c_rag_not_soaked_decrease],
        goal_clause_pattern="""
(is-soaked {rag})
"""
    )
    sketch_dict["clean plate"].append(rule_soak_rag)
    # 3.4 clean plate
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    c_sink_toggle_equal_0 = Count(
        type_tag='sink',
        wanted_value='=0',
        count_func=count_not_toggled,
    )
    c_rag_soaked_equal_0 = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=count_not_soaked,
    )
    # * effects
    n_plate_clean_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=count_not_cleaned, 
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_clean_plate = Rule(
        preconditions = [H_r, n_plate_clean_larger_0, c_sink_toggle_equal_0, c_rag_soaked_equal_0],
        effects = [n_plate_clean_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-stained {plate})
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["clean plate"].append(rule_clean_plate)
    
    # 4. place plate in cabinet distinct from oil
    # 4.1 pick up plate
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    neg_H_p = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    # * effects
    H_p = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    rule_pick_up_plate = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, neg_H_p, n_plate_store_larger_0],
        effects = [H_p],
        goal_clause_pattern="""
(inhandofrobot agent-01 {plate})
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_pick_up_plate)
    # 4.2 move towards cabinet to place plate
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    H_p = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    d_cabinet_larger_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='>0',
        obj_name='cabinet_0',
    )
    # * effects
    d_cabinet_decrease = DistanceToNearest(
        type_tag = 'cabinet',
        wanted_value = 'decrease',
        obj_name='cabinet_0',
    )
    
    rule_move_to_cabinet_for_place_plate = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, H_p, n_plate_store_larger_0, d_cabinet_larger_0],
        effects = [d_cabinet_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 cabinet_0)
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_move_to_cabinet_for_place_plate)
    # 4.3 place plate in cabinet
    
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    H_p = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    d_cabinet_equal_0 = DistanceToNearest(
        type_tag='cabinet',
        wanted_value='=0',
        obj_name='cabinet_0',
    )
    # * effects
    n_plate_store_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    neg_H_p = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    rule_place_plate_in_cabinet = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, H_p, n_plate_store_larger_0, d_cabinet_equal_0],
        effects = [n_plate_store_decrease, neg_H_p],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {plate} cabinet_0 ?loc ?dim)
)
(not (inhandofrobot agent-01 {plate}))
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_place_plate_in_cabinet)
    
    # 5. place blender on countertop
    # 5.1 pick up blender
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    neg_H_b = Holding(
        wanted_value=False,
        type_tag='blender',
    )

    # * effects
    H_b = Holding(
        wanted_value=True,
        type_tag='blender',
    )
    rule_pick_up_blender = Rule(
        preconditions = [n_blender_larger_0, neg_H_b],
        effects = [H_b],
        goal_clause_pattern="""
(inhandofrobot agent-01 {blender})
"""
    )
    sketch_dict["place blender on countertop"].append(rule_pick_up_blender)
    # 5.2 put down blender on countertop
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    H_b = Holding(
        wanted_value=True,
        type_tag='blender',
    )
    # * effects
    n_blender_decrease = Count(
        type_tag='blender',
        wanted_value='decrease',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    neg_H_b = Holding(
        wanted_value=False,
        type_tag='blender',
    )
    rule_place_blender_on_countertop = Rule(
        preconditions = [n_blender_larger_0, H_b],
        effects = [n_blender_decrease, neg_H_b],
        goal_clause_pattern="""
(onTop {blender} countertop_0)
(not (inhandofrobot agent-01 {blender}))
"""
    )
    sketch_dict["place blender on countertop"].append(rule_place_blender_on_countertop)
    
    # 6. place apple in fridge
    # 6.1 pick up apple
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    neg_H_a = Holding(
        wanted_value=False,
        type_tag='apple',
    )
    n_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    H_a = Holding(
        wanted_value=True,
        type_tag='apple',
    )
    rule_pick_up_apple = Rule(
        preconditions = [neg_H_a, N_close_equal_0, n_apple_larger_0],
        effects = [H_a],
        goal_clause_pattern="""
(inhandofrobot agent-01 {apple})
"""
    )
    sketch_dict["place apple in fridge"].append(rule_pick_up_apple)
    # 6.2 place apple in fridge
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    H_a = Holding(
        wanted_value=True,
        type_tag='apple',
    )
    N_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_apple_decrease = Count(
        type_tag='apple',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    neg_H_a = Holding(
        wanted_value=False,
        type_tag='apple',
    )
    rule_place_apple_in_fridge = Rule(
        preconditions = [H_a, N_close_equal_0, N_apple_larger_0],
        effects = [N_apple_decrease, neg_H_a],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {apple} {electric_refrigerator} ?loc ?dim)
)
(not (inhandofrobot agent-01 {apple}))
"""
    )
    sketch_dict["place apple in fridge"].append(rule_place_apple_in_fridge)
    
    # 7. place casserole in fridge
    # 7.1 pick up casserole
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    neg_H_c = Holding(
        wanted_value=False,
        type_tag='casserole',
    )
    n_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    H_c = Holding(
        wanted_value=True,
        type_tag='casserole',
    )
    rule_pick_up_casserole = Rule(
        preconditions = [neg_H_c, N_close_equal_0, n_casserole_larger_0],
        effects = [H_c],
        goal_clause_pattern="""
(inhandofrobot agent-01 {casserole})
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_pick_up_casserole)
    # 7.2 place casserole in fridge
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    H_c = Holding(
        wanted_value=True,
        type_tag='casserole',
    )
    N_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_casserole_decrease = Count(
        type_tag='casserole',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    neg_H_c = Holding(
        wanted_value=False,
        type_tag='casserole',
    )
    rule_place_casserole_in_fridge = Rule(
        preconditions = [H_c, N_close_equal_0, N_casserole_larger_0],
        effects = [N_casserole_decrease, neg_H_c],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {casserole} {electric_refrigerator} ?loc ?dim)
)
(not (inhandofrobot agent-01 {casserole}))
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_place_casserole_in_fridge)
    # 8. wipe cabinet
    # 8.1 pick up rag
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag_for_wipe = Rule(
        preconditions = [neg_H_r, n_cabinet_dust_larger_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["wipe cabinet"].append(rule_pick_up_rag_for_wipe)
    # 8.2 wipe cabinet
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    # * effects
    n_cabinet_dust_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_wipe_cabinet = Rule(
        preconditions = [H_r, n_cabinet_dust_larger_0],
        effects = [n_cabinet_dust_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-dusted {cabinet})
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["wipe cabinet"].append(rule_wipe_cabinet)
    # 9. rag near/inside sink
    # 9.1 pick up rag
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag_for_put_away = Rule(
        preconditions = [neg_H_r, n_rag_not_stored, n_cab_wiped_equal_0, n_plate_clean],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_pick_up_rag_for_put_away)
    # 9.2 put away rag near/inside sink
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    # * effects
    n_rag_not_stored_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_put_away_rag = Rule(
        preconditions = [H_r, n_cab_wiped_equal_0, n_plate_clean, n_rag_not_stored],
        effects = [n_rag_not_stored_decrease, neg_H_r],
        goal_clause_pattern="""
(or
    (nextto {rag} {sink})
    (exists (?loc - location ?dim - dimension)
        (inside {rag} {sink} ?loc ?dim)
    )
)
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_put_away_rag)
    
    # 10. soap next to sink
    # 10.1 pick up soap
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    
    # * effects
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    rule_pick_up_soap = Rule(
        preconditions = [neg_H_s, n_plate_clean, n_cab_wiped_equal_0, n_soap_not_nextto_sink],
        effects = [H_s],
        goal_clause_pattern="""
(inhandofrobot agent-01 {soap})
"""
    )
    sketch_dict["soap next to sink"].append(rule_pick_up_soap)
    # 10.2  put down soap
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )

    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    # * effects
    n_soap_not_nextto_sink_decrease = Count(
        type_tag='soap',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    rule_put_down_soap = Rule(
        preconditions = [H_s, n_plate_clean, n_cab_wiped_equal_0, n_soap_not_nextto_sink],
        effects = [n_soap_not_nextto_sink_decrease, neg_H_s],
        goal_clause_pattern="""
(nextto {soap} {sink})
(not (inhandofrobot agent-01 {soap}))
"""
    )
    sketch_dict["soap next to sink"].append(rule_put_down_soap)
    
    goal_count_feature = _create_goal_count_feature()
    
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
    
    
#### Sketch width **W=2** (split “carry & place” into carry / place)

# - **Open Cabinet and Fridge** -- width 1
#    $\{\ N_{close} > 0 \}\ \mapsto\ \{\ N_{close}\downarrow \,\}$
# - **Oil placement — width 2**
#   $\{N_{close} = 0, N_{oil} > 0 \}\ \mapsto\ \{ N_{oil}\downarrow\}$
# - **Plate cleaning — width 2**
#   $\{\ \neg H_{rag} \}\ \mapsto\ \{H_{rag}\}$
#   $\{\ H_{rag}, \neg Soaked \}\ \mapsto\ \{Soaked\}$ (width 2, need to ensure sink is toggled and next to sink)
#    $\{\  H_{rag}, Soaked, N_{p\_clean} > 0 \}\ \mapsto\ \{\ N_{p\_clean}\downarrow \,\}$ width = 1
# - **Plate placement distinct from oil — width 2 (dominant)**
#    $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 , \neg H_{plate} \}\ \mapsto\ \{\, H_{plate}\}$ (width = 1)
#    $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 , H_{plate}, N_{p\_store} > 0 \}\ \mapsto\ \{\, N_{p\_store}\downarrow, \neg H_{plate}\}$ (next_to, and c1 != c2)
# - **Blender → countertop — width 2**
#    $\{\ N_b > 0 \}\ \mapsto\ \{\, N_b\downarrow\}$
# - **Apple → fridge — width 2**
#    $\{N_{close} =0 ,N_a > 0 \}\ \mapsto\ \{\, N_a\downarrow\}$
# - **Casserole → fridge — width 2**
#    $\{N_{close} =0 , N_{cas} >0 \}\ \mapsto\ \{\, N_{cas}\downarrow\}$
# - **Cabinet dusting — width 2**
#    $\{N_{cab} > 0 \}\ \mapsto\ \{ N_{cab}\downarrow \}$
# - **Rag near/inside sink — width 2**
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_r > 0\}\ \mapsto\ \{\, N_r\downarrow\}$
# - **Soap next to sink — width 2**
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_s > 0\}\ \mapsto\ \{\, N_s\downarrow\}$    
    
def create_width_2_sketch(env, map_id, domain_name, window=None):    
    width = 2
    sketch_dict = {
        "open cabinet and fridge": [],
        "place oil in cabinet": [],
        "clean plate": [],
        "place plate in cabinet distinct from oil": [],
        "place blender on countertop": [],
        "place apple in fridge": [],
        "place casserole in fridge": [],
        "wipe cabinet": [],
        "rag near/inside sink": [],
        "soap next to sink": [],
    }
    # 1. open cabinet and fridge
    # 1.1 open them 
    N_close_larger_0 = Count(
        type_tag='cabinet electric_refrigerator',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    N_close_decrease = Count(
        type_tag='cabinet electric_refrigerator',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet_fridge = Rule(
        preconditions = [N_close_larger_0],
        effects = [N_close_decrease],
        goal_clause_pattern="""
(is-opened {cabinet electric_refrigerator})
"""
    )
    sketch_dict["open cabinet and fridge"].append(rule_open_cabinet_fridge)
    # 2. place oil in cabinet
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_larger_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    # * effects
    N_oil_decrease = Count(
        type_tag='vegetable_oil',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    rule_place_oil_in_cabinet = Rule(
        preconditions = [N_oil_larger_0, N_close_equal_0],
        effects = [N_oil_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {vegetable_oil} cabinet_1 ?loc ?dim)
)
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_place_oil_in_cabinet)
    # 3. clean plate
    # 3.1 pick up rag
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag_for_clean = Rule(
        preconditions = [neg_H_r, n_plate_clean_larger_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["clean plate"].append(rule_pick_up_rag_for_clean)
    # 3.2 soak rag
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    
    c_rag_not_soaked_larger_0 = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    # * effects
    c_rag_not_soaked_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=count_not_soaked,
    )
    rule_soak_rag = Rule(
        preconditions = [H_r, n_plate_clean_larger_0, c_rag_not_soaked_larger_0],
        effects = [c_rag_not_soaked_decrease],
        goal_clause_pattern="""
(is-soaked {rag})
"""
    )
    sketch_dict["clean plate"].append(rule_soak_rag)
    # 3.3 clean plate
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    c_rag_soaked_equal_0 = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=count_not_soaked,
    )
    # * effects
    n_plate_clean_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=count_not_cleaned, 
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_clean_plate = Rule(
        preconditions = [H_r, n_plate_clean_larger_0, c_rag_soaked_equal_0],
        effects = [n_plate_clean_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-stained {plate})
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["clean plate"].append(rule_clean_plate)
    
    # 4 . place plate in cabinet distinct from oil
    # 4.1 pick up plate
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    neg_H_p = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    # * effects
    H_p = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    rule_pick_up_plate = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, neg_H_p, n_plate_store_larger_0],
        effects = [H_p],
        goal_clause_pattern="""
(inhandofrobot agent-01 {plate})
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_pick_up_plate)

    # 4.2 place plate in cabinet
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    H_p = Holding(
        wanted_value=True,
        type_tag='plate',
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    
    # * effects
    n_plate_store_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    neg_H_p = Holding(
        wanted_value=False,
        type_tag='plate',
    )
    rule_place_plate_in_cabinet = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, H_p, n_plate_store_larger_0],
        effects = [n_plate_store_decrease, neg_H_p],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {plate} cabinet_0 ?loc ?dim)
)
(not (inhandofrobot agent-01 {plate}))
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_place_plate_in_cabinet)
    
    # 5. place blender on countertop
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    n_blender_decrease = Count(
        type_tag='blender',
        wanted_value='decrease',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    rule_place_blender_on_countertop = Rule(
        preconditions = [n_blender_larger_0],
        effects = [n_blender_decrease],
        goal_clause_pattern="""
(onTop {blender} countertop_0)
"""
    )
    sketch_dict["place blender on countertop"].append(rule_place_blender_on_countertop)
    
    # 6. place apple in fridge
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_apple_decrease = Count(
        type_tag='apple',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    rule_place_apple_in_fridge = Rule(
        preconditions = [N_close_equal_0, n_apple_larger_0],
        effects = [N_apple_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {apple} {electric_refrigerator} ?loc ?dim)
)
"""
    )
    sketch_dict["place apple in fridge"].append(rule_place_apple_in_fridge)
    # 7. place casserole in fridge
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_casserole_decrease = Count(
        type_tag='casserole',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    rule_place_casserole_in_fridge = Rule(
        preconditions = [N_close_equal_0, n_casserole_larger_0],
        effects = [N_casserole_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {casserole} {electric_refrigerator} ?loc ?dim)
)
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_place_casserole_in_fridge)
    # 8. wipe cabinet
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    # * effects
    n_cabinet_dust_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    rule_wipe_cabinet = Rule(
        preconditions = [n_cabinet_dust_larger_0],
        effects = [n_cabinet_dust_decrease],
        goal_clause_pattern="""
(is-not-dusted {cabinet})
"""
    )
    sketch_dict["wipe cabinet"].append(rule_wipe_cabinet)
    
    # 9. rag near/inside sink
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    # * effects
    n_rag_not_stored_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    rule_put_away_rag = Rule(
        preconditions = [n_cab_wiped_equal_0, n_plate_clean, n_rag_not_stored],
        effects = [n_rag_not_stored_decrease],
        goal_clause_pattern="""
(or
    (nextto {rag} {sink})
    (exists (?loc - location ?dim - dimension)
        (inside {rag} {sink} ?loc ?dim)
    )
)
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_put_away_rag)
    # 10. soap next to sink
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    # * effects
    n_soap_not_nextto_sink_decrease = Count(
        type_tag='soap',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    rule_put_down_soap = Rule(
        preconditions = [n_plate_clean, n_cab_wiped_equal_0, n_soap_not_nextto_sink],
        effects = [n_soap_not_nextto_sink_decrease],
        goal_clause_pattern="""
(nextto {soap} {sink})
"""
    )
    sketch_dict["soap next to sink"].append(rule_put_down_soap) 
    
    goal_count_feature = _create_goal_count_feature()
    
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

#### Sketch width **W=3** (separate oil selection and plate prep)
# - **Open Cabinet and Fridge** -- width 1
#    $\{\ N_{close} > 0 \}\ \mapsto\ \{\ N_{close}\downarrow \,\}$
# - **Oil placement — width 2**
#   $\{N_{close} = 0, N_{oil} > 0 \}\ \mapsto\ \{ N_{oil}\downarrow\}$
# - **Plate cleaning — width 3** CHANGED
#    $\{\ N_{p\_clean} > 0 \}\ \mapsto\ \{\ N_{p\_clean}\downarrow \,\}$
# - **Plate placement distinct from oil — width 3 (dominant)**, carrying, next_to, not same CHANGED
#    $\{N_{close} =0 , N_{oil}=0, N_{p\_clean} =0 , N_{p\_store} > 0 \}\ \mapsto\ \{\, N_{p\_store}\downarrow\}$
# - **Blender → countertop — width 2**
#    $\{\ N_b > 0 \}\ \mapsto\ \{\, N_b\downarrow\}$
# - **Apple → fridge — width 2**
#    $\{N_{close} =0 ,N_a > 0 \}\ \mapsto\ \{\, N_a\downarrow\}$
# - **Casserole → fridge — width 2**
#    $\{N_{close} =0 , N_{cas} >0 \}\ \mapsto\ \{\, N_{cas}\downarrow\}$
# - **Cabinet dusting — width 2**
#    $\{N_{cab} > 0 \}\ \mapsto\ \{ N_{cab}\downarrow \}$
# - **Rag near/inside sink — width 2**
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_r > 0\}\ \mapsto\ \{\, N_r\downarrow\}$
# - **Soap next to sink — width 2**
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_s > 0\}\ \mapsto\ \{\, N_s\downarrow\}$

def create_width_3_sketch(env, map_id, domain_name, window=None):    
    width = 3
    sketch_dict = {
        "open cabinet and fridge": [],
        "place oil in cabinet": [],
        "clean plate": [],
        "place plate in cabinet distinct from oil": [],
        "place blender on countertop": [],
        "place apple in fridge": [],
        "place casserole in fridge": [],
        "wipe cabinet": [],
        "rag near/inside sink": [],
        "soap next to sink": [],
    }
    # 1. open cabinet and fridge
    # 1.1 open them 
    N_close_larger_0 = Count(
        type_tag='cabinet electric_refrigerator',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    N_close_decrease = Count(
        type_tag='cabinet electric_refrigerator',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet_fridge = Rule(
        preconditions = [N_close_larger_0],
        effects = [N_close_decrease],
        goal_clause_pattern="""
(is-opened {cabinet electric_refrigerator})
"""
    )
    sketch_dict["open cabinet and fridge"].append(rule_open_cabinet_fridge)
    # 2. place oil in cabinet
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_larger_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    # * effects
    N_oil_decrease = Count(
        type_tag='vegetable_oil',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    rule_place_oil_in_cabinet = Rule(
        preconditions = [N_oil_larger_0, N_close_equal_0],
        effects = [N_oil_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {vegetable_oil} cabinet_1 ?loc ?dim)
)
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_place_oil_in_cabinet)
    # 3. clean plate
    n_plate_clean_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=count_not_cleaned, 
    )
    # * effects
    n_plate_clean_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=count_not_cleaned, 
    )
    rule_clean_plate = Rule(
        preconditions = [n_plate_clean_larger_0],
        effects = [n_plate_clean_decrease],
        goal_clause_pattern="""
(is-not-stained {plate})
"""
    )
    sketch_dict["clean plate"].append(rule_clean_plate)
    
    # 4 . place plate in cabinet distinct from oil
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_clean_equal_0 = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    # * effects
    n_plate_store_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0'),
    )
    rule_place_plate_in_cabinet = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_clean_equal_0, n_plate_store_larger_0],
        effects = [n_plate_store_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {plate} cabinet_0 ?loc ?dim)
)
"""
    )
    sketch_dict["place plate in cabinet distinct from oil"].append(rule_place_plate_in_cabinet)
    
    # rest is the same as width 2
    # 5. place blender on countertop
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    n_blender_decrease = Count(
        type_tag='blender',
        wanted_value='decrease',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    rule_place_blender_on_countertop = Rule(
        preconditions = [n_blender_larger_0],
        effects = [n_blender_decrease],
        goal_clause_pattern="""
(onTop {blender} countertop_0)
"""
    )
    sketch_dict["place blender on countertop"].append(rule_place_blender_on_countertop)
    
    # 6. place apple in fridge
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_apple_decrease = Count(
        type_tag='apple',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    rule_place_apple_in_fridge = Rule(
        preconditions = [N_close_equal_0, n_apple_larger_0],
        effects = [N_apple_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {apple} {electric_refrigerator} ?loc ?dim)
)
"""
    )
    sketch_dict["place apple in fridge"].append(rule_place_apple_in_fridge)
    # 7. place casserole in fridge
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_casserole_decrease = Count(
        type_tag='casserole',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    rule_place_casserole_in_fridge = Rule(
        preconditions = [N_close_equal_0, n_casserole_larger_0],
        effects = [N_casserole_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {casserole} {electric_refrigerator} ?loc ?dim)
)
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_place_casserole_in_fridge)
    # 8. wipe cabinet
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    # * effects
    n_cabinet_dust_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    rule_wipe_cabinet = Rule(
        preconditions = [n_cabinet_dust_larger_0],
        effects = [n_cabinet_dust_decrease],
        goal_clause_pattern="""
(is-not-dusted {cabinet})
"""
    )
    sketch_dict["wipe cabinet"].append(rule_wipe_cabinet)
    
    # 9. rag near/inside sink
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    # * effects
    n_rag_not_stored_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    rule_put_away_rag = Rule(
        preconditions = [n_cab_wiped_equal_0, n_plate_clean, n_rag_not_stored],
        effects = [n_rag_not_stored_decrease],
        goal_clause_pattern="""
(or
    (nextto {rag} {sink})
    (exists (?loc - location ?dim - dimension)
        (inside {rag} {sink} ?loc ?dim)
    )
)
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_put_away_rag)
    # 10. soap next to sink
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    # * effects
    n_soap_not_nextto_sink_decrease = Count(
        type_tag='soap',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    rule_put_down_soap = Rule(
        preconditions = [n_plate_clean, n_cab_wiped_equal_0, n_soap_not_nextto_sink],
        effects = [n_soap_not_nextto_sink_decrease],
        goal_clause_pattern="""
(nextto {soap} {sink})
"""
    )
    sketch_dict["soap next to sink"].append(rule_put_down_soap) 
    
    goal_count_feature = _create_goal_count_feature()
    
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

#### Sketch width **W=4** (factor one atom from the W=5 plate rule)
# - **Open Cabinet and Fridge**
#    $\{\ N_{close} > 0 \}\ \mapsto\ \{\ N_{close}\downarrow \,\}$
# - **Oil placement — width 2**
# ​	 $\{N_{close} = 0, N_{oil} > 0 \}\ \mapsto\ \{ N_{oil}\downarrow\}$
# - **Plate placement distinct from oil — width 4 (dominant)**
#    $\{N_{close} =0 , N_{oil}=0, N_{p\_store} > 0 \}\ \mapsto\ \{\, N_{p\_store}\downarrow\}$
# - **Blender → countertop — width 2**
#    $\{\ N_b > 0 \}\ \mapsto\ \{\, N_b\downarrow\}$
# - **Apple → fridge — width 2**
#    $\{N_{close} =0 ,N_a > 0 \}\ \mapsto\ \{\, N_a\downarrow\}$
# - **Casserole → fridge — width 2**
#    $\{N_{close} =0 , N_{cas} >0 \}\ \mapsto\ \{\, N_{cas}\downarrow\}$
# - **Cabinet dusting — width 2**
#    $\{N_{cab} > 0 \}\ \mapsto\ \{ N_{cab}\downarrow \}$
# - **Rag near/inside sink — width 2**
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_r > 0\}\ \mapsto\ \{\, N_r\downarrow\}$
# - **Soap next to sink — width 2**
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_s > 0\}\ \mapsto\ \{\, N_s\downarrow\}$

def create_width_4_sketch(env, map_id, domain_name, window=None):
    width = 4 
    sketch_dict = {
        "open cabinet and fridge": [],
        "place oil in cabinet": [],
        "place cleaned plate in cabinet distinct from oil": [],
        "place blender on countertop": [],
        "place apple in fridge": [],
        "place casserole in fridge": [],
        "wipe cabinet": [],
        "rag near/inside sink": [],
        "soap next to sink": [],
    }
    # 1. open cabinet and fridge
    # 1.1 open them 
    N_close_larger_0 = Count(
        type_tag='cabinet electric_refrigerator',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    N_close_decrease = Count(
        type_tag='cabinet electric_refrigerator',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet_fridge = Rule(
        preconditions = [N_close_larger_0],
        effects = [N_close_decrease],
        goal_clause_pattern="""
(is-opened {cabinet electric_refrigerator})
"""
    )
    sketch_dict["open cabinet and fridge"].append(rule_open_cabinet_fridge)
    # 2. place oil in cabinet
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_larger_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    # * effects
    N_oil_decrease = Count(
        type_tag='vegetable_oil',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    rule_place_oil_in_cabinet = Rule(
        preconditions = [N_oil_larger_0, N_close_equal_0],
        effects = [N_oil_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {vegetable_oil} cabinet_1 ?loc ?dim)
)
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_place_oil_in_cabinet)
    # 3-4: place cleaned plate in cabinet distinct from oil
    N_close_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_closed,
    )
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0', additional_predicate_on_object_type='is-not-stained'),
    )
    # * effects
    n_plate_store_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0', additional_predicate_on_object_type='is-not-stained'),
    )
    rule_place_plate_in_cabinet = Rule(
        preconditions = [N_close_equal_0, N_oil_equal_0, n_plate_store_larger_0],
        effects = [n_plate_store_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {plate} cabinet_0 ?loc ?dim)
)
(is-not-stained {plate})
"""
    )
    sketch_dict["place cleaned plate in cabinet distinct from oil"].append(rule_place_plate_in_cabinet)
    # rest is the same as width 2
    # 5. place blender on countertop
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    n_blender_decrease = Count(
        type_tag='blender',
        wanted_value='decrease',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    rule_place_blender_on_countertop = Rule(
        preconditions = [n_blender_larger_0],
        effects = [n_blender_decrease],
        goal_clause_pattern="""
(onTop {blender} countertop_0)
"""
    )
    sketch_dict["place blender on countertop"].append(rule_place_blender_on_countertop)
    
    # 6. place apple in fridge
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_apple_decrease = Count(
        type_tag='apple',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    rule_place_apple_in_fridge = Rule(
        preconditions = [N_close_equal_0, n_apple_larger_0],
        effects = [N_apple_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {apple} {electric_refrigerator} ?loc ?dim)
)
"""
    )
    sketch_dict["place apple in fridge"].append(rule_place_apple_in_fridge)
    # 7. place casserole in fridge
    N_close_equal_0 = Count(
        type_tag='electric_refrigerator',
        wanted_value='=0',
        count_func=count_closed,
    )
    n_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_casserole_decrease = Count(
        type_tag='casserole',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    rule_place_casserole_in_fridge = Rule(
        preconditions = [N_close_equal_0, n_casserole_larger_0],
        effects = [N_casserole_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {casserole} {electric_refrigerator} ?loc ?dim)
)
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_place_casserole_in_fridge)
    # 8. wipe cabinet
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    # * effects
    n_cabinet_dust_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    rule_wipe_cabinet = Rule(
        preconditions = [n_cabinet_dust_larger_0],
        effects = [n_cabinet_dust_decrease],
        goal_clause_pattern="""
(is-not-dusted {cabinet})
"""
    )
    sketch_dict["wipe cabinet"].append(rule_wipe_cabinet)
    
    # 9. rag near/inside sink
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    # * effects
    n_rag_not_stored_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    rule_put_away_rag = Rule(
        preconditions = [n_cab_wiped_equal_0, n_plate_clean, n_rag_not_stored],
        effects = [n_rag_not_stored_decrease],
        goal_clause_pattern="""
(or
    (nextto {rag} {sink})
    (exists (?loc - location ?dim - dimension)
        (inside {rag} {sink} ?loc ?dim)
    )
)
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_put_away_rag)
    # 10. soap next to sink
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    # * effects
    n_soap_not_nextto_sink_decrease = Count(
        type_tag='soap',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    rule_put_down_soap = Rule(
        preconditions = [n_plate_clean, n_cab_wiped_equal_0, n_soap_not_nextto_sink],
        effects = [n_soap_not_nextto_sink_decrease],
        goal_clause_pattern="""
(nextto {soap} {sink})
"""
    )
    sketch_dict["soap next to sink"].append(rule_put_down_soap) 
    
    goal_count_feature = _create_goal_count_feature()
    
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

### Sketch width **W=5** (decomposed by sub-problem; max width is 5)

# Only the **plate–oil distinct-cabinet placement** needs width 5; all other rules are ≤3.
# - **Oil placement — width 3**
#    $\{N_{oil} > 0 \}\ \mapsto\ \{ N_{oil}\downarrow\}$
# - **Plate placement distinct from oil — width 5 (dominant)**
#    $\{ N_{oil} =0 ,N_{p\_store} > 0 \}\ \mapsto\ \{\, N_{p\_store}\downarrow\}$
# - **Blender → countertop — width 2**
#    $\{\ N_b > 0 \}\ \mapsto\ \{\, N_b\downarrow\}$
# - **Apple → fridge — width 3**
#    $\{N_a > 0 \}\ \mapsto\ \{\, N_a\downarrow\}$
# - **Casserole → fridge — width 3**
#    $\{N_{cas} >0 \}\ \mapsto\ \{\, N_{cas}\downarrow\}$
# - **Cabinet dusting — width 2**
#    $\{N_{cab} > 0 \}\ \mapsto\ \{ N_{cab}\downarrow \}$
# - **Rag near/inside sink — width 2**
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_r > 0\}\ \mapsto\ \{\, N_r\downarrow\}$
# - **Soap next to sink — width 2**
#    $\{N_{cab} = 0, N_{p\_clean} = 0, N_s > 0\}\ \mapsto\ \{\, N_s\downarrow\}$
    
def create_width_5_sketch(env, map_id, domain_name, window=None):    
    width = 5
    sketch_dict = {
        "place oil in cabinet": [],
        "place cleaned plate in cabinet distinct from oil": [],
        "place blender on countertop": [],
        "place apple in fridge": [],
        "place casserole in fridge": [],
        "wipe cabinet": [],
        "rag near/inside sink": [],
        "soap next to sink": [],
    }
    # 1. place oil in cabinet
    N_oil_larger_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    # * effects
    N_oil_decrease = Count(
        type_tag='vegetable_oil',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    rule_place_oil_in_cabinet = Rule(
        preconditions = [N_oil_larger_0],
        effects = [N_oil_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {vegetable_oil} cabinet_1 ?loc ?dim)
)
"""
    )
    sketch_dict["place oil in cabinet"].append(rule_place_oil_in_cabinet)
    # 3-4: place cleaned plate in cabinet distinct from oil
    N_oil_equal_0 = Count(
        type_tag='vegetable_oil',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_1'),
    )
    n_plate_store_larger_0 = Count(
        type_tag='plate',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0', additional_predicate_on_object_type='is-not-stained'),
    )
    # * effects
    n_plate_store_decrease = Count(
        type_tag='plate',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_object, container_obj_name='cabinet_0', additional_predicate_on_object_type='is-not-stained'),
    )
    rule_place_plate_in_cabinet = Rule(
        preconditions = [N_oil_equal_0, n_plate_store_larger_0],
        effects = [n_plate_store_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {plate} cabinet_0 ?loc ?dim)
)
(is-not-stained {plate})
"""
    )
    sketch_dict["place cleaned plate in cabinet distinct from oil"].append(rule_place_plate_in_cabinet)
    
    # 5. place blender on countertop
    n_blender_larger_0 = Count(
        type_tag='blender',
        wanted_value='>0',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    n_blender_decrease = Count(
        type_tag='blender',
        wanted_value='decrease',
        count_func=partial(count_not_ontop, surface_obj_name='countertop_0'),
    )
    rule_place_blender_on_countertop = Rule(
        preconditions = [n_blender_larger_0],
        effects = [n_blender_decrease],
        goal_clause_pattern="""
(onTop {blender} countertop_0)
"""
    )
    sketch_dict["place blender on countertop"].append(rule_place_blender_on_countertop)
    
    # 6. place apple in fridge
    n_apple_larger_0 = Count(
        type_tag='apple',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_apple_decrease = Count(
        type_tag='apple',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    rule_place_apple_in_fridge = Rule(
        preconditions = [n_apple_larger_0],
        effects = [N_apple_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {apple} {electric_refrigerator} ?loc ?dim)
)
"""
    )
    sketch_dict["place apple in fridge"].append(rule_place_apple_in_fridge)
    # 7. place casserole in fridge
    n_casserole_larger_0 = Count(
        type_tag='casserole',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    # * effects
    N_casserole_decrease = Count(
        type_tag='casserole',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='electric_refrigerator'),
    )
    rule_place_casserole_in_fridge = Rule(
        preconditions = [n_casserole_larger_0],
        effects = [N_casserole_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {casserole} {electric_refrigerator} ?loc ?dim)
)
"""
    )
    sketch_dict["place casserole in fridge"].append(rule_place_casserole_in_fridge)
    
    # 8. wipe cabinet
    n_cabinet_dust_larger_0 = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_not_wiped,
    ) 
    # * effects
    n_cabinet_dust_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    rule_wipe_cabinet = Rule(
        preconditions = [n_cabinet_dust_larger_0],
        effects = [n_cabinet_dust_decrease],
        goal_clause_pattern="""
(is-not-dusted {cabinet})
"""
    )
    sketch_dict["wipe cabinet"].append(rule_wipe_cabinet)
    
    # 9. rag near/inside sink
    # 9. rag near/inside sink
    n_rag_not_stored = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    # * effects
    n_rag_not_stored_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=True),
    )
    rule_put_away_rag = Rule(
        preconditions = [n_cab_wiped_equal_0, n_plate_clean, n_rag_not_stored],
        effects = [n_rag_not_stored_decrease],
        goal_clause_pattern="""
(or
    (nextto {rag} {sink})
    (exists (?loc - location ?dim - dimension)
        (inside {rag} {sink} ?loc ?dim)
    )
)
"""
    )
    sketch_dict["rag near/inside sink"].append(rule_put_away_rag)
    # 10. soap next to sink
    n_plate_clean = Count(
        type_tag='plate',
        wanted_value='=0',
        count_func=count_not_cleaned, 
    )
    n_cab_wiped_equal_0 = Count(
        type_tag='cabinet',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    n_soap_not_nextto_sink = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    # * effects
    n_soap_not_nextto_sink_decrease = Count(
        type_tag='soap',
        wanted_value='decrease',
        count_func=partial(count_not_near_target_type, target_type_name='sink', include_inside=False),
    )
    rule_put_down_soap = Rule(
        preconditions = [n_plate_clean, n_cab_wiped_equal_0, n_soap_not_nextto_sink],
        effects = [n_soap_not_nextto_sink_decrease],
        goal_clause_pattern="""
(nextto {soap} {sink})
"""
    )
    sketch_dict["soap next to sink"].append(rule_put_down_soap) 
    
    goal_count_feature = _create_goal_count_feature()
    
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/cleaning_up_the_kitchen_only.py --map_id 15644487 --domain_name cleaning_up_the_kitchen_only --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/cleaning_up_the_kitchen_only/p15644487-cleaning_up_the_kitchen_only_plans.json', want_render=False)
    
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