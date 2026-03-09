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
11
### Feature set (mirroring the plywood features)
- $H_r$: “holding a scrub_brush”
- $H_f$: “holding a focused item (teapot, kettle, or pan)”
- $p_r$: distance to the nearest scrub_brush
- $p_f$: distance to the nearest focused item
- $p_s$: distance to the sink
- $p_c$: distance to a valid cabinet target
- $Clean_f$: whether the focused item is clean (not stained)
- $N_f$: number of focused items not stored
- $is\_{toggled}$: whether the sink is toggled on
- $is\_{soaked}$: whether the scrub_brush is soaked
- $is\_{opened}$: whether the cabinet is opened


### Sketch with W = 0 (approximate)

**Clean task**
$\{\neg H_r, N_f > 0, p_r > 0\} \mapsto \{p_r \downarrow, p_f ?, p_s ?\}$  ; move toward the nearest scrub_brush
$\{\neg H_r, N_f > 0, p_r = 0\} \mapsto \{H_r, p_f ?, p_s ?\}$  ; pick up the scrub_brush when reachable
$\{\neg is\_toggled, p_s > 0\} \mapsto \{p_s \downarrow, p_f ?, p_r ?\}$  ; move toward the sink
$\{\neg is\_toggled, p_s = 0\} \mapsto \{is\_toggled, p_f ?, p_r ?\}$  ; toggle the sink on when reachable
$\{H_r, is\_toggled, \neg is\_soaked, p_s =0 \} \mapsto \{is\_soaked, \neg H_r, p_f ?, p_r = 0\}$  ; soak the scrub_brush when at the sink
$\{\neg H_r, is\_soaked, \neg Clean_f, p_r = 0\} \mapsto \{H_r, p_f ? \}$  ; pick up a soaked scrub_brush when reachable
$\{H_r, is\_soaked, \neg Clean_f, p_f > 0\} \mapsto \{p_f \downarrow, p_r ?, p_s ?\}$  ; move toward the nearest focused item
$\{H_r, is\_soaked, \neg Clean_f, p_f = 0\} \mapsto \{Clean_f, \neg H_r, p_r ?, p_s ?\}$  ; clean the focused item when reachable

**Store task**

$\{\neg is\_opened, p_c > 0\} \mapsto \{p_c \downarrow\}$  ; move toward the cabinet
$\{\neg is\_opened, p_c = 0\} \mapsto \{is\_opened\}$  ; open the cabinet when reachable
$\{\neg H_f, N_f > 0, Clean_f, p_f > 0\} \mapsto \{p_f \downarrow, p_c ?\}$  ; move toward the nearest unboxed focused item
$\{\neg H_f, N_f > 0, Clean_f, p_f = 0\} \mapsto \{H_f, p_c ?\}$  ; pick up the focused item when reachable
$\{H_f, N_f > 0, Clean_f, p_c > 0\} \mapsto \{p_c \downarrow\}$  ; move toward a chosen cabinet spot
$\{H_f, N_f > 0, Clean_f, p_c = 0\} \mapsto \{\neg H_f, N_f \downarrow, p_f ?, p_r ?, p_s ?\}$  ; place the focused item into the cabinet when at the spot
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "washing pots and pans": [],
        "storing pots and pans": []
    }
    
    # 1 washing pots and pans
    # 1.1 move toward the nearest scrub_brush
    # * precons
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    n_f_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    
    p_r_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='scrub_brush',
    )
    # * effects
    p_r_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='scrub_brush',
    )
    rule_move_to_scrub_brush = Rule(
        preconditions=[neg_H_r, n_f_larger_0, p_r_larger_0],
        effects=[p_r_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {scrub_brush})
"""
    )
    sketch_dict["washing pots and pans"].append(rule_move_to_scrub_brush)
    # 1.2 pick up the scrub_brush when reachable
    # * precons
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    n_f_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    p_r_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='scrub_brush',
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    rule_pick_up_scrub_brush = Rule(
        preconditions=[neg_H_r, n_f_larger_0, p_r_equal_0],
        effects=[H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {scrub_brush})
"""
    )
    sketch_dict["washing pots and pans"].append(rule_pick_up_scrub_brush)
    # 1.3 move toward the sink if scrub_brush not soaked
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    neg_soaked = isSoaked(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    p_s_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='sink',
    )
    # * effects
    p_s_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='sink',
    )
    rule_move_to_sink = Rule(
        preconditions=[H_r, neg_soaked, p_s_larger_0],
        effects=[p_s_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    sketch_dict["washing pots and pans"].append(rule_move_to_sink)
    # 1.4 toggle the sink on when reachable
    # * precons
    neg_is_toggled = isToggled(
        wanted_value=False,
        type_tag='sink',
    )
    p_s_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='sink',
    )
    # * effects 
    is_toggled = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    rule_toggle_sink = Rule(
        preconditions=[neg_is_toggled, p_s_equal_0],
        effects=[is_toggled],
        goal_clause_pattern="""
(is-toggled {sink})
"""
    )
    sketch_dict["washing pots and pans"].append(rule_toggle_sink)
    # 1.5 soak the scrub_brush when at the sink and it is toggled on
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    is_toggled = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    neg_soaked = isSoaked(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    p_s_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='sink',
    )
    # * effects
    soaked = isSoaked(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    rule_soak_scrub_brush = Rule(
        preconditions=[H_r, is_toggled, neg_soaked, p_s_equal_0],
        effects=[soaked, neg_H_r],
        goal_clause_pattern="""
(is-soaked {scrub_brush})
(not (inhandofrobot agent-01 {scrub_brush}))
"""
    )
    sketch_dict["washing pots and pans"].append(rule_soak_scrub_brush)
    
    # 1.6 move towards the nearest focused item if holding a soaked scrub_brush and focused item not clean
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    soaked = isSoaked(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    n_f_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    p_f_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='teapot kettle pan',
    )
    # * effects
    p_f_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='teapot kettle pan',
    )
    rule_move_to_focused = Rule(
        preconditions=[H_r, soaked, n_f_larger_0, p_f_larger_0],
        effects=[p_f_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {teapot kettle pan})
"""
    )
    sketch_dict["washing pots and pans"].append(rule_move_to_focused)
    # 1.7 clean the focused item when reachable
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    soaked = isSoaked(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    n_f_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    p_f_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='teapot kettle pan',
    )
    # * effects
    n_f_decrease = Count(
        type_tag='teapot kettle pan',
        wanted_value='decrease',
        count_func=count_not_cleaned,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    rule_clean_focused = Rule(
        preconditions=[H_r, soaked, n_f_larger_0, p_f_equal_0],
        effects=[n_f_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-stained {teapot kettle pan})
(not (inhandofrobot agent-01 {scrub_brush}))
"""
    )
    sketch_dict["washing pots and pans"].append(rule_clean_focused)
    # 2 storing pots and pans
    # 2.1 move towards nearest item 
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='teapot kettle pan',
    )
    n_clean_equal_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    n_store_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    p_f_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='teapot kettle pan',
    )
    # * effects
    p_f_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='teapot kettle pan',
    )
    rule_move_to_focused = Rule(
        preconditions=[neg_H, n_clean_equal_0, n_store_larger_0, p_f_larger_0],
        effects=[p_f_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {teapot kettle pan})
"""
    )
    sketch_dict["storing pots and pans"].append(rule_move_to_focused)
    # 2.2 pick up the item when reachable
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='teapot kettle pan',
    )
    n_clean_equal_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    n_store_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    p_f_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='teapot kettle pan',
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='teapot kettle pan',
    )
    rule_pick_up_focused = Rule(
        preconditions=[neg_H, n_clean_equal_0, n_store_larger_0, p_f_equal_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {teapot kettle pan})
"""
    )
    sketch_dict["storing pots and pans"].append(rule_pick_up_focused)
    # 2.3 move toward a chosen cabinet spot
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='teapot kettle pan',
    )
    n_clean_equal_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    n_store_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
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
        preconditions=[H, n_clean_equal_0, n_store_larger_0, p_c_larger_0],
        effects=[p_c_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {cabinet})
"""
    )
    sketch_dict["storing pots and pans"].append(rule_move_to_cabinet)
    # 2.4 open the cabinet when reachable
    # * precons
    neg_is_opened = isOpened(
        wanted_value=False,
        type_tag='cabinet',
    )
    p_c_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='cabinet',
    )
    # * effects
    is_opened = isOpened(
        wanted_value=True,
        type_tag='cabinet',
    )
    rule_open_cabinet = Rule(
        preconditions=[neg_is_opened, p_c_equal_0],
        effects=[is_opened],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["storing pots and pans"].append(rule_open_cabinet)
    # 2.5 place the item into the cabinet when at the spot
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='teapot kettle pan',
    )
    n_clean_equal_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    n_store_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    is_opened = isOpened(
        wanted_value=True,
        type_tag='cabinet',
    )
    p_c_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='cabinet',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='teapot kettle pan',
    )
    n_store_decrease = Count(
        type_tag='teapot kettle pan',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    
    rule_place_focused_in_cabinet = Rule(
        preconditions=[H, n_clean_equal_0, n_store_larger_0, is_opened, p_c_equal_0],
        effects=[neg_H, n_store_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {teapot kettle pan} {cabinet} ?loc ?dim)
)
(not (inhandofrobot agent-01 {teapot kettle pan}))
"""
    )
    sketch_dict["storing pots and pans"].append(rule_place_focused_in_cabinet)
    
    goal_count_feature = [
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
        ),
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=count_not_cleaned,
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

# **Clean task**
# $\{\neg H_r, \neg is\_soaked\} \mapsto \{H_r\}$  ; pick up a rag (width 1)
# $\{\neg toggled\} \mapsto \{is\_toggled\}$  ; toggle the sink on (width 1)
# $\{H_r, is\_toggled, \neg is\_soaked\} \mapsto \{is\_soaked, \neg H_r\}$  ; soak the rag (width 1)
# $\{\neg H_r, is\_soaked, \neg Clean_f\} \mapsto \{H_r\}$  ; pick up a soaked rag (width 1)
# $\{H_r, is\_soaked, \neg Clean_f\} \mapsto \{Clean_f, \neg H_r\}$  ; clean a focused item (width 1)
# **Store task**
# $\{\neg is\_opened\} \mapsto \{is\_opened\}$  ; open the cabinet (width 1)
# $\{\neg H_f, Clean_f\} \mapsto \{H_f\}$  ; pick up an unboxed focused item (width 1)
# $\{H_f, N_f > 0, Clean_f, is\_opened\} \mapsto \{\neg H_f, N_f \downarrow\}$  ; place the carried focused item into some valid cabinet spot (width 1)

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "washing pots and pans": [],
        "storing pots and pans": []
    }
    
    # 1 washing pots and pans
    # 1.1 pick up the scrub_brush
    # * precons
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    n_f_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    rule_pick_up_scrub_brush = Rule(
        preconditions=[neg_H_r, n_f_larger_0],
        effects=[H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {scrub_brush})
"""
    )
    sketch_dict["washing pots and pans"].append(rule_pick_up_scrub_brush)
    # 1.2 toggle the sink on
    # * precons
    c_sink_not_toggled = Count(
        type_tag='sink',
        wanted_value='>0',
        count_func=count_not_toggled,
    )
    # * effects
    c_sink_decrease = Count(
        type_tag='sink',
        wanted_value='decrease',
        count_func=count_not_toggled,
    )
    rule_toggle_sink = Rule(
        preconditions=[c_sink_not_toggled],
        effects=[c_sink_decrease],
        goal_clause_pattern="""
(is-toggled {sink})
"""
    )
    sketch_dict["washing pots and pans"].append(rule_toggle_sink)
    # 1.3 soak the scrub_brush
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    c_sink_equal_0 = Count(
        type_tag='sink',
        wanted_value='=0',
        count_func=count_not_toggled,
    )
    neg_soaked = isSoaked(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    # * effects
    soaked = isSoaked(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    rule_soak_scrub_brush = Rule(
        preconditions=[H_r, c_sink_equal_0, neg_soaked],
        effects=[soaked, neg_H_r],
        goal_clause_pattern="""
(is-soaked {scrub_brush})
(not (inhandofrobot agent-01 {scrub_brush}))
"""
    )
    sketch_dict["washing pots and pans"].append(rule_soak_scrub_brush)
    # 1.4  clean the focused item
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    soaked = isSoaked(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    n_f_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    n_f_decrease = Count(
        type_tag='teapot kettle pan',
        wanted_value='decrease',
        count_func=count_not_cleaned,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    rule_clean_focused = Rule(
        preconditions=[H_r, soaked, n_f_larger_0],
        effects=[n_f_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-stained {teapot kettle pan})
(not (inhandofrobot agent-01 {scrub_brush}))
"""
    )
    sketch_dict["washing pots and pans"].append(rule_clean_focused)
    # 2 storing pots and pans
    # 2.1 open the cabinet
    # * precons
    c_cabinet_not_opened = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    c_cabinet_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet = Rule(
        preconditions=[c_cabinet_not_opened],
        effects=[c_cabinet_decrease],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["storing pots and pans"].append(rule_open_cabinet)
    # 2.2 pick up the focused item
    # * precons
    neg_H = Holding(
        wanted_value=False,
        type_tag='teapot kettle pan',
    )
    n_clean_equal_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    n_store_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    # * effects
    H = Holding(
        wanted_value=True,
        type_tag='teapot kettle pan',
    )
    rule_pick_up_focused = Rule(
        preconditions=[neg_H, n_clean_equal_0, n_store_larger_0],
        effects=[H],
        goal_clause_pattern="""
(inhandofrobot agent-01 {teapot kettle pan})
"""
    )
    sketch_dict["storing pots and pans"].append(rule_pick_up_focused)
    # 2.3 place the carried focused item into some valid cabinet spot
    # * precons
    H = Holding(
        wanted_value=True,
        type_tag='teapot kettle pan',
    )
    n_clean_equal_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    n_store_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    is_opened = isOpened(
        wanted_value=True,
        type_tag='cabinet',
    )
    # * effects
    neg_H = Holding(
        wanted_value=False,
        type_tag='teapot kettle pan',
    )
    n_store_decrease = Count(
        type_tag='teapot kettle pan',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    rule_place_focused_in_cabinet = Rule(
        preconditions=[H, n_clean_equal_0, n_store_larger_0, is_opened],
        effects=[neg_H, n_store_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {teapot kettle pan} {cabinet} ?loc ?dim)
)
(not (inhandofrobot agent-01 {teapot kettle pan}))
"""
    )
    sketch_dict["storing pots and pans"].append(rule_place_focused_in_cabinet)
    
    goal_count_feature = [
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
        ),
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=count_not_cleaned,
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

### Sketch with W = 2 ; decompose cleaning item and separate opening cabinet
# **Clean task**
# $\{\neg H_r, \neg is\_soaked\} \mapsto \{H_r\}$  ; pick up a rag (width 1)
# $\{H_r, \neg is\_soaked\} \mapsto \{is\_soaked, \neg H_r\}$  ; soak the rag (width 2)
# $\{\neg H_r, is\_soaked, \neg Clean_f\} \mapsto \{H_r\}$  ; pick up a soaked rag (width 1)
# $\{H_r, is\_soaked, \neg Clean_f\} \mapsto \{Clean_f, \neg H_r\}$  ; clean a focused item (width 1)
# **Store task**
# $\{\neg is\_opened\} \mapsto \{is\_opened\}$  ; open the cabinet (width 1)
# $\{N_f > 0, Clean_f, is\_opened\} \mapsto \{N_f \downarrow, Clean_f?\}$  ; store a focused item (width 2)

def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "washing pots and pans": [],
        "storing pots and pans": []
    }
    
    # 1 washing pots and pans
    # 1.1 pick up the scrub_brush
    # * precons
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    n_f_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    rule_pick_up_scrub_brush = Rule(
        preconditions=[neg_H_r, n_f_larger_0],
        effects=[H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {scrub_brush})
"""
    )
    sketch_dict["washing pots and pans"].append(rule_pick_up_scrub_brush)
    # 1.2 soak the scrub_brush
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    c_scrub_brush_not_soaked = Count(
        type_tag='scrub_brush',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    # * effects
    c_scrub_brush_decrease = Count(
        type_tag='scrub_brush',
        wanted_value='decrease',
        count_func=count_not_soaked,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    rule_soak_scrub_brush = Rule(
        preconditions=[H_r, c_scrub_brush_not_soaked],
        effects=[c_scrub_brush_decrease, neg_H_r],
        goal_clause_pattern="""
(is-soaked {scrub_brush})
(not (inhandofrobot agent-01 {scrub_brush}))
"""
    )
    sketch_dict["washing pots and pans"].append(rule_soak_scrub_brush)
    # 1.3  clean the focused item
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    soaked = isSoaked(
        wanted_value=True,
        type_tag='scrub_brush',
    )
    n_f_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    n_f_decrease = Count(
        type_tag='teapot kettle pan',
        wanted_value='decrease',
        count_func=count_not_cleaned,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='scrub_brush',
    )
    rule_clean_focused = Rule(
        preconditions=[H_r, soaked, n_f_larger_0],
        effects=[n_f_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-stained {teapot kettle pan})
(not (inhandofrobot agent-01 {scrub_brush}))
"""
    )
    sketch_dict["washing pots and pans"].append(rule_clean_focused)
    # 2 storing pots and pans
    # 2.1 open the cabinet
    # * precons
    c_cabinet_not_opened = Count(
        type_tag='cabinet',
        wanted_value='>0',
        count_func=count_closed,
    )
    # * effects
    c_cabinet_decrease = Count(
        type_tag='cabinet',
        wanted_value='decrease',
        count_func=count_closed,
    )
    rule_open_cabinet = Rule(
        preconditions=[c_cabinet_not_opened],
        effects=[c_cabinet_decrease],
        goal_clause_pattern="""
(is-opened {cabinet})
"""
    )
    sketch_dict["storing pots and pans"].append(rule_open_cabinet)
    # 2.2 store a focused item
    # * precons
    n_clean_equal_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    n_store_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    is_opened = isOpened(
        wanted_value=True,
        type_tag='cabinet',
    )
    # * effects
    n_store_decrease = Count(
        type_tag='teapot kettle pan',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    rule_store_focused = Rule(
        preconditions=[n_clean_equal_0, n_store_larger_0, is_opened],
        effects=[n_store_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {teapot kettle pan} {cabinet} ?loc ?dim)
)
"""
    )
    sketch_dict["storing pots and pans"].append(rule_store_focused)
    
    goal_count_feature = [
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
        ),
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=count_not_cleaned,
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

### Sketch with W = 3 ; factor out cleaning item
# $\{\neg Clean_f\} \mapsto \{Clean_f\}$  ; clean a focused item (width 3)
# $\{N_f > 0, Clean_f\} \mapsto \{N_f \downarrow, Clean_f?\}$  ; store a focused item (width 3)
    
def create_width_3_sketch(env, map_id, domain_name, window=None):
    width = 3
    sketch_dict = {
        "washing pots and pans": [],
        "storing pots and pans": []
    }
    # 1 washing pots and pans
    # 1.1 clean the focused item
    # * precons
    n_f_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    n_f_decrease = Count(
        type_tag='teapot kettle pan',
        wanted_value='decrease',
        count_func=count_not_cleaned,
    )
    rule_clean_focused = Rule(
        preconditions=[n_f_larger_0],
        effects=[n_f_decrease],
        goal_clause_pattern="""
(is-not-stained {teapot kettle pan})
"""
    )
    sketch_dict["washing pots and pans"].append(rule_clean_focused)
    # 2 storing pots and pans
    # 2.1 store a focused item
    # * precons
    n_clean_equal_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    n_store_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    # * effects
    n_store_decrease = Count(
        type_tag='teapot kettle pan',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
    )
    rule_store_focused = Rule(
        preconditions=[n_clean_equal_0, n_store_larger_0],
        effects=[n_store_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {teapot kettle pan} {cabinet} ?loc ?dim)
)
"""
    )
    sketch_dict["storing pots and pans"].append(rule_store_focused)
    
    goal_count_feature = [
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
        ),
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=count_not_cleaned,
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

### Sketch with W = 4 ; state each sub-goal (no mentioning of is-not-stained, inside, is-toggled, is-soaked, is-opened)
# $\{N_f > 0\} \mapsto \{N_f \downarrow\}$  ; wash and store a focused item (width 4)
def create_width_4_sketch(env, map_id, domain_name, window=None):
    width = 4
    sketch_dict = {
        "storing cleaned pots and pans": []
    }
    # 1 storing cleaned pots and pans
    # precons
    n_store_larger_0 = Count(
        type_tag='teapot kettle pan',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet', additional_predicate_on_object_type='is-not-stained'),
    )
    
    # effects
    n_store_decrease = Count(
        type_tag='teapot kettle pan',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='cabinet', additional_predicate_on_object_type='is-not-stained'),
    )
    rule_store_cleaned_focused = Rule(
        preconditions=[n_store_larger_0],
        effects=[n_store_decrease],
        goal_clause_pattern="""
(exists (?loc - location ?dim - dimension)
    (inside {teapot kettle pan} {cabinet} ?loc ?dim)
)
(is-not-stained {teapot kettle pan})
"""
    )
    sketch_dict["storing cleaned pots and pans"].append(rule_store_cleaned_focused)
    
    goal_count_feature = [
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=partial(count_not_inside_target_type, target_type_name='cabinet'),
        ),
        Count(
            type_tag='teapot kettle pan',
            wanted_value='=0',
            count_func=count_not_cleaned,
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/washing_pots_and_pans.py --map_id 219607168 --domain_name washing_pots_and_pans --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/washing_pots_and_pans/p219607168-washing_pots_and_pans_plans.json', want_render=False)
    
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