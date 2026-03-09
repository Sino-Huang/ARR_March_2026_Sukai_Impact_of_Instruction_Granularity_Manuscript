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
6
### Feature set (mirroring the plywood features)
**features related to slice lemon**
- $H_l$: holding a lemon
- $p_l$: distance to the nearest lemon
- $is_{knife}$: whether the knife is currently held
- $p_{knife}$: distance to the knife
- $H_{knife}$: whether the knife is currently held
  
**features related to putting teapot on stove**
- $H_t$: holding a teapot
- $p_t$: distance to the nearest teapot

**features related to tea bag**
- $H_{tb}$: holding a tea bag
- $p_{tb}$: distance to the nearest tea bag
- $at_{tb}$: whether the tea bag is at the same location as the teapot
- $is_{soaked}$: whether the tea bag is soaked
- $p_{sink}$: distance to the sink
- $toggle_{sink}$: whether the sink is toggled on


**features related to stove**
- $p_s$: distance to the stove
- $toggle_{stove}$: whether the stove is toggled on

### Sketch with W = 0 (approximate, including distance features)

**Toggle stove**
$\{ \neg is\_toggled(stove), p_s > 0 \} \mapsto \{ p_s \downarrow \}$  ; move toward the stove
$\{ \neg is\_toggled(stove), p_s = 0 \} \mapsto \{ is\_toggled(stove) \}$  ; toggle the stove on when reachable


**Slice lemon**
$\{ \neg H_{knife}, p_{knife} > 0, \neg \text{is\_sliced(lemon)} \} \mapsto \{ p_{knife} \downarrow \}$  ; move toward the knife
$\{ \neg H_{knife}, p_{knife} = 0, \neg \text{is\_sliced(lemon)} \} \mapsto \{ H_{knife} \}$  ; pick up the knife when reachable
$\{ H_{knife}, \neg \text{is\_sliced(lemon)}, p_{l} > 0 \} \mapsto \{ p_{l} \downarrow \}$  ; move toward the nearest lemon
$\{ H_{knife}, \neg \text{is\_sliced(lemon)}, p_{l} = 0 \} \mapsto \{ \text{is\_sliced(lemon)} \}$  ; slice the lemon when reachable
$\{ H_{knife},  \text{is\_sliced(lemon)} \} \mapsto \{ \neg H_{knife}, p_{knife} ? \}$  ; put down the knife 

**Put teapot on stove and put tea bag in teapot**
$\{ \neg atsamelocation(tea\_bag, teapot), \neg onTop(teapot, stove), \neg H_t, p_t > 0 \} \mapsto \{ p_t \downarrow \}$  ; move toward the nearest teapot
$\{ \neg atsamelocation(tea\_bag, teapot), \neg onTop(teapot, stove), \neg H_t, p_t = 0 \} \mapsto \{ H_t \}$  ; pick up the teapot when reachable
$\{ \neg atsamelocation(tea\_bag, teapot), \neg onTop(teapot, stove), H_t, p_s > 0 \} \mapsto \{ p_s \downarrow \}$  ; move toward the stove
$\{ \neg atsamelocation(tea\_bag, teapot), \neg onTop(teapot, stove), H_t, p_s = 0 \} \mapsto \{ onTop(teapot, stove), \neg H_t, p_t ? \}$  ; put the teapot on the stove when reachable
$\{ \neg atsamelocation(tea\_bag, teapot), onTop(teapot, stove), \neg H_{tb},  p_{tb} > 0 \} \mapsto \{ p_{tb} \downarrow \}$  ; move toward the nearest soaked tea bag
$\{ \neg atsamelocation(tea\_bag, teapot), onTop(teapot, stove), \neg H_{tb},  p_{tb} = 0 \} \mapsto \{ H_{tb} \}$  ; pick up the soaked tea bag when reachable
$\{ \neg atsamelocation(tea\_bag, teapot), onTop(teapot, stove), H_{tb},  p_t > 0 \} \mapsto \{ p_t \downarrow \}$  ; move toward the teapot
$\{ H_{tb}, \neg atsamelocation(tea\_bag, teapot), onTop(teapot, stove),  p_t = 0 \} \mapsto \{ atsamelocation(tea\_bag, teapot), \neg H_{tb}, p_{tb} ? ,is\_soaked(tea\_bag)\}$  ; put the tea bag in the teapot when reachable
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        "toggle stove": [],
        "slice lemon": [],
        "put teapot on stove and put tea bag in teapot": []
    }
    # 1. toggle stove
    # 1.1 move toward the stove
    # * precons
    neg_is_toggled_stove = isToggled(
        wanted_value=False,
        type_tag='stove',
    )
    p_s_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='stove',
    )
    # * effects
    p_s_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='stove',
    )
    rule_move_to_stove = Rule(
        preconditions=[neg_is_toggled_stove, p_s_larger_0],
        effects=[p_s_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {stove})
"""
    )
    sketch_dict["toggle stove"].append(rule_move_to_stove)
    # 1.2 toggle the stove on when reachable
    # * precons
    neg_is_toggled_stove = isToggled(
        wanted_value=False,
        type_tag='stove',
    )
    p_s_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='stove',
    )
    # * effects
    is_toggled_stove = isToggled(
        wanted_value=True,
        type_tag='stove',
    )
    rule_toggle_stove = Rule(
        preconditions=[neg_is_toggled_stove, p_s_equal_0],
        effects=[is_toggled_stove],
        goal_clause_pattern="""
(is-toggled {stove})
"""
    )
    sketch_dict["toggle stove"].append(rule_toggle_stove)
    
    # 2. slice lemon
    # 2.1 move toward the knife
    # * precons
    neg_H_knife = Holding(
        wanted_value=False,
        type_tag='knife',
    )
    n_sliced = Count(
        type_tag='lemon',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    p_knife_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='knife',
    )
    # * effects
    p_knife_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='knife',
    )
    rule_move_to_knife = Rule(
        preconditions=[neg_H_knife, p_knife_larger_0, n_sliced],
        effects=[p_knife_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {knife})
"""
    )
    sketch_dict["slice lemon"].append(rule_move_to_knife)
    # 2.2 pick up the knife when reachable
    # * precons
    neg_H_knife = Holding(
        wanted_value=False,
        type_tag='knife',
    )
    n_sliced = Count(
        type_tag='lemon',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    p_knife_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='knife',
    )
    # * effects
    H_knife = Holding(
        wanted_value=True,
        type_tag='knife',
    )
    
    rule_pick_up_knife = Rule(
        preconditions=[neg_H_knife, p_knife_equal_0, n_sliced],
        effects=[H_knife],
        goal_clause_pattern="""
(inhandofrobot agent-01 {knife})
"""
    )
    sketch_dict["slice lemon"].append(rule_pick_up_knife)
    # 2.3 move toward the nearest lemon
    # * precons
    H_knife = Holding(
        wanted_value=True,
        type_tag='knife',
    )
    n_sliced = Count(
        type_tag='lemon',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    p_lemon_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='lemon',
    )
    # * effects
    p_lemon_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='lemon',
    )
    rule_move_to_lemon = Rule(
        preconditions=[H_knife, p_lemon_larger_0, n_sliced],
        effects=[p_lemon_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {lemon})
"""
    )
    sketch_dict["slice lemon"].append(rule_move_to_lemon)
    # 2.4 slice the lemon when reachable
    # * precons
    H_knife = Holding(
        wanted_value=True,
        type_tag='knife',
    )
    n_sliced = Count(
        type_tag='lemon',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    p_lemon_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='lemon',
    )
    # * effects
    n_sliced_decrease = Count(
        type_tag='lemon',
        wanted_value='decrease',
        count_func=count_not_sliced,
    )
    rule_slice_lemon = Rule(
        preconditions=[H_knife, p_lemon_equal_0, n_sliced],
        effects=[n_sliced_decrease],
        goal_clause_pattern="""
(is-sliced {lemon})
"""
    )
    sketch_dict["slice lemon"].append(rule_slice_lemon)
    # 2.5 put down the knife
    # * precons
    H_knife = Holding(
        wanted_value=True,
        type_tag='knife',
    )
    n_sliced_0 = Count(
        type_tag='lemon',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    # * effects
    neg_H_knife = Holding(
        wanted_value=False,
        type_tag='knife',
    )
    rule_put_down_knife = Rule(
        preconditions=[H_knife, n_sliced_0],
        effects=[neg_H_knife],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {knife}))
"""
    )
    sketch_dict["slice lemon"].append(rule_put_down_knife)
    # 3. put teapot on stove and put tea bag in teapot
    # 3.1 move toward the nearest teapot
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    
    neg_H_teapot = Holding(
        wanted_value=False,
        type_tag='teapot',
    )
    p_teapot_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='teapot',
    )
    # * effects
    p_teapot_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='teapot',
    )
    rule_move_to_teapot = Rule(
        preconditions=[n_not_ontop_teapot_stove, neg_H_teapot, p_teapot_larger_0],
        effects=[p_teapot_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {teapot})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_move_to_teapot)
    # 3.2 pick up the teapot when reachable
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    neg_H_teapot = Holding(
        wanted_value=False,
        type_tag='teapot',
    )
    p_teapot_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='teapot',
    )
    # * effects
    H_teapot = Holding(
        wanted_value=True,
        type_tag='teapot',
    )
    rule_pick_up_teapot = Rule(
        preconditions=[n_not_ontop_teapot_stove, neg_H_teapot, p_teapot_equal_0],
        effects=[H_teapot],
        goal_clause_pattern="""
(inhandofrobot agent-01 {teapot})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_pick_up_teapot)
    # 3.3 move toward the stove when holding a teapot
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    H_teapot = Holding(
        wanted_value=True,
        type_tag='teapot',
    )
    p_s_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='stove',
    )
    # * effects
    p_s_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='stove',
    )
    rule_move_to_stove_holding_teapot = Rule(
        preconditions=[n_not_ontop_teapot_stove, H_teapot, p_s_larger_0],
        effects=[p_s_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {stove})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_move_to_stove_holding_teapot)
    # 3.4 put the teapot on the stove when reachable
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    H_teapot = Holding(
        wanted_value=True,
        type_tag='teapot',
    )
    p_s_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='stove',
    )
    # * effects
    n_not_ontop_teapot_stove_decrease = Count(
        type_tag='teapot',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    neg_H_teapot = Holding(
        wanted_value=False,
        type_tag='teapot',
    )
    rule_put_teapot_on_stove = Rule(
        preconditions=[n_not_ontop_teapot_stove, H_teapot, p_s_equal_0],
        effects=[n_not_ontop_teapot_stove_decrease, neg_H_teapot],
        goal_clause_pattern="""
(onTop {teapot} {stove})
(not (inhandofrobot agent-01 {teapot}))
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_put_teapot_on_stove)
    # 3.5 move toward the nearest soaked tea bag
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    neg_H_tea_bag = Holding(
        wanted_value=False,
        type_tag='tea_bag',
    )
    
    n_not_same_location_tea_bag_teapot = Count(
        type_tag='tea_bag',
        wanted_value='>0',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    p_tea_bag_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='tea_bag',
    )
    
    # * effects
    p_tea_bag_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='tea_bag',
    )
    rule_move_to_soaked_tea_bag = Rule(
        preconditions=[n_not_ontop_teapot_stove, neg_H_tea_bag, n_not_same_location_tea_bag_teapot, p_tea_bag_larger_0],
        effects=[p_tea_bag_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {tea_bag})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_move_to_soaked_tea_bag)
    # 3.6 pick up the soaked tea bag when reachable
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    neg_H_tea_bag = Holding(
        wanted_value=False,
        type_tag='tea_bag',
    )
    
    n_not_same_location_tea_bag_teapot = Count(
        type_tag='tea_bag',
        wanted_value='>0',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    p_tea_bag_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='tea_bag',
    )
    # * effects
    H_tea_bag = Holding(
        wanted_value=True,
        type_tag='tea_bag',
    )
    rule_pick_up_soaked_tea_bag = Rule(
        preconditions=[n_not_ontop_teapot_stove, neg_H_tea_bag, n_not_same_location_tea_bag_teapot, p_tea_bag_equal_0],
        effects=[H_tea_bag],
        goal_clause_pattern="""
(inhandofrobot agent-01 {tea_bag})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_pick_up_soaked_tea_bag)
    # 3.7 move toward the teapot
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
 
    
    n_not_same_location_tea_bag_teapot = Count(
        type_tag='tea_bag',
        wanted_value='>0',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    H_tea_bag = Holding(
        wanted_value=True,
        type_tag='tea_bag',
    )
    p_teapot_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='teapot',
    )
    # * effects
    p_teapot_down = DistanceToNearest(
        wanted_value="decrease",
        type_tag='teapot',
    )
    rule_move_to_teapot_holding_tea_bag = Rule(
        preconditions=[n_not_ontop_teapot_stove, n_not_same_location_tea_bag_teapot, H_tea_bag, p_teapot_larger_0],
        effects=[p_teapot_down],
        goal_clause_pattern="""
(inreachofrobot agent-01 {teapot})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_move_to_teapot_holding_tea_bag)
    # 3.8 put the tea bag in the teapot when reachable
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    
    n_not_same_location_tea_bag_teapot = Count(
        type_tag='tea_bag',
        wanted_value='>0',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    H_tea_bag = Holding(
        wanted_value=True,
        type_tag='tea_bag',
    )
    p_teapot_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='teapot',
    )
    
    neg_isSoaked_tea_bag = isSoaked(
        wanted_value=False,
        type_tag='tea_bag',
    )
    
    # * effects
    n_not_same_location_tea_bag_teapot_decrease = Count(
        type_tag='tea_bag',
        wanted_value='decrease',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    
    isSoaked_tea_bag = isSoaked(
        wanted_value=True,
        type_tag='tea_bag',
    )
    
    neg_H_tea_bag = Holding(
        wanted_value=False,
        type_tag='tea_bag',
    )
    rule_put_tea_bag_in_teapot = Rule(
        preconditions=[n_not_ontop_teapot_stove, n_not_same_location_tea_bag_teapot, H_tea_bag, p_teapot_equal_0, neg_isSoaked_tea_bag],
        effects=[n_not_same_location_tea_bag_teapot_decrease, neg_H_tea_bag, isSoaked_tea_bag],
        goal_clause_pattern="""
(atsamelocation {tea_bag} {teapot})
(not (inhandofrobot agent-01 {tea_bag}))
(is-soaked {tea_bag})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_put_tea_bag_in_teapot)
    
    goal_count_feature = [
        Count(
            type_tag='lemon',
            wanted_value='=0',
            count_func=count_not_sliced,
        ),
        Count(
            type_tag='tea_bag',
            wanted_value='=0',
            count_func=count_not_soaked,
        ),
        Count(
            type_tag='teapot',
            wanted_value='=0',
            count_func=partial(count_not_ontop_sometype, surface_type='stove')
        ),
        Count(
            type_tag='tea_bag',
            wanted_value='=0',
            count_func=partial(count_not_samelocation_sometype, target_type_name='teapot', additional_predicate_on_object_type='is-soaked')
        ),
        Count(
            type_tag='stove',
            wanted_value='=0',
            count_func=count_not_toggled,
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
# **Toggle stove**
# $\{\neg is\_toggled(stove)\} \mapsto \{is\_toggled(stove)\}$  ; toggle the stove on (width 1)
# **Slice lemon**
# $\{\neg H_{knife}\} \mapsto \{H_{knife}\}$  ; pick up the knife (width 1)
# $\{H_{knife}, \neg \text{is\_sliced(lemon)}\} \mapsto \{\text{is\_sliced(lemon)}, \neg H_{knife}\}$  ; slice a lemon (width 1)
# **Put teapot on stove and put tea bag in teapot**
# $\{\neg H_t\} \mapsto \{H_t\}$  ; pick up a teapot (width 1)
# $\{H_t\} \mapsto \{onTop(teapot, stove), \neg H_t\}$  ; put the teapot on the stove (width 1)
# $\{\neg H_{tb}, is\_soaked(tea\_bag)\} \mapsto \{H_{tb}\}$  ; pick up a soaked tea bag (width 1)
# $\{H_{tb}, onTop(teapot, stove)\} \mapsto \{atsamelocation(tea\_bag, teapot), \neg H_{tb}\}$  ; put the tea bag in the teapot (width 1)

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        "toggle stove": [],
        "slice lemon": [],
        "put teapot on stove and put tea bag in teapot": []
    }
    # 1. toggle stove
    # * precons
    c_not_toggled_stove = Count(
        type_tag='stove',
        wanted_value='>0',
        count_func=count_not_toggled,
    )
    # * effects
    c_not_toggled_stove_decrease = Count(
        type_tag='stove',
        wanted_value='decrease',
        count_func=count_not_toggled,
    )
    rule_toggle_stove = Rule(
        preconditions=[c_not_toggled_stove],
        effects=[c_not_toggled_stove_decrease],
        goal_clause_pattern="""
(is-toggled {stove})
"""
    )
    sketch_dict["toggle stove"].append(rule_toggle_stove)
    
    # 2. slice lemon
    # 2.1 pick up the knife 
    # * precons
    neg_H_knife = Holding(
        wanted_value=False,
        type_tag='knife',
    )
    n_sliced = Count(
        type_tag='lemon',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    # * effects
    H_knife = Holding(
        wanted_value=True,
        type_tag='knife',
    )
    rule_pick_up_knife = Rule(
        preconditions=[neg_H_knife, n_sliced],
        effects=[H_knife],
        goal_clause_pattern="""
(inhandofrobot agent-01 {knife})
"""
    )
    sketch_dict["slice lemon"].append(rule_pick_up_knife)
    # 2.2 slice a lemon
    # * precons
    H_knife = Holding(
        wanted_value=True,
        type_tag='knife',
    )
    n_sliced = Count(
        type_tag='lemon',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    # * effects
    n_sliced_decrease = Count(
        type_tag='lemon',
        wanted_value='decrease',
        count_func=count_not_sliced,
    )
    rule_slice_lemon = Rule(
        preconditions=[H_knife, n_sliced],
        effects=[n_sliced_decrease],
        goal_clause_pattern="""
(is-sliced {lemon})
"""
    )
    sketch_dict["slice lemon"].append(rule_slice_lemon)
    # 2.3 put down the knife
    # * precons
    H_knife = Holding(
        wanted_value=True,
        type_tag='knife',
    )
    n_sliced_0 = Count(
        type_tag='lemon',
        wanted_value='=0',
        count_func=count_not_sliced,
    )
    # * effects
    neg_H_knife = Holding(
        wanted_value=False,
        type_tag='knife',
    )
    rule_put_down_knife = Rule(
        preconditions=[H_knife, n_sliced_0],
        effects=[neg_H_knife],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {knife}))
"""
    )
    sketch_dict["slice lemon"].append(rule_put_down_knife)
    
    # 3. put teapot on stove and put tea bag in teapot
    # 3.1 pick up a teapot
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    neg_H_teapot = Holding(
        wanted_value=False,
        type_tag='teapot',
    )
    # * effects
    H_teapot = Holding(
        wanted_value=True,
        type_tag='teapot',
    )
    rule_pick_up_teapot = Rule(
        preconditions=[n_not_ontop_teapot_stove, neg_H_teapot],
        effects=[H_teapot],
        goal_clause_pattern="""
(inhandofrobot agent-01 {teapot})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_pick_up_teapot)
    # 3.2 put the teapot on the stove
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    H_teapot = Holding(
        wanted_value=True,
        type_tag='teapot',
    )
    # * effects
    n_not_ontop_teapot_stove_decrease = Count(
        type_tag='teapot',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    neg_H_teapot = Holding(
        wanted_value=False,
        type_tag='teapot',
    )
    rule_put_teapot_on_stove = Rule(
        preconditions=[n_not_ontop_teapot_stove, H_teapot],
        effects=[n_not_ontop_teapot_stove_decrease, neg_H_teapot],
        goal_clause_pattern="""
(onTop {teapot} {stove})
(not (inhandofrobot agent-01 {teapot}))
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_put_teapot_on_stove)
    # 3.3 pick up a soaked tea bag
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    neg_H_tea_bag = Holding(
        wanted_value=False,
        type_tag='tea_bag',
    )
    n_not_same_location_tea_bag_teapot = Count(
        type_tag='tea_bag',
        wanted_value='>0',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    # * effects
    H_tea_bag = Holding(
        wanted_value=True,
        type_tag='tea_bag',
    )
    rule_pick_up_soaked_tea_bag = Rule(
        preconditions=[n_not_ontop_teapot_stove, neg_H_tea_bag, n_not_same_location_tea_bag_teapot],
        effects=[H_tea_bag],
        goal_clause_pattern="""
(inhandofrobot agent-01 {tea_bag})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_pick_up_soaked_tea_bag)
    # 3.4 put the tea bag in the teapot
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
 
    
    n_not_same_location_tea_bag_teapot = Count(
        type_tag='tea_bag',
        wanted_value='>0',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    H_tea_bag = Holding(
        wanted_value=True,
        type_tag='tea_bag',
    )
    neg_isSoaked_tea_bag = isSoaked(
        wanted_value=False,
        type_tag='tea_bag',
    )
    # * effects
    n_not_same_location_tea_bag_teapot_decrease = Count(
        type_tag='tea_bag',
        wanted_value='decrease',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    
    isSoaked_tea_bag = isSoaked(
        wanted_value=True,
        type_tag='tea_bag',
    )
    
    neg_H_tea_bag = Holding(
        wanted_value=False,
        type_tag='tea_bag',
    )
    rule_put_tea_bag_in_teapot = Rule(
        preconditions=[n_not_ontop_teapot_stove, n_not_same_location_tea_bag_teapot, H_tea_bag, neg_isSoaked_tea_bag],
        effects=[n_not_same_location_tea_bag_teapot_decrease, neg_H_tea_bag, isSoaked_tea_bag],
        goal_clause_pattern="""
(atsamelocation {tea_bag} {teapot})
(not (inhandofrobot agent-01 {tea_bag}))
(is-soaked {tea_bag})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_put_tea_bag_in_teapot)
    
    goal_count_feature = [
        Count(
            type_tag='lemon',
            wanted_value='=0',
            count_func=count_not_sliced,
        ),
        Count(
            type_tag='tea_bag',
            wanted_value='=0',
            count_func=count_not_soaked,
        ),
        Count(
            type_tag='teapot',
            wanted_value='=0',
            count_func=partial(count_not_ontop_sometype, surface_type='stove')
        ),
        Count(
            type_tag='tea_bag',
            wanted_value='=0',
            count_func=partial(count_not_samelocation_sometype, target_type_name='teapot', additional_predicate_on_object_type='is-soaked')
        ),
        Count(
            type_tag='stove',
            wanted_value='=0',
            count_func=count_not_toggled,
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

### Sketch with W = 2 ; factor out toggling stove and decompose soaking tea bag
# **Toggle stove**
# $\{\neg is\_toggled(stove)\} \mapsto \{is\_toggled(stove)\}$  ; toggle the stove on (width 1)
# **Slice lemon**
# ${\neg \text{is\_sliced(lemon)}} \mapsto \{\text{is\_sliced(lemon)}\}$  ; slice a lemon (width 2)
# **Put teapot on stove and put tea bag in teapot**
# ```lisp
# (exists (?t - teapot ?tb - tea_bag ?s - stove)
#       (and
#           (onTop ?t ?s)
#           (atsamelocation ?tb ?t)
#       )
# )
# ```
# ; put a teapot on the stove, and put a tea bag in the teapot (width 2)
# $\{\neg onTop(teapot, stove)\} \mapsto \{onTop(teapot, stove)\}$  ; put the teapot on the stove
# $\{onTop(teapot, stove) \neg atsamelocation(tea\_bag, teapot)\} \mapsto \{atsamelocation(tea\_bag, teapot)}$  ; put the tea bag in the teapot (width 1)

def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        "toggle stove": [],
        "slice lemon": [],
        "put teapot on stove and put tea bag in teapot": []
    }
    # 1. toggle stove
    # * precons
    c_not_toggled_stove = Count(
        type_tag='stove',
        wanted_value='>0',
        count_func=count_not_toggled,
    )
    # * effects
    c_not_toggled_stove_decrease = Count(
        type_tag='stove',
        wanted_value='decrease',
        count_func=count_not_toggled,
    )
    rule_toggle_stove = Rule(
        preconditions=[c_not_toggled_stove],
        effects=[c_not_toggled_stove_decrease],
        goal_clause_pattern="""
(is-toggled {stove})
"""
    )
    sketch_dict["toggle stove"].append(rule_toggle_stove)
    # 2. slice lemon
    # * precons
    n_sliced = Count(
        type_tag='lemon',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    # * effects
    n_sliced_decrease = Count(
        type_tag='lemon',
        wanted_value='decrease',
        count_func=count_not_sliced,
    )
    rule_slice_lemon = Rule(
        preconditions=[n_sliced],
        effects=[n_sliced_decrease],
        goal_clause_pattern="""
(is-sliced {lemon})
"""
    )
    sketch_dict["slice lemon"].append(rule_slice_lemon)
    # 3. put teapot on stove and put tea bag in teapot
    # 3.1 put the teapot on the stove
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    # * effects
    n_not_ontop_teapot_stove_decrease = Count(
        type_tag='teapot',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    rule_put_teapot_on_stove = Rule(
        preconditions=[n_not_ontop_teapot_stove],
        effects=[n_not_ontop_teapot_stove_decrease],
        goal_clause_pattern="""
(onTop {teapot} {stove})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_put_teapot_on_stove)
    # 3.2 put the tea bag in the teapot
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='=0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    n_not_same_location_tea_bag_teapot = Count(
        type_tag='tea_bag',
        wanted_value='>0',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    neg_isSoaked_tea_bag = isSoaked(
        wanted_value=False,
        type_tag='tea_bag',
    )
    # * effects
    n_not_same_location_tea_bag_teapot_decrease = Count(
        type_tag='tea_bag',
        wanted_value='decrease',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    
    isSoaked_tea_bag = isSoaked(
        wanted_value=True,
        type_tag='tea_bag',
    )
    rule_put_tea_bag_in_teapot = Rule(
        preconditions=[n_not_ontop_teapot_stove, n_not_same_location_tea_bag_teapot, neg_isSoaked_tea_bag],
        effects=[n_not_same_location_tea_bag_teapot_decrease, isSoaked_tea_bag],
        goal_clause_pattern="""
(atsamelocation {tea_bag} {teapot})
(is-soaked {tea_bag})
"""
    )
    sketch_dict["put teapot on stove and put tea bag in teapot"].append(rule_put_tea_bag_in_teapot)
    
    goal_count_feature = [
        Count(
            type_tag='lemon',
            wanted_value='=0',
            count_func=count_not_sliced,
        ),
        Count(
            type_tag='tea_bag',
            wanted_value='=0',
            count_func=count_not_soaked,
        ),
        Count(
            type_tag='teapot',
            wanted_value='=0',
            count_func=partial(count_not_ontop_sometype, surface_type='stove')
        ),
        Count(
            type_tag='tea_bag',
            wanted_value='=0',
            count_func=partial(count_not_samelocation_sometype, target_type_name='teapot', additional_predicate_on_object_type='is-soaked')
        ),
        Count(
            type_tag='stove',
            wanted_value='=0',
            count_func=count_not_toggled,
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

### Sketch with W = 3 ; factor out soaking tea bag
# $\{\neg \text{is\_sliced(lemon)}\} \mapsto \{\text{is\_sliced(lemon)}\}$  ; slice a lemon (width 2)
# ```lisp
# (exists (?t - teapot ?tb - tea_bag ?s - stove)
#     (and
#         (onTop ?t ?s)
#         (atsamelocation ?tb ?t)
#         (is-soaked ?tb)
#         (is-toggled ?s)
#     )
# )
# ```  
# ; put a teapot on the stove, put a tea bag in the teapot, and toggle the stove on (width 3)

def create_width_3_sketch(env, map_id, domain_name, window=None):
    width = 3
    sketch_dict = {
        "slice lemon": [],
        "put teapot on toggled-on stove and put tea bag in teapot": []
    }
    # 1. slice lemon
    # * precons
    n_sliced = Count(
        type_tag='lemon',
        wanted_value='>0',
        count_func=count_not_sliced,
    )
    # * effects
    n_sliced_decrease = Count(
        type_tag='lemon',
        wanted_value='decrease',
        count_func=count_not_sliced,
    )
    rule_slice_lemon = Rule(
        preconditions=[n_sliced],
        effects=[n_sliced_decrease],
        goal_clause_pattern="""
(is-sliced {lemon})
"""
    )
    sketch_dict["slice lemon"].append(rule_slice_lemon)
    # 2. put teapot on toggled-on stove and put tea bag in teapot
    # * precons
    n_not_ontop_teapot_stove = Count(
        type_tag='teapot',
        wanted_value='>0',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    n_not_same_location_tea_bag_teapot = Count(
        type_tag='tea_bag',
        wanted_value='>0',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    neg_isSoaked_tea_bag = isSoaked(
        wanted_value=False,
        type_tag='tea_bag',
    )
    neg_toggled_stove = isToggled(
        wanted_value=False,
        type_tag='stove',
    )
    # * effects
    n_not_ontop_teapot_stove_decrease = Count(
        type_tag='teapot',
        wanted_value='decrease',
        count_func=partial(count_not_ontop_sometype, surface_type='stove')
    )
    n_not_same_location_tea_bag_teapot_decrease = Count(
        type_tag='tea_bag',
        wanted_value='decrease',
        count_func=partial(count_not_samelocation_sometype, target_type_name='teapot')
    )
    isSoaked_tea_bag = isSoaked(
        wanted_value=True,
        type_tag='tea_bag',
    )
    toggled_stove = isToggled(
        wanted_value=True,
        type_tag='stove',
    )
    rule_put_teapot_on_stove_and_put_tea_bag_in_teapot_and_toggle_stove = Rule(
        preconditions=[n_not_ontop_teapot_stove, n_not_same_location_tea_bag_teapot, neg_isSoaked_tea_bag, neg_toggled_stove],
        effects=[n_not_ontop_teapot_stove_decrease, n_not_same_location_tea_bag_teapot_decrease, isSoaked_tea_bag, toggled_stove],
        goal_clause_pattern="""
(onTop {teapot} {stove})
(atsamelocation {tea_bag} {teapot})
(is-soaked {tea_bag})
(is-toggled {stove})
"""
    )
    sketch_dict["put teapot on toggled-on stove and put tea bag in teapot"].append(rule_put_teapot_on_stove_and_put_tea_bag_in_teapot_and_toggle_stove)
    
    goal_count_feature = [
        Count(
            type_tag='lemon',
            wanted_value='=0',
            count_func=count_not_sliced,
        ),
        Count(
            type_tag='tea_bag',
            wanted_value='=0',
            count_func=count_not_soaked,
        ),
        Count(
            type_tag='teapot',
            wanted_value='=0',
            count_func=partial(count_not_ontop_sometype, surface_type='stove')
        ),
        Count(
            type_tag='tea_bag',
            wanted_value='=0',
            count_func=partial(count_not_samelocation_sometype, target_type_name='teapot', additional_predicate_on_object_type='is-soaked')
        ),
        Count(
            type_tag='stove',
            wanted_value='=0',
            count_func=count_not_toggled,
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/making_tea.py --map_id 18581263 --domain_name making_tea --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/making_tea/p18581263-making_tea_plans.json', want_render=False)
    
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