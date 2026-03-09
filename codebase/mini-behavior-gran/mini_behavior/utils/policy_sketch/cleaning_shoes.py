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
12
### Feature set (mirroring the plywood features)
- $H_r$: “holding a rag”
- $p_r$: distance to the nearest rag
- $p_s$: distance to the sink
- $p_f$: distance to the nearest focused shoe
- $N_f$: number of focused shoes not cleaned
- $is\_{toggled}$: whether the sink is toggled on
- $is\_{soaked}$: whether the rag is soaked



### Sketch with W = 0 (approximate, including distance features)

**Pick up rag and soak it**
$\{\neg H_r, N_f > 0, p_r > 0\} \mapsto \{p_r \downarrow, p_f ?, p_s ?\}$  ; move toward the nearest rag
$\{\neg H_r, N_f > 0, p_r = 0\} \mapsto \{H_r, p_f ?, p_s ?\}$  ; pick up the rag when reachable
$\{H_r, \neg is\_toggled, p_s > 0\} \mapsto \{p_s \downarrow, p_f ?, p_r ?\}$  ; move toward the sink
$\{H_r, \neg is\_toggled, p_s = 0\} \mapsto \{is\_toggled, p_f ?, p_r ?\}$  ; toggle the sink on when reachable
$\{H_r, is\_toggled, \neg is\_soaked, p_s >0 \} \mapsto \{p_s \downarrow\}$  ; move toward the sink when holding the rag
$\{H_r, is\_toggled, \neg is\_soaked, p_s =0 \} \mapsto \{is\_soaked, \neg H_r, p_f ?, p_r = 0\}$  ; soak the rag when at the sink

**Clean task**
$\{\neg H_r, is\_soaked, N_f > 0, p_r > 0\} \mapsto \{p_r \downarrow \}$  ; move toward the nearest soaked rag
$\{\neg H_r, is\_soaked, N_f > 0, p_r = 0\} \mapsto \{H_r, p_f ? \}$  ; pick up a soaked rag when reachable
$\{H_r, is\_soaked, N_f > 0, p_f > 0\} \mapsto \{p_f \downarrow, p_r ?, p_s ?\}$  ; move toward the nearest focused shoe
$\{H_r, is\_soaked, N_f > 0, p_f = 0\} \mapsto \{N_f \downarrow, \neg H_r, p_r ?, p_s ?\}$  ; clean the focused shoe when reachable

**put towel on floor at the end**
$\{N_f = 0, \neg onfloor, H_r\} \mapsto \{\neg H_r, onfloor\}$  ; put the towel on the floor when holding it
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0 
    sketch_dict = {
        "pick up rag and soak it": [],
        "clean shoes": [],
        "put towel on floor at the end": []
    }
    # 1. pick up rag and soak it
    # 1.1 move toward the nearest rag
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    p_r_larger_0 = DistanceToNearest(
        type_tag = 'rag',
        wanted_value = '>0',
    )
    # * effects
    p_r_decrease = DistanceToNearest(
        type_tag = 'rag',
        wanted_value = 'decrease',
    )
    rule_move_towards_rag = Rule(
        preconditions = [neg_H_r, N_f_larger_0, p_r_larger_0],
        effects = [p_r_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {rag})
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_move_towards_rag)
    
    # 1.2 pick up the rag when reachable
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    p_r_equal_0 = DistanceToNearest(
        type_tag = 'rag',
        wanted_value = '=0',
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag = Rule(
        preconditions = [neg_H_r, N_f_larger_0, p_r_equal_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_pick_up_rag)
    # 1.3 move toward the sink to toggle it
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    neg_is_toggled_sink = isToggled(
        wanted_value=False,
        type_tag='sink',
    )
    p_s_larger_0 = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = '>0',
    )
    # * effects
    p_s_decrease = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = 'decrease',
    )
    rule_move_towards_sink = Rule(
        preconditions = [H_r, neg_is_toggled_sink, p_s_larger_0],
        effects = [p_s_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_move_towards_sink)
    # 1.4 toggle the sink on when reachable
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    neg_is_toggled_sink = isToggled(
        wanted_value=False,
        type_tag='sink',
    )
    p_s_equal_0 = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = '=0',
    )
    # * effects
    is_toggled_sink = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    rule_toggle_sink = Rule(
        preconditions = [H_r, neg_is_toggled_sink, p_s_equal_0],
        effects = [is_toggled_sink],
        goal_clause_pattern="""
(is-toggled {sink})
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_toggle_sink)
    # 1.5 move toward the sink when holding the rag
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    is_toggled_sink = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    neg_is_soaked = isSoaked(
        wanted_value=False,
        type_tag='rag',
    )
    p_s_larger_0 = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = '>0',
    )
    # * effects
    p_s_decrease = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = 'decrease',
    )
    rule_move_towards_sink = Rule(
        preconditions = [H_r, is_toggled_sink, neg_is_soaked, p_s_larger_0],
        effects = [p_s_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {sink})
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_move_towards_sink)
    # 1.6 soak the rag when at the sink
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    is_toggled_sink = isToggled(
        wanted_value=True,
        type_tag='sink',
    )
    neg_is_soaked = isSoaked(
        wanted_value=False,
        type_tag='rag',
    )
    p_s_equal_0 = DistanceToNearest(
        type_tag = 'sink',
        wanted_value = '=0',
    )
    # * effects
    is_soaked = isSoaked(
        wanted_value=True,
        type_tag='rag',
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    
    rule_soak_rag = Rule(
        preconditions = [H_r, is_toggled_sink, neg_is_soaked, p_s_equal_0],
        effects = [is_soaked, neg_H_r],
        goal_clause_pattern="""
(is-soaked {rag})
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_soak_rag)
    
    # 2. clean shoes
    # 2.1 pick up a soaked rag when reachable
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    is_soaked = isSoaked(
        wanted_value=True,
        type_tag='rag',
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    p_r_larger_0 = DistanceToNearest(
        type_tag = 'rag',
        wanted_value = '>0',
    )
    # * effects
    p_r_decrease = DistanceToNearest(
        type_tag = 'rag',
        wanted_value = 'decrease',
    )
    rule_move_towards_rag = Rule(
        preconditions = [neg_H_r, is_soaked, N_f_larger_0, p_r_larger_0],
        effects = [p_r_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {rag})
"""
    )
    sketch_dict["clean shoes"].append(rule_move_towards_rag)
    
    # 2.2 pick up a soaked rag when reachable
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    is_soaked = isSoaked(
        wanted_value=True,
        type_tag='rag',
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    p_r_equal_0 = DistanceToNearest(
        type_tag = 'rag',
        wanted_value = '=0',
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag = Rule(
        preconditions = [neg_H_r, is_soaked, N_f_larger_0, p_r_equal_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    
    sketch_dict["clean shoes"].append(rule_pick_up_rag)
    
    # 2.3 move toward the nearest focused shoe
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    is_soaked = isSoaked(
        wanted_value=True,
        type_tag='rag',
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    p_f_larger_0 = DistanceToNearest(
        type_tag = 'shoe',
        wanted_value = '>0',
    )
    # * effects
    p_f_decrease = DistanceToNearest(
        type_tag = 'shoe',
        wanted_value = 'decrease',
    )
    rule_move_towards_shoe = Rule(
        preconditions = [H_r, is_soaked, N_f_larger_0, p_f_larger_0],
        effects = [p_f_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {shoe})
"""
    )
    sketch_dict["clean shoes"].append(rule_move_towards_shoe)
    # 2.4 clean the focused shoe when reachable
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    is_soaked = isSoaked(
        wanted_value=True,
        type_tag='rag',
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    p_f_equal_0 = DistanceToNearest(
        type_tag = 'shoe',
        wanted_value = '=0',
    )
    # * effects
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_f_decrease = Count(
        type_tag='shoe',
        wanted_value='decrease',
        count_func=count_not_cleaned,
    )
    rule_clean_shoe = Rule(
        preconditions = [H_r, is_soaked, N_f_larger_0, p_f_equal_0],
        effects = [N_f_decrease, neg_H_r],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {rag}))
(is-not-stained {shoe})
"""
    )
    sketch_dict["clean shoes"].append(rule_clean_shoe)
    
    # 3. put towel on floor at the end
    N_f_equal_0 = Count(
        type_tag='shoe',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )

    C_r_not_onfloor = Count(
        type_tag='towel',
        wanted_value='>0',
        count_func=count_not_onfloor,
    )
    # * effects
    C_r_not_onfloor_decrease = Count(
        type_tag='towel',
        wanted_value='decrease',
        count_func=count_not_onfloor,
    )
    
    rule_put_rag_on_floor = Rule(
        preconditions = [N_f_equal_0, C_r_not_onfloor],
        effects = [C_r_not_onfloor_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {towel}))
"""
    )
    sketch_dict["put towel on floor at the end"].append(rule_put_rag_on_floor)
    
    goal_count_feature = []
    goal_count_feature.append(
        Count(
            type_tag='shoe',
            wanted_value='=0',
            count_func=count_not_cleaned,
        )
    )
    goal_count_feature.append(
        Count(
            type_tag='towel',
            wanted_value='=0',
            count_func=count_not_onfloor,
        )
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

### Sketch with W = 1 ; separate carrying and cleaning

# **Pick up rag and soak it**
# $\{\neg H_r, \neg is\_soaked\} \mapsto \{H_r\}$  ; pick up a rag (width 1)
# $\{\neg is\_toggled\} \mapsto \{is\_toggled\}$  ; toggle the sink on (width 1)
# $\{H_r, is\_toggled, \neg is\_soaked\} \mapsto \{is\_soaked, \neg H_r\}$  ; soak the rag (width 1)
# **Clean task**
# $\{\neg H_r, is\_soaked, N_f > 0\} \mapsto \{H_r\}$  ; pick up a soaked rag (width 1)
# $\{H_r, is\_soaked, N_f > 0\} \mapsto \{N_f \downarrow, \neg H_r\}$  ; clean a focused shoe (width 1)
# **put towel on floor at the end**
# $\{N_f = 0, \neg onfloor\} \mapsto \{onfloor\}$  ; put a towel on the floor (width 1)

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1 
    sketch_dict = {
        "pick up rag and soak it": [],
        "clean shoes": [],
        "put towel on floor at the end": []
    }
    # 1. pick up rag and soak it
    # 1.1 pick up a rag (width 1)
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag = Rule(
        preconditions = [neg_H_r, N_f_larger_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_pick_up_rag)
    # 1.2 toggle the sink on (width 1)
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    C_sink_not_toggled = Count(
        type_tag='sink',
        wanted_value='>0',
        count_func=count_not_toggled,
    )
    # * effects
    C_sink_not_toggled_decrease = Count(
        type_tag='sink',
        wanted_value='decrease',
        count_func=count_not_toggled,
    )
    rule_toggle_sink = Rule(
        preconditions = [C_sink_not_toggled, H_r],
        effects = [C_sink_not_toggled_decrease],
        goal_clause_pattern="""
(is-toggled {sink})
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_toggle_sink)
    # 1.3 soak the rag (width 1)
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    C_sink_not_toggled_zero = Count(
        type_tag='sink',
        wanted_value='=0',
        count_func=count_not_toggled,
    )
    C_rag_not_soaked = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    # * effects
    C_rag_not_soaked_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=count_not_soaked,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_soak_rag = Rule(
        preconditions = [H_r, C_sink_not_toggled_zero, C_rag_not_soaked],
        effects = [C_rag_not_soaked_decrease, neg_H_r],
        goal_clause_pattern="""
(is-soaked {rag})
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_soak_rag)
    
    # 2. clean shoes
    # 2.1 pick up a soaked rag (width 1)
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    C_rag_not_soaked = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=count_not_soaked,
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag = Rule(
        preconditions = [neg_H_r, C_rag_not_soaked, N_f_larger_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["clean shoes"].append(rule_pick_up_rag)
    # 2.2 clean a focused shoe (width 1)
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    C_rag_not_soaked = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=count_not_soaked,
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    N_f_decrease = Count(
        type_tag='shoe',
        wanted_value='decrease',
        count_func=count_not_cleaned,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_clean_shoe = Rule(
        preconditions = [H_r, C_rag_not_soaked, N_f_larger_0],
        effects = [N_f_decrease, neg_H_r],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {rag}))
(is-not-stained {shoe})
"""
    )
    sketch_dict["clean shoes"].append(rule_clean_shoe)  
    # 3. put towel on floor at the end
    N_f_equal_0 = Count(
        type_tag='shoe',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
  
    C_r_not_onfloor = Count(
        type_tag='towel',
        wanted_value='>0',
        count_func=count_not_onfloor,
    )
    # * effects
    C_r_not_onfloor_decrease = Count(
        type_tag='towel',
        wanted_value='decrease',
        count_func=count_not_onfloor,
    )
    rule_put_rag_on_floor = Rule(
        preconditions = [N_f_equal_0, C_r_not_onfloor],
        effects = [C_r_not_onfloor_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {towel}))
"""
    )
    sketch_dict["put towel on floor at the end"].append(rule_put_rag_on_floor)
    goal_count_feature = []
    goal_count_feature.append(
        Count(
            type_tag='shoe',
            wanted_value='=0',
            count_func=count_not_cleaned,
        )
    )
    goal_count_feature.append(
        Count(
            type_tag='towel',
            wanted_value='=0',
            count_func=count_not_onfloor,
        )
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
    
### Sketch with W = 2 ; factor out cleaning shoe
# **Pick up rag and soak it**
# $\{\neg H_r, \neg is\_soaked\} \mapsto \{H_r\}$  ; pick up a rag (width 1)
# $\{H_r, \neg is\_soaked\} \mapsto \{is\_soaked, \neg H_r\}$  ; soak the rag (width 2)
# **Clean task**
# $\{is\_soaked, N_f > 0\} \mapsto \{N_f \downarrow\}$  ; clean a focused shoe (width 2)
# **put towel on floor at the end**
# $\{N_f = 0, \neg onfloor\} \mapsto \{onfloor\}$  ; put a towel on the floor (width 1)

def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2 
    sketch_dict = {
        "pick up rag and soak it": [],
        "clean shoes": [],
        "put towel on floor at the end": []
    }
    # 1. pick up rag and soak it
    # 1.1 pick up a rag (width 1)
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag = Rule(
        preconditions = [neg_H_r, N_f_larger_0],
        effects = [H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_pick_up_rag)
    # 1.2 soak the rag (width 2)
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    C_rag_not_soaked = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=count_not_soaked,
    )
    # * effects
    C_rag_not_soaked_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=count_not_soaked,
    )
    
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_soak_rag = Rule(
        preconditions = [H_r, C_rag_not_soaked],
        effects = [C_rag_not_soaked_decrease, neg_H_r],
        goal_clause_pattern="""
(is-soaked {rag})
(not (inhandofrobot agent-01 {rag}))
"""
    )
    sketch_dict["pick up rag and soak it"].append(rule_soak_rag)
    
    # 2. clean shoes
    # 2.1 clean a focused shoe (width 2)
    C_rag_not_soaked = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=count_not_soaked,
    )
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    N_f_decrease = Count(
        type_tag='shoe',
        wanted_value='decrease',
        count_func=count_not_cleaned,
    )
    rule_clean_shoe = Rule(
        preconditions = [C_rag_not_soaked, N_f_larger_0],
        effects = [N_f_decrease],
        goal_clause_pattern="""
(is-not-stained {shoe})
"""
    )
    sketch_dict["clean shoes"].append(rule_clean_shoe)  
    # 3. put towel on floor at the end
    N_f_equal_0 = Count(
        type_tag='shoe',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    C_r_not_onfloor = Count(
        type_tag='towel',
        wanted_value='>0',
        count_func=count_not_onfloor,
    )
    # * effects
    C_r_not_onfloor_decrease = Count(
        type_tag='towel',
        wanted_value='decrease',
        count_func=count_not_onfloor,
    )
    rule_put_rag_on_floor = Rule(
        preconditions = [N_f_equal_0, C_r_not_onfloor],
        effects = [C_r_not_onfloor_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {towel}))
"""
    )
    sketch_dict["put towel on floor at the end"].append(rule_put_rag_on_floor)
    goal_count_feature = []
    goal_count_feature.append(
        Count(
            type_tag='shoe',
            wanted_value='=0',
            count_func=count_not_cleaned,
        )
    )
    goal_count_feature.append(
        Count(
            type_tag='towel',
            wanted_value='=0',
            count_func=count_not_onfloor,
        )
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

### Sketch with W = 3 ; only $N_f$ matters
# $\{N_f > 0\} \mapsto \{N_f \downarrow\}$  ; clean a focused shoe (width 3)
# $\{N_f = 0, \neg onfloor\} \mapsto \{onfloor\}$  ; put a towel on the floor (width 1)

def create_width_3_sketch(env, map_id, domain_name, window=None):
    width = 3 
    sketch_dict = {
        "clean shoes": [],
        "put towel on floor at the end": []
    }
    
    # 2. clean shoes
    # 2.1 clean a focused shoe (width 3)
    N_f_larger_0 = Count(
        type_tag='shoe',
        wanted_value='>0',
        count_func=count_not_cleaned,
    )
    # * effects
    N_f_decrease = Count(
        type_tag='shoe',
        wanted_value='decrease',
        count_func=count_not_cleaned,
    )
    rule_clean_shoe = Rule(
        preconditions = [N_f_larger_0],
        effects = [N_f_decrease],
        goal_clause_pattern="""
(is-not-stained {shoe})
"""
    )
    sketch_dict["clean shoes"].append(rule_clean_shoe)  
    # 3. put towel on floor at the end
    N_f_equal_0 = Count(
        type_tag='shoe',
        wanted_value='=0',
        count_func=count_not_cleaned,
    )
    C_r_not_onfloor = Count(
        type_tag='towel',
        wanted_value='>0',
        count_func=count_not_onfloor,
    )
    # * effects
    C_r_not_onfloor_decrease = Count(
        type_tag='towel',
        wanted_value='decrease',
        count_func=count_not_onfloor,
    )
    rule_put_rag_on_floor = Rule(
        preconditions = [N_f_equal_0, C_r_not_onfloor],
        effects = [C_r_not_onfloor_decrease],
        goal_clause_pattern="""
(not (inhandofrobot agent-01 {towel}))
"""
    )
    sketch_dict["put towel on floor at the end"].append(rule_put_rag_on_floor)
    goal_count_feature = []
    goal_count_feature.append(
        Count(
            type_tag='shoe',
            wanted_value='=0',
            count_func=count_not_cleaned,
        )
    )
    goal_count_feature.append(
        Count(
            type_tag='towel',
            wanted_value='=0',
            count_func=count_not_onfloor,
        )
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/cleaning_shoes.py --map_id 26467857 --domain_name cleaning_shoes --width 1 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/cleaning_shoes/p26467857-cleaning_shoes_plans.json', want_render=False)
    
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