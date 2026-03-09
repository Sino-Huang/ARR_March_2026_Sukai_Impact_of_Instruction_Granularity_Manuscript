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
    count_not_sliced)
"""
16
### Feature set (mirroring the plywood features)
- $H_r$: “holding a rag”
- $H_s$: “holding a soap”
- $p_r$: distance to the nearest rag
- $p_s$: distance to the nearest soap
- $p_c$: distance to the nearest focused car
- $p_b$: distance to the bucket
- $N_c$: number of focused cars not cleaned
- $N_{in\_bucket}$: number of items (rag and soap) not in the bucket
- $inside\_rag$: whether the rag is in the bucket


### Sketch with W = 0 (approximate, including distance features)

**Clean task**
$\{\neg H_r, N_c > 0, p_r > 0\} \mapsto \{p_r \downarrow, p_c ?\}$  ; move toward the nearest rag
$\{\neg H_r, N_c > 0, p_r = 0\} \mapsto \{H_r, p_c ?\}$  ; pick up the rag when reachable
$\{H_r, N_c > 0, p_c > 0\} \mapsto \{p_c \downarrow, p_r ?\}$  ; move toward the nearest focused car
$\{H_r, N_c > 0, p_c = 0\} \mapsto \{N_c \downarrow, \neg H_r, p_r ?\}$  ; clean the focused car when reachable

**Put rag in the bucket**
$\{\neg inside\_rag, \neg H_r, N_c = 0, N_{in\_bucket} > 0, p_r > 0\} \mapsto \{p_r \downarrow, p_b ?\}$  ; move toward the nearest rag
$\{\neg inside\_rag, \neg H_r, N_c = 0, N_{in\_bucket} > 0, p_r = 0\} \mapsto \{H_r, p_b ?\}$  ; pick up the rag when reachable
$\{H_r, \neg inside\_rag, N_c = 0, N_{in\_bucket} > 0, p_b > 0\} \mapsto \{p_b \downarrow, p_r ?\}$  ; move toward the bucket
$\{H_r, \neg inside\_rag, N_c = 0, N_{in\_bucket} > 0, p_b = 0\} \mapsto \{inside\_rag, \neg H_r, N_{in\_bucket} \downarrow, p_r ?\}$  ; put the rag in the bucket when at the spot

**Put soap in the bucket**
$\{\neg inside\_rag, \neg H_s, N_c = 0, N_{in\_bucket} > 0, p_s > 0\} \mapsto \{p_s \downarrow, p_b ?\}$  ; move toward the nearest soap
$\{\neg inside\_rag, \neg H_s, N_c = 0, N_{in\_bucket} > 0, p_s = 0\} \mapsto \{H_s, p_b ?\}$  ; pick up the soap when reachable
$\{H_s, inside\_rag, N_c = 0, N_{in\_bucket} > 0, p_b > 0\} \mapsto \{p_b \downarrow, p_s ?\}$  ; move toward the bucket
$\{H_s, inside\_rag, N_c = 0, N_{in\_bucket} > 0, p_b = 0\} \mapsto \{\neg H_s, N_{in\_bucket} \downarrow, p_r ?, p_c ?\}$  ; put the soap in the bucket when at the spot
"""

def create_width_0_sketch(env, map_id, domain_name, window=None):
    width = 0
    sketch_dict = {
        'cleaning task': [],
        'putting rag in the bucket': [],
        'putting soap in the bucket': []
    }
    
    # 1. cleaning task
    ## 1.1 move toward the nearest rag
    # * precons
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_c_larger_0 = Count(
        type_tag='car',
        wanted_value='>0',
        count_func=count_not_wiped,
    )
    p_r_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='rag',
    )
    # * effects
    p_r_decrease = DistanceToNearest(
        wanted_value='decrease',
        type_tag='rag',
    )
    rule_move_to_rag = Rule(
        preconditions=[neg_H_r, N_c_larger_0, p_r_larger_0],
        effects=[p_r_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {rag})
"""
    )
    sketch_dict['cleaning task'].append(rule_move_to_rag)
    ## 1.2 pick up the rag when reachable
    # * precons
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_c_larger_0 = Count(
        type_tag='car',
        wanted_value='>0',
        count_func=count_not_wiped,
    )
    p_r_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='rag',
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag = Rule(
        preconditions=[neg_H_r, N_c_larger_0, p_r_equal_0],
        effects=[H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict['cleaning task'].append(rule_pick_up_rag)
    ## 1.3 move toward the nearest focused car
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    N_c_larger_0 = Count(
        type_tag='car',
        wanted_value='>0',
        count_func=count_not_wiped,
    )
    p_c_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='car',
    )
    # * effects
    p_c_decrease = DistanceToNearest(
        wanted_value='decrease',
        type_tag='car',
    )
    rule_move_to_car = Rule(
        preconditions=[H_r, N_c_larger_0, p_c_larger_0],
        effects=[p_c_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {car})
"""
    )
    sketch_dict['cleaning task'].append(rule_move_to_car)
    ## 1.4 clean the focused car when reachable
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    N_c_larger_0 = Count(
        type_tag='car',
        wanted_value='>0',
        count_func=count_not_wiped,
    )
    p_c_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='car',
    )
    # * effects
    N_c_decrease = Count(
        type_tag='car',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_clean_car = Rule(
        preconditions=[H_r, N_c_larger_0, p_c_equal_0],
        effects=[N_c_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-dusted {car})
"""
    )
    sketch_dict['cleaning task'].append(rule_clean_car)
    
    # 2. putting rag in the bucket
    ## 2.1 move toward the nearest rag
    # * precons

    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    p_r_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='rag',
    )
    # * effects
    p_r_decrease = DistanceToNearest(
        wanted_value='decrease',
        type_tag='rag',
    )
    rule_move_to_rag_2 = Rule(
        preconditions=[neg_H_r, N_c_equal_0, p_r_larger_0],
        effects=[p_r_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {rag})
"""
    )
    sketch_dict['putting rag in the bucket'].append(rule_move_to_rag_2)
    ## 2.2 pick up the rag when reachable
    # * precons
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    p_r_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='rag',
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag_2 = Rule(
        preconditions=[neg_H_r, N_c_equal_0, p_r_equal_0],
        effects=[H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict['putting rag in the bucket'].append(rule_pick_up_rag_2)
    ## 2.3 move toward the bucket
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    p_b_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='bucket',
    )
    # * effects
    p_b_decrease = DistanceToNearest(
        wanted_value='decrease',
        type_tag='bucket',
    )
    rule_move_to_bucket = Rule(
        preconditions=[H_r, N_c_equal_0, p_b_larger_0],
        effects=[p_b_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {bucket})
"""
    )
    sketch_dict['putting rag in the bucket'].append(rule_move_to_bucket)
    ## 2.4 put the rag in the bucket when at the spot
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    p_b_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='bucket',
    )
    N_in_bucket_larger_0 = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    # * effects
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_in_bucket_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    rule_put_rag_in_bucket = Rule(
        preconditions=[H_r, N_c_equal_0, p_b_equal_0, N_in_bucket_larger_0],
        effects=[neg_H_r, N_in_bucket_decrease],
        goal_clause_pattern="""
(exists (?b - bucket ?loc - location ?dim - dimension)
    (inside {rag} ?b ?loc ?dim)
)
"""
    )
    sketch_dict['putting rag in the bucket'].append(rule_put_rag_in_bucket)
    # 3. putting soap in the bucket
    ## 3.1 move toward the nearest soap
    # * precons
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_r_equal_0 = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    p_s_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='soap',
    )
    # * effects
    p_s_decrease = DistanceToNearest(
        wanted_value='decrease',
        type_tag='soap',
    )
    rule_move_to_soap = Rule(
        preconditions=[neg_H_s, N_c_equal_0, N_r_equal_0, p_s_larger_0],
        effects=[p_s_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {soap})
"""
    )
    sketch_dict['putting soap in the bucket'].append(rule_move_to_soap)
    ## 3.2 pick up the soap when reachable
    # * precons
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_r_equal_0 = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    p_s_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='soap',
    )
    # * effects
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    rule_pick_up_soap = Rule(
        preconditions=[neg_H_s, N_c_equal_0, N_r_equal_0, p_s_equal_0],
        effects=[H_s],
        goal_clause_pattern="""
(inhandofrobot agent-01 {soap})
"""
    )
    sketch_dict['putting soap in the bucket'].append(rule_pick_up_soap)
    ## 3.3 move toward the bucket
    # * precons
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_r_equal_0 = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    p_b_larger_0 = DistanceToNearest(
        wanted_value='>0',
        type_tag='bucket',
    )
    # * effects
    p_b_decrease = DistanceToNearest(
        wanted_value='decrease',
        type_tag='bucket',
    )
    rule_move_to_bucket_2 = Rule(
        preconditions=[H_s, N_c_equal_0, N_r_equal_0, p_b_larger_0],
        effects=[p_b_decrease],
        goal_clause_pattern="""
(inreachofrobot agent-01 {bucket})
"""
    )
    sketch_dict['putting soap in the bucket'].append(rule_move_to_bucket_2)
    ## 3.4 put the soap in the bucket when at the spot
    # * precons
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_r_equal_0 = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    p_b_equal_0 = DistanceToNearest(
        wanted_value='=0',
        type_tag='bucket',
    )
    N_s_larger_0 = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    # * effects
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    N_s_decrease = Count(
        type_tag='soap',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    rule_put_soap_in_bucket = Rule(
        preconditions=[H_s, N_c_equal_0, N_r_equal_0, N_s_larger_0, p_b_equal_0],
        effects=[neg_H_s, N_s_decrease],
        goal_clause_pattern="""
(exists (?b - bucket ?loc - location ?dim - dimension)
    (inside {soap} ?b ?loc ?dim)
)
"""
    )
    sketch_dict['putting soap in the bucket'].append(rule_put_soap_in_bucket)
    
    goal_count_feature = []
    goal_count_feature.append(Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    ))
    goal_count_feature.append(Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    ))
    goal_count_feature.append(Count(
        type_tag='soap',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    ))
    
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

### Sketch with W = 1 ; separate carrying and cleaning/placing
# **Clean task**
# $\{\neg H_r, N_c > 0\} \mapsto \{H_r\}$  ; pick up a rag (width 1)
# $\{H_r, N_c > 0\} \mapsto \{N_c \downarrow, \neg H_r\}$  ; clean a focused car (width 1)
# **Put soap and rag in the bucket**
# $\{N_c = 0, \neg H_r\} \mapsto \{H_r\}$  ; pick up a rag (width 1)
# $\{H_r, N_c = 0, N_{in\_bucket} > 0 \} \mapsto \{inside\_rag, \neg H_r, N_{in\_bucket} \downarrow\}$  ; put the rag in the bucket (width 1)
# $\{N_c = 0, \neg H_s\} \mapsto \{H_s\}$  ; pick up a soap (width 1)
# $\{H_s, inside\_rag, N_{in\_bucket} > 0\} \mapsto \{\neg H_s, N_{in\_bucket} \downarrow\}$  ; put the soap in the bucket (width 1)

def create_width_1_sketch(env, map_id, domain_name, window=None):
    width = 1
    sketch_dict = {
        'cleaning task': [],
        'putting rag in the bucket': [],
        'putting soap in the bucket': []
    }
    
    # 1. cleaning task
    ## 1.1 pick up a rag
    # * precons
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_c_larger_0 = Count(
        type_tag='car',
        wanted_value='>0',
        count_func=count_not_wiped,
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag = Rule(
        preconditions=[neg_H_r, N_c_larger_0],
        effects=[H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict['cleaning task'].append(rule_pick_up_rag)
    ## 1.2 clean a focused car
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    N_c_larger_0 = Count(
        type_tag='car',
        wanted_value='>0',
        count_func=count_not_wiped,
    )
    # * effects
    N_c_decrease = Count(
        type_tag='car',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    rule_clean_car = Rule(
        preconditions=[H_r, N_c_larger_0],
        effects=[N_c_decrease, neg_H_r],
        goal_clause_pattern="""
(is-not-dusted {car})
"""
    )
    sketch_dict['cleaning task'].append(rule_clean_car)
    # 2. putting rag in the bucket
    ## 2.1 pick up a rag
    # * precons
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_in_bucket_larger_0 = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    # * effects
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    rule_pick_up_rag_2 = Rule(
        preconditions=[neg_H_r, N_c_equal_0, N_in_bucket_larger_0],
        effects=[H_r],
        goal_clause_pattern="""
(inhandofrobot agent-01 {rag})
"""
    )
    sketch_dict['putting rag in the bucket'].append(rule_pick_up_rag_2)
    ## 2.2 put the rag in the bucket
    # * precons
    H_r = Holding(
        wanted_value=True,
        type_tag='rag',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_in_bucket_larger_0 = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    # * effects
    neg_H_r = Holding(
        wanted_value=False,
        type_tag='rag',
    )
    N_in_bucket_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    rule_put_rag_in_bucket = Rule(
        preconditions=[H_r, N_c_equal_0, N_in_bucket_larger_0],
        effects=[neg_H_r, N_in_bucket_decrease],
        goal_clause_pattern="""
(exists (?b - bucket ?loc - location ?dim - dimension)
    (inside {rag} ?b ?loc ?dim)
)
"""
    )
    sketch_dict['putting rag in the bucket'].append(rule_put_rag_in_bucket)
    # 3. putting soap in the bucket
    ## 3.1 pick up a soap
    # * precons
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_r_equal_0 = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    N_in_bucket_larger_0 = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    # * effects
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    rule_pick_up_soap = Rule(
        preconditions=[neg_H_s, N_c_equal_0, N_r_equal_0],
        effects=[H_s],
        goal_clause_pattern="""
(inhandofrobot agent-01 {soap})
"""
    )
    sketch_dict['putting soap in the bucket'].append(rule_pick_up_soap)
    ## 3.2 put the soap in the bucket
    # * precons
    H_s = Holding(
        wanted_value=True,
        type_tag='soap',
    )
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_r_equal_0 = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    N_s_larger_0 = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    # * effects
    neg_H_s = Holding(
        wanted_value=False,
        type_tag='soap',
    )
    N_s_decrease = Count(
        type_tag='soap',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    rule_put_soap_in_bucket = Rule(
        preconditions=[H_s, N_c_equal_0, N_r_equal_0, N_s_larger_0],
        effects=[neg_H_s, N_s_decrease],
        goal_clause_pattern="""
(exists (?b - bucket ?loc - location ?dim - dimension)
    (inside {soap} ?b ?loc ?dim)
)
"""
    )
    sketch_dict['putting soap in the bucket'].append(rule_put_soap_in_bucket)
    goal_count_feature = []
    goal_count_feature.append(Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    ))
    goal_count_feature.append(Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    ))
    goal_count_feature.append(Count(
        type_tag='soap',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    ))
    
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

### Sketch with W = 2 ; factor out putting soap first 
# **Clean task**
# $\{N_c > 0\} \mapsto \{N_c \downarrow\}$  ; clean a focused car (width 2)
# **Put soap and rag in the bucket**
# $\{N_c = 0, N_{in\_bucket} > 0, \neg inside\_rag \} \mapsto \{N_{in\_bucket} \downarrow, inside\_rag \}$  ; put rag in the bucket (width 2)
# $\{N_c = 0, N_{in\_bucket} > 0, inside\_rag \} \mapsto \{N_{in\_bucket} \downarrow\}$  ; put soap in the bucket (width 2)

def create_width_2_sketch(env, map_id, domain_name, window=None):
    width = 2
    sketch_dict = {
        'cleaning task': [],
        'putting rag in the bucket': [],
        'putting soap in the bucket': []
    }
    
    # 1. cleaning task
    ## 1.1 clean a focused car
    # * precons
    N_c_larger_0 = Count(
        type_tag='car',
        wanted_value='>0',
        count_func=count_not_wiped,
    )
    # * effects
    N_c_decrease = Count(
        type_tag='car',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    rule_clean_car = Rule(
        preconditions=[N_c_larger_0],
        effects=[N_c_decrease],
        goal_clause_pattern="""
(is-not-dusted {car})
"""
    )
    sketch_dict['cleaning task'].append(rule_clean_car)
    # 2. putting rag in the bucket
    ## 2.1 put the rag in the bucket
    # * precons
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_in_bucket_larger_0 = Count(
        type_tag='rag',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    # * effects
    N_in_bucket_decrease = Count(
        type_tag='rag',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    rule_put_rag_in_bucket = Rule(
        preconditions=[N_c_equal_0, N_in_bucket_larger_0],
        effects=[N_in_bucket_decrease],
        goal_clause_pattern="""
(exists (?b - bucket ?loc - location ?dim - dimension)
    (inside {rag} ?b ?loc ?dim)
)
"""
    )
    sketch_dict['putting rag in the bucket'].append(rule_put_rag_in_bucket)
    # 3. putting soap in the bucket
    ## 3.1 put the soap in the bucket
    # * precons
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_in_bucket_larger_0 = Count(
        type_tag='soap',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    N_r_equal_0 = Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    # * effects
    N_in_bucket_decrease = Count(
        type_tag='soap',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    rule_put_soap_in_bucket = Rule(
        preconditions=[N_c_equal_0, N_in_bucket_larger_0, N_r_equal_0],
        effects=[N_in_bucket_decrease],
        goal_clause_pattern="""
(exists (?b - bucket ?loc - location ?dim - dimension)
    (inside {soap} ?b ?loc ?dim)
)
"""
    )
    sketch_dict['putting soap in the bucket'].append(rule_put_soap_in_bucket)
    goal_count_feature = []
    goal_count_feature.append(Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    ))
    goal_count_feature.append(Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    ))
    goal_count_feature.append(Count(
        type_tag='soap',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    ))
    
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


### Sketch with W = 3 ; only $N_c$ matters
# $\{N_c > 0\} \mapsto \{N_c \downarrow\}$  ; clean a focused car (width 2)
# $\{N_c = 0, N_{in\_bucket} > 0\} \mapsto \{N_{in\_bucket} \downarrow\}$  ; put rag and soap in the bucket (width 3)

def create_width_3_sketch(env, map_id, domain_name, window=None):
    width = 3
    sketch_dict = {
        'cleaning task': [],
        'putting rag and soap in the bucket': [],
    }
    # 1. cleaning task
    ## 1.1 clean a focused car
    # * precons
    N_c_larger_0 = Count(
        type_tag='car',
        wanted_value='>0',
        count_func=count_not_wiped,
    )
    # * effects
    N_c_decrease = Count(
        type_tag='car',
        wanted_value='decrease',
        count_func=count_not_wiped,
    )
    rule_clean_car = Rule(
        preconditions=[N_c_larger_0],
        effects=[N_c_decrease],
        goal_clause_pattern="""
(is-not-dusted {car})
"""
    )
    sketch_dict['cleaning task'].append(rule_clean_car)
    # 2. putting rag and soap in the bucket
    ## 2.1 put rag and soap in the bucket
    # * precons
    N_c_equal_0 = Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    )
    N_in_bucket_larger_0 = Count(
        type_tag='rag soap',
        wanted_value='>0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    # * effects
    N_in_bucket_decrease = Count(
        type_tag='rag soap',
        wanted_value='decrease',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    )
    rule_put_rag_and_soap_in_bucket = Rule(
        preconditions=[N_c_equal_0, N_in_bucket_larger_0],
        effects=[N_in_bucket_decrease],
        goal_clause_pattern="""
(exists (?b - bucket ?loc - location ?dim - dimension)
    (inside {rag soap} ?b ?loc ?dim)
)
"""
    )
    sketch_dict['putting rag and soap in the bucket'].append(rule_put_rag_and_soap_in_bucket)
    goal_count_feature = []
    goal_count_feature.append(Count(
        type_tag='car',
        wanted_value='=0',
        count_func=count_not_wiped,
    ))
    goal_count_feature.append(Count(
        type_tag='rag',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    ))
    goal_count_feature.append(Count(
        type_tag='soap',
        wanted_value='=0',
        count_func=partial(count_not_inside_target_type, target_type_name='bucket')
    ))
    
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
    # python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/policy_sketch/cleaning_a_car.py --map_id 325990298 --domain_name cleaning_a_car --width 3 --rewrite
    # env, map_id, domain_name, window, env_id = init_env_for_policy_sketch('data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/cleaning_a_car/p325990298-cleaning_a_car_plans.json', want_render=False)
    
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