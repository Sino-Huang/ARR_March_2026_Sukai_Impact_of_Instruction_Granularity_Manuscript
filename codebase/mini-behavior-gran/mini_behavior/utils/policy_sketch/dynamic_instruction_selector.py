import os 
import sys 
import json 
import random
import time
from pathlib import Path 
from subprocess import Popen, PIPE
import subprocess
from glob import glob
from multiprocessing import Pool  # multiprocessing for parallel execution
import tempfile
from tqdm.auto import tqdm
import argparse
import pickle
from mini_behavior.utils.policy_sketch.generate_instructions import verbalize_rule
import importlib  # Import the library for dynamic loading
import sys
from mini_behavior.utils.policy_sketch.general_policy_sketch import GeneralPolicySketch
from pddl.core import Domain, Problem, Predicate
from mini_behavior.utils.pddl_gen.parse_pddl import obtain_objs_from_problem_instance, obtain_preidicates_categorized_by_objs, format_predicate_dict_to_text


class InstructorSelector:
    def __init__(self, instruction_width, instruction_domain_name, update_seed=True, default_seed=42):
        self.instruction_width = instruction_width
        self.instruction_domain_name = instruction_domain_name
        self.general_policy_sketch : GeneralPolicySketch  = None
        self.idx = 0 
        self.previous_rule_str = ""
        self.ready_to_generate_text_world_state_str = False
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        if cur_dir not in sys.path:  # add current directory to sys.path
            sys.path.append(cur_dir)
            
        self.update_seed = update_seed
        self.seed = default_seed
        
        
    def reset(self, env):
        # reset the policy sketch object, maybe load the policy sketch here
        dummy_map_id = 0
        dummy_domain_name = "dummy_domain"
        sketch_creation_name = f'create_width_{self.instruction_width}_sketch'
        # here need to load teh sketch creation function dynamically and use the domain_name 
        module = importlib.import_module(self.instruction_domain_name)
        sketch_creation_func = getattr(module, sketch_creation_name)
        self.general_policy_sketch = sketch_creation_func(
            env, dummy_map_id, dummy_domain_name, None
        )
        self.idx = 0 
        self.previous_rule_str = ""
        # update the seed if needed
        if self.update_seed:
            self.seed = int(time.time()) % (2**32 - 1)

        
    
    def select_instruction(self):
        
        focus_subgoal, focus_rule_str = self.general_policy_sketch.update_and_get_focus_rule()
        # set ready_to_generate_text_world_state_str to True once update_and_get_focus_rule is called
        self.ready_to_generate_text_world_state_str = True
        
        if focus_subgoal == "finish":
            return "You have completed all the tasks. Well done!"
        elif focus_subgoal is None:
            return "No available instruction at the moment. Explore the environment by yourself."
        else:
            if focus_rule_str != self.previous_rule_str:
                self.idx += 1
                self.previous_rule_str = focus_rule_str
                
            # postprocessing the str
            sent, _ = verbalize_rule(focus_rule_str)
            return f"At step {self.idx}: {sent}"
        
        
    def generate_text_world_state_str(self):
        assert self.general_policy_sketch is not None, "Please reset the InstructorSelector with an environment first."
        if not self.ready_to_generate_text_world_state_str:
            self.general_policy_sketch._calculate_current_problem_model_nogoal()
            self.ready_to_generate_text_world_state_str = True
            
        problem_model_nogoal : Problem = self.general_policy_sketch.curr_problem_model_nogoal
        domain_model : Domain = self.general_policy_sketch.domain_model
        objs = obtain_objs_from_problem_instance(problem_model_nogoal, domain_model, filter_obj_name_prefix='pos-')
        obj_to_preds = obtain_preidicates_categorized_by_objs(problem_model_nogoal, objs)
        text_world_state_str = format_predicate_dict_to_text(
            obj_to_predicates=obj_to_preds,
            num_of_variants=1,
            object_shuffle_prob=1.0,
            predicate_shuffle_prob=1.0,
            seed=self.seed,
        )[0]
        self.ready_to_generate_text_world_state_str = False # set to False after generating
        return text_world_state_str
        
    
    def close(self):
        # detach 
        self.general_policy_sketch = None
      
      


if __name__ == "__main__":
    # test_domain_name = "collect_misplaced_items"
    # test_instruction_width = 1
    # sketch_creation_name = f'create_width_{test_instruction_width}_sketch'
    # if '.' not in sys.path:  # add current directory to sys.path
    #     sys.path.append('.')
        
    # module = importlib.import_module(test_domain_name)
    # func = getattr(module, sketch_creation_name)
    
    
    # test pddl problem object 
    problem_fp = "/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_problems/cleaning_shoes/p339833990-cleaning_shoes.pddl"
    domain_fp = "/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/minibehavior.pddl"
    
    from mini_behavior.utils.pddl_gen.parse_pddl import obtain_domain_and_problem_instance
    domain, problem = obtain_domain_and_problem_instance(domain_fp, problem_fp)
    
    # Test obtain_objs_from_problem_instance
    objs = obtain_objs_from_problem_instance(problem, domain, filter_obj_name_prefix='pos-')
    print(f"Number of objects: {len(objs)}")
    print(f"Sample objects: {[obj.name for obj in objs[:15]]}")
    
    # Test obtain_preidicates_categorized_by_objs
    obj_to_preds = obtain_preidicates_categorized_by_objs(problem, objs)
    print(f"\nNumber of objects in dictionary: {len(obj_to_preds)}")
    
    # Test format_predicate_dict_to_text with first 5 objects
    first_n_objs = {obj: obj_to_preds[obj] for obj in objs[:15]}
    
    # Generate 3 variants with different shuffle probabilities
    print("\n" + "="*60)
    print("Testing with 3 variants (full shuffling):")
    print("="*60)
    variants = format_predicate_dict_to_text(
        first_n_objs, 
        num_of_variants=3,
        object_shuffle_prob=1.0,
        predicate_shuffle_prob=1.0
    )
    for i, variant in enumerate(variants, 1):
        print(f"\n--- Variant {i} ---")
        print(variant)
    
    # Test with partial shuffling
    print("\n" + "="*60)
    print("Testing with 2 variants (50% object shuffle, 100% predicate shuffle):")
    print("="*60)
    variants_partial = format_predicate_dict_to_text(
        first_n_objs,
        num_of_variants=2,
        object_shuffle_prob=0.5,
        predicate_shuffle_prob=1.0
    )
    for i, variant in enumerate(variants_partial, 1):
        print(f"\n--- Variant {i} ---")
        print(variant) 
    
    
    
    # conda activate /scratch/xxxplace0478/xxxnameh/nsai_behavior