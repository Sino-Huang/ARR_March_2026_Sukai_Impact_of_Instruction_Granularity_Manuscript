# abstract class for policy sketch features
# should contain the following functions and attributes 
# Attributes:
# width : int, the width of the overall sketch 
# subgoal_dict : dict, mapping from subgoal name to its collection of rules 
# env : the environment object
# domain_model : the PDDL domain model
# prev_problem_model : the PDDL problem model
# curr_problem_model : the PDDL problem model
# curr_focus_rule: the current focus rule, if None, means no focus rule
# curr_focus_subgoal: the current focus subgoal, if None, means no focus subgoal

# Functions:
# get_available_rules() -> return a list of available rules

# reset_rule -> after completing a rule, reset the rule status, this shall be called inside get_available_rules

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
import random
import time
from typing import List
from mini_behavior.utils.pddl_gen.parse_pddl import obtain_domain_model, obtain_domain_and_problem_instance
from mini_behavior.utils.policy_sketch.general_features import Feature, Count, DistanceToNearest, Holding, count_not_complete_first_layer_salad, isOpened, isToggled, isSoaked, isDustFree, isCleaned
from mini_behavior.window import Window, redraw, reset, key_handler_primitive
from pddl.core import Domain, Problem, Action, Requirements, Predicate
from mini_behavior.utils.pddl_gen.pddl_problem_gen import generate_behavior_pddl, FURNITURE_TYPES
import tempfile
from mini_behavior.utils.pddl_gen.parse_pddl import replace_problem_goal
from mini_behavior.utils.pddl_gen.gen_and_plan import FastDownward
from loguru import logger


def parse_action(action_str: str, env):
    
    action_str = action_str.replace('turn-left', 'left').replace('turn-right', 'right')
    actions = env.actions

    for action in actions:
        action_name = action.name
        
        if action_str.startswith(action_name):
            return action
        
    raise ValueError(f"Action '{action_str}' not recognized in environment.")
        

class GeneralPolicySketch():
    def __init__(self, env, domain_filepath: str, width: int, sketch_dict: dict, goal_count_feature: Count, map_id, domain_name, window=None):
        self.env = env
        self.domain_model = obtain_domain_model(domain_filepath)
        self.domain_filepath = domain_filepath
        self.problem_filepath = None
        self.prev_problem_model_nogoal = None
        self.curr_problem_model_nogoal = None
        self.map_id = int(map_id)
        self.goal_count_feature = goal_count_feature
        self.domain_name = domain_name
        self.width = width
        self.extra_object_to_goal = True # whether to add extra objects to the goal clause, default True (avoid undoing the progress of other objects)
        self.window = window
        self.subgoal_dict = sketch_dict # mapping from subgoal name to its collection of rules 
        self.curr_focus_rule = None # the current focus rule, if None, means no focus rule
        self.curr_focus_subgoal = None # the current focus subgoal, if None, means no focus subgoal
        self.curr_focus_rule_index = -1 # the index of the current focus rule in the subgoal's rule list
        self.curr_ignore_object_dict = None # the current ignore object dict, if None, means no ignore object
        self.curr_focus_rule_done = True # whether the current focus rule is done, initially True, so that we can pick a new rule
        
        
        self.stored_info ={
            "performed_rules": [], # list of (count, subgoal, rule, actions)
            "executed_actions": [], # list of executed action strings
            "map_id": self.map_id,
            "domain_name": self.domain_name,
        }

    def _get_available_rules(self, assert_done=True) -> dict:
        """
        Return a list of available rules.
        """
        # we assume that the current problem_model is up to date as we call _calculate_current_problem_model_nogoal during perform_solution function.
        curr_ignore_object_dict_dict = self._auto_reset_rule(assert_done=assert_done)
        available_rules = dict()
        available_rules_index = dict()
        for subgoal, rules in self.subgoal_dict.items():
            available_rules[subgoal] = []
            available_rules_index[subgoal] = []
            for i, rule in enumerate(rules):
                if rule.check_preconditions(self.domain_model, self.curr_problem_model_nogoal):
                    available_rules[subgoal].append(rule)
                    available_rules_index[subgoal].append(i)
        return available_rules, available_rules_index, curr_ignore_object_dict_dict
    
    def _query_solution(self) -> List[str]:
        assert self.curr_focus_rule, "No current focus rule selected."
        assert self.curr_focus_subgoal, "No current focus subgoal selected."
        assert not self.curr_focus_rule_done, "Current focus rule is already done."
        # obtain the goal_actual_clause from the curr_focus_rule
        goal_actual_clause = self.curr_focus_rule.goal_actual_clause
        assert goal_actual_clause is not None, "The goal_actual_clause of the current focus rule is not set."
        # get domain and problem instance with the new goal
        
        # consider adding ignore_objects to the goal clause
        goal_lines = []
        str_ver_goal_clause = str(goal_actual_clause)
        # split by lines
        split_lines = str_ver_goal_clause.split('\n')
        for line in split_lines:
            goal_lines.append(line)
        additional_goal_lines = []
        if self.extra_object_to_goal or self.domain_name in ['laying_wood_floors', 'making_tea']: 
            if self.curr_ignore_object_dict is not None and len(self.curr_ignore_object_dict) > 0:
                for object_type, ignore_objects in self.curr_ignore_object_dict.items():
                    for obj_instance in ignore_objects:
                        # get the obj_instance_name 
                        obj_instance_name = obj_instance.name if hasattr(obj_instance, 'name') else str(obj_instance)
                        # ! special case, if the goal_lines contain something related to this object, we also skip
                        skip_adding_at_clause = False
                        for existing_line in split_lines:
                            if len(existing_line.split(' '))>2 and obj_instance_name in existing_line.split(' ')[1]:
                                skip_adding_at_clause = True
                                break
                        if skip_adding_at_clause:
                            continue
                        # ! another special case, skip furniture objects
                        skip_adding_at_clause = False 
                        furniture_name_prefixes = FURNITURE_TYPES
                        for furniture_prefix in furniture_name_prefixes:
                            if obj_instance_name.startswith(furniture_prefix):
                                skip_adding_at_clause = True
                                break
                        if skip_adding_at_clause:
                            continue
                        # ! special case, crucial part, we iterate over existing goal_lines, if we see a line that contain both inhandofrobot and obj_instance_name, we skip adding at clause for this object
                        skip_adding_at_clause = False
                        for goal_line in goal_lines:
                            if 'inhandofrobot' in goal_line and obj_instance_name in goal_line:
                                skip_adding_at_clause = True
                                break
                        if skip_adding_at_clause:
                            continue                        
                        # iterate over predicates and add (at obj_name pos, dim) to the goal clause if applicable
                        for predicate in self.curr_problem_model_nogoal.init:
                            if predicate.name == 'at':
                                terms = predicate.terms
                                obj, pos, dim = terms
                                if obj_instance == obj:
                                    additional_goal_lines.append(f"(at {obj_instance.name} {pos} {dim})")
            
        # goal_lines = list(set(goal_lines)) # remove duplicates
        goal_lines.extend(additional_goal_lines)

        logger.info(f"Planning for rule: {self.curr_focus_rule} with goal:\n" + '\n'.join(goal_lines))
        domain_str, problem_str = replace_problem_goal('\n'.join(goal_lines), self.domain_filepath, self.problem_filepath)

        # save domain and problem to temporary files
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_domain_file:
            temp_domain_file.write(domain_str)
            temp_domain_file_path = temp_domain_file.name
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_problem_file:
            temp_problem_file.write(problem_str)
            temp_problem_file_path = temp_problem_file.name
        
        my_planner = FastDownward()
        # plan
        plans = my_planner(temp_domain_file_path, temp_problem_file_path, timeout='7m', optimal=True)
        assert len(plans) > 0, "No plans found."
        
        # delete temporary files
        Path(temp_domain_file_path).unlink()
        Path(temp_problem_file_path).unlink()
        
        return plans
        
        
    
    def perform_solution(self):
        """
        Perform a sequence of actions in the environment.
        will update curr_focus_rule_done to True, otherwise raise an error
        """
        sequence_of_actions = self._query_solution()[0]
        for action_str in sequence_of_actions:
            action = parse_action(action_str, self.env)
            obs, reward, done, info = self.env.step(action)
            if self.window is not None:
                redraw(obs, self.env, self.window)
                
        # ! storage process, we need to store what we have done. 
        self.stored_info["performed_rules"].append((len(self.stored_info["performed_rules"])+1, self.curr_focus_subgoal, str(self.curr_focus_rule), sequence_of_actions))
        self.stored_info["executed_actions"].extend(sequence_of_actions)
            
        # now the env is updated, we need to call _calculate_current_problem_model_nogoal to update the problem model
        self._calculate_current_problem_model_nogoal()
        effects = self.curr_focus_rule.effects

        logger.info(f"Execute Solution for {self.curr_focus_rule}")

        for effect in effects:
            if not effect.verify(self.domain_model, self.curr_problem_model_nogoal):
                breakpoint()
                # str(self.curr_focus_rule)
                # str(self.curr_focus_rule.effects[0].ground_var_value_dict) # pot_plant_1
                # self.env.env.obj_instances['pot_plant_1']
                # self.env.env.carrying
                # self.env.env.agent_pos
                # self.env.env.agent_dir
                
                raise ValueError("The effects of the current focus rule are not satisfied after performing the actions.")
        self.curr_focus_rule_done = True
        
        # after performing the solution, we need to reset the curr_focus_rule and curr_focus_subgoal
        # but we keep the curr_focus_subgoal, so that we can prefer to pick a new rule from the same subgoal
        

    def determine_new_focus_rule(self, assert_done=True):
        """
        Pick one available rule from the available rules.
        """
        if assert_done:
            assert self.curr_focus_rule_done, "Current focus rule is not done yet."
        
        temp_save_subgoal = None
        temp_save_rule_index = None
        if not assert_done:
            # this is the update and get focus rule function call, a quick shortcut is to check if the current focus rule is still available
            if self.curr_focus_rule is not None:
                # check if the current focus rule is still available
                cur_rule_finished = self.curr_focus_rule.check_effects(self.domain_model, self.curr_problem_model_nogoal)
                if not cur_rule_finished:
                    # it means we do not need to pick a new rule
                    # but refresh it may help 
                    
                    # so we store the index of the current focus rule
                    temp_save_rule_index = self.curr_focus_rule_index
                    temp_save_subgoal = self.curr_focus_subgoal

        available_rules, available_rules_index, curr_ignore_object_dict_dict = self._get_available_rules(assert_done=assert_done)
        logger.info(f"Available rules")
        
        if temp_save_subgoal is not None and temp_save_rule_index is not None:
            
            self.curr_focus_subgoal = temp_save_subgoal
            self.curr_focus_rule_index = temp_save_rule_index
            self.curr_focus_rule = self.subgoal_dict[self.curr_focus_subgoal][self.curr_focus_rule_index]
            self.curr_focus_rule_done = False
            self.curr_ignore_object_dict = curr_ignore_object_dict_dict.get(self.curr_focus_subgoal, dict())
            logger.info(f'Shortcut as current focus rule is still available: {self.curr_focus_rule}')
            return self.curr_focus_rule
        
        for subgoal, rules in available_rules.items():
            logger.info(f"Subgoal: {subgoal}, Available Rules: {len(rules)}")
            for rule in rules:
                logger.info(f"   {rule}")
        # prefer to pick the one at the same subgoal
        if self.curr_focus_subgoal is None:
            # pick any available rule 
            for subgoal, rules in available_rules.items():
                if len(rules) > 0:
                    self.curr_focus_subgoal = subgoal
                    # randomly pick one rule
                    local_rule_random_index = random.randint(0, len(rules)-1)
                    selected_rule = rules[local_rule_random_index]
                    selected_rule_index = available_rules_index[subgoal][local_rule_random_index]
                    self.curr_focus_rule = selected_rule
                    self.curr_focus_rule_index = selected_rule_index
                    # reset the rule status
                    self.curr_focus_rule_done = False
                    self.curr_ignore_object_dict = curr_ignore_object_dict_dict.get(subgoal, dict())
                    return self.curr_focus_rule
            raise ValueError("No available rule found.")
        else:
            # pick from the current focus subgoal 
            rules = available_rules.get(self.curr_focus_subgoal, [])
            if len(rules) > 0:
                local_rule_random_index = random.randint(0, len(rules)-1)
                selected_rule = rules[local_rule_random_index]
                selected_rule_index = available_rules_index[self.curr_focus_subgoal][local_rule_random_index]
                self.curr_focus_rule = selected_rule
                self.curr_focus_rule_index = selected_rule_index
                self.curr_focus_rule_done = False
                self.curr_ignore_object_dict = curr_ignore_object_dict_dict.get(self.curr_focus_subgoal, dict())
                return self.curr_focus_rule
            else:
                # no available rule in the current focus subgoal, pick any available rule
                for subgoal, rules in available_rules.items():
                    if len(rules) > 0:
                        self.curr_focus_subgoal = subgoal
                        local_rule_random_index = random.randint(0, len(rules)-1)
                        selected_rule = rules[local_rule_random_index]
                        selected_rule_index = available_rules_index[subgoal][local_rule_random_index]
                        # reset the rule status
                        self.curr_focus_rule = selected_rule
                        self.curr_focus_rule_index = selected_rule_index
                        self.curr_focus_rule_done = False
                        self.curr_ignore_object_dict = curr_ignore_object_dict_dict.get(subgoal, dict())
                        return self.curr_focus_rule
                raise ValueError("No available rule found.")
    
    def _calculate_current_problem_model_nogoal(self):
        """
        Calculate the current problem model from the environment.
        This function should be called at the beginning of get_available_rules.
        this will be called in the perform_solution function. 
        """
        # * Save the previous problem model
        self.prev_problem_model_nogoal = self.curr_problem_model_nogoal
        # * Obtain the current problem model from the environment
        pddl_problem_str = generate_behavior_pddl(self.env)
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_problem_file:
            temp_problem_file.write(pddl_problem_str)
            temp_problem_file_path = temp_problem_file.name
        self.curr_problem_model_nogoal = obtain_domain_and_problem_instance(self.domain_filepath, temp_problem_file_path)[1]
        
        if self.problem_filepath is not None:
            Path(self.problem_filepath).unlink()  # Delete the previous temporary file
            
        self.problem_filepath = temp_problem_file_path
  

    def _auto_reset_rule(self, assert_done=True):
        """
        After completing a rule, reset the rule status, this shall be called inside get_available_rules. this result involves reset all the rules, and then determine the 
        """
        if assert_done:
            assert self.curr_focus_rule_done, "Current focus rule is not done yet."
        # if self.curr_focus_rule is None, reset all rules
        self.curr_ignore_object_dict = dict()
        curr_ignore_object_dict_dict = dict()
        for subgoal, rules in self.subgoal_dict.items():
            curr_ignore_object_dict_local = dict()
            Rule.reset(rules, self.domain_model, self.curr_problem_model_nogoal, curr_ignore_object_dict_local, self.width)
            curr_ignore_object_dict_dict[subgoal] = curr_ignore_object_dict_local
        return curr_ignore_object_dict_dict

            
    def evaluate_full_sketch(self):
        # first step 
        self._calculate_current_problem_model_nogoal()
        while True:
            if not isinstance(self.goal_count_feature, list):
                focus_goal_count_features = [self.goal_count_feature]
            else:
                focus_goal_count_features = self.goal_count_feature
                
            goal_achieved = True
            for gcf in focus_goal_count_features:
                if not gcf.activated:
                    gcf.activate_in_precon(self.domain_model, self.curr_problem_model_nogoal)
                if not gcf.verify(self.domain_model, self.curr_problem_model_nogoal):
                    goal_achieved = False
                    break
                
            if goal_achieved:
                break
            try: 
                self.determine_new_focus_rule()
            except ValueError:
                print("No available rule found, finishing evaluation.")
                break
            self.perform_solution()
            
        # finally, re-check if the goal is achieved
        # update the problem model
        self._calculate_current_problem_model_nogoal()
        if not isinstance(self.goal_count_feature, list):
            focus_goal_count_features = [self.goal_count_feature]
        else:
            focus_goal_count_features = self.goal_count_feature
            
        goal_achieved = True
        for gcf in focus_goal_count_features:
            if not gcf.activated:
                gcf.activate_in_precon(self.domain_model, self.curr_problem_model_nogoal)
            if not gcf.verify(self.domain_model, self.curr_problem_model_nogoal):
                goal_achieved = False
                logger.warning(f"Goal not achieved for feature: {gcf}")
                break
            
        if goal_achieved:
            logger.success("Goal achieved!")
        return goal_achieved, self.stored_info
    
    def update_and_get_focus_rule(self):
        """
        Similar to evaluate_full_sketch, but this time, we need to combine the function of determine_new_focus_rule and perform_solution, and the actual solution comes from outside.
        the return is (focus_subgoal, focus_rule_str)
        if there is no available rule,
        if achieved, return ("finished", "")
        else:
        return (None, None)
        """
        self._calculate_current_problem_model_nogoal()
        
        if not isinstance(self.goal_count_feature, list):
            focus_goal_count_features = [self.goal_count_feature]
        else:
            focus_goal_count_features = self.goal_count_feature
            
        goal_achieved = True
        for gcf in focus_goal_count_features:
            if not gcf.activated:
                gcf.activate_in_precon(self.domain_model, self.curr_problem_model_nogoal)
            if not gcf.verify(self.domain_model, self.curr_problem_model_nogoal):
                goal_achieved = False
                logger.warning(f"Goal not achieved for feature: {gcf}")
                break
            
        if goal_achieved:
            logger.success("Goal achieved!")
            return "finished", ""
        
        # now we need to determine the current focus rule
        try:
            self.determine_new_focus_rule(assert_done=False) # not assert done here because we will call this function every time when we perform a chunk of actions even if we have not completed the current rule
            
        except ValueError:
            logger.warning("No available rule found.")
            return None, None
        focus_subgoal = self.curr_focus_subgoal
        focus_rule_str = str(self.curr_focus_rule) if self.curr_focus_rule else None
        
        
        return focus_subgoal, focus_rule_str
        

def sort_heuristic(feature: Feature):
    """
    A heuristic function to sort features.
    Priority:
    1. Count feature with wanted_value '>0' first
    2. DistanceToNearest feature next
    3. Others
    """
    if isinstance(feature, Count) and feature.wanted_value == '>0':
        return 0
    elif isinstance(feature, DistanceToNearest):
        return 1
    else:
        return 2

class Rule(ABC):
    def __init__(self, preconditions=None, effects=None, goal_clause_pattern=None):
        """
        Initialize a rule with preconditions and effects.
        
        Args:
            preconditions (list): List of Feature objects that must be satisfied
            effects (list): List of Feature objects that represent the outcome
        """
        self.preconditions = preconditions if preconditions is not None else []
        self.effects = effects if effects is not None else []
        assert goal_clause_pattern is not None, "goal_clause_pattern must be provided"
        self.goal_clause_pattern = goal_clause_pattern 
        self.goal_actual_clause = None 

    def check_preconditions(self, domain_model, problem_model):
        """
        Check if all preconditions are satisfied.
        
        Returns:
            bool: True if all preconditions are met, False otherwise
        """
        for precondition in self.preconditions:
            if not precondition.verify(domain_model, problem_model):
                return False
        return True
    
    def check_effects(self, domain_model, problem_model):
        """
        Check if all effects are satisfied.
        
        Returns:
            bool: True if all effects are met, False otherwise
        """
        for effect in self.effects:
            if not effect.verify(domain_model, problem_model):
                return False
        return True
    
    @classmethod
    def reset(cls, rules, domain_model, problem_model, ignore_objects_dict, width):
        """
        Reset the rule status. call set_ground in the features of the preconditions and effects. call reset_ground for each feature first and then do some specific reset operations, leave it as empty here. this is where we also reset goal_actual_clause.
        """
        # for each rule, we accumulate the ground_var_value_dict 
        ground_var_value_dict = dict()
        # reverse rules, as the last rule contains count feature
        ongoing_objects_dict = dict()
        
        precon_reset_later = [] 
        for rule in reversed(rules):
            # sort the precondition so that Count feature whose wanted_value is '>0' is processed first
            # second sort, sort DistanceToNearest feature to be processed after Count feature
            rule.preconditions.sort(key=sort_heuristic)
            precondition : Feature
            for precondition in rule.preconditions:
                precondition.force_invalid = False
                # special case for Count feature 
                if isinstance(precondition, Count):
                    precondition.reset_ground_precon(
                        updated_var_value_dict = None,
                        ignore_objects = None,
                        domain_model = domain_model,
                        problem_model = problem_model,
                    )
                    if hasattr(precondition.count_func, "func"):
                        count_func_name = precondition.count_func.func.__name__
                    else:
                        count_func_name = precondition.count_func.__name__
                    additional_info = precondition.additional_info
                    # ignore_objects are in the second value
                    values = list(additional_info.values())
                    if len(values) > 1 and (precondition.wanted_value in ['>0']
                                            or (precondition.wanted_value == '=0' and 'count_not_inside' in count_func_name)
                                            or (precondition.wanted_value == '=0' and 'count_not_near' in count_func_name)
                                            or (precondition.wanted_value == '=0' and 'count_not_ontop' in count_func_name)
                                            ): # ! only >0 will generate ignore_objects
                        ignore_objects = values[1]
                        ongoing_objects_dict[precondition.object_type] = [x.name if getattr(x, 'name', None) else str(x) for x in values[0]] # the first value is ongoing_objects, convert to name (str)
                        if precondition.object_type in ignore_objects_dict:
                            ignore_objects_dict[precondition.object_type].extend(ignore_objects)
                            ignore_objects_dict[precondition.object_type] = list(set(ignore_objects_dict[precondition.object_type]))
                        else:
                            ignore_objects_dict[precondition.object_type] = ignore_objects
                elif isinstance(precondition, DistanceToNearest):
                    ignore_objects = ignore_objects_dict.get(precondition.object_type, [])
                    precondition.reset_ground_precon(
                        updated_var_value_dict = None,
                        ignore_objects = ignore_objects,
                        domain_model = domain_model,
                        problem_model = problem_model,
                    )
                    # update the ground_var_value_dict
                    ground_var_value_dict.update(precondition.ground_var_value_dict)
                else:
                    try:
                        precondition.reset_ground_precon(
                            updated_var_value_dict = ground_var_value_dict,
                            ignore_objects = None,
                            domain_model = domain_model,
                            problem_model = problem_model,
                        )
                    except Exception as e:
                        # if failed, we store it to reset later 
                        precon_reset_later.append(precondition)
        # process the reset_later list
        for precondition in precon_reset_later:
            try:
                precondition.reset_ground_precon(
                    updated_var_value_dict = ground_var_value_dict,
                    ignore_objects = None,
                    domain_model = domain_model,
                    problem_model = problem_model,
                )
            except Exception as e:
                if width == 0:
                    logger.warning(f"Failed to reset precondition {precondition}, force it to be invalid.")
                    precondition.force_invalid = True
                else:
                    try:
                        # use ongoing_objects_dict to simulate a ground_var_value_dict
                        simulated_ground_var_value_dict = dict()
                        for var, obj_list in ongoing_objects_dict.items():
                            # just use the first object
                            if len(obj_list) > 0:
                                simulated_ground_var_value_dict[var] = obj_list[0]
                        precondition.reset_ground_precon(
                            updated_var_value_dict = simulated_ground_var_value_dict,
                            ignore_objects = None,
                            domain_model = domain_model,
                            problem_model = problem_model,
                        )
                        ground_var_value_dict.update(precondition.ground_var_value_dict)
                    except Exception as e:
                        try:
                            if len(ongoing_objects_dict) == 0 :
                                logger.warning(f"Failed to reset precondition {precondition}, force it to be invalid.")
                                precondition.force_invalid = True
                            else:
                                # randomly pick one object of the required type
                                object_type = precondition.object_type
                                all_objects = [obj.name for obj in problem_model.objects if obj.type_tag == object_type]
                                random_obj = random.choice(all_objects)
                                simulated_ground_var_value_dict = {object_type: random_obj}
                                precondition.reset_ground_precon(
                                    updated_var_value_dict = simulated_ground_var_value_dict,
                                    ignore_objects = None,
                                    domain_model = domain_model,
                                    problem_model = problem_model,
                                )
                                ground_var_value_dict.update(precondition.ground_var_value_dict)
                        except Exception as e:
                            # if still failed, we force it to be invalid
                            logger.warning(f"Failed to reset precondition {precondition}, force it to be invalid.")
                            precondition.force_invalid = True
                
        
        # now we reset effect 
        
        for rule in reversed(rules):
            # process the effects
            effect_var_value_dict = ground_var_value_dict.copy()
            effect : Feature
            for effect in rule.effects:
                # find corresponding precondition feature
                corresponding_precon = None
                for precondition in rule.preconditions:
                    if type(precondition) == type(effect) and precondition.object_type == effect.object_type:
                        corresponding_precon = precondition
                        break
                if corresponding_precon is None:
                    raise ValueError("No corresponding precondition found for effect.")
                effect.reset_ground_effect(corresponding_precon)
                # update the effect_var_value_dict
                effect_var_value_dict.update(effect.ground_var_value_dict)
            # update the goal_actual_clause using the effect's ground_var_value_dict
            # goal_clause_pattern looks like (and (is_toggled {object_type}) (is_opened {object_type})) and <object_type> is a key in effect_var_value_dict
            goal_actual_clause = str(rule.goal_clause_pattern)
            for var, value in effect_var_value_dict.items():
                if isinstance(value, str):
                    goal_actual_clause = goal_actual_clause.replace(f"{{{var}}}", value)
            # further check if there is any remaining {var} in the goal_actual_clause, if yes, we need to do again with the ongoing_objects_dict
            if '{' in goal_actual_clause and '}' in goal_actual_clause:
                for var, obj_list in ongoing_objects_dict.items():
                    if len(obj_list) > 0:
                        goal_actual_clause = goal_actual_clause.replace(f"{{{var}}}", obj_list[0])
            # further check if there is any remaining {var} in the goal_actual_clause, if yes, we loop over all objects of that type and replace it with the first one
            if '{' in goal_actual_clause and '}' in goal_actual_clause:
                # iterate over all {var} in the goal_actual_clause
                goal_actual_clause_temp = deepcopy(goal_actual_clause)
                for goal_actual_clause_idx in range(len(goal_actual_clause_temp)):
                    if goal_actual_clause_temp[goal_actual_clause_idx] == '{':
                        end_idx = goal_actual_clause_temp.find('}', goal_actual_clause_idx)
                        if end_idx != -1:
                            var = goal_actual_clause_temp[goal_actual_clause_idx+1:end_idx]
                            # find all objects of that type
                            focus_objects = []
                            for object in problem_model.objects:
                                if object.type_tag == var:
                                    focus_objects.append(object)
                            if len(focus_objects) > 0:
                                goal_actual_clause = goal_actual_clause.replace(f"{{{var}}}", focus_objects[0].name)
            # finally, assign the goal_actual_clause to the rule
            # additional consideration
            # split the goal_actual_clause by '\n`, loop over each line, if we see there is (nextto obj1 obj2), we need to also add (at obj2 pos dim) to the goal clause to ensure obj2 does not move 
            split_lines = goal_actual_clause.split('\n')
            final_goal_lines = []
            additional_goal_lines = []
            for line in split_lines:
                final_goal_lines.append(line)
                line = line.strip()
                if line.startswith('(nextto'):
                    terms = line[len('(nextto'): -1].strip().split()
                    terms = [term.strip() for term in terms]
                    terms = [x for x in terms if x != '']
                    if len(terms) == 2:
                        obj1_name, obj2_name = terms
                        # find the (at obj2 pos dim) predicate in the problem model
                        for predicate in problem_model.init:
                            if predicate.name == 'at':
                                p_terms = predicate.terms
                                p_obj, p_pos, p_dim = p_terms
                                if p_obj.name == obj2_name:
                                    additional_goal_lines.append(f"(at {obj2_name} {p_pos} {p_dim})")
            final_goal_lines.extend(additional_goal_lines)  
            goal_actual_clause = '\n'.join(final_goal_lines)
            rule.goal_actual_clause = goal_actual_clause
            

    def __str__(self):
        preconds_str = ', '.join(str(p) for p in self.preconditions)
        effects_str = ', '.join(str(e) for e in self.effects)
        return f"Rule(Preconditions: [{preconds_str}] -> Effects: [{effects_str}])"
    
    def __repr__(self):
        return self.__str__()