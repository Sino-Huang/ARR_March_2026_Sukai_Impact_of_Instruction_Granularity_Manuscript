from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from mini_behavior.utils.pddl_gen.parse_pddl import obtain_domain_model, obtain_domain_and_problem_instance
import argparse
from pddl.core import Domain, Problem, Action, Requirements, Predicate

# * helper function 
# get the location of agent-01

def get_agent_location(problem_model):
    """Input: problem model
    Output: (x, y) of agent-01
    """
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.name == 'agent-01':
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                return int(pos_x), int(pos_y)
    raise ValueError("Agent location not found in problem model.")

def get_closest_object_location(problem_model, object_type, agent_pos, ignore_objects: List[str] = None, consider_carried_objects=True):
    """Get the closest object of a specific type to the agent's position.
    1. Find all objects of the specified type in the problem model.
    2. Calculate the Manhattan distance from the agent's position to each object's position.
    3. Return the location and distance of the closest object.
    """
    predicate: Predicate
    agent_pos_x, agent_pos_y = agent_pos
    min_distance = float('inf')
    closest_location = None
    closest_object_name = None
    additional_info = dict()
    
    ignore_closest_object = None 
    ignore_min_distance = float('inf')
    ignore_closest_object_name = None
    
    
    if " " in object_type: # means it can be split 
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
    
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag in object_type_lst:
                
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                pos_x, pos_y = int(pos_x), int(pos_y)
                distance = abs(agent_pos_x - pos_x) + abs(agent_pos_y - pos_y)  # Manhattan distance
                settled_flag = False
                if ignore_objects:
                    if first_term.name in ignore_objects or first_term in ignore_objects: # either it can be name or Term type obj
                        settled_flag = True 
                        if distance < ignore_min_distance:
                            ignore_min_distance = distance
                            ignore_closest_object = (pos_x, pos_y)
                            ignore_closest_object_name = first_term.name
                
                if distance < min_distance and not settled_flag:
                    min_distance = distance
                    closest_location = (pos_x, pos_y)
                    closest_object_name = first_term.name
        elif predicate_name == 'inhandofrobot':
            # if the object is in hand of robot, then we directly return this object and the min_distance = 0
            terms = predicate.terms
            object_term = terms[1]
            
            if object_term.type_tag in object_type_lst:
                if ignore_objects:
                    if object_term.name in ignore_objects or object_term in ignore_objects: # either it can be name or Term type obj
                        continue
                if consider_carried_objects:
                    closest_location = agent_pos  # same as agent position
                    min_distance = -1 # indicate it is in hand of robot
                    closest_object_name = object_term.name
                    break  # no need to check further, since this is the closest possible
                else:
                    additional_info['carried_object'] = (object_type, object_term.name)
    
    if closest_location is None and not consider_carried_objects:
        return ignore_closest_object, ignore_min_distance, ignore_closest_object_name, additional_info

    return closest_location, min_distance, closest_object_name, additional_info


    
def calculate_distance(problem_model, obj_name_1, obj_name_2):
    """Calculate the Manhattan distance between two objects in the problem model."""
    predicate: Predicate
    obj1_pos = []
    obj2_pos = []
    min_distance = float('inf')
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.name == obj_name_1:
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                obj1_pos.append((int(pos_x), int(pos_y)))
            elif first_term.name == obj_name_2:
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                obj2_pos.append((int(pos_x), int(pos_y)))
        elif predicate_name == 'inhandofrobot':
            terms = predicate.terms
            agent_term = terms[0]
            object_term = terms[1]
            if object_term.name == obj_name_1:
                # if obj_name_1 is in hand of robot, then its position is the same as agent
                agent_pos = get_agent_location(problem_model)
                obj1_pos.append(agent_pos)
            elif object_term.name == obj_name_2:
                agent_pos = get_agent_location(problem_model)
                obj2_pos.append(agent_pos)

    if not obj1_pos or not obj2_pos:
        raise ValueError(f"One or both objects {obj_name_1}, {obj_name_2} not found in problem model.")
    obj1_pos = list(set(obj1_pos))  # unique positions
    obj2_pos = list(set(obj2_pos))  # unique positions
    
    for pos1 in obj1_pos:
        for pos2 in obj2_pos:
            distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
            if distance < min_distance:
                min_distance = distance
    return min_distance
# * --- end of helper function

class Feature(ABC):
    # Class-level attribute to indicate if grounding is required (defaults to False)
    def __init__(self, require_ground=False):
        self.require_ground = require_ground
        self.ground_var_value_dict = dict()
        self.object_type = None  # to be defined in subclasses if needed
        self.force_invalid = False  # if set to True, then verify always return False


    @abstractmethod
    def verify(self, domain_model, problem_model):
        """Check if the feature holds in the given environment."""
        pass

    @classmethod
    def set_ground(cls, list_of_features, ground_var_value_dict):
        """Set the class-level grounding requirement."""
        # first, assert all features have True require_ground
        for feature in list_of_features:
            assert feature.require_ground, "All features must have require_ground=True to set grounding."
        for feature in list_of_features:
            feature.ground_var_value_dict.update(ground_var_value_dict)
            
        return list_of_features
    
    def reset_ground_precon(self, updated_var_value_dict, ignore_objects, domain_model, problem_model):
        """Reset the grounding dictionary."""
        self.ground_var_value_dict = dict()
        # if activated attribute exists, set it to False
        if hasattr(self, 'activated'):
            self.activated = False
            assert updated_var_value_dict is None
        else:
            # means there is no activated attribute, so updated_var_value_dict must be provided
            # if grounding is required, then updated_var_value_dict must be provided
            if self.require_ground:
                assert updated_var_value_dict is not None, "updated_var_value_dict must be provided to reset grounding."
                if isinstance(self, Holding) and 'carried_object' in updated_var_value_dict and updated_var_value_dict['carried_object'][0] == self.object_type:
                    self.ground_var_value_dict[self.object_type] = updated_var_value_dict['carried_object'][1]
                  
                else:
                    self.ground_var_value_dict[self.object_type] = updated_var_value_dict[self.object_type]
        if hasattr(self, 'value_in_precon'):
            self.value_in_precon = None
            
        if hasattr(self, 'fixed_object'):
            if self.fixed_object is not None:
                self.ground_var_value_dict[self.object_type] = self.fixed_object
            
        # if self is Count feature, and count_func is provided, then we can set value_in_precon
        if isinstance(self, Count):
            self.activate_in_precon(domain_model, problem_model)
            
        elif isinstance(self, DistanceToNearest):
            assert ignore_objects is not None, "ignore_objects must be provided to reset DistanceToNearest feature."
            self.activate_in_precon(domain_model, problem_model, ignore_objects)
            
        if hasattr(self, 'activated'):
            assert self.activated, "Feature must be activated after resetting grounding."
        if hasattr(self, 'value_in_precon'):
            assert self.value_in_precon is not None, "value_in_precon must be set after resetting grounding."
            
            

    def reset_ground_effect(self, precon_feature):
        """Reset the grounding dictionary based on a precondition feature."""
        # ensure precon_feature has the same type as self
        assert type(precon_feature) == type(self), "precon_feature must be of the same type as self."
        self.ground_var_value_dict = precon_feature.ground_var_value_dict.copy()
        # if activated attribute exists, first of all, ensure precon_feature already set it to True
        if hasattr(self, 'activated'):
            assert precon_feature.activated, "precon_feature must be activated to reset effect grounding."
            self.activated = True
        if hasattr(self, 'value_in_precon'):
            self.value_in_precon = precon_feature.value_in_precon
        
    def __repr__(self):
        return self.__str__()
    

    
# holding class, require ground = True, lifted as well (require object_type)

class Holding(Feature):
    def __init__(self, type_tag, wanted_value):
        super().__init__(require_ground=True)
        assert wanted_value in [True, False], "wanted_value must be True or False"
        self.object_type = type_tag
        self.wanted_value = wanted_value  # True or False
        
    def verify(self, domain_model, problem_model):
        # ! Note that domain and problem model is depend on the env, so env must not be None
        # get domain and problem models
        assert domain_model is not None and problem_model is not None, "Domain and problem models must be provided."
        if self.force_invalid:
            return False
        # also assert the grounding dict is provided
        assert self.object_type in self.ground_var_value_dict, f"Grounding value for type {self.object_type} not provided."
        
        object_name = self.ground_var_value_dict[self.object_type]
        if " " in self.object_type:
            object_type_lst = self.object_type.split(" ")
        else:
            object_type_lst = [self.object_type]
        ver_value = False
        predicate: Predicate
        for predicate in problem_model.init: 
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) < 2:
                continue
            else:
                term = terms[1]
                if predicate_name == 'inhandofrobot' and term.name == object_name and term.type_tag in object_type_lst:
                    ver_value = True
                    break
        if ver_value == self.wanted_value:
            return True
        else:
            return False
    def __str__(self):
        if self.object_type in self.ground_var_value_dict:
            obj_name = self.ground_var_value_dict[self.object_type]
        else:
            obj_name = self.object_type
        if self.wanted_value:
            
            return f"Holding({obj_name})"
        else:
            return f"Not Holding({obj_name})"
    

                
    
class DistanceToNearest(Feature):
    def __init__(self, type_tag, wanted_value=None, value_in_precon=None, obj_name=None, consider_carried_objects=True):
        super().__init__(require_ground=True)
        self.object_type = type_tag
        if value_in_precon is not None:
            self.activated = True
            self.value_in_precon = value_in_precon
        else:
            self.activated = False 
            self.value_in_precon = None  # distance value when activated
        assert wanted_value in ['>0', '=0', 'increase', 'decrease', 'change'], "wanted_value must be one of '>0', '=0', 'increase', 'decrease'"
        
        self.wanted_value = wanted_value
        if value_in_precon is not None:
            assert obj_name is not None, "obj_name must be provided when value_in_precon is given"
            self.ground_var_value_dict[self.object_type] = obj_name
            
        if obj_name is not None and value_in_precon is None:
            self.ground_var_value_dict[self.object_type] = obj_name
            self.fixed_object = obj_name
            
        self.consider_carried_objects = consider_carried_objects

    def activate_in_precon(self, domain_model, problem_model, ignore_objects):
        if self.activated:
            raise ValueError("DistanceToNearest feature is already activated. That oftens means this feature is in the effect, not precondition.")
        agent_pos = get_agent_location(problem_model)
        calculate_closest_flag = False
        if self.object_type in self.ground_var_value_dict and not self.consider_carried_objects:
            if self.ground_var_value_dict[self.object_type]:
            # ! it means we already have a specific object to calculate distance
                obj_name = self.ground_var_value_dict[self.object_type]
                distance = calculate_distance(problem_model, 'agent-01', obj_name)
                self.value_in_precon = distance
                self.activated = True
            else:
                calculate_closest_flag = True
                
        if getattr(self, 'fixed_object', None) is not None:
            calculate_closest_flag = False
            obj_name = self.fixed_object
            distance = calculate_distance(problem_model, 'agent-01', obj_name)
            self.value_in_precon = distance
            self.activated = True
                
        if calculate_closest_flag or self.object_type not in self.ground_var_value_dict:
            closest_location, min_distance, closest_object_name, additional_info = get_closest_object_location(problem_model, self.object_type, agent_pos, ignore_objects, self.consider_carried_objects)
            # update the grounding dict
            self.ground_var_value_dict.update(additional_info)
            if closest_location is not None:
                self.ground_var_value_dict[self.object_type] = closest_object_name
                self.activated = True
                self.value_in_precon = min_distance
            else:
                self.activated = True
                self.value_in_precon = "invalid"
                self.force_invalid = True
        
    def verify(self, domain_model, problem_model):
        # call this kind of value changes 
        assert self.activated, "DistanceToNearest feature must be activated before verification."
        if self.force_invalid:
            return False
        agent_name = 'agent-01'
        object_name = self.ground_var_value_dict[self.object_type]
        distance = calculate_distance(problem_model, agent_name, object_name)
        # 5 cases
        if self.wanted_value == '>0':
            # check if the distance is > 0
            if distance > 1:
                return True
            else:
                return False
        elif self.wanted_value == '=0':
            # check if the distance is <= 1
            if distance <= 1: # include 1 as well
                return True
            else:
                return False
        elif self.wanted_value in ['increase', 'decrease', 'change']:
            assert self.value_in_precon is not None, "value_in_precon must be set for increase/decrease/change checks."
            if self.wanted_value == 'increase':
                if distance > self.value_in_precon:
                    return True
                else:
                    return False
            elif self.wanted_value == 'decrease':
                if distance < self.value_in_precon:
                    return True
                else:
                    return False
            elif self.wanted_value == 'change':
                if distance != self.value_in_precon:
                    return True
                else:
                    return False
        else:
            raise ValueError(f"Invalid wanted_value: {self.wanted_value}")
        
    def __str__(self):
        if self.object_type in self.ground_var_value_dict:
            obj_name = self.ground_var_value_dict[self.object_type]
        else:
            obj_name = self.object_type
        return f"DistanceToNearest({obj_name}, {self.wanted_value})"
        


class isOpened(Feature):
    def __init__(self, type_tag, wanted_value):
        super().__init__(require_ground=True)
        assert wanted_value in [True, False], "wanted_value must be True or False"
        self.object_type = type_tag
        self.wanted_value = wanted_value  # True or False
        
    def verify(self, domain_model, problem_model):
        # ! Note that domain and problem model is depend on the env, so env must not be None
        # get domain and problem models
        assert domain_model is not None and problem_model is not None, "Domain and problem models must be provided."
        if self.force_invalid:
            return False
        # also assert the grounding dict is provided
        assert self.object_type in self.ground_var_value_dict, f"Grounding value for type {self.object_type} not provided."
        
        object_name = self.ground_var_value_dict[self.object_type]
        ver_value = False
        predicate: Predicate
        for predicate in problem_model.init: 
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) > 1:
                continue
            else:
                term = terms[0]
                if predicate_name == 'is-opened' and term.name == object_name and term.type_tag == self.object_type:
                    ver_value = True
                    break
        if ver_value == self.wanted_value:
            return True
        else:
            return False
    def __str__(self):
        if self.wanted_value:
            return f"IsOpened({self.object_type})"
        else:
            return f"Not IsOpened({self.object_type})"
        
class isToggled(Feature):
    def __init__(self, type_tag, wanted_value):
        super().__init__(require_ground=True)
        assert wanted_value in [True, False], "wanted_value must be True or False"
        self.object_type = type_tag
        self.wanted_value = wanted_value  # True or False
        
    def verify(self, domain_model, problem_model):
        # ! Note that domain and problem model is depend on the env, so env must not be None
        # get domain and problem models
        assert domain_model is not None and problem_model is not None, "Domain and problem models must be provided."
        if self.force_invalid:
            return False
        # also assert the grounding dict is provided
        assert self.object_type in self.ground_var_value_dict, f"Grounding value for type {self.object_type} not provided."
        
        object_name = self.ground_var_value_dict[self.object_type]
        ver_value = False
        predicate: Predicate
        for predicate in problem_model.init: 
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) > 1:
                continue
            else:
                term = terms[0]
                if predicate_name == 'is-toggled' and term.name == object_name and term.type_tag == self.object_type:
                    ver_value = True
                    break
        if ver_value == self.wanted_value:
            return True
        else:
            return False
    def __str__(self):
        if self.wanted_value:
            return f"isToggled({self.object_type})"
        else:
            return f"Not isToggled({self.object_type})"
        
class isSoaked(Feature):
    def __init__(self, type_tag, wanted_value):
        super().__init__(require_ground=True)
        assert wanted_value in [True, False], "wanted_value must be True or False"
        self.object_type = type_tag
        self.wanted_value = wanted_value  # True or False
        
    def verify(self, domain_model, problem_model):
        # ! Note that domain and problem model is depend on the env, so env must not be None
        # get domain and problem models
        assert domain_model is not None and problem_model is not None, "Domain and problem models must be provided."
        if self.force_invalid:
            return False
        # also assert the grounding dict is provided
        assert self.object_type in self.ground_var_value_dict, f"Grounding value for type {self.object_type} not provided."
        
        object_name = self.ground_var_value_dict[self.object_type]
        ver_value = False
        predicate: Predicate
        for predicate in problem_model.init: 
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) > 1:
                continue
            else:
                term = terms[0]
                if predicate_name == 'is-soaked' and term.name == object_name and term.type_tag == self.object_type:
                    ver_value = True
                    break
        if ver_value == self.wanted_value:
            return True
        else:
            return False
    def __str__(self):
        if self.wanted_value:
            return f"isSoaked({self.object_type})"
        else:
            return f"Not isSoaked({self.object_type})"
        
class isDustFree(Feature):
    def __init__(self, type_tag, wanted_value):
        super().__init__(require_ground=True)
        assert wanted_value in [True, False], "wanted_value must be True or False"
        self.object_type = type_tag
        self.wanted_value = wanted_value  # True or False
        
    def verify(self, domain_model, problem_model):
        # ! Note that domain and problem model is depend on the env, so env must not be None
        # get domain and problem models
        assert domain_model is not None and problem_model is not None, "Domain and problem models must be provided."
        if self.force_invalid:
            return False
        # also assert the grounding dict is provided
        assert self.object_type in self.ground_var_value_dict, f"Grounding value for type {self.object_type} not provided."
        
        object_name = self.ground_var_value_dict[self.object_type]
        ver_value = False
        predicate: Predicate
        for predicate in problem_model.init: 
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) > 1:
                continue
            else:
                term = terms[0]
                if predicate_name == 'is-not-dusted' and term.name == object_name and term.type_tag == self.object_type:
                    ver_value = True
                    break
        if ver_value == self.wanted_value:
            return True
        else:
            return False
    def __str__(self):
        if self.wanted_value:
            return f"isDustFree({self.object_type})"
        else:
            return f"Not isDustFree({self.object_type})"

class isCleaned(Feature):
    def __init__(self, type_tag, wanted_value):
        super().__init__(require_ground=True)
        assert wanted_value in [True, False], "wanted_value must be True or False"
        self.object_type = type_tag
        self.wanted_value = wanted_value  # True or False
        
    def verify(self, domain_model, problem_model):
        # ! Note that domain and problem model is depend on the env, so env must not be None
        # get domain and problem models
        assert domain_model is not None and problem_model is not None, "Domain and problem models must be provided."
        if self.force_invalid:
            return False
        # also assert the grounding dict is provided
        assert self.object_type in self.ground_var_value_dict, f"Grounding value for type {self.object_type} not provided."
        
        object_name = self.ground_var_value_dict[self.object_type]
        ver_value = False
        predicate: Predicate
        for predicate in problem_model.init: 
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) > 1:
                continue
            else:
                term = terms[0]
                if predicate_name == 'is-not-stained' and term.name == object_name and term.type_tag == self.object_type:
                    ver_value = True
                    break
        if ver_value == self.wanted_value:
            return True
        else:
            return False
    def __str__(self):
        if self.wanted_value:
            return f"isCleaned({self.object_type})"
        else:
            return f"Not isCleaned({self.object_type})"
        
        
        
class Count(Feature):
    def __init__(self, type_tag, value_in_precon=None, wanted_value=None, count_func=None):
        """count_func: function to count objects of the given type in the problem model"""
        super().__init__(require_ground=False)
        self.object_type = type_tag
        if value_in_precon is not None:
            self.activated = True
            self.value_in_precon = value_in_precon
        else:
            self.activated = False 
            self.value_in_precon = None  # count value when activated
        assert wanted_value in ['>0', '=0', 'increase', 'decrease', 'change'], "wanted_value must be one of '>0', '=0', 'increase', 'decrease'"
        
        
        self.wanted_value = wanted_value
        self.count_func = count_func  # function to count objects of the given type in the problem model
        self.additional_info = dict()  # to store additional info from count_func

    def activate_in_precon(self, domain_model, problem_model):
        if self.activated:
            raise ValueError("Count feature is already activated. That oftens means this feature is in the effect, not precondition.")

        self.value_in_precon, self.additional_info = self.count_func(problem_model, self.object_type)
        self.activated = True
        
    def verify(self, domain_model, problem_model):
        # call this kind of value changes 
        assert self.activated, "Count feature must be activated before verification."
        count = self.count_func(problem_model, self.object_type)[0]
        # 5 cases
        if self.wanted_value == '>0':
            # check if the count is > 0
            if count > 0:
                return True
            else:
                return False
        elif self.wanted_value == '=0':
            # check if the count is = 0
            if count == 0:
                return True
            else:
                return False
        elif self.wanted_value in ['increase', 'decrease', 'change']:
            assert self.value_in_precon is not None, "value_in_precon must be set for increase/decrease/change checks."
            if self.wanted_value == 'increase':
                if count > self.value_in_precon:
                    return True
                else:
                    return False
            elif self.wanted_value == 'decrease':
                if count < self.value_in_precon:
                    return True
                else:
                    return False
            elif self.wanted_value == 'change':
                if count != self.value_in_precon:
                    return True
                else:
                    return False
        else:
            raise ValueError(f"Invalid wanted_value: {self.wanted_value}")
        
    def __str__(self):
        if hasattr(self.count_func, "__name__"):
            count_func_name = self.count_func.__name__  
            count_func_repr = count_func_name
        elif hasattr(self.count_func, "func") and hasattr(self.count_func.func, "__name__"):
            # This is a partial function
            count_func_name = self.count_func.func.__name__
            # Extract keyword arguments from the partial function
            kwargs_parts = []
            if hasattr(self.count_func, 'keywords') and self.count_func.keywords:
                for key, value in self.count_func.keywords.items():
                    kwargs_parts.append(f"{key}={value}")
            if kwargs_parts:
                count_func_repr = f"{count_func_name}({', '.join(kwargs_parts)})"
            else:
                count_func_repr = count_func_name
        else:
            count_func_name = "count_func"
            count_func_repr = count_func_name
        return f"Count({self.object_type}, {self.wanted_value}, {count_func_repr})"
    
# ---- End of Features -----

# * Create count functions
# count function input : problem model, object type
# output: count value (int), additional info (dict)

def count_isolated(problem_model, object_type):
    # steps: get all the positions of the given object type
    # for each position, check if its 4-neighbors have the same object type object, thus, we shall generate a pool of positions
    
    position_pool = []
    obj_pool = [] 
    isolated_count = 0
    isolated_objs = [] 
    not_isolated_objs = []
    
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == object_type:
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                pos_x, pos_y = int(pos_x), int(pos_y)
                position_pool.append((pos_x, pos_y))
                obj_pool.append(first_term)
        elif predicate_name == 'inhandofrobot':
            terms = predicate.terms
            object_term = terms[1]
            if object_term.type_tag == object_type:
                # directly add to isolated objects since it is in hand of robot
                isolated_count += 1
                isolated_objs.append(object_term)

    # filter same obj 
    temp_obj_pool = []
    temp_position_pool = []
    for idx, obj in enumerate(obj_pool):
        if obj not in temp_obj_pool:
            temp_obj_pool.append(obj)
            temp_position_pool.append(position_pool[idx])
    obj_pool = temp_obj_pool
    position_pool = temp_position_pool

    
    for idx, (pos_x, pos_y) in enumerate(position_pool):
        neighbors = [(pos_x+1, pos_y), (pos_x-1, pos_y), (pos_x, pos_y+1), (pos_x, pos_y-1)]
        is_isolated = True
        for neighbor in neighbors:
            if neighbor in position_pool:
                is_isolated = False
                break
        if is_isolated:
            isolated_count += 1
            isolated_objs.append(obj_pool[idx])
        else:
            not_isolated_objs.append(obj_pool[idx])
    additional_info = {
        "isolated_objs": isolated_objs,
        "not_isolated_objs": not_isolated_objs
    }
    return isolated_count, additional_info

def count_not_cleaned(problem_model, object_type):
    focus_objects = []
    if " " in object_type:
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
    predicate: Predicate
    for object in problem_model.objects:
        if object.type_tag in object_type_lst:
            focus_objects.append(object)
    not_cleaned_count = 0
    not_cleaned_objs = []
    cleaned_objs = []
    for obj in focus_objects:
        is_cleaned = False
        for predicate in problem_model.init:
            predicate_name = predicate.name
            if predicate_name == 'is-not-stained':
                terms = predicate.terms
                first_term = terms[0]
                if first_term == obj:
                    is_cleaned = True
                    break
        if not is_cleaned:
            not_cleaned_count += 1
            not_cleaned_objs.append(obj)
        else:
            cleaned_objs.append(obj)
    additional_info = {
        "not_cleaned_objs": not_cleaned_objs,
        "cleaned_objs": cleaned_objs
    }
    return not_cleaned_count, additional_info
        
def count_not_wiped(problem_model, object_type):
    focus_objects = []
    if " " in object_type:
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
    predicate: Predicate
    for object in problem_model.objects:
        if object.type_tag in object_type_lst:
            focus_objects.append(object)
    not_cleaned_count = 0
    not_cleaned_objs = []
    cleaned_objs = []
    for obj in focus_objects:
        is_cleaned = False
        for predicate in problem_model.init:
            predicate_name = predicate.name
            if predicate_name == 'is-not-dusted':
                terms = predicate.terms
                first_term = terms[0]
                if first_term == obj:
                    is_cleaned = True
                    break
        if not is_cleaned:
            not_cleaned_count += 1
            not_cleaned_objs.append(obj)
        else:
            cleaned_objs.append(obj)
    additional_info = {
        "not_cleaned_objs": not_cleaned_objs,
        "cleaned_objs": cleaned_objs
    }
    return not_cleaned_count, additional_info

def count_not_sliced(problem_model, object_type):
    # this function does not need to use object_type, it just first of all, get all objects that have sliceable tag
    # then if not is-sliced predicate exists, then count +1
    sliceable_objects = []
    if " " in object_type:
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
    
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'sliceable':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag in object_type_lst:
                sliceable_objects.append(first_term)
            
    not_sliced_count = 0
    not_sliced_objs = []
    sliced_objs = []
    for obj in sliceable_objects:
        is_sliced = False
        for predicate in problem_model.init:
            predicate_name = predicate.name
            if predicate_name == 'is-sliced':
                terms = predicate.terms
                first_term = terms[0]
                if first_term == obj:
                    is_sliced = True
                    break
        if not is_sliced:
            not_sliced_count += 1
            not_sliced_objs.append(obj)
        else:
            sliced_objs.append(obj)
    additional_info = {
        "not_sliced_objs": not_sliced_objs,
        "sliced_objs": sliced_objs
    }
    return not_sliced_count, additional_info

def count_incomplete_salad_place(problem_model, object_type):
    # a complete salad place means two cookable objects are place at the same location but dimension is either on top or middle, the plate dimension is at the bottom
    # first step, get all plate name 
    plate_objects = []
    plate_positions = []
    plate_dimensions = []
    cookable_objects = []
    sliceable_objects = []
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == 'plate':
                plate_objects.append(first_term)
                position = str(terms[1])
                position = position.split('-')[-2:]
                position = (int(position[0]), int(position[1]))
                plate_positions.append(position)
                dimension = str(terms[2])
                plate_dimensions.append(dimension)
                
        elif predicate_name == 'cookable':
            terms = predicate.terms
            first_term = terms[0]
            cookable_objects.append(first_term)
            
        elif predicate_name == 'sliceable':
            terms = predicate.terms
            first_term = terms[0]
            sliceable_objects.append(first_term)
                
    for obj in problem_model.objects:
        if obj.type_tag == 'lettuce':
            cookable_objects.append(obj)
        
    # make sure sliceable objects are also cookable
    for sliceable in sliceable_objects:
        assert sliceable in cookable_objects, f"Sliceable object {sliceable} is not in cookable objects."
            
    cookable_positions = dict()
    cookable_dimensions = dict()
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term in cookable_objects:
                position = str(terms[1])
                position = position.split('-')[-2:]
                position = (int(position[0]), int(position[1]))
                cookable_positions[first_term] = position
                dimension = str(terms[2])
                cookable_dimensions[first_term] = dimension
        elif predicate_name == 'inhandofrobot':
            terms = predicate.terms
            object_term = terms[1]
            if object_term in cookable_objects:
                # directly add to cookable positions since it is in hand of robot, but dimension is top
                position = get_agent_location(problem_model)
                cookable_positions[object_term] = position
                cookable_dimensions[object_term] = 'bottom' # in hand of robot, consider it at bottom dimension
    # convert dict back to list 
    cookable_positions_list = []
    cookable_dimensions_list = []
    for cookable in cookable_objects:
        cookable_positions_list.append(cookable_positions[cookable])
        cookable_dimensions_list.append(cookable_dimensions[cookable])
    cookable_positions = cookable_positions_list
    cookable_dimensions = cookable_dimensions_list

    incomplete_salad_place_count = 0
    incomplete_salad_place_objs = []
    complete_salad_place_objs = []
    for idx, plate in enumerate(plate_objects):
        plate_pos = plate_positions[idx]
        plate_dim = plate_dimensions[idx]
        if plate_dim != 'bottom':
            incomplete_salad_place_count += 1
            incomplete_salad_place_objs.append(plate)
            continue
        
        else:
            # check if there are two cookable objects at the same position
            cookable_at_same_pos = []
            for jdx, cookable in enumerate(cookable_objects):
                try:
                    cookable_pos = cookable_positions[jdx]
                except Exception as e:
                    breakpoint()
                    a = 1 
                cookable_dim = cookable_dimensions[jdx]
                if cookable_pos == plate_pos and cookable_dim in ['top', 'middle']:
                    cookable_at_same_pos.append(cookable)
            if len(cookable_at_same_pos) >= 2:
                # final check, check those cookable object if they are sliceable, make sure they have is-sliced predicate
                all_sliceable_sliced = True
                for cookable in cookable_at_same_pos:
                    if cookable in sliceable_objects:
                        # check if is-sliced predicate exists
                        is_sliced = False
                        for predicate in problem_model.init:
                            predicate_name = predicate.name
                            if predicate_name == 'is-sliced':
                                terms = predicate.terms
                                first_term = terms[0]
                                if first_term == cookable:
                                    is_sliced = True
                                    break
                        if not is_sliced:
                            all_sliceable_sliced = False
                            break
                if all_sliceable_sliced:
                    complete_salad_place_objs.append(plate)
                else:
                    incomplete_salad_place_count += 1
                    incomplete_salad_place_objs.append(plate)
            else:
                incomplete_salad_place_count += 1
                incomplete_salad_place_objs.append(plate)
                
    additional_info = {
        "incomplete_salad_place_objs": incomplete_salad_place_objs,
        "complete_salad_place_objs": complete_salad_place_objs,
        "cookable_objects": cookable_objects,
        "cookable_positions": cookable_positions,
        "cookable_dimensions": cookable_dimensions,
        "plate_objects": plate_objects,
        "plate_positions": plate_positions,
        "plate_dimensions": plate_dimensions
    }
    return incomplete_salad_place_count, additional_info
                    

def count_closed(problem_model, object_type):
    closed_count = 0
    focus_objects = []
    if " " in object_type:
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
        
    for obj in problem_model.objects:
        if obj.type_tag in object_type_lst:
            focus_objects.append(obj)
    closed_objs = []
    not_closed_objs = []
    predicate: Predicate
    for focus_obj in focus_objects:
        is_closed = True
        for predicate in problem_model.init:
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) > 1:
                continue
            else:
                term = terms[0]
                if predicate_name == 'is-opened' and term == focus_obj:
                    is_closed = False
                    break
        if is_closed:
            closed_count += 1
            closed_objs.append(focus_obj)
        else:
            not_closed_objs.append(focus_obj)
    additional_info = {
        "closed_objs": closed_objs,
        "not_closed_objs": not_closed_objs
    }
    return closed_count, additional_info

def count_not_inside_target_object(problem_model, object_type, container_obj_name, additional_predicate_on_object_type=None):
    # this is a count of objects of a certain type that are not inside a specific container
    not_inside_count = 0
    focus_objects = []
    if " " in object_type:
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
    for obj in problem_model.objects:
        if obj.type_tag in object_type_lst:
            focus_objects.append(obj)
    
    # check container's which dimension is containable
    container_containable_dimensions = [] 
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'containable':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.name == container_obj_name:
                dimension = str(terms[1])
                container_containable_dimensions.append(dimension)
                
    # get container position
    container_position = [] # container can cover multiple positions
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.name == container_obj_name:
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                pos_x, pos_y = int(pos_x), int(pos_y)
                container_position.append((pos_x, pos_y))
    
                
    inside_objs = []
    # count how many focus_objects are inside the container and also satisfy the additional predicate if provided
    for focus_obj in focus_objects:
        for predicate in problem_model.init:
            predicate_name = predicate.name
            if predicate_name == 'at':
                terms = predicate.terms
                first_term = terms[0]
                if first_term == focus_obj:
                    location = str(terms[1])
                    pos_x, pos_y = location.split('-')[-2:]
                    pos_x, pos_y = int(pos_x), int(pos_y)
                    dimension = str(terms[2])
                    if (pos_x, pos_y) in container_position and dimension in container_containable_dimensions:
                        if additional_predicate_on_object_type is not None:
                            # check if the additional predicate holds
                            additional_holds = False
                            for pred in problem_model.init:
                                pred_name = pred.name
                                if pred_name == additional_predicate_on_object_type:
                                    terms = pred.terms
                                    first_term = terms[0]
                                    if first_term == focus_obj:
                                        additional_holds = True
                                        break
                            if additional_holds:
                                inside_objs.append(focus_obj)
                                break
                            else:
                                continue
                        else:
                            inside_objs.append(focus_obj)
                            break
                    else:
                        continue
                else:
                    continue
                
    not_inside_count = len(focus_objects) - len(inside_objs)
    not_inside_objs = [obj for obj in focus_objects if obj not in inside_objs]
    additional_info = {
        "not_inside_objs": not_inside_objs,
        "inside_objs": inside_objs,
        "container_containable_dimensions": container_containable_dimensions,
        "container_position": list(set(container_position))
    }
    return not_inside_count, additional_info

def count_not_ontop(problem_model, object_type, surface_obj_name, additional_predicate_on_object_type=None):
    not_ontop_count = 0
    focus_objects = []
    if " " in object_type:
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
    for obj in problem_model.objects:
        if obj.type_tag in object_type_lst:
            focus_objects.append(obj)
            
    # get surface position
    surface_position = [] # surface can cover multiple positions
    surface_dimension = None
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.name == surface_obj_name:
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                pos_x, pos_y = int(pos_x), int(pos_y)
                surface_position.append((pos_x, pos_y))
                surface_dimension = str(terms[2])
                continue

    assert surface_dimension is not None, f"Surface object {surface_obj_name} not found in the problem model."
    if surface_dimension == 'bottom':
        valid_obj_dimensions = ['middle', 'top']
    elif surface_dimension == 'middle':
        valid_obj_dimensions = ['top']
    else:
        raise ValueError(f"Surface object {surface_obj_name} has invalid dimension {surface_dimension}.")
                
    ontop_objs = []
    # count how many focus_objects are ontop the surface and also satisfy the additional predicate if provided
    for focus_obj in focus_objects:
        for predicate in problem_model.init:
            predicate_name = predicate.name
            if predicate_name == 'at':
                terms = predicate.terms
                first_term = terms[0]
                if first_term == focus_obj:
                    location = str(terms[1])
                    pos_x, pos_y = location.split('-')[-2:]
                    pos_x, pos_y = int(pos_x), int(pos_y)
                    dimension = str(terms[2])
                    if (pos_x, pos_y) in surface_position and dimension in valid_obj_dimensions:
                        if additional_predicate_on_object_type is not None:
                            # check if the additional predicate holds
                            additional_holds = False
                            for pred in problem_model.init:
                                pred_name = pred.name
                                if pred_name == additional_predicate_on_object_type:
                                    terms = pred.terms
                                    first_term = terms[0]
                                    if first_term == focus_obj:
                                        additional_holds = True
                                        break
                            if additional_holds:
                                ontop_objs.append(focus_obj)
                                break
                            else:
                                continue
                        else:
                            ontop_objs.append(focus_obj)
                            break
                    else:
                        continue
                else:
                    continue
                
    not_ontop_count = len(focus_objects) - len(ontop_objs)
    not_ontop_objs = [obj for obj in focus_objects if obj not in ontop_objs]
    additional_info = {
        "not_ontop_objs": not_ontop_objs,
        "ontop_objs": ontop_objs,
        "surface_position": list(set(surface_position)),
        "surface_dimension": surface_dimension
    }
    return not_ontop_count, additional_info


def count_not_ontop_sometype(problem_model, object_type, surface_type, additional_predicate_on_object_type=None):
    not_ontop_count = 0
    focus_objects = []
    if " " in object_type:
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
    for obj in problem_model.objects:
        if obj.type_tag in object_type_lst:
            focus_objects.append(obj)
            
    # get surface position
    surface_position = [] # surface can cover multiple positions
    surface_dimension = None
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == surface_type:
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                pos_x, pos_y = int(pos_x), int(pos_y)
                surface_position.append((pos_x, pos_y))
                surface_dimension = str(terms[2])
                continue

    assert surface_dimension is not None, f"Surface type {surface_type} not found in the problem model."
    if surface_dimension == 'bottom':
        valid_obj_dimensions = ['middle', 'top']
    elif surface_dimension == 'middle':
        valid_obj_dimensions = ['top']
    else:
        raise ValueError(f"Surface object {surface_type} has invalid dimension {surface_dimension}.")
                
    ontop_objs = []
    # count how many focus_objects are ontop the surface and also satisfy the additional predicate if provided
    for focus_obj in focus_objects:
        for predicate in problem_model.init:
            predicate_name = predicate.name
            if predicate_name == 'at':
                terms = predicate.terms
                first_term = terms[0]
                if first_term == focus_obj:
                    location = str(terms[1])
                    pos_x, pos_y = location.split('-')[-2:]
                    pos_x, pos_y = int(pos_x), int(pos_y)
                    dimension = str(terms[2])
                    if (pos_x, pos_y) in surface_position and dimension in valid_obj_dimensions:
                        if additional_predicate_on_object_type is not None:
                            # check if the additional predicate holds
                            additional_holds = False
                            for pred in problem_model.init:
                                pred_name = pred.name
                                if pred_name == additional_predicate_on_object_type:
                                    terms = pred.terms
                                    first_term = terms[0]
                                    if first_term == focus_obj:
                                        additional_holds = True
                                        break
                            if additional_holds:
                                ontop_objs.append(focus_obj)
                                break
                            else:
                                continue
                        else:
                            ontop_objs.append(focus_obj)
                            break
                    else:
                        continue
                else:
                    continue
                
    not_ontop_count = len(focus_objects) - len(ontop_objs)
    not_ontop_objs = [obj for obj in focus_objects if obj not in ontop_objs]
    additional_info = {
        "not_ontop_objs": not_ontop_objs,
        "ontop_objs": ontop_objs,
        "surface_position": list(set(surface_position)),
        "surface_dimension": surface_dimension
    }
    return not_ontop_count, additional_info


def count_not_near_target_type(problem_model, object_type, target_type_name, include_inside=False):
    # get target positions
    target_positions = []
    target_containable_dimensions = []
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == target_type_name:
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                pos_x, pos_y = int(pos_x), int(pos_y)
                target_positions.append((pos_x, pos_y))
        elif predicate_name == 'containable':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == target_type_name:
                dimension = str(terms[1])
                target_containable_dimensions.append(dimension)

    target_positions = list(set(target_positions))
    target_containable_dimensions = list(set(target_containable_dimensions))


    assert len(target_positions) > 0, "No target found in the problem model."

    not_near_target_count = 0
    focus_objects = []
    if " " in object_type:
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
    for obj in problem_model.objects:
        if obj.type_tag in object_type_lst:
            focus_objects.append(obj)
            # we collect all positions that are near the target, also include the target position itself, also with dimension, but for target_position, we only want the containable dimensions

    valid_pos_and_dims = set()
    for target_pos in target_positions:
        pos_x, pos_y = target_pos
        neighbors = [(pos_x+1, pos_y), (pos_x-1, pos_y), (pos_x, pos_y+1), (pos_x, pos_y-1)]
        for neighbor in neighbors:
            valid_pos_and_dims.add((neighbor, 'bottom'))
            valid_pos_and_dims.add((neighbor, 'middle'))
            valid_pos_and_dims.add((neighbor, 'top'))
        if include_inside:
            for dim in target_containable_dimensions:
                valid_pos_and_dims.add((target_pos, dim))
                
    if not include_inside:
        # remove any positions that are exactly the target positions
        for target_pos in target_positions:
            valid_pos_and_dims.discard((target_pos, 'bottom'))
            valid_pos_and_dims.discard((target_pos, 'middle'))
            valid_pos_and_dims.discard((target_pos, 'top'))

    near_target_objs = []
    for focus_obj in focus_objects:
        for predicate in problem_model.init:
            predicate_name = predicate.name
            if predicate_name == 'at':
                terms = predicate.terms
                first_term = terms[0]
                if first_term == focus_obj:
                    location = str(terms[1])
                    pos_x, pos_y = location.split('-')[-2:]
                    pos_x, pos_y = int(pos_x), int(pos_y)
                    dimension = str(terms[2])
                    if ((pos_x, pos_y), dimension) in valid_pos_and_dims:
                        near_target_objs.append(focus_obj)
                        break
                    else:
                        continue
                else:
                    continue
    not_near_target_count = len(focus_objects) - len(near_target_objs)
    not_near_target_objs = [obj for obj in focus_objects if obj not in near_target_objs]
    additional_info = {
        "not_near_target_objs": not_near_target_objs,
        "near_target_objs": near_target_objs,
        "target_positions": target_positions,
        "target_containable_dimensions": target_containable_dimensions,
        "valid_pos_and_dims": list(valid_pos_and_dims)
    }
    return not_near_target_count, additional_info

def count_not_soaked(problem_model, object_type):
    not_soaked_count = 0
    focus_objects = []
    for obj in problem_model.objects:
        if obj.type_tag == object_type:
            focus_objects.append(obj)
    not_soaked_objs = []
    soaked_objs = []
    predicate: Predicate
    for focus_obj in focus_objects:
        is_soaked = False
        for predicate in problem_model.init:
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) > 1:
                continue
            else:
                term = terms[0]
                if predicate_name == 'is-soaked' and term == focus_obj:
                    is_soaked = True
                    break
        if not is_soaked:
            not_soaked_count += 1
            not_soaked_objs.append(focus_obj)
        else:
            soaked_objs.append(focus_obj)
    additional_info = {
        "not_soaked_objs": not_soaked_objs,
        "soaked_objs": soaked_objs
    }
    return not_soaked_count, additional_info

def count_not_inside_target_type(problem_model, object_type, target_type_name, additional_predicate_on_object_type=None):
    # get target positions
    target_positions = []
    target_containable_dimensions = []
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == target_type_name:
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                pos_x, pos_y = int(pos_x), int(pos_y)
                target_positions.append((pos_x, pos_y))
        elif predicate_name == 'containable':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == target_type_name:
                dimension = str(terms[1])
                target_containable_dimensions.append(dimension)

    target_positions = list(set(target_positions))
    target_containable_dimensions = list(set(target_containable_dimensions))

    assert len(target_containable_dimensions) > 0, "No target containable dimension found in the problem model."
    assert len(target_positions) > 0, "No target found in the problem model."

    not_inside_count = 0
    focus_objects = []
    if " " in object_type:
        object_type_lst = object_type.split(" ")
    else:
        object_type_lst = [object_type]
    for obj in problem_model.objects:
        if obj.type_tag in object_type_lst:
            focus_objects.append(obj)
            # we collect all positions that are near the target, also include the target position itself, also with dimension, but for target_position, we only want the containable dimensions

    inside_objs = []
    for focus_obj in focus_objects:
        for predicate in problem_model.init:
            predicate_name = predicate.name
            if predicate_name == 'at':
                terms = predicate.terms
                first_term = terms[0]
                if first_term == focus_obj:
                    location = str(terms[1])
                    pos_x, pos_y = location.split('-')[-2:]
                    pos_x, pos_y = int(pos_x), int(pos_y)
                    dimension = str(terms[2])
                    if (pos_x, pos_y) in target_positions and dimension in target_containable_dimensions:
                        if additional_predicate_on_object_type is not None:
                            # check if the additional predicate holds
                            additional_holds = False
                            for pred in problem_model.init:
                                pred_name = pred.name
                                if pred_name == additional_predicate_on_object_type:
                                    terms = pred.terms
                                    first_term = terms[0]
                                    if first_term == focus_obj:
                                        additional_holds = True
                                        break
                            if additional_holds:
                                inside_objs.append(focus_obj)
                                break
                            else:
                                continue
                        else:
                            inside_objs.append(focus_obj)
                            break
                    else:
                        continue
                else:
                    continue
    not_inside_count = len(focus_objects) - len(inside_objs)
    not_inside_objs = [obj for obj in focus_objects if obj not in inside_objs]
    additional_info = {
        "not_inside_objs": not_inside_objs,
        "inside_objs": inside_objs,
        "target_positions": target_positions,
        "target_containable_dimensions": target_containable_dimensions
    }
    return not_inside_count, additional_info

def count_not_toggled(problem_model, object_type):
    not_toggled_count = 0
    focus_objects = []
    for obj in problem_model.objects:
        if obj.type_tag == object_type:
            focus_objects.append(obj)
    not_toggled_objs = []
    toggled_objs = []
    predicate: Predicate
    for focus_obj in focus_objects:
        is_toggled = False
        for predicate in problem_model.init:
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) > 1:
                continue
            else:
                term = terms[0]
                if predicate_name == 'is-toggled' and term == focus_obj:
                    is_toggled = True
                    break
        if not is_toggled:
            not_toggled_count += 1
            not_toggled_objs.append(focus_obj)
        else:
            toggled_objs.append(focus_obj)
    additional_info = {
        "not_toggled_objs": not_toggled_objs,
        "toggled_objs": toggled_objs
    }
    return not_toggled_count, additional_info

def count_not_samelocation_sometype(problem_model, object_type, target_type_name, additional_predicate_on_object_type=None):
    target_type_positions = []
    focus_objects = []
    for obj in problem_model.objects:
        if obj.type_tag == object_type:
            focus_objects.append(obj)
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == target_type_name:
                location = str(terms[1])
                pos_x, pos_y = location.split('-')[-2:]
                pos_x, pos_y = int(pos_x), int(pos_y)
                target_type_positions.append((pos_x, pos_y))
                
    # loop over focus_objects, check if they are at the same location as any of the target_type_positions
    not_samelocation_count = 0
    not_samelocation_objs = []
    samelocation_objs = []
    for focus_obj in focus_objects:
        is_samelocation = False
        for predicate in problem_model.init:
            predicate_name = predicate.name
            if predicate_name == 'at':
                terms = predicate.terms
                first_term = terms[0]
                if first_term == focus_obj:
                    location = str(terms[1])
                    pos_x, pos_y = location.split('-')[-2:]
                    pos_x, pos_y = int(pos_x), int(pos_y)
                    if (pos_x, pos_y) in target_type_positions:
                        if additional_predicate_on_object_type is not None:
                            # check if the additional predicate holds
                            additional_holds = False
                            for pred in problem_model.init:
                                pred_name = pred.name
                                if pred_name == additional_predicate_on_object_type:
                                    terms = pred.terms
                                    first_term = terms[0]
                                    if first_term == focus_obj:
                                        additional_holds = True
                                        break
                            if additional_holds:
                                is_samelocation = True
                                break
                            else:
                                continue
                        else:
                            is_samelocation = True
                            break
                    else:
                        continue
                else:
                    continue
        if not is_samelocation:
            not_samelocation_count += 1
            not_samelocation_objs.append(focus_obj)
        else:
            samelocation_objs.append(focus_obj)
    additional_info = {
        "not_samelocation_objs": not_samelocation_objs,
        "samelocation_objs": samelocation_objs
    }
    return not_samelocation_count, additional_info

def count_not_onfloor(problem_model, object_type):
    not_onfloor_count = 0
    focus_objects = []
    for obj in problem_model.objects:
        if obj.type_tag == object_type:
            focus_objects.append(obj)
    not_onfloor_objs = []
    onfloor_objs = []
    predicate: Predicate
    for focus_obj in focus_objects:
        is_onfloor = True
        for predicate in problem_model.init:
            predicate_name = predicate.name
            terms = predicate.terms
            if len(terms) < 2:
                continue
            else:
                term = terms[1]
                if predicate_name == 'inhandofrobot' and term == focus_obj:
                    is_onfloor = False
                    break
        if not is_onfloor:
            not_onfloor_count += 1
            not_onfloor_objs.append(focus_obj)
        else:
            onfloor_objs.append(focus_obj)
    additional_info = {
        "not_onfloor_objs": not_onfloor_objs,
        "onfloor_objs": onfloor_objs
    }
    return not_onfloor_count, additional_info
     

def count_not_complete_first_layer_salad(problem_model, type_tag):
    """Count function to check if there is at least one plate with first layer salad (lettuce or radish)"""
    assert type_tag == 'plate', "type_tag must be 'plate' for CompleteFirstLayerSalad"
    # get plate locations
    plate_objs = []
    plate_locations = []
    plate_dimensions = [] 
    salad_locations = []
    salad_dimensions = []
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == 'plate':
                plate_objs.append(first_term.name)
                location = str(terms[1])
                location = location.split('-')[-2:]
                plate_locations.append((int(location[0]), int(location[1])))
                dimension = terms[2].name
                plate_dimensions.append(dimension)
            elif first_term.type_tag in ['lettuce', 'radish']:
                location = str(terms[1])
                location = location.split('-')[-2:]
                salad_locations.append((int(location[0]), int(location[1])))
                dimension = terms[2].name
                salad_dimensions.append(dimension)
                
    not_complete_number = 0
    not_complete_plates = []
    completed_plates = []
    for idx, plate_obj in enumerate(plate_objs):
        plate_location = plate_locations[idx]
        plate_dimension = plate_dimensions[idx]
        if plate_dimension != 'bottom':
            not_complete_number += 1
            not_complete_plates.append(plate_obj)
            continue
        # check if there is any salad on this plate and salad dimension is middle 
        has_salad = False
        for s_idx, salad_location in enumerate(salad_locations):
            salad_dimension = salad_dimensions[s_idx]
            if salad_location == plate_location and salad_dimension == 'middle':
                has_salad = True
                break
        if not has_salad:
            not_complete_number += 1
            not_complete_plates.append(plate_obj)
        else:
            completed_plates.append(plate_obj)
            
    additional_info = {
        'not_complete_plates': not_complete_plates,
        'completed_plates': completed_plates,
        'plate_locations': plate_locations,
        'plate_dimensions': plate_dimensions,
        'plate_objs': plate_objs,
        'salad_locations': salad_locations,
        'salad_dimensions': salad_dimensions,
    }

    return not_complete_number, additional_info


def count_not_complete_tables(problem_model, type_tag):
    """Count function to check number of tables that do not have 3 candles on them."""
    assert type_tag == 'table', "type_tag must be 'table' for NotCompleteCandleTable"
    # get table locations
    table_objs = []
    table_locations = []
    table_dimensions = [] 
    candle_locations = []
    candle_dimensions = []
    predicate: Predicate
    for predicate in problem_model.init:
        predicate_name = predicate.name
        if predicate_name == 'at':
            terms = predicate.terms
            first_term = terms[0]
            if first_term.type_tag == 'table':
                table_objs.append(first_term.name)
                location = str(terms[1])
                location = location.split('-')[-2:]
                table_locations.append((int(location[0]), int(location[1])))
                dimension = terms[2].name
                table_dimensions.append(dimension)
            elif first_term.type_tag == 'candle':
                location = str(terms[1])
                location = location.split('-')[-2:]
                candle_locations.append((int(location[0]), int(location[1])))
                dimension = terms[2].name
                candle_dimensions.append(dimension)
    table_obj_set = set(table_objs)
    not_complete_number = 0
    not_complete_tables = []
    completed_tables = []
    table_candle_count = dict()
    for table_obj in table_obj_set:
        table_candle_count[table_obj] = 0
    
    # loop over table locations and check how many candles are on each table
    for tab_idx, table_location in enumerate(table_locations):
        for idx, candle_location in enumerate(candle_locations):
            candle_dimension = candle_dimensions[idx]
            if candle_location == table_location and candle_dimension == 'top':
                # that means we can find the associated table obj
                table_obj = table_objs[tab_idx]
                if table_obj not in table_candle_count:
                    table_candle_count[table_obj] = 0
                table_candle_count[table_obj] += 1
                
    for table_obj, candle_count in table_candle_count.items():
        if candle_count < 3:
            not_complete_number += 1
            not_complete_tables.append(table_obj)
        else:
            completed_tables.append(table_obj)
            
    additional_info = {
        'not_complete_tables': not_complete_tables,
        'completed_tables': completed_tables,
        'table_locations': table_locations,
        'table_dimensions': table_dimensions,
        'candle_locations': candle_locations,
        'candle_dimensions': candle_dimensions,
    }
    return not_complete_number, additional_info


if __name__ == "__main__":
    problem_filepath = "/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_problems/preparing_salad/test.pddl"
    
    domain_filepath = "/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/minibehavior.pddl"
    
    # test incomplete salad place count
    domain_model, problem_model = obtain_domain_and_problem_instance(domain_filepath, problem_filepath)
    count, info = count_incomplete_salad_place(problem_model, 'plate')
    print(f"incomplete salad place count: {count}")
    print(info)