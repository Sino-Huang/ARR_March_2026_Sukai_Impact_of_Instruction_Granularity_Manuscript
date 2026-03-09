import os 
from pathlib import Path
import json
import pickle
from typing import Dict, List, Optional, Tuple, Sequence, Iterable
from collections import Counter
import itertools
from collections import defaultdict
from gym_minigrid.wrappers import *
import re
import argparse
import random
import numpy as np
import numpy.random as npr
from mini_behavior.window import Window, redraw, reset, key_handler_primitive
from functools import partial


OBJ_ACTION_JSON_FP = Path(os.path.join(os.path.abspath(__file__))).parent.parent / 'object_actions.json'

assert OBJ_ACTION_JSON_FP.exists(), f"Object action JSON file not found at {OBJ_ACTION_JSON_FP}"

FURNITURE_TYPES = [
    "ashcan",
    "bed",
    "bin",
    "box",
    "bucket",
    "cabinet",
    "car",
    "chair",
    "countertop",
    "door",
    "electric_refrigerator",
    "wall",
    "shelf",
    "shower",
    "sink",
    "sofa",
    "stove",
    "table",
]

GOAL_CLAUSES = {
    "laying_wood_floors": """
(forall (?x - plywood)
    (and 
        (onfloor ?x)
        (exists (?y - plywood)
            (nextto ?x ?y)
        )
    )
)
""",
    "preparing_salad": """
(forall (?a - apple)
    (and
        (is-sliced ?a)
        (exists (?p - plate)
            (onTop ?a ?p)
        )
    )
)
(forall (?a - tomato)
    (and
        (is-sliced ?a)
        (exists (?p - plate)
            (onTop ?a ?p)
        )
    )
)
(forall (?a - radish)
    (exists (?p - plate)
        (onTop ?a ?p)
    )
)
(forall (?a - lettuce)
    (exists (?p - plate)
        (onTop ?a ?p)
    )
)
(forall (?p - plate)
    (exists (?o - normal-item)
        (and
            (cookable ?o)
            (onTop ?o ?p)
        )
    )
)
""",
    "cleaning_up_the_kitchen_only": """
(forall (?b - blender)
    (exists (?ct - countertop)
        (onTop ?b ?ct)
    )
)
(forall (?a - apple)
    (exists (?e - electric_refrigerator ?loc - location ?dim - dimension)
        (inside ?a ?e ?loc ?dim)
    )
)
(forall (?a - casserole)
    (exists (?e - electric_refrigerator ?loc - location ?dim - dimension)
        (inside ?a ?e ?loc ?dim)
    )
)
(forall (?s - soap)
    (exists (?si - sink)
        (nextto ?s ?si)
    )
)
(forall (?p - plate)
    (is-not-stained ?p)
)
(forall (?c - cabinet)
    (is-not-dusted ?c)
)
(forall (?r - rag)
    (exists (?s - sink)
        (or
            (nextto ?r ?s)
            (exists (?loc - location ?dim - dimension)
                (inside ?r ?s ?loc ?dim)
            )
        )
    )
)
(forall (?p - plate ?v - vegetable_oil)
   (exists (?loc1 ?loc2 - location ?dim1 ?dim2 - dimension ?c1 ?c2 - cabinet)
      (and
          (inside ?p ?c1 ?loc1 ?dim1)
          (inside ?v ?c2 ?loc2 ?dim2)
          (not (= ?c1 ?c2))
      )
   )
)
""",
    "organizing_file_cabinet": """
(exists (?m - marker ?t - table)
    (onTop ?m ?t)
)
(forall (?f - folder)
    (exists (?c - cabinet ?loc - location ?dim - dimension)
        (inside ?f ?c ?loc ?dim)
    )
)
(forall (?f - document)
    (exists (?c - cabinet ?loc - location ?dim - dimension)
        (inside ?f ?c ?loc ?dim)
    )
)
""",
    "thawing_frozen_food": """
(exists (?f - fish ?d - date)
    (nextto ?d ?f)
)
(forall (?f - fish)
    (exists (?s - sink)
        (nextto ?f ?s)
    )
)
(exists (?o - olive ?s - sink)
    (nextto ?o ?s)
)
""",
    "making_tea": """
(exists (?t - teapot ?tb - tea_bag ?l - lemon ?s - stove)
    (and
        (is-sliced ?l)
        (onTop ?t ?s)
        (atsamelocation ?tb ?t)
        (is-soaked ?tb)
        (is-toggled ?s)
    )
)
""",
    "opening_packages": """
(forall (?p - package)
    (is-opened ?p)
)
""",
    "boxing_books_up_for_storage": """
(forall (?b - book)
    (exists (?bo - box ?loc - location ?dim - dimension)
        (inside ?b ?bo ?loc ?dim)
    )
)
""",
    "collect_misplaced_items": """
(forall (?g - gym_shoe)
    (exists (?t - table)
        (onTop ?g ?t)
    )
)
(forall (?g - necklace)
    (exists (?t - table)
        (onTop ?g ?t)
    )
)
(forall (?g - notebook)
    (exists (?t - table)
        (onTop ?g ?t)
    )
)
(forall (?g - sock)
    (exists (?t - table)
        (onTop ?g ?t)
    )
)
""",
    "putting_away_dishes_after_cleaning": """
(forall (?p - plate)
    (exists (?c - cabinet ?loc - location ?dim - dimension)
        (inside ?p ?c ?loc ?dim)
    )
)
""",
    "washing_pots_and_pans": """
(forall (?t - teapot)
    (and
        (is-not-stained ?t)
        (exists (?c - cabinet ?loc - location ?dim - dimension)
            (inside ?t ?c ?loc ?dim)
        )
    )
)
(forall (?t - kettle)
    (and
        (is-not-stained ?t)
        (exists (?c - cabinet ?loc - location ?dim - dimension)
            (inside ?t ?c ?loc ?dim)
        )
    )
)
(forall (?t - pan)
    (and
        (is-not-stained ?t)
        (exists (?c - cabinet ?loc - location ?dim - dimension)
            (inside ?t ?c ?loc ?dim)
        )
    )
)
""",
    "cleaning_shoes": """
(exists (?t - towel)
    (onfloor ?t)
)
(forall (?s - shoe)
    (and
        (is-not-stained ?s)
        (is-not-dusted ?s)
    )
)
""",
    "installing_a_printer": """
(exists (?p - printer ?t - table)
    (and
        (onTop ?p ?t)
        (is-toggled ?p)
    )
)
""",
    "setting_up_candles": """
;; (forall (?t - table)
;;     (exists (?c1 ?c2 ?c3 - candle)
;;         (and
;;             (onTop ?c1 ?t)
;;             (onTop ?c2 ?t)
;;             (onTop ?c3 ?t)
;;             (not (= ?c1 ?c2))
;;             (not (= ?c1 ?c3))
;;             (not (= ?c2 ?c3))
;;         )
;;     )
;; )
  (onTop candle_0 table_0)
  (onTop candle_1 table_0)
  (onTop candle_2 table_0)
  (onTop candle_3 table_1)
  (onTop candle_4 table_1)
  (onTop candle_5 table_1)
""",
    "watering_houseplants": """
(forall (?p - pot_plant)
    (is-soaked ?p)
)
""",
    "cleaning_a_car": """
(exists (?c - car)
    (is-not-dusted ?c)
)
(exists (?r - rag ?s - soap ?b - bucket ?loc ?loc2 - location ?dim - dimension ?otherdim - dimension)
    (and
        (inside ?r ?b ?loc ?dim)
        (inside ?s ?b ?loc2 ?otherdim)
    )
)
""",
    "storing_food": """
(forall (?x - normal-item)
  (imply (cookable ?x)
      (exists (?c - cabinet ?loc - location ?dim - dimension)
          (inside ?x ?c ?loc ?dim)
      )
  )
)
""",
    "throwing_away_leftovers": """
(forall (?h - hamburger)
    (exists (?a - ashcan ?loc - location ?dim - dimension)
        (inside ?h ?a ?loc ?dim)
    )
)
""",
    "moving_boxes_to_storage": """
(forall (?x - carton)
    (and 
        (exists (?c - shelf ?loc - location ?dim - dimension)
            (inside ?x ?c ?loc ?dim)
        )
    )
)
""",
    "sorting_books": """
(forall (?x - book)
    (exists (?s - shelf)
        (onTop ?x ?s)
    )
)
(forall (?x - hardback)
    (exists (?s - shelf)
        (onTop ?x ?s)
    )
)
"""
}

GYM_ENV_REGISTER_DICT = {
    "laying_wood_floors": ['MiniGrid-LayingWoodFloors-8x8-N2-v0'], 
    "preparing_salad": ['MiniGrid-PreparingSalad-12x12-N2-v0'], 
    "cleaning_up_the_kitchen_only": ['MiniGrid-CleaningUpTheKitchenOnly-13x13-N2-v0'], # * it can take 140 seconds to find satisfying plan 
    "organizing_file_cabinet": ['MiniGrid-OrganizingFileCabinet-10x10-N2-v0'],  
    "thawing_frozen_food": ['MiniGrid-ThawingFrozenFood-12x12-N2-v0'],  
    "making_tea": ['MiniGrid-MakingTea-12x12-N2-v0'],   
    "opening_packages": ['MiniGrid-OpeningPackages-8x8-N2-v0'],   
    "boxing_books_up_for_storage": ['MiniGrid-BoxingBooksUpForStorage-12x12-N2-v0'],   
    "collect_misplaced_items": ['MiniGrid-CollectMisplacedItems-14x14-N2-v0'],  
    "putting_away_dishes_after_cleaning": ['MiniGrid-PuttingAwayDishesAfterCleaningDense-11x11-N2-v0'],
    "washing_pots_and_pans": ['MiniGrid-WashingPotsAndPansDense-14x14-N2-v0'], # * it takes around 95 seconds to find satisfying plan
    "cleaning_shoes": ['MiniGrid-CleaningShoes-12x12-N2-v0'],   #* it takes 65 seconds to find satisfying plan
    "installing_a_printer": ['MiniGrid-InstallingAPrinter-8x8-N2-v0'],   
    "setting_up_candles": ['MiniGrid-SettingUpCandles-10x10-N2-v0'],  # * it takes 140 seconds, very slow
    "watering_houseplants": ['MiniGrid-WateringHouseplants-8x8-N2-v0'],  
    "cleaning_a_car": ['MiniGrid-CleaningACar-14x14-N2-v0'],
    "storing_food": ['MiniGrid-StoringFood-11x11-N2-v0'],
    "throwing_away_leftovers": ['MiniGrid-ThrowingAwayLeftovers-12x12-N2-v0'],   
    "moving_boxes_to_storage": ['MiniGrid-MovingBoxesToStorage-8x8-N2-v0'],  
    "sorting_books": ['MiniGrid-SortingBooks-10x10-N2-v0']
}

PROPERTY_FORWARD_DICT = {
    "cookable": ['apple', 'banana', 'beef', 'bread', 'cake', 'candy', 'casserole', 'chicken', 'chip', 'cookie', 'date', 'egg', 'fish', 'hamburger', 'lemon', 'oatmeal', 'olive', 'radish', 'salad', 'sandwich', 'soup', 'strawberry', 'sugar', 'tomato', 'vegetable_oil'],
    "freezable": ['apple', 'banana', 'beef', 'bread', 'cake', 'casserole', 'chicken', 'cookie', 'date', 'egg', 'fish', 'hamburger', 'juice', 'lemon', 'lettuce', 'oatmeal', 'olive', 'pop', 'radish', 'salad', 'sandwich', 'soup', 'strawberry', 'tomato', 'vegetable_oil', 'water'],
    "sliceable": ['apple', 'lemon', 'strawberry', 'tomato'],
    "dustyable": ['ashcan', 'ball', 'banana', 'basket', 'bed', 'bin', 'blender', 'book', 'bow', 'box', 'broom', 'bucket', 'cabinet', 'calculator', 'candle', 'car', 'carton', 'carving_knife', 'chair', 'countertop', 'document', 'door', 'dustpan', 'electric_refrigerator', 'floor', 'folder', 'fork', 'gym_shoe', 'hardback', 'hammer', 'highlighter', 'jewelry', 'kettle', 'knife', 'marker', 'necklace', 'notebook', 'package', 'pan', 'pen', 'pencil', 'plate', 'plywood', 'printer', 'saw', 'scrub_brush', 'shelf', 'shoe', 'sink', 'sock', 'sofa', 'spoon', 'stove', 'table', 'tea_bag', 'teapot', 'toilet', 'window'],
    "stainable": ['ashcan', 'ball', 'banana', 'basket', 'bed', 'bin', 'blender', 'bucket', 'cabinet', 'car', 'carving_knife', 'casserole', 'chair', 'countertop', 'door', 'dustpan', 'electric_refrigerator', 'floor', 'fork', 'gym_shoe', 'hammer', 'highlighter', 'jewelry', 'kettle', 'knife', 'marker', 'necklace', 'pan', 'pen', 'pencil', 'plate', 'plywood', 'radish', 'rag', 'saw', 'scrub_brush', 'shelf', 'shoe', 'shower', 'sink', 'sock', 'sofa', 'spoon', 'stove', 'table', 'tea_bag', 'teapot', 'toilet', 'towel', 'window'],
    "toggleable": ['blender', 'calculator', 'printer', 'sink', 'stove'],
    "openable": ['cabinet', 'car', 'door', 'electric_refrigerator', 'package', 'stove', 'window'],
    "soakable": ['pot_plant', 'rag', 'scrub_brush', 'tea_bag', 'towel'],
    "overlapable": ['bed', 'door'],
    "containable": ['ashcan', 'bin', 'box', 'cabinet', 'car', 'electric_refrigerator', 'shelf', 'sink', 'stove', 'bucket'],
    "seebehindable": ['countertop', 'door', 'sink'],
    "pickable": ['apple', 'ball', 'banana', 'beef', 'blender', 'book', 'bow', 'bread', 'broom', 'cake', 'calculator', 'candy', 'candle', 'carton', 'carving_knife', 'casserole', 'chicken', 'chip', 'cookie', 'date', 'document', 'dustpan', 'egg', 'fish', 'folder', 'fork', 'gym_shoe', 'hamburger', 'hammer', 'hardback', 'highlighter', 'jewelry', 'juice', 'kettle', 'knife', 'lemon', 'lettuce', 'marker', 'necklace', 'notebook', 'oatmeal', 'olive', 'package', 'pan', 'pen', 'pencil', 'plate', 'plywood', 'pop', 'pot_plant', 'printer', 'radish', 'rag', 'salad', 'sandwich', 'saw', 'scrub_brush', 'shoe', 'soap', 'sock', 'soup', 'spoon', 'strawberry', 'sugar', 'tea_bag', 'teapot', 'tomato', 'towel', 'vegetable_oil', 'water'],
    "slicer": ['carving_knife', 'knife'],
    "cleaningTool": ['broom', 'rag', 'scrub_brush', 'towel'],
    "coldSource": ['electric_refrigerator'],
    "heatSource": ['stove'],
    "waterSource": ['sink', 'teapot'],
}

REVERSE_PROPERTY_FORWARD_DICT = dict()
for key, values in PROPERTY_FORWARD_DICT.items():
    for value in values:
        if value not in REVERSE_PROPERTY_FORWARD_DICT:
            REVERSE_PROPERTY_FORWARD_DICT[value] = []
        REVERSE_PROPERTY_FORWARD_DICT[value].append(key)
        
FOCUS_ABILITIES = [
    'cookable',
    'dustyable',
    'freezable',
    'openable',
    'sliceable',
    'soakable',
    'stainable',
    'toggleable',
]

FOCUS_ABILITIES_MAP = {
    'cookable': "is-cooked",
    'dustyable': "is-not-dusted",
    'freezable': "is-freezed",
    'openable': "is-opened",
    'sliceable': "is-sliced",
    'soakable': "is-soaked",
    'stainable': "is-not-stained",
    'toggleable': "is-toggled",
}

def get_walkable_and_wall_positions(grid):
    walkable_positions = []
    wall_positions = []
    
    for wall in grid.walls:
        wall_positions.append(wall.cur_pos)
        
    for x in range(grid.width):
        for y in range(grid.height):
            if (x, y) not in wall_positions:
                walkable_positions.append((x, y))
                
    return walkable_positions, wall_positions


def generate_behavior_pddl(
    env,
    problem_name : str = "mini_behavior_instance",
    domain_name : str = "mini_behavior",
    goal_clauses : Optional[Sequence[str]] = None,

):
    """Generate PDDL problem definition from the environment.


    Returns:
        problem_str: str, the PDDL problem definition string
    """
    loc_name = lambda x, y: f"pos-{x}-{y}"
    grid = env.grid
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir
    obj_instances = env.obj_instances
    AGENT_DIR_DICT = {0: 'east', 1: 'south', 2: 'west', 3: 'north'}
    DIM_DICT = {0: 'bottom', 1: 'middle', 2: 'top'}
    agent_dir = AGENT_DIR_DICT[agent_dir]
    
    walkable_positions, wall_positions = get_walkable_and_wall_positions(grid)
    
    with open(OBJ_ACTION_JSON_FP, 'r') as f:
        obj_action_json = json.load(f)
        
    obj_types = list(obj_action_json.keys())
    objs_by_type = dict()
    
    for obj_type_name in obj_types:
        objs_by_type[obj_type_name] = set()
        
    objs_by_type['location'] = set()
    objs_by_type['basket'] = set()  # special case for basket, which is not in obj_action_json
    objs_by_type['backpack'] = set()  # special case for backpack, which is not in obj_action_json
    
    # ------- location objects 
    for x, y in walkable_positions:
        loc_name_ = loc_name(x, y)
        objs_by_type['location'].add(loc_name_)
        
    # add door locations
    for x, y in wall_positions:
        items = grid.get_all_items(x, y)
        for item in items:
            if item is not None and item.type != 'wall':
                loc_name_ = loc_name(x, y)
                objs_by_type['location'].add(loc_name_)

    at_info_init_list = []
    
    # ------- object properties
    object_property_dict = dict()
    obj_name_list = list(obj_instances.keys())


    for walkable_pos in list(set(walkable_positions + wall_positions)):
        x, y = walkable_pos
        loc_name_ = loc_name(x, y)
        all_items = grid.get_all_items(x, y) # this gives [furniture_at_bottom, object_at_bottom, furniture_at_middle, object_at_middle, furniture_at_top, object_at_top]
        for dim_i, dim in enumerate(['bottom', 'middle', 'top']):
            furniture = all_items[dim_i * 2]
            obj = all_items[dim_i * 2 + 1]
            
            if furniture is not None and furniture.type != 'wall':
                furniture_type = furniture.type
                furniture_name = furniture.name
                if furniture_name not in obj_name_list: # this means we need to find the right furniture name in the obj_instances
                    # get all objs in the obj_instances that has the same type 
                    temp_obj_collection_lst = []
                    for obj_id, obj_instance in obj_instances.items():
                        if obj_instance.type == furniture_type:
                            temp_obj_collection_lst.append(obj_id)
                    for temp_obj_id in temp_obj_collection_lst:
                        temp_obj_instance = obj_instances[temp_obj_id]
                        # get the coverage of the temp obj instancce 
                        coverage_obj_instance = []
                        temp_obj_width = temp_obj_instance.width
                        temp_obj_height = temp_obj_instance.height
                        temp_obj_x, temp_obj_y = temp_obj_instance.cur_pos
                        for dx in range(temp_obj_width):
                            for dy in range(temp_obj_height):
                                c_x = temp_obj_x + dx
                                c_y = temp_obj_y + dy
                                coverage_obj_instance.append((c_x, c_y))
                        if (x, y) in coverage_obj_instance: # this means the temp_obj_instance is the one we are looking for
                            furniture_name = temp_obj_id
                            break
                    
                
                objs_by_type[furniture_type].add(furniture_name)
                furniture_at_str = f"(at {furniture_name} {loc_name_} {dim})"
            
                at_info_init_list.append(furniture_at_str)
                
                # add furniture properties
                if furniture_name not in object_property_dict:
                    object_property_dict[furniture_name] = set()
                properties = REVERSE_PROPERTY_FORWARD_DICT[furniture_type]
             
                for prop_name in properties:
                    if prop_name == 'containable':  # containable is special case
                        # check can_contain value 
                        if furniture.can_contain:
                            contain_dim = list(furniture.can_contain)
                            for contain_d_id in contain_dim:
                                contain_d = DIM_DICT[contain_d_id]
                                property_str = f"({prop_name} {furniture_name} {contain_d})"
                                object_property_dict[furniture_name].add(property_str)
                                
                            
                        else:
                            raise ValueError(f"Furniture {furniture_name} has can_contain set to False, but is in containable category.")
                    else:
                        property_str = f"({prop_name} {furniture_name})"
                        object_property_dict[furniture_name].add(property_str)
          
                
            if obj is not None:
                obj_type = obj.type
                obj_name = obj.name

                objs_by_type[obj_type].add(obj_name)
                obj_at_str = f"(at {obj_name} {loc_name_} {dim})"
            
                at_info_init_list.append(obj_at_str)
                
                if not (obj_type in ["backpack", 'basket']): 
                    # add object properties
                    if obj_name not in object_property_dict:
                        object_property_dict[obj_name] = set()
                    properties = REVERSE_PROPERTY_FORWARD_DICT[obj_type]
                    
                    for prop_name in properties:
                        if prop_name == 'containable':
                            # containable is special case
                            if obj.can_contain:
                                contain_dim = list(obj.can_contain)
                                for contain_d_id in contain_dim:
                                    contain_d = DIM_DICT[contain_d_id]
                                    property_str = f"({prop_name} {obj_name} {contain_d})"
                                    object_property_dict[obj_name].add(property_str)
                                
                            else:
                                raise ValueError(f"Object {obj_name} has can_contain set to False, but is in containable category.")
                        else:
                            property_str = f"({prop_name} {obj_name})"
                            object_property_dict[obj_name].add(property_str)
                
   
    # add agent location 
    agent_loc_name = loc_name(agent_pos[0], agent_pos[1])
    agent_at_str = f"(at agent-01 {agent_loc_name} bottom)"
    at_info_init_list.append(agent_at_str)
    
    # add carrying stuffs 
    for obj_name in obj_name_list:
        obj_instance = obj_instances[obj_name]
        if obj_instance.cur_pos[0] == -1:
            carrying_str = f"(inhandofrobot agent-01 {obj_name})"
            at_info_init_list.append(carrying_str)
            # also add this to objs_by_type[obj_type].add(obj_name)
            obj_type = obj_instance.type
            if obj_type not in objs_by_type:
                objs_by_type[obj_type] = set()
            objs_by_type[obj_type].add(obj_name)
            
            # update object properties if not already added
            if obj_name not in object_property_dict:
                object_property_dict[obj_name] = set()
                properties = REVERSE_PROPERTY_FORWARD_DICT[obj_type]
                    
                for prop_name in properties:
                    if prop_name == 'containable':
                        # containable is special case
                        if obj_instance.can_contain:
                            contain_dim = list(obj_instance.can_contain)
                            for contain_d_id in contain_dim:
                                contain_d = DIM_DICT[contain_d_id]
                                property_str = f"({prop_name} {obj_name} {contain_d})"
                                object_property_dict[obj_name].add(property_str)
                                
                        else:
                            raise ValueError(f"Object {obj_name} has can_contain set to False, but is in containable category.")
                    else:
                        property_str = f"({prop_name} {obj_name})"
                        object_property_dict[obj_name].add(property_str)
    
    # add agent direction
    agent_dir_str = f"(facing agent-01 {agent_dir})"
    at_info_init_list.append(agent_dir_str)
    
    def _dump_obj_block() -> str:
        return "\n        ".join(
            f"{' '.join(sorted(v))} - {k}"
            for k, v in objs_by_type.items() if v
        )
    
    init_lines : list[str] = []
    static_inits_str = """;; Static initial conditions
(succ bottom middle)
(succ middle top)
(succ north east)
(succ east south)
(succ south west)
(succ west north)
"""
    static_init_splits = static_inits_str.splitlines()
    init_lines.extend(static_init_splits)
    
    # extend at info init lines
    init_lines.extend(at_info_init_list)
    
    # add object properties
    for obj_name, properties in object_property_dict.items():
        init_lines.append(f";; Object properties for {obj_name}")

        for prop_str in properties:
            init_lines.append(prop_str)
            
    # add walkable positions move-dir 
    init_lines.append(";; Walkable positions and move directions")
    MOVE_DIRS = {
        'north': (0, -1),
        'south': (0, 1),
        'east': (1, 0),
        'west': (-1, 0)
    }
    
    actual_location_list = []
    for pos_str in objs_by_type['location']:
        x, y = map(int, pos_str.split('-')[1:])
        actual_location_list.append((x, y))
        
    for x, y in actual_location_list:
        for d_name, (dx, dy) in MOVE_DIRS.items():
            new_x = x + dx
            new_y = y + dy
            if (new_x, new_y) in actual_location_list:
                move_str = f"(move-dir {loc_name(x, y)} {loc_name(new_x, new_y)} {d_name})"
                init_lines.append(move_str)

    # Add focus abilities value 
    init_lines.append(";; Init Abilities")
    for obj_id, obj_instance in obj_instances.items():
        obj_states = obj_instance.states
        for focus_ability in FOCUS_ABILITIES:
            if focus_ability in obj_states:
                ability_value = obj_states[focus_ability].value
                # if value is True, then add the ability
                if focus_ability in ['dustyable', 'stainable']: # this is reversed predicate, <not>
                    if not ability_value:
                        predicate_str = f"({FOCUS_ABILITIES_MAP[focus_ability]} {obj_id})"
                        init_lines.append(predicate_str)
                else:
                    if ability_value and focus_ability not in ['freezable']: # is-freezed is derived predicate
                        predicate_str = f"({FOCUS_ABILITIES_MAP[focus_ability]} {obj_id})"
                        init_lines.append(predicate_str)

    # parse goal clauses
    if goal_clauses is None:
        goal_clauses = []
    elif isinstance(goal_clauses, str):
        goal_clauses = goal_clauses.splitlines()
        
    goal_sec = "\n".join(f"      {clause}" for clause in goal_clauses)

    # ---------------------------------------------------------------- render
    indent_join = lambda seq: "\n        ".join(seq)
    problem_str = f"""(define (problem {problem_name})
  (:domain {domain_name})

  (:objects
        { _dump_obj_block() }
  )

  (:init
        { indent_join(init_lines) }
  )

  (:goal
    (and
{goal_sec}
    )
  )
)"""
    return problem_str


def main(path, map_id, domain_name, if_display):
    """
    Main function to generate PDDL problem file for MiniGrid behavior tasks.
    
    Args:
        path (str): Path to the state data file.
        map_id (int): Map ID to use for the problem generation.
        domain_name (str): Domain name to use for the problem generation.
        if_display (bool): Whether to display the environment window.
    """
    if path:
        state_data_fp = Path(args.path)
        re_pattern = re.compile(r'p(\d+)')
        re_result = re.search(re_pattern, state_data_fp.stem)
        if re_result is not None:
            map_id = int(re_result.group(1))
        else:
            raise ValueError(f"Invalid state data file name format: {state_data_fp.stem}. Expected format like 'p123'.")
        
        domain_name = state_data_fp.stem.split('-')[1:]
        domain_name = '-'.join(domain_name)
        
    else:
        map_id = map_id
        state_data_fp = None
        domain_name = domain_name
        
    assert domain_name in GOAL_CLAUSES, f"Domain name {domain_name} not found in GOAL_CLAUSES."
    
    goal_clauses = GOAL_CLAUSES[domain_name]
    
    problem_name = f"p{map_id}-{domain_name}"

    # load env

    register_env_choice = random.choice(GYM_ENV_REGISTER_DICT[domain_name])
    env = gym.make(register_env_choice)
    env.teleop_mode()
    # load environment 
    # set seed 
    random.seed(map_id)
    npr.seed(map_id)
  
    # load window 
    
    if if_display:
        window = Window('mini_behavior - ' + register_env_choice + ' ' + problem_name)
        key_handler_primitive_partial = partial(key_handler_primitive, window=window, env=env)
        
        window.reg_key_handler(
            key_handler_primitive_partial
        )
    else:
        window = None

    # reset environment or load the state
    reset(map_id, env=env, window=window)
    env.step(env.actions.left)
    env.step(env.actions.right)
    
    if state_data_fp is not None:
        pass # ! we will not load the state from file, the state is only used for generating the PDDL problem file
        # obs = env.load_state(state_data_fp) 
        # if if_display:
        #     redraw(
        #         obs,
        #         env,
        #         window
        #     )
    
    # Deprecated save the env state into a pickle file state pickle file is not used anymore
    # state_save_dir = Path(__file__).parent.parent / 'level'/ domain_name
    # state_save_dir.mkdir(parents=True, exist_ok=True)
    
    # state_saving_fp = state_save_dir / f"p{map_id}-{domain_name}.pkl"
    # state_saving_fp = state_saving_fp.resolve().absolute()
    
    # if not state_saving_fp.exists():
    #     print(f"Saving state to {state_saving_fp}")
    #     env.unwrapped.save_state(str(state_saving_fp))
    #     with open(state_saving_fp, 'rb') as f:
    #         state_data = pickle.load(f)
    # else:
    #     with open(state_saving_fp, 'rb') as f:
    #         state_data = pickle.load(f)
    
        
    # generate the PDDL problem file
    pddl_problem_str = generate_behavior_pddl(
        env=env,
        problem_name = register_env_choice+ "-" + f"p{map_id}",
        domain_name = 'mini_behavior',
        goal_clauses = goal_clauses,
    )
    
    pddl_problem_dir = Path(__file__).parent.parent / 'pddl_problems' / domain_name
    pddl_problem_dir.mkdir(parents=True, exist_ok=True)
    
    pddl_problem_fp = pddl_problem_dir / f"{problem_name}.pddl"
    pddl_problem_fp = pddl_problem_fp.resolve().absolute()
    pddl_problem_fp.write_text(pddl_problem_str, encoding='utf-8')
    
    if if_display:
        window.show(block=True)
        
    # return pddl_problem_fp
    return pddl_problem_fp, register_env_choice
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDDL problem file for MiniGrid behavior tasks.")
    parser.add_argument("--path", type=str, default="", help="Path to the state data file.")
    parser.add_argument("--map_id", type=int, default=0, help="Map ID to use for the problem generation.")
    parser.add_argument("--domain_name", type=str, default="laying_wood_floors", help="Domain name to use for the problem generation.")
    parser.add_argument("--display", action='store_true', help="Whether to display the environment window.")
    
    args = parser.parse_args()
    
    path = args.path
    map_id = args.map_id
    domain_name = args.domain_name
    if_display = args.display
    
    
    main(path, map_id, domain_name, if_display)