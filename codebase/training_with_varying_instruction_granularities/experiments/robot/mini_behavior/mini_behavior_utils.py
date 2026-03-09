"""Utils for evaluating policies in Mini behavior AI environments"""

import argparse
from termcolor import colored
import imageio
import tempfile
import cv2 
import os 
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Sequence, Iterable
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
import time 
from glob import glob
from tqdm.auto import tqdm
from mini_behavior.grid import GridDimension
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache

from mini_behavior.utils.policy_sketch.save_n_load_env import (
    get_mini_behavior_image
)

USE_COMPACT_INSTRUCTION = False

def parse_action(action_str: str, env):
    
    action_str = action_str.replace('turn-left', 'left').replace('turn-right', 'right')
    actions = env.actions

    for action in actions:
        action_name = action.name
        
        if action_str.startswith(action_name):
            return action
        
    raise ValueError(f"Action '{action_str}' not recognized in environment.")
        
ACTION_PARSING_DICT = {
    "left" : 0, 
    "right" : 1, 
    "forward" : 2, 
    "toggle" : 3, 
    "open" : 4, 
    "close" : 5, 
    "slice" : 6, 
    "cook" : 7, 
    "drop_in" : 8, 
    "pickup_0" : 9, 
    "pickup_1" : 10, 
    "pickup_2" : 11, 
    "drop_0" : 12, 
    "drop_1" : 13, 
    "drop_2" : 14
}
        
DOMAIN_NAME_LIST = [
    "laying_wood_floors",
    "sorting_books",
    "collect_misplaced_items",
    "organizing_file_cabinet",
    "watering_houseplants",
    "boxing_books_up_for_storage",
    "cleaning_a_car",
    "thawing_frozen_food",
    "installing_a_printer",
    "storing_food",
    "making_tea",
    "washing_pots_and_pans",
    "putting_away_dishes_after_cleaning",
    "preparing_salad",
    "opening_packages",
    "cleaning_shoes",
    "moving_boxes_to_storage",
    "setting_up_candles",
    "throwing_away_leftovers",
    "cleaning_up_the_kitchen_only",
]

if not USE_COMPACT_INSTRUCTION:

    NL_INSTRUCT_DICT = {
    "laying_wood_floors": [
        "Lay every plywood sheet on the floor, each touching at least one other sheet.",
        "Cover the floor with plywood so every panel is adjacent to another.",
        "Place all plywood on the floor and ensure each has a neighboring sheet.",
        "Install plywood across the floor with every piece next to a second piece.",
        "Put each plywood board flat on the floor and beside another board.",
        "Arrange all plywood on the floor so no sheet is isolated.",
        "Floor the room with plywood, making every sheet edge meet another sheet.",
        "Set every plywood panel on the floor; each must be next to one.",
        "Lay plywood everywhere on the floor, keeping sheets side-by-side.",
        "Place every plywood slab on the floor with at least one neighbor.",
        "Ensure all plywood rests on the floor and is adjacent to a peer.",
        "Position each plywood tile on the floor next to some other tile.",
        "Install plywood on the floor so every piece contacts another piece.",
        "Set down every plywood sheet on the floor and pair it with a neighbor.",
        "Arrange plywood sheets on the floor in a continuous, touching layout.",
        "Put all plywood on the floor; no single sheet stands alone.",
        "Line the floor with plywood with each sheet abutting another.",
        "Lay every plywood panel on the floor, maintaining adjacency throughout.",
        "Place plywood on the floor such that each sheet shares an edge.",
        "Fit all plywood sheets on the floor, each next to at least one."
    ],
    "preparing_salad": [
        "Slice all apples and tomatoes; place them on plates. Put every radish and lettuce on plates. Ensure each plate holds at least one cookable item.",
        "Cut apples and tomatoes, plate them. Plate all radishes and lettuce. Make sure every plate includes a cookable item.",
        "Thin-slice the apples and tomatoes and set them on plates; plate all radish and lettuce; every plate must contain a cookable item.",
        "Slice apples and tomatoes, move to plates. Put radish and lettuce on plates. Each plate should have some cookable item.",
        "Prepare apples and tomatoes by slicing and plating; plate every radish and lettuce; guarantee every plate has a cookable item.",
        "Make slices of apples and tomatoes and plate them; plate all radish and lettuce; ensure a cookable item appears on every plate.",
        "Cut apples and tomatoes into slices onto plates; place radish and lettuce on plates; include at least one cookable thing per plate.",
        "Slice apples/tomatoes and plate; plate radish/lettuce; add a cookable item to each plate.",
        "Create salad by slicing apples and tomatoes and plating; also plate radish and lettuce; every plate must feature a cookable item.",
        "Apples and tomatoes: slice and plate. Radish and lettuce: plate. Verify each plate has a cookable object.",
        "Finely slice apples and tomatoes onto plates; arrange radish and lettuce on plates; each plate needs one cookable item.",
        "Cut apples, cut tomatoes, plate both; plate lettuce and radish; put some cookable item on each plate.",
        "Slice all apples/tomatoes; plate them. Plate all lettuce/radish. Each plate must contain a cookable ingredient.",
        "Produce slices of apples and tomatoes and put on plates; place radish and lettuce on plates; ensure per-plate a cookable item exists.",
        "Apples and tomatoes are sliced and plated; radish and lettuce are plated; every plate includes a cookable food.",
        "Chop apples and tomatoes into slices and plate; plate radish and lettuce; guarantee one cookable object per plate.",
        "Slice apples, slice tomatoes; plate them. Plate every radish and lettuce. Add a cookable element to each plate.",
        "Prepare by slicing apples/tomatoes and plating; plate lettuce/radish; make sure each plate hosts a cookable item.",
        "All apples and tomatoes sliced and on plates; all radish and lettuce on plates; each plate contains a cookable thing.",
        "Slice apples and tomatoes and plate them; also plate radish and lettuce; include at least one cookable item on every plate."
    ],
    "cleaning_up_the_kitchen_only": [
        "Put every blender on a countertop. Refrigerate all apples and casseroles. Keep soap next to a sink. Make plates unstained and cabinets dust-free. Place each rag either beside or inside the sink. Store plates and vegetable oil in different cabinets.",
        "Move blenders to countertops. Store apples and casseroles in the fridge. Put soap near a sink. Ensure plates are clean and cabinets dust-free. Rags go next to or inside the sink. Keep plates and vegetable oil in separate cabinets.",
        "All blenders on countertops; apples and casseroles in the refrigerator; soap next to a sink; plates unstained; cabinets dust-free; rags by or in the sink; plates and oil kept in different cabinets.",
        "Countertop every blender. Fridge all apples and casseroles. Position soap beside a sink. Clean plates; dust-free cabinets. Rags either next to or inside the sink. Separate cabinets for plates and vegetable oil.",
        "Place blenders on countertops, apples/casseroles in the fridge, and soap by a sink. Ensure clean plates, dust-free cabinets. Rags belong next to or in the sink. Plate storage must differ from oil storage.",
        "Set each blender onto a countertop. Chill apples and casseroles. Keep soap adjacent to a sink. Plates must be unstained; cabinets dustless. Rags live next to or inside the sink. Do not use the same cabinet for plates and vegetable oil.",
        "Arrange blenders on countertops; refrigerate apples and casseroles; station soap near a sink. Leave plates spotless and cabinets free of dust. Place rags by or in the sink. Use separate cabinets for plates and vegetable oil.",
        "Countertops hold every blender. The refrigerator holds apples and casseroles. Soap sits next to a sink. Plates are clean; cabinets dust-free. Rags are adjacent to or inside the sink. Plates and oil go to different cabinets.",
        "Move blenders to counters; apples and casseroles to the refrigerator; soap next to a sink. Make plates unstained, cabinets not dusty. Rags should be beside or inside the sink. Separate plate storage from oil storage.",
        "All blenders belong on countertops. Apples and casseroles belong in the fridge. Soap must be placed by a sink. Plates must be clean and cabinets dust-free. Rags are either next to or inside the sink. Store plates and vegetable oil in non-matching cabinets.",
        "Place blenders on counters; refrigerate apples/casseroles; keep soap near a sink. Ensure plates are unstained and cabinets dust-free. Park rags next to or in the sink. Use different cabinets for plates versus vegetable oil.",
        "Every blender: countertop. Apples and casseroles: refrigerator. Soap: next to a sink. Plates: clean. Cabinets: dust-free. Rags: by or in sink. Plates and oil: different cabinets.",
        "Set blenders on counters. Put apples and casseroles in the fridge. Keep soap adjacent to a sink. Leave plates spotless and cabinets dustless. Rags stay beside or inside the sink. Separate cabinet for oil from the one for plates.",
        "Countertop all blenders. Fridge the apples/casseroles. Soap goes by a sink. Plates are unstained; cabinets are dust-free. Rags are by or in the sink. Don't co-store plates with vegetable oil.",
        "Place each blender on a countertop; stow apples and casseroles in the refrigerator; set soap next to a sink; keep plates clean and cabinets dust-free; position rags by or in the sink; store plates and oil in different cabinets.",
        "All blenders to countertops; apples/casseroles to fridge; soap to sink-side. Plates must be clean; cabinets dust-free. Rags placed next to or inside the sink. Use separate storage for plates and vegetable oil.",
        "Put blenders up on counters. Refrigerate apples and casseroles. Stage soap at a sink. Verify plates are unstained and cabinets dust-free. Drop rags next to or into the sink. Keep plates and oil in distinct cabinets.",
        "Counters: blenders. Fridge: apples, casseroles. Sink-side: soap. Plates clean; cabinets dust-free. Rags next to or in sink. Plates and oil in separate cabinets.",
        "On countertops place blenders; in the refrigerator place apples and casseroles; put soap near a sink; ensure plates are clean and cabinets dustless; place rags by or inside the sink; do not share a cabinet between plates and vegetable oil.",
        "Position blenders on countertops, chill apples and casseroles, put soap by a sink, keep plates spotless and cabinets dust-free, set rags beside or in the sink, and separate plate cabinets from oil cabinets."
    ],
    "organizing_file_cabinet": [
        "Put a marker on a table and file every folder and document in a cabinet.",
        "Place one marker on a table; store all folders and documents inside cabinets.",
        "Ensure a marker sits on a table; put every folder and document into a cabinet.",
        "Leave a marker on a table and organize all folders and documents in cabinets.",
        "Set a marker on a table; file all folders and documents away in a cabinet.",
        "Have a marker on a table; move every folder and document into cabinets.",
        "Keep a marker on any table; place all folders and documents inside a cabinet.",
        "Place a marker on the table; cabinet every folder and document.",
        "Marker goes on a table; folders and documents go into cabinets.",
        "Put down a marker on a table; file all documents and folders in cabinets.",
        "Ensure there is a marker on a table; store all folders/documents in a cabinet.",
        "Set a marker atop a table and file every folder plus document in cabinets.",
        "Position a marker on a table; place all documents and folders inside cabinets.",
        "Make sure a marker is on a table; cabinet all folders and documents.",
        "Put a marker on any table, then file all folders and documents in cabinets.",
        "Keep a marker on the table while stowing every folder and document in a cabinet.",
        "Table holds a marker; cupboards hold every folder and document.",
        "Place one marker on a table; archive all documents and folders in cabinets.",
        "Leave a marker on a table; all folders and documents must be inside cabinets.",
        "Lay a marker on a table and relocate all folders/documents to cabinets."
    ],
    "thawing_frozen_food": [
        "Place a date label next to a fish. Put every fish next to a sink. Set an olive next to a sink.",
        "Keep some date item beside a fish; position all fish by a sink; place an olive by a sink.",
        "Set a date near one fish, ensure all fish are next to a sink, and put an olive by a sink.",
        "Have a date marker next to a fish; move all fish beside a sink; add an olive near a sink.",
        "A date goes next to a fish. Every fish sits next to a sink. An olive rests by a sink.",
        "Put a date object adjacent to a fish; place all fish adjacent to a sink; place an olive adjacent to a sink.",
        "Next to a fish, add a date. For every fish, place it by a sink. Also put an olive near a sink.",
        "Add a date beside some fish; align all fish near a sink; keep an olive near a sink.",
        "Place one date tag beside a fish; station all fish next to sinks; position an olive next to a sink.",
        "Date next to fish; all fish beside sink; olive beside sink.",
        "Attach a date label near a fish; arrange every fish by a sink; set an olive by a sink.",
        "A date should be next to a fish; every fish should be next to a sink; an olive should be next to a sink.",
        "Keep a date at the side of a fish; put each fish next to a sink; place an olive near a sink.",
        "Date beside fish, fish beside sink, olive beside sink.",
        "Ensure a date item is near one fish; all fish are sink-side; an olive is sink-side.",
        "Position a date near a fish; situate all fish near sinks; include an olive near a sink.",
        "Place a date close to a fish; keep each fish close to a sink; add an olive close to a sink.",
        "Set a date label next to a fish; move all fish next to sinks; put an olive next to a sink.",
        "Have a date adjacent to a fish; make all fish adjacent to sinks; keep an olive adjacent to a sink.",
        "Date beside fish, every fish near sink, one olive near sink."
    ],
    "making_tea": [
        "Slice a lemon. Put a teapot on an activated stove. Keep a soaked teabag with the teapot.",
        "Turn on the stove and place a teapot on it; slice a lemon; have a soaked teabag with the teapot.",
        "Set a teapot on a powered stove; cut a lemon; keep a soaked tea bag at the teapot's location.",
        "Boil with the stove on: teapot on the stove, lemon sliced, teabag soaked and beside the teapot.",
        "Power the stove, place the teapot on top, slice a lemon, and keep a soaked teabag with it.",
        "Stove toggled on, teapot atop; lemon sliced; teabag soaked and colocated with the teapot.",
        "Activate stove; set teapot on it; slice lemon; ensure teabag is soaked and at the teapot.",
        "Put the teapot on a turned-on stove; slice the lemon; keep a wet teabag with the pot.",
        "Switch the stove on; place teapot on stove; cut lemon; tea bag soaked and near the pot.",
        "Heat on: teapot on stove; lemon sliced; soaked teabag at the teapot.",
        "With the stove on, rest a teapot on it; slice a lemon; teabag must be soaked and co-located.",
        "Turn stove on; teapot on top; lemon sliced; teabag soaked and kept with the teapot.",
        "Place teapot on an active stove and slice a lemon; keep a soaked teabag with the pot.",
        "Engage the stove, set the teapot on it, slice lemon, and soak the teabag at the pot.",
        "Teapot goes on a hot stove; lemon gets sliced; teabag is soaked and at the pot.",
        "Power on stove, seat teapot, slice lemon, ensure teabag is soaked and with teapot.",
        "Activate the burner; teapot on top; cut lemon; teabag soaked and co-located.",
        "Teapot positioned on a lit stove; lemon sliced; soaked teabag placed with the teapot.",
        "Stove on; teapot on stove; sliced lemon; soaked teabag at same spot.",
        "Heat the stove, place teapot, slice lemon, and keep a soaked teabag with it."
    ],
    "opening_packages": [
        "Open every package.",
        "Make sure all packages are opened.",
        "Unseal all packages.",
        "Ensure every package is open.",
        "Open all the packages.",
        "Have every package in an opened state.",
        "Leave no package unopened.",
        "Confirm all packages are opened.",
        "Set all packages to opened.",
        "Open each and every package.",
        "Open all packages you find.",
        "Guarantee that all packages are open.",
        "All packages should be opened.",
        "Make all packages open.",
        "Ensure all packages are unsealed.",
        "Complete by opening every package.",
        "Turn every package into an opened package.",
        "Open 100% of the packages.",
        "Get all packages opened.",
        "Finish with all packages open."
    ],
    "boxing_books_up_for_storage": [
        "Place every book inside a box.",
        "Box up all the books.",
        "Put all books into boxes.",
        "Ensure each book is boxed.",
        "Pack every book into a box.",
        "Have all books contained in boxes.",
        "Store all books inside boxes.",
        "Move every book into a box.",
        "Get all books boxed for storage.",
        "Make sure every book ends up in a box.",
        "Collect and box all books.",
        "All books should be inside boxes.",
        "Box each book individually or together.",
        "Put books away by boxing them.",
        "Every single book goes in a box.",
        "Load all books into boxes.",
        "Secure all books within boxes.",
        "Place the entire set of books in boxes.",
        "Contain all books in boxes.",
        "Pack up the books into boxes."
    ],
    "collect_misplaced_items": [
        "Place all gym shoes, necklaces, notebooks, and socks on a table.",
        "Gather gym shoes, necklaces, notebooks, and socks and set them on a table.",
        "Put every gym shoe, necklace, notebook, and sock onto a table.",
        "Ensure gym shoes, necklaces, notebooks, and socks are resting on a table.",
        "Move all gym shoes, necklaces, notebooks, and socks to a tabletop.",
        "Collect each gym shoe, necklace, notebook, and sock and place on a table.",
        "All gym shoes, necklaces, notebooks, socks should be on a table.",
        "Relocate gym shoes, necklaces, notebooks, and socks to the top of a table.",
        "Arrange gym shoes, necklaces, notebooks, and socks on a table surface.",
        "Set the gym shoes, necklaces, notebooks, and socks on any table.",
        "Consolidate gym shoes, necklaces, notebooks, and socks onto a table.",
        "Everything --- gym shoes, necklaces, notebooks, socks --- must be on a table.",
        "Put the listed items (shoes, necklaces, notebooks, socks) on a table.",
        "Table all gym shoes, all necklaces, all notebooks, and all socks.",
        "Move those items to a table and leave them there.",
        "Every gym shoe, necklace, notebook, and sock should end on a table.",
        "Gather and table all gym shoes, necklaces, notebooks, and socks.",
        "Place each item --- gym shoe, necklace, notebook, sock --- on the table.",
        "Ensure the table holds all gym shoes, necklaces, notebooks, and socks.",
        "Transfer gym shoes, necklaces, notebooks, and socks onto a tabletop."
    ],
    "putting_away_dishes_after_cleaning": [
        "Put every plate inside a cabinet.",
        "Store all plates in cabinets.",
        "Place plates away in a cabinet.",
        "Ensure each plate is inside some cabinet.",
        "Cabinet all plates.",
        "Move every plate into a cabinet.",
        "Stow plates in cabinets.",
        "File all plates into cabinets.",
        "All plates should be in a cabinet.",
        "Put plates away by placing them in cabinets.",
        "Keep every plate inside a cabinet space.",
        "Relocate plates into cabinets.",
        "Place the entire set of plates into cabinets.",
        "Every plate must end up in a cabinet.",
        "House plates within cabinets.",
        "Store plates neatly in a cabinet.",
        "Return all plates to cabinets.",
        "Put the plates back into cabinets.",
        "Ensure no plate remains outside a cabinet.",
        "Tidy up by cabining all plates."
    ],
    "washing_pots_and_pans": [
        "Clean every teapot, kettle, and pan so they're unstained, then store each inside a cabinet.",
        "Ensure teapots, kettles, and pans are unstained and placed inside cabinets.",
        "Wash teapots, kettles, and pans; once clean, put them in cabinets.",
        "Make teapots, kettles, and pans spotless and stow them in cabinets.",
        "Remove stains from all teapots, kettles, and pans and cabinet them.",
        "Have every teapot, kettle, and pan clean, then put each into a cabinet.",
        "All teapots/kettles/pans: clean (unstained) and stored in cabinets.",
        "Scrub teapots, kettles, and pans clean and place inside cabinets.",
        "Leave no stains on teapots, kettles, or pans; store them in cabinets.",
        "Wash and dry teapots, kettles, and pans; cabinet them afterward.",
        "Make sure teapots, kettles, and pans are clean, then cabinet each.",
        "Teapots, kettles, pans must be unstained and inside cabinets.",
        "Clean cookware (teapots, kettles, pans) and relocate to cabinets.",
        "Shine teapots, kettles, pans; put them away in cabinets.",
        "Get all teapots, kettles, and pans spotless; store them in cabinets.",
        "Remove stains from the cookware and house it in cabinets.",
        "Finish with all teapots, kettles, pans clean and inside cabinets.",
        "Ensure cleanliness of teapots, kettles, pans, then put them in cabinets.",
        "All specified cookware should be clean and stored in cabinets.",
        "Unstained teapots, kettles, and pans belong in cabinets."
    ],
    "cleaning_shoes": [
        "Leave a towel on the floor and make sure every shoe is unstained and dust-free.",
        "Place a towel on the floor; clean all shoes of stains and dust.",
        "A towel goes on the floor; every shoe must be clean and not dusty.",
        "Put a towel on the floor and ensure shoes are spotless and dustless.",
        "Have a towel lying on the floor; all shoes must be clean and dust-free.",
        "Keep one towel on the floor; make each shoe unstained and not dusty.",
        "Floor a towel; verify all shoes are both unstained and undusted.",
        "Drop a towel on the floor and clean/dedust every shoe.",
        "Set a towel on the floor; shoes should be clear of stains and dust.",
        "Place towel on floor, and ensure shoes are clean and free of dust.",
        "Ensure a towel is on the floor, and every shoe is clean and dust-free.",
        "Put down a towel on the floor; remove stains and dust from all shoes.",
        "Have a towel on the floor; confirm shoes aren't stained or dusty.",
        "Towel on the floor; shoes must be both unstained and undusted.",
        "Keep a towel on the floor; cleanse and dedust all shoes.",
        "Lay a towel on the floor and finish with clean, dust-free shoes.",
        "Set towel on floor; all shoes to be unstained and not dusty.",
        "A towel should be on the floor; shoes should be spotless and dust-free.",
        "Place a towel on the floor; every shoe ends up clean and dustless.",
        "Lay a towel down and ensure no shoe is stained or dusty."
    ],
    "installing_a_printer": [
        "Place a printer on a table and turn it on.",
        "Set the printer atop a table and power it up.",
        "Put the printer on a table; ensure it is switched on.",
        "Install the printer on a table and toggle it on.",
        "Position a printer on a table and activate it.",
        "Rest a printer on a tabletop and turn the power on.",
        "Place printer on table; printer must be on.",
        "Put the printer on the table and ensure it's powered.",
        "Set a printer on a table surface and switch it on.",
        "Keep the printer on a table and in the on state.",
        "Mount the printer on a table and power it on.",
        "Put a printer onto a table and enable it.",
        "Place the printer on the table; toggle power to on.",
        "Install the printer by setting it on a table and turning it on.",
        "Printer goes on a table and must be on.",
        "Locate a table for the printer and turn it on.",
        "Tabletop the printer and switch it on.",
        "Set the printer on a table; confirm it is on.",
        "Place and power on a printer on a table.",
        "Put a printer on a table and leave it turned on."
    ],
    "setting_up_candles": [
        "On every table, place three distinct candles on top.",
        "Ensure each table holds exactly three different candles.",
        "Arrange three unique candles on each table.",
        "Put three separate candles on every table.",
        "For every table, set three non-identical candles on it.",
        "Top each table with three distinct candles.",
        "Every table should have three different candles on top.",
        "Place a trio of unique candles on each table.",
        "Each table must display three separate candles.",
        "Set three individual candles on each table.",
        "Populate every table with three distinct candles.",
        "Arrange a set of three different candles per table.",
        "Lay three unique candles on top of every table.",
        "Place three discrete candles on each table.",
        "Every table is to hold three non-matching candles.",
        "Put a group of three distinct candles on every table.",
        "Provide each table with three different candles on top.",
        "Ensure per table there are three separate candles.",
        "Each table needs three distinct candles placed on it.",
        "Top off each table with three unique candles."
    ],
    "watering_houseplants": [
        "Water every potted plant until it is soaked.",
        "Ensure all pot plants are soaked.",
        "Saturate every potted plant with water.",
        "Soak all houseplants thoroughly.",
        "Drench each potted plant.",
        "Leave every pot plant fully soaked.",
        "Make sure every houseplant is soaked with water.",
        "All potted plants should be soaked.",
        "Soak every plant in a pot.",
        "Water until every potted plant is soaked.",
        "Saturate all pot plants completely.",
        "Bring every houseplant to a soaked state.",
        "Ensure each potted plant is well soaked.",
        "Get all potted plants soaked.",
        "Soak every indoor pot plant.",
        "Leave no potted plant unsoaked.",
        "All houseplants must end up soaked.",
        "Completely soak the potted plants.",
        "Make each pot plant soaked thoroughly.",
        "Water plants so every pot plant is soaked."
    ],
    "cleaning_a_car": [
        "Make at least one car dust-free. Place a rag and soap together inside a bucket.",
        "Have one car cleaned of dust; put a rag and soap in a bucket.",
        "Ensure some car is not dusty; store a rag and soap inside a bucket.",
        "Leave one car undusted; keep rag and soap inside the bucket.",
        "Get a car clean (no dust) and place rag and soap into a bucket.",
        "At least one car must be dust-free; both rag and soap go in a bucket.",
        "Clean a car of dust; contain the rag and the soap in a bucket.",
        "Produce one dustless car; put rag and soap together in a bucket.",
        "Achieve a dust-free car; have rag and soap inside the bucket.",
        "One car should be undusted; stash rag and soap in a bucket.",
        "Have a car not dusty; place the rag and soap inside a bucket.",
        "Make some car clean from dust; put rag plus soap into a bucket.",
        "Ensure one car is dust-free; keep both rag and soap in a bucket.",
        "Deliver at least one dustless car; store rag and soap in a bucket.",
        "Obtain a clean (not dusty) car; bucket the rag and the soap.",
        "Get at least one car cleaned; place rag and soap together in a bucket.",
        "Provide one car that is not dusty; keep rag and soap inside a bucket.",
        "Have one car come out dust-free; leave rag and soap in a bucket.",
        "Produce a clean car; ensure rag and soap are contained in a bucket.",
        "Make a car dust-free and place a rag and soap in a bucket."
    ],
    "storing_food": [
        "Store every cookable item inside a cabinet.",
        "Place all cookable items in cabinets.",
        "Put any cookable normal item into a cabinet.",
        "Ensure all cookable things are stored in cabinets.",
        "Cabinet every item that is cookable.",
        "All cookable objects should be inside cabinets.",
        "Move cookable items into cabinets.",
        "Stow cookable goods in a cabinet.",
        "Keep every cookable item within a cabinet.",
        "Any cookable item must be in a cabinet.",
        "Place cookable foods inside cabinets.",
        "Make sure cookable items end up in cabinets.",
        "House all cookable items in cabinets.",
        "Relocate cookable items to cabinets.",
        "Every cookable object belongs in a cabinet.",
        "Cabinet all items that are cookable.",
        "Put cookable normal-items into cabinets.",
        "Ensure cabinet storage for every cookable item.",
        "Keep cookable materials stored in cabinets.",
        "Stash cookable items away in cabinets."
    ],
    "throwing_away_leftovers": [
        "Throw away every hamburger by placing it in a trash can.",
        "Put all hamburgers into an ashcan.",
        "Ensure each hamburger is inside a trash can.",
        "Discard every hamburger in the bin.",
        "Place every hamburger in an ashcan for disposal.",
        "All hamburgers must end up in a trash can.",
        "Move every hamburger into a garbage can.",
        "Stow hamburgers inside the ashcan.",
        "Deposit each hamburger in a trash receptacle.",
        "Make sure every hamburger is trashed in an ashcan.",
        "Throw all hamburgers into the can.",
        "Place the hamburgers into the ashcan.",
        "Every hamburger belongs inside a trash can.",
        "Get every hamburger into the bin.",
        "Ensure no hamburger remains outside the ashcan.",
        "Dispose of all hamburgers in a trash can.",
        "Put each hamburger in a waste can.",
        "All leftover hamburgers should be in the ashcan.",
        "Drop all hamburgers into the trash can.",
        "Finish with every hamburger placed in an ashcan."
    ],
    "moving_boxes_to_storage": [
        "Place every carton inside a shelf.",
        "Store all cartons in shelves.",
        "Put each carton into a shelf compartment.",
        "Ensure every box (carton) is within a shelf.",
        "Move all cartons into shelving.",
        "Cabinet-shelve every carton (inside the shelf).",
        "Stow every carton in a shelf space.",
        "Keep all cartons housed inside shelves.",
        "Relocate cartons to inside the shelves.",
        "Place the entire set of cartons in shelves.",
        "All cartons belong inside shelves.",
        "Transfer each carton into a shelf.",
        "Shelve every carton by placing it inside.",
        "Insert every carton into a shelf bay.",
        "Make sure each carton ends up inside a shelf.",
        "Store cartons by putting them into shelves.",
        "Contain all cartons within shelves.",
        "Load every carton inside the shelving.",
        "Ensure shelves contain all cartons.",
        "Put every carton away in shelves."
    ],
    "sorting_books": [
        "Place every book and every hardback on a shelf.",
        "Put all books, including hardbacks, onto shelves.",
        "Shelve every book and also every hardback.",
        "Ensure each book and each hardback is on a shelf.",
        "Move books and hardbacks to shelves.",
        "All books and all hardbacks should be on shelves.",
        "Set every book and hardback atop a shelf.",
        "Arrange books and hardbacks on shelves.",
        "Every book, including hardbacks, belongs on a shelf.",
        "Place the entire set of books and hardbacks on shelves.",
        "Put each book and each hardback up on a shelf.",
        "Shelve the books and shelve the hardbacks.",
        "Ensure shelves hold all books and all hardbacks.",
        "Store both books and hardbacks on shelving.",
        "Lay every book and hardback on shelf surfaces.",
        "Return books and hardbacks to shelves.",
        "Organize by placing books and hardbacks on shelves.",
        "Books go to shelves; hardbacks go to shelves too.",
        "Every book and hardback must end up on a shelf.",
        "Finish with books and hardbacks shelved."
    ]
    }
else:
    # create compact instruction that basically the same as the task title 
    NL_INSTRUCT_DICT = {domain_name: [f"Complete the task: {domain_name.replace('_', ' ')}."] for domain_name in DOMAIN_NAME_LIST}

NL_INSTRUCT_DICT_TEMP = {}

# encode NL to ascii, if failed, then skip that 
for k, v in NL_INSTRUCT_DICT.items():
    try:
        NL_INSTRUCT_DICT_TEMP[k] = [s.encode("ascii", "ignore").decode("ascii") for s in v]
    except Exception as e:
        print(f"Error encoding NL instructions for {k}: {e}")

NL_INSTRUCT_DICT = NL_INSTRUCT_DICT_TEMP


assert set(NL_INSTRUCT_DICT.keys()) == set(DOMAIN_NAME_LIST), "NL_INSTRUCT_DICT keys do not match DOMAIN_NAME_LIST"


import sys
import os

# Get the current directory of main_script.py
current_dir = os.path.dirname(os.path.abspath(__file__))
openvla_oft_dir = Path(current_dir).parent.parent.parent

# Add the current directory to sys.path
sys.path.insert(0, str(openvla_oft_dir))

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)

assert os.environ.get("WORKING_DIR", None) is not None, "WORKING_DIR is not set"

PDDL_PROBLEM_PATTERN = os.path.join(os.environ["WORKING_DIR"], "data/00_envs/mini_behavior/mini_behavior/utils/pddl_problems/{domain_name}/*.pddl")

PDDL_PLAN_FOLDER_PATTERN = os.path.join(os.environ["WORKING_DIR"], "data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/{domain_name}")


# all_files = glob(PDDL_PROBLEM_PATTERN.format(domain_name="*"), recursive=True)


def get_mini_behavior_env(domain_name, problem_id):

    problem_files = glob(PDDL_PROBLEM_PATTERN.format(domain_name=domain_name), recursive=True)

    
    if not (problem_id == "" or problem_id is None):
        problem_file = Path(os.path.join(str(Path(PDDL_PROBLEM_PATTERN.format(domain_name=domain_name)).parent), f"p{problem_id}-{domain_name}.pddl"))
        assert problem_file.exists(), f"Problem file does not exist: {problem_file}"
    else:
        problem_file = Path(random.choice(problem_files))

    assert problem_file.exists(), f"Problem file does not exist: {problem_file}"

    problem_file_stem = problem_file.stem
    # obtain problem_id from stem
    problem_id = problem_file_stem.split("-")[0][1:]

    plan_fp = Path(os.path.join(PDDL_PLAN_FOLDER_PATTERN.format(domain_name=domain_name), f"p{problem_id}-{domain_name}_plans.json"))

    assert plan_fp.exists(), f"Plan file {plan_fp} does not exist."
    
    # ! Load the plans 
    with open(plan_fp, 'r') as f:
        pddl_plan_info = json.load(f)
        
    env_id = pddl_plan_info['env_id']
    
    plans = pddl_plan_info["plans"]
    
    # load the env 
    env = gym.make(env_id)
    env.teleop_mode()
    env.seed(int(problem_id))


    actions_lst = [] 
    for plan in plans:
        actions = [parse_action(p, env) for p in plan]
        actions = [a for a in actions if a is not None]  # filter out None actions
        if len(actions) > 0:
            actions_lst.append(actions)

    print(colored(f"Using env_id: {env_id}, problem_id: {problem_id}", "green"))

    nl_instruction_list = NL_INSTRUCT_DICT[domain_name]
    
    nl_instruction = random.choice(nl_instruction_list) # pick one randomly

    return env, nl_instruction, nl_instruction_list, int(problem_id), actions_lst, plans



# def _fit_into_box(img_np, box_w, box_h, pad=0, bg=(255, 255, 255)):
#     """
#     Aspect-fit a HxWxC numpy RGB image into a box (box_w x box_h),
#     add uniform padding `pad`, and return a PIL.Image with background `bg`.
#     """
#     img = Image.fromarray(img_np.astype(np.uint8), mode="RGB")
#     W, H = img.size

#     # Available area inside padding
#     aw = max(1, box_w - 2 * pad)
#     ah = max(1, box_h - 2 * pad)

#     # Scale by min ratio
#     scale = min(aw / W, ah / H)
#     new_w = max(1, int(round(W * scale)))
#     new_h = max(1, int(round(H * scale)))
#     img_rs = img.resize((new_w, new_h), Image.LANCZOS)

#     # Center on canvas
#     canvas = Image.new("RGB", (box_w, box_h), bg)
#     ox = (box_w - new_w) // 2
#     oy = (box_h - new_h) // 2
#     canvas.paste(img_rs, (ox, oy))
#     return canvas

# def _draw_title_bar(img_pil, title, height_px=24, bg=(255,255,255), fg=(0,0,0)):
#     """
#     Add a title bar on TOP of img_pil and return a new image with the bar.
#     """
#     W, H = img_pil.size
#     bar = Image.new("RGB", (W, height_px), bg)
#     d = ImageDraw.Draw(bar)
#     # Default PIL font (portable). If you have a font file, you can load it via ImageFont.truetype
#     # and pass it in --- but we stay dependency-free here.
#     d.text((W // 2, height_px // 2), title, fill=fg, anchor="mm")
#     out = Image.new("RGB", (W, H + height_px), bg)
#     out.paste(bar, (0, 0))
#     out.paste(img_pil, (0, height_px))
#     return out



# def render_inventory_image_old(on_grid, carrying,
#                            fontsize_ratio=200,
#                            title_fontsize=10,      # bumped up
#                            body_fontsize=9,        # nudged down
#                            state_images=None,      # list of 4 np arrays [Top, Middle, Bottom, Container]
#                            section_gap_px=12,
#                            main_image=None,        # np array (agent view)
#                            out_size=1024,          # final square size
#                            inv_width_frac=0.28,    # fraction of full width for the left inventory strip
#                            bottom_height_frac=0.20,# fraction of full height for the bottom row of 4 states
#                            pad_px=1,
#                            bbox_kwargs=None,
#                            bg=(255, 255, 255),
#                            fg=(0, 0, 0)):
#     """
#     If `main_image` and `state_images` (length 4) are provided, compose a square panel:
#       - Left: inventory (on_grid / carrying) stacked vertically
#       - Middle/Right: main_image (agent view)
#       - Bottom: four state images titled Top, Middle, Bottom, Container (full width)
#     Otherwise, returns just the inventory panel as before.

#     Assumes `state_images` order = [Top, Middle, Bottom, Container].
#     """
#     # ---------- 1) Render the inventory panel (as before, with a content bbox) ----------
#     if bbox_kwargs is None:
#         bbox_kwargs = dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="none", linewidth=0.5)

#     # ---------- 2) Prepare layout metrics for the final square ----------
#     S = int(out_size)
#     S = max(256, S)
#     inv_w = int(round(inv_width_frac * S))
#     bottom_h = int(round(bottom_height_frac * S))
#     top_h = S - bottom_h               # height for (inventory + main_image) row
#     main_w = S - inv_w                 # width for main image

#     # ---------- 1) Render the inventory panel as a SINGLE AXES at target size ----------
#     dpi_inv = 200
#     figsize_inv = (max(1, inv_w) / dpi_inv, max(1, top_h) / dpi_inv)

#     # content + spacing params (in pixels, will be scaled if needed)
#     top_margin_px    = max(4, int(0.04 * top_h))
#     section_gap_px   = section_gap_px
#     title_gap_px     = 10
#     line_spacing     = 1.05

#     # initial font sizes (pixels)  ---  will auto-shrink if needed
#     # start with generous sizes; we'll scale down to fit if required
#     title_px = title_fontsize
#     body_px  = body_fontsize

#     on_grid_lines  = max(1, len(on_grid) if on_grid else 1)
#     carrying_lines = max(1, len(carrying) if carrying else 1)
#     h_on_grid   = title_px + title_gap_px + int(on_grid_lines * body_px * line_spacing)
#     h_carrying  = title_px + title_gap_px + int(carrying_lines * body_px * line_spacing)
#     required_px = top_margin_px + h_on_grid + section_gap_px + h_carrying + top_margin_px

#     # scale down fonts if content won't fit vertically
#     if required_px > top_h:
#         scale = (top_h - 2) / max(1, required_px)  # tiny 1–2px breathing room
#         title_px = max(10, int(title_px * scale))
#         body_px  = max(8,  int(body_px  * scale))
#         # recompute heights
#         h_on_grid   = title_px + title_gap_px + int(on_grid_lines * body_px * line_spacing)
#         h_carrying  = title_px + title_gap_px + int(carrying_lines * body_px * line_spacing)
#         required_px = top_margin_px + h_on_grid + section_gap_px + h_carrying + top_margin_px

#     # convert px -> pt for Matplotlib font sizes
#     title_pt = fontsize_ratio * title_px / dpi_inv
#     body_pt  = fontsize_ratio * body_px  / dpi_inv

#     # build figure/axes that exactly match target pixels
#     fig = plt.figure(figsize=figsize_inv, dpi=dpi_inv)
#     ax  = fig.add_axes([0, 0, 1, 1])  # full-bleed axes
#     ax.set_axis_off()

#     # helpers
#     def _px_to_axes_y(px):  # convert pixel Y offset to axes fraction
#         return px / top_h

#     def _draw_section(title, items, y_top_px):
#         # draw title INSIDE axes at y_top_px (pixels from bottom); anchor top
#         ax.text(0.5, 1.0 - _px_to_axes_y(y_top_px),
#                 title, ha='center', va='top', fontsize=title_pt, transform=ax.transAxes)

#         # body directly beneath the title with a small gap
#         body_y_top = y_top_px + title_px + title_gap_px
#         text = "\n".join(items) if items else " --- "
#         ax.text(0.5, 1.0 - _px_to_axes_y(body_y_top),
#                 text, ha='center', va='top', fontsize=body_pt,
#                 bbox=bbox_kwargs, transform=ax.transAxes)

#         # return height consumed by this section
#         lines = max(1, len(items)) if items else 1
#         body_h = int(lines * body_px * line_spacing)
#         return (title_px + title_gap_px + body_h)

#     # lay out sections from the top down
#     y_cursor = top_margin_px
#     y_cursor += _draw_section("on grid", on_grid, y_cursor)
#     y_cursor += section_gap_px
#     y_cursor += _draw_section("carrying", carrying, y_cursor)

#     # render to array
#     fig.canvas.draw()
#     iw, ih = fig.canvas.get_width_height()
#     inv_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(ih, iw, 3)
#     plt.close(fig)

#     # ensure exact target size (rare off-by-one)
#     if (iw, ih) != (inv_w, top_h):
#         from PIL import Image
#         inv_np = np.array(Image.fromarray(inv_np).resize((inv_w, top_h), Image.NEAREST))

#     inv_panel  = Image.fromarray(inv_np)
#     main_panel = _fit_into_box(main_image, main_w, top_h, pad=pad_px, bg=bg)

#     # ---------- 4) Build the bottom strip with 4 state images + titles ----------
#     titles = ["Top", "Middle", "Bottom", "Container"]
#     tile_w = S // 4
#     title_bar_h = max(18, int(0.03 * S))  # proportional, but not tiny
#     tiles = []
#     for img_np, title in zip(state_images, titles):
#         tile_img = _fit_into_box(img_np, tile_w, bottom_h - title_bar_h, pad=pad_px, bg=bg)
#         tile_img = _draw_title_bar(tile_img, title, height_px=title_bar_h, bg=bg, fg=fg)
#         tiles.append(tile_img)

#     # If integer division leaves a gap on the right, widen the last tile to fill exactly S
#     last_extra = S - (tile_w * 4)
#     if last_extra != 0:
#         w, h = tiles[-1].size
#         tiles[-1] = tiles[-1].resize((w + last_extra, h), Image.NEAREST)

#     # ---------- 5) Compose final square ----------
#     canvas = Image.new("RGB", (S, S), bg)
#     # Top row
#     canvas.paste(inv_panel, (0, 0))
#     canvas.paste(main_panel, (inv_w, 0))
#     # Bottom row
#     x = 0
#     y = top_h
#     for t in tiles:
#         canvas.paste(t, (x, y))
#         x += t.size[0]

#     return np.array(canvas)

def safe_filename(task_description: str, max_length: int = 50) -> str:
    # Convert to lowercase
    s = task_description.lower()
    # Replace spaces and newlines with underscores
    s = s.replace(' ', '_').replace('\n', '_')
    # Replace any character that is not alphanumeric or underscore with underscore
    s = re.sub(r'[^a-z0-9_]', '_', s)
    # Collapse multiple consecutive underscores into a single underscore
    s = re.sub(r'_+', '_', s)
    # Strip leading and trailing underscores
    s = s.strip('_')
    # Truncate to max_length characters
    return s[:max_length]

def create_captioned_frame(img, text, img_width=256):
    """
    Takes an image and text, and returns a new image with the
    text wrapped and drawn on a black canvas underneath.
    """
    
    # --- 1. Define Text Parameters ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45  # Small font for 256px width
    thickness = 1
    line_type = cv2.LINE_AA
    font_color = (255, 255, 255) # White
    bg_color = (0, 0, 0)         # Black
    
    # Padding for the text area
    v_padding = 10 # Top/bottom padding for text area
    h_padding = 5  # Left/right padding for text area
    line_spacing = 5 # Pixels between lines of text
    
    # --- 2. Wrap Text ---
    words = text.split(' ')
    lines = []
    current_line = ""
    
    # Get the height of a single line of text
    (text_w, text_h), baseline = cv2.getTextSize("A", font, font_scale, thickness)
    
    for word in words:
        # Test a line with the new word
        test_line = f"{current_line} {word}".strip()
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        # Check if the test line fits within the width (with padding)
        if text_width < (img_width - 2 * h_padding):
            current_line = test_line
        else:
            lines.append(current_line) # Add the line that fit
            current_line = word        # Start a new line with the current word
    lines.append(current_line) # Add the last line
    
    # --- 3. Create Caption Canvas ---
    num_lines = len(lines)
    
    # Calculate total height needed for the caption area
    caption_height = (2 * v_padding) + (num_lines * text_h) + max(0, num_lines - 1) * line_spacing
    caption_canvas = np.full((caption_height, img_width, 3), bg_color, dtype=np.uint8)
    
    # --- 4. Draw Each Line of Text on the Canvas ---
    y = v_padding + text_h # Initial y position (top-padding + first-line-height)
    
    for line in lines:
        cv2.putText(caption_canvas,
                    line,
                    (h_padding, y), # Position (x, y)
                    font,
                    font_scale,
                    font_color,
                    thickness,
                    line_type)
        y += text_h + line_spacing # Move y down for the next line
        
    # --- 5. Stack Original Image and New Caption Canvas ---
    final_frame = np.vstack((img, caption_canvas))
    
    return final_frame

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, notes='eval', domain_name="", rollout_dir=None):
    """Saves an MP4 replay of an episode."""
    if rollout_dir is None:
        rollout_dir = os.path.join(os.environ['WORKING_DIR'], "rollouts/mini_behavior", notes, DATE)
    os.makedirs(rollout_dir, exist_ok=True)
    # print abspath of rollout_dir
    print(f"Rollout directory: {os.path.abspath(rollout_dir)}")
    
    
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={domain_name}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=4)
    for idx, img in enumerate(rollout_images):
        # img_with_caption
        # put the task_description[i] into the rollout_images[i]
        
        text_to_show = task_description[idx]
        
        # Make a copy to avoid drawing on the original image array
        img_captioned = create_captioned_frame(img, text_to_show, img.shape[0])
        video_writer.append_data(img_captioned)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path



if __name__ == "__main__":
    all_fail_domains = [] 
    for domain_name in DOMAIN_NAME_LIST:
        problem_id = None 
    # for domain_name in ['watering_houseplants']:
    #     problem_id = 122968710
        while True:
            try:
                env, nl_instruction, nl_instruction_list, problem_id, actions_lst, raw_action_lst = get_mini_behavior_env(
                    domain_name=domain_name,
                    problem_id=problem_id,
                )
                print(colored(f"Domain_name: {domain_name}", "red"))
                break
            except Exception as e:
                print(colored(f"Error occurred while getting environment for domain {domain_name}: {e}", "red"))
                continue
        ever_done = False
        for plan_id in tqdm(range(len(actions_lst)), desc="Processing plans", leave=False):
            actions = actions_lst[plan_id]
            plan = raw_action_lst[plan_id]
            # ! Reset is important here
            env.seed(problem_id) # ! important to set seed before reset()
            obs_info = env.reset()
            image = get_mini_behavior_image(obs_info, env, temp_save=True)

            rollout_images = [image]

            # print("Actions")
            # print(actions)
            # print("Action len")
            # print(len(actions))

            for idx, action in enumerate(tqdm(actions,  desc="Executing actions")):
                # print(f"Executing action {idx + 1}/{len(actions)}: {plan[idx]}")
                obs, reward, done, info = env.step(action)
                image = get_mini_behavior_image(obs, env, temp_save=False)
                rollout_images.append(image)
                
                
            print(f"done is {done}")
            if done:
                ever_done = True
            else:
                print(colored(f"pls run python /home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/single_agent_gameplay_test.py --map_id {problem_id} --domain_name {domain_name} --no_display", 'red'))

            # save mp4
            save_rollout_video(rollout_images, 0, done, f"mini_behavior_{domain_name}", notes="example")
        if not ever_done:
            all_fail_domains.append(domain_name)
            
    for fail_domain_name in all_fail_domains:
        print(colored(f"Domain {fail_domain_name} did not complete any tasks successfully.", "red"))

    print(f"Len original: {len(DOMAIN_NAME_LIST)}")
    print(f"Len failed: {len(all_fail_domains)}")