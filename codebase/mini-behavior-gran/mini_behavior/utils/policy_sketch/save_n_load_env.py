import os 
import pickle 
import json 
import hashlib
from gym_minigrid.wrappers import *
import re
import argparse
import random
import numpy as np
import numpy.random as npr
from mini_behavior.window import Window, redraw, reset, key_handler_primitive
import tempfile
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache
import cv2
from mini_behavior.grid import GridDimension
from pathlib import Path
from mini_behavior.utils.pddl_gen.parse_pddl import obtain_objs_from_problem_instance, obtain_preidicates_categorized_by_objs, format_predicate_dict_to_text
from mini_behavior.utils.pddl_gen.pddl_problem_gen import generate_behavior_pddl, FURNITURE_TYPES
from mini_behavior.utils.pddl_gen.parse_pddl import obtain_domain_model, obtain_domain_and_problem_instance
import time

@lru_cache(maxsize=64)
def _get_font(font_path, size):
    """
    Cached font loader. Provide a TTF path for best quality & scaling.
    Falls back to DejaVuSans or PIL's default if needed.
    """
    try:
        if font_path:
            return ImageFont.truetype(font_path, size)
        # DejaVuSans is shipped with many Pillow builds; fallback to default otherwise
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def _to_pil(img_np, bg=(255, 255, 255)):
    """
    Normalize an incoming image (np array or PIL.Image) to a PIL RGB image.
    Handles grayscale and RGBA (composites onto bg).
    """
    if img_np is None:
        return None
    if isinstance(img_np, Image.Image):
        im = img_np
    else:
        arr = np.asarray(img_np)
        if arr.ndim == 2:  # grayscale
            im = Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.shape[-1] == 4:
            im = Image.fromarray(arr, mode="RGBA")
        else:
            im = Image.fromarray(arr, mode="RGB")

    if im.mode == "RGBA":
        bg_im = Image.new("RGB", im.size, bg)
        bg_im.paste(im, mask=im.split()[3])  # alpha composite
        return bg_im
    elif im.mode != "RGB":
        return im.convert("RGB")
    return im

def _fit_into_box(img_np, box_w, box_h, pad=1, bg=(255, 255, 255), resample=Image.BILINEAR):
    """
    Fit (keep aspect) an image into a target box (box_w x box_h) with padding and background.
    Returns a PIL RGB image of EXACT size (box_w, box_h).
    """
    im = _to_pil(img_np, bg=bg)
    if im is None:
        return Image.new("RGB", (box_w, box_h), bg)

    inner_w = max(1, box_w - 2*pad)
    inner_h = max(1, box_h - 2*pad)
    iw, ih = im.size
    if iw == 0 or ih == 0:
        return Image.new("RGB", (box_w, box_h), bg)

    scale = min(inner_w / iw, inner_h / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    if (new_w, new_h) != (iw, ih):
        im = im.resize((new_w, new_h), resample=resample)

    canvas = Image.new("RGB", (box_w, box_h), bg)
    x = (box_w - new_w) // 2
    y = (box_h - new_h) // 2
    canvas.paste(im, (x, y))
    return canvas

def _draw_title_bar(img, title, height_px=24, bg=(255, 255, 255), fg=(0, 0, 0), font_path=None):
    """
    Append a title bar ABOVE the given image (so total height increases).
    """
    w, h = img.size
    out = Image.new("RGB", (w, h + height_px), bg)
    out.paste(img, (0, height_px))  # image goes under the bar

    draw = ImageDraw.Draw(out)
    # Choose a font size that fits comfortably in the bar height
    size = max(10, min(height_px - 6, 32))
    font = _get_font(font_path, size)
    text = str(title)

    # Center vertically within the bar
    ascent, descent = font.getmetrics()
    text_h = ascent + descent
    y = (height_px - text_h) // 2

    # Center horizontally
    try:
        text_w = draw.textlength(text, font=font)
    except Exception:
        text_w = draw.textbbox((0, 0), text, font=font)[2]
    x = (w - text_w) // 2

    draw.text((x, y), text, fill=fg, font=font)
    return out

def _rounded_rect(draw, xy, radius=6, outline=(0,0,0), width=1, fill=None):
    """
    A thin wrapper over rounded_rectangle with sane defaults.
    """
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except Exception:
        # Fallback: plain rectangle if rounded not available
        draw.rectangle(xy, fill=fill, outline=outline, width=width)

def _measure_block(lines, font, line_spacing=1.05, draw=None):
    """
    Measure a block of multi-line text. Returns (block_width, block_height, line_height).
    """
    lines = lines if lines else [" --- "]
    ascent, descent = font.getmetrics()
    lh = int(round((ascent + descent) * line_spacing))

    max_w = 0
    if draw is None:
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)

    for ln in lines:
        try:
            w = draw.textlength(ln, font=font)
        except Exception:
            w = draw.textbbox((0, 0), ln, font=font)[2]
        if w > max_w:
            max_w = w

    block_h = max(1, lh * max(1, len(lines)))
    return max_w, block_h, lh

# ---------------------- main: optimized renderer ----------------------

def render_inventory_image(on_grid, carrying,
                           fontsize_ratio=200,      # kept for API compatibility (unused)
                           title_fontsize=7,       # interpreted as pixels 
                           body_fontsize=7,         # interpreted as pixels
                           state_images=None,       # list of 4 np arrays [Top, Middle, Bottom, Container]
                           section_gap_px=12,
                           main_image=None,         # np array (agent view)
                           out_size=1024,           # final square size
                           inv_width_frac=0.28,     # fraction of full width for the left inventory strip
                           bottom_height_frac=0.20, # fraction of full height for the bottom row of 4 states
                           pad_px=1,
                           bbox_kwargs=None,
                           bg=(255, 255, 255),
                           fg=(0, 0, 0),
                           font_path=None,
                           resample_main=Image.BILINEAR,
                           resample_tiles=Image.BILINEAR):
    """
    Matplotlib-free, high-performance renderer.

    If `main_image` and `state_images` (len==4) are provided, compose a square panel:
      - Left: inventory (on_grid / carrying) stacked vertically in a single column
      - Middle/Right: main_image (agent view)
      - Bottom: four state images titled Top, Middle, Bottom, Container (full width)

    Otherwise, returns just the inventory panel (same layout region as the top-left strip).

    NOTES
    - `title_fontsize` / `body_fontsize` are in **pixels** (not points).
    - For best results & speed, pass a TrueType `font_path` (e.g., DejaVuSans.ttf).
    - `bbox_kwargs` supported keys: radius, pad, outline, width, fill (fill=None recommended).
    """

    # ------------- layout metrics -------------
    S = int(max(256, out_size))
    inv_w = int(round(inv_width_frac * S))
    bottom_h = int(round(bottom_height_frac * S))
    top_h = S - bottom_h
    main_w = S - inv_w

    # ------------- defaults & caches -------------
    if bbox_kwargs is None:
        bbox_kwargs = dict(radius=6, pad=6, outline=fg, width=1, fill=None)

    on_grid = list(on_grid) if on_grid else []
    carrying = list(carrying) if carrying else []
    titles = ["Top", "Middle", "Bottom", "Container"]

    # ---------- inventory panel (single PIL image) ----------
    inv_panel = Image.new("RGB", (inv_w, top_h), bg)
    draw = ImageDraw.Draw(inv_panel)

    top_margin_px = max(4, int(0.04 * top_h))
    section_gap = int(section_gap_px)
    title_px = int(title_fontsize)
    body_px = int(body_fontsize)
    title_gap_px = 3 # gap between title and box 
    line_spacing = 1.05

    # Fonts (will be resized if needed)
    title_font = _get_font(font_path, title_px)
    body_font  = _get_font(font_path, body_px)

    # Measure required height with current sizes
    # Title heights
    ascent_t, descent_t = title_font.getmetrics()
    title_h = ascent_t + descent_t

    # Body heights
    _, grid_h, lh_body = _measure_block(on_grid, body_font, line_spacing, draw)
    _, carry_h, _       = _measure_block(carrying, body_font, line_spacing, draw)

    required = top_margin_px + (title_h + title_gap_px + grid_h) + section_gap + (title_h + title_gap_px + carry_h) + top_margin_px

    # If doesn't fit, scale both fonts down proportionally (integer px)
    if required > top_h:
        scale = max(0.3, (top_h - 2) / max(1, required))
        new_title_px = max(9, int(title_px * scale))
        new_body_px  = max(8, int(body_px  * scale))
        if new_title_px != title_px:
            title_px = new_title_px; title_font = _get_font(font_path, title_px)
        if new_body_px != body_px:
            body_px  = new_body_px;  body_font  = _get_font(font_path, body_px)

        ascent_t, descent_t = title_font.getmetrics()
        title_h = ascent_t + descent_t
        _, grid_h, lh_body = _measure_block(on_grid, body_font, line_spacing, draw)
        _, carry_h, _       = _measure_block(carrying, body_font, line_spacing, draw)
        required = top_margin_px + (title_h + title_gap_px + grid_h) + section_gap + (title_h + title_gap_px + carry_h) + top_margin_px

    # Draw section helper
    def _draw_section(title, items, y_cursor):
        # Title (centered)
        try:
            tw = draw.textlength(title, font=title_font)
        except Exception:
            tw = draw.textbbox((0, 0), title, font=title_font)[2]
        tx = (inv_w - tw) // 2
        draw.text((tx, y_cursor), title, fill=fg, font=title_font)
        y_cursor += title_h + title_gap_px

        # Body (boxed, centered with rounded rectangle)
        lines = items if items else [" --- "]

        # Measure bounding box for lines
        max_w, block_h, lh = _measure_block(lines, body_font, line_spacing, draw)
        pad = int(bbox_kwargs.get("pad", 6))
        left = max(0, (inv_w - max_w) // 2 - pad)
        right = min(inv_w - 0, left + max_w + 2*pad)
        top = y_cursor
        bottom = y_cursor + block_h + 2*pad

        _rounded_rect(
            draw,
            (left, top, right, bottom),
            radius=int(bbox_kwargs.get("radius", 6)),
            outline=bbox_kwargs.get("outline", fg),
            width=int(bbox_kwargs.get("width", 1)),
            fill=bbox_kwargs.get("fill", None),
        )

        # Draw lines
        y_text = top + pad
        for ln in lines:
            try:
                lw = draw.textlength(ln, font=body_font)
            except Exception:
                lw = draw.textbbox((0, 0), ln, font=body_font)[2]
            lx = (inv_w - lw) // 2
            draw.text((lx, y_text), ln, fill=fg, font=body_font)
            y_text += lh

        return bottom  # new cursor baseline

    y = top_margin_px
    y = _draw_section("on grid", on_grid, y)
    y += section_gap
    y = _draw_section("carrying", carrying, y)

    # If only inventory requested, return that region as numpy array
    full_compose = (main_image is not None) and (state_images is not None) and (len(state_images) == 4)
    if not full_compose:
        return np.array(inv_panel)

    # ---------- main image panel ----------
    main_panel = _fit_into_box(main_image, main_w, top_h, pad=pad_px, bg=bg, resample=resample_main)

    # ---------- bottom strip (4 states + titles) ----------
    tile_w = S // 4
    title_bar_h = max(18, int(0.03 * S))
    tiles = []
    for img_np, t in zip(state_images, ["Top", "Middle", "Bottom", "Container"]):
        tile_img = _fit_into_box(img_np, tile_w, bottom_h - title_bar_h, pad=pad_px, bg=bg, resample=resample_tiles)
        tile_img = _draw_title_bar(tile_img, t, height_px=title_bar_h, bg=bg, fg=fg, font_path=font_path)
        tiles.append(tile_img)

    # Fix last tile if integer division left a gap
    last_extra = S - (tile_w * 4)
    if last_extra != 0:
        w, h = tiles[-1].size
        tiles[-1] = tiles[-1].resize((w + last_extra, h), resample=Image.NEAREST)

    # ---------- compose final square ----------
    canvas = Image.new("RGB", (S, S), bg)
    canvas.paste(inv_panel, (0, 0))
    canvas.paste(main_panel, (inv_w, 0))

    x, y = 0, top_h
    for t in tiles:
        canvas.paste(t, (x, y))
        x += t.size[0]

    return np.array(canvas)


def get_mini_behavior_image(obs, env, add_noise= True, image_size= 256, temp_save=False):
    
    # raw image
    # main_image_start_time = time.time()
    image = env.render('rgb_array', tile_size = 32)
    # print(f"Main image rendering time: {time.time() - main_image_start_time:.2f} seconds")
    # furniture image 
    # furniture_image_time = time.time()
    furniture_img = np.copy(env.furniture_view)

    # i, j = env.agent.cur_pos
    i, j = env.agent_pos
    ymin = j * 32
    ymax = (j + 1) * 32
    xmin = i * 32
    xmax = (i + 1) * 32

    furniture_img[ymin:ymax, xmin:xmax, :] = GridDimension.render_agent(
        furniture_img[ymin:ymax, xmin:xmax, :], env.agent_dir)
    furniture_img = env.render_furniture_states(furniture_img)

    # print(f"Furniture image rendering time: {time.time() - furniture_image_time:.2f} seconds")
    # stack image and img, image 0.8, img 0.2

    image = cv2.addWeighted(image, 0.8, furniture_img, 0.2, 0)

    # state_and_invent_image_start_time = time.time()
    state_images = env.render_states(want_print=False)
    # reverse 
    state_images = state_images[::-1]
    on_grid, carrying = compute_inventory_lists(env)
    # print(f"State and inventory image rendering time: {time.time() - state_and_invent_image_start_time:.2f} seconds")

    # compose_start_time = time.time()
    image = render_inventory_image(on_grid, carrying, state_images=state_images, main_image=image, inv_width_frac=0.2, bottom_height_frac=0.15, out_size=image_size, pad_px=1, fontsize_ratio=72, section_gap_px=1)
    
    # print(f"Composing image time: {time.time() - compose_start_time:.2f} seconds")
    
    if add_noise:
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.addWeighted(image, 0.9, noise, 0.1, 0)
        
        
    if temp_save:
        # opencv2 save this 
        temp_file = os.path.join(os.environ['WORKING_DIR'], "temp_images", "mini_behavior_img.jpg")
        Path(temp_file).parent.mkdir(parents=True, exist_ok=True)
        # convert from RGB to BGR
        image_new = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_file, image_new)

    return image



def compute_inventory_lists(env):
    """
    Pure function: derive inventory lists from env.
    Returns (on_grid, carrying) as lists of strings.
    """
    carrying = [
        obj.name
        for obj in env.obj_instances.values()
        if obj.check_abs_state(env, 'inhandofrobot')
    ]

    on_grid = []
    for objs in env.objs.values():
        for obj in objs:
            if obj.name not in carrying and obj.type != 'door':
                on_grid.append(obj.name)

    return on_grid, carrying


def test_load_and_save():
    env_filepath = "/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/cleaning_a_car/p335829852-cleaning_a_car_plans.json"
     
    exec_length = 10
    obs_lst, reward_lst, done_lst, info_lst, env, env_info_dict = load_env_from_json(env_filepath, exec_length=exec_length)
    
    # now save this env and partial action again into a tempfile

    lefted_executed_actions = env_info_dict['executed_actions'][exec_length :]
    executed_actions = env_info_dict['executed_actions'][:exec_length]
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # use save_env
        save_env_to_json(env, executed_actions, env_info_dict['map_id'], env_info_dict['domain_name'], save_path=tmp_file.name)
        
        tmp_filepath = tmp_file.name
        print(f"Temporary file saved at {tmp_filepath}")
        
    # now load again
    obs_lst2, reward_lst2, done_lst2, info_lst2, env2, env_info_dict2 = load_env_from_json(tmp_filepath)
    
    # continue the lefted actions
    for action_str in lefted_executed_actions:
        action = parse_action(action_str, env2)
        obs, reward, done, info = env2.step(action)
        obs_lst2.append(obs)
        reward_lst2.append(reward)
        done_lst2.append(done)
        info_lst2.append(info)
    
    assert done_lst2[-1], "The loaded actions did not reach the goal."
    assert not done_lst[-1], "The original actions should not reach the goal with partial execution."
    os.remove(tmp_filepath)
        
        
        
def test_load_and_reach_goal():
    env_filepath = "/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_plans/cleaning_a_car/p335829852-cleaning_a_car_plans.json"
    obs_lst, reward_lst, done_lst, info_lst, env, env_info_dict = load_env_from_json(env_filepath)
    assert done_lst[-1], "The loaded actions did not reach the goal."

def parse_action(action_str: str, env):
    
    action_str = action_str.replace('turn-left', 'left').replace('turn-right', 'right')
    actions = env.actions

    for action in actions:
        action_name = action.name
        
        if action_str.startswith(action_name):
            return action
        
    raise ValueError(f"Action '{action_str}' not recognized in environment.")
        

def hash8hex(s):
    return hashlib.sha256(s.encode()).hexdigest()[:8]

# ! Main function 1
def save_env_to_json(env, executed_actions, map_id, domain_name, additional_info_dict=None, save_path=None):
    # obtain env_id, map_id, domain_name 
    env_id = env.spec.id
    env_info_dict = { # it will be the same structure as the plan info dict
        "env_id": env_id,
        "map_id": map_id,
        "domain_name": domain_name,
        "executed_actions": executed_actions,
    }
    if additional_info_dict is not None:
        env_info_dict.update(additional_info_dict)

    assert 'WORKING_DIR' in os.environ, "Please set WORKING_DIR environment variable."
    
    save_dir = os.path.join(os.environ['WORKING_DIR'], 'data/02_intermediate/env_cache')
    os.makedirs(save_dir, exist_ok=True)
    # give 8 char hash for the plans
    hash_str = hash8hex(str(executed_actions))

    if save_path is not None:
        env_filepath = save_path
    else:
        env_filepath = os.path.join(save_dir, f'env_{domain_name}_map_{map_id}_hash_{hash_str}.json')
    with open(env_filepath, 'w') as f:
        json.dump(env_info_dict, f, indent=2)


# ! Main function 2
def load_env_from_json(env_filepath, exec_length=None, run_plan=False, want_render=False):
    assert 'WORKING_DIR' in os.environ, "Please set WORKING_DIR environment variable."
    full_env_filepath = os.path.join(os.environ['WORKING_DIR'], env_filepath)
    assert os.path.exists(full_env_filepath), f"File {full_env_filepath} does not exist."
    with open(full_env_filepath, 'r') as f:
        env_info_dict = json.load(f)
    # now we load the env and execute the actions
    env = gym.make(env_info_dict['env_id'])
    env.teleop_mode()
    # load window if want_render
    if want_render:
        window = Window('mini_behavior - ' + str(env_info_dict['env_id']) + ' ' + str(env_info_dict['map_id']))
        key_handler_primitive_partial = partial(key_handler_primitive, window=window, env=env)
    else:
        window = None
    
    
    
    # set seed 
    random.seed(int(env_info_dict['map_id']))
    npr.seed(int(env_info_dict['map_id']))
    reset(int(env_info_dict['map_id']), env=env, window=window)
    # refresh the env 
    env.step(env.actions.left) 
    env.step(env.actions.right) 
    
    # execute the actions
    if "executed_actions" in env_info_dict:
        action_list = env_info_dict['executed_actions']
    else:
        if run_plan:
            if "plans" in env_info_dict:
                action_list = env_info_dict['plans'][0]
            else:
                action_list = []
        else:
            action_list = []
        
    env_info_dict['executed_actions'] = action_list
        
    obs_collection = []
    reward_collection = []
    done_collection = []
    info_collection = []
    
    if isinstance(exec_length, int):
        action_list = action_list[:exec_length]
    
    for action_str in action_list:
        action = parse_action(action_str, env)
        obs, reward, done, info = env.step(action)
        obs_collection.append(obs)
        reward_collection.append(reward)
        done_collection.append(done)
        info_collection.append(info)
        
    return obs_collection, reward_collection, done_collection, info_collection, env, env_info_dict, window
    
# ! Main function 3
def load_env_from_pickled_sketch_data(sketch_data, want_render=False, image_for_vla=False):

    # now we load the env and execute the actions
    env = gym.make(sketch_data['env_id'])
    env.teleop_mode()
    # load window if want_render
    if want_render:
        window = Window('mini_behavior - ' + str(sketch_data['env_id']) + ' ' + str(sketch_data['map_id']))
        key_handler_primitive_partial = partial(key_handler_primitive, window=window, env=env)
    else:
        window = None
    
    # set seed 
    random.seed(int(sketch_data['map_id']))
    npr.seed(int(sketch_data['map_id']))
    reset(int(sketch_data['map_id']), env=env, window=window)
    # refresh the env 
    env.step(env.actions.left) 
    obs, reward, done, info = env.step(env.actions.right) 
    
    # execute the actions
    obs_collection = []
    text_obs_collection = []
    action_collection = [] 
    reward_collection = []
    done_collection = []
    info_collection = []
    action_list = sketch_data['stored_info']['executed_actions']
    action_list_among_performed_rules = []
    action_count = 0 
    action_map_to_rule_id = dict()
    for local_rule_id, (rule_step, subgoal_str, rule_str, action_l) in enumerate(sketch_data['stored_info']['performed_rules']):
        action_list_among_performed_rules.extend(action_l)
        for a_str in action_l:
            action_map_to_rule_id[action_count] = local_rule_id
            action_count += 1
    if action_list != action_list_among_performed_rules:
        print("Action list from stored_info and performed_rules do not match!")
        raise ValueError("Action list mismatch in sketch data.")
    
    domain_filepath = Path(__file__).parent.parent / 'pddl_gen' / 'minibehavior.pddl'
    seed_for_text_obs = int(time.time()) % (2**32 - 1)
    assert domain_filepath.exists(), f"Domain file {domain_filepath} does not exist."
    for action_id, action_str in enumerate(action_list):
        if image_for_vla:
            obs_collection.append(get_mini_behavior_image(None, env, add_noise=True))
        else:
            obs_collection.append(obs)
            
        # ! get text observation
        pddl_problem_str = generate_behavior_pddl(env)
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_problem_file:
            temp_problem_file.write(pddl_problem_str)
            temp_problem_file_path = temp_problem_file.name
        domain_model, problem_instance = obtain_domain_and_problem_instance(domain_filepath, temp_problem_file_path)
        
        Path(temp_problem_file_path).unlink()  # Delete the previous temporary file
        objs_in_env = obtain_objs_from_problem_instance(problem_instance, domain_model, "pos-")
        predicate_dict = obtain_preidicates_categorized_by_objs(problem_instance, objs_in_env)
        formatted_state_text = format_predicate_dict_to_text(
            obj_to_predicates = predicate_dict,
            num_of_variants = 1,
            object_shuffle_prob = 1.0,
            predicate_shuffle_prob = 1.0,
            seed = seed_for_text_obs,
        )[0]
        text_obs_collection.append(formatted_state_text)
        
        action = parse_action(action_str, env)
        action_collection.append(action)
        obs, reward, done, info = env.step(action)
        if window is not None:
            redraw(obs, env, window)
        reward_collection.append(reward)
        done_collection.append(done)
        # info add instruction info
        rule_id = action_map_to_rule_id[action_id]
        if 'nl_instructions' in sketch_data:
            single_instr = sketch_data['nl_instructions']['instr_list'][rule_id]
            info['instr_at_cur_step'] = single_instr
            info['overall_instr'] = sketch_data['nl_instructions']['compact_instr']
            info['rule_at_cur_step'] = sketch_data['nl_instructions']['rule_list'][rule_id]
        info_collection.append(info)
        if done:
            print("Goal reached during loading from pickled sketch data.")
        
    # # DEBUG
    # if image_for_vla:
    #     # save the images 
    #     temp_dir = os.path.join(os.environ['WORKING_DIR'], "temp_images", "vla_images")
    #     Path(temp_dir).mkdir(parents=True, exist_ok=True)
    #     for i, img in enumerate(obs_collection):
    #         img_path = os.path.join(temp_dir, f"img_{i}.png")
    #         img_c = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(img_path, img_c)
    #         print(f"Saved image to {img_path}")
    #     breakpoint()

    return obs_collection, text_obs_collection, action_collection, reward_collection, done_collection, info_collection, env, sketch_data, window
    
    
# ! function for creating general policy sketch
def init_env_for_policy_sketch(env_filepath, want_render=False):
    obs_lst, reward_lst, done_lst, info_lst, env, env_info_dict, window = load_env_from_json(env_filepath, want_render=want_render)
    env_id = env_info_dict['env_id']
    map_id = env_info_dict['map_id']
    domain_name = env_info_dict['domain_name']
    return env, map_id, domain_name, window, env_id


def save_gps_data(
    stored_info,
    goal_achieved,
    map_id,
    domain_name,
    width,
    env_id,
    goal_count_feature,
    output_dir="data/02_intermediate/gps_data",
):
    output_dir = os.path.join(os.environ['WORKING_DIR'], output_dir, domain_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_pickle_filename = os.path.join(output_dir, f'gps_data_map_{map_id}_width_{width}_env_{env_id}_goal_{goal_achieved}.pkl')
    
    data_to_save = {
        "stored_info": stored_info,
        "goal_achieved": goal_achieved,
        "map_id": map_id,
        "domain_name": domain_name,
        "width": width,
        "env_id": env_id,
        'goal_count_feature': goal_count_feature,
    }
    with open(output_pickle_filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"GPS data saved to {output_pickle_filename}")

    
if __name__ == '__main__':
    # test_load_and_reach_goal()
    # test_load_and_save()
    
    test_pickle_fp = "/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/02_intermediate/gps_data/cleaning_shoes/gps_data_map_339833990_width_3_env_MiniGrid-CleaningShoes-12x12-N2-v0_goal_True.pkl"
    with open(test_pickle_fp, 'rb') as f:
        sketch_data = pickle.load(f)
    obs_collection, text_obs_collection, action_collection, reward_collection, done_collection, info_collection, env, sketch_data, window = load_env_from_pickled_sketch_data(sketch_data, want_render=True)