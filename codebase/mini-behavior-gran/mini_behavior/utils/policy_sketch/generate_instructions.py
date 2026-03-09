from pprint import pprint
import re
from typing import List, Tuple, Dict, Any, Optional
import pickle
from mini_behavior.utils.pddl_gen.pddl_problem_gen import GOAL_CLAUSES
from collections import Counter as CollectionCounter
from deprecated import deprecated
from tqdm.auto import tqdm
import os 
from glob import glob
from multiprocessing import Pool
from time import sleep

# -------------------------------
# Utils
# -------------------------------

def normalize_step(step_line):
    """
    Splits a step line into its content and a normalized version.
    
    It replaces all instances of "(word) (number)" with "(word) <ID>"
    to find repeating patterns. It also returns the first object
    it found (e.g., "plate 2") for use in the summary.
    
    Example:
    Input: "...at the location of plate 2... not holding plate 2..."
    Returns: 
    - Normalized: "...at the location of plate <ID>... not holding plate <ID>..."
    - Object: "plate 2"
    """
    try:
        # Split the line at the first colon
        prefix, content = step_line.split(':', 1)
        content = content.strip()
    except ValueError:
        # Not a line with a ':', return as-is
        return step_line, None

    # Find the first instance of object types followed by a number
    # We look for patterns like "plate 2", "cabinet 0" but exclude common words like "above", "count", etc.
    # The pattern looks for word + number, but we filter by looking for specific object-indicating contexts
    
    # Define all possible object types (using spaces instead of underscores)
    normal_items = [
        'apple', 'backpack', 'ball', 'banana', 'basket', 'beef', 'blender', 'book', 'bow', 'bread',
        'cake', 'calculator', 'candy', 'candle', 'carton', 'casserole', 'chicken', 'chip', 'cookie',
        'date', 'document', 'dustpan', 'egg', 'fish', 'floor', 'folder', 'fork', 'gym shoe',
        'hamburger', 'hammer', 'hardback', 'highlighter', 'jewelry', 'juice', 'kettle', 'lemon',
        'lettuce', 'marker', 'necklace', 'notebook', 'oatmeal', 'olive', 'package', 'pan', 'pen',
        'pencil', 'plate', 'plywood', 'pop', 'pot plant', 'printer', 'radish', 'salad', 'sandwich',
        'saw', 'shoe', 'soap', 'sock', 'soup', 'spoon', 'strawberry', 'sugar', 'tea bag', 'teapot',
        'toilet', 'tomato', 'vegetable oil', 'water', 'window'
    ]
    
    cutting_items = ['carving knife', 'knife']
    
    cleaning_items = ['broom', 'rag', 'scrub brush', 'towel']
    
    furniture_items = [
        'ashcan', 'bed', 'bin', 'box', 'bucket', 'cabinet', 'car', 'chair', 'countertop', 'door',
        'electric refrigerator', 'wall', 'shelf', 'shower', 'sink', 'sofa', 'stove', 'table'
    ]
    
    all_objects = normal_items + cutting_items + cleaning_items + furniture_items
    
    # Create regex pattern that matches any of these objects followed by a number
    # Handle multi-word objects (e.g., "tea bag", "pot plant") by escaping and preserving spaces
    # Sort by length (longest first) to match longer object names first (e.g., "carving knife" before "knife")
    all_objects_sorted = sorted(all_objects, key=len, reverse=True)
    
    # Escape each object name for regex, preserving spaces
    escaped_objects = [re.escape(obj) for obj in all_objects_sorted]
    pattern = r'((?:' + '|'.join(escaped_objects) + r')\s+\d+)'
    
    object_matches = re.findall(pattern, content, re.IGNORECASE)
    found_object = object_matches[0] if object_matches else None
    
    # Create a normalized version by replacing ALL instances of (word) (number)
    # This makes "...move toward plate 2" and "...move toward plate 7"
    # identical ("...move toward plate <ID>")
    normalized_content = re.sub(r'(\b[a-zA-Z]+\b)\s+\d+', r'\1 <ID>', content)
    
    return normalized_content, found_object

def condense_planning_script(script):
    """
    Parses a verbose planning script, dynamically identifies ALL 
    repeating loops, and condenses the output by summarizing repetitive steps.
    This function can handle multiple different templates in one script.
    """
    
    # 1. Parse all steps and create normalized versions
    all_steps = [s.strip() for s in script.strip().split('\n') if s.strip()]
    if not all_steps:
        return ""

    normalized_steps = []
    found_objects = []
    for step in all_steps:
        norm, obj = normalize_step(step)
        normalized_steps.append(norm)
        found_objects.append(obj)

    num_steps = len(all_steps)
    
    # Track which steps have been processed (to avoid overlapping templates)
    processed = [False] * num_steps
    condensed_output = []
    
    current_pos = 0
    
    # 2. Iteratively find templates throughout the script
    while current_pos < num_steps:
        # Skip already processed steps (shouldn't happen with current logic, but safety check)
        if processed[current_pos]:
            current_pos += 1
            continue
        
        # Find the best template starting at current_pos
        best_template = find_best_template_at_position(
            normalized_steps, found_objects, current_pos, num_steps, processed
        )
        
        if best_template is None:
            # No template found at this position, add the step as-is
            condensed_output.append(all_steps[current_pos])
            processed[current_pos] = True
            current_pos += 1
        else:
            # Unpack template information
            template_start, template_length, repeat_positions = best_template
            
            # Add any steps between current_pos and template_start that weren't processed
            for i in range(current_pos, template_start):
                if not processed[i]:
                    condensed_output.append(all_steps[i])
                    processed[i] = True
            
            # Add the first instance of the template
            for i in range(template_start, template_start + template_length):
                condensed_output.append(all_steps[i])
                processed[i] = True
            
            # Add summary lines for each repetition
            for rep_start in repeat_positions:
                rep_end = rep_start + template_length - 1
                obj = found_objects[rep_start] or "the next target"
                
                if template_start + 1 < template_start + template_length:
                    summary = (
                        f"At step {rep_start } to {rep_end }, "
                        f"repeat step {template_start } to {template_start + template_length} "
                        f"for {obj}."
                    )
                else:
                    summary = (
                        f"At step {rep_start }, "
                        f"repeat step {template_start} "
                        f"for {obj}."
                    )
                condensed_output.append(summary)
                
                # Mark these steps as processed
                for i in range(rep_start, rep_start + template_length):
                    processed[i] = True
            
            # Move to the position right after the last processed repetition
            last_rep_end = repeat_positions[-1] + template_length if repeat_positions else template_start + template_length
            current_pos = last_rep_end
    
    return "\n".join(condensed_output)


def find_best_template_at_position(normalized_steps, found_objects, start_pos, num_steps, processed):
    """
    Find the best repeating template starting at start_pos.
    
    Returns:
        tuple: (template_start_index, template_length, list_of_repeat_positions) or None
    """
    best_template = None
    best_score = -1
    
    for start in range(start_pos, num_steps):
        # Skip if this position is already processed
        if processed[start]:
            continue
        
        # Max length is half the remaining steps
        for length in range(1, max(2, (num_steps - start) // 2 + 1)):
            # Skip if any step in the template range is already processed
            if any(processed[i] for i in range(start, min(start + length, num_steps))):
                continue
            
            # Define the template and the block immediately after it
            template = normalized_steps[start : start + length]
            next_block = normalized_steps[start + length : start + 2 * length]
            
            # Check if we found a matching pattern
            if template == next_block:
                # Check if any steps in next_block are already processed
                if any(processed[i] for i in range(start + length, min(start + 2 * length, num_steps))):
                    continue
                
                # Count how many times this pattern repeats
                repeat_positions = []
                check_index = start + length
                
                while check_index + length <= num_steps:
                    check_block = normalized_steps[check_index : check_index + length]
                    
                    # Skip if any step in this block is processed
                    if any(processed[i] for i in range(check_index, check_index + length)):
                        break
                    
                    if check_block == template:
                        repeat_positions.append(check_index)
                        check_index += length
                    else:
                        break
                
                # We need at least one repetition to consider it a template
                if not repeat_positions:
                    continue
                
                repeat_count = len(repeat_positions)
                
                # Check if the objects change between cycles (preferred pattern)
                obj_first = found_objects[start]
                obj_next = found_objects[start + length] if start + length < num_steps else None
                objects_differ = obj_first and obj_next and obj_first != obj_next
                
                # Score: prioritize patterns with different objects and high repeat counts
                # For patterns with different objects, prefer higher repeat counts over longer patterns
                # This ensures we find the smallest repeating unit rather than larger composite patterns
                if objects_differ:
                    score = 10000 + repeat_count * 100 - length  # Higher repeats better, shorter patterns better
                else:
                    score = length * 100 + repeat_count * 10  # Fallback to length preference
                
                if score > best_score:
                    best_score = score
                    best_template = (start, length, repeat_positions)
        
        # If we found a template starting at start_pos, return it immediately
        # This ensures we process templates in order
        if best_template and best_template[0] == start_pos:
            return best_template
    
    return best_template


def _split_top_level(s: str, sep: str = ",") -> List[str]:
    """Split by sep but only at top level (ignoring commas inside (), [], {})."""
    out, buf, depth = [], [], 0
    brackets = {"(": ")", "[": "]", "{": "}"}
    closing = set(brackets.values())
    for ch in s:
        if ch in brackets:
            depth += 1
            buf.append(ch)
        elif ch in closing:
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
        else:
            buf.append(ch)
    piece = "".join(buf).strip()
    if piece:
        out.append(piece)
    return out

def _strip_outer(s: str, left: str, right: str) -> str:
    s = s.strip()
    if s.startswith(left) and s.endswith(right):
        return s[len(left):-len(right)]
    return s

# -------------------------------
# Rule parsing & verbalization
# -------------------------------

_RULE_RE = re.compile(
    r"^Rule\(\s*Preconditions:\s*\[(?P<pre>.*)\]\s*->\s*Effects:\s*\[(?P<eff>.*)\]\s*\)\s*$",
    re.S
)

def parse_rule(rule_str: str) -> Tuple[List[str], List[str]]:
    """
    Parse a 'Rule(Preconditions: [ ... ] -> Effects: [ ... ])' string
    into two lists of tokens (preconditions, effects).
    """
    m = _RULE_RE.match(rule_str.strip())
    if not m:
        # Fallback: try to recover something useful
        return [], []
    pre_raw, eff_raw = m.group("pre"), m.group("eff")
    pre = [x.strip() for x in _split_top_level(pre_raw)]
    eff = [x.strip() for x in _split_top_level(eff_raw)]
    return pre, eff

def _unnegate(token: str) -> Tuple[bool, str]:
    token = token.strip()
    if token.startswith("Not "):
        return True, token[4:].strip()
    return False, token

def _fn_and_args(token: str) -> Tuple[str, List[str]]:
    """Extract function name and args from 'Fn(arg1, arg2, ...)'."""
    token = token.strip()
    i = token.find("(")
    j = token.rfind(")")
    if i == -1 or j == -1 or j < i:
        return token, []
    fn = token[:i].strip()
    args = token[i+1:j].strip()
    args = [a.strip() for a in _split_top_level(args)]
    return fn, args

# Templates for turning structured predicates into English
# Extend/override these per domain as needed.
COMPARATOR_WORDS = {
    ">0": "above 0",
    "=0": "0",
    "<=0": "0 or less",
    ">=1": "at least 1",
    ">": "greater than",
    "<": "less than",
    "=": "equal to",
}

def _clean_name(x: str) -> str:
    # also turn "apple tomato" into "apple and tomato"
    output = re.sub(r"\b(\w+)\s+(\w+)\b", r"\1 and \2", x)
    # Turn 'plywood_3' into 'plywood 3'
    output = output.replace("_", " ")
    return output

def _extract_partial_kwargs(count_func_name: str, token: str) -> Dict[str, str]:
    """
    Extract keyword arguments from a partial function reference in the Count token.
    E.g., "count_not_inside_target_type(target_type_name=shelf)" -> {"target_type_name": "shelf"}
    """
    kwargs = {}
    if "(" in count_func_name and ")" in count_func_name:
        # Extract the part between parentheses
        start = count_func_name.index("(")
        end = count_func_name.rindex(")")
        kwargs_str = count_func_name[start+1:end]
        
        # Parse comma-separated key=value pairs
        if kwargs_str.strip():
            for pair in kwargs_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    kwargs[key.strip()] = value.strip()
    
    return kwargs

def _verbalize_count_feature(entity: str, comp: str, count_func_name: str, context: str) -> str:
    """
    Verbalize a Count feature based on the count function name.
    
    Args:
        entity: The object type being counted (e.g., 'carton', 'plywood')
        comp: The comparison operator (e.g., '>0', '=0', 'decrease')
        count_func_name: The name of the count function (e.g., 'count_isolated', 'count_not_inside_target_type(target_type_name=shelf)')
        context: 'precondition' or 'effect'
    """
    entity_phrase = _clean_name(entity)
    comp = comp.strip()
    
    # Extract any kwargs from the count function name
    kwargs = _extract_partial_kwargs(count_func_name, count_func_name)
    
    # Get the base function name (without kwargs)
    base_func_name = count_func_name.split("(")[0] if "(" in count_func_name else count_func_name
    
    # Map count function names to natural language descriptions
    count_descriptions = {
        "count_isolated": f"isolated {entity_phrase}",
        "count_not_cleaned": f"uncleaned {entity_phrase}",
        "count_not_wiped": f"unwiped {entity_phrase}",
        "count_not_sliced": f"unsliced {entity_phrase}",
        "count_closed": f"closed {entity_phrase}",
        "count_not_soaked": f"dry {entity_phrase}",
        "count_not_toggled": f"untoggled {entity_phrase}",
        "count_not_onfloor": f"{entity_phrase} not on floor",
        "count_incomplete_salad_place": f"incomplete salad places",
        "count_not_complete_first_layer_salad": f"incomplete first layer salads",
        "count_not_complete_tables": f"incomplete tables",
    }
    
    # For functions with target parameters, extract the target from kwargs
    if base_func_name == "count_not_inside_target_type":
        target = kwargs.get("target_type_name", "target")
        target_phrase = _clean_name(target)
        feature_desc = f"{entity_phrase} not inside {target_phrase}"
    elif base_func_name == "count_not_inside_target_object":
        target = kwargs.get("container_obj_name", "target")
        target_phrase = _clean_name(target)
        feature_desc = f"{entity_phrase} not inside {target_phrase}"
    elif base_func_name == "count_not_ontop":
        target = kwargs.get("surface_obj_name", "target")
        target_phrase = _clean_name(target)
        feature_desc = f"{entity_phrase} not on {target_phrase}"
    elif base_func_name == "count_not_ontop_sometype":
        target = kwargs.get("surface_type", "target type")
        target_phrase = _clean_name(target)
        feature_desc = f"{entity_phrase} not on {target_phrase}"
    elif base_func_name == "count_not_near_target_type":
        target = kwargs.get("target_type_name", "target")
        target_phrase = _clean_name(target)
        feature_desc = f"{entity_phrase} not near {target_phrase}"
    elif base_func_name == "count_not_samelocation_sometype":
        target = kwargs.get("target_type_name", "target")
        target_phrase = _clean_name(target)
        feature_desc = f"{entity_phrase} not at same location as {target_phrase}"
    else:
        # Use the mapping or fall back to cleaning the function name
        feature_desc = count_descriptions.get(
            base_func_name,
            _clean_name(base_func_name.replace("count_", "")) + f" {entity_phrase}"
        )
    
    # Handle comparison/delta
    if comp.lower() in ("decrease", "increase"):
        if comp.lower() == "decrease":
            phrase = f"reduce the count of {feature_desc}"
        else:
            phrase = f"increase the count of {feature_desc}"
        return phrase if context == "effect" else phrase
    else:
        comp_word = COMPARATOR_WORDS.get(comp, comp)
        phrase = f"the count of {feature_desc} is {comp_word}"
        return f"ensure {phrase}" if context == "effect" else phrase

def verbalize_predicate(token: str, context: str = "precondition") -> str:
    """
    Convert common symbolic tokens to English phrases.
    `context` is 'precondition' or 'effect'.
    """
    neg, core = _unnegate(token)
    fn, args = _fn_and_args(core)

    # Handle Count in two shapes:
    #  - Count(entity, >0, count_func_name)    (state/comparison)
    #  - Count(entity, decrease, count_func_name) (effect)
    if fn == "Count" and len(args) == 3:
        entity, comp, count_func_name = args
        return _verbalize_count_feature(entity, comp, count_func_name, context)

    if fn == "DistanceToNearest":
        # DistanceToNearest(obj, >0 | =0 | decrease)
        if len(args) == 2:
            obj, state = args
            obj = _clean_name(obj)
            state_l = state.lower()
            if state_l == ">0":
                phrase = f"not yet at {obj}"
            elif state_l == "=0":
                phrase = f"you are at {obj}"
            elif state_l == "decrease":
                phrase = f"move toward {obj}"
            else:
                phrase = f"distance condition ({state}) for {obj}"
            return f"try to {phrase}" if context == "effect" else phrase

    if fn == "Holding" and len(args) == 1:
        obj = _clean_name(args[0])
        if neg:
            return "not holding " + obj
        else:
            # Effect-wise we can make it imperative
            return ("pick up " + obj) if context == "effect" else ("holding " + obj)

    if (fn == "isOpened" or fn == "IsOpened") and len(args) == 1:
        obj = _clean_name(args[0])
        if neg:
            phrase = f"{obj} is closed"
            return f"close {obj}" if context == "effect" else phrase
        else:
            phrase = f"{obj} is opened"
            return f"open {obj}" if context == "effect" else phrase

    if (fn == "isToggled" or fn == "IsToggled") and len(args) == 1:
        obj = _clean_name(args[0])
        if neg:
            phrase = f"{obj} is off"
            return f"turn off {obj}" if context == "effect" else phrase
        else:
            phrase = f"{obj} is on"
            return f"turn on {obj}" if context == "effect" else phrase

    if (fn == "isSoaked" or fn == "IsSoaked") and len(args) == 1:
        obj = _clean_name(args[0])
        if neg:
            phrase = f"{obj} is not soaked"
            return f"dry {obj}" if context == "effect" else phrase
        else:
            phrase = f"{obj} is soaked"
            return f"soak {obj}" if context == "effect" else phrase

    if (fn == "isDustFree" or fn == "IsDustFree") and len(args) == 1:
        obj = _clean_name(args[0])
        if neg:
            phrase = f"{obj} is dusty"
            return f"make {obj} dusty" if context == "effect" else phrase
        else:
            phrase = f"{obj} is dust-free"
            return f"dust {obj}" if context == "effect" else phrase

    if (fn == "isCleaned" or fn == "IsCleaned") and len(args) == 1:
        obj = _clean_name(args[0])
        if neg:
            phrase = f"{obj} is dirty"
            return f"make {obj} dirty" if context == "effect" else phrase
        else:
            phrase = f"{obj} is cleaned"
            return f"clean {obj}" if context == "effect" else phrase

    # Generic fallback
    generic = core
    if neg:
        generic = "not " + generic
    return (f"make {generic} true" if context == "effect" else generic)

# -------------------------------
# Primitive action summarizer (optional)
# -------------------------------

def summarize_actions(actions: List[str]) -> Optional[str]:
    """
    Turn action primitives into a short hint.
    Looks for tokens like 'turn-left', 'turn-right', 'forward', 'pickup_0', 'drop_0'.
    """
    if not actions:
        return None

    verbs = []
    for line in actions:
        tok = line.strip().split()
        if not tok:
            continue
        op = tok[0]
        if op == "turn-left":
            verbs.append("turn left")
        elif op == "turn-right":
            verbs.append("turn right")
        elif op == "forward":
            verbs.append("move forward")
        elif op.startswith("pickup"):
            # try to grab object name if present
            obj = tok[2] if len(tok) >= 3 else ""
            verbs.append("pick up " + _clean_name(obj) if obj else "pick up")
        elif op.startswith("drop"):
            obj = tok[2] if len(tok) >= 3 else ""
            verbs.append("drop " + _clean_name(obj) if obj else "drop")
        else:
            # keep as is, lightly cleaned
            verbs.append(op.replace("_", " "))
    # Deduplicate mild repeats but keep order
    collapsed = []
    for v in verbs:
        if not collapsed or collapsed[-1] != v:
            collapsed.append(v)
    hint = ", ".join(collapsed)
    return hint if hint else None

# -------------------------------
# PDDL-ish goal clause verbalization (lightweight)
# -------------------------------

def _sexpr_tokens(s: str) -> List[str]:
    s = s.replace("\n", " ")
    # Add spaces around parentheses
    s = s.replace("(", " ( ").replace(")", " ) ")
    return [t for t in s.split() if t]

def _parse_sexpr(tokens: List[str], i: int = 0):
    """Return (expr, next_index)."""
    if i >= len(tokens):
        return None, i
    if tokens[i] != "(":
        return tokens[i], i+1
    lst = []
    i += 1
    while i < len(tokens) and tokens[i] != ")":
        elem, i = _parse_sexpr(tokens, i)
        lst.append(elem)
    return lst, i+1

def _render_goal(expr, binder_set= None) -> str:
    """
    Render a subset of PDDL-like S-exprs to English.
    Handles: forall, exists, and, onfloor, nextto.
    """
    if not isinstance(expr, list) or not expr:
        return str(expr)

    head = expr[0]
    # (forall (?x - type) body)
    if head == "forall" and len(expr) == 3:
        binder, body = expr[1], expr[2]
        if binder_set is None:
            binder_set = set()
        binder_set.add(str(binder))
        # binder looks like: ['?p', '-', 'plate', '?v', '-', 'vegetable_oil']
        front_end_phrase = ""
        previous_slash_index = -2
        for binder_part_id, binder_part_val in enumerate(binder):
            if binder_part_val == '-':
                # get the current binder_val 
                binder_val = binder[binder_part_id + 1]
                binder_keys = binder[previous_slash_index + 2: binder_part_id]
                assert len(binder_keys) == 1, 'Forall binder and the outtest should have only one variable'
                if previous_slash_index == -2:
                    front_end_phrase += f"for every {_clean_name(binder_val)}, "
                else:
                    front_end_phrase += f"and for every {_clean_name(binder_val)}, "
                previous_slash_index = binder_part_id
      
        return front_end_phrase + _render_goal(body, binder_set)

    # (exists (?y - type) body)
    if head == "exists" and len(expr) == 3:
        binder, body = expr[1], expr[2]
        if binder_set is None:
            binder_set = set()
        binder_set.add(str(binder))
        
        front_end_phrase = ""
        previous_slash_index = -2
        for binder_part_id, binder_part_val in enumerate(binder):
            if binder_part_val == '-':
                # get the current binder_val 
                binder_val = binder[binder_part_id + 1]
                binder_keys = binder[previous_slash_index + 2: binder_part_id]
                binder_key_len = len(binder_keys)
                if previous_slash_index == -2:
                    front_end_phrase += f"there exists {binder_key_len} {_clean_name(binder_val)}(s), "
                else:
                    front_end_phrase += f"and there exists {binder_key_len} {_clean_name(binder_val)}(s), "
                previous_slash_index = binder_part_id
        return front_end_phrase + "such that " + _render_goal(body, binder_set)
        

    # (and a b c ...)
    if head == "and":
        parts = [_render_goal(x, binder_set) for x in expr[1:]]
        parts = [p for p in parts if p]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    
    if head == 'not':
        parts = [_render_goal(x, binder_set) for x in expr[1:]]
        parts = [p for p in parts if p]
        if not parts:
            return ""
        if len(parts) == 1:
            return "it is not the case that " + parts[0]
        return "it is not the case that " + ", ".join(parts[:-1]) + " and " + parts[-1]
    
    # imply 
    # example ['imply', ['cookable', '?x'], ['exists', ['?c', '-', 'cabinet', '?loc', '-', 'location', '?dim', '-', 'dimension'], ['inside', '?x', '?c', '?loc', '?dim']]]
    if head == 'imply' and len(expr) == 3:
        premise = _render_goal(expr[1], binder_set)
        conclusion = _render_goal(expr[2], binder_set)
        return f"if {premise}, then {conclusion}"
    
    # or 
    # example ['or', ['nextto', '?r', '?s'], ['exists', ['?loc', '-', 'location', '?dim', '-', 'dimension'], ['inside', '?r', '?s', '?loc', '?dim']]]
    if head == 'or' and len(expr) == 3:
        part1 = _render_goal(expr[1], binder_set)
        part2 = _render_goal(expr[2], binder_set)
        return f"either {part1}, or {part2}"

    # Predicates
    if head == "onfloor" and len(expr) == 2:
        arg = str(expr[1])
        # arg change to binder value if exists
        output = f"{arg} is on the floor"
    elif head == "nextto" and len(expr) == 3:
        a = str(expr[1])
        b = str(expr[2])
        a_clean = a.split(" ")[0]
        b_clean = b.split(" ")[0]
        if a_clean == b_clean:
            if b_clean == b:
                output = f"{a} is next to another {b}"
            else:
                output = f"{a} is next to {b}"
        else:
            if b_clean == b:
                output = f"{a} is next to the {b}"
            else:
                output = f"{a} is next to {b}"
            
    elif head == 'is-sliced' and len(expr) == 2:
        arg = str(expr[1])
        output = f"{arg} is sliced"
    elif head == 'onTop' and len(expr) == 3:
        a = str(expr[1])
        b = str(expr[2])
        a_clean = a.split(" ")[0]
        b_clean = b.split(" ")[0]
        if a_clean == b_clean:
            if b_clean == b:
                output = f"{a} is on top of another {b}"
            else:
                output = f"{a} is on top of {b}"
        else:
            if b_clean == b:
                output = f"{a} is on top of the {b}"
            else:
                output = f"{a} is on top of {b}"
        
    elif head == 'atsamelocation' and len(expr) == 3:
        a = str(expr[1])
        b = str(expr[2])
        output = f"{a} is at the same location as {b}"
    elif head == 'is-soaked' and len(expr) == 2:
        arg = str(expr[1])
        output = f"{arg} is soaked"
    elif head == 'is-toggled' and len(expr) == 2:
        arg = str(expr[1])
        output = f"{arg} is toggled on"
    elif head == 'is-opened' and len(expr) == 2:
        arg = str(expr[1])
        output = f"{arg} is opened"
    elif head == 'is-not-dusted' and len(expr) == 2:
        arg = str(expr[1])
        output = f"{arg} is not dusty"
    elif head == 'is-not-stained' and len(expr) == 2:
        arg = str(expr[1])
        output = f"{arg} is not stained"
    elif head == 'inreachofrobot' and len(expr) == 3:
        a = str(expr[1])
        b = str(expr[2])
        output = f"{a} is in reach of {b}"
    elif head == 'inside' and len(expr) == 5:
        a = str(expr[1])
        b = str(expr[2])
        output = f"{a} is inside {b}"
    elif head == 'is-freezed' and len(expr) == 2:
        arg = str(expr[1])
        output = f"{arg} is frozen"
    elif head == 'under' and len(expr) == 3:
        a = str(expr[1])
        b = str(expr[2])
        output = f"{a} is under {b}"
    elif head == "=" and len(expr) == 3:
        a = str(expr[1])
        b = str(expr[2])
        output = f"{a} is equal to {b}"
    elif head == 'cookable' and len(expr) == 2:
        arg = str(expr[1])
        output = f"{arg} is cookable"
    else:
        print(f"Warning: unhandled goal predicate: {expr}")
        raise NotImplementedError(f"Unhandled goal predicate: {expr}")
    if binder_set is not None:
        for binder in binder_set:
            binder = eval(binder) if isinstance(binder, str) else binder
            previous_slash_index = -2
            for binder_part_id, binder_part_val in enumerate(binder):
                if binder_part_val == '-':
                    # get the current binder_val 
                    binder_val = binder[binder_part_id + 1]
                    # binder_val change _ to space
                    binder_val = binder_val.replace("_", " ")
                    binder_keys = binder[previous_slash_index + 2: binder_part_id]
                    for binder_key_idx, binder_key in enumerate(binder_keys):
                        regex_pattern = re.escape(binder_key) + r'\b'
                        if len(binder_keys) == 1:
                            output = re.sub(regex_pattern, binder_val, output)
                        else:
                            # add idx to the binder_val 
                            output = re.sub(regex_pattern, binder_val + " " + str(binder_key_idx), output)

                    previous_slash_index = binder_part_id
    return output

def verbalize_goal_clause(goal_clause: str) -> str:
    tokens = _sexpr_tokens(goal_clause)
    expr, _ = _parse_sexpr(tokens, 0)
    text = _render_goal(expr)
    # Minor stylistic touch-ups
    text = text.replace(" ,", ",").strip()
    text = text[0].upper() + text[1:] if text and text[0].islower() else text
    return text

def verbalize_goal_count_feature(goal_count_feature) -> str:
    """
    Verbalize goal count feature(s). Can handle both single Count and list of Count objects.
    
    Ex: 'Count(plywood, =0, count_isolated)' -> 'isolated plywood is 0'
    Ex: 'Count(carton, =0, count_not_inside_target_type(target_type_name=shelf))' -> 'carton not inside shelf is 0'
    Ex: [Count(...), Count(...)] -> 'X is 0, Y is 0, and Z is 0'
    """
    goal_count_str = str(goal_count_feature)
    
    # Check if it's a list by looking for opening bracket
    if goal_count_str.strip().startswith('['):
        # Parse the list of Count objects
        # Remove the outer brackets
        inner = goal_count_str.strip()[1:-1]
        
        # Split by Count( to find individual Count objects
        count_tokens = []
        depth = 0
        current = []
        i = 0
        while i < len(inner):
            if inner[i:i+6] == 'Count(' and depth == 0:
                if current:
                    # Skip the current buffer if it's just whitespace/comma
                    pass
                current = ['Count(']
                i += 6
                depth = 1
            elif depth > 0:
                current.append(inner[i])
                if inner[i] == '(':
                    depth += 1
                elif inner[i] == ')':
                    depth -= 1
                    if depth == 0:
                        # Found complete Count object
                        count_str = ''.join(current)
                        count_tokens.append(count_str)
                        current = []
                i += 1
            else:
                i += 1
        
        # Verbalize each Count and join with commas and 'and'
        verbalized = []
        for count_str in count_tokens:
            fn, args = _fn_and_args(count_str)
            if fn == "Count" and len(args) == 3:
                entity, comp, count_func_name = args
                verbalized.append(_verbalize_count_feature(entity, comp, count_func_name, context="precondition"))
        
        if not verbalized:
            return _clean_name(goal_count_str)
        elif len(verbalized) == 1:
            return verbalized[0]
        else:
            # Join with commas and 'and' for the last item
            return ", ".join(verbalized[:-1]) + ", and " + verbalized[-1]
    else:
        # Single Count object
        fn, args = _fn_and_args(goal_count_str)
        if fn == "Count" and len(args) == 3:
            entity, comp, count_func_name = args
            return _verbalize_count_feature(entity, comp, count_func_name, context="precondition")
        return _clean_name(goal_count_str)

# -------------------------------
# End-to-end generator
# -------------------------------

def verbalize_rule(rule_str: str) -> Tuple[str, List[str]]:
    pre, eff = parse_rule(rule_str)
    if not pre and not eff:
        return "(unparsed rule)", []
    pre_en = [verbalize_predicate(p, context="precondition") for p in pre]
    eff_en = [verbalize_predicate(e, context="effect") for e in eff]

    # Build a compact sentence
    if pre_en:
        pre_txt = ", ".join(pre_en[:-1]) + (" and " if len(pre_en) > 1 else "") + pre_en[-1] if len(pre_en) >= 2 else pre_en[0]
        prefix = f"When {pre_txt}, "
    else:
        prefix = ""
    if eff_en:
        eff_txt = "please " + ", then ".join(eff_en)
        sentence = prefix + eff_txt + "."
    else:
        sentence = prefix.rstrip(", ") + "."

    # We also return the list (pre_en + eff_en) in case the caller wants finer control
    return sentence, pre_en + eff_en


def split_goal_str(goal_str: str) -> List[str]:
    # split by ()
    parts = [] 
    cur_content = ""
    cur_bracket_count = 0 # ( + 1 and ) -1
    
    for char in goal_str:
        if char == '(':
            cur_bracket_count += 1
            cur_content += char
        elif char == ')':
            cur_bracket_count -= 1
            cur_content += char
        else:
            cur_content += char
        if cur_bracket_count == 0:
            if cur_content.strip():
                parts.append(cur_content.strip())
            cur_content = ""
        
    # return the parts 
    return parts

#! MAIN FUNCTION
def generate_pseudo_nl(
    demo_info: Dict[str, Any],
    goal_clause: Any,
    goal_count_feature: Any,
    include_actions: bool = False,
    compact: bool = False,
    compose: bool = True,
) -> str:
    """
    Deterministically generate a concise pseudo-NL description from
    (goal clause, goal-count feature, and performed rules).
    """
    # Goal strings
    # clean the goal clause str, split by \n and remove lines that start with ;
    goal_clause = str(goal_clause).split("\n")
    goal_clause = [line for line in goal_clause if not line.strip().startswith(";")]
    goal_clause = "\n".join(goal_clause)
    goal_clause_parts = split_goal_str(str(goal_clause))
    goal_clause_text_lst = [] 
    # part = goal_clause_parts[7]
    # goal_clause_text_part = verbalize_goal_clause(part)
    for part in goal_clause_parts:
        goal_clause_text_part = verbalize_goal_clause(part)
        goal_clause_text_lst.append(goal_clause_text_part)
    goal_clause_text = ""
    for idx, part in enumerate(goal_clause_text_lst):
        if idx == 0:
            goal_clause_text += part 
        else:
            goal_clause_text += ". And " + part[0].lower() + part[1:]
    
    goal_count_text = verbalize_goal_count_feature(str(goal_count_feature))

    lines = []
    # High-level goal
    lines.append(f"Goal: {goal_clause_text}.")
    # lines.append(f"Stop condition: {goal_count_text}.") # no need to add this

    # Steps from performed rules
    rules = demo_info.get("stored_info", {}).get("performed_rules", [])
    for idx, subgoal, rule_str, action_seq in rules:
        sent, _ = verbalize_rule(rule_str)
        header = f"At step {idx}: {sent}"
        # header = f"At step {idx}, for subgoal {subgoal}: {sent}"
        # if "reduce book and hardback not on shelf" in header:
        #     print("debug")
        #     breakpoint()
        if include_actions:
            hint = summarize_actions(action_seq)
            if hint:
                header += f" (actions: {hint})"
        lines.append(header)

    # Optional compact final guidance
    normal_output = "\n".join(lines)
    condensed_result =  condense_planning_script(normal_output)
    compact_output = condensed_result
        
    if compose: 
        # return {list, normal_output, compact_output}
        return {
            "instr_list": lines[1:],
            "normal_instr": normal_output,
            "compact_instr": compact_output,
            "rule_list": [normalize_rule_str(rule_str) for _, _, rule_str, _ in rules],
        }
        
    else:
        if compact:
            return compact_output
        else:
            return normal_output
            
    # TODO 
    # compact_output seem problematic for width = 3 

def normalize_rule_str(rule_str: str) -> str:
    """
    Re-formats a rule string to be more friendly for LLM tokenizers.

    This function applies the following transformations:
    1. Adds spaces around all parentheses, brackets, and braces.
    2. Replaces all underscores with spaces (e.g., "plywood_0" -> "plywood 0").
    3. Splits CamelCase words (e.g., "DistanceToNearest" -> "Distance To Nearest").
    4. Converts the entire string to lowercase.
    5. Collapses multiple spaces into a single space.

    Args:
        rule_str: The original rule string.

    Returns:
        The reformatted string.
    """
    
    # 1. Add spaces around all special grouping characters
    # This will turn "Rule(A)" into " Rule ( A ) "
    # We include {} as well, just in case.
    s = re.sub(r'([\[\]\(\)\{\}])', r' \1 ', rule_str)
    
    # 2. Replace underscores with spaces
    s = s.replace('_', ' ')
    
    # 3. Split CamelCase (e.g., "DistanceToNearest" -> "Distance To Nearest")
    # This regex finds a lowercase letter or digit followed by an uppercase letter
    # and inserts a space between them.
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s)
    
    # 4. Convert the entire string to lowercase
    s = s.lower()
    
    # 5. Collapse multiple spaces (created in step 1) into a single space
    # and strip leading/trailing whitespace.
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s


def run_demo():
    domain_name_list = [
        "laying_wood_floors",
        "preparing_salad",
        "cleaning_up_the_kitchen_only",
        "organizing_file_cabinet",
        "thawing_frozen_food",
        "making_tea",
        "opening_packages",
        "boxing_books_up_for_storage",
        "collect_misplaced_items",
        "putting_away_dishes_after_cleaning", # need to deal with compact mode for this one
        "washing_pots_and_pans",
        "cleaning_shoes",
        "installing_a_printer",
        "setting_up_candles",
        "watering_houseplants",
        "cleaning_a_car",
        "storing_food",
        "throwing_away_leftovers",
        "moving_boxes_to_storage",
        "sorting_books",
    ]
    map_id = "*"
    for domain_name in tqdm(domain_name_list, desc="Domains"):
        print(f"--- Domain: {domain_name} ---")
        # Locate a demo file

        for width in tqdm([0,1,2,3,4,5,6,7], leave=False, desc="Widths"):
            pattern_str = f'/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/02_intermediate/gps_data/{domain_name}/gps_data_map_{map_id}_width_{width}_env_MiniGrid-*_goal_True.pkl'
            from glob import glob
            found_files = glob(pattern_str)
            if not found_files:
                if width == 3:
                    continue
                else:
                    raise FileNotFoundError(f"No files found matching pattern: {pattern_str}")
            file_path = found_files[0]  # Just take the first match for demonstration
            with open(file_path, 'rb') as f:
                demo_info_dict = pickle.load(f)
                
            pddl_goal_clause = GOAL_CLAUSES[domain_name]
            
            output = generate_pseudo_nl(
                demo_info=demo_info_dict,
                goal_clause=pddl_goal_clause,
                goal_count_feature=demo_info_dict["goal_count_feature"],
                include_actions=False,       # show a short hint from primitive actions
                compact=True,                # slightly tighter phrasing,
                compose=True
            )
            print(f"Goal clause: {pddl_goal_clause}")
            print(f"Goal count feature: {demo_info_dict['goal_count_feature']}")
            print(output['normal_instr'])
            print("----- Compact version -----")
            print(output['compact_instr'])
            print(f"Width: {width}")
            input("Press Enter to continue to next domain...")

def update_instr_helper(arg):
    file_path, domain_name = arg
    with open(file_path, 'rb') as f:
        demo_info_dict = pickle.load(f)
        
    pddl_goal_clause = GOAL_CLAUSES[domain_name]
    
    output = generate_pseudo_nl(
        demo_info=demo_info_dict,
        goal_clause=pddl_goal_clause,
        goal_count_feature=demo_info_dict["goal_count_feature"],
        include_actions=False,       # show a short hint from primitive actions
        compact=True,                # slightly tighter phrasing,
        compose=True
    )
    # Update the pickle file with new instructions
    demo_info_dict['nl_instructions'] = output
    with open(file_path, 'wb') as f:
        pickle.dump(demo_info_dict, f)

def update_instr_into_pickle_files():
    domain_name_list = [
        "laying_wood_floors",
        "preparing_salad",
        "cleaning_up_the_kitchen_only",
        "organizing_file_cabinet",
        "thawing_frozen_food",
        "making_tea",
        "opening_packages",
        "boxing_books_up_for_storage",
        "collect_misplaced_items",
        "putting_away_dishes_after_cleaning", # need to deal with compact mode for this one
        "washing_pots_and_pans",
        "cleaning_shoes",
        "installing_a_printer",
        "setting_up_candles",
        "watering_houseplants",
        "cleaning_a_car",
        "storing_food",
        "throwing_away_leftovers",
        "moving_boxes_to_storage",
        "sorting_books",
    ]
    map_id = "*"
    args = [] 
    for domain_name in tqdm(domain_name_list, desc="Domains"):
        print(f"--- Domain: {domain_name} ---")
        # Locate a demo file

        for width in tqdm([0,1,2,3,4,5,6,7], leave=False, desc="Widths"):
            pattern_str = f'/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/02_intermediate/gps_data/{domain_name}/gps_data_map_{map_id}_width_{width}_env_MiniGrid-*_goal_True.pkl'
            found_files = glob(pattern_str)
            # print how many files found
            print(f"\nFound {len(found_files)} files for width {width}")
            # sleep(5)
            for file_path in tqdm(found_files, desc="Found files", leave=False):
                args.append((file_path, domain_name))
                
    max_processes = int(os.cpu_count() * 0.93)
    with Pool(processes=max_processes) as pool:
        a = list(tqdm(
            pool.imap_unordered(update_instr_helper, args, chunksize=1), total=len(args), desc="Updating pickle files"
        ))

    # update_instr_helper(args[0])

if __name__ == "__main__":
    update_instr_into_pickle_files()