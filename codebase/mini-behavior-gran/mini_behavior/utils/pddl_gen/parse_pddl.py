# from tarski.io import PDDLReader
from pddl.parser.domain import DomainParser
from pddl.parser.problem import ProblemParser

from pddl.logic import Predicate, constants, variables
from pddl.core import Domain, Problem, Action, Requirements
from pddl.formatter import domain_to_string, problem_to_string
from collections import defaultdict
import re


def remove_goal_clause(problem_str):
    lines = problem_str.splitlines()
    new_lines = []
    skip_lines = []
    skip = False
    parenthesis_count = 0
    for line in lines:
        if not skip:
            new_lines.append(line)
        if '(:goal' in line:
            skip = True
            # add (and 
            new_lines.append('    (and')
        if skip:
            skip_lines.append(line)
            parenthesis_count += line.count('(') - line.count(')')
            
        if skip and parenthesis_count == 0:
            skip = False
            # add the last line 
            new_lines.append('    )')  # close the (and
            new_lines.append(line)
        
    # pop first and last line from skip_lines
    if skip_lines:
        skip_lines = skip_lines[1:-1]
    return '\n'.join(new_lines), '\n'.join(skip_lines)
    
def form_problem_with_goal(problem_str_no_goal, goal_clause_str, add_optimize_cost=True):
    lines = problem_str_no_goal.splitlines()
    goal_clause_flag = False 
    and_clause_flag = False
    added_goal_clause = False
    new_lines = []
    for line_idx, line in enumerate(lines):
        if line_idx == len(lines) - 1:
            if add_optimize_cost:
                new_lines.append('    (:metric minimize (total-cost))')
        new_lines.append(line)
        if '(:goal' in line:
            goal_clause_flag = True
        if goal_clause_flag and '(and' in line:
            and_clause_flag = True
        if goal_clause_flag and and_clause_flag and not added_goal_clause:
            # add the goal clause str here
            goal_lines = goal_clause_str.splitlines()
            for goal_line in goal_lines:
                new_lines.append('    ' + goal_line)
            
            added_goal_clause = True
    return '\n'.join(new_lines)    
    
def add_goal_action_in_domain(domain_str, goal_str):
    
    # get variable from goal_str, variable start with ?
    # find all ?<var_name> in goal_str
    goal_vars = []
    goal_split = goal_str.split('\n')
    for line in goal_split:
        # find var and also its type
        vars_and_types = re.findall(r'\?(\w+)\s*-\s*(\w+)', line)
        
        if vars_and_types and 'forall' not in line and 'exists' not in line:
            goal_vars.extend(vars_and_types)
    # print(f'Goal variables: {goal_vars}')
    
    goal_vars = list(set(goal_vars))  # unique
    
    lines = domain_str.splitlines()
    # remove trailing empty lines
    while lines and lines[-1].strip() == '':
        lines.pop()
    num_lines = len(lines)
    new_lines = []
    for line_idx, line in enumerate(lines):

        new_lines.append(line)
        
        if line_idx == num_lines - 2:
            # mean we are at the end of the domain
            # add the new action before this line
            # add the new action after this line
            new_lines.append('(:action achieve-goal')
            # parameters
            parameter_str = ' '.join([f'?{var} - {type_}' for var, type_ in goal_vars])
            new_lines.append(f' :parameters ({parameter_str})')
            # precondition and effect
            # add the goal_str as precondition
            precondition_str = goal_str.replace('\n', '\n        ')
            new_lines.append(f' :precondition {precondition_str}')
            new_lines.append(' :effect (and )')  # empty effect
            new_lines.append(')')
    return '\n'.join(new_lines)



def obtain_domain_model(domain_file):
    domain_parser = DomainParser()
    
    with open(domain_file, 'r') as f:
        dtext = f.read()
    domain = domain_parser(dtext)
    return domain


# ! Main function
def obtain_domain_and_problem_instance(
    domain_file, problem_file, 
):
    domain_parser = DomainParser()
    
    with open(domain_file, 'r') as f:
        dtext = f.read()
    domain = domain_parser(dtext)
    
    with open(problem_file, 'r') as f:
        ptext = f.read()
        
    ptext_no_goal, ptext_goal = remove_goal_clause(ptext)
    
    
    # add goal action in domain
    updated_domain_str = add_goal_action_in_domain(dtext, ptext_goal)
    # print('Updated domain:')
    # print(updated_domain_str)
    # print('\n\n')
    # new domain parser
    new_domain_parser = DomainParser()
    new_domain = new_domain_parser(updated_domain_str)
    

    # print('ptext_no_goal:')
    # print(ptext_no_goal)
    # print('\n\n')
    problem_parser = ProblemParser()
    problem = problem_parser(ptext_no_goal)
    
    # Obtain goal action
    domain_actions = list(new_domain.actions)
    domain_action_dict = {a.name: a for a in domain_actions}
    
    achieve_goal_action = domain_action_dict['achieve-goal']
    goal_clause = achieve_goal_action.precondition
    
    # * Form new problem to replace the goal
    
    new_problem = Problem(
        name = problem.name,
        domain = domain,
        domain_name = domain.name,
        requirements = domain.requirements,
        objects = problem.objects,
        init = problem.init,
        goal = goal_clause,
    )
    
    # * Get the final domain and problem instance
    
    final_domain = domain
    task_instance = new_problem
    return final_domain, task_instance

# ! Main function 2
def replace_problem_goal(goal_clause_str, domain_file, problem_file):
    """
    Replace the goal of the problem with the new goal clause string.
    """
    
    with open(problem_file, 'r') as f:
        ptext = f.read()
        
    ptext_no_goal, ptext_goal = remove_goal_clause(ptext)
    
    
    with open(domain_file, 'r') as f:
        dtext = f.read()

    problem_with_goal = form_problem_with_goal(ptext_no_goal, goal_clause_str)

    return dtext, problem_with_goal


def obtain_objs_from_problem_instance(problem_instance: Problem, domain:Domain, filter_obj_name_prefix=None):
    objs = list(problem_instance.objects)
    if filter_obj_name_prefix is not None:
        objs = [obj for obj in objs if not obj.name.startswith(filter_obj_name_prefix)]
        
    # add agent constant if exists in domain
    agent_constants = [const for const in domain.constants if const.name.startswith('agent')]
    objs.extend(agent_constants)
    return objs

def obtain_preidicates_categorized_by_objs(problem_instance: Problem, objs: list):
    """
    Categorize predicates by objects.
    
    Args:
        problem_instance: PDDL Problem instance
        objs: List of objects obtained from obtain_objs_from_problem_instance
        
    Returns:
        Dictionary where key is the object and value is the list of predicates 
        whose terms contain that object
    """
    # Initialize dictionary with empty lists for each object
    obj_to_predicates = {obj: [] for obj in objs}
    
    # Get object names for quick lookup
    obj_names = {obj.name for obj in objs}
    
    # Iterate through all predicates in the initial state
    for predicate in problem_instance.init:
        # Check if this is a Predicate instance (not EqualTo or other types)
        if isinstance(predicate, Predicate) and hasattr(predicate, 'terms'):
            # Check each term in the predicate
            for term in predicate.terms:
                # If the term's name matches an object in our list
                if term.name in obj_names:
                    # Find the corresponding object and add the predicate
                    for obj in objs:
                        if obj.name == term.name:
                            obj_to_predicates[obj].append(predicate)
                            break
    
    return obj_to_predicates


def format_predicate_dict_to_text(
    obj_to_predicates: dict, 
    num_of_variants: int = 1,
    object_shuffle_prob: float = 1.0,
    predicate_shuffle_prob: float = 1.0,
    seed: int = 42
) -> list:
    """
    Convert the object-to-predicates dictionary into readable text with optional variants.
    
    Args:
        obj_to_predicates: Dictionary mapping objects to their predicates
        num_of_variants: Number of variants to generate (default: 1)
        object_shuffle_prob: Probability of shuffling object order for each variant (0.0-1.0, default: 1.0)
        predicate_shuffle_prob: Probability of shuffling predicate order for each object (0.0-1.0, default: 1.0)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        List of formatted strings (one per variant). If num_of_variants=1, returns a single-item list.
    """
    import random
    def format_name(name: str) -> str:
        """Replace - and _ with spaces in names and convert camelCase to separated words."""
        # First, handle camelCase by inserting space before uppercase letters
        import re
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        # Then replace - and _ with spaces
        return name.replace('-', ' ').replace('_', ' ')
    
    def format_position(term_str: str) -> str:
        """Convert pos-x-y format to readable position format."""
        if term_str.startswith('pos '):
            parts = term_str.split()
            if len(parts) >= 3:
                # Extract x and y from "pos x y"
                return f"x = {parts[1]} , y = {parts[2]}"
            elif len(parts) >= 2:
                return f"position {parts[1]}"
        return term_str
    
    def format_predicate(predicate) -> str:
        """Format a single predicate without the Predicate() wrapper."""
        # Get predicate name
        pred_name = format_name(predicate.name)
        
        # Get and format terms
        if hasattr(predicate, 'terms') and len(predicate.terms) > 0:
            formatted_terms = []
            for term in predicate.terms:
                term_str = format_name(str(term))
                # Check if this looks like a position
                if term_str.startswith('pos '):
                    term_str = format_position(term_str)
                formatted_terms.append(term_str)
            return f"{pred_name} {' '.join(formatted_terms)}"
        else:
            return pred_name
    
    def condense_position_strings(pos_strs: list) -> str:
        pattern = re.compile(r"at\s+(.+?)\s+(\d+)\s+x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s+(.+)")
        groups = defaultdict(list)
        for pos_str in pos_strs:
            match = pattern.match(pos_str)
            if match:
                name, index, x, y, label = match.groups()
                # Key by (name, index, label)
                groups[(name, index, label)].append((int(x), int(y)))
                
        condensed_lines = []
        # Sort by name and index for clean output
        for name, index, label in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
            coords = groups[(name, index, label)]
            
            # Extract unique, sorted x and y values
            xs = sorted(list(set(c[0] for c in coords)))
            ys = sorted(list(set(c[1] for c in coords)))
            
            # Format as requested
            line_out = f"at {name} {index} x = {xs} , y = {ys} , [{label}]"
            condensed_lines.append(line_out)
            
        return "\n".join(condensed_lines)
        
    # Save the current random state
    random_state = random.getstate()
    
    try:
        # Set random seed for reproducibility within this function
        random.seed(seed)
    

        # Generate variants
        variants = []
        
        for variant_idx in range(num_of_variants):
            # Decide whether to shuffle objects in this variant
            shuffle_objects = random.random() < object_shuffle_prob
            
            # Get objects list (shuffle if needed)
            objects = list(obj_to_predicates.keys())
            if shuffle_objects:
                random.shuffle(objects)
            
            # Build the output text for this variant
            output_lines = []
            position_outputs = []
            for obj in objects:
                predicates = obj_to_predicates[obj]
                
                # Format object name
                obj_name = format_name(obj.name)
                
                # Format all predicates for this object
                pred_strs = []
                pos_strs = []
                for pred in predicates:
                    format_predicate_str = format_predicate(pred)
                    if format_predicate_str.startswith('at') and 'x = 'in format_predicate_str:
                        pos_strs.append(format_predicate_str)
                    else:
                        pred_strs.append(format_predicate_str)
                
                # Decide whether to shuffle predicates for this object
                shuffle_predicates = random.random() < predicate_shuffle_prob
                if shuffle_predicates:
                    random.shuffle(pred_strs)
                
                # Create the object section
                obj_section = f"{obj_name}:\n" + "\n".join(f"  - {pred}" for pred in pred_strs)
                output_lines.append(obj_section)
                pos_strs = condense_position_strings(pos_strs)
                position_outputs.append(pos_strs)
            
            # Join all sections with double newlines
            final_output = "\n\n".join(output_lines) + "\n\n" + "\n".join(position_outputs)
            variants.append(final_output)
        
        return variants
    finally:
        # Restore the original random state
        random.setstate(random_state)

if __name__ == '__main__':

    domain_file = '/home/xxxname/Project/granularity_instruction_nsai/granularity-instruction-nsai/data/00_envs/mini_behavior/mini_behavior/utils/pddl_gen/minibehavior.pddl'
    problem_file = '/home/xxxname/Project/tarski_test/pddl_problems/cleaning_a_car/p325990298-cleaning_a_car.pddl'
    
    domain, problem = obtain_domain_and_problem_instance(domain_file, problem_file)
    
    domain = obtain_domain_model(domain_file)
    
    

   

    
    