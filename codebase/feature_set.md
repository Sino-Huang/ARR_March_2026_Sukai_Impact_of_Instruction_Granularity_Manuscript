# Feature Set for Mini-Behavior-Gran Environment

## Overview

To bridge raw states and high-level instructions, we define a set of **features** $\Phi = \{\phi_1, \ldots, \phi_n\}$, where $\phi_i$ is a Boolean or integer-valued function over state space $S$. These features serve as domain-level abstractions that capture task-relevant aspects of the environment. Thus, each state $s$ is associated with a **feature valuation vector** $\phi(s) = \left(\phi_1(s), \ldots, \phi_n(s)\right)$.

## Running Example

We first illustrate *planning width* computation using a running example. Consider a task where two shoes are cleaned using a rag, and the rag is placed on the floor afterwards. The key **environment dynamics** are:

1. The rag must be soaked before it can clean shoes
2. The sink must be turned on to soak the rag

We represent states using a set of task-relevant feature variables ($\Phi$):
- **Boolean features**: $H_r$ (holding rag), $T_s$ (sink is turned on), $S_r$ (rag is soaked), $C_i$ (shoe $i$ is cleaned), $onfloor_r$ (rag is on the floor)
- **Spatial features**: $p_r, p_s, p_f$ (distance to the rag, sink, and nearest uncleaned shoe, respectively)

## Feature Categories

The mini-behavior-gran environment provides two main categories of features:

### 1. Boolean State Features

These features check whether an object satisfies a specific predicate or property. They require grounding (i.e., binding to specific object instances).

#### 1.1 Object Manipulation Features

- **`Holding(object_type, wanted_value)`**
  - Checks if the robot is holding an object of the specified type
  - `wanted_value`: `True` (holding) or `False` (not holding)
  - Example: `Holding('rag', True)` checks if the robot is holding a rag

#### 1.2 Object State Features

- **`isOpened(object_type, wanted_value)`**
  - Checks if an openable object is in the opened state
  - Applicable to: cabinets, drawers, doors, etc.
  - Example: `isOpened('cabinet', True)`

- **`isToggled(object_type, wanted_value)`**
  - Checks if a toggleable object is in the toggled-on state
  - Applicable to: sinks, stoves, light switches, etc.
  - Example: `isToggled('sink', True)`

- **`isSoaked(object_type, wanted_value)`**
  - Checks if an object is in the soaked state
  - Applicable to: rags, cloths, sponges, etc.
  - Example: `isSoaked('rag', True)`

- **`isCleaned(object_type, wanted_value)`**
  - Checks if an object is in the cleaned state (not stained)
  - Applicable to: shoes, dishes, pots, pans, etc.
  - Example: `isCleaned('shoe', True)`

- **`isDustFree(object_type, wanted_value)`**
  - Checks if an object is dust-free (not dusted)
  - Applicable to: furniture, surfaces, etc.
  - Example: `isDustFree('table', True)`

### 2. Spatial and Quantitative Features

These features capture distances, counts, and spatial relationships.

#### 2.1 Distance Feature

- **`DistanceToNearest(object_type, wanted_value, value_in_precon=None, obj_name=None, consider_carried_objects=True)`**
  - Measures the Manhattan distance from the agent to the nearest object of the specified type
  - `wanted_value` options:
    - `'=0'`: Distance is 0 or 1 (adjacent)
    - `'>0'`: Distance is greater than 1 (not adjacent)
    - `'increase'`: Distance has increased from precondition value
    - `'decrease'`: Distance has decreased from precondition value
    - `'change'`: Distance has changed from precondition value
  - `value_in_precon`: The distance value in the precondition state (for comparison)
  - `obj_name`: Optional specific object name to measure distance to
  - `consider_carried_objects`: Whether to consider objects being carried (distance = -1)
  - Example: `DistanceToNearest('sink', '=0')` checks if the agent is next to a sink

#### 2.2 Count Features

- **`Count(object_type, value_in_precon=None, wanted_value=None, count_func=None)`**
  - Counts objects of a specific type that satisfy certain conditions
  - `wanted_value` options: `'>0'`, `'=0'`, `'increase'`, `'decrease'`, `'change'`
  - `count_func`: Custom counting function that defines what to count
  - `value_in_precon`: The count value in the precondition state (for comparison)
  
**Available Count Functions:**

1. **`count_isolated(problem_model, object_type)`**
   - Counts objects that have no neighbors of the same type in 4-connected grid
   - Returns: count, {isolated_objs, not_isolated_objs}

2. **`count_not_cleaned(problem_model, object_type)`**
   - Counts objects that are not cleaned (stained)
   - Returns: count, {not_cleaned_objs, cleaned_objs}

3. **`count_not_wiped(problem_model, object_type)`**
   - Counts objects that are not wiped
   - Returns: count, {not_cleaned_objs, cleaned_objs}

4. **`count_not_sliced(problem_model, object_type)`**
   - Counts sliceable objects that have not been sliced
   - Returns: count, {not_sliced_objs, sliced_objs}

5. **`count_incomplete_salad_place(problem_model, object_type)`**
   - Counts incomplete salad preparations (plates without 2 cookable items)
   - Returns: count, {incomplete_salad_place_objs, complete_salad_place_objs, ...}

6. **`count_closed(problem_model, object_type)`**
   - Counts objects that are closed
   - Returns: count, {closed_objs, not_closed_objs}

7. **`count_not_inside_target_object(problem_model, object_type, container_obj_name, additional_predicate_on_object_type=None)`**
   - Counts objects not inside a specific container object
   - Returns: count, {not_inside_objs, inside_objs, container_containable_dimensions, container_position}

8. **`count_not_ontop(problem_model, object_type, surface_obj_name, additional_predicate_on_object_type=None)`**
   - Counts objects not on top of a specific surface object
   - Returns: count, {not_ontop_objs, ontop_objs, surface_position, surface_dimension}

9. **`count_not_ontop_sometype(problem_model, object_type, surface_type, additional_predicate_on_object_type=None)`**
   - Counts objects not on top of any surface of a specific type
   - Returns: count, {not_ontop_objs, ontop_objs, surface_position, surface_dimension}

10. **`count_not_near_target_type(problem_model, object_type, target_type_name, include_inside=False)`**
    - Counts objects not near (within 1 step) of a target type
    - `include_inside`: Whether to count objects inside the target as "near"
    - Returns: count, {not_near_target_objs, near_target_objs, target_positions, ...}

11. **`count_not_soaked(problem_model, object_type)`**
    - Counts objects that are not soaked
    - Returns: count, {not_soaked_objs, soaked_objs}

12. **`count_not_inside_target_type(problem_model, object_type, target_type_name, additional_predicate_on_object_type=None)`**
    - Counts objects not inside any container of a specific type
    - Returns: count, {not_inside_objs, inside_objs, target_positions, target_containable_dimensions}

13. **`count_not_toggled(problem_model, object_type)`**
    - Counts objects that are not toggled on
    - Returns: count, {not_toggled_objs, toggled_objs}

14. **`count_not_samelocation_sometype(problem_model, object_type, target_type_name, additional_predicate_on_object_type=None)`**
    - Counts objects not at the same location as any object of target type
    - Returns: count, {not_samelocation_objs, samelocation_objs}

15. **`count_not_onfloor(problem_model, object_type)`**
    - Counts objects not on the floor (bottom dimension)
    - Returns: count, {not_onfloor_objs, onfloor_objs}

16. **`count_not_complete_first_layer_salad(problem_model, type_tag='plate')`**
    - Counts plates without first layer salad (lettuce or radish)
    - Returns: count, {not_complete_plates, completed_plates, plate_locations, ...}

17. **`count_not_complete_tables(problem_model, type_tag='table')`**
    - Counts tables that don't have 3 candles on them
    - Returns: count, {not_complete_tables, completed_tables, table_locations, ...}

## Feature Properties

### Grounding Requirement

- **Grounded Features**: Require binding to specific object instances (e.g., `Holding`, `isOpened`, `DistanceToNearest`)
  - Use `ground_var_value_dict` to store object bindings
  - Example: `{object_type: 'rag-01'}` binds the feature to a specific rag instance

- **Lifted Features**: Do not require grounding (e.g., some `Count` features)
  - Operate over all objects of a type

### Activation Pattern

Some features (e.g., `DistanceToNearest`, `Count`) support an **activation pattern** for precondition-effect relationships:

1. **`activate_in_precon()`**: Called in precondition to set the reference value
2. **`verify()`**: Called in effect to check if the feature holds relative to the precondition value

This enables features like:
- `Count(..., wanted_value='decrease')`: Count has decreased from precondition
- `DistanceToNearest(..., wanted_value='change')`: Distance has changed from precondition

## Helper Functions

The feature system includes helper functions for spatial reasoning:

- **`get_agent_location(problem_model)`**: Returns agent's (x, y) position
- **`get_closest_object_location(problem_model, object_type, agent_pos, ignore_objects, consider_carried_objects)`**: Finds closest object of specified type
- **`calculate_distance(problem_model, obj_name_1, obj_name_2)`**: Computes Manhattan distance between two objects

## Usage Example

```python
from mini_behavior.utils.policy_sketch.general_features import *

# Boolean feature: Check if holding a rag
holding_rag = Holding('rag', True)
holding_rag.ground_var_value_dict = {'rag': 'rag-01'}
is_holding = holding_rag.verify(domain_model, problem_model)

# Distance feature: Check if next to sink
near_sink = DistanceToNearest('sink', '=0')
near_sink.activate_in_precon(domain_model, problem_model, ignore_objects=[])
is_near = near_sink.verify(domain_model, problem_model)

# Count feature: Count uncleaned shoes
uncleaned_shoes = Count('shoe', wanted_value='>0', count_func=count_not_cleaned)
uncleaned_shoes.activate_in_precon(domain_model, problem_model)
has_uncleaned = uncleaned_shoes.verify(domain_model, problem_model)
```