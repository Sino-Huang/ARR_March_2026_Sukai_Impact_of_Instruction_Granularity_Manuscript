## PDDL Domain Model Statistics

### Overview
The Mini-BEHAVIOR-Gran environment uses a **universal PDDL domain model** (`minibehavior_gran_domain_model.pddl`) that is shared across all 20 household tasks. This unified model provides comprehensive support for embodied AI planning in multi-room household environments with complex object interactions.

### File Statistics
- **Total Lines**: 1,070 lines
- **File Size**: ~50 KB
- **Format**: PDDL fully compatible with LAMA planner, partially compatible with BFWS planner

### PDDL Requirements
The domain leverages advanced PDDL features:
- `:strips` - Basic action schemas
- `:typing` - Hierarchical type system
- `:conditional-effects` - Context-dependent action outcomes
- `:derived-predicates` - Computed spatial relations
- `:negative-preconditions` - Explicit negative conditions
- `:equality` - Object identity checks
- `:disjunctive-preconditions` - OR conditions
- `:existential-preconditions` - Existential quantification
- `:universal-preconditions` - Universal quantification (forall)
- `:action-costs` - Plan quality metric

### Type Hierarchy
- **7 top-level types**: `location`, `geo-info`, `item-and-furniture`, `direction`, `dimension`, `normal-item`, `furniture`
- **72 object types**: Including 67 normal items (apple, book, knife, etc.) and 18 furniture types (table, cabinet, stove, etc.)
- **4 specialized subtypes**: 
  - `cutting-item` (2 types: carving_knife, knife)
  - `cleaning-item` (4 types: broom, rag, scrub_brush, towel)

### Constants
- **4 directions**: north, east, south, west
- **3 dimensions**: bottom, middle, top (for vertical stacking)
- **1 agent**: agent-01

### Predicates (Total: 41)

#### State Predicates (10)
Dynamic predicates tracking object states:
1. `is-cooked` - Food cooking state
2. `is-sliced` - Object slicing state
3. `is-not-dusted` - Dust accumulation state
4. `is-not-stained` - Stain presence state
5. `is-toggled` - Toggle state (appliances)
6. `is-opened` - Open/closed state
7. `is-soaked` - Water absorption state
8. `can-overlap` - Traversability state
9. `can-seebehind` - Visibility state
10. `inhandofrobot` - Grasping state

#### Derived Predicates (10)
Computed spatial and relational predicates:
1. `atsamelocation` - Co-location relation
2. `infovofrobot` - Field-of-view relation
3. `inreachofrobot` - Reachability relation (critical for manipulation)
4. `insameroomasrobot` - Room co-presence
5. `inside` - Containment relation
6. `nextto` - Adjacency relation
7. `onfloor` - Floor placement (derived from not in hand)
8. `onTop` - Vertical stacking relation
9. `under` - Reverse stacking relation
10. `is-freezed` - Freezing state (derived from refrigerator containment)

#### Property Predicates (13)
Static capability predicates initialized per object:
1. `cookable` - Can be cooked
2. `freezable` - Can be frozen
3. `sliceable` - Can be sliced
4. `dustyable` - Can accumulate dust
5. `stainable` - Can be stained
6. `toggleable` - Can be toggled on/off
7. `openable` - Can be opened/closed
8. `soakable` - Can absorb water
9. `overlapable` - Can be walked through
10. `containable` - Can contain objects (with dimension parameter)
11. `seebehindable` - Can be seen behind when opened
12. `pickable` - Can be picked up/dropped
13. `slicer`, `cleaningTool`, `coldSource`, `heatSource`, `waterSource` - Tool/appliance roles

#### Spatial Predicates (4)
Environment topology and agent state:
1. `move-dir` - Grid connectivity (location × location × direction)
2. `succ` - Successor relation (for directions and dimensions)
3. `facing` - Agent orientation
4. `at` - Object/agent position (location × dimension)

### Primitive Actions (Total: 15)

#### Navigation Actions (3)
Low-level movement primitives:
1. **`turn-left`** - 90° counterclockwise rotation
2. **`turn-right`** - 90° clockwise rotation
3. **`forward`** - Move one grid cell forward (respects obstacles)

#### Object Manipulation Actions (12)

**Container/Furniture Interaction (2)**

4. **`open`** - Open containers/doors (updates `can-overlap`, `can-seebehind`)
5. **`close`** - Close containers/doors

**Pickup Actions (5)** - Layer-aware grasping with auto-stacking collapse

6. **`pickup_0`** - Pick from bottom layer (not in container)
7. **`pickup_1`** - Pick from middle layer (not in container)
8. **`pickup_2`** - Pick from top layer
9. **`pickup_0_container`** - Pick from bottom layer inside container
10. **`pickup_1_container`** - Pick from middle layer inside container

**Drop Actions (4)** - Layer-aware placement with auto-state updates

11. **`drop_0`** - Drop to bottom layer (target location)
12. **`drop_1`** - Drop to middle layer (requires bottom object)
13. **`drop_2`** - Drop to top layer (requires middle object)
14. **`drop_in`** - Drop inside open container (any dimension)

**State-Change Actions (2)**
15. **`slice`** - Slice food with cutting tool
16. **`toggle`** - Toggle appliances on/off

### Action Complexity Metrics

#### Lines of Code per Action
- **Navigation actions**: 15-25 lines each
- **Simple actions** (open, close, toggle, slice): 10-20 lines each
- **Pickup actions**: 40-80 lines each (due to stacking logic)
- **Drop actions**: 60-90 lines each (due to auto-cleaning/soaking effects)
- **Most complex action**: `drop_in` (~90 lines) - handles containment, stacking, cleaning, and soaking

#### Precondition Complexity
- **Average preconditions per action**: 5-8 conditions
- **Most complex preconditions**: `drop_in` (11 conditions including nested existential quantifiers)
- **Derived predicates used**: `inreachofrobot` (14 actions), `containable` (9 actions)

#### Effect Complexity
- **Conditional effects**: All drop actions include 4 `forall` loops for auto-dusting/staining
- **Automatic state propagation**: Drop actions automatically trigger:
  - Dust removal when cleaning tool contacts dusty objects
  - Stain removal when soaked cleaning tool contacts stained objects
  - Water absorption when object enters water source
- **Stacking mechanics**: Pickup actions include automatic collapse of vertical stacks

### Design Highlights

1. **Universal Coverage**: Single domain model supports 20 diverse household tasks without task-specific modifications
2. **Three-Layer Vertical Stacking**: Explicit bottom/middle/top dimensions enable complex spatial reasoning (objects on tables, items in containers)
3. **Automatic Effect Propagation**: Drop actions intelligently handle secondary effects (cleaning, soaking) without separate actions
4. **Derived Spatial Relations**: 10 derived predicates reduce planning complexity by precomputing reachability and spatial relations
5. **Container Semantics**: Sophisticated containment logic with open/close state and dimension-specific containable properties
6. **Tool-Object Interactions**: Specialized predicates for tools (slicer, cleaningTool) and appliances (heatSource, waterSource, coldSource)

### Scalability Metrics
- **Objects per task**: 15-40 objects (furniture + items)
- **Locations per task**: 20-60 grid cells (multi-room layouts)
- **Average plan length**: 50-200 primitive actions
- **Grounding size**: ~10K-100K ground actions per task (depends on object count and layout)

This unified PDDL model demonstrates the feasibility of **task-agnostic planning representations** for household robotics while maintaining semantic richness for instruction grounding at multiple granularities.

