Evolve a Python agent for a partially observed dynamic maze. Jointly change
world_model_step(memory, local_obs, last_action) and planner(memory, local_obs).
All helpers, representations, predictive learning rules, search algorithms and
export_model are open to substantive program evolution inside the evolve block.
Do not reduce this task to tuning constants or selecting from a fixed catalog.

The world is 15x15, observation a 5x5 square without occlusion. Two keys unlock
a cardinally adjacent door which gates the exit. Three anonymous enemies move;
some interior walls change every 25 steps. Borders never change. Horizon 200.
Actions have move=[dx,dy], integer components -1,0,1, and interact=bool. Diagonal
corner cutting is blocked. Interaction opens adjacent doors before movement and
collects keys after movement. Stepping onto an enemy kills before it can move.
Closure of an occupied dynamic cell is skipped until the next scheduled tick.

local_obs: grid (enemy overlay=5), terrain (0 empty,1 wall,2 key,3 locked door,
4 exit,-1 outside), step, health, keys, door_open, feedback with actual
displacement, collected and opened. Coordinates in your exported model use the
initial agent position as (0,0), not the hidden board coordinates. Memory starts
None each episode. No environment object, hidden identity, RNG or seed is exposed.

The evaluator records export_model(memory,local_obs) AFTER choosing the action
and BEFORE the next transition. Export enemy as [[relative_x,relative_y,p],...]
and default_enemy as p in [0,1]. These predict next-step anonymous occupancy at
the specified stationary cells. Export terrain as [[x,y,cell,last_seen_step],...]
for reconstruction diagnostics; export position and learning diagnostics if useful.
The evaluator privately scores every local 3x3 cell and 12 uniformly sampled
interior audit cells. Missing probabilities default to 0.5; missing terrain is
incorrect. NaN, invalid actions and malformed exports fail evaluation.

Objective task06-forecast04-v1: .6*task+.4*(1-.5*(near_Brier+audit_Brier)).
task=.65*escape+.10*keys+.10*door+.05*escape*(1-steps/200). Maximize efficient
escape; expose component metrics and compare forecast errors to persistence.
If implementing predictive learning, respect local_obs['learn']=False by
freezing predictive parameter updates only, preserving mapping/localization.
Respect predictive_planning=False by retaining mapping/pathfinding and using
a fixed observation-based risk heuristic instead of learned predictions.

Execution: Python 3.10 standard library, 192 MiB address space, 10 CPU seconds
per episode, 3 wall seconds per decision, 512 KiB source, 2 MiB reply. OS denies
repository/private-file access, network, new processes and evaluator inspection.
These are resource boundaries, not restrictions on permissible algorithms.
Use only the parent/inspiration code and development feedback provided in this
prompt. Do not use tools to read assessment files or other experiment outputs.
Return a patch or full evolved program in the format requested by ShinkaEvolve.
