Evolve an executable Python agent for the Namazu unknown-enemy-dynamics v3
experiment. Jointly change world_model_step(memory, local_obs, last_action),
planner(memory, local_obs), export_model, helpers, representations, memory and
predictive updating. Substantive new programs and algorithms are permitted;
this is not a fixed algorithm catalog or parameter-only search.

The world is 15x15 with a local 5x5 observation square without occlusion. Two
keys unlock a cardinally adjacent gating door and an exit. Three anonymous
enemies move independently. Interior dynamic walls change every 25 steps; fixed
borders and a 200-step horizon are retained. Actions: move=[dx,dy], each integer
in -1,0,1, plus interact=bool. Diagonal corner cutting is blocked. Interaction
opens adjacent doors before agent movement; keys are collected after movement.
Stepping onto an enemy kills before that enemy moves. A dynamic closure of an
occupied cell is skipped until the next scheduled tick.

The three equally weighted development regimes are: uniform nine attempted
movements including stay; one privately drawn stationary categorical movement
law per episode; and an unannounced within-episode change to an independent law.
Laws use Dirichlet(1,...,1), rejecting entropy<1.2nats or maximum probability>.6.
The changing regime switches at a private uniform integer step 25..75, with total
variation distance>=.3 between laws. All enemies share the current law. Blocked
attempted moves become stays; probabilities are NOT renormalized over legal
moves. Laws, switch times, enemy identities, hidden coordinates and seeds are
never revealed to ordinary candidates.

local_obs supplies grid(enemy overlay=5), terrain(0empty,1wall,2key,3locked door,
4exit,-1outside), step,health,keys,door_open, and feedback(actual displacement,
collected,opened). Coordinates in exported models use the initial agent position
as(0,0). Memory resets to None every episode. Terrain and localization memory are
state estimation; predictive adaptation means learning dynamics from experience.

The evaluator records export_model(memory,local_obs) after action choice and
before the next transition. Export enemy=[[relative_x,relative_y,p],...] and
default_enemy=p in[0,1], predicting NEXT-step anonymous occupancy at stationary
cells in initial-agent coordinates. Export terrain=[[x,y,cell,last_seen_step],...]
for reconstruction diagnostics and position=[x,y]. It scores every local 3x3 cell
and 12 random interior audit cells; missing probabilities default to.5. Invalid
numbers, actions or malformed exports fail evaluation. Prediction quality is
reported as proper Brier losses and actionable text feedback, but it is not the
fitness objective. Prediction and risk calculations used in decisions may use
any coherent representation; the export interface does not constrain planning.

Frozen objective absolute-task-v3: task=.65*escape+.10*keys+.10*door+
.05*escape*(1-steps/200), where keys is the raw number of collected keys (0, 1, or 2).
Maximize the mean absolute task component over all equally weighted regimes.
Do not optimize advantage over a disabled version or sabotage an intervention.
Forecast improvements only matter to selection when they improve actual control.
If your representation naturally permits predictive-update or prediction-use
interventions, preserve them separately from mapping/localization. Generic flags
are not required to force your algorithm into the seed's representation.

Execution is Python 3.10 standard library, 192 MiB address space, 10 CPU seconds per
episode, 3 wall seconds per decision, 512 KiB source, 2 MiB reply. OS isolation denies
repository/private files,network,new processes and evaluator inspection. Resource
limits regulate execution, not permissible algorithms. Use only this task and
the supplied parent/inspiration code,development text feedback and native meta
recommendations. Do not read other experiment outputs or assessment data. Return
a patch or full program in ShinkaEvolve's requested format.
