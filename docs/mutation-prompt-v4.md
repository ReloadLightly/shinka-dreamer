Evolve a complete executable Python world-model learner and planner for the
Namazu persistence/information maze experiment (wave v4 RUN 1). Jointly change
world_model_step(memory, local_obs, last_action), planner(memory, local_obs),
export_model, helpers, representations, inference, learning and planning.
Substantive new programs are permitted, not an algorithm catalog or weight-only
search. Start from and improve the supplied parent using native inspirations.

Physical task: 15x15 dynamic maze, 5x5 terrain observation, two keys, a gating door
and exit, three anonymous enemies, 200 transitions. Walls change every25steps.
Actions {'move':[dx,dy], 'interact':bool} with integer dx,dy in[-1,1], including
waiting. No diagonal corner cutting. Interact opens a cardinally adjacent door
when both keys have been collected, before movement. Keys are collected after
movement. Entering a currently occupied enemy cell kills BEFORE enemies move;
then all enemies attempt their movement, including on the terminal tick. Blocked
enemy attempts STAY; never renormalize their law over legal moves. Enemies share
one unknown categorical law over the nine attempted directions. They have no
observable identities. Occupied dynamic-wall closures are skipped until the
next scheduled tick. Terrain and anonymous occupancy must remain distinct.

Six equally weighted development conditions: uniform/full enemy observation,
stationary/full, one-switch/full, stationary/late, repeated-25/full and
repeated-25/late. Full means radius2; late means radius1 enemy visibility, while
terrain remains visible throughout the5x5window. The ordinary observation
explicitly provides enemy_visibility_radius. Outside it, grid equals terrain,
so lack of enemy overlay is NOT evidence of an empty cell. Stationary laws are
privately drawn from Dirichlet(1,...,1), rejecting entropy<1.2nats or max>.6.
One-switch first replacement transition is privately uniform25..75. Repeated
laws persist for privately drawn integer20..30transitions (mean25). Consecutive laws differ by total
variation>=.3. Ordinary candidates receive no laws, identities, seeds, actual switch
schedule, future innovations, private files or hidden world positions.

Observation fields: grid(enemy=5), terrain(0empty,1wall,2key,3locked door,4exit,
-1outside), enemy_visibility_radius, step, health, keys(integer0..2), door_open,
feedback(displacement,collected,opened), learn, predictive_planning. Memory
resets to None per episode. Mapping/localization/occupancy inference alone are
not evidence of predictive adaptation. Generic intervention flags need not
force a new representation into a fixed family; preserve coherent hooks if
natural, but your absolute task score governs selection.

Primary fitness is the mean absolute task score over a fixed48episode panel:
.65*escape+.10*number_of_keys+.10*door_open+.05*escape*(1-steps/200).
The two keys contribute at most.20; invalid execution scores0. No reward for
merely waiting. Never maximize advantage over your own disabled counterpart.
Prediction loss is diagnostic textual feedback, not part of fitness.

export_model(memory,local_obs) runs after planning, BEFORE the transition.
Export enemy=[[x,y,p],...] and default_enemy=p in[0,1] for NEXT-transition
anonymous occupancy at FIXED cells in initial-agent coordinates (initial
position=(0,0)). Export terrain=[[x,y,cell,last_seen_step],...] and optionally
position=[x,y]. Missing cell forecasts use default_enemy, default.5. Near Brier
scores cover the OLD agent-centered3x3; audit samples12interiorcells. Destination
Brier measures enemy occupancy AFTER enemy movement at the actual destination,
not total collision probability: entry-contact risk is separate. Forecast
coordinates/horizon/conditioning may differ from internal planning risks; be
precise about how prediction informs choices. Malformed actions/exports fail.

Development evidence motivating this wave (not assessment results): historical
v3 task performance improved rapidly, but many learned-vs-prior action changes
occurred where both actions had zero immediate estimated risk. A bounded new
selected-planner branch diagnostic held the learned state fixed, changed only
one first action, then used the same observation-only continuation for12steps.
For40 valid learned-vs-prior changed-action/future pairs, there was no collision
or escape difference;39returns were identical and one lost a key under the
learned recommendation. Only6disagreementstates had valid consequences and
5others were missing because a memory-check validator failed. This small,
selected, short-horizon negative result does not exclude useful adaptation.
Look for consequential ranking errors, uncertainty, stale evidence, anonymous
matching ambiguity and observations arriving too late to learn. A better generic
occupancy loss need not improve control. Track/update useful dynamics only where
its learning/inference cost and persistence make it relevant. All algorithms
remain open, including active information acquisition and uncertainty-aware
planning. No particular hand-designed adaptation architecture is required.

Runtime: Python3.10 standard library only, 192MiB address space,10CPU seconds per
episode,3wall seconds per reply,512KiB source,2MiB reply. Isolation denies network,
process creation, repository/private files and evaluator inspection. These
limits regulate execution rather than permissible algorithm families. Use the
provided task, parent/inspiration code, development feedback and native meta
recommendations. Return the patch/full code requested by the native operator.
