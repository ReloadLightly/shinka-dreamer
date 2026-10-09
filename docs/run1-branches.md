# RUN1 bounded first-action consequences: proposed execution contract

This is a separate development experiment. It preserves the v2/v3 sources,
assessments and completed diagnostic20 unchanged. Execution requires a recorded
RUN1 branch protocol; this document alone does not authorize transitions.

The question is whether a different first action recommended by the selected
generation-4 planner changes short-horizon survival or progress. It is narrower
than the value of online learning throughout an episode.

Use the first four existing fitting-subvalidation recordings in each v3 regime,
the same exposed development recordings used by diagnostic20. At most 80 frames
per recording are eligible. Reconstruct physical states inside the evaluator
from their registered private development seeds and recorded actions, checking
each current public observation against the existing recording. Count these
reconstruction transitions separately; their ceiling is 960. They are repeated
development execution, not fresh assessment cases.

An encounter is eligible from observation step 10 when an anonymous enemy is
visible and at least two distinct destinations are legal and currently empty
under the observed terrain and ordinary interaction rule. Within each regime,
retain the earliest four states where all three recommendations agree and the
earliest four where at least one differs. Traverse recording index then frame
index; accept at most two states per recording, separated by at least eight
recorded frames. Total ceiling: 24 states. Shortfalls remain shortfalls, with no
extra pool, longer recording, relaxed criterion or outcome-driven replacement.

At each eligible state, replay the public history through the immutable selected
source. Enable its normal predictive learning and planning, supplying preceding
recorded actions and actual observed displacement. This is memory-policy
physical experience with selected-planner internal target history. Clone its
memory immediately after the current model update. Compute recommendations from
the learned forecast, the same predictor with its movement law replaced by the
uniform initial prior, and the same predictor with the evaluator's true current
movement law. Recompute three-horizon forecasts before the replacement-law
planner calls. These replacements do not freeze learning history or replace
localization, map, belief, visits or predictive parameters. Only the true-law
copy is privileged. Discard both replacement-law memories after recommendation.

Every consequence branch starts from the same physical snapshot and the same
ordinary learned memory after its original planner call. Force only the selected
first action; subsequent calls use the original learned policy and each branch's
own observations and displacement. This deliberately holds the original target
memory fixed as well: any target write by the replacement-law recommendation is
discarded. Thus the treatment is the first action, not an entire alternative
model/planner state. Collapse duplicate actions and alias their outcomes.

For each state draw eight independent tagged enemy-innovation streams. Pair
each stream across distinct first actions. Each branch lasts at most 12 physical
transitions, stopping earlier at death, escape or the original episode horizon.
The original laws, hidden switch timing, dynamic-wall schedule and occupancy
rules remain unchanged. Changing agent occupancy may change which scheduled wall
closures occur; this is a real action consequence. The ceiling is
24 × 8 × 3 × 12 = 6,912 branch transitions. Hidden snapshots, enemy identities,
private seeds and future innovations remain evaluator-only.

One restricted worker per state/future replicate replays at most 80 public
frames, restores the common memory for each distinct action, and processes at
most 33 subsequent observations. The original 10 CPU second/192 MiB boundary and
three-second response deadline remain. Maximum 192 branch workers, 15,360 replay
warmup frames and 6,336 continuation calls; selection adds at most 960 public
frames. A global cap of 22,272 candidate operations additionally applies: one
public-observation update/planner call counts as one operation, as does each
extra replacement-law planner call. Selection operations are included. Reserve
each branch worker's maximum cost before launching it; report unstarted future
replicates when the combined cap would be exceeded. Source, worker, driver,
protocol and recording hashes are bound before
execution. Persist every attempt and its failure without automatic retry. A
deadline leaves an explicit incomplete unit, never a completed zero-effect case.

Primary descriptive outcome: death within 12 transitions. Also report escape,
survival without escape, horizon exhaustion, keys gained and door opening.
Estimate paired first-action effects by common future stream, then average
within state. Preserve dependence between states from the same recording in any
uncertainty calculation. The maximum of 12 precursor recordings gives limited
precision; present state-level effects and descriptive cluster intervals, not
an assertion of reliable discovery or whole-episode benefit. Agreement states
are explicit zero-treatment controls; identical actions share one execution.
Invalid executions remain in denominators and are shown separately from death.
The registered scalar return is minus one for collision, plus one for escape,
plus 0.25 per key gained and 0.25 for opening the door, minus 0.01 per executed
transition. Invalid branches receive the conservative worst-window return
−1.12; report invalid execution separately and also report collision-or-invalid
as a failure composite. No invalid execution is called an observed collision.

The separately specified new-wave factors retain ordinary v3 as the reference:
episode-persistent versus repeated private 25-tick laws, crossed with original
radius-two versus late radius-one enemy visibility. Terrain remains visible in
the full 5×5 window. This is an explicit information extension, not a repair.
That environment is owned by the main RUN1 protocol, not by this branch driver.
