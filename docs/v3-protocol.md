# Unknown dynamics, wave 1

This separately versioned experiment (`namazu-unknown-dynamics-v3`) asks whether
an evolved online predictor helps decisions under unknown enemy motion. It does
not extend the closed v2 database or reinterpret v2's exposed assessment.
The current user instruction authorizes one initial search of **50 total slots**,
including its seed and failed proposals/evaluations. It supersedes the earlier
24-search proposal; independent discovery reliability remains unmeasured.

## Design before mutation

Retain the 15×15 maze, 5×5 local view, two keys, gating door, exit, three anonymous
enemies, nine attempted moves, dynamic walls every 25 actions and 200-action
horizon. The experimental extension changes enemy attempted-movement laws only.
Uniform is the original categorical law. Stationary laws are independently drawn
Dirichlet(1,…,1), rejecting entropy below 1.2 nats or maximum probability above
0.6. Switching cases draw another law at total variation distance at least 0.3;
the first transition under it is privately scheduled uniformly at 25–75.
These are unvalidated initial design choices, not discovered constants.
Blocked attempts stay; probabilities are never renormalized over legal moves.
Independent tagged streams separate layout, laws, switch timing, enemy
innovations, forecast targets and candidate randomness. All regimes share layout
and underlying enemy innovations for each case, without exposing those streams.

The search starts from the immutable original predictive source in
`controls/v1/predictive.py`, with no comparator architecture added. Original
memory and v2 generation 14 remain distinct historical controls. The new
direction-aware comparator is explicitly manual engineering, fitted only on
development trajectories and its own fitting subvalidation.

The frozen objective is the existing **absolute task component**:
`0.65*escape + 0.10*keys + 0.10*door + 0.05*escape*(1-steps/200)`.
Invalid executions score zero. Proper one-step occupancy losses remain separate
diagnostics and textual feedback. Before mutation, a focused development check
will quantify score variation, historical-composite ranking, sparse targets and
the decision response to frozen predictions. No weight sweep is authorized by
this design. Predictive adaptation must earn control benefit empirically.

The development search uses 24 case seeds × three equally weighted regimes
(72 episodes/candidate). Comparator fitting uses 64 separate cases × three;
its subvalidation uses another 24 × three. Champion selection uses 64 additional
cases × three. All splits are disjoint and identities are in `splits.json`.
After all 50 slots, assess up to five distinct valid sources with the highest
development task score (ties: lower slot first) on selection validation. Select
highest mean task, then escape count, then earlier slot. Do not use this split
for fitting or send its feedback to mutation. Report selection bias explicitly.

## Search mechanics and evidence

Use the existing pinned upstream Shinka revision, four native islands, native
parent/inspiration sampling, 0.6 diff / 0.4 full mutations and recommendations
every ten slots. One subscription-backed model retains existing high effort;
no model-bandit claim. Novelty embeddings/judge and prompt evolution are inactive.
Keep one controller, preserve pending proposals, restore recommendation history
on resume, and save every compact candidate source and actual lineage/event
record. All mutation, repair, probe and recommendation calls are counted;
evaluations and analyses make no model calls. API-list-price estimates are not
subscription charges. Installed source and resolved runtime settings, rather
than nominal configuration alone, determine the feature audit.

## Frozen-program assessment

Before generating a fresh private pool, freeze selected source, fitted comparator,
coherent interventions, analysis code and sample size. Size will be chosen from
development paired discordance and a meaningful 3-percentage-point escape effect,
with the computation and limitations published; it will not depend on assessment
significance. Baseline: 1,024 paired cases per regime, subject to that recorded
precision calculation. The primary family is selected online versus coherently
frozen learning, stationary and switch escape; apply Holm across those two tests.
The fitted-online versus fitted-frozen contrast is a separate prespecified
secondary family across stationary/switch. Uniform is continuity/descriptive.
Report exact paired binary tests and conservative paired intervals; forecast
differences/ratios use whole-episode bootstrap recomputing pooled sums and counts.
Pairing is preserved across conditions and regimes. Include invalid executions
in outcome denominators. On-policy losses do not establish prediction learning.

Matched recorded experience compares adaptive and frozen predictions under the
same observations/actions. Distinguish parameter adaptation from localization,
mapping and occupancy filtering. Validate interventions on state and action
traces. If a meaningful intervention is unavailable for the selected representation,
retain it and limit causal claims. Known-law reference sees current probabilities
only; partial observation remains and it is not an optimal-policy bound.
Count cases encountering a switch and informative visible transitions. All
post-switch comparisons explicitly condition on surviving to the analyzed window;
unconditional outcomes remain primary. A single selected program's fresh outcomes
cannot establish reliable discovery across searches.

## Reporting and preservation

Publish all candidates/failures, task and forecast components, actual ancestry,
paired outcomes, matched prediction evidence and resource measurements in
Chromatic Field figures and compact tables. Preserve all historical source,
controls, artifacts and reproduction commands byte-for-byte. Hidden pools,
traces, credentials and large databases stay outside Git. Reproduction and exact
resume commands accompany each checkpoint. Complete historical restyling is a
separate task.
