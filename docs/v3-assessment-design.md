# V3 wave 1: frozen-program assessment design

This assessment asks whether the program selected after one native 50-slot
search benefits from its own predictive updates under unknown or changing enemy
movement. It separates task performance, prediction on identical experience,
control benefit from adaptation, and discovery across searches. Only the first
three can be addressed by this wave; there is one independent search.

The executable preregistration is
`artifacts/campaign-v3/assessment/preregistration.json`. Its source hashes,
conditions, analysis files, sample-size record and case-exclusion inputs are
committed before the fresh pool is drawn. The original search protocol and its
three equally weighted regimes remain unchanged. The assessment does not feed
back into this search, baseline fitting or champion selection.

## Selection and interventions

The five distinct sources with highest development fitness were evaluated on
64 separately reserved layouts in all three regimes: 960 condition-episodes.
Generation 4 won by the prespecified absolute task score, then escape count,
then earlier generation. It escaped in 180/192 validation episodes; this is a
selection-biased result. One execution of another finalist failed and remained
in the denominator with zero task fitness. All five sources and all outcomes
are retained. The selected original source is immutable.

Generation 4 combines a uniform movement expert with slow and fast directional
learners. Adaptive state comprises their raw concentrations, mixture log
weights and update counters. Localization, terrain memory, visible occupancy,
diffuse unseen occupancy, visits and controller targets are separate state.
The selected-source review and development audit specify and verify the exact
interventions; their artifacts are bound in the executable preregistration.

Ten on-policy conditions are assessed on the same cases:

| Condition | Source and intervention |
|:--|:--|
| Original memory | Immutable historical mapping/pathfinding agent |
| Original predictive seed | Immutable source that initialized this search |
| V2 generation 14 | Immutable source selected in the closed uniform-law study |
| Fitted predictor, frozen | Manually engineered direction-aware predictor; development-fitted initialization held fixed |
| Fitted predictor, online | Identical initialization and planner; online directional updates enabled |
| Known-law reference | Same comparator and planner; current true attempted-movement probabilities, with ordinary partial observation |
| Selected evolved | Generation 4, online learning and learned-law planning |
| Selected, frozen | Adaptive concentrations and mixture weights frozen at initialization; mapping and occupancy inference continue |
| Selected, uniform-law planning | Online learner continues; planning forecasts and collision hazards are recomputed with the uniform movement law |
| Selected, frozen and uniform-law planning | Both interventions together |

The known-law reference has privileged dynamics information, not an optimal
policy or full state information. The two selected uniform-law conditions still
predict and plan. They substitute the movement law used in those calculations;
they do not remove all predictive reasoning. No module-only search arms are
introduced. Additive diagnostic exports are source-bound and checked against
the immutable original for unchanged actions, task results and original model
exports on development experience. Their additional computation is disclosed.

## Sample size and case retirement

The meaningful escape difference is three percentage points. The development-only
precision calculation uses the selected program's 24 registered development
layouts in each regime, with online and frozen executions paired. It uses only
the number of discordant outcomes to choose the assessment size, not their
direction. Invalid executions count as non-escapes.

The planning discordance is the greater of 0.08 and the individual one-sided
95% binomial upper bounds for stationary and switching discordances. Starting
at 1,024 cases per regime, the calculation adds blocks of 256 until exact paired
test power reaches 80% for a three-point difference at two-sided alpha 0.025.
This plans for the most stringent step of the two-test primary Holm family.
The resulting integer and sensitivity calculations are in `precision.json`.
They are frozen before drawing cases and never changed in response to assessment
significance. Reused development cases and selection make this a conditional
planning calculation, not a guarantee of power or selection-adjusted coverage.

Fresh case identifiers are drawn privately after the complete freeze. The same
identifier supplies paired layouts and tagged randomness across all conditions
and regimes. Previously exposed assessment pools, development/selection/fitting
pools and historical published development identifiers are excluded. The
additional half-open interval [0, 10,000,000) conservatively retires small
synthetic identifiers used in checks; it is not a count of executed episodes.
Exclusion files are hash-bound. Every case in this assessment is retired from
future tuning, including cases not shown in figures.

## Outcomes and uncertainty

Unconditional escape, death, timeout, invalid execution, keys, door opening,
absolute task fitness, prediction losses and computational cost are reported
separately. Invalid executions stay in every applicable outcome denominator
and have zero combined fitness. Their partial raw task/forecast records are
identified; failures are not retried to obtain a more favorable outcome.
Successful escape time is explicitly conditional on success.

The primary family has two paired escape comparisons: selected online minus
selected frozen in the stationary and switching regimes. Exact two-sided
McNemar tests receive Holm correction within that family. The corresponding
two fitted-online minus fitted-frozen comparisons form a separately labeled
secondary family. Uniform-law comparisons, absolute comparisons against the
controls, prediction-use interventions and interactions between regime effects
are descriptive. A difference in significance between regimes is not an
interaction test. The executable plan lists every contrast and its family.

Escape effects use paired estimates and conservative 95% intervals obtained
from a simultaneous binomial rectangle for discordance probability and
conditional win probability. Regime-effect interactions subtract two 97.5%
component intervals to obtain conservative pointwise 95% coverage without
assuming independent regimes. Shared-case bootstrap interaction intervals are
diagnostic, with sparse/degenerate warnings. Ordinary paired 95% intervals are
also pointwise, not Holm-adjusted simultaneous intervals; Holm adjusts the
specified families' p-values only.

Prediction losses, task means and resource means use 5,000 bootstrap samples of
whole case identifiers, with the same sampled multiplicities across conditions
and regimes. Pooled loss sums/counts and relative reductions are recomputed in
each sample. Forecast cells and time bins are not independent replicates.
Prediction intervals are descriptive and pointwise; they are not another
confirmatory testing family. All exposed outcomes are reported regardless of
significance or direction.

## Identical experience and decision relevance

Five passive predictors receive each immutable-memory policy's recorded public
observations and preceding actions: fitted frozen, fitted online, known law,
selected frozen and selected online. The source-specific selected adapter
omits planner execution because its only planner writes are controller targets
that its learner and forecast do not read. This is verified against source and
development replay. Ordinary shadows receive no laws, enemy identities,
environment identifiers or target labels. Only the known-law shadow receives
current attempted-movement probabilities. No extra environment transitions are
drawn by replaying these records.

All five shadows are scored on identical evaluator-owned targets, including the
terminal transition. The target is horizon-one union enemy occupancy, in
origin-relative coordinates. Near targets are the pre-action 3×3 neighborhood;
audit targets are 12 independently sampled cells. The actual destination target
is occupancy after enemy movement, excluding entry contact; it is not total
death probability. A failed shadow retains all targets with the declared 0.5
forecast fallback and an explicit failure record. Empty policy records remain
in case resampling with zero forecast targets and in outcome denominators.

The selected uniform-planning pair supplies a second, policy-specific matched
comparison only if full trajectory and observation hashes agree on all
registered cases in the regime and executions are valid. No favorable subset
of matching trajectories is selected. Its results are separated from the common
memory-policy replay and from on-policy losses under differing actions.

State audits check raw adaptive quantities, constancy under freezing, online
changes, maps and localization. Action/observation sequence differences are
reported separately from beneficial control. Selected-source instrumentation
records the risks actually consumed by planning, with their event, horizon,
action conditioning and coordinate contract. Exported one-step occupancy and
multi-step heuristic route costs are not treated as interchangeable quantities.

The number of episodes that encounter the switch and the amount of subsequent
visible experience are reported with all-episode denominators. Consecutive
observations also count anonymous source-cell occupancy contrasts with known
attempt terrain. This is an opportunity proxy, not confirmed enemy identity or
measured information gain. Later prediction bins and post-switch forecast
analyses are explicitly survivor-conditioned.

## Execution, examples and reproduction

Assessment uses four local workers and no model calls. Immutable episode
checkpoints preserve partial cases. The pool is reserved once; resumption uses
the same frozen plan and omits `--reserve-pool`. Resource ledgers distinguish
candidate, evaluator, assessment-attempt worker and controller CPU, shadow
costs, elapsed time, and unavailable measurements. The attempt worker timer
includes evaluation and audit/retention, but excludes imports, dispatch and
case assembly outside the timer. Its sum is a measured component, not complete
worker-process CPU. Evaluator CPU is already included in that component and is
not added a second time. Whole-command host timing is reported separately.

Original selected/selected-frozen traces are retained for the first 32 cases in
each regime. After analysis closes, the example script chooses the lowest-index
case in each available paired outcome stratum, reports absent strata, and uses
the first action divergence or last common frame. Published images label
retrospective hidden-world context separately from the agent's partial view.
No example-dependent tuning or new rollout occurs. Figures and tables use the
repository's Chromatic Field theme and preserve negative or unavailable results.

The completed report provides exact commands for the frozen runner, saved-data
statistics, figures and retained examples. Native search archives preserve all
50 sources, branches, patch attempts, parent/inspiration relationships,
recommendations and actual calls. Private pools, full hidden traces and the
runtime database remain outside Git. The closed v2 assessment and its published
numbers retain their original scope and reproduction paths.
