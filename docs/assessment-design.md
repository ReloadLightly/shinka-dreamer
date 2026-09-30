# Frozen generation-14 assessment: prespecified design

This design was recorded before reserving or evaluating the new case pool. The
machine-readable authority is [preregistration.json](../artifacts/campaign-v2/assessment-1024/preregistration.json).
It supersedes the old suggested 256-case default for this assessment only. The
historical v1 pool and the completed 50-slot campaign remain unchanged.

There are exactly **1,024 fresh paired mazes and six conditions (6,144 episodes)**:
original memory, original predictive seed, generation 14, frozen generation 14,
fixed-risk generation 14 and frozen-plus-fixed-risk generation 14. The selected
source, controls, evaluator kernel and objective are frozen by hashes. No further
evolution, parameter tuning, model calls or sample-size adjustment is allowed.

The primary outcome is generation 14 minus frozen-weights **escape probability**.
Its two-sided exact McNemar test has alpha 0.05. Three secondary escape contrasts
(selected minus memory, original seed and fixed-risk) form a separate Holm-corrected
family at alpha 0.05, evaluated regardless of the primary outcome. This controls
the secondary family, not one combined family over every research question.
Death, timeout, door and two-key contrasts are exploratory; all directions and
unadjusted exact tests are reported. Forecast results are prespecified interval
estimation, with no additional confirmatory p-value claims.

For each binary contrast report paired wins, losses, both-positive and
both-negative counts. Let q be the probability of discordance, and theta the
conditional probability that a discordance favors the left condition. The paired
risk difference is q(2 theta − 1). Construct separate 97.5% Clopper–Pearson
intervals for q and theta, then take the extrema over their rectangle. By a union
bound, the resulting interval has at least 95% coverage. Conditional coverage for
theta holds at each discordant count; when there are no discordances its interval
is [0,1]. Thus zero discordances do not produce a misleading zero-width interval.
These intervals are intentionally conservative and need not invert McNemar's
test. Also report 98.333% versions for simultaneous coverage of the three
secondary differences. Per-condition rates use Wilson 95% intervals.

The mechanistic prediction contrast is learned versus frozen weights under
fixed-risk planning. Both complete action/world/enemy hashes and map/localization
hashes must match before calling it matched experience. Every episode is replayed
inside the unchanged evaluator, compressed to hashes and small diagnostics, and
its full trace discarded immediately afterward. Parameter constancy, updates,
position correctness, visible-map correctness and memory activity are recorded.
Mapping activity is described rather than assumed in episodes ending early.

For Brier differences and relative reductions, resample 10,000 whole paired
episodes and recompute pooled error sums divided by target counts on each draw.
Use RNG seed 20260930. Near-cell loss is the principal prediction measure; audit
and threat-conditioned loss are additional diagnostics. The 25-step learning
curves show pointwise intervals and episode/target counts, with explicit survivor
conditioning. On-policy losses under different actions remain separate.

Report task score, model score, keys, door completion, unconditional episode steps
and host runtime separately. Steps and runtime among successful escapes are
conditional on success, not unconditional efficiency. Invalid episodes remain
recorded non-escapes; no case is replaced because of its outcome.

Case files checkpoint all six conditions atomically. Condition order rotates by
case index. Descriptive progress is permitted; inferential tests wait for the fixed
sample to finish. Four local workers use the existing candidate isolation and
limits. Concurrency does not change the case pool or estimands.

After analysis is closed, choose the lowest case index in each selected/frozen
escape stratum: selected-only, frozen-only, neither and both. Display any available
strata, including failures, without selecting for effect magnitude or visual
appeal. Replayed examples must reproduce their saved metrics and hashes. Publish
their seeds and compact behavioral data as **exposed cases**, ineligible for any
future fresh assessment; the rest of the private pool remains unpublished.
