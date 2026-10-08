# Unknown-dynamics wave 1: methods and execution

The frozen design is [v3-protocol.md](v3-protocol.md), with executable identities
in [protocol.json](../artifacts/campaign-v3/protocol.json). This is one independently
versioned search, not a continuation of the closed v2 population. All v2 controls,
selected source, results and original Namazu proposal are preserved.

## What changed

Only the enemy attempted-movement law changes the environment. Geometry,
observations, interaction/movement/collision order, dynamic walls, keys, door,
exit and episode horizon reuse the unchanged original implementation. V3 uses
one inverse-CDF uniform innovation per enemy per tick. Its uniform regime retains
the original categorical law but does **not** reproduce v2's `random.choice`
mapping from seeds to trajectories. V3 uniform scores are new-environment
continuity results, never entries in the v2 database.

The law draws, rejection criteria and switch schedule are the proposed initial
choices. Switch transition K is the first transition using the replacement law.
An ordinary candidate receives no regime label, seed, law, switch announcement,
enemy identity or forecast target. A separately identified known-law reference
receives only the probabilities for the next transition. Its future planning
still lacks unseen terrain, hidden enemy positions, future innovations and the
future switch schedule.

The chosen-cell diagnostic predicts **post-transition enemy occupancy at the
actual destination**. It excludes pre-movement enemy contact, and is not the
probability of death. Ordinary near/audit forecasts use fixed cells in coordinates
relative to the agent's initial position. Near targets surround the pre-action
position. A planner's longer-horizon heuristic cost need not equal these forecasts.
Memory's near/audit diagnostics use the evaluator's persistence reference;
its destination diagnostic uses the original source's exported prior prediction.

## Comparators and matched experience

The evolutionary starting source is the unchanged original predictive seed.
Original memory and historical v2 generation 14 are immutable controls.
`controls/v3/directional.py` is a manually engineered comparator, derived from
generation 14's map, occupancy filtering and planner but replacing its transition
family with nine absolute attempted-direction logits. Illegal attempted moves
accumulate at the source cell. Anonymous occupancy uses an approximate union of
source contributions; it does not observe identities or perfectly track overlap.

Fitting collected 264 development condition-episodes under the fixed memory
policy: 64 training layouts and 24 predictor-subvalidation layouts, each in three
regimes. Fits use visible next-observation targets only; terminal transitions are
excluded from that training corpus. L-BFGS-B fits Brier loss with three specified
ridge candidates; subvalidation chooses ridge 0.01 and online step size 2.0 from
0.5, 2.0 and 8.0. Case counts are equal across regimes, but pooled target counts
give slightly unequal realized loss weights. See the exact
[fitting record](../controls/v3/fit-results.json) and
[provenance](../controls/v3/manifest.json). Frozen and online sources and initial
parameters are identical. Frozen disables transition-parameter updating while
mapping and occupancy-state filtering continue.

Final matched comparisons replay the frozen memory policy's experience through
isolated predictors. Each receives the same current observation and preceding
recorded action. The reviewed comparator is action independent apart from door
interaction, which is always enabled by this policy; no shadow planner executes.
Near, audit and destination labels remain in the evaluator. All recorded
transitions, including terminal transitions, are scored. A failed shadow retains
every target with an explicit 0.5 fallback and error record. This endpoint measures
the executable predictor, including failures, and is distinct from on-policy loss.

The selected evolved program receives its own source-specific mechanism review.
Its prediction-use intervention is described by the computation actually removed
or replaced. The analysis only labels fixed-policy predictions as matched if
every prespecified trajectory and physical-observation hash agrees; it does not
select a favorable matching subset after assessment.

Outcome denominators always include invalid executions. Ordinary on-policy
forecast losses pool only targets produced before an execution fails, with target
counts and invalids reported beside them; they are not reliability-adjusted loss
comparisons. The passive matched predictor endpoint instead keeps every recorded
target through its declared missing-prediction fallback.

## Native execution and corrections

Actual upstream ShinkaEvolve is pinned at
`9912af12d423504b8d580f4179fd15f5f88b8c50`; Headless is pinned at
`93cd9b06b85f848af1308c41e018991b33907c5e`. Installed source hashes and native
resolved settings are recorded. The effective model is `gpt-6-astra`, effort
`high`, through the subscription route. Temperature and `max_tokens` remain native
metadata but are not forwarded Codex controls. API-list-price cost estimates in
logs are not subscription charges. Mutation, repair, recommendation and readiness
calls are audited separately; evaluation, fitting and statistics use no model.

Four native islands, native parent and inspiration sampling, diff/full mutation
and meta recommendations remain configured. Actual usage, migrations, repairs,
rejections, failures and model roles are recorded by `scripts/native_v3_audit.py`.
Novelty embeddings and judge calls are inactive; no authorized embedding route
was enabled. One model configuration cannot demonstrate model-bandit selection.

The first mutation's prompt incorrectly described keys as a fraction. The
evaluator always used the correct raw count, 0–2. After slot 1 completed, the
controller stopped with no active model request and resumed using the exact
one-phrase correction. The original prompt/configuration and affected request
remain preserved in the [amendment](../artifacts/campaign-v3/prompt-amendment.json).
The prompts' label `absolute-task-v3` refers to the v3 study; the canonical evaluator
objective identity is `absolute-task-v1`, with the same formula.

The first model call also obeyed inherited repository instructions by reading
`CODEX_TASK.md` and `docs/design.md`, then ran two synthetic self-checks. The
audited tool requests showed no protected v3 pool access. Campaign-local mutation
instructions now require use of supplied context only; these are task-specific
instructions, not a change to authentication or sandbox policy. Candidate workers
retain the existing Landlock/seccomp restrictions. The LLM's read-only tool policy
alone is not an adversarial filesystem isolation guarantee.

The first diff against the original seed exposed a separate transport defect:
the pinned Headless stdout parser discarded interior blank lines, invalidating
an exact SEARCH block. Native repair recovered the candidate; its rejected
attempt and extra call remain recorded. After slot 7, a normal drain checkpoint
saved all eight slots and native recommendations. An additive runtime hook now
preserves assistant-text whitespace. The pinned upstream files, candidate
objective and configured search mechanisms are unchanged. The checkpoint's
additional native summary/recommendation calls are counted separately.

## Reproduction and continuation

For the existing local campaign, use exactly one controller:

```bash
HEADLESS_BILLING=subscription .venv/bin/python scripts/resume_v3_transport.py \
  --results results/campaign-v3 --generations 50
```

Do not run this command while its controller already holds the campaign lock.
`SIGUSR1` requests a normal checkpoint after pending native work; the same resume
command restores saved proposals, native population state, random state and
recommendation history. Total slots include seed and failures; seed island copies
are not additional slots. The original `scripts/evolve_v3.py` is preserved as the
initial-launch record; use the additive resume entrypoint for this amended run.

Development pools can be recreated locally without drawing assessment cases:

```bash
.venv/bin/python scripts/v3_development_pools.py
.venv/bin/python scripts/v3_fit.py --workers 2 \
  --out results/v3-comparator-refit \
  --source-out results/v3-comparator-refit.py
.venv/bin/python scripts/v3_comparator_check.py --workers 2
.venv/bin/python scripts/v3_report.py
.venv/bin/python scripts/native_v3_audit.py
```

After all 50 slots, `scripts/v3_select.py --workers 4` evaluates up to five distinct
development leaders on the separate 64-layout selection pool. That pool is not
fitting subvalidation and its outcomes remain selection-biased. Before a fresh
assessment pool is drawn, freeze the selected source, comparator/interventions,
precision calculation, analysis and driver hashes. `scripts/v3_assessment.py`
then enforces those identities and preserves each completed condition episode.
Private seeds and traces and large runtime databases remain outside Git. Committed
compact outcomes and analysis scripts reproduce numerical reporting without model
calls; execution on another host requires an explicitly recorded runtime identity.

`scripts/v3_precision.py` uses the selected program's completed development
freeze audit to plan for a three-percentage-point escape difference, near the
historical v2 selected-versus-memory effect. It computes exact paired-binomial
power at alpha 0.025 for each of the two primary regime comparisons, targeting
80% power. The planning discordance is the larger individual one-sided 95%
development upper bound, with a floor of 0.08. Sample size starts at 1,024 per
regime and increases in steps of 256. These are conditional planning calculations:
the 24 development cases per regime were reused by search, and their bounds are
neither selection-adjusted nor a simultaneous two-regime guarantee. The calculation
also reports sensitivity and paired intervals at rounded expected counts. Final
sample size is fixed before drawing assessment cases and never depends on their
significance.

The two primary escape comparisons are selected versus frozen in the stationary
and switching regimes, with Holm correction within that family. Fitted online
versus fitted frozen uses a separate secondary two-regime family. Uniform-law
and absolute-control comparisons are descriptive. Prespecified differences
between regime-specific selected-versus-frozen effects use the same paired cases;
conservative intervals subtract two 97.5% paired-effect intervals, giving at least
95% coverage by a union bound without assuming independence. These descriptive
intervals are pointwise, not simultaneous across all regime contrasts. A shared-case
bootstrap is also reported as a diagnostic, with sparse-discordance and degenerate
resampling warnings. Different within-regime significance results alone do not
establish an interaction.

The assessment passively retains selected/frozen traces only for the first 32
registered case indices. After the numerical analysis closes, the example renderer
uses the lowest available index in each escape-outcome stratum and regime. It
shows the first differing action, or the last shared frame. Missing strata remain
missing. This produces retrospective examples without rerunning or replacing an
inferential episode; a pictured decision need not explain the complete outcome.
