# Current task: original proposal restored

The user requested implementing the original Sakana proposal and a paper-style
README containing only Chromatic Fields figures from that proposal's experiments.
The active implementation is `proposal/`, evaluator
`namazu-proposal-reconstruction-v1`: original0.6task+0.4current-map-accuracy.
Predictive-learning requirements and full-machinery RUN1 plans below are historical,
not prerequisites for this track. Preserve their evidence; do not resume them.

This implementation step authorized local verification and an initial32-episode
development characterization, not another mutation-model campaign. The initial
seed ran32valid episodes/4,386transitions,0escapes,20captures,12timeouts. Native
search is configured and prepared but unexecuted. No automatic continuation.
The README describes actual results, limitations and reproduction commands.

---

# Current checkpoint: RUN 1 stopped incomplete

RUN1 stopped at 2026-10-09 01:51:46 UTC after a mutation timeout with missing
usage triggered its accounting guard. All RUN1 workers/services are stopped.
The user has objected to further allowance spent on infrastructure and reporting.
Do not restart, launch RUN2, resume v3 assessment, or perform additional analysis
without a new instruction. Completed results and recovery limitations are in the
README and artifacts/campaign-v4/run1/run-state.json. Earlier instructions below
are preserved history, not current authorization to continue execution.

---

# Authorized current work: RUN 1 of the reviewed roadmap

The user's 9 October 2026 instruction supersedes the earlier pause on **new** work,
but leaves the large historical v3 assessment paused. Execute **RUN 1 only**, then
checkpoint and stop. Later runs require a fresh user instruction. The durable
plan is [docs/research-roadmap.md](docs/research-roadmap.md), prospective settings
[docs/run1-protocol.md](docs/run1-protocol.md), and current machine state
[artifacts/campaign-v4/run1/run-state.json](artifacts/campaign-v4/run1/run-state.json).

RUN 1 is separately versioned wave v4: a bounded development branch-consequence
experiment and full native Shinka machinery from the original predictive seed.
The initial maximum is32slots including seed/failures,1.5Mreported uncached
input-plus-output tokens across all roles,120calls and the four-hour wall ceiling.
Admission stops by2026-10-09 04:24:23UTC; hard checkpoint by04:54:23UTC.
Prospective saturation reviews may stop earlier. Preserve all v2/v3 scientific
sources/data/statistics. Repository-wide Chromatic Fields restyling is now
explicitly authorized, using saved data with separate rendering provenance.

State recovery found current local and origin at reviewedcommit`ec7e4c3`, with
completed v3 interim analysis and profiling diagnostic; neither is repeated.
The new branch experiment completed23states/184futureattempts with56validator
failures preserved, no retries and2,032branchtransitions. Its limited valid
changed-action contrasts show no collision/escape improvement; see
[docs/run1-branch-findings.md](docs/run1-branch-findings.md). This is development
evidence, not an escape-population assessment. The subsequent search's actual
status/counts belong in the machine state and README, never inferred from the
configuration. Do not relaunch closed branch drivers or any historical campaign.

---

# Current research state and unknown-dynamics experiment

## Paused by the user: v3 wave 1

On 8 October 2026 UTC / 9 October Europe/Berlin, the user requested a checkpoint
and a full report before deciding direction. The subsequent request authorized
a detailed assessment of saved results and recommendations; that interim
analysis is now complete. **Do not resume assessment or start another search
until the next scope is decided.** All
assessment workers and its controller have exited; the reporting watcher is
stopped. The completed checkpoint preserves 140 cases, 4,200 world episodes and
2,100 passive prediction passes, with no unfinished attempts. The saved-data
interim analysis found improved matched prediction under stationary/switch laws,
but no established escape benefit from updating. It launched no new episodes
or experiment-route model calls. See [the interim report](docs/v3-interim-assessment.md),
[the original checkpoint report](docs/v3-checkpoint-report.md)
and [operator checkpoint](artifacts/campaign-v3/assessment/operator-checkpoint-complete.json).
The full 1,536-case plan remains preserved but incomplete. Commands below are
recovery documentation, not instructions to resume now.

The separate interim protocol was committed as `a69110f` before aggregation.
Results are in `artifacts/campaign-v3/assessment-interim140`; uncertainty is
nominal and descriptive, not the completed 1,536-case assessment. Preserve the
original frozen sources and the checkpoint as a historical snapshot. The exact
saved-data reproduction command is
`OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/v3_interim_analysis.py --compact`.
Any new experiment or resumed inference must disclose this interim look.

## Completed bounded development diagnostic

The user subsequently approved the proposed 20-minute development diagnostic.
Protocol `5914034` set zero new worlds, 48 replay passes capped at 80 recorded
frames each, and unchanged per-worker resource limits. Execution completed
8 October at 23:02:22 UTC, before the 23:10:04 execution and 23:15:04 task
deadlines: 42 passes complete, six traced comparator passes failed at the CPU
boundary, no retries, 3,306 returned frame records. All 24 profiled passes
completed. Existing fitting-subvalidation recordings only; no mutation or other
experiment-model calls, no assessment resumption. See
[the report](docs/v3-development-diagnostic20.md) and
[evidence](artifacts/campaign-v3/development-diagnostic20/summary.json).

Reproduce saved-data analysis with
`.venv/bin/python scripts/v3_diagnostic20_report.py`. The execution driver is
one-shot and deadline-bound; do not relaunch it or extend its closed budget.
Source-specific local caching and duplicate forecast removal are proposals,
not implemented or benchmarked changes. No further optimization experiment,
independent search or assessment resumption has been authorized.

## Previously authorized execution: v3 wave 1

The 8 October 2026 instruction authorizes **one 50-total-slot native search**,
followed by selection validation, frozen-program assessment and scientific
reporting. It supersedes the earlier proposed 24-search execution requirement.
Do not start another copy or extend the closed v2 database. The active campaign is
`results/campaign-v3`; the frozen design and identities are in
[docs/v3-protocol.md](docs/v3-protocol.md) and
[artifacts/campaign-v3/protocol.json](artifacts/campaign-v3/protocol.json).
The task-only objective, 24 development layouts × three regimes, original
predictive seed and native settings were frozen before mutation. The search is
now **complete at 50 slots**; do not restart it. Selection validation chose
generation 4. A 288-episode development audit verified its interventions.

Use the compact [search summary](artifacts/campaign-v3/search-summary.json) and
[native audit](artifacts/campaign-v3/native-audit.json) for the latest exported
counts. The frozen assessment has 1,536 paired layout cases, three regimes and
ten conditions. Inspect its existing checkpoints and controller before using
the exact continuation command:

```bash
.venv/bin/python scripts/v3_assessment.py \
  --plan artifacts/campaign-v3/assessment/preregistration.json \
  --out results/campaign-v3-assessment --workers 4
```

Run only one controller. Reserve the assessment pool once, only after its
complete preregistration is committed; resumption never draws another pool.
Original native driver/configuration files are hash-frozen;
the additive resume driver applies the preserved prompt-description correction.
Follow [methods and execution](docs/v3-methods.md) for that correction, comparator
fitting, matched prediction, checkpoints and remaining assessment gates. A final
assessment pool must not be drawn until all sources/interventions, analysis and
sample size are frozen. Existing v2 cases remain retired from fresh assessment.

The older study design below remains context for later independent repetitions;
it does not authorize or require launching 24 searches in this wave.

Read [AGENTS.md](AGENTS.md), the [README](README.md), the
[assessment design](docs/assessment-design.md), [development findings](docs/campaign-findings.md),
[evolved-agent walkthrough](docs/evolved-agent.md) and [protocol](docs/protocol.md).
Preserve [Namazu's proposal](docs/namazu-proposal.md), original controls, historical
evidence and the selected program unchanged.

## Completed campaign and current assessment

The original campaign is closed at **50 candidate slots, 0–49**: one seed,
45 valid descendants and four failed descendants. **Do not extend it.** Generation
14 remains selected, with source SHA-256
`588eeb7c10b978fe86c7e7b572f177e8b755c9c25699f3c75bfadd41192ec5e0`.

The subsequent assessment uses exactly **1,024 fresh paired mazes and six
conditions (6,144 condition-episodes)**. Its design was committed before generating
cases. The README and versioned assessment artifacts record its final findings.
The selected agent escaped in 962/1024 cases versus 966/1024 with frozen weights:
−0.39 percentage points (95% interval −2.23 to +1.21; exact p=0.557). Matched
near-cell Brier loss fell 3.31% (2.91–3.71%), while escape gains over the original
seed and memory were +13.87 and +2.83 points. Updating or using predictions did
not establish an escape advantage. All fixed-risk trajectory/map checks passed.
The case pool is an existing assessment, not a fresh pool available for another
claim. Published behavioral examples are explicitly exposed; exclude them and the
entire assessment pool from future fresh tests. Reuse them only as labelled
historical analysis. The assessment uses local execution and no model calls.

## Proposed next study: unknown enemy dynamics

**Original proposed design; v3 wave 1 is now launched under the scope above.** Give each study its own protocol,
objective/evaluator fingerprints, directory and preregistration, for example
`namazu-unknown-dynamics-v3`. Never append its scores to the completed v2 database.
Its hypothesis is that online updating improves prediction and control when enemy
transition laws vary across episodes or change within an episode, beyond what a
well-fitted static predictor achieves.

Preserve the original 15×15 partially observed maze, 5×5 observations, two keys,
locked door, exit, three moving enemies, dynamic walls every 25 steps and 200-step
horizon. Keep unrestricted joint evolution of `world_model_step`, `planner`, their
representations and helpers in the principal arm. Do not replace this problem
with another benchmark, a parameter sweep or a predefined algorithm catalog.
The new controlled change concerns the enemy transition law; record it as an
experimental extension, not a correction to the original environment.

### Environments and information boundaries

1. **Stationary hidden law:** sample one categorical distribution over the nine
   attempted enemy displacements for each maze. All three enemies share it and
   draw independently; the law remains fixed for the episode. Draw from a
   continuous, recorded distribution (initial proposal: Dirichlet concentration
   one per displacement, rejecting entropy below 1.2 nats or a component above
   0.6). Retain original collision/blocked-move handling: illegal attempts stay.
   The agent sees neither the distribution nor enemy identities.
2. **Unannounced change:** draw a second law independently, require total variation
   distance at least 0.3 from the first, and switch at a private uniformly sampled
   step from 25 through 75. This timing is independent of the policy and is not
   announced. Report unconditional episode outcomes and the number of cases that
   actually encounter the change. Post-change forecast windows are explicitly
   conditional on surviving to that window, preferably on matched experience.
3. Retain a **fixed original-law condition** as a continuity control. Original maze
   generation, objective reachability, wall changes and action order stay intact.
   Any training-only feasibility pilot must be completed before fixing the new
   protocol; do not tune these choices on the new assessment.
4. Use independent tagged streams for layout, enemy draws, law parameters, switch
   time and audit targets. Keep the enemy draw count fixed per tick. Test fresh
   layouts and fresh law draws; evaluate a separately declared distribution-shift
   suite only if its parameter support is fixed before search.

### Comparators and what is frozen

Include the immutable original memory baseline, a copy of v2 generation 14,
selected new programs with their verified learning/planning interventions, and:

- **Training-fitted then frozen predictor.** Fit the generation-14 softmax predictor
  on a separate training-only trajectory corpus under a fixed exploration policy;
  optimize Brier loss and select regularization/stopping using training-validation
  data only. Freeze its fitted weights before assessment. Compare it with an
  online version initialized at those exact fitted weights, using the same
  planner, observations and per-episode reset. Publish fitting data identities,
  optimization settings and final weight hashes. This tests adaptation beyond an
  imperfect hand-coded prior. It is a comparator, not a restriction on the
  representations permitted in joint evolution.
- **Known-law reference with partial observability.** Supply the current true
  displacement probabilities to a reference predictor, including the new law at
  a switch. It must still infer hidden enemy occupancy and terrain from the same
  local observations. It receives no unseen map, enemy positions, identities,
  future draws or future switch time. Use the same planner where compatible and
  also compare forecast quality on matched trajectories. Label this as privileged
  dynamics knowledge, not an optimal-policy or fully observable upper bound.

For every selected program, inspect what counts as adaptive predictive parameters
versus state estimation and ordinary map memory. Validate freezing and prediction
use on actual assessment traces. A source flag is insufficient. Preserve failed
interventions in the record and withhold causal learning claims when validation
fails; do not select a replacement program using final results.

### Independent search repetitions and comparison arms

Propose **eight independent searches per arm**, with **50 total candidate slots
per search**, four native islands and equal evaluation budgets: 24 searches in
all. This budget requires a separate execution authorization and subscription
capacity; none of these searches is part of the completed assessment task.

- **Joint arm:** both model updating and planning, representations and helpers may
  evolve substantively. Keep algorithm choice unrestricted within resource limits.
- **Model-only edit arm:** freeze the planner and its planning helpers. Permit
  arbitrary estimator/update representations behind a documented model-to-planner
  interface. An adapter may expose map/risk queries; it must not choose actions.
- **Planner-only edit arm:** freeze the world-model implementation and predictive
  exports. Evolve planning and its private state while supplying read-only model
  outputs. Audit that planner execution cannot modify the frozen predictor state.

The controlled arms need explicit module boundaries and checks of unchanged
components on identical observation streams. They measure the value of editing
those modules under that interface; they do not prove that all internal planning
computation is free of implicit predictive reasoning. Keep these interface
limitations separate from the unrestricted joint arm.

Use independent search RNG streams, databases, LLM calls, archives and
recommendation histories for each repetition. Share prespecified training-case
blocks across corresponding arms for comparability, but never pass discoveries
or recommendation text across repetitions. Record provider/model revisions and
resolved native settings; model sampling need not be exactly reproducible merely
because evaluation streams are controlled. Native parent/inspiration sampling,
lineage and recommendations remain available on the authorized subscription
route. Do not claim novelty embeddings or model-bandit experiments when disabled.

### Selection, assessment and inference

Fix training and validation splits, per-run selection rule, sample sizes, primary
contrasts and multiplicity before launching search. Freeze each run's selected
program before reserving a new common assessment pool. Keep assessment cases,
observations and outcomes out of mutation and recommendation context.

A concrete initial assessment budget is 1,024 paired mazes **per environment
regime**, shared across the frozen programs and controls. Size changes must be
based on a training-only power/precision calculation before drawing these cases,
never on assessment significance. The principal proposed control endpoint is
online versus training-fitted-frozen escape; define arm-level generalization
contrasts separately. Prespecify a multiplicity family for stationary and
changing-law control tests. Forecast changes use matched-trajectory pooled Brier
ratios with episode-level uncertainty; report near, audit and threat diagnostics
without treating linked losses as independent replications.

The search repetition is the unit for claims about evolutionary search. Report
all selected runs, their distribution and failures, with a crossed bootstrap that resamples repetition blocks and common case IDs
independently, preserving the same sampled cases across arms and programs. Do not
resample shared test cases separately within each run and lose their dependence.
A large number of test mazes
cannot substitute for independent searches. Use exact paired binary inference
within each frozen-program comparison when discordances are sparse. Report
escape, death, timeout, keys, door, task, forecast losses and compute separately;
successful-escape efficiency remains conditional on success. Resource failures
are outcomes, not grounds for deleting difficult cases.

## Working and delivery rules

- No paid model calls or silent billing fallback. Credentials do not authorize
  spending. The unknown-dynamics study requires a separately authorized launch.
- Never alter global authentication, approval or sandbox settings to bypass a
  blocker. Keep evaluator/hidden state independent of candidate code.
- Keep source, raw evidence, private pools and historical artifacts intact. Store
  large databases, credentials and hidden traces outside Git. Commit compact
  episode data, hashes, lineage, uncertainty and figures sufficient for review.
- Keep mechanisms and scientific results central in README; put execution
  logistics in supporting documentation. Negative findings are valid results.
- Verify repository root and origin, commit coherently and push only to
  `ReloadLightly/shinka-dreamer`, without force-pushing.
