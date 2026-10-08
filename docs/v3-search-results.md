# Unknown-dynamics v3: completed search

One native Shinka search completed its 50-slot budget on 8 October 2026: the
unchanged original predictive seed and 49 valid descendants, evaluated on the
same 24 development layouts in each of three regimes. All 3,600 condition-episodes
were valid. The four island copies of the seed produce 53 database rows but count
as one candidate slot. Native final meta analysis processed all 50 programs;
normal completion, process exit and lock release are independently recorded in
[native-completion.json](../artifacts/campaign-v3/native-completion.json).

Generation 46 led the reused development pool. The separate selection-validation
stage then selected **generation 4**, with 180/192 escapes and exact mean task
score **0.93088671875**. Selection evaluated five distinct candidate sources on
64 layouts in each of three regimes: **960 condition-episodes**, including one
invalid execution. The selected source is frozen in
[selection.json](../artifacts/campaign-v3/selection/selection.json). These are
selection-biased validation results, not fresh assessment or a verified benefit
from predictive adaptation.

| Selection rank | Generation | Mean task score (rounded) | Escapes / 192 | Invalid / 192 |
|---|---:|---:|---:|---:|
| 1, selected | 4 | 0.930886719 | 180 | 0 |
| 2 | 18 | 0.926182292 | 179 | 0 |
| 3 | 20 | 0.926182292 | 179 | 0 |
| 4 | 46 | 0.926046875 | 179 | 0 |
| 5 | 19 | 0.925738281 | 179 | 1 |

Selection sums exact integer task units, retaining invalid executions in the
denominator with zero task contribution. Generations 18 and 20 tie in both task
and escapes; the earlier generation wins that tie. The following comparison
retains the original development results to distinguish the search leader from
the selected program.

| Development measure | Original predictive seed | Generation 46 |
|---|---:|---:|
| Absolute task fitness | 0.799156 | 0.952483 |
| Escape / death / timeout, out of 72 | 56 / 15 / 1 | 69 / 2 / 1 |
| Near-cell Brier loss | 0.006127 | 0.007106 |
| Audit-cell Brier loss | 0.016230 | 0.016141 |
| Mean candidate CPU seconds per episode | 0.1691 | 0.6976 |

Task improvement did not imply improvement on every prediction diagnostic:
generation 46 had worse near-cell Brier loss on its own trajectories. These are
different experiences, so this comparison does not isolate prediction learning.
It encountered the switch in 17/24 switching episodes, with 401 subsequent steps
and 142 steps with a visible enemy. Analyses restricted to those survivors are
conditional. All 72 outcomes remain primary. Frozen-program assessment and
matched-experience analyses determine the claims
about the chosen program; this single search cannot establish reliable discovery.

The source changed substantially: native descendants introduced anonymous motion
inference, directional transition learning, persistent occupancy beliefs and
survival-conditioned planning. These changes are described against actual parents
in [source-changes.md](../artifacts/campaign-v3/source-changes.md), with complete
[programs](../artifacts/campaign-v3/programs/) and
[ancestry](../artifacts/campaign-v3/figures/ancestry.svg). Model-written patch
descriptions are retained as descriptions, not mechanism evidence. The historical
v2 generation 14 remains an immutable comparator. The manually engineered,
development-fitted direction-aware comparator is separate from the unchanged
v1 predictive source that initialized this search.

The selected generation 4 is a direct descendant of the seed. Its
[source review](../artifacts/campaign-v3/mechanism-gen4-review.json) identifies
uniform, slow and fast directional experts updated from anonymous observations,
plus three-stage planning with before/after enemy-occupancy hazards. Its hidden
occupancy estimate is rebuilt on each observation; future hazards are not
conditioned on surviving earlier planned positions. Thus the persistent-belief
and survival-conditioning changes reviewed in generation 46 do not describe the
winner. Source inspection alone establishes neither a learning benefit nor the
behavior of a freezing or prediction-use intervention.

The evaluator, splits and task-only objective were frozen before mutation. The
initial configuration used upstream ShinkaEvolve
`9912af12d423504b8d580f4179fd15f5f88b8c50` and Headless
`93cd9b06b85f848af1308c41e018991b33907c5e`. Resolved settings and actual execution
are distinguished in the [native audit](../artifacts/campaign-v3/native-audit.json).

| Mechanism | Configured | Observed |
|---|---|---|
| Islands and parent choice | Four islands; uniform island, weighted parent sampling | 49 sampling contexts used all four islands |
| Inspirations | One archive and one top-k slot | Archive context in 45 samples; top-k in 42 |
| Mutation | Diff/full probabilities 0.6/0.4; two patch attempts and two resamples allowed | 28 diff and 21 full descendants; one patch retry; zero resamples |
| Migration | Every 10 generations, rate 0.1, elitism | 13 recorded moves at generations 10, 20, 30 and 40 |
| Recommendations | Interval 10; one recommendation sampled into a request | Outputs at 8, 18, 28, 38, 48 and final 50; 42 requests contained recommendations |
| Concurrency | One proposal, evaluation and database worker | One campaign controller; native meta work could overlap evaluation |
| Model choice | `gpt-6-astra`, effort `high` | Single effective configuration; no model-bandit test |
| Novelty and prompt evolution | Disabled | No embeddings, novelty-judge or prompt-evolution calls |

No permitted embedding route was enabled, so native novelty was unavailable in
this configuration; it is not ruled out for a subsequent controlled study.
Temperature and token-limit metadata were not forwarded to Codex. All mutation,
repair, meta and probe calls used the strict subscription route; no paid fallback
occurred. Evaluation used local Python without a model.

There were **116 model calls**: 49 initial mutation requests, one native repair,
50 program summaries, six global-insight calls, six recommendation calls and four
readiness probes. All 112 native requests completed without nonzero return codes.
The one rejected patch arose because the pinned stdout parser discarded interior
blank lines. Native repair succeeded within the same candidate slot. The
[original stdout](../artifacts/campaign-v3/attempts/gen_5/attempts/novelty_1/resample_1/patch_1/headless-raw-stdout.txt),
parsed rejection and repair remain preserved. A narrow transport hook followed
a normal eight-slot checkpoint; recommendation continuity was verified in the
[complete resumed request](../artifacts/campaign-v3/request-example-gen8.md) and
its [binding record](../artifacts/campaign-v3/request-example-gen8.json).

An earlier amendment corrected the first request's description of the key count;
the evaluator formula never changed. The first mutation also made three observed
tool requests under inherited repository instructions. No protected pool access
appeared in those requests. Later campaign-local instructions restricted mutation
workers to supplied context. This instruction boundary is not an adversarial
filesystem guarantee. Original prompts, amendments and every rejected attempt
remain recorded; no global authentication or sandbox settings changed.

The following resource totals cover evolution and its development evaluations;
the 960 selection episodes are additional. Reported model usage totals
**3,811,874 tokens**: 1,969,941 uncached input, 1,511,168 cached
input and 330,765 output. The 103,212 reasoning tokens are an output subset.
Native-call monotonic durations sum to 10,972.70 seconds; probe durations were not
separately measured. Saved evaluations used 4,131.36 candidate CPU seconds,
248.84 evaluator CPU seconds and 4,022.97 episode wall seconds. Native logs span
16:03:43–20:39:24 UTC, or 4h 35m 41s including restart gaps; per-invocation logged
intervals sum to 16,376 seconds. These timestamps differ from observed host
process elapsed time for an undetermined reason. Remote model CPU, complete
controller CPU and actual subscription charges were not measured; native dollar
figures are API-list-price estimates. The original tool-session exit code was
lost during an interruption, so completion evidence does not assert that code.

The [archive check](../artifacts/campaign-v3/archive-readiness.json) binds all
sources and relationships to the closed native database. Repeated conversation
and meta text in lineage metadata is represented by canonical hashes, byte counts
and database-field pointers; individual attempt responses and meta outputs remain
public. Private pools and the large runtime database remain outside Git.

Regenerate the public figures and source-change table without a runtime database,
new episodes or model calls:

```bash
.venv/bin/python scripts/v3_report.py --saved
```

The search is complete. The exact recovery command for an interrupted copy of
this campaign is below; do not start a second controller or extend its budget:

```bash
HEADLESS_BILLING=subscription .venv/bin/python scripts/resume_v3_transport.py \
  --results results/campaign-v3 --generations 50
```
