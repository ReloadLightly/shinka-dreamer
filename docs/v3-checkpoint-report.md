# ShinkaDreamer: checkpoint and decision report

Recorded after the user-requested stop on 8 October 2026 UTC / 9 October in
Europe/Berlin. This report separates completed evidence from the unfinished
assessment. No further experiment or automatic resumption is authorized by it.

## Status and main finding

**The assessment is stopped.** All assessment workers and its controller have
exited; the automatic analysis watcher was also stopped. The closed evolutionary
search has 50 candidate slots. Its selected program is generation 4. We retained
140 fully evaluated assessment cases: **4,200 world episodes and 2,100 passive
prediction passes**, with no unfinished attempts. Four drained cases were
assembled from existing checkpoints without rerunning anything.

The central finding so far is an **early development-performance plateau, with
small development prediction improvements but no demonstrated control benefit
from online learning**. The fresh assessment remains incomplete and its treatment
effects have not been aggregated. It must not be presented as a completed or
confirmatory result.

The operational mistake was launching a statistically ambitious assessment
without first converting its total workload into a measured runtime budget and
making that cost clear. The batch scheduler also left avoidable idle capacity.
Scientific precision alone did not justify treating the next working day as an
automatic execution budget.

## What the experiment is trying to determine

The question is when an evolved learner that updates its model of enemy movement
actually makes better decisions and escapes more often. Four claims need
different evidence: better task performance; better prediction on identical
experience; a control benefit caused by predictive adaptation; and reliable
discovery across independent searches.

The earlier, closed v2 campaign had already illustrated the gap: its selected
generation 14 improved escape over original memory by 2.83 percentage points and
matched prediction loss by 3.31%, but online versus frozen escape was −0.39 points
(95% interval −2.23 to +1.21). Those historical results and their reproduction
paths remain unchanged; their exposed cases are not fresh v3 tests.

V3 retains the 15×15 maze, 5×5 observations, two keys, gating door, exit, three
anonymous enemies, dynamic walls and 200-action horizon. Its separately versioned
extension has three equally weighted regimes: uniform attempted movement; a
privately sampled stationary directional law; and an unannounced replacement law
between transitions 25 and 75. Blocked attempted moves become stays. Layouts,
laws, switch timing, enemy innovations, forecast targets and candidate randomness
have separate tagged streams. Ordinary candidates do not receive hidden laws or
enemy identities.

These are defensible initial dynamics choices, not evidence that the task has a
large adaptation-dependent performance gap. Fitness is absolute task performance;
proper prediction losses are diagnostics and textual feedback. This avoids
rewarding an agent for sabotaging its frozen counterpart, but does not force useful
adaptation when mapping and planning can already solve most cases.

## What actually happened in the 50-slot search

The seed was the immutable original predictive program, not the manually
engineered comparator. Actual upstream ShinkaEvolve used four islands, native
parent/inspiration sampling, diff/full mutations, migration and recommendations.
There was one effective model configuration, `gpt-6-astra` at `high` effort, through
the subscription route. No paid fallback occurred. Novelty embeddings/judge calls
were inactive; a single model does not test a model bandit.

Each candidate saw the same 24 development layouts in three regimes: 72 episodes
per slot. The budget contained the seed and 49 valid descendants. All 3,600 search
episodes were valid. One rejected patch required a repair within its slot.

| Reused-development measure | Seed, generation 0 | Generation 4 | Development leader, generation 46 |
|:--|--:|--:|--:|
| Mean task fitness | 0.799156 | 0.950906 | 0.952483 |
| Escapes / 72 | 56 | 69 | 69 |
| Deaths / timeouts | 15 / 1 | 0 / 3 | 2 / 1 |

Generation 4 is the fifth total slot, including the seed, after four mutations.
It captured **98.97% of the eventual task-fitness improvement**. Of the next
45 candidates, only generation 46 improved task fitness, and **none improved
escape count**. That final 0.001576 fitness increment came from the aggregate
successful-escape speed term. Five later candidates tied 69 escapes; 40 escaped
less often. All branches and failures remain available in the
[search explorer](../artifacts/campaign-v3/search-explorer.html).

The user's concern about diminishing returns is therefore supported by the data.
Completing 50 slots followed the requested initial budget, but most later
computation did not improve the observed escape result. This plateau does not
prove that five slots will reliably suffice on another independent search.

Separate selection validation tested five finalists on 64 layouts in three
regimes: 960 episodes. Generation 4 won with 180/192 escapes; the other finalists
had 179/192, with one invalid execution among them. This narrow selected-validation
margin is selection-biased. It is not a fresh generalization result.

Generation 4 is a direct descendant of the seed. It introduced uniform, slow and
fast directional movement experts learned from anonymous observations, together
with three-stage hazard-aware planning. The result is substantive executable code,
but this run does not show that a long evolutionary lineage, migration or
recommendations were necessary to discover it.

## What we know about learning and control

The selected-source audit distinguishes predictive parameters from localization,
mapping and occupancy-state inference. Freezing stops predictive updates and
forgetting while leaving mapping active. A second intervention substitutes the
uniform movement law in planning; it still predicts and plans. Instrumentation
was checked against the original selected source for unchanged actions and
original exports on development experience.

In the 288-episode development mechanism audit, all four selected-program
conditions escaped on the **same 69/72 cases**. Online versus frozen learning
changed action sequences on **50/72 cases**, but changed no escape outcome.

On verified identical uniform-planning trajectories, online updates changed
near-cell Brier loss by these relative reductions:

| Regime | Reduction from learning |
|:--|--:|
| Uniform | −0.072%: slightly worse |
| Stationary hidden law | +1.560% |
| Unannounced switch | +1.278% |

These are reused-development point estimates, not fresh inferential results.
Better prediction and different decisions do not establish better control.

A manually engineered, development-fitted directional comparator and its online
counterpart share the same planner and initialization. A known-law reference
receives current movement probabilities while retaining partial observation; it
is not an optimal-policy bound. In the short development diagnostic, known-law
information changed actions on 15/24 cases, but all three controls escaped on
the same 23/24 cases. That also gave little positive evidence that more accurate
dynamics information was the limiting factor for escape.

Lack of switching exposure is not a sufficient general explanation: the selected
program encountered the switch in 21/24 development switching episodes, with
701 post-switch steps and 340 steps with a visible enemy. Exposure still does not
guarantee identifiable information or a decision where that information matters.

## Why the assessment became expensive

The frozen plan used 1,536 shared layouts, three regimes and ten conditions:
**46,080 world episodes**, plus five predictors replaying each memory-policy
recording: **23,040 passive passes**. The ten conditions are original memory,
original predictive seed, v2 generation 14, fitted frozen, fitted online,
known-law, selected online, selected frozen, and the two selected uniform-law
planning interventions.

The chosen target was a three-percentage-point paired escape difference, with
at least 80% conditional planning power for each primary selected-online versus
frozen comparison (stationary and switching), planning at alpha 0.025 for the
two-test Holm family. This is not joint power to detect both effects. Zero
development escape discordances on only 24 cases per regime did not imply zero uncertainty; the
calculation used a conservative discordance value of 0.11735 and selected 1,536
cases. This was a statistical planning choice, not a demonstrated effect size or
an agreed wall-clock budget. Reused development data and selection limit the
interpretation of that power calculation.

Execution was already parallel: four worker processes on an Intel i5-1035G1 with
four physical cores and eight logical CPUs. There was no provisioned distributed
cluster. However, the driver submitted four cases together and waited for the
slowest before dispatching the next batch. The resource-only audit estimated
**21.3% idle worker capacity** within those batches. A rolling queue could recover
some of this; it cannot produce a 30-fold speedup.

At the 136-case audit, 86.8% of measured CPU time was spent in candidate and shadow
predictor processes. The remaining workload was approximately 26.6–30.0 CPU-hours.
Even ideal scaling across eight independent CPUs implied 3.3–3.8 hours; actual SMT
scaling would be worse. A 15–20 minute completion would need roughly 80–120
comparable fully utilized CPU equivalents, before coordination overhead.

The stopped assessment used **10,751.56 whole-command CPU seconds, about 2.99
CPU-hours**. GNU elapsed time was **58 minutes 19 seconds**. The driver and wrapper
recorded approximately 52 minutes 55 seconds on their monotonic clock; UTC and GNU
elapsed were about 10% longer, for an unresolved reason. These measurements are
retained separately. Whole-command CPU overlaps per-episode counters and is not
added to them. The [throughput audit](../artifacts/campaign-v3/assessment/throughput-review.json)
and [checkpoint](../artifacts/campaign-v3/assessment/operator-checkpoint-complete.json)
preserve the exact scopes.

## Exact accounting at the pause

| V3 activity | World episodes executed |
|:--|--:|
| Development, fitting corpus and diagnostics | 830 |
| Native search | 3,600 |
| Selection validation | 960 |
| Incomplete assessment | 4,200 |
| **Total to date** | **9,590** |

There were additionally **2,132 isolated shadow passes**: 32 development and
2,100 assessment. The 372 in-process prediction passes and 264 feature-construction
passes are separate work on recorded data; none creates another world episode.
Tests are not added to these scientific counts. Shared cases are dependent, not
9,590 independent experimental units.

The incomplete assessment has 27 invalid world executions, no shadow failures
and no unfinished attempts. Recorded failures were consistent with the fixed
CPU boundary; they remain in the saved outcomes and are not replaced.

The v3 experiment route recorded **116 model calls**: 49 mutation requests, one
repair, 50 program summaries, six global-insight calls, six recommendations and
four readiness probes. Reported usage was **3,811,874 tokens**, including cached
input; reasoning tokens are an output subset. The assessment runner makes no
model calls. These counts do **not** include this assistant conversation and
subagent orchestration, whose usage is not measured by the experiment ledger.
Consequently, “no assessment model calls” must not be interpreted as “no assistant
subscription usage while discussing or supervising it.” Actual subscription
charges are unmeasured; API price estimates are not bills.

All 85 protected historical files remain unchanged. Source archives, rejected
patches, ancestry and native execution records are preserved. Private pools and
large runtime databases remain outside Git.

## Design weaknesses and decisions now needed

The weaknesses are substantive as well as operational:

- Most task improvement appeared in one early direct rewrite. One run cannot
  establish reliable discovery or the benefit of longer evolution.
- Reusing 24 development layouts and reaching 69/72 escapes leaves little room
  for additional escape gains and allows rankings to depend on speed or small
  case differences.
- Dynamics uncertainty was added, but pilot controls did not establish that
  dynamics knowledge was a consequential control bottleneck.
- Generic occupancy loss can improve without improving the risks that decide
  the next action. Later/post-switch analyses also select survivors.
- Absolute task fitness is defensible, but the experiment cannot assume it will
  discover adaptation when other planning improvements yield the reward.
- The full ten-condition assessment allocated substantial computation to
  precision before the value of that precision and the operational cost were
  made clear. A runtime estimate and a hard budget should have preceded launch.

Reasonable options for discussion, with nothing started automatically:

1. **Analyze the existing 140-case checkpoint descriptively.** This needs no new
   simulation or experiment-model calls. Preserve the original preregistration,
   label the administrative early stop and reduced precision, and do not claim
   the planned assessment was completed. It can identify whether further testing
   is likely to answer a useful question. If these effects inform whether to
   resume, record that interim look and decision; a later report must disclose
   them rather than describe an uninterrupted blinded fixed-sample assessment.
2. **Design a small development-only decision-relevance pilot.** Inspect cases
   where better dynamics information changes the actual chosen action/risk and
   whether that can affect escape. Use a written case, CPU and wall-time budget.
   Any changed environment or objective belongs to a new versioned wave; do not
   tune on this assessment pool and relabel it fresh.
3. **Study discovery efficiency with short independent searches.** Compare early
   successes under fixed budgets before claiming that long searches or specific
   Shinka mechanisms add value. Predeclare model-call and wall-time caps.
4. **Complete the original assessment only if its precision is worth the cost.**
   Improve scheduling under an explicit execution amendment or use separately
   approved compute. Do not promise twenty-minute local completion.

My recommendation for discussion is to consider the existing checkpoint first,
then decide whether the next uncertainty is decision relevance, discovery
reliability or statistical precision. This is a recommendation, not authorization
to start analysis or another experiment. ShinkaDreamer is not neural Dreamer, and
the current work makes no novelty or publication claim.

For a second opinion: **given the early search plateau, improved development
prediction without improved escape, near-ceiling task performance, and this cost
profile, which uncertainty is worth resolving next—and what explicit compute
budget should be spent resolving it?**

The exact original recovery command is preserved for a future scope decision;
**do not run it now**:

```bash
.venv/bin/python scripts/v3_assessment.py \
  --plan artifacts/campaign-v3/assessment/preregistration.json \
  --out results/campaign-v3-assessment --workers 4
```

It reuses the original pool and completed checkpoints. The old automatic reporting
watcher was stopped and must not be blindly restarted after its interrupted-launch
history. Current status: **paused for discussion, not completed**.
