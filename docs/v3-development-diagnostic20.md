# Bounded development diagnostic: measured cost and decision sensitivity

The authorized diagnostic is complete. It used **zero new world episodes, zero
assessment cases and zero experiment-model calls**. The long assessment remains
paused. The replay execution took 115 monotonic seconds (125 seconds by UTC),
well inside the 20-minute task limit recorded before execution.

The useful result is specific: **the fitted comparator spends most measured
main-pipeline CPU in planning, particularly continuation risk. Supplying the
true dynamics law sometimes changes its chosen action at the same state, but
most recorded decisions stay unchanged.** This motivates a small, separately
tested optimization; it does not justify restarting the large assessment.

## Budget, data and executed work

Protocol commit `5914034` records a task start of 22:55:04 UTC on 8 October 2026,
an execution deadline of 23:10:04 and a task deadline of 23:15:04. The caps were
zero worlds, 48 restricted replay passes, 80 frames per pass, two workers and
the unchanged 10-second CPU/192 MiB boundary per worker. No retries were allowed.

We used the first four existing fitting-subvalidation recordings in each of
uniform, stationary and switching dynamics: 12 development recordings. The
recording policy was original memory. These data were already exposed during
comparator fitting and are neither champion selection nor fresh assessment.
The original selected source and fitted comparator were each replayed with and
without profiling, using identical recorded observations and preceding actions.

| Execution count | Actual |
|:--|--:|
| Replay attempts | 48 |
| Complete / failed / unstarted | 42 / 6 / 0 |
| Requested recorded-frame executions | 3,368 |
| Returned frame records | 3,306 |
| Complete selected passes | 24 / 24 |
| Complete profiled comparator passes | 12 / 12 |
| Complete traced comparator passes | 6 / 12 |
| New world episodes / mutation calls | 0 / 0 |

All six failures were comparator passes carrying the native-ranking trace
observer; they exited −9 at the unchanged execution boundary. Their saved
prefixes are retained. These are **diagnostic instrumentation failures**, not
additional on-policy comparator failures. We did not relax limits or rerun them.

Across all 24 plain/profiled pairs, all **1,622 common-prefix action/model hashes
matched exactly**. Eighteen pairs completed in full. All **780 common-prefix
known-law actions** also matched. Every observed first-ranked native tuple
agreed with the actual planner action. Missing tails are unavailable, not
assumed equivalent.

## Where the time goes

The following shares use measured process CPU around the three main functions
in the 12 complete profiled passes per source. Profiling adds overhead, so these
are diagnostic shares, not unbiased production timing or a speedup benchmark.

| Main function | Selected program | Fitted comparator |
|:--|--:|--:|
| World-model update and forecast | 50.02% | 13.66% |
| Planner | 49.24% | 85.94% |
| Model export | 0.74% | 0.40% |

For the selected source, `_forecast` accounts for about 98% of profiled
world-model cumulative time, and `_distances` for about 69% of planner cumulative
time. The main costs are spatial forecast propagation and graph search.

For the comparator, `_continuation_risk` accounts for about 61% of profiled
planner cumulative time. Its nested source/action loops repeatedly construct
transition rows. `_transition_rows` appears prominently in both prediction and
planning. Export is a small fraction for both sources.

Cumulative timings overlap: a parent includes its children. They must not be
added as independent costs. Profiled phase measurements exclude the known-law
clone and recorded-action alignment. Whole-worker CPU includes those operations.
The traced comparator's other pass is also instrumented and cannot serve as an
uninstrumented runtime baseline.

## Does correct dynamics information change actual choices?

For the comparator, the diagnostic freezes fitted parameters, updates on a
recorded public observation, and then deep-copies that exact internal state.
Only the copy receives the true law for the next transition; its forecasts are
recomputed before invoking the identical planner. The copy is then discarded.
The original belief, map, localization and planner history therefore agree
before the intervention. No hidden law enters the ordinary state or selected
agent. Tagged law draws reproduce the frozen environment without constructing
or stepping a maze.

The recording policy had disabled predictive planning. The diagnostic explicitly
enables it in both sources; otherwise the test would incorrectly leave the law
unused. Comparator waits and previous forecast center are aligned to recorded
actions after export, while target history follows the fitted planner. These
are recommendations on forced recorded experience, **not executed trajectories**.

All 12 profiled comparator passes completed and give the full action denominator:

| Regime | Different chosen moves / recorded decisions | Recordings with a changed move |
|:--|--:|--:|
| Uniform | 2 / 281 (0.71%) | 2 / 4 |
| Unknown stationary | 14 / 287 (4.88%) | 4 / 4 |
| Switching | 4 / 274 (1.46%) | 2 / 4 |

In switching recordings, one of 102 decisions after a recorded replacement-law
outcome changed. This is a small development prefix, not a precise estimate of
the population rate or information gained from that outcome.

Native ranking/score details are available for 780 decisions in the traced
passes, with 17 changed choices: two uniform, 11 stationary and four switching.
For 12 of those 17 changes, both alternatives have zero immediate occupancy risk
under the known-law model. The known-law choice reduces immediate risk in three
changes and increases it in two; future costs can outweigh immediate risk.
This provides a concrete reason to inspect continuation and navigation, not
just a generic prediction score.
The comparison concerns a change in model-based rankings. Native heuristic
cost differences are not measured reward improvements, death probabilities or
counterfactual escape effects. A perfect movement law also leaves partial
observability and the planner's approximations unresolved.

## Small proposed change

**First candidate: cache directional transition probabilities once per source
cell within a comparator planner call.** The inspected directional
`_transition_rows` ignores its `center`, `velocity` and `degrees` arguments.
Nevertheless, `_continuation_risk` reconstructs velocity information and full
rows for different destination queries. A local probability lookup keyed by
source cell could reuse those rows for all destination queries and remove
velocity work that has no effect on this comparator's transition law.

This proposal is specific to `controls/v3/directional.py`; it must not be applied
blindly to historical v2, where the feature family differs. Keep the cache local
to one planner call so changing laws, walls, doors and state cannot make it stale.
Preserve blocked-attempt staying mass, summation order, survival conditioning
and tie breaking. Learning's derivative rows must remain available where needed.

A second, smaller selected-source opportunity is visible in `_forecast`: it
computes `following = spread(previous)` and then repeats the same propagation
when assigning the next visible distributions. Reusing `following` could remove
that duplicate work. It addresses only part of forecasting, not the whole run.

**Neither optimization has been applied or benchmarked.** The replay budget is
closed. The next proposed task is a separate bounded optimization check: make
one source-specific copy, require exact action/forecast/state equivalence on
development recordings, and measure untraced paired runtime before deciding
whether the gain is worth using. Do not claim a speedup, overwrite frozen sources,
or resume the large assessment on this evidence alone. For the scientific
question, better predictions reaching only a few action changes remains a
reason to examine decision relevance before paying for much greater precision.

## Cost, limits and reproduction

Execution finished at 23:02:22 UTC. Reaped workers used **250.26 CPU seconds**;
the controller used **1.23 CPU seconds**, disjoint from worker CPU. UTC elapsed
was 125.25 seconds versus 114.98 monotonic seconds; the previously observed clock
discrepancy remains explicit. Reporting and assistant/subagent usage are not
experiment-model calls and are not included in these worker measurements.

World and assessment totals did not change. These 48 diagnostic replay attempts
are a separate cost category, not additional simulated episodes or passive
prediction assessments. All prior v2/v3 frozen sources and data remain preserved.
The raw replay inputs and seed pool stay private; compact outputs contain no
seed values, laws or hidden maps.

Reproduce the saved-data report and figure, without executing a candidate:

```bash
.venv/bin/python scripts/v3_diagnostic20_report.py
```

The one-shot execution driver refuses a second run when its start record exists
and refuses to start after the recorded deadline. It is not a continuation
instruction. See the [protocol and evidence](../artifacts/campaign-v3/development-diagnostic20)
and [summary](../artifacts/campaign-v3/development-diagnostic20/summary.json).
