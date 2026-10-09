# RUN1 saved development results

One selected development panel, repeatedly exposed to evolutionary selection; no held-out assessment or independent-search replication.

Current export: **3 persisted slots**, including the seed; 96 saved condition-episodes. 3 administrative seed-copy rows are excluded from slot counts.

| Slot | Status | Native valid | Recorded / expected | Mean task | Escape | Death | Timeout | Invalid | Missing |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 0 | persisted_complete_panel | True | 48/48 | 0.86111 | 41 | 7 | 0 | 0 | 0 |
| 1 | persisted_no_panel | False | 0/48 | Incomplete | 0 | 0 | 0 | 0 | 48 |
| 2 | persisted_complete_panel | True | 48/48 | 0.81093 | 39 | 9 | 0 | 0 | 0 |
| 3 | held_before_valid_proposal | None | 0/48 | Incomplete | 0 | 0 | 0 | 0 | 48 |

Native eligibility additionally requires every episode to validate. Failed candidates remain in the slot and evidence records; their episode means retain invalid executions with score zero. Missing episodes are unavailable, not silently counted as successes or observed failures.

Slot1 infrastructure finding: Original native local scheduler uses job.start_time, which this job records as proposal start. A timeout before evaluation can complete is infrastructure evidence, not an empirical failure of the candidate algorithm.

Reserved slot3 is held at `before_valid_proposal` with generated source available: False. Its 48 planned episodes were unattempted. Reason: Reported usage unavailable; token budget cannot be verified. It is not a completed candidate slot.

The seed remains the best eligible development program. The paired export is therefore an identity comparison, not evidence about a new program.

The evaluated descendant at slot2 versus the seed has paired task difference -0.05018 (descriptive 95% layout-bootstrap interval -0.21336 to +0.10052) and escape difference -4.17 percentage points (-20.83 to +12.50). This is a development comparison, not a frozen-adaptation causal contrast.

| Condition | Task difference | Escape difference (pp) |
|:--|--:|--:|
| uniform-full | -0.15037 | -12.5 |
| stationary-full | -0.00172 | +0.0 |
| switch-full | -0.00175 | +0.0 |
| stationary-late | -0.24466 | -25.0 |
| repeat25-full | +0.21988 | +25.0 |
| repeat25-late | -0.12247 | -12.5 |

The condition-specific intervals and pooled forecast ratios are in `paired-descendants.json`; all six conditions from each layout remain together in resampling. Exposure counts are in `exposure-by-condition.csv`. Visibility and consecutive visibility are opportunity proxies, not identified enemy transitions or measured information gain. Post-switch experience is only available for episodes that survive to a replacement; unconditional task outcomes remain primary.

On-policy forecast losses; actions and target exposure differ between programs. This is not matched-experience prediction-learning evidence.

No selected-program freeze or prediction-use intervention was assessed in this RUN1 campaign. Predictive-adaptation control benefit remains untested.

1000 percentile bootstrap draws, fixed seed4137, resampling all8 layout cases with all6 conditions and both programs kept together. Nominal descriptive uncertainty; it does not remove winner-selection bias. With only8 layouts and sparse discordance, intervals may be unstable or degenerate; a zero-width interval does not establish equivalence.

Candidate and evaluator episode-process CPU sums are disjoint; native host evaluation elapsed and summed episode wall time overlap and must not be added. Uncheckpointed work can be unavailable.

Program changes are reported from source hashes and structural AST comparisons. Named-call reachability flags possible unused helpers but does not prove inactivity or useful learning. Source-specific causal analysis remains separate.

Reproduce this saved-data export without executing candidates, worlds or model calls:

```bash
.venv/bin/python scripts/run1_science_report.py
```

Compact data and the source-analysis record are under `artifacts/campaign-v4/run1/science`; the separate native export preserves all source branches and actual search mechanisms.
