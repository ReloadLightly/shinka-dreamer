# RUN1 saved development results

One selected development panel, repeatedly exposed to evolutionary selection; no held-out assessment or independent-search replication.

Current export: **3 persisted slots**, including the seed; 96 saved condition-episodes. 3 administrative seed-copy rows are excluded from slot counts.

| Slot | Native valid | Recorded / expected | Mean task | Escape | Death | Timeout | Invalid | Missing |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 0 | True | 48/48 | 0.86111 | 41 | 7 | 0 | 0 | 0 |
| 1 | False | 0/48 | Incomplete | 0 | 0 | 0 | 0 | 48 |
| 2 | True | 48/48 | 0.81093 | 39 | 9 | 0 | 0 | 0 |

Native eligibility additionally requires every episode to validate. Failed candidates remain in the slot and evidence records; their episode means retain invalid executions with score zero. Missing episodes are unavailable, not silently counted as successes or observed failures.

Slot1 infrastructure finding: Original native local scheduler uses job.start_time, which this job records as proposal start. A timeout before evaluation can complete is infrastructure evidence, not an empirical failure of the candidate algorithm.

The seed remains the best eligible development program. The paired export is therefore an identity comparison, not evidence about a new program.

On-policy forecast losses; actions and target exposure differ between programs. This is not matched-experience prediction-learning evidence.

No selected-program freeze or prediction-use intervention was assessed in this RUN1 campaign. Predictive-adaptation control benefit remains untested.

1000 percentile bootstrap draws, fixed seed4137, resampling all8 layout cases with all6 conditions and both programs kept together. Nominal descriptive uncertainty; it does not remove winner-selection bias.

Candidate and evaluator episode-process CPU sums are disjoint; native host evaluation elapsed and summed episode wall time overlap and must not be added. Uncheckpointed work can be unavailable.

Program changes are reported from source hashes and structural AST comparisons. Named-call reachability flags possible unused helpers but does not prove inactivity or useful learning. Source-specific causal analysis remains separate.

Reproduce this saved-data export without executing candidates, worlds or model calls:

```bash
.venv/bin/python scripts/run1_science_report.py
```

Compact data and the source-analysis record are under `artifacts/campaign-v4/run1/science`; the separate native export preserves all source branches and actual search mechanisms.
