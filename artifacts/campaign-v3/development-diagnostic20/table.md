# Bounded development replay diagnostic

Development-only saved memory-policy experience; diagnostic actions were not executed. No control-performance inference.

Completed passes: 42/48; failed: 6; unstarted: 0. Returned frames: 3306/3368; missing: 62. Complete plain/profiled pairs: 18/24. Shared-prefix main action/export hashes equal: True; comparator known actions equal: True.

Complete profiled comparator passes provide all recorded-prefix action recommendations. Rank/margin diagnostics separately use returned plain-pass frames, including partial prefixes from CPU-limit failures; missing ranks are not imputed.

| Regime | Changed choices / complete profiled frames | Returned plain rank frames | Changed choices / rank frames | Recordings with changed choices | Same / different target | Changed full rank order | Changed choices with both immediate risks zero |
|:--|--:|--:|--:|--:|--:|--:|--:|
| uniform | 2/281 | 265 | 2/265 | 2/4 | 264 / 1 | 19 | 2 |
| stationary | 14/287 | 260 | 11/260 | 4/4 | 248 / 12 | 50 | 7 |
| switch | 4/274 | 255 | 4/255 | 2/4 | 245 / 10 | 39 | 3 |

Known-law minus fitted dynamics at the same fitted hidden-occupancy/map/location/history state. Known-score gap=score(baseline choice)-score(known choice), both under known-law scores. Immediate delta likewise compares both choices under known-law occupancy risk, not calibrated death risk. Navigation targets may change. Rank changes compare ordered eligible moves, not merely changed score values.

| Regime | Known-score gap + / 0 / − | Immediate-risk delta + / 0 / − | Mean known-score gap |
|:--|--:|--:|--:|
| uniform | 0 / 265 / 0 | 0 / 265 / 0 | 0.0 |
| stationary | 10 / 250 / 0 | 2 / 256 / 2 | 0.32477726457297573 |
| switch | 4 / 251 / 0 | 1 / 254 / 0 | 0.18635758346540593 |

Only profiled main WM/planner/export calls; known-law clone, copying and recorded-action resets are excluded. Each pass retained only its top15 functions by cumulative time per phase. Aggregated function rows are truncated evidence, not full function self-time totals; nested cumulative times must not be added.

Instrumentation costs, not an uninstrumented speed benchmark. Plain comparator includes ranking-trace overhead; profiled passes include cProfile overhead. Worker and phase CPU overlap and are not added.

| Source | Phase | Calls | Profiled CPU s | Largest retained cumulative-time functions |
|:--|:--|--:|--:|:--|
| selected | world_model_step | 842 | 12.270 | world_model_step (11.149s); _forecast (10.905s); kernel (5.812s); spread (4.779s) |
| selected | planner | 842 | 12.079 | planner (10.921s); _distances (7.552s); _legal (4.204s); _blocked (3.055s) |
| selected | export_model | 842 | 0.181 | export_model (0.141s); <listcomp> (0.060s); <listcomp> (0.055s); _mean (0.012s) |
| comparator | world_model_step | 842 | 7.730 | world_model_step (7.055s); _record_forecast (6.681s); _predict (6.651s); _transition_rows (5.679s) |
| comparator | planner | 842 | 48.639 | planner (43.913s); _continuation_risk (26.678s); _transition_rows (9.995s); _paths (8.608s) |
| comparator | export_model | 842 | 0.226 | export_model (0.189s); <listcomp> (0.110s); <listcomp> (0.052s); <built-in method builtins.min> (0.025s) |

Uses the final logged post-switch-observation frames of each requested prefix (obs.step>=switch). The first forecast governed by the replacement law occurs one observation earlier; this diagnostic does not relabel that boundary. Failed prefixes contribute only returned frames meeting that cutoff.

existing development-only fitting-subvalidation memory-policy recordings; prior/rate tuning exposed; not validation of champion or fresh assessment
Four recordings per regime, capped at80 observations each; dependent frames are not independent samples. No significance tests.
Improving the planner's own score is mechanically expected after recomputing its optimum and does not prove actual safety or escape benefit.
