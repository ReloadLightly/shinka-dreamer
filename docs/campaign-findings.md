# What the first evolutionary campaign established

ShinkaDreamer’s first completed campaign produced an agent with a measurable capacity to improve its enemy forecasts from experience. Evolution also increased escape success on the development mazes. These are distinct findings: the experiment has not yet established that learning those forecasts causes a reliable improvement in escape success on unfamiliar mazes.

The evidence is a single evolutionary campaign evaluated repeatedly on 64 development mazes, followed by controlled interventions on its selected program. The analysis below uses the published episode records. The fresh final assessment remains untouched.

## Evolution improved control, with an early plateau

The campaign contains 50 candidate slots: the seed and 49 generated descendants, of which 45 were valid. The selected lineage is **0 → 2 → 5 → 14**. Generation numbers identify candidate slots, not independent replications of the experiment.

| Ancestor | Escape successes | Mean task score | Mean model score | Selection score |
|---|---:|---:|---:|---:|
| Seed, generation 0 | 56/64 | 0.878910 | 0.988651 | 0.922806 |
| Generation 2 | 61/64 | 0.946730 | 0.988171 | 0.963307 |
| Generation 5 | 62/64 | 0.958055 | 0.987991 | 0.970029 |
| Selected, generation 14 | 62/64 | 0.958121 | 0.987943 | 0.970050 |

The selection objective is `0.6 × task + 0.4 × model`. From seed to selected program, its increase decomposes as:

| Contribution | Calculation | Change in selection score |
|---|---|---:|
| Task outcome and efficiency | `0.6 × (0.958121 − 0.878910)` | +0.047527 |
| Forecast component | `0.4 × (0.987943 − 0.988651)` | −0.000283 |
| Total | Sum of the two contributions | **+0.047243** |

The task improvement accounts for the gain; the forecast component slightly offsets it. The nominal 40% model weight does not imply that forecasting drove 40% of the observed progress. Model scores occupy a narrow, high range because enemy occupancy is sparse and many forecast targets are empty.

Generation 5 had already achieved **99.96% of the final best-score improvement**. Generation 14 adds only 0.00002036 to its score. The two agents have the same escape/death outcome in every paired maze. Eleven episodes change length; total steps decrease from 3,184 to 3,167, with unchanged key outcomes. The subsequent 35 slots, containing 31 valid descendants, produce no new best.

This is evidence of rapid initial progress followed by a plateau on this objective and development set. It gives no basis for assuming that another fixed block of generations would necessarily produce a breakthrough.

Sources: [generation metrics](../artifacts/campaign-v2/completed-50/generation-metrics.json), [episode records](../artifacts/campaign-v2/completed-50/episodes.csv), [lineage and source programs](../artifacts/campaign-v2/completed-50/lineage-programs/).

## Better escape does not automatically mean better prediction

The selected program escapes 62/64 development mazes, compared with 57/64 for the original memory/pathfinding baseline and 56/64 for the original predictive seed. Its paired escape advantage over memory is **+7.8 percentage points**, with a 95% bootstrap interval of **[−1.6, +17.2]**. Against the seed, it is **+9.4 points [0.0, +18.8]**.

Meanwhile, near-cell Brier loss increases from **0.006989** in the seed to **0.009565** in the selected agent; lower is better. The paired difference is **+0.002576 [0.001045, 0.004032]**. This does not establish that evolution damaged the underlying predictor: changing actions changes which situations the agent encounters, how long it survives, and which observations it receives. These are losses along each agent’s own trajectories, not a comparison on identical experience.

The meaningful conclusion is that the campaign improved the joint behavior while leaving a prediction–control tradeoff to explain. A planner can make better decisions without minimizing average occupancy error. Conversely, a predictor can reduce average error without changing the few decisions that determine survival. Both measurements are necessary.

Reported Brier losses pool target errors across episodes. The model component used for selection first computes a score within each episode and then averages episodes. These different weightings are intentional; the displayed pooled losses should not be substituted into the selection formula to reconstruct fitness.

Sources: [campaign comparison](../artifacts/campaign-v2/completed-50/summary.json), [original control episodes](../artifacts/campaign-v2/controls/episodes.csv).

## A controlled test isolates predictive learning

The selected program updates ten enemy-transition weights within an episode. The frozen intervention disables those parameter updates while retaining localization, map updates, and state estimation. A second intervention replaces forecast-dependent risk planning with fixed-risk planning. Applying both interventions produces four versions of the same program:

| Version | Predictive parameter updates | Forecast-dependent planning | Escapes |
|---|---|---|---:|
| Selected agent | Yes | Yes | 62/64 |
| Frozen learner | No | Yes | 60/64 |
| Fixed-risk planner | Yes | No | 61/64 |
| Frozen + fixed-risk | No | No | 61/64 |

The fixed-risk pair provides the clean prediction comparison. Saved hashes match the complete world/action/enemy trajectory and the map/localization history in **all 64 paired episodes**. The forecasts therefore face the same experience. Every episode begins with the same initial weight vector; learned weights change in 61/64 episodes, and frozen weights remain constant in all 64.

| Matched prediction measure | Learned | Frozen | Learned minus frozen, 95% paired interval |
|---|---:|---:|---:|
| Near-cell Brier loss | 0.009960 | 0.010320 | **−0.000360 [−0.000531, −0.000199]** |
| Threat-conditioned Brier loss | 0.028139 | 0.029156 | **−0.001017 [−0.001501, −0.000569]** |

Learning reduces aggregate near-cell error by **3.49%** relative to frozen weights. Per-episode near-cell error improves in 39 mazes, ties in four, and worsens in 21. This supports a predictive benefit from adaptation on matched development experience, with substantial variation between episodes.

The two Brier rows are related views of the same effect. Their total error sums are identical here; threat conditioning reduces the denominator from 28,782 near-cell targets to 10,188 targets during nearby-enemy encounters. The threat result locates the improvement in relevant encounters but is not an independent replication.

The remaining causal question concerns action quality. With forecast planning active, learning yields two extra escapes: **+3.1 points [0.0, +7.8]**. With learning active, forecast-dependent rather than fixed-risk planning yields **+1.6 points [−4.7, +7.8]**. The latter wins three paired mazes and loses two. These small samples do not establish a reliable escape benefit from either intervention. The matched fixed-risk versions have identical outcomes by construction, so their prediction improvement cannot itself demonstrate a task advantage.

Sources: [intervention summary](../artifacts/campaign-v2/mechanism-gen14/audit-summary.json), [intervention episodes](../artifacts/campaign-v2/mechanism-gen14/episodes.csv), [trajectory and parameter audit](../artifacts/campaign-v2/mechanism-gen14/trajectory-audit.json).

## What would resolve the central question

The intervals above resample paired episodes, with 5,000 bootstrap samples. They describe uncertainty within the reused development set; they do not correct for selecting the best program after search. Repeated evaluations of these 64 mazes are not thousands of independent test mazes. A single campaign also cannot establish that ShinkaEvolve reliably discovers this behavior across independent searches.

The next decisive experiment is to **freeze generation 14 and the comparison definitions**, then evaluate fresh, paired mazes using the original memory baseline, original predictive seed, selected agent, frozen learner, fixed-risk planner, and frozen-plus-fixed-risk version. Predeclare escape differences against memory and between interventions, alongside matched forecast differences. Verify that fixed-risk trajectories still match on those new cases. Report episode-level outcomes and paired uncertainty before making further changes based on the findings.

That assessment would distinguish three outcomes: generalization of the improved controller, generalization of within-episode predictive learning, and a causal contribution of prediction to successful action. If prediction generalizes while its control benefit remains unresolved, the research target becomes how the planner uses uncertainty and future risk. Independent evolutionary campaigns would then address search repeatability. This is a proposed scientific comparison, not a new run: no additional campaign or held-out evaluation was performed for this report.
