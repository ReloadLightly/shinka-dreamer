# Original predictive seed: assessment before evolution

This is the preserved v1 seed study. These results concern the original handwritten program; the evolved agent is analyzed in the [campaign findings](campaign-findings.md). The old assessment is not a fresh test of the evolved agent.



The seed was fixed before validation and final assessment. Development used 64
paired mazes; validation used a separate 64; assessment used 256 fresh mazes
reserved after mutation attempts stopped. There were **2,240 condition-episodes**
across these comparisons, excluding integration checks and replays. This is one
handwritten seed experiment, not evidence of evolutionary improvement.

| Agent / intervention | Escapes, /256 | Escape rate [95% interval] | Death rate | Steps per escape | Near-cell Brier ↓ |
|---|---:|---:|---:|---:|---:|
| Reactive local movement | 0 | 0.0% [0.0, 1.5] | 78.5% | — | 0.25000¹ |
| Memory + paths | 231 | 90.2% [86.0, 93.3] | 9.8% | 61.8 | 0.01167² |
| Learned prediction + planning | 214 | 83.6% [78.6, 87.6] | 16.4% | 55.7 | 0.00737 |
| Freeze predictive updates | 213 | 83.2% [78.1, 87.3] | 16.8% | 50.7 | 0.00696 |
| Learn, use fixed-risk planning | 231 | 90.2% [86.0, 93.3] | 9.8% | 61.8 | 0.00767 |

¹ Missing forecasts receive the declared 0.5 default, hence 0.25 loss.
² The memory control reports the evaluator's persistence forecast.
Only the reactive agent timed out (21.5%); all other episodes ended in death or
escape. Brier losses in this table are on each policy's own trajectory, so their
comparison alone does not isolate learning. Successful-escape times are conditional
on success and should not be interpreted as an unconditional efficiency advantage.

![Withheld outcomes and prediction errors](../artifacts/assessment/outcomes.png)

Using learned predictions in planning reduced escape by **6.6 percentage points**
versus the competent baseline / no-prediction-planning control (paired 95% bootstrap
interval **−12.5 to −0.8 points**). Learning versus frozen updates changed escape
by **+0.4 points [−5.5, +6.3]**: no demonstrated benefit. Freezing leaves localization,
map integration, inventory and ordinary memory functioning.

On **identical baseline actions and observations**, learned forecasts had near-cell
Brier **0.007673**, versus **0.007635** for frozen priors: a small deterioration of
**0.0000375 [0.0000102, 0.0000625]**. Both beat persistence (**0.011669**) and the
fixed 0.02 base forecast (**0.008867**). Thus beating persistence is attributable
to the predictive representation/prior here; it does not demonstrate useful online
learning. The matched threat-conditioned losses were 0.026867 learned, 0.026692
frozen and 0.040866 persistence. Uniform-audit loss was 0.017113 learned versus
0.017103 frozen; the full assessment components are retained.

![Matched-experience forecast loss by episode step](../artifacts/assessment/learning.png)

The predictive seed made **356.7 observation-based parameter updates per episode**
on average; the fixed-planning learner made 400.0. Later curve bins contain fewer,
longer surviving episodes. They do not establish an unconditional learning trend.

Development escapes were 56/64 predictive versus 57/64 baseline. Validation was
41/64 versus 58/64. We retained that regression and did not tune on validation or
assessment. Escape intervals use Wilson's method; difference intervals use 5,000
paired episode-bootstrap samples. These intervals describe case variability, not
repeatability across evolutionary campaigns.

Compact evidence: [assessment](../artifacts/assessment/summary.json),
[paired episode rows](../artifacts/assessment/episodes.csv),
[development](../artifacts/development/summary.json),
[validation](../artifacts/validation/summary.json).
Task, model, coverage, reconstruction, keys, door, timing and ablation metrics remain
separate. The scalar selection objective is versioned `task06-forecast04-v1`:
`.6 * task + .4 * (1 − mean(near Brier, audit Brier))`; it replaces the draft's
post-observation reconstruction term. High model scores mostly reflect empty cells.
