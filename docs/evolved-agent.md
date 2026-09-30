# Inside the evolved agent

ShinkaDreamer's selected program combines a learned enemy-motion model with a planner that uses its predictions. Its most informative result is that **learning reduces prediction error under identical experience**. Whether that improvement reliably produces more escapes remains an open question.

This document explains the [generation-14 program](../artifacts/campaign-v2/completed-50/selected.py), its [ancestors](../artifacts/campaign-v2/completed-50/lineage-programs), and the [mechanism experiment](../artifacts/campaign-v2/mechanism-gen14/audit-summary.json). All reported results here concern the 64 development mazes reused during search.

## Two timescales of adaptation

[Namazu's original proposal](namazu-proposal.md) supplies the central idea: evolve both `world_model_step(memory, local_obs, last_action)` and `planner(memory, local_obs)` in a partially observed, changing maze. The representation and helper functions can evolve alongside those entry points. Selection acts on executable programs; within an episode, the selected program updates its own memory and predictive parameters.

The [implemented extension](design.md) makes predictive learning an explicit, testable requirement. The original proposal scored reconstruction of the current world. The implemented model objective scores enemy-occupancy forecasts made before the next transition. Reconstruction remains a separate diagnostic. This change goes beyond repairs to coordinates, door mechanics and environment generation: it asks whether experience improves predictions that can inform actions.

Episode memory resets between mazes. The evolved source code carries across episodes, but its learned weights do not. This is currently a symbolic and statistical world model, with short lookahead, rather than a neural latent model trained through long imagined trajectories.

## From observations to a predictive state

`world_model_step` first localizes the agent using observed displacement, then integrates the 5×5 view into an agent-relative terrain map. It records when each cell was seen, visit counts, keys and door state. Terrain and enemies remain separate, so a passing enemy does not erase a wall or key from memory.

Observed enemy occupancy is certain at the current instant. Outside the view, the program carries forward its preceding occupancy forecast, blending it slightly toward a small background prior. It limits total hidden occupancy mass using the known number of enemies. `_infer_flow` estimates incoming displacement from the preceding transition probabilities and current sightings. This is anonymous state inference: the agent receives no enemy identities.

These map and state-estimation operations still run when predictive learning is frozen. Consequently, the frozen comparison retains localization, memory and filtering rather than disabling the agent's entire internal state.

## A spatial transition model

`_transition_rows` considers staying and the eight neighboring displacements that are legal under remembered terrain. For a source cell $e$, destination $q$, and assumed agent position $c$, it computes a ten-component feature vector φ. Features describe staying, diagonal movement, approach toward the agent, nearby threats, corridor geometry, destination connectivity and alignment with inferred motion.

The transition probabilities are a spatial softmax:

$$
P_\theta(q\mid e,c)=
\frac{\exp(\theta^\top\phi(e,q,c))}
{\sum_{r\in L(e)}\exp(\theta^\top\phi(e,r,c))},
$$

where $L(e)$ contains the permitted destinations. Ten learned weights therefore adjust a distribution over movement, rather than independently assigning unrelated risk scores to cells.

`_predict` transports the current occupancy mass $b(e)$ through those transitions. It combines contributions using an anonymous union approximation:

$$
\widehat p(q)=1-\prod_e\left[1-b(e)P_\theta(q\mid e,c)\right].
$$

This represents the chance that at least one source occupies the destination. It does not require matching enemy identities between observations. The approximation is imperfect: alternative hidden source locations are not represented as a complete joint distribution, and overlapping enemies are observed only as occupancy.

## Learning from prediction errors

`_record_forecast` saves the forecast and transition rows. After the next observation, `_learn` obtains occupancy labels for cells in the previous agent position's 3×3 neighborhood, excluding unsuitable terrain. The next 5×5 view covers this neighborhood after a legal one-cell move. The labels arrive after the prediction being assessed.

For each eligible cell, the loss is Brier error, $\ell=(\widehat p-y)^2$. The learner differentiates the union forecast through the saved softmax rows for previously visible sources. This is a local update, not differentiation through the entire history of hidden-state inference.

The implementation adds a small penalty toward the initial weights, clips each gradient component to ±3, and applies an AdaGrad update:

$$
G_i\leftarrow G_i+g_i^2,\qquad
\theta_i\leftarrow\operatorname{clip}_{[-6,6]}
\left(\theta_i-\frac{0.20g_i}{\sqrt{0.20+G_i}}\right).
$$

Here $g_i$ includes the regularization term $0.003(\theta_i-\theta_{i,0})$. Updates occur when the observed transitions provide a nonzero predictive gradient. The selected agent averages **351.3 labeled-cell updates and 14.0 parameter-update steps per episode**. These counts describe different things: many observed labels need not change a parameter.

## Turning forecasts into actions

`_paths` computes risk-weighted shortest paths through known terrain. `_choose_target` prioritizes visible keys and the exit, uses unexplored view coverage to choose frontiers, and revisits stale views when remembered connectivity provides no useful goal. Diagonal and cardinal moves both consume one environment step.

`planner` then compares legal first moves and possible second moves. Immediate predicted danger receives a large penalty. `_continuation_risk` preserves each source's alternative destinations instead of treating the first-step occupancy field as independent sources on the second step.

If a source's first-step branch assigns mass $b_e(q)$ to the agent's first destination $q$, surviving there rules that location out. The remaining alternatives are normalized by $1-b_e(q)$. The planner evaluates second-step hazards under this conditional branch, including danger from entering an already occupied cell and from subsequent enemy movement.

This is approximate, heuristic planning. Entry hazard is discounted by 0.65 because another observation will intervene; numerical penalties are evolved constants rather than a calibrated expected-return objective. First-step planning forecasts use the current agent position, while the final exported forecast is recomputed around the chosen destination. Thus the forecast that is scored and the forecast used while comparing actions are not perfectly identical.

## What changed along the lineage

The selected parent chain is **0 → 2 → 5 → 14**. The [published generation metrics](../artifacts/campaign-v2/completed-50/generation-metrics.json) and saved programs make the changes inspectable.

| Program | Representation and planning changes | Escapes |
|---|---|---:|
| [Seed](../artifacts/campaign-v2/completed-50/lineage-programs/gen_0.py) | Categorical occupancy-rate estimates and risk-weighted pathfinding | 56/64 |
| [Generation 2](../artifacts/campaign-v2/completed-50/lineage-programs/gen_2.py) | Learned mixture of stationary, cardinal and diagonal motion fields; revised door handling and waiting | 61/64 |
| [Generation 5](../artifacts/campaign-v2/completed-50/lineage-programs/gen_5.py) | Spatial softmax learner, hidden occupancy propagation, inferred motion, two-step planning and revised exploration | 62/64 |
| [Generation 14](../artifacts/campaign-v2/completed-50/lineage-programs/gen_14.py) | Preserved source alternatives and survival-conditioned continuation hazards | 62/64 |

Most task improvement preceded generation 14. Its score exceeds generation 5 by only **0.00002036**, with unchanged escape count. The softmax learner already exists in generation 5. Earlier changes modify model and planner together, so these comparisons cannot assign task gains to individual innovations.

## What the intervention establishes

The [audit](../scripts/audit_candidate.py) compares learned and frozen models while both use the same fixed-risk planner. [Saved trace hashes](../artifacts/campaign-v2/mechanism-gen14/trajectory-audit.json) match for actions, worlds, enemy trajectories, maps and localization in all 64 pairs. Learned parameters change in 61 episodes; frozen parameters never change.

On this identical experience, near-cell Brier loss falls from **0.010320 to 0.009960**, a **3.5% reduction**. The paired bootstrap interval for learned minus frozen is **[−0.000531, −0.000199]**. Threat-conditioned error also improves. This is evidence that the update rule improves prediction on these cases.

With predictive planning enabled, the learned agent escapes 62 mazes versus 60 with frozen parameters and 61 with fixed-risk planning. Those small differences do not establish a reliable escape benefit. Nor does this development comparison demonstrate performance on unseen mazes: the selected program was chosen using the same cases.

## Reading one episode

The [representative replay](../artifacts/campaign-v2/mechanism-gen14/replay/replay.gif) links hidden world, internal map and forecast. Its [trace](../artifacts/campaign-v2/mechanism-gen14/replay/replay.json) shows the first key collected at step **13**, the second at **39**, and wall changes at **25** and **50**. By step 50, the learner has made **19 parameter-update steps**. The door opens at **58**, followed by escape on action **59**. Frames show pre-action states; episode metadata records the final escape.

The [environment](../dreamer/world.py) gives enemies independent random displacement draws. They do not pursue the agent or retain momentum. The model's motion features should therefore be understood as hypotheses whose weights can adapt, not evidence that it discovered purposeful enemies. Likewise, the agent refreshes its terrain map but does **not** learn wall-transition dynamics. Extending useful prediction from local enemy occupancy to changing geometry, and establishing that those predictions improve decisions on fresh mazes, remain the central research opportunities.
