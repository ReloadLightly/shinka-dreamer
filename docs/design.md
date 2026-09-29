# Proposed implementation design

This design preserves the submitted project and identifies one substantive extension: learned prediction used in planning. It is a specification for the next Codex implementation, not a report of completed experiments.

## Scientific core

The main question is whether joint evolution of model updating and planning produces agents whose experience-derived predictions help them escape unfamiliar changing mazes.

Retain the proposed 15×15 world, 5×5 local observation window, two keys, locked door, exit, three moving enemies, 25-step wall-change interval, and 200-step episode horizon as the source-aligned main regime. Do not call it a faithful implementation until the discrepancies in docs/audit.md are repaired and documented. Additional sizes, intervals and stochastic dynamics belong to explicitly labelled assessment conditions or later extensions.

Retain the declared nine displacement choices including staying still. Define whether diagonal corner cutting is legal; the recommended repair disallows cutting through blocked orthogonal corners. The door must be a real bottleneck, impassable until opened from an adjacent tile with the required keys. Specify the action ordering and collision rule, including moving onto an enemy before that enemy moves. Observation visibility is a local square in the source; do not silently introduce line-of-sight occlusion.

Use fixed boundary walls. Dynamic closures must not put a wall on an occupied cell; document whether the toggle is deferred and how that affects its subsequent schedule. Generation must produce a feasible objective structure rather than an arbitrary exit and passable 'door'. Check route feasibility under the actual dynamic rules, not only static connectedness; this does not guarantee success against enemies.

## Agent and evaluator boundary

The evaluator owns the hidden world, seeds, transitions, reward, forecast targets and all score calculations. Candidate code sees local observations, the previous attempted action and observable outcomes such as actual displacement, collected key and door interaction. This local feedback is an explicit interface correction; it does not reveal the world map, enemy identities, hidden RNG state or future schedules.

The candidate retains these entry points:

```python
world_model_step(memory, local_obs, last_action) -> memory
planner(memory, local_obs) -> action
```

Allow extra helpers, classes, representations and learning rules within the evolve block. A prediction export function or structured forecast fields may be added for assessment. Its contract specifies what can be scored, not how the model must work. Never pass a live maze object into the candidate.

Keep candidate execution isolated from the evaluator process and protected assessment files. Passing a dictionary alone does not sandbox Python: candidate code can inspect files, imports or other state if those are accessible. Implement an appropriate process/container boundary for the actual host, with only the intended observations crossing it. Do not describe process separation alone as an adversarial security guarantee. Keep this task-specific and small.

## Two learning loops

Within the episode, an initial CPU agent can maintain a map, estimates of uncertain occupancy, and empirical transition statistics updated from observed outcomes. It should forecast relevant future events and use forecasts to rank action sequences or paths. Avoid hard-coding the hidden transition law and then calling it learned.

Across generations, native ShinkaEvolve changes the code implementing representation, estimation, uncertainty handling, exploration, forecasting and planning. These are not restricted to a selected algorithm catalog. The LLM's prior knowledge may contribute to evolved code; within-episode learning must therefore be measured separately.

Memory resets at the start of each independent episode in the initial design. Persistent learning across episodes is an additional experiment, with explicit train/test separation, rather than an accidental carryover.

## Measurement

Task outcomes: escape, death, timeout, keys, door opened, and steps to successful escape. Treat efficient escape as the main objective; any survival shaping must not silently prefer indefinite delay.

Reconstruction diagnostics: evaluator-selected positions in the agent-relative frame, coverage, last-seen age, observed versus unobserved cells. Keep static terrain, transient enemies and inventory separate rather than allowing an enemy overlay to erase remembered terrain.

Predictive diagnostics: probabilistic forecasts made before future observations, scored on evaluator-chosen targets and horizons. Publish a proper scoring rule such as Brier score, missing-output handling, and comparison to persistence/base-rate forecasts. Include both spatially relevant targets and a fixed audit sample so easy empty cells do not dominate the metric. Impossible-to-know unseen details should permit calibrated uncertainty rather than reward confident guessing.

The source's 0.6 task + 0.4 model weighting is a provisional hypothesis. Define and normalize each component before search. If the repaired model component uses predictive accuracy rather than reconstruction, version and disclose the change. Keep all component metrics; a scalar is for selection, not a substitute for evidence. Do not tune scoring weights on withheld assessment results.

Comparisons: a reproducible reactive agent, a competent memory/pathfinding agent, and the predictive seed. The primary competent comparison should retain mapping and pathfinding; beating random movement alone is weak evidence. Use no-update and no-prediction-planning ablations that preserve the rest of the agent as far as possible. Publish what the intervention actually removes.

Use common episode seeds for paired candidate comparisons, with agent randomness independent of environment randomness. Keep development, validation and final assessment separate. Make 64 development episodes a starting execution configuration, not a scientific ceiling; measure runtime and increase replication as needed for the uncertainty of the actual comparison. Reserve 256 or more unseen episodes for a useful first assessment, without exposing them to proposals. These counts are proposed defaults and must be recorded with the measured campaign; they are not results or a new mandatory readiness stage.

Report episode-level outcomes and uncertainty on differences. A single evolutionary campaign cannot establish search repeatability; add independent search repeats when making claims about the search method.

## Native ShinkaEvolve and execution

Use official upstream code and record the actual commit/version. Start from the proposal's 100-generation, four-island campaign. Generation/slot count and accepted evaluated descendants are distinct; report both. Preserve parent/inspiration sampling, database, lineage, resumability and meta-recommendations where supported. Resource configuration may adapt to the WSL host without shrinking the candidate algorithm space.

Use the currently supported subscription-backed Headless/Codex route, verified against installed upstream code. Audit every model role for billing; explicitly require subscription billing where the backend supports it and prohibit paid fallback. Embeddings are separate from mutation calls. Native embedding-based novelty requires an available authorized local embedding service or equivalent supported route; otherwise record it as unavailable instead of claiming it is active. Do not start a heavyweight local model blindly on a memory-constrained host.

Use a mutation bandit only over actual supported authorized model configurations. Recommendations use their own client/settings; they are not automatically governed by the mutation bandit. Send model-specific prediction failures and task failures through native text feedback, not only a scalar.

Start with one evaluation worker on the user's potentially constrained WSL host, then use measured memory and evaluation time to set concurrency. The remote LLM is the likely dominant cost in a small gridworld, but runtime must be measured rather than promised. No GPU is inherently required for the initial symbolic/statistical design.

## First substantive deliverable

A running maze, competent agents with a predictive learner, auditable metrics, a working native Shinka adapter, representative replay, and the first actual evolution results when the authorized runtime is available. A setup-only scaffold is not this deliverable. Fix concrete defects, then proceed with the experiment; avoid an additional permanent readiness subsystem.
