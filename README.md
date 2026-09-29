# ShinkaDreamer

**Evolving agents that build predictive world models and use them to plan in partially observed, changing mazes.**

Status: research design and implementation handoff, 29 September 2026. The repository does not contain a working simulator or trained/evolved agent yet. No evolution run or benchmark result is claimed.

The starting proposal was supplied by Roland Löchli and attributed by him to Sakana AI's Namazu model. Its full text and code are preserved in [docs/namazu-proposal.md](docs/namazu-proposal.md). This project is independent; that provenance does not imply endorsement by Sakana AI.

## Research question

Can evolutionary program search discover agents whose learned predictions about a changing world improve their decisions in previously unseen instances?

Namazu proposes a 15×15 maze with 5×5 local observations, moving enemies, changing walls, two keys, a locked door, and an exit. ShinkaEvolve evolves both `world_model_step(memory, local_obs, last_action)` and `planner(memory, local_obs)` together. The representations, learning rules, helper functions, and planning algorithms remain open to program evolution.

Two timescales matter:

| Timescale | What changes | Evidence required |
|---|---|---|
| Within an episode | Beliefs and predictive model estimates change using the agent's observations and action outcomes | Predictions recorded before outcomes; learning/update ablations |
| Across generations | ShinkaEvolve selects and changes model-and-planner programs | Candidate lineage, evaluated descendants, development results and withheld assessment |

The seed in the pasted proposal only accumulates observations. Adding learned transition prediction and planning through predicted outcomes is a proposed extension that makes the stronger world-model claim testable. It is separate from correcting bugs in the supplied code. See [the design](docs/design.md) and [the audit](docs/audit.md).

The name ShinkaDreamer does not mean this is an implementation of DreamerV3. ShinkaEvolve performs LLM-guided evolutionary program search; classical numeric genetic algorithms, neural world models, and gradient-trained actors are not implemented by naming the project.

## Start in WSL / VS Code

Run these commands in your WSL terminal for a fresh clone:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/ReloadLightly/shinka-dreamer.git
cd shinka-dreamer
code -n .
```

If you already have a local clone, open that folder and inspect `git status` before updating it. The public repository already exists; no repository-creation script is needed.

In Codex, use:

> Read AGENTS.md, CODEX_TASK.md, docs/design.md, docs/audit.md, and the complete original proposal. Carry out CODEX_TASK.md in this repository. Preserve the dynamic-maze problem and substantive program evolution, implement the learning agent, and report measured results accurately.

The main experiment is CPU-first. A symbolic or small statistical predictive model does not inherently require a GPU. Actual evaluation concurrency and episode throughput must be measured on the local WSL host. The proposed 100-generation campaign should be resumable; it is not a claim that 100 candidates will finish within a particular subscription allowance or runtime.

## Evidence to publish as work proceeds

- Escape rate, deaths, key/door completion, and time to escape, with uncertainty.
- Prediction quality on evaluator-selected targets and at specified horizons.
- Comparisons with a competent reactive agent and memory/pathfinding agent.
- Effects of disabling predictive updates and predictive planning.
- Learning curves, representative true-world/belief/forecast replays, candidate lineage and substantive program changes.
- Unseen-instance results separately from selection/development scores.

No fitness increase alone proves understanding, general intelligence, or recursive self-improvement. Negative findings remain publishable project results.

## References

- [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)
- [ShinkaEvolve agentic usage](https://sakanaai.github.io/ShinkaEvolve/agentic_usage/)
- [DreamerV3, for terminology and comparison](https://danijar.com/project/dreamerv3/)

The original proposal's references to specific Neuroevolution book chapters have not been independently audited in this handoff and are not source-fidelity claims.
