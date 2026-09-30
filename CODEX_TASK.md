# Current research state and next scientific task

Read [AGENTS.md](AGENTS.md), [the findings](docs/campaign-findings.md),
[the evolved-agent walkthrough](docs/evolved-agent.md) and
[the protocol](docs/protocol.md) before further implementation. Preserve
[Namazu's proposal](docs/namazu-proposal.md) unchanged. The
[initial implementation task](docs/initial-implementation-task.md) is historical.

## Completed experiment

The authorized campaign is complete at **50 total slots, 0–49**: one seed,
45 valid descendants and four failed descendants. Generation 14 is selected;
its source SHA-256 is
`588eeb7c10b978fe86c7e7b572f177e8b755c9c25699f3c75bfadd41192ec5e0`.
Do not extend this completed campaign because an old manifest or archived task
mentions 100 generations.

Development evidence establishes improved selected-program performance and
better matched-trajectory prediction from online learning. It does not establish
an escape benefit caused by learning or generalization to fresh cases.
The repository contains a completed 2 × 2 intervention audit and a saved-data
research analysis. Use those results before proposing new execution.

## Next scientific question

Test whether the selected agent's learned predictions improve decisions on
fresh mazes. At the user-authorized assessment point, freeze the candidate,
evaluator, objective, baseline and interventions before opening fresh cases.
Use paired cases for the selected agent, frozen predictive weights, fixed-risk
planning, the joint intervention and original competent memory baseline.
Report paired escape differences as the primary outcome, then forecast losses,
intervention validity and interpretable failure cases. Preserve a separate
development record; do not select or revise the agent on final-assessment results.

The current instruction leaves the evolved agent's final held-out pool
**unreserved and unevaluated**. Do not allocate or evaluate it as part of
documentation work. New search campaigns or dynamics-shift studies are separate
experiments and require a concrete scientific hypothesis and user direction.

## Working rules

- Keep the full dynamic maze and joint evolution of world-model updating and
  planning. Do not replace program evolution with a fixed algorithm catalog.
- Keep code and raw experimental evidence intact when improving presentation.
  Derive figures from committed data; distinguish inference from measured results.
- Put mechanisms, experiments and findings in the README. Put execution history,
  billing and recovery details in supporting documentation.
- No paid model API calls or silent billing fallback are authorized.
- Work autonomously on ordinary reversible analysis, implementation and fixes.
  Commit coherent changes and push completed work to the verified
  `ReloadLightly/shinka-dreamer` origin. Do not force-push.
- For any future authorized run, report useful checkpoints while it proceeds:
  best-program changes, component metrics, mechanism changes, remaining uncertainty
  and a readable evolution curve. The user should not have to extract manual logs
  to understand what an experiment is discovering.
