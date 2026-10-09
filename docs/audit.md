# Audit of the supplied Namazu proposal

Reviewed in full on 29 September 2026. Locations below refer to the original pasted text as numbered from its first line. The text is preserved as docs/namazu-proposal.md.

## Feasibility and claim boundaries

The project is feasible as CPU-based evolutionary search over agents with symbolic or statistical internal models. Both model updating and planning can remain freely evolvable. A stored map is an internal state representation, but the supplied code has no learned transition predictor or imagination-based planning. Better map reconstruction alone would not demonstrate those properties.

ShinkaEvolve supplies evolutionary program search with LLM-generated mutations. That is a defensible evolutionary/world-model combination. It is not automatically a classical numeric genetic algorithm, a neural Dreamer implementation, a multiagent learning system, or recursive self-improvement.

## Necessary code repairs

| Problem | Source location | Required repair |
|---|---|---|
| Relative agent map is compared directly to absolute world coordinates | Lines 57–83 and 449–458 | Define the frame; evaluator privately translates relative coordinates using the initial origin |
| New observations are mapped before applying the preceding movement; attempted movement is treated as successful | 76–101, 344–345 | Apply observable movement outcome before observation integration; distinguish blocked motion |
| Full hidden environment is passed into candidate code; candidate returns the reported metrics | 197–203, 467–479 | Evaluator owns environment, transitions, auditing and scores; candidate receives observation/action-result data only |
| Arbitrary displacement values can bypass the intended action space | 118–120, 341–345 | Validate integer displacements and interaction values before executing actions; define diagonal/corner rules |
| Candidate chooses which map cells contribute to accuracy | 449–460 | Evaluator chooses targets; missing predictions receive a defined score; report coverage separately |
| Audit after observing measures reconstruction, not prediction | 399–414 | This is intentional in the original proposal. Future forecasting is a separate extension, not a necessary repair |
| Door does not physically block passage, and is unrelated to exit geometry | 287–294, 344 | Implement the prose: locked door genuinely gates the exit and keys permit opening it |
| Dynamic-wall selection includes borders; walls may close on occupants | 271–278, 312–317 | Fixed borders and explicit occupant-closure semantics, without removing dynamic changes |
| Random placement can create infeasible instances | 257–304 | Generate valid key→door→exit tasks under documented dynamic mechanics; distinguish feasibility from guaranteed survival |
| Repeated five seeds provide no independent generalization evidence | 467–469, 521 | Separate development, validation and withheld assessment; controlled RNGs |
| Reward grows with elapsed steps | 430–440 | Clarify survival versus efficient escape; expose component metrics and make the primary objective consistent |
| Seed never updates key/door memory; enemy IDs are position labels | 57–60, 88–94, 145–150 | Use observable outcomes and anonymous occupancy/risk unless observations actually expose identity |

Additional checks: reset all episode state before placement; do not use stale agent/enemy positions from a previous build; stream aggregate audit statistics rather than deep-copying every growing memory unless a replay was requested. These affect correctness and memory usage.

## Verified ShinkaEvolve integration errors

Upstream HEAD was checked on 29 September 2026 at `9912af12d423504b8d580f4179fd15f5f88b8c50`. Recheck the installed revision when implementing.

- `get_experiment_kwargs()` returns `maze_world` and `seed`, but `run_evaluation(env_config)` accepts a different argument. The wrapper calls the candidate as `experiment_fn(**run_kwargs)`. As written, this raises an unexpected-keyword error.
- `aggregate_metrics(results, results_dir)` requires two arguments, while upstream calls the aggregation callback with the result list alone.
- The scheduler supplies `--program_path PATH --results_dir DIR`. The draft's two positional `sys.argv` reads do not parse that invocation.
- The shown `shinka_run --task-dir ... --set ...` syntax is supported at the checked revision. The evaluator callback mismatches, not the mere existence of that command, are the immediate integration defects.
- Passing a Python maze object is not inherently a JSON-serialization error: the sequential wrapper passes objects directly and the parallel route uses pickle. The hidden-state exposure remains a scientific validity problem regardless of transport.

See [docs/upstream.md](upstream.md) for source links and subscription routing details.

## Stronger world-model evidence: proposed extension

Preserve the two entry functions and allow helpers inside the evolve block. Add a documented forecast output interface, while leaving internal representation and learning unrestricted. An empirical transition estimator is one reasonable CPU seed, not a restriction on descendants.

Collect predictions of action outcomes and future dynamics before the outcome is revealed. Test whether estimates improve with experience and whether the planner uses them beneficially. Use fixed target sets, a probabilistic scoring rule, a persistence baseline, and planning/update ablations.

Independent environment seeds are necessary but not sufficient. Report withheld layouts and, separately, deliberate dynamics shifts. Repeatedly selecting on the same held-out results would turn them into development data.

Do not promise generation-specific discoveries, superiority to baselines, or 'understanding' from a 0.6/0.4 weighted fitness. Those are empirical questions.
