# Upstream integration notes

Checked 29 September 2026. ShinkaEvolve HEAD observed at `9912af12d423504b8d580f4179fd15f5f88b8c50`; record and inspect the actual installed version during implementation.

## Evaluation contract

[wrap_eval.py](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/core/wrap_eval.py) calls `experiment_fn(**run_kwargs)` and `aggregate_metrics_fn(all_run_results)`. The original proposal mismatches both signatures. Sequential runs pass Python objects directly; parallel runs use a process executor and require pickle-compatible data.

[scheduler.py](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/launch/scheduler.py) invokes the evaluator with named `--program_path` and `--results_dir` options. Use argparse. Native CLI task invocation is described in the [official guide](https://sakanaai.github.io/ShinkaEvolve/cli_usage/).

These interface corrections do not resolve hidden-state exposure. The trusted evaluator must still drive the episode and compute its own metrics. Validate finite values, completeness, and action legality rather than accepting candidate-reported success.

## Subscription route and auxiliary calls

Native `headless/codex` support exists in the checked [provider](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/llm/providers/headless.py). The [Headless billing documentation](https://github.com/RobertTLange/headless-cli#billing-and-subscription-fallback) describes `auto` billing and fallback behavior. Explicitly require:

```bash
export HEADLESS_BILLING=subscription
```

That setting is necessary for the Headless route; it is not a blanket control over separate API clients or embeddings. Audit all model roles and environment/config resolution. No paid fallback is authorized.

The Shinka default embedding model is `text-embedding-3-small`, which is a separate API route. Select an available authorized local embedding backend or disable embeddings explicitly and report embedding-based novelty as inactive. Do not treat that optional limitation as a reason to leave the entire experiment unfinished.

Native meta-recommendations require both an enabled `meta_rec_interval` and a nonempty `meta_llm_models`. The meta client can use Headless, but it is separate from the mutation-model bandit. Enable `use_text_feedback=True`; send feedback as a string and inspect an actual mutation prompt to confirm it reaches the proposer.

At this revision, the Headless effort parser accepts `low`, `medium`, `high`, and `xhigh`; do not invent an `ultra` or `max` query value. Check the available authenticated model and supported effort on the host. A single model provides no between-model bandit experiment. Headless calls are guarded by a process lock, so four evolutionary islands do not imply four parallel subscription mutation calls.

`max_api_costs=0` is not a substitute for correct routing and may prevent useful execution depending on the runner. Disable paid routes explicitly and label subscription usage/estimates accurately.

Reference: [configuration](https://sakanaai.github.io/ShinkaEvolve/configuration/), [async runner](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/core/async_runner.py).
