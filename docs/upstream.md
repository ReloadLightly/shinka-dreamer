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

## Executed integration, 29 September 2026

Installed actual upstream `9912af12d423504b8d580f4179fd15f5f88b8c50` (package
version 0.0.7) into the repository-local `.venv`. The older unrelated checkout
under the user's home was inspected but left unchanged. The current repository
origin is `https://github.com/ReloadLightly/shinka-dreamer.git`.

The installed npm Headless 0.6.1 release did **not** implement `HEADLESS_BILLING`.
The official source at `93cd9b06b85f848af1308c41e018991b33907c5e` does, despite
retaining the same package version. It was built locally under `.runtime/headless`.
The launcher refuses the release without billing support; no model call was made
through that old release. `scripts/subscription_headless.sh` requires subscription
billing and removes paid API environment variables in its child process. Source
inspection verified that explicit subscription mode cannot take the paid fallback.
No global authentication files, Codex configuration or permission policy was edited.

Actual native execution evaluated `initial.py` on all 64 development episodes and
persisted four generation-0 rows: one evaluated seed and three native island copies.
There is **one unique evaluated native program, zero evaluated descendants and zero
accepted mutations**. Native parent selection sampled across those islands. Saved
native prompts contain the evaluator's string `text_feedback`; an actual excerpt
and source hash are committed in `artifacts/native/`. No evolution curve is justified
by these data.

The live subscription probe and native mutation attempts failed before inference:

```
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Codex CLI 0.159.0 reports a ChatGPT login, but this session's filesystem boundary
prevents its app-server startup. This is an execution blocker, not evidence that
the account lacks subscription access. DNS/network restrictions also prevent the
optional models.dev price-catalog refresh; upstream uses its bundled pricing snapshot.
Those estimates are not subscription charges. No paid model or embedding route was
used, and the blocked mutation attempts generated no candidate program.

Role audit:

| Role | Resolved behavior |
|---|---|
| Mutation/fix | `headless/codex@gpt-6-astra?effort=high`, subscription required |
| Meta/recommendations | Same route, independent client, interval 10; upstream also flushes at shutdown |
| Final meta during resume check | Attempted through subscription route; same startup failure |
| Novelty LLM | Disabled |
| Embeddings | `None`; native embedding novelty inactive |
| Evaluation | Pure local Python; no LLM |
| Prompt evolution | Disabled |
| Model bandit | Disabled; one configuration is not a bandit experiment |

Four islands, weighted native parent selection, within-island archive/top-k
inspirations, fitness archive, lineage and 10-generation migration remain configured.
Recommendation generation is configured but no successful recommendation exists.
One evaluation/proposal/database worker fits the observed host; the initial native
64-episode evaluation took 22.04 seconds, recorded in `evolution_run.log`.

Integration checks against installed code:

* Scheduler invokes named `--program_path` and `--results_dir`; `--episodes` is
  supplied through native `LocalJobConfig.extra_cmd_args`.
* `run_shinka_eval` loads **trusted** `dreamer/bridge.py` and calls
  `run_evaluation(candidate_path=..., seed=...)`; hidden state never enters a
  candidate wrapper. Aggregation accepts one result list. JSON metrics use
  `combined_score`, `public`, and string `text_feedback`; `correct.json` uses the
  actual `correct`/`error` schema. Native output passed all 64 runs after repairing
  an initially missing output-directory creation in our adapter.
* `max_novelty_attempts` must remain at least 1 even with novelty disabled: upstream
  uses that loop for all proposals. This was corrected before any proposal executed.
* SQLite commits completed programs through native `add_program_async`; island copies
  must not be counted as separate evaluations. No cross-model bandit state is needed.
* Upstream checks `last_iteration > 0` to resume, which misses a seed-only checkpoint.
  `dreamer/native.py` narrowly reuses persisted generation-0 rows and invokes native
  `_restore_resume_progress`. A real run on a SQLite backup reused exactly four rows
  without re-evaluation or duplication. It does not change generation numbers, scores,
  sampling or mutation behavior.
* This host denies `socketpair.send`, preventing asyncio's cross-thread wakeup.
  A 50 ms event-loop timer allows completed futures to be drained without changing
  permissions. This resolved the native file-read stall; no upstream source was edited.

The exact resume command from the repository root is:

```bash
HEADLESS_BILLING=subscription .venv/bin/python scripts/evolve.py \
  --results results/campaign-v1 --generations 100
```

Run it in an authorized normal terminal where the existing Codex login can start.
The driver now performs a fail-fast subscription probe before native resampling.
It does not relocate auth or weaken sandbox settings. For a fresh installation,
`bash scripts/bootstrap.sh` installs both pinned dependencies locally. The first
assessment is now published; reserve fresh final cases after further evolution.
