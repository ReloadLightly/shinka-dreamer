# ShinkaEvolve: implementation research for the original proposal

**Research date: 9 October 2026.** This guide supports the
[step-by-step execution plan](original-proposal-plan.md). It is a source audit,
not a new experimental result. No provider probes, candidate evaluations or
searches were launched during this planning task.

## 1. Sources, revision and scope

The research used the [official repository](https://github.com/SakanaAI/ShinkaEvolve),
the [final ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/7886b9bafe76c52fd568db10ff9772df-Paper-Conference.pdf),
official [configuration](https://sakanaai.github.io/ShinkaEvolve/configuration/),
[bandit](https://sakanaai.github.io/ShinkaEvolve/bandit_selection/),
[asynchronous evolution](https://sakanaai.github.io/ShinkaEvolve/async_evolution/),
[local-model](https://sakanaai.github.io/ShinkaEvolve/support_local_models/) and
[WebUI](https://sakanaai.github.io/ShinkaEvolve/webui/) documentation, plus the
installed implementations, including their failure and resumption paths.

The installed distribution is ShinkaEvolve **0.0.7**, revision
**`9912af12d423504b8d580f4179fd15f5f88b8c50`**, verified from distribution provenance.
Headless is pinned to **`93cd9b06b85f848af1308c41e018991b33907c5e`**. Public GitHub
metadata identified current upstream main as
**`8adc053a2ce4511ad2ac310e004c530a73fb974a`**, dated 5 October 2026.

Direct comparisons found these **15 relevant files byte-identical** between the
installed version and that current upstream revision:

```text
core/config.py             core/async_runner.py       core/sampler.py
core/async_novelty_judge.py core/prompt_evolver.py     core/summarizer.py
database/parents.py        database/inspirations.py  database/islands.py
database/prompt_dbase.py   llm/prioritization.py     llm/providers/headless.py
embed/client.py            edit/async_apply.py       launch/scheduler.py
```

This is not a claim that the entire repositories are identical. It establishes
that an upgrade is unnecessary for the mechanisms examined here. Local adapter
code remains separately identified and must be included in a run's source freeze.

The paper motivates fitness/diversity-aware sampling, conditional novelty and
adaptive LLM selection. Its comparisons on tasks such as circle packing are
evidence for those experiments, not proof that the same settings improve this
maze. The current implementation also supports prompt coevolution; that code
feature should not be presented as an ablation result from the paper. The bandit
documentation's simulated teaching examples are not our search data.

## 2. What happens during one native generation

The [async runner](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/core/async_runner.py)
reserves a generation ID, samples context and a mutation arm, asks for a program,
applies/repairs the edit, computes an embedding, conditionally checks novelty,
then submits accepted code to the external evaluator. Database insertion updates
the population and credits, and can schedule migration, meta work and prompt
evolution. Proposal workers, evaluation workers and database maintenance are
separate activities. Their counters are not interchangeable.

Generation zero is the evaluated seed. Four islands also receive three
administrative seed copies; these are not three extra evaluations or independent
discoveries. A rejected proposal attempt is a model call and saved source, but
need not be a new generation. A terminal failure uses a reserved generation and
must remain in the total slot count. Report these quantities separately.

## 3. Mechanisms and what we can learn from them

### Population, parent selection and inspiration

The native database selects an island/context and then an eligible parent. In
`WeightedSamplingStrategy.sample_parent`, the actual implementation weights both
fitness and how often a program has already produced offspring. Within the
eligible score population, let `m` be the median and `d` the median absolute
deviation, floored at `1e-6`. The normalized weight is proportional to

```math
\frac{\mathrm{sigmoid}(\lambda(F_i-m)/d)}{1+\mathrm{children}_i}.
```

This **MAD scaling is in the implementation**, although absent from the paper's
displayed expression. An underused parent can retain sampling probability without
having the highest score. This offspring-count term is not embedding novelty or
measured behavioral diversity. See
[parents.py, lines 365–423](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/database/parents.py#L365).

Native archive and top-performing inspirations supply additional code and
evaluation context. They are selected by
[inspirations.py](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/database/inspirations.py),
not manually curated by the supervisor. Preserve actual IDs, eligible context and
scores. Inspect whether a descendant reuses a useful idea; an inspiration link
alone is not evidence of substantive recombination.

### Operators, feedback and recommendations

The [sampler](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/core/sampler.py#L90)
supports diff, full rewrite and crossover. Without inspirations it removes
crossover and renormalizes remaining probabilities. With crossover it selects an
inspiration for an LLM-generated combination. This is code recombination, not a
guarantee that both sources contribute behavior.

With `use_text_feedback=True`, evaluator **string** feedback enters parent and
inspiration context. Native recommendations enter diff/full system prompts, but
are deliberately omitted from the crossover prompt. Preserve that distinction;
do not modify upstream merely to claim recommendation consumption on every
operator. Save real requests so we can verify exactly which feedback was seen.

### Migration

The [island manager](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/database/islands.py#L548)
schedules migration when a persisted generation number is divisible by the
interval; maintenance executes it later. Eligible correct, non-seed programs may
move, while elites are protected. An interval event can therefore move no
programs in a small population. Inspect actual migration history, birth/current
island and parentage. Merely drawing four colored clusters does not establish
four independent searches or functioning migration.

### Adaptive model allocation

`llm_dynamic_selection='ucb'` selects
[`AsymmetricUCB`](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/llm/prioritization.py#L470).
The runner selects an arm once per generation before native retry loops. A
correct child supplies its **combined evaluator objective**; the baseline is the
larger of parent and seed fitness. Negative improvements are clipped under the
native asymmetric scaling, with additional adaptive normalization/utility
handling; incorrect outcomes receive the native imputed failure reward. This is
not simply a bandit over raw escape rates or raw child-minus-parent scores.

The default cost-aware coefficient is unsuitable for interpreting subscription
allowance: native estimated API prices are not subscription charges. The proposed
treatment uses supported reward-only allocation (`cost_aware_coef=0`) with
separate hard call/time budgets. It does not invent a dollar conversion.

Use genuinely distinct available models when possible. Two effort settings on
one model would be a **configuration bandit**. Aliases resolving to the same model
and effort do not provide diversity. The Headless route does not forward the
usual temperature/max-token fields to Codex; varying ignored settings creates no
meaningful treatment. Record actual effective settings, selection counts and
rewards, not just the requested configuration. Auxiliary roles have separate
clients and are not automatically chosen by this mutation bandit.

### Embeddings, novelty and rejection

The [embedding client](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/embed/client.py)
supports the repository's real local encoder through an OpenAI-compatible
loopback endpoint. The
[edit implementation](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/edit/async_apply.py#L361)
embeds only the **first 10,000 source characters**. A cached BGE sentence encoder
used on code is real embedding computation, but not validated semantic program
equivalence. Hashes, random vectors and empty embeddings are not substitutes.

The [native conditional judge](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/core/async_novelty_judge.py#L33)
requires a usable embedding, a parent island and initialized islands. It compares
with programs in that island. A similarity at or below threshold accepts without
a judge call; a higher similarity can invoke the judge against the nearest
stored source. Rejection returns to bounded native proposal attempts; exhausting
them is a failure, not permission to accept the last rejected program.

Some upstream exception/missing-embedding paths accept without a successful
check. Reuse the existing adapter's explicit failure handling, preserving pending
source instead of misreporting a successful novelty gate. Report similarity
distribution, compared IDs, explanations and rejected sources. Native novelty is
a proposal diversity heuristic; it is not evidence of scientific novelty or
useful behavioral diversity. A threshold that worked for another task is not a
validated constant for this one.

### Summaries, insights, scratchpad and recommendations

Meta updates comprise one summary per pending evaluated program, then global
insights and recommendations. The interval counts **pending evaluated programs,
including the seed**, rather than every fourth numeric generation ID. Four
pending programs usually mean six model requests before retries, not one.
Finalization can flush another partial batch. See the
[summarizer](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/core/summarizer.py#L76)
and [async implementation](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/core/async_summarizer.py#L70).

Retain full history and pending programs across a checkpoint. The existing
adapter guards a native partial-response problem: filtering failed summaries
before attaching generation IDs can misassign surviving text. A failed batch
must not manufacture valid guidance. Inspect whether a later supported operator
actually receives the recommendation; a scratchpad file alone proves neither
consumption nor benefit.

### Prompt coevolution

The [prompt evolver](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/core/prompt_evolver.py)
maintains an archive and selects guidance with UCB plus epsilon exploration.
Prompts can themselves receive diff/full mutations. Cadence depends on attributed
persisted descendants. The
[prompt database's credit](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/database/prompt_dbase.py#L850)
averages percentile ranks of **correct** descendants. Incorrect descendants add
to counts but are excluded from that fitness denominator; disclose this bias
rather than silently changing native credit.

An initial archive row is not a prompt mutation. A mutated prompt without a later
attributed child has not yet received outcome credit. Save prompt versions,
selection and child links. Keep evaluator identity, information boundary,
original objective and spending rules outside the evolvable guidance. Evolving
instructions may change search advice, never the scientific task.

## 4. Scheduling, throughput and subscription boundaries

The [native Headless provider](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/llm/providers/headless.py#L489)
serializes subprocess execution with a CLI lock. Concurrent summary coroutines
therefore do not establish parallel remote execution. The async runner can
overlap evaluation and proposal work, but its available workers and backend
limits still determine throughput. Upstream concurrency results on other API
routes are not measurements for this subscription route.

Native retries nest: novelty attempts, parent resampling, patch attempts and
[`AsyncLLMClient.query`](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/llm/llm.py#L659)
transport retries (default three). One proposal worker does not mean one request
per candidate. Count actual dispatches and set each layer explicitly.

Every permitted model role must use `scripts/subscription_headless.sh`, explicit
subscription billing and no paid fallback. Local embeddings have separate CPU,
memory and elapsed measurements. No global authentication or permission changes
are needed. The supervisor's own work is additional usage; report observable
limits and never equate zero API dollars with free computation or a known weekly
allowance percentage.

The previous failure supplies a practical lesson: a timeout with missing tokens
should not by itself invalidate a campaign whose prospectively declared primary
budgets are actual calls and time. Missing usage is unknown, not zero. Changing
that failure policy belongs in the **new** treatment, not a rewrite of historical
records. Use bounded retries, preserve partial work and stop safely at deadlines.

## 5. Resumption is more than reopening SQLite

The current original-task launcher uses `dreamer/native.py::CheckpointRunner`,
which prevents duplicate seed execution but does not restore all native state.
The repository already has useful narrow fixes in
[dreamer/native_run1.py](../dreamer/native_run1.py). Reuse them selectively:

| Concrete risk in the pinned path | Required handling |
|---|---|
| Evaluation timer includes proposal time | Existing fix starts its clock at evaluation launch. |
| Terminal pre-evaluation failure writes files but no database program row | Persist the failed slot and apply native credits once; distinguish reserved and completed counts. |
| Resume infers progress from database rows and rejects existing generation directories | Reconcile pending stage/source before scheduling; do not overwrite or regenerate accepted proposals. |
| Native bandit serialization omits its dedicated RNG | Restore that RNG with bandit statistics and global Python/NumPy state. |
| Prompt cadence and generation-zero initialization restart | Restore counters and reuse the existing prompt archive. |
| Meta history or pending batch is lost | Restore full state and keep completed-side-effect identities. |
| Pre-novelty generated source is interrupted | Resume its remaining gate; saved source alone is not an accepted proposal. Existing adapter still needs bounded work here. |
| Live evaluator files change after launch | Bind/check evaluator and dependency identities before every evaluation; an after-the-fact hash is insufficient. |
| Seed exception path can fabricate `correct=True` | Require real valid seed artifacts before mutation. |

Do not inherit the older adapter's predictive task wording, task-only score
assumptions, token-missing abort rule or escape-based saturation rule. Those do
not implement the original proposal. SQLite `ORDER BY RANDOM()` and asynchronous
side effects also limit exact replay: graceful continuation is a defensible claim;
bitwise replay of arbitrary crashes is not yet established.

## 6. What “full machinery” means in the next run

For each feature, keep three evidence levels:

| Level | Evidence |
|---|---|
| Configured | Frozen resolved settings enable it with a valid role/backend. |
| Verified reachable | Source path and bounded checks show its prerequisites can be met. Historical route fixtures are labelled as such. |
| Actually exercised | Real campaign requests, decisions, database events or descendant credit demonstrate it ran. |

Our original-task campaign currently has **no exercised search mechanisms**,
because it has not started. Earlier v4 fixtures verify useful routes but are not
new original-task discoveries. A conditional judge may never run on naturally
dissimilar proposals; crossover may lack an inspiration; migration may lack an
eligible migrant. Report these facts, not extra manufactured activity.

The next deliverable is evaluated code and measured behavior with this machinery,
not another broad audit. The [execution plan](original-proposal-plan.md) makes
adapter work finite, preserves the original proposal, and reserves the scientific
claims about individual mechanisms for genuine controlled comparisons.
