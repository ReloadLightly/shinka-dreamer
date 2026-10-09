# RUN1 native execution and checkpoint scope

RUN1 uses pinned upstream Shinka `9912af12d423504b8d580f4179fd15f5f88b8c50`
and Headless `93cd9b06b85f848af1308c41e018991b33907c5e`. The
[prospective protocol](run1-protocol.md) and
[resolved execution evidence](../artifacts/campaign-v4/run1/native/native-report.json)
distinguish configuration from observed activity. This is one development search;
mechanisms were enabled together and their individual causal contributions are
not identified. No independent-search reliability claim follows.

The original predictive program is the evolutionary seed. Neither the hand-built
direction-aware comparator nor the selected v3 program was incorporated into it.
The four native seed rows represent one evaluated slot and three administrative
island copies. Generated proposals, patch failures, novelty rejections, completed
candidate slots and actual model calls are separate counts.

RUN1 stopped normally at **01:51:46 UTC on 9 October 2026**, after a remote
request timed out and its token usage could not be recovered. There are **three
persisted slots and a fourth admitted slot held before any valid proposal**.
Generation 3 retains its sampled context and actual request; there is no generated
generation-3 source, evaluation or outcome. The 32-slot ceiling was not completed.
The automatic call gate is blocked; no restart or additional call is authorized.

Across both invocations there were **12 actual model requests: 11 successful
returns and one timeout exit 124 without a response**. Raw ledger status
`completed` means the subprocess returned; it does not mean the request succeeded.
The 125,978 uncached input-plus-output tokens and 135,168 cached tokens cover only
the 11 calls with reported usage and are **lower bounds**, not exact totals for all
12 requests. The timed-out call's native session also lacks a token report.
Summed measured remote elapsed time is **1,445.235 seconds**, including its
590.600-second failed request. Eleven usage reports independently confirm
subscription billing; all twelve commands used the strict subscription wrapper.
Separate [supervisor usage](../artifacts/campaign-v4/run1/supervisor-usage.json)
must not be conflated with this experiment-route budget.

Native patch metadata contains four attempt records: two successful applications
and two generation-3 no-response failures. The latter include one local retry
after the call gate had blocked admission. There were **zero actual remote repair
calls**, zero returned-patch application rejections and zero discovery novelty
rejections. The duplicate-source rejection belongs to the separate readiness
fixture. Neither the local retry nor the held context is an extra remote request
or an evaluated candidate.

At the first normal checkpoint, three slots had persisted: the seed, generation
1 with an infrastructure failure, and a valid generation 2. There were 96 measured
world episodes; generation 1 produced no episode records. Eleven model calls
completed: two model readiness checks, one separate native novelty fixture, two
mutations, one discovery novelty judge, three program summaries, one global
insight call and one recommendation call. Their total was 125,978 uncached input
plus output tokens, 135,168 cached input tokens and 854.635 seconds of summed
remote-call elapsed time. These are checkpoint counts, not the eventual RUN1
totals; the compact ledger carries its snapshot time.

The first mutation used Sol/high and the second Astra/high. Native UCB recorded
both outcomes. Generation 1's embedding passed the low-similarity gate; generation
2 exceeded the 0.95 cosine threshold and its actual native judge accepted it.
The separate identical-source fixture had been rejected by the same native judge
path. These observations establish exercised plumbing, not validated semantic
novelty or model superiority. Native bandit `s` is a log-sum-exp accumulator under
exponential scaling, not a mean reward. Cost coefficient zero prevents estimated
API prices from influencing reward-only selection.

The normal checkpoint generated a final meta update after three slots, before
the configured periodic interval of four. On actual resume, native metadata
verified restoration of the dedicated bandit RNG, prompt evolution counter 2,
percentile counter 1 and one recommendation-history entry. The generation-3
[actual request](../artifacts/campaign-v4/run1/native/request-example-with-context.md)
contains its full parent, generation-2 inspiration, string feedback, a recommendation
from the saved meta text and the fixed experiment boundary. Its
[audit](../artifacts/campaign-v4/run1/native/request-audit.json) records exact hashes.
Migration, crossover and prompt mutation were configured and reachable but
**unexercised before the stop**. The initial prompt received native fitness credit,
but that is not an observed prompt-coevolution step. One archive inspiration was
actually supplied; no top-k inspiration was supplied. Model selection, local
embedding, conditional discovery judging, meta recommendations and actual
recommendation/state restoration were exercised. Their small number of events
does not establish that the complete mechanism ecosystem improves discovery.

Native Codex session metadata corroborates actual model and `high` effort.
Only session IDs, model/effort context, tool counts and token totals are exported;
reasoning, tool arguments and raw rollouts are excluded. Session token reports
describe the same calls as Headless usage and must not be counted again. All
observed billing attempts are subscription routes. Dollar amounts are API list
price estimates, not actual subscription charges. Local embedding computation
and candidate/evaluator CPU are separate from remote-call elapsed time.

## Preserved timing defect and correction

The installed native local scheduler measured its 480-second evaluation timeout
from `job.start_time`, which the asynchronous runner populated with proposal
start. Generation 1 consumed 526.33 seconds in proposal/embedding work, so its
evaluation was killed after approximately 0.03 seconds. Its zero score is not
evidence that the candidate algorithm performs poorly. The original incorrect
row, source, costs and bandit credit remain preserved. No retry was performed.

The [immutable runtime amendment](../artifacts/campaign-v4/run1/runtime-amendment.json)
adds a scheduler-only copy of the job with its evaluation-start timestamp;
original pipeline timing and all candidate limits remain unchanged. Every resumed
Headless command also receives the supported `--timeout 590`, allowing its own
child-process cleanup inside the unchanged outer 600-second cap. A factual note
about generation 1 is appended to future mutation/fix context without rewriting
historical feedback or meta text. Three deterministic fixtures checked these
changes with no model calls or world episodes. Frozen original files and installed
upstream files remain byte-identical. The amended child timeout was exercised by
the final request; no post-amendment candidate evaluation completed, so the
corrected evaluation clock has fixture evidence only in this RUN1.

## Reproduction and limited recovery

Export saved native evidence without running candidates or requesting models:

```bash
.venv/bin/python scripts/run1_native_report.py
```

The exporter opens runtime databases read-only and publishes complete canonical
sources, generated branches, raw patch attempts, lineage, call records, prompt
credit, meta text and representative actual requests. Private pools, observation
records, embedding vectors, large databases and global RNG arrays stay outside
the public archive. Native nonfinite unobserved bandit extrema become JSON null;
runtime originals remain unchanged.

The same-run corrected launcher is:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/resume_v4_run1.py --run
```

**Do not run this command now:** the missing-usage gate blocks automatic restart.
It documents the launcher used for the actual earlier resume, not authorization
for another request, RUN2 or a renewed budget. Original state and database snapshots are retained under
`results/campaign-v4-run1/checkpoints/before-runtime-fix`.

Automatic recovery covers a saved proposal **after native novelty acceptance**:
its exact source and acceptance evidence are reused for evaluation. A generated
proposal held **before novelty acceptance** is preserved but cannot yet be
automatically resumed through its remaining stage. The launcher deliberately
rejects blind continuation in that case. A later authorized recovery must complete
that saved proposal's novelty stage with its recorded context, count any actual
calls, and then evaluate it; regenerating or bypassing the gate would change the
experiment. No universal automatic-resume claim is made.

The actual held generation 3 is an earlier case: it has no generated source at
all. Its unknown token cost must first be resolved or explicitly bounded under a
new authorized accounting decision. A future authorized request must retain and
count the failed admission and its context; no completed program can be recovered
from an absent response. This stop is a preserved resource/accounting limitation,
not a negative evaluation of a generation-3 algorithm.
