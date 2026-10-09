# Implementing the original Sakana proposal, step by step

**Prospective plan recorded 9 October 2026; Stage 1, 3 and 4 checkpoints below.**
The plan was originally commissioned as research and planning only. The user chose
budgets **stage by stage** and subsequently authorized Stages 1, 3 and 4. Later-stage
ceilings remain proposals for review, not spending authorization. Each execution
ends at a checkpoint and awaits a new instruction. There is no unattended sequence
of stages.

## Execution checkpoint

The user authorized Stage 1. Its invocation is checkpointed;
see the [results](../README.md#42-first-native-evolutionary-block) and
[machine state](../proposal/plan-state.json). Two valid descendants escaped all
five development cases. The run stopped after two successive full-rewrite
request timeouts, before reaching six completed slots.

The separately authorized Stage 3 is also complete. Generation 2 escaped all
32 existing development cases versus 0/32 for the saved seed. It used 32 new
episodes, 2,024 transitions, 5.55 seconds elapsed evaluation and 5.45 CPU-seconds,
with zero experiment-model calls. Original sources and five-case search scores
are unchanged. See the [paired results](../README.md#43-paired-32-case-development-comparison).
The separately authorized Stage 4 is now complete too: on 64 disjoint private
cases, both descendants escaped 61/64 and the seed 0/64. Generation 2 was selected
by original F, but its advantage over generation 1 remains unresolved. The 192
episodes used 28.18 seconds elapsed and 27.62 total CPU-seconds, with zero model
calls. See [selection results](../README.md#44-selection-validation-and-assessment-freeze)
and the [assessment freeze](../artifacts/proposal/stage4-selection/assessment-freeze.json).
No subsequent stage is authorized. Stage 5's proposed 256 fresh pairs remain
undrawn; the next step is its separately authorized evaluation and report.

## 1. Destination and fixed scope

Implement and investigate the [preserved Sakana proposal](namazu-proposal.md):
jointly evolve executable `world_model_step(memory, local_obs, last_action)` and
`planner(memory, local_obs)` functions for the dynamic, partially observed maze.
Evolution can change helpers, memory representations, inference, exploration and
planning. `believed_map` is an assessment export, not a required internal algorithm.

Keep the original 15×15 world, 5×5 observation, two keys, gating door, exit, three
anonymous enemies, nine attempted moves including waiting, wall changes every
25 transitions and 200-transition horizon. Preserve the documented executable
repairs, including localization feedback, blocked movement, interaction and
contact ordering. Do not introduce hidden movement-law regimes or a forecasting
contract into this track.

Keep the original objective, including its limitations:

```math
S=\mathrm{clip}_{[0,1]}((0.1T+20K+30D+100E-50C+50)/250),\quad F=0.6S+0.4A.
```

Here, `A` is current-map accuracy over reported in-bounds cells, pooled over an
episode after receiving the observation and before the transition. Fitness is
the mean episode `F`; invalid executions contribute zero. Do not silently replace
this with future prediction, task-only selection, accuracy over a different
denominator, or a reward for outperforming the program's own ablation.

The proposal's claim that reconstruction rewards imply world understanding is a
hypothesis to examine, not an established fact. Separate:

1. Higher composite fitness from better task outcomes.
2. More accurate reported cells from greater coverage and useful memory.
3. A useful selected program from a repeatable discovery process.
4. Observing a Shinka mechanism operate from showing that it improves search.

The main research questions are whether joint program evolution improves this
task, what executable changes account for observed behavior, and what the native
search machinery actually contributes under measured budgets. No paper acceptance
or novel algorithm is promised.

## 2. What is already complete

At repository checkpoint `b1ebded`:

| Completed item | Evidence and implication |
|---|---|
| Original-task implementation | [proposal/](../proposal/README.md): repaired seed, isolated candidate process, evaluator, native launcher and plotting code. Preserve this work. |
| Initial characterization | 32 development episodes; 4,386 transitions; zero escapes, 20 captures, 12 timeouts, zero invalid executions. Mean `S=0.172325`, `A=0.983149`, `F=0.496655`; final map coverage 32.90%. |
| Measured seed runtime | 2.84 summed episode wall-seconds, including candidate execution. This is not a forecast for evolved programs or LLM calls. |
| Scientific README | Two Chromatic Fields figures from these actual original-task measurements. |
| Native integration | Prepared but unexecuted for this objective. The current launcher does **not** yet enable embeddings, novelty judging or prompt evolution, or full checkpoint restoration. |
| Shinka implementation research | [Source-level findings and mechanism guide](shinkaevolve-research.md), including current upstream comparison and reusable adapters. |

Do not repeat the initial 32-case experiment, historical profiling or prior v3
assessment. No v2/v3/v4 scores enter the original-task search database. Their
programs and conclusions remain in the [historical archive](experiments-archive.md).

## 3. Proposed execution sequence and budgets

All limits apply together; stop at the first binding boundary. Counts are maxima,
not targets to exhaust. Model-call limits count actual provider subprocess
attempts across **all** roles, including failed requests and retries. Wall time
includes implementation, checking, execution, analysis and checkpointing.

| Stage | Deliverable | Proposed ceiling | Review decision |
|---|---|---|---|
| **1. First full-native block** | Reuse adapters, then obtain the first original-task evolved programs and a compact result page. | 40 min total: at most 12 min integration, 23 min execution, 5 min checkpoint. Up to **6 total slots**, seed included; **30 all-role calls**, **1,200 summed provider seconds**; 30 episodes / 6,000 transitions; 300 candidate CPU-seconds. | Did real proposals run? Did behavior or fitness change? Is continuing worth its measured cost? |
| **2. Continue only after review** | Another bounded block in the same unchanged treatment. | Each separately authorized block: **25 min**, up to **6 additional slots**, **24 all-role calls**, **900 provider seconds**, 30 episodes / 6,000 transitions / 300 candidate CPU-seconds. Reserve the last 5 min for checkpointing. | Continue, finish selection, or propose a separately versioned correction. Never automatically fill 100 slots. |
| **3. Development behavior review** | Source comparison and paired replay/outcome checks for at most two nominated programs. | **20 min**, **zero model calls**, up to **64 new episodes / 12,800 transitions / 640 candidate CPU-seconds**; last 5 min for reporting. Reuse the seed's saved 32 cases. | Does apparent progress survive more development cases, and is it navigation or metric exploitation? |
| **4. Select and freeze** | Choose from at most three nominated descendants on a separate selection panel; freeze comparisons and assessment protocol. | **25 min**, zero model calls, **64 paired cases × at most 4 programs = 256 episodes / 51,200 transitions**, **800 total evaluator CPU-seconds**. Reserve 5 min for the freeze/report. | Is there a defensible program and comparison worth assessing? |
| **5. Fresh assessment** | Paired selected-program versus seed assessment on the original task. | Proposed **256 fresh cases × 2 programs = 512 episodes / 102,400 transitions**, zero model calls, **30 min elapsed**, **1,200 evaluator CPU-seconds**; reserve 5 min for analysis. Confirm feasibility from Stage 3–4 runtime before drawing cases. | What can be claimed for this frozen program, with uncertainty? |
| **6. Paper-style synthesis** | Update README, figures and a concise manuscript draft from completed evidence. | **30 min**, zero experiment-model calls and zero new episodes. No new campaign hidden inside writing. | Identify the smallest scientifically useful follow-up. |

Stages 3 and 4 have different purposes: the 32 known development cases may inform
further mutations; the selection panel may choose a champion but must not become
mutation feedback. Stage 3 is scheduled after an informative search block, not a
mandatory repeated cost after every six proposals. Stage 2 requires an explicit
continuation each time. The source's 100-slot target remains a possible cumulative
campaign size; it is neither the current budget nor a completion requirement.

These are deliberately finite review units. Earlier predictive descendants took
roughly 174 seconds of mutation/repair plus 92 evaluation CPU-seconds on average;
their evaluator was different. Five such mutation requests alone would take about
14.5 minutes before auxiliary work. Therefore six completed slots are an aim,
not a throughput guarantee. If Stage 1 cannot finish adapter reuse within its
12-minute allocation, checkpoint the concrete changes and blocker; do not consume
the rest of the invocation on open-ended infrastructure or begin provider probes.

For stages with an aggregate CPU ceiling, admit complete paired case batches only
while worst-case remaining CPU fits. If a ceiling prevents the declared panel
from completing, report an incomplete panel and preserve its unexposed remainder;
do not treat its partial result as the promised full assessment or enlarge the
budget automatically. Final sample size is confirmed prospectively, never extended
because a result missed significance.

## 4. Stage 1: minimum changes before the first mutation

Use a separate directory such as `results/proposal-full-native-01`, with evaluator
identity `namazu-proposal-reconstruction-v1` and its own search-treatment identity.
Do not overwrite the prepared `results/proposal-reconstruction` configuration.

1. **Reuse the working boundary.** Keep `proposal/initial.py`, the evaluator and
   the world unchanged. Bind source/dependency hashes before execution and check
   them at every evaluation launch, including the seed. Keep the fixed candidate
   random stream; do not change its policy during this treatment. Record this
   limitation in generalization claims.
2. **Reuse narrow adapters.** Transfer the existing timer, transport, native
   state, terminal-failure, novelty and meta-batch fixes from
   `dreamer/native_run1.py`. Replace its predictive prompt boundary and task-only
   assumptions with the original task. Do not copy its token-missing abort rule,
   old saturation rule or old budget. Keep this small; do not rewrite Shinka.
3. **Apply one budget gate to every role.** Count actual calls before dispatch,
   enforce deadline and request-time reserves, and preserve missing token usage
   as unknown. Disable nested transport retries by explicitly setting
   `SHINKA_LLM_MAX_RETRIES=1`; permit the declared native repair/novelty attempts
   only inside the shared budgets. No separate readiness-model campaign.
4. **Check concrete failure risks offline.** Reuse fixtures for timer origin,
   valid seed artifacts, failed-slot counting, prompt/meta restoration, and
   accepted-proposal recovery. Add only checks needed by these changes. Synthetic
   fixture replies are not discoveries. Do not repeat model availability or
   duplicate-code judge probes already evidenced in the historical adapter work.
5. **Start actual native evolution.** Evaluate the seed on the same five cases
   used for descendants. Inspect the first real request for the intended task,
   parent, inspirations where available, string feedback and effective model.
   Verify actual embeddings and native state as this work executes.

A runtime dependency that is genuinely unavailable is a concrete blocker. Do not
silently disable a required mechanism and label the reduced run “full Shinka”.
Do not spend extra requests merely to make a mechanism's event count nonzero.

### Stable proposed full-native configuration

Freeze this configuration before the first proposal. All unspecified defaults
must be exported in the resolved configuration; material changes start a new
treatment. No tuning of mechanism probabilities from early results inside this
campaign.

| Mechanism | Initial setting and interpretation |
|---|---|
| Fitness panel | Five fixed development cases, IDs 0–4, equally weighted; original `F`. Seed evaluated anew on these five, not assigned the 32-case mean. |
| Population | Four native islands with uniform island selection; weighted parent sampling with lambda 10; fitness archive size 40; one archive and one top-performing inspiration where eligible; native island separation and elitism enabled. |
| Migration | Interval 4; rate 0.25. Native elite protection and eligibility remain intact. Scheduled maintenance can legitimately move zero programs. |
| Program operators | `diff/full/cross = 0.4/0.3/0.3`; native removal of crossover when inspirations are unavailable. Whole-program editing remains open. |
| Mutation allocation | Native `AsymmetricUCB`, reward-only `cost_aware_coef=0`, over `headless/codex@gpt-6-astra?effort=high` and `headless/codex@gpt-6.1-sol?effort=high`. These routes have prior successful repository records; actual effective identities are checked on real requests. No extra availability probes. “Sol/high” below means this exact 6.1 route. |
| Genuine embeddings | Existing local CPU BGE/ONNX route: `qdrant/bge-small-en-v1.5-onnx-q` at `aa8f8b060edb00e03bfdd08813a2949946c8ba55`, endpoint `bge-small-code-chunks-v1`, 384 dimensions. Reuse the cached model/tokenizer; no download or paid fallback. Record source-prefix truncation, hashes and inference time. |
| Conditional novelty | Cosine threshold 0.95; Sol/high judge through strict subscription route; `max_novelty_attempts=2`. This is an initial engineering choice, not a scientifically calibrated code-similarity threshold. |
| Repairs/resampling | One parent resample, at most two patch attempts per novelty attempt; provider retry count one. All nested attempts consume the same all-role call/time caps. |
| Text feedback | String-valued `F`, `S`, `A`, coverage, outcomes, keys/door and actual execution failures; no withheld cases or hidden state. |
| Meta memory | Native per-program summaries, insights and recommendations after 4 pending evaluated programs; Sol/high auxiliary model. Preserve full scratchpad/history. |
| Prompt evolution | Enabled, interval 4 attributed descendants, archive 10, native UCB constant 1 and epsilon 0.1; native diff/full prompt probabilities 0.7/0.3; native percentile credit with recomputation interval 4. Sol/high. Immutable task/resource rules appended outside evolvable guidance. |
| Execution | One controller; one proposal, one evaluation and one DB worker; one numeric thread. No controlled oversubscription. Embedding service limited to one inference worker. |
| Per-request boundary | Proposed 240-second child timeout inside a 250-second provider timeout; include lock wait and process cleanup in elapsed accounting. Admit no request whose maximum duration would cross checkpoint reserves. |
| Candidate boundary | Existing 512 KiB source limit, 192 MiB address space, 10 CPU-seconds per episode, 3-second response limit, 2 MiB response limit. Budget these separately from controller/embedding memory. |

The local route serializes Headless subprocesses. Four islands do not mean four
simultaneous model calls. Increasing evaluation workers cannot remove that
bottleneck. Keep the known modest concurrency first; measure role times from
actual work. Set a 2 GiB combined resident-memory ceiling for controller, evaluator
and local embedding service, monitored without changing global settings. If the
cached service cannot fit, report that concrete constraint. Faster parallelism is
a later declared execution change, not a promise of reduced subscription usage.

## 5. Failure, checkpoint and review rules

A timeout consumes a call and elapsed budget; record its tokens as unknown if
unreported. A failed mutation can consume its reserved candidate slot without an
evaluation. Neither event erases successful candidates nor invalidates their
measurements. Stop admitting remote work after two consecutive provider transport
failures, a missing required route, or a resource ceiling; checkpoint immediately.

Novelty transport failure must not masquerade as acceptance. Preserve a generated
source pending its actual novelty stage. Preserve accepted proposals pending
evaluation without generating them again. If pre-novelty continuation cannot be
implemented within the bounded stage, report that specific paused stage rather
than skipping the gate. Partial meta batches retain raw replies and last valid
guidance; do not attach summaries to the wrong programs or invent prompt credit.

Native finalization can launch another meta batch. It must use the same admission
gate: queue unpaid pending work for a later authorized continuation instead of
overrunning the deadline. Distinguish an incomplete feature update from evidence
that the feature is unsupported. Save database, prompt archive, bandit state and
its RNG, Python/NumPy RNG, pending proposals, full meta state and applied-side-effect
IDs. This supports continuation; it does not promise bitwise replay of native SQL
random sampling or every asynchronous crash.

At each checkpoint, provide one concise result table and answer:

- Which slots, calls, episodes and failures actually happened?
- What changed in the best and other substantive programs? What happened in play?
- Did fitness improvement come from task reward, reported-map accuracy, or both?
- Which mechanisms were configured, reachable and exercised? Which remained
  conditional or pending?
- What did mutation, auxiliary work, embeddings and evaluation cost separately?
- Is the next marginal block justified? If solved on five cases, prefer broader
  development checking or selection over filling the remaining slot target.

An early high score is not a reason to declare generalization, and a plateau is
not a reason to spend the entire nominal budget. End every block for review even
if budget remains. Stop/reap experiment workers and automatic continuations.
Keep a local-only dashboard while the run is active if requested; it is a view of
the database, not a separate source of measurements.

## 6. Evidence beyond five selection cases

Stage 3 nominates at most two programs by a rule fixed before its evaluation:
the best search-fitness program, and, if different, the best task-score program
(ties use earlier generation). Evaluate them on the existing 32 development IDs.
Reuse the exact seed evidence only if candidate, environment, evaluator, RNG and
resource identities match. Inspect source differences and prespecified replay
indices, including failures. Results may become subsequent development feedback.

Before Stage 4, stop search and nominate up to three descendants: best `F`, best
`S`, and best escape count on the common five-case panel, deduplicated with earlier
generation as tie-breaker. Evaluate these and the seed on 64 disjoint private
selection-validation cases. Choose the descendant with highest mean original
`F`; break ties by mean `S`, then earlier generation. Publish selection results as
selection-biased. Do not tune programs on this panel. If it is later exposed for
tuning, retire it and make that change explicit.

Stage 5 freezes selected source, seed, evaluator, analysis and sample size before
drawing 256 fresh paired cases, disjoint from all exposed repository pools.
Primary contrast: paired mean `F` difference. Report paired `S`, `A`, coverage,
escape, capture, timeout, keys, door, episode lengths and computational cost
separately. Resample **whole paired episodes**, keeping their denominators and
invalid outcomes, for descriptive 95% bootstrap intervals. The original `A`
estimand averages episode-level pooled ratios; do not silently replace it with a
global cell-weighted ratio. Report binary discordant-case counts as well.

For scale, 256 cases give at most roughly ±6.1 percentage points for a single
escape proportion under a normal approximation; the worst-case paired escape
difference can be roughly ±12.3 points. These are planning bounds, not promised
observed intervals. If a smaller meaningful effect is the scientific target,
choose a different sample size **before exposure**, with measured runtime and a
separately approved budget. Do not resurrect the large historical assessment.

Selected versus seed establishes an improvement over this starting program, not
superiority over competent maze planners. A stronger paper may justify one
existing competent memory/pathfinding comparator adapted to this exact interface
and evaluated on the same panel. Decide that addition and its budget before the
assessment freeze; label its manual engineering and never use historical scores
as its new result. It is not a prerequisite for the first mutation.

Inspect whether map coverage shrinks, old cells are discarded, or long survival
increases the original score without escape. Report such findings rather than
changing the objective after seeing them. A memory intervention is optional and
source-specific: only perform it if it has a coherent meaning that preserves
localization and control interfaces. Do not reuse a predictive-model freeze flag
and call it evidence about the original proposal.

## 7. Learning about Shinka without confusing observation and causation

During the main run, answer mechanistic questions from saved execution evidence:
why a parent was eligible and sampled; whether inspirations supplied code used in
a child; which operator and arm ran; why novelty accepted/rejected a proposal;
whether migration moved a program; which recommendation or prompt reached a
later proposal; and how much each role cost. The companion
[research guide](shinkaevolve-research.md) specifies native behavior and caveats.

This combined configuration tests the complete system. It cannot attribute gains
to individual features. If the original task produces substantive results, a
separately proposed follow-up can compare full Shinka with independent rewrites
from the same seed, objective, cases and available model pool. Use **actual
all-role provider-call count** as the primary declared comparison budget and
report request time, model allocation, tokens with missingness, CPU and wall time
alongside it. Equal calls are not equal subscription allowance or equal compute.
Charge full Shinka for its auxiliary calls. Specify the rewrite allocation policy
before execution and withhold other candidates/history from rewrites.

Replication must consist of independent searches; islands and more episodes do
not supply it. A provisional design of three independent runs per treatment is
an exploratory starting point for review, not permission to launch six searches
or a guarantee of statistical power. Each run gets a separate budget and
checkpoint. Individual mechanism ablations require their own targeted hypotheses;
do not add an automatic combinatorial ablation campaign to this plan.

## 8. Figures, scientific writing and durable handoff

Keep the README in paper form: abstract, question, methods, results, limitations
and reproducibility. After each substantive block update the results, rather than
turning its opening into an operations diary. Keep compact settings and state
beside the data; do not make bookkeeping the project output.

All new figures use [Chromatic Fields](VISUAL_STYLE.md) through
`scripts/visual_theme.py`: background `#FFFEFC`, text `#161625`, secondary
`#6F6A78`, cobalt `#3534CF`, magenta `#C93683`, orange `#F26A37`, rules `#DCD7E0`.
Use shared CSS for project-controlled browser views. The upstream WebUI is an
operational inspection aid; any screenshot or exported panel published in the
README must first be rendered in this theme. Inspect actual renders.

Prioritize four evidence-driven figure groups, adding panels only when measured:

1. Every candidate's task, map accuracy, coverage, fitness and outcome, with failed
   slots visible; progress against calls and elapsed/CPU time.
2. Actual ancestry, inspirations, operators and migrations, annotated with
   substantive source changes; bandit, novelty and prompt/meta events.
3. Actual maze/map replays and paired development/assessment outcomes with
   uncertainty; no illustrative invented results.
4. Resources by model role and program evaluation, including missing usage and
   failures.

The existing two README figures already use original-proposal data and this
style. Keep historical extension figures out of this paper's evidence. Their
preservation or restyling must not launch experiments or delay actual mutations.

At every execution checkpoint, commit and push completed coherent work to the
verified origin, exclude credentials/private pools/large databases, save the exact
invocation and resume command, and stop. Do not invent a command for features not
implemented yet. Stage 1 supplied the bounded full-native launcher; its saved
failure gate/deadline prevent further dispatch without a newly authorized block.

**Next action after the Stage 4 checkpoint:** await the user's instruction for
Stage 5 fresh assessment of frozen generation 2 versus the original seed. No further
mutation or historical assessment is authorized. The prospective stage budgets
above remain preserved. The original
Stage 1 launcher is now implemented; its held pre-source slot still requires
explicit recovery before any newly authorized evolutionary continuation.
