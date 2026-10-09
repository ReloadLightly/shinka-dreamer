# RUN 1 prospective protocol — wave v4

Registered before new candidate outcomes. Start 2026-10-09 00:54:23 UTC; stop issuing work by 04:24:23 UTC; hard checkpoint deadline 04:54:23 UTC. RUN 1 only. Historical v3 assessment stays paused.

## Questions and environment

This is one discovery execution, not an independent-search reliability test or fresh assessment. Does full native Shinka produce substantive proposals and improved development task performance from the immutable original predictive seed? Separately, do selected v3 planner recommendation changes have consequences under paired short branches? The branch protocol is `run1-branches.md`; positive branch results are not a prerequisite for search.

The wave v4 maze preserves physical v3 mechanics and law distribution. Six equally weighted unique conditions: ordinary uniform/full visibility, stationary/full visibility, one-switch/full visibility, stationary/late visibility, repeated-25/full visibility, repeated-25/late visibility. The first three retain the ordinary v3 reference distribution; the last four relevant conditions form persistence × information (stationary versus private replacement dwell times uniform20..30 transitions (mean25); enemy sensor radius 2 versus 1). Stationary/full occurs once in fitness, shared between reference and factorial summaries.

Full visibility is the ordinary 5×5 enemy/terrain window. Late visibility keeps the 5×5 terrain/key/door window but reveals enemies only within Chebyshev radius 1. Outer enemy overlays become underlying terrain. The observation explicitly declares `enemy_visibility_radius`, so outer-ring absence is not evidence of no enemy. This changes encounter information rather than making layouts harder; it may also change state inference difficulty. It does not guarantee higher learning value. The new repeated law lasts a privately drawn integer20..30 transitions per segment (mean25), with the next law first applied on the following transition. Every replacement uses the v3 truncated Dirichlet distribution and total variation ≥0.3 from its predecessor. All schedules/laws remain evaluator-private; a known-law reference receives only current probabilities. Blocked attempts stay. Tags separate laws, switch timing, innovations, targets and candidate randomness; paired visibility variants share physical innovations/laws.

Selection uses eight newly drawn development layout cases × six conditions = 48 episodes per evaluated slot, fixed across this execution. Draw cases disjoint from all registered historical pools. No new selection-validation or final-assessment cases are drawn or scored in RUN 1. The development winner is selection-biased and is not a held-out champion. Absolute task fitness remains `.65*escape+.10*keys+.10*door+.05*escape*(1-steps/200)`; invalid execution contributes zero. Prediction losses and exposure are diagnostics/text feedback only. Report unconditional escape, death, timeout, invalid, keys/door/steps, pooled Brier sums/counts and computational cost separately. No composite-weight sweep.

## Search and resource limits

Initial source: immutable `controls/v1/predictive.py`, disclosed as the historical executable seed; no fitted comparator or v3 champion capabilities are manually incorporated. Whole-program representation/update/helper/planning evolution remains open. A warm start is not part of this treatment.

Primary method-comparison budget: **1,500,000 reported uncached input + output tokens**, summed over ALL model roles including readiness. Cached input tokens are a separate resource measure, not free computation; reasoning tokens are not added twice to output. Enforce at request boundaries: one already admitted request can overshoot, and its full usage counts. Missing usage is recorded and blocks further calls until resolved. Report observable supervising-assistant usage separately if available (not charged into a falsely exact experiment token measure).

Additional binding ceilings: 32 total candidate slots including seed and terminal failures; 120 remote model calls total; 7,200 summed remote request seconds; 600 seconds per request; the wall deadline above. Per-role maxima: mutation/repair 64, novelty judge 16, program summaries 32, global insights 8, recommendations 8, prompt mutation 16, readiness/fixtures 4; the total cap is binding even if role caps sum higher. Fixture calls count resources but not evolutionary discoveries. No repeated broad probes. These limits supersede the earlier default 50-slot budget for this separately declared run.

Maximum discovery evaluation: 1,536 episodes, 307,200 physical transitions, 15,360 candidate CPU-seconds (10 s per episode), plus up to180 evaluator CPU-seconds per slot (60 per coordinator/pool-worker process), at most5,760 across32slots. Each evaluation has an eight-minute wall limit and writes every completed episode before aggregation. These are conservative outer bounds, not expected use. One proposal and one candidate evaluation in flight, one DB worker; at most two episode workers, each 192 MiB address space, 10 CPU-seconds, 3-second reply deadline, source ≤512 KiB, response ≤2 MiB. Embedding service CPU-only, one inference thread, ≤1 GiB resident-memory target (measured RSS reported); combined host pressure checked before launch. Branch work is capped separately in its protocol. Focused deterministic correctness fixtures create no scored research episodes.

Planning evidence from v3: later descendants averaged ~174 seconds mutation/repair and ~92 evaluator CPU-seconds for 72 episodes before meta overhead. Forty-eight episodes suggest ~61 CPU-seconds if behavior costs transfer, not a guarantee. Multiple auxiliary roles mean the 32-slot cap may not be reached. Stop on the first binding resource; preserve incomplete work rather than spend the ceiling automatically.

Native configuration: four islands, native weighted parent/population/archive selection, one archive and one top inspiration; migration every four generations at rate .25; archive 32; diff/full/crossover probabilities .4/.3/.3; native UCB over genuinely distinct authorized `gpt-6-astra/high` and `gpt-6.1-sol/high`, reward-only `cost_aware_coef=0`; meta interval four; prompt evolution interval four. Auxiliary model Sol/high via the same strict subscription adapter. Actual resolved field names, conditional behavior, callable models and local embedding identity are verified and frozen in the campaign manifest before mutation. Native prompt coevolution cannot edit appended immutable scientific/resource boundaries. Native code embedding truncation is disclosed; embeddings are not a semantic proof of novelty. No single-feature causal attribution: full machinery is a combined treatment.

## Accounting and checkpoint

Store complete native databases privately and compact candidate sources/events publicly, including failed/nonwinning branches, parent/inspirations, actual island/migration events, model allocation/rewards, embedding/judge/rejection/repair activity, prompt credit and meta consumption. Ledger status is configured → verified reachable → actually exercised; fixture events remain separate. Cache and resume bind evaluator/seed pool/code/resource/model/dependency identities. Save dedicated bandit RNG as well as global RNG, prompt counters/archive, meta history and accepted-proposal stage. Never silently skip novelty on resume.

No decision about RUN 2, replication, fresh assessment or expanding this budget is authorized by the present execution. At checkpoint give actual counts and resource use, program/behavior changes, limits, visual migration remainder, exact commands and next recommendation, then stop.

Native correctness is an additional eligibility rule: every returned episode,
including invalid episodes with score zero, remains in the 48-episode mean.
However, any invalid episode marks the whole candidate `correct=False`; native
parent/inspiration sampling normally excludes such candidates. Preserve that
candidate's source, nonzero mean if any, and failures in the record. Thus native
selection is absolute task performance subject to whole-panel validity, not just
an average zero-score penalty. An evaluator-level exception may produce native
default-error metrics; durable per-episode checkpoints retain completed work for
accounting without pretending the entire panel completed.

Replacement exposure entries are prepared just before a transition's candidate
query. A failed query can leave an entry with zero executed steps; count an
encountered replacement only when that entry has `steps>0`. Consecutive visible
enemy observations are opportunity proxies, not identified enemy correspondences
or verified predictive updates. Analyses after replacement remain conditional
on surviving to that window.

Pre-execution correction: repeated-25 was initially drafted with fixed25-tick replacements. Before any v4 candidate result, this was changed to private independent uniform20..30 dwell times: exact fixed timing could be inferred from the public step, contradicting the unannounced-change information boundary. This is a declared initial design choice, not a fitted constant.

Prospective saturation review (before the first v4 candidate outcome): at completed total slots8,12,16,20,24,28, inspect the latest three nonseed generation rows. If all three are valid, each escapes all48development episodes, and their absolute-task score range is≤.002, checkpoint and stop admitting proposals. This detects a near-ceiling development panel and avoids exhausting32slots merely because they are available. It makes no held-out or learning-benefit claim. A matched RUN2 must use the same rule and retain the final incumbent over unspent resource intervals when comparing resource curves. Binding token/call/time limits still take precedence.
