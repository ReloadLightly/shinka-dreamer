# Original-proposal implementation

This path implements the original [Sakana proposal](../docs/namazu-proposal.md),
with joint evolution of `world_model_step` and `planner`. It is a separate
`namazu-proposal-reconstruction-v1` evaluator; historical forecast/task scores
must never be inserted into its campaign database.

The first bounded full-native block has now executed through
[`evolve_full.py`](evolve_full.py). It produced two valid descendants, each escaping
all five development cases, before two successive full-rewrite timeouts triggered
the declared checkpoint. The six-slot ceiling was not filled. See the
[scientific results](../README.md#42-first-native-evolutionary-block) and
[run state](plan-state.json). Stage 3 subsequently checked generation 2 on the
existing 32 development cases: **32 escapes versus 0 for the saved seed**, with
zero model calls and 5.55 seconds of evaluation. See the
[paired development results](../README.md#43-paired-32-case-development-comparison).
Stage 4 then compared the seed and both descendants on 64 disjoint private
selection cases: each descendant escaped **61/64**, the seed **0/64**. Generation 2
won the declared fitness rule by an unresolved margin of 0.000854. The 192 episodes
took 28.18 seconds with zero model calls. See the
[selection results](../README.md#44-selection-validation-and-assessment-freeze).
Stage 5 assessed the frozen generation 2 on 256 fresh paired cases: **239/256
escapes versus 2/256 for the seed**, with a fitness gain of **0.402573
[0.383537, 0.420061]**. Its 512 episodes took 64.06 seconds and 62.61 total
CPU-seconds with zero experiment-model calls. See the
[fresh assessment](../README.md#45-fresh-assessment-of-the-frozen-program).
The completed execution was recovered from the working tree and verified without
new episodes. Its pool is retired; further stages await instruction.

## Preserved design

The world is 15×15 with a 5×5 local view, two keys, a gating door, exit, three
anonymous random-walking enemies, nine attempted movements including waiting,
wall changes every 25 steps and a 200 step horizon. There are no private directional
laws, law switches, restricted enemy sensors or forecast requirements.

The seed retains map integration, five-step enemy-sighting avoidance, greedy
nearest-target movement and random wandering. Both functions and added helpers
may evolve. A dictionary `memory['believed_map']` exposes current reconstructed
cell types in initial-agent coordinates; other memory representations are free.

The original task score is
`clip((.1*steps+20*keys+30*door_open+100*escaped-50*caught+50)/250,0,1)`.
Accuracy is correct reported in-bounds map entries divided by all such entries,
pooled across an episode, audited **after observation and before movement**.
Fitness is the episode mean of `.6*task+.4*accuracy`. Invalid executions count 0.
Mean episode accuracy is used, preserving the proposal's aggregation rather
than pooling all episodes by map size. Coverage is diagnostic, not extra fitness.
The original positive step reward and candidate-chosen map coverage remain;
their limitations are disclosed, not silently changed.

## Explicit implementation repairs

Reuse the existing `dreamer/world.py` repaired maze, without changing it:
connected closed-wall layouts; a real locked door gating the exit; fixed borders;
closures skipped on occupants; no diagonal corner cutting; entry contact is fatal;
independent layout/enemy RNGs. Interaction opens an adjacent door before movement
and collects a key at the destination. These realize the prose and resolve
ambiguities in the example, so this is source-aligned rather than a byte-for-byte
reproduction of its buggy example. Full details are in that source.

Seed repairs apply actual observed displacement before mapping, maintain observable
inventory/door state, use anonymous sightings, and permit interaction while standing
on a target key. The evaluator privately translates relative coordinates and owns
truth/metrics. Candidates run in the existing isolated process; they never receive
the live maze, world seed or metrics. The original callback/CLI mismatch is repaired
with a fixed scheduler entrypoint. The shared transport has bounded replies and
the existing 192 MiB/10 CPU-second episode limits.

The previous audit incorrectly classified replacing reconstruction with future
prediction as a necessary repair. It is a different scientific extension and is
deliberately absent here. High reconstruction accuracy does not prove predictive
learning, useful memory or understanding.

## Initial development characterization — declared before execution

Evaluate only `proposal/initial.py` on 32 public development cases 0–31, one episode
each, at most 6,400 transitions. No parameter fitting, comparator engineering,
selection, held-out assessment or model calls. Save every outcome including invalids;
save the first case's replay by case order, without selecting an attractive outcome.
This small panel describes the initial implementation and seed, not evolutionary
improvement or generalization. Cases are not claimed fresh relative to historical
work. Figures use only these results and the shared Chromatic Fields theme.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python proposal/evaluate.py \
  --results_dir artifacts/proposal/initial-evaluation --episodes 32 --replay-first
```

The evaluator refuses to overwrite completed results; use a new directory for
reproduction. The original five-episode default remains the native search default.

## Original preparation entrypoint — retained separately

The [staged plan](../docs/original-proposal-plan.md) specifies the proposed bounded
full-native treatment. The preparation example below predates that integration;
it is not the execution command for a stage of the plan. Stage 1 used the separate
bounded launcher; do not use this unbounded example to continue its checkpoint.

Prepare an explicit configuration with zero model calls:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python proposal/evolve.py \
  --results results/proposal-reconstruction --slots 100 --episodes 5 \
  --model 'headless/codex@gpt-6-astra?effort=medium'
```

Adding `--run` launches upstream Shinka. No search is authorized or launched by
this implementation step.100 total slots and four islands are the source proposal's
configuration, not a commitment to spend them now. Choose a bounded execution
and review point before launch. Explicit model identifiers replace the proposal's
paid-provider examples; all roles use the existing subscription-only wrapper.

Parent/archive/inspiration sampling, migration, diff/full/crossover and meta
recommendations are native. Multiple explicitly supplied models enable native
UCB; one model does not demonstrate a bandit. Embedding novelty and prompt
coevolution are not part of this initial proposal configuration. No capability
is claimed exercised by preparation alone. Existing Headless blank-line and
evaluation-clock fixes are reused; request timeout is 290 seconds inside 300 seconds.
Timeouts do not invoke the v4 token-accounting stop guard. Token usage may remain
unknown for failed requests; do not report unknown usage as zero.
