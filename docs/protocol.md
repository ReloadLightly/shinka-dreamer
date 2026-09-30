# Executed protocol: v1 mechanics, v2 initialization and controls

This describes implemented mechanics. The Namazu text remains unchanged in
`namazu-proposal.md`. The primary regime is 15×15, 5×5 square visibility, three
moving enemies, two keys, a locked door, 25-step changes and a 200-step horizon.

## Correctness repairs and explicit choices

* Coordinates are `(x,y)`; grids index `[y][x]`. Agent memory is relative to its
  initial position. The evaluator privately translates exports with that origin.
* The nine integer displacements, including stay, are validated strictly; booleans
  are not accepted as integers. `interact` must be a boolean. Diagonal moves cannot
  cross a blocked orthogonal corner. Enemies obey the same movement geometry.
* Each turn: open a cardinally adjacent door if interacting with two keys; move;
  collect a key at the destination if interacting; record immediate enemy collision;
  move all enemies; toggle walls on scheduled ticks; check death, exit, timeout.
  Death takes precedence over escape. An enemy cannot move away to undo collision.
* The exit is a one-cell corner chamber with one door entrance, randomly rotated.
  This is a generation repair, not a claim to preserve the original arbitrary
  exit placement. Random interior walls number 20–34; a quarter are dynamic.
  Border/chamber walls never toggle. The fully closed dynamic configuration has
  cardinal paths connecting all outside floor cells, both keys and the door
  approach. Since opening adds paths, objectives remain reachable in every dynamic
  phase. This is a constructive dynamic-feasibility guarantee, not survival assurance.
* A closing wall occupied by an agent or enemy remains open. Closure is attempted
  again only at the next 25-step tick; no hidden per-cell deferred countdown.
* Reset rebuilds every episode state and RNG. Initial enemies occupy distinct
  non-object cells outside the agent's initial 5×5 view. Subsequently enemies can
  overlap, with binary occupancy observations; no identities are supplied.
* Local feedback reports actual displacement, key count and door state. Terrain
  and enemy overlay are separate local layers. The original single layer hid
  terrain underneath enemies; exposing the local terrain layer is an explicit
  observable-interface extension, not hidden-map access. Visibility remains square,
  without line-of-sight occlusion.

## Learning and planning extension

The seed maintains a terrain map, seen times, visits, localization and inventory.
Its Dijkstra search chooses visible objectives or frontiers with information gain.
The competent memory control uses the same mapping, goals, path search and local
enemy avoidance, with fixed conservative risk penalties. In v1 it was selected by
flags on `initial.py`. In v2 `--variants memory` always resolves to the immutable
original program in `controls/v1/memory.py`, regardless of `--program`. This is
the exact original source plus wrappers binding its original two interventions.
`original_predictive` resolves to the unchanged original predictive snapshot.
Both hashes and their `1ce4fb7` provenance are in `controls/v1/manifest.json`;
loading a changed control fails. Only selected-candidate ablations use the
selected candidate's code. All v1 result files remain unchanged.

The predictive agent also estimates binary enemy-occupancy transition frequencies
with Beta-style success/opportunity counts. Features distinguish currently occupied
open/tight cells, one/multiple neighboring enemies and no nearby enemy. Priors are
handwritten and imperfect; the transition law is not supplied as an exact model.
Labels come from consecutive observations of the previous view's inner 3×3, where
the possible enemy origins were all visible. Predictions are one-step spatial
occupancy probabilities. Planning adds expected risk to path costs and compares a
move to waiting. The seed does not learn a dynamic-wall clock, roll out a learned
multi-step simulator, or implement neural Dreamer. These remain possible evolved
program changes: both entry functions, representations, learning and helpers are
inside one unrestricted evolve block.

Controls are `reactive`, `memory`, `predictive`, `frozen`, `no_planning`, and
`frozen_no_planning`. Freezing suppresses only predictive count updates; ordinary
map, location, inventory, visits and current enemy sightings continue. Disabling
prediction in planning substitutes the fixed risk heuristic and preserves online
prediction training. `no_planning` and `frozen_no_planning` therefore follow exactly
the memory baseline's actions and observations, enabling a matched-experience
forecast comparison. Future descendants must be inspected to verify that they
respect these interventions; merely passing flags would not establish an ablation.

## Measurement and objective

`export_model` runs after action selection, before the world advances. The evaluator
privately selects the old agent position's nine 3×3 cells and twelve uniformly
sampled interior cells (with replacement). It scores their **stationary positions**
at `t+1`, including terminal turns. Targets are never supplied to the candidate.
The horizon is exactly one step; longer horizons are not measured in this experiment.

Enemy loss is binary Brier `(p-y)^2`. Omitted individual cells use the candidate's
declared `default_enemy`; an omitted default is 0.5. Invalid probabilities, actions
or export structure fail the candidate. Persistence uses visible binary occupancy
and 0.02 elsewhere; the fixed base comparator is 0.02 everywhere. The memory
control's reported forecast is this evaluator-computed persistence comparator.
Reactive supplies no forecasts and consequently receives 0.25 Brier loss.

Near and uniform-audit losses are separate. A third diagnostic conditions near
targets on at least one enemy within the current visible 5×5 square. It exposes
performance in threatening situations; it is not used for selection. Most cells
are empty, so a high absolute model score is weak evidence. Reconstruction uses
evaluator-selected uniform targets with coverage, visible/stale/unseen strata,
and last-observed ages maintained independently of candidate claims.

Objective **task06-forecast04-v1** replaces the draft's after-observation map
accuracy term with before-outcome prediction quality:

```
task = .65*escape + .10*keys + .10*door + .05*escape*(1-steps/200)
model = 1 - (near_Brier + audit_Brier)/2
combined = .6*task + .4*model
```

Each episode component is bounded [0,1]; the candidate score averages episodes
equally. Death and timeout have no survival-duration reward. Outcomes, keys, door
completion, successful-escape time and model metrics are retained separately.
Published Brier diagnostics pool targets; the scalar averages episode means.

## Splits, randomness and uncertainty

Development uses seeds 10000–10063. Validation uses 20000–20063. The first seed
was held fixed after development; validation exposed a planning regression and
was not used to tune it away. Final assessment contains 256 fresh random 63-bit
world seeds generated **after** the campaign's mutation process was stopped and
the seed source was frozen. They reside only in ignored local results. All six
conditions use the same episode pool and reset memory and process globals.

Layout, enemy transitions and audit targets have independently hashed RNG streams.
Enemy draw count is fixed at three per tick, independent of audit and policy;
policies can still change realized enemy paths by opening the door or preventing
wall closure. V1 incorrectly used `-I` (which ignores `PYTHONHASHSEED`) and seeded
`random` after module execution. V2 uses hash seed 0 and seeds module `random` with
712934 before candidate compilation/execution, independent of world state. Six
fresh workers executing the same stochastic program gave identical import-time
draws, string-set order/hashes and five subsequent decisions each; see
`artifacts/campaign-v2/reproducibility.json`. Newly created `random.Random` objects
must be explicitly seeded for reproducibility; OS entropy and clocks are not
virtualized. Descendant repeatability must still be checked if it uses these.
There is one native search campaign, with no independent search repeat. Native
sampling after a restart is not claimed to reproduce an uninterrupted proposal
sequence. This does not change the independently controlled evaluation streams.

Report Wilson 95% escape intervals and 5,000 paired episode-bootstrap replicates
for differences. Brier intervals resample whole episodes and recompute ratios of
pooled sums/counts. Step bins compare matched experience but late bins contain
fewer/longer surviving episodes; they are not an unconditional learning curve.
On-policy Brier differences can reflect different visited states. Reusing the
published assessment results to improve later agents makes these cases development
data: a resumed scientific campaign requires a newly reserved final pool.

## Execution boundary

The evaluator loads a **trusted bridge**, never the candidate module. Each episode
starts `/usr/bin/python3 -s -S` with a minimal environment and JSON pipes. System
and user site loading are disabled; the worker replaces `sys.path` with the two
explicit system-standard-library paths before importing its dependencies. The
temporary trusted isolation-module path is removed before candidate execution.
This replaces v1's `-I` while retaining the same OS boundary. The worker
reads source text, then installs `no_new_privs`, Landlock rules allowing only the
Python 3.10 standard-library tree, and a seccomp syscall allowlist **before** compiling
or executing the source. Repository files, credentials, `/proc`, network, process
creation, signals to other processes and evaluator memory access are unavailable.
Only observations/action feedback cross into that process. Candidates return
actions and model exports, never scores. A fresh worker prevents cross-episode
global-state leakage.

Limits: 192 MiB address space, 10 CPU seconds per episode, three wall seconds per
response, 512 KiB source, 2 MiB response, 64 descriptors. No algorithm catalog or
representation restriction is imposed. Optional numerical packages are not
provisioned in this first standard-library execution environment. Linux Landlock
and libseccomp are required; unsupported hosts fail closed. This is an OS execution
boundary, not a claim against kernel exploits, side channels or all hostile code.
The tests execute file/proc/socket/fork probes as real candidate code.

Raw episode logs and private seeds stay under ignored `results/`; compact paired
CSV rows omit withheld seeds. Replay world states are restricted to development
and validation examples. Native SQLite/WAL, prompts and detailed run logs stay local.

## Campaign v2 provenance and assessment point

`results/campaign-v1` and the published assessment pool are historical and remain
untouched. Initialization changes can affect stochastic programs, so all new search
uses `namazu-repair-predictive-v2` in `results/campaign-v2`. The maze and objective
remain exactly v1. Native evaluation rejects an evaluator or episode-pool mismatch.
The campaign manifest records source hashes for the evaluator, world, worker,
OS restrictions, bridge, original seed/controls and driver, plus the exact objective,
runtime limits, interpreter, installed Shinka/Headless fingerprints and resolved
native settings. Development has an explicit 64-seed file and byte hash.

The first v2 launch made one failed subscription probe before entering native
search. Later execution through the existing subscription login generated native
descendants. The early blocked-state artifacts remain historical records; the
current campaign report distinguishes slots, valid descendants, failed slots and
island seed copies. Local replication ran 320 control condition-episodes and a
separate 64-episode native evaluator check. All scientific control fields matched v1.

Recovery preserves the immutable original 100-slot manifest and records the
user's revised **50-total-slot** stopping point in separate execution records.
The evaluator, objective, cases and native search settings are unchanged. The
additive `scripts/recover_campaign.py` holds one controller lock, restores saved
unpersisted proposal lineage, and reloads native recommendation text and pending
programs. New recovery evaluation outputs use a separate directory. It delegates
proposal sampling, evaluation submission, database insertion, archive maintenance,
migration and subsequent evolution to the pinned upstream runner.

Final assessment requires an explicit campaign-specific path, recorded up front as
`results/private/campaign-v2-assessment-seeds.json`. After selecting and inspecting
an evaluated native descendant, `--reserve-assessment` locks its native ID and source
hash in `selection.json` **before** generating 256 fresh cases. No v2 final cases
have been generated yet. The new pool excludes the published v1 seeds and any prior
campaign pools available locally. `assessment-manifest.json` and the experiment's
manifest record the pool's exact byte hash; resume checks selection and evaluation
identity and refuses replacing/reusing an existing unregistered pool. Never put
these private files or observations into mutation prompts. The original v1 pool
hash is `8e05f239e7067822fb200bb2fa9ad27f4c034b3dc62c52cb18cf00488a2f06d6`.

Inspect a descendant before interpreting its ablations: locate the actual learned
state and update sites, verify unchanged mapping/localization under freezing, and
verify that planning no longer depends on predictions under the planning ablation.
For prediction comparisons replay identical observations and executed actions to
the learned/frozen models, or demonstrate exact trajectory equality. The original
seed's fixed-risk branch supplies matched experience; that fact cannot be assumed
for new code. The report script consequently emits its matched learning curve only
for the hash-verified original seed. `scripts/audit_candidate.py` now audits
development interventions using complete action/world trajectory hashes and
exported map/localization hashes. It checks exported parameter constancy under
freezing, records update diagnostics, and only plots a matched-experience learning
comparison when every trajectory matches. The generation-14 audit and its source
inspection satisfy these checks; this is development evidence, not a held-out
result or a guarantee for arbitrary future descendants. Report fixed-rule
improvements separately from benefits of online learning. The user's recovery
request expressly leaves the final assessment untouched: no v2 assessment pool
is reserved and no assessment command is run.
