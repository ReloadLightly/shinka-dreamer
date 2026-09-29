# Executed protocol v1

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
enemy avoidance, with fixed conservative risk penalties. It is selected with
`--variants memory`, rather than maintained as a diverging copy of the same code.

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
wall closure. Agent randomness has a reproducible independent constant seed and
cannot reveal a world seed. Search's own stochastic ordering has not been studied
because no descendant was generated.

Report Wilson 95% escape intervals and 5,000 paired episode-bootstrap replicates
for differences. Brier intervals resample whole episodes and recompute ratios of
pooled sums/counts. Step bins compare matched experience but late bins contain
fewer/longer surviving episodes; they are not an unconditional learning curve.
On-policy Brier differences can reflect different visited states. Reusing the
published assessment results to improve later agents makes these cases development
data: a resumed scientific campaign requires a newly reserved final pool.

## Execution boundary

The evaluator loads a **trusted bridge**, never the candidate module. Each episode
starts `/usr/bin/python3 -I` with a minimal environment and JSON pipes. The worker
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
