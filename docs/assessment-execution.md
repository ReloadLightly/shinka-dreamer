# Generation-14 assessment execution record

Scientific findings and interpretation are in the [README](../README.md); the
[complete numerical report](assessment-results.md) retains every comparison.
This record documents execution and preservation rather than scientific claims.

## State inspected before execution

The repository root was `/home/roland/projects/shinka-dreamer`, with origin
`https://github.com/ReloadLightly/shinka-dreamer.git`. The clean local checkout at
`19df1f4` was behind the reviewed remote commit
`02d15793637b30a653c1e5c7c113cd33dfc2d576`. The newer documentation and analysis were
inspected and fast-forwarded; no work newer than that reviewed remote commit was
present. Existing processes were inspected on the host. There was no active
ShinkaDreamer controller or evaluator; unrelated processes were left alone.

The campaign database already contained 50 distinct slots (0–49), with 53 native
rows because island seed copies are separate records. The campaign was not
resumed or extended. Generation 14 had the requested SHA-256
`588eeb7c10b978fe86c7e7b572f177e8b755c9c25699f3c75bfadd41192ec5e0`.
The immutable original controls and selected source were checked before execution.

The existing private v1 assessment pool and saved results were found and retained.
Its pool hash is
`8e05f239e7067822fb200bb2fa9ad27f4c034b3dc62c52cb18cf00488a2f06d6`.
No campaign-v2 selection, reservation or assessment results existed. An inventory
of 950 historical/protected files was hashed before making changes. It includes
the old results, native campaign, original controls, Namazu proposal, selected
source and evaluation implementation; SQLite transient WAL/SHM files were excluded.

## Reservation, runtime and checkpoints

The fixed design and audit driver were committed at `38e437a` before case
generation. The existing reservation helper first froze the selected native ID,
then generated exactly 1,024 unique 63-bit seeds, excluding development/validation
ranges and previous local assessment pools. The new pool remains local at
`results/private/campaign-v2-assessment-seeds.json`; its byte hash and selection
hash are in the committed manifests. No substitute pool was created.

The existing environment ran this command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u \
  scripts/assess_selected.py --workers 4
```

One controller held an exclusive `flock`; four child workers handled paired
cases, each executing all six conditions in rotating order. Host process
inspection confirmed that the four children were workers of that controller,
not independent controllers. Case JSON was fsynced and published atomically
without replacing existing records. A completed file contains all six conditions.
Restarting the same command verifies the registered pool, source and kernel
hashes, and skips saved cases. Incomplete checkpoint bytes are preserved and any
necessary rerun is explicitly logged. This execution used one launch, starting with zero saved cases; no controller restart or case rerun was necessary.

The evaluator kernel and all original boundaries stayed unchanged: 192 MiB
candidate address space, 10 CPU seconds per episode, three wall seconds per
response, 512 KiB source and 2 MiB response. Candidate isolation still uses Linux
Landlock and seccomp, standard-library Python, controlled initialization and
independent layout/enemy/audit RNG streams. No authentication, global Codex,
approval or sandbox settings were changed.

The launch record is `execution-20260930T182112655919Z.json`. All 1,024 cases and
6,144 episodes completed in **2,873.1 seconds (47.9 minutes)**.
Summed episode host time was 8,334.7 seconds, which overlaps across
workers and is not wall time. Each condition has 1,024 saved observations. Descriptive
progress was reported throughout; inferential tests waited until completion.

All generation slots and assessment episodes used their existing authorized
routes. This assessment itself made **zero model calls**: mutation, recommendation,
novelty, embeddings and evaluator model calls were all absent. No paid API or
billing fallback was used. The next-study campaign was not launched.

## Failures and intervention evidence

Four episodes ended with `RuntimeError: Candidate worker exited (-9)`:

| Case | Condition | Completed actions | Host seconds |
|---|---|---:|---:|
| 0036 | Frozen | 76 | 11.616 |
| 0509 | Selected | 130 | 10.599 |
| 0588 | Selected | 142 | 10.522 |
| 0589 | Selected | 122 | 10.597 |

Exit −9 indicates SIGKILL. These durations are consistent with the worker's
10-CPU-second hard limit, but the existing evaluator does not retain the signal's
sender or child CPU accounting. The precise sender is therefore not established.
No kernel or resource-limit change was made to investigate them. They remain
invalid non-escapes with their recorded partial task/forecast data and zero
selection score. No case was replaced, dropped or rerun for inference. All four
are discordant escape pairs for selected versus frozen; the control comparison
includes execution reliability under the prespecified limits.

Every actual assessment episode used `compact_trace` from
`scripts/audit_candidate.py`. It hashed action/world/enemy and exported
map/localization histories, recorded parameter constancy and memory checks, and
discarded the full trace. The evaluator was not modified to add reporting.
Both frozen conditions retained constant exported transition weights on all
1,024 episodes. Weights changed in 991 selected episodes and 993 learned
fixed-risk episodes. Mapping and localized movement remained active in every
condition-episode. All 1,024 fixed-risk pairs had identical action/world and
map/localization hashes, with zero localization errors and zero visible-map errors
across 8,116,112 checks. The matched learning claim is therefore supported by
actual assessment checks, not inferred from development flags.

## Analysis closure and published examples

Analysis and figure code were committed at `f3279e4` while evaluation was still
running. Statistical analysis waited until all 1,024 case files were complete.
The completion record hashes every case; the exported episode archive contains
6,144 compact records without seeds or full hidden traces. The analysis uses the
prespecified exact paired binary inference, Holm secondary correction and 10,000
paired episode bootstrap draws. Neither sample size nor agent source changed
after looking at results.

`analysis-closed.json` fixes the numerical analysis hash before any assessment
example is exposed. The lowest-index case in each prespecified selected/frozen
escape stratum is then replayed. Cases 0000, 0073, 0009 and 0001 represent selected-only, frozen-only,
both-fail and both-escape outcomes, respectively. All eight diagnostic replays
reproduced all original scientific fields and audit hashes exactly, excluding
host runtime from deterministic equality.
These additional diagnostic replays are not extra inferential episodes and never
replace original observations. Only these examples' seeds and compact display
frames are published, with an explicit exclusion from future fresh tests.

## Reproduction and final checks

The analysis environment used Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3 and
Matplotlib 3.10.9 on Linux/WSL2 x86-64. Exact analysis dependencies are listed in
`requirements-analysis.txt`. Candidate execution uses `/usr/bin/python3` and
the runtime identity in the manifest. The original campaign used upstream
ShinkaEvolve `9912af12d423504b8d580f4179fd15f5f88b8c50`; no new search was run.

Recompute the closed statistics and regenerate tables and figures using the
[reproduction instructions](reproduction.md). SVG and PDF figures preserve text
and vector marks; PNG copies provide convenient previews. The original development
replay remains unchanged.

All **16 tests passed**, including sparse-discordance uncertainty, exact McNemar
direction, unequal-denominator pooled bootstraps, checkpoint preservation and
intervention auditing. Recomputing from the compact archive reproduced the closed
analysis hash exactly. All four publication figures were visually inspected. Regenerating all 12
SVG/PDF/PNG files reproduced their bytes exactly; private-pool uniqueness,
registered hash and exclusion of the v1/development/validation cases were verified.
The [preservation record](../artifacts/campaign-v2/assessment-1024/preservation.json)
confirms all 950 protected files unchanged, the selected source hash unchanged,
and the native database still at 53 rows / 50 distinct slots (0–49). A compressed
input-hash inventory makes the preservation check inspectable without publishing
raw databases or private seeds. The machine-readable
[execution audit](../artifacts/campaign-v2/assessment-1024/execution-audit.json)
records counts, failures, checks and runtime. The original v1 assessment and
historical development evidence remain separate.
