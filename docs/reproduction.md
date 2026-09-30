# Reproduction and saved artifacts

## Recompute the published analysis

A fresh checkout contains the episode CSVs, summaries, selected and ancestor
programs, intervention checks and representative replay. With Python, NumPy
and Matplotlib installed, run from the repository root:

```bash
python scripts/research_figure.py
```

This recomputes paired intervals and score decomposition from committed records,
checks them against the published summaries, and writes
[`research-review/`](../artifacts/campaign-v2/research-review/).
It runs no candidate episodes, makes no model calls and requires no login.
The generated `analysis.json` records input hashes and interpretation limits.

## Repeat the development intervention experiment

Candidate execution uses Linux, Python 3.10, Landlock and libseccomp.
The published runtime identity is in the
[audit manifest](../artifacts/campaign-v2/mechanism-gen14/manifest.json).
The evaluator itself needs no network, GPU or model subscription.

For the full project environment, `bash scripts/bootstrap.sh` installs the
Python requirements and pinned Headless tooling into repository-local
directories. That installation requires network access and Node 22+ for
Headless. An authorized subscription login is needed only for evolutionary
model calls, not for evaluation of saved programs.

With that environment installed, the following command repeats the four
generation-14 interventions on the 64 development mazes:

```bash
.venv/bin/python scripts/audit_candidate.py \
  --program artifacts/campaign-v2/completed-50/selected.py \
  --out results/development-audit-repeat
```

Use a new output directory for a fresh repeat; the audit script resumes rows
already present in its output directory. It emits progress every 16 episodes
per condition, paired summaries, parameter checks and trajectory hashes.
The condition name `no_planning` in saved data means **fixed-risk planning**:
pathfinding and lookahead still operate, but learned forecasts no longer
provide their risk estimates.

## Export or recover an existing local campaign

The following commands require the original local `results/campaign-v2`
database and generation directories. Those raw run files are not included in a
fresh GitHub checkout. The report also uses the original local control results
under `results/campaign-v2-controls`.

```bash
.venv/bin/python scripts/campaign_report.py --out results/report-50
```

The completed campaign is already exported under
[`completed-50/`](../artifacts/campaign-v2/completed-50/). For a saved copy that
was interrupted before slot 49, the historical recovery entry point is:

```bash
HEADLESS_BILLING=subscription .venv/bin/python scripts/recover_campaign.py \
  --results results/campaign-v2 --generations 50
```

This is a recovery command for the existing campaign, not a new experiment.
The original controller completed slots 0–49. Recovery does not add slots after
that stop. See [execution history](execution-history.md) for preserved failures,
upstream versions and the subscription route.

Raw databases, private seeds, backups, credentials and runtime caches remain
outside Git. No fresh final assessment of the evolved agent has been reserved
or run. The [protocol](protocol.md), [design](design.md),
[current task](../CODEX_TASK.md) and [working instructions](../AGENTS.md)
record the experimental contract and current scope.
