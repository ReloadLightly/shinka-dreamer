"""Export compact native evolution evidence without reading assessment cases."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
from dreamer.provenance import sha256
from report import paired, wilson


def load_episodes(campaign, row):
    metadata = json.loads(row["metadata"])
    directory = Path(metadata.get("recovery_results_dir", campaign / f"gen_{row['generation']}/results"))
    path = directory / "episodes.json"
    return json.loads(path.read_text()) if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="results/campaign-v2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--slots", type=int, default=50)
    args = parser.parse_args()
    campaign, out = Path(args.campaign).resolve(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{campaign / 'programs.sqlite'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("select * from programs order by generation,timestamp,id")]
    generations = sorted({r["generation"] for r in rows})
    native_log = (campaign / "evolution_run.log").read_text().splitlines()
    unique = [next(r for r in rows if r["generation"] == g) for g in generations]
    best = max((r for r in unique if r["correct"]), key=lambda r: r["combined_score"])
    lineage, metrics, episode_rows, failures = [], [], [], []
    for row in rows:
        metadata, public = json.loads(row["metadata"]), json.loads(row["public_metrics"])
        lineage.append({
            **{k: row[k] for k in ("id", "parent_id", "generation", "island_idx", "correct", "combined_score")},
            "source_sha256": hashlib.sha256(row["code"].encode()).hexdigest(),
            "archive_inspiration_ids": json.loads(row["archive_inspiration_ids"] or "[]"),
            "top_k_inspiration_ids": json.loads(row["top_k_inspiration_ids"] or "[]"),
            "migration_history": json.loads(row["migration_history"] or "[]"),
            "patch_name": metadata.get("patch_name"), "patch_type": metadata.get("patch_type"),
            "patch_description": metadata.get("patch_description"),
            "model": metadata.get("model_name"), "evaluation_seconds": metadata.get("evaluation_seconds"),
            "failure_stage": metadata.get("failure_stage"), "failure_reason": metadata.get("failure_reason"),
            "metrics": public})
    for row in unique:
        public = json.loads(row["public_metrics"])
        episodes = load_episodes(campaign, row)
        if not row["correct"]:
            metadata = json.loads(row["metadata"])
            directory = Path(metadata.get("recovery_results_dir", campaign / f"gen_{row['generation']}/results"))
            correct_path = directory / "correct.json"
            failures.append({
                "generation": row["generation"],
                "evaluator_status": json.loads(correct_path.read_text()) if correct_path.exists() else
                    {"correct": False, "error": "No saved evaluator correctness file; see native timeout evidence when available"},
                "saved_episodes": len(episodes),
                "native_timeout_evidence": [line for line in native_log
                                            if "exceeded timeout" in line and line.rstrip().endswith(f"=> Gen. {row['generation']}")],
                "invalid_episodes": [{"episode": i, "steps": e["steps"], "error": e["error"], "seconds": e["seconds"]}
                                     for i, e in enumerate(episodes) if e["error"]]})
        metrics.append({"generation": row["generation"], "correct": bool(row["correct"]),
                        "combined_score": row["combined_score"], "saved_episode_count": len(episodes), **public})
        for index, episode in enumerate(episodes):
            record = {"generation": row["generation"], "episode": index,
                      **{k: episode[k] for k in ("reason", "steps", "keys", "door", "task", "model_score", "combined_score")}}
            for key in ("brier_near", "brier_audit", "brier_threat"):
                record[key + "_sum"], record[key + "_n"] = episode["stats"].get(key, (0, 0))
            episode_rows.append(record)
    with (out / "episodes.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(episode_rows)
    comparisons = {"selected": load_episodes(campaign, best)}
    for name, filename in (("memory", "memory"), ("original_predictive", "original_predictive")):
        path = ROOT / f"results/campaign-v2-controls/{filename}.jsonl"
        comparisons[name] = [json.loads(line) for line in path.read_text().splitlines()]
    from dreamer.evaluation import aggregate
    summary = {
        "split": "development", "selection_warning": "64 reused search cases; intervals do not correct selection bias",
        "target_slots": args.slots, "persisted_slots": len(unique), "database_rows": len(rows),
        "complete": generations == list(range(args.slots)),
        "seed_slots": 1, "seed_island_rows": sum(r["generation"] == 0 for r in rows),
        "descendant_slots": sum(r["generation"] > 0 for r in unique),
        "descendants_with_generated_code": sum(r["generation"] > 0 and bool(r["code"].strip()) for r in unique),
        "valid_descendants": sum(r["generation"] > 0 and r["correct"] for r in unique),
        "failed_slots": [r["generation"] for r in unique if not r["correct"]],
        "failure_details": failures,
        "unique_source_hashes": len({r["source_sha256"] for r in lineage}),
        "candidates_with_saved_64_episode_evaluations": sum(m["saved_episode_count"] == 64 for m in metrics),
        "saved_native_condition_episodes": len(episode_rows),
        "best": {k: best[k] for k in ("id", "generation", "combined_score")},
        "agents": {}, "paired": {},
        "billing": {"mutation": "subscription", "recommendations": "subscription",
                    "embeddings": None, "novelty_llm": None, "evaluator": "local Python, no LLM",
                    "model_bandit": "disabled; one configuration"},
        "assessment": "Not run or reserved; final assessment left untouched",
        "campaign_sha256": sha256(campaign / "campaign-manifest.json"),
        "generation_stop_records": [json.loads(p.read_text()) for p in sorted(campaign.glob("execution-*.json"))]}
    for name, episodes in comparisons.items():
        summary["agents"][name] = {
            **aggregate(episodes)["public"],
            "escapes": sum(r["reason"] == "escaped" for r in episodes),
            "escape_ci95": wilson(sum(r["reason"] == "escaped" for r in episodes), len(episodes))}
    for name in ("memory", "original_predictive"):
        summary["paired"][f"selected minus {name}"] = {
            key: paired(comparisons["selected"], comparisons[name], key)
            for key in ("escape", "task", "brier_near", "brier_threat")}
    (out / "selected.py").write_text(best["code"])
    summary["best"]["source_sha256"] = sha256(out / "selected.py")
    by_id = {r["id"]: r for r in rows}
    ancestor = best
    ancestry = []
    source_dir = out / "lineage-programs"
    source_dir.mkdir(exist_ok=True)
    while ancestor is not None:
        source_path = source_dir / f"gen_{ancestor['generation']}.py"
        source_path.write_text(ancestor["code"])
        ancestry.append({"id": ancestor["id"], "generation": ancestor["generation"],
                         "source_sha256": sha256(source_path)})
        ancestor = by_id.get(ancestor["parent_id"])
    summary["best"]["parent_chain"] = ancestry[::-1]
    for filename, data in (("summary.json", summary), ("lineage.json", lineage), ("generation-metrics.json", metrics)):
        (out / filename).write_text(json.dumps(data, indent=2) + "\n")
    for name in ("campaign-manifest.json", "dreamer-resolved.json", "recovery-jobs.json"):
        shutil.copy2(campaign / name, out / name)
    shutil.copytree(campaign / "meta", out / "recommendations", dirs_exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    valid = [m for m in metrics if m["correct"]]
    x = [m["generation"] for m in valid]
    scores = [m["combined_score"] for m in valid]
    axes[0].scatter(x, scores, color="#38788a", s=23, alpha=.75, label="Candidate")
    axes[0].step(x, np.maximum.accumulate(scores), where="post", color="#bf6d36", label="Best so far")
    axes[0].set(ylabel="Selection score", ylim=(min(scores)-.015, 1))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].scatter(x, [m["escape"] for m in valid], color="#38788a", s=23)
    axes[1].axhline(summary["agents"]["memory"]["escape"], color="#bf6d36", linestyle="--", label="Original memory baseline")
    axes[1].set(ylabel="Escape fraction", ylim=(.65, 1.02))
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(x, [m["brier_near"] for m in valid], "o", color="#38788a", markersize=4, label="Near cells")
    axes[2].plot(x, [m["brier_threat"] for m in valid], "o", color="#bf6d36", markersize=4, label="Threat-conditioned")
    axes[2].set(ylabel="On-policy Brier loss (lower is better)")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set(xlabel="Generation slot (seed = 0)", xlim=(-1, args.slots))
        for failed in summary["failed_slots"]:
            ax.axvline(failed, color="#a0a0a0", alpha=.4, linewidth=1)
    fig.suptitle(f"ShinkaDreamer · {len(unique)} native slots · 64 reused development mazes", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "evolution.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("persisted_slots", "valid_descendants", "failed_slots", "best", "paired")}, indent=2))


if __name__ == "__main__":
    main()
