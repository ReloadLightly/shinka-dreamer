"""Compact paired evidence, uncertainty, learning plots and native lineage."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
from dreamer.evaluation import aggregate
import numpy as np


def wilson(successes, n):
    z = 1.959964
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-radius), float(center+radius)]


def paired(a, b, metric):
    assert [r["seed"] for r in a] == [r["seed"] for r in b]
    rng = np.random.default_rng(382940)
    indices = rng.integers(0, len(a), size=(5000, len(a)))
    def estimates(rows):
        if metric.startswith("brier_"):
            values = np.array([r["stats"].get(metric, [0, 0]) for r in rows])
            samples = values[indices].sum(axis=1)
            estimates = np.divide(samples[:, 0], samples[:, 1], out=np.zeros(len(samples)), where=samples[:, 1] > 0)
            totals = values.sum(axis=0)
            return totals[0]/totals[1], estimates
        values = np.array([float(r["reason"] == "escaped") if metric == "escape" else r[metric] for r in rows])
        return values.mean(), values[indices].mean(axis=1)
    ma, ba = estimates(a)
    mb, bb = estimates(b)
    return {"difference": float(ma-mb), "ci95": list(map(float, np.quantile(ba-bb, [.025, .975])))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--extra-input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--campaign", default="results/campaign-v1")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for folder in filter(None, [args.input, args.extra_input]):
        for path in Path(folder).glob("*.jsonl"):
            data[path.stem] = [json.loads(line) for line in path.read_text().splitlines()]
    summary = {"protocol": json.loads(Path(args.input, "manifest.json").read_text()), "agents": {}, "paired": {}}
    compact = []
    for variant, rows in sorted(data.items()):
        values = aggregate(rows)["public"]
        values["escape_ci95"] = wilson(sum(r["reason"] == "escaped" for r in rows), len(rows))
        values["total_seconds"] = sum(r["seconds"] for r in rows)
        values["mean_predictive_updates"] = sum(r["learning"].get("updates", 0) for r in rows)/len(rows)
        summary["agents"][variant] = values
        for i, row in enumerate(rows):
            # Do not publish withheld seeds; paired IDs suffice for resampling.
            record = {"episode": i, "agent": variant, **{k: row[k] for k in ("reason", "steps", "keys", "door", "task", "model_score", "combined_score")}}
            for key in ("brier_near", "brier_audit", "brier_threat", "persistence_near", "base_near"):
                total, count = row["stats"].get(key, [0, 0])
                record[key+"_sum"], record[key+"_n"] = total, count
            record["updates"] = row["learning"].get("updates", 0)
            compact.append(record)
    for left, right in (("predictive", "memory"), ("predictive", "frozen"), ("predictive", "no_planning"), ("no_planning", "frozen_no_planning")):
        if left in data and right in data:
            summary["paired"][left+" minus "+right] = {metric: paired(data[left], data[right], metric)
                for metric in ("escape", "task", "brier_near", "brier_threat")}
    with (out / "episodes.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(compact[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(compact)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    order = [v for v in ("reactive", "memory", "predictive", "frozen", "no_planning") if v in data]
    names = {"reactive": "Reactive", "memory": "Memory + paths", "predictive": "Learn + predict", "frozen": "Frozen learning", "no_planning": "Fixed-risk planning"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    means = [summary["agents"][v]["escape"] for v in order]
    intervals = np.array([summary["agents"][v]["escape_ci95"] for v in order]).T
    colors = {"reactive": "#8a98a5", "memory": "#2a6976", "predictive": "#d77c3e", "frozen": "#bbad88", "no_planning": "#6c8b68"}
    axes[0].barh([names[v] for v in order], means, color=[colors[v] for v in order])
    axes[0].errorbar(means, range(len(order)), xerr=[np.array(means)-intervals[0], intervals[1]-means], fmt="none", color="#252d36", capsize=3)
    axes[0].set(xlim=(0, 1), xlabel="Escape fraction · 95% Wilson interval")
    axes[0].invert_yaxis()
    prediction_order = [v for v in order if v != "reactive"]
    axes[1].barh([names[v] for v in prediction_order], [summary["agents"][v]["brier_threat"] for v in prediction_order], color="#5a8194")
    axes[1].set(xlabel="Brier loss near a visible threat · lower is better")
    axes[1].invert_yaxis()
    fig.suptitle(f"ShinkaDreamer · {summary['protocol']['split']} · {len(next(iter(data.values())))} paired mazes", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "outcomes.png", dpi=160)
    plt.close(fig)

    if "frozen_no_planning" in data and "no_planning" in data:
        fig, ax = plt.subplots(figsize=(8.5, 4))
        curve = []
        for name, key, style in [("Learned", "no_planning", "-o"), ("Frozen prior", "frozen_no_planning", "-s")]:
            xs, ys, counts = [], [], []
            for b in range(8):
                acc = np.array([r["bins"].get(str(b), {}).get("brier", [0, 0]) for r in data[key]]).sum(axis=0)
                if acc[1]:
                    xs.append(b*25+12.5); ys.append(acc[0]/acc[1]); counts.append(int(acc[1]//9))
                    curve.append({"agent": key, "start_step": b*25, "end_step_exclusive": (b+1)*25,
                                  "loss_sum": float(acc[0]), "target_count": int(acc[1]),
                                  "episodes_present": sum(str(b) in r["bins"] for r in data[key])})
            ax.plot(xs, ys, style, label=name)
        ax.set(xlabel="Episode step (25-step bins; later bins have fewer surviving episodes)", ylabel="Near-cell Brier loss", title="Matched experience: same fixed-risk actions, learned vs frozen forecasts")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "learning.png", dpi=160)
        (out / "learning-curve.json").write_text(json.dumps(curve, indent=2))
        plt.close(fig)

    db_path = Path(args.campaign, "programs.sqlite")
    if db_path.exists():
        conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        columns = {r[1] for r in conn.execute("pragma table_info(programs)")}
        fields = [k for k in ("id", "parent_id", "generation", "island_idx", "combined_score", "correct", "timestamp") if k in columns]
        lineage = [dict(row) for row in conn.execute("select " + ",".join(fields) + " from programs order by generation")]
        (out / "lineage.json").write_text(json.dumps(lineage, indent=2))
        conn.close()
    print(json.dumps({"agents": {v: {k:a[k] for k in ('escape','brier_near','mean_predictive_updates')} for v,a in summary['agents'].items()}, "paired": summary["paired"]}, indent=2))


if __name__ == "__main__":
    main()
