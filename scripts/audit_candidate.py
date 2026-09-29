"""Development-only intervention and repeatability evidence for a native program."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.evaluation import run_episode, aggregate
from dreamer.provenance import evaluation_identity, record, sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--parameter-key", default="transition_weights")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = ("predictive", "frozen", "no_planning", "frozen_no_planning")
    record(out / "manifest.json", {
        "split": "development", "episodes": 64, "seeds": [10000, 10063],
        "program_sha256": sha256(args.program), "variants": list(variants),
        "parameter_key": args.parameter_key, "evaluation": evaluation_identity(),
        "audit_driver_sha256": sha256(__file__),
        "purpose": "Post-selection diagnostic; never a final held-out assessment"})
    data = {}
    for variant in variants:
        path = out / f"{variant}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        assert [r["seed"] for r in rows] == list(range(10000, 10000 + len(rows)))
        for seed in range(10000 + len(rows), 10064):
            row = run_episode(args.program, seed, variant, replay=True)
            trace = row.pop("trace")
            parameters = [f["model"].get("learning", {}).get(args.parameter_key) for f in trace]
            row["audit"] = {
                "trajectory_sha256": digest([{k: f[k] for k in ("world", "action", "next_enemies")} for f in trace]),
                "map_position_sha256": digest([{
                    "position": f["model"].get("position"),
                    "terrain": sorted(f["model"].get("terrain", []))} for f in trace]),
                "parameters_exported": bool(parameters) and all(p is not None for p in parameters),
                "parameters_constant": bool(parameters) and all(p == parameters[0] for p in parameters),
                "first_parameters": parameters[0] if parameters else None,
                "final_parameters": parameters[-1] if parameters else None}
            if row["error"]:
                raise RuntimeError(f"{variant} / {seed}: {row['error']}")
            with path.open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
            rows.append(row)
            if len(rows) % 16 == 0:
                print(variant, len(rows), "escapes", sum(r["reason"] == "escaped" for r in rows), flush=True)
        data[variant] = rows
    from report import paired, wilson
    pairs = list(zip(data["no_planning"], data["frozen_no_planning"]))
    matched = all(a["audit"]["trajectory_sha256"] == b["audit"]["trajectory_sha256"] for a, b in pairs)
    maps_match = all(a["audit"]["map_position_sha256"] == b["audit"]["map_position_sha256"] for a, b in pairs)
    summary = {
        "development_only": True, "program_sha256": sha256(args.program),
        "matched_fixed_risk_trajectories": matched,
        "matched_fixed_risk_maps_and_localization": maps_match,
        "agents": {}, "paired": {}}
    for variant, rows in data.items():
        summary["agents"][variant] = {
            **aggregate(rows)["public"],
            "escape_ci95": wilson(sum(r["reason"] == "escaped" for r in rows), len(rows)),
            "mean_updates": sum(r["learning"].get("updates", 0) for r in rows) / len(rows),
            "mean_parameter_updates": sum(r["learning"].get("parameter_updates", 0) for r in rows) / len(rows),
            "parameter_export_episodes": sum(r["audit"]["parameters_exported"] for r in rows),
            "parameter_change_episodes": sum(not r["audit"]["parameters_constant"] for r in rows)}
    for left, right in (("predictive", "frozen"), ("predictive", "no_planning"),
                        ("no_planning", "frozen_no_planning")):
        summary["paired"][f"{left} minus {right}"] = {
            key: paired(data[left], data[right], key)
            for key in ("escape", "task", "brier_near", "brier_threat")}
    record(out / "audit-summary.json", summary)
    if matched:
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 4))
        curve = []
        for variant, label in (("no_planning", "Learned"), ("frozen_no_planning", "Frozen prior")):
            points = []
            for bin_index in range(8):
                values = [r["bins"].get(str(bin_index), {}).get("brier", [0, 0]) for r in data[variant]]
                count = sum(v[1] for v in values)
                if count:
                    total = sum(v[0] for v in values)
                    points.append((bin_index * 25 + 12.5, total / count))
                    curve.append({"variant": variant, "start_step": bin_index * 25,
                                  "loss_sum": total, "targets": count,
                                  "episodes": sum(v[1] > 0 for v in values)})
            ax.plot(*zip(*points), marker="o", label=label)
        ax.set(xlabel="Episode step (later bins contain fewer surviving episodes)", ylabel="Near-cell Brier loss",
               title="Selected program: identical fixed-risk actions and observations")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "matched-learning.png", dpi=160)
        plt.close(fig)
        record(out / "matched-learning.json", curve)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
