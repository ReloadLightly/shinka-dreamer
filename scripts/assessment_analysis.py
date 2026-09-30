"""Analyze the fixed assessment; reproduce statistics from committed compact rows."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import beta, binomtest
from dreamer.provenance import record, sha256
from assess_selected import CONDITIONS

PAIRS = (("predictive", "frozen"), ("predictive", "memory"),
         ("predictive", "original_predictive"), ("predictive", "no_planning"))
METRICS = ("brier_near", "brier_audit", "brier_threat")


def cp_interval(k, n, alpha=.05):
    if n == 0:
        return [0., 1.]
    return [0. if k == 0 else float(beta.ppf(alpha/2, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-alpha/2, k+1, n-k))]


def paired_exact_interval(wins, losses, n, alpha=.05):
    q = cp_interval(wins+losses, n, alpha/2)
    theta = cp_interval(wins, wins+losses, alpha/2)
    corners = [discordance * (2*direction-1) for discordance in q for direction in theta]
    return [min(corners), max(corners)]


def event(row, name):
    return (row["reason"] == {"escape": "escaped", "death": "caught", "timeout": "timeout"}[name]
            if name in ("escape", "death", "timeout") else
            row["keys"] == 2 if name == "two_keys" else bool(row["door"]))


def binary_pair(left, right, name="escape"):
    a = np.array([event(r, name) for r in left], dtype=bool)
    b = np.array([event(r, name) for r in right], dtype=bool)
    wins, losses = int((a & ~b).sum()), int((~a & b).sum())
    m, n = wins+losses, len(a)
    return {"left_only": wins, "right_only": losses, "both": int((a & b).sum()),
            "neither": int((~a & ~b).sum()), "discordant": m,
            "favorable_wins": losses if name in ("death", "timeout") else wins,
            "favorable_losses": wins if name in ("death", "timeout") else losses,
            "difference": (wins-losses)/n,
            "ci95": paired_exact_interval(wins, losses, n),
            "secondary_simultaneous_ci": paired_exact_interval(wins, losses, n, .05/3),
            "mcnemar_exact_p": float(binomtest(wins, m, .5).pvalue) if m else 1.}


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted, running = [None]*len(values), 0.
    for rank, i in enumerate(order):
        running = max(running, (len(values)-rank)*values[i])
        adjusted[i] = min(1., running)
    return adjusted


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {"ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            "bootstrap_retained": len(finite)}


def boot_ratio(weights, totals, counts):
    denominator = weights @ counts
    numerator = weights @ totals
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def sums(rows, metric, bin_index=None):
    if bin_index is None:
        pairs = [r["stats"].get(metric, [0., 0]) for r in rows]
    else:
        pairs = [r["bins"].get(str(bin_index), {}).get("brier", [0., 0]) for r in rows]
    array = np.asarray(pairs, dtype=float)
    return array[:, 0], array[:, 1]


def pooled_pair(left, right, metric, weights, bin_index=None):
    ls, ln = sums(left, metric, bin_index)
    rs, rn = sums(right, metric, bin_index)
    if not ln.sum() or not rn.sum():
        return None
    l, r = float(ls.sum()/ln.sum()), float(rs.sum()/rn.sum())
    bl, br = boot_ratio(weights, ls, ln), boot_ratio(weights, rs, rn)
    relative = np.divide(br-bl, br, out=np.full(len(br), np.nan), where=br != 0)
    return {"left_loss": l, "right_loss": r, "difference": l-r,
            "left_loss_sum": float(ls.sum()), "right_loss_sum": float(rs.sum()),
            "relative_reduction": 1-l/r if r else None,
            **interval(bl-br), "relative_reduction_interval": interval(relative),
            "left_interval": interval(bl), "right_interval": interval(br),
            "left_targets": int(ln.sum()), "right_targets": int(rn.sum()),
            "left_episodes": int((ln > 0).sum()), "right_episodes": int((rn > 0).sum())}


def export_rows(raw, out):
    completion = json.loads((raw / "execution-complete.json").read_text())
    assert completion["cases"] == 1024 and completion["condition_episodes"] == 6144
    paths = sorted((raw / "cases").glob("*.json"))
    assert len(paths) == 1024
    for p in paths:
        assert completion["case_file_sha256"][p.name] == sha256(p)
    rows = [r for p in paths for r in json.loads(p.read_text())["conditions"]]
    # No seeds or hidden world/model traces are published in normal episode data.
    assert all("seed" not in r and "trace" not in r for r in rows)
    with (out / "episodes.jsonl.gz").open("wb") as file:
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as text:
                for row in rows:
                    text.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
    simple = []
    for r in rows:
        row = {k: r[k] for k in ("case", "variant", "reason", "steps", "keys", "door", "task", "model_score", "combined_score", "seconds", "error")}
        for metric in METRICS:
            row[metric+"_sum"], row[metric+"_n"] = r["stats"].get(metric, [0., 0])
        row["parameter_updates"] = r["learning"].get("parameter_updates")
        simple.append(row)
    with (out / "episodes.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(simple[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(simple)
    for filename in ("manifest.json", "execution-complete.json"):
        shutil.copy2(raw / filename, out / filename)
    for filename in ("selection.json", "assessment-manifest.json"):
        shutil.copy2(ROOT / "results/campaign-v2" / filename, out / filename)
    return rows


def analyze(rows, plan):
    data = {v: sorted((r for r in rows if r["variant"] == v), key=lambda r: r["case"]) for v in CONDITIONS}
    assert all([r["case"] for r in group] == list(range(1024)) for group in data.values())
    n = 1024
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    weights = rng.multinomial(n, np.full(n, 1/n), size=plan["bootstrap"]["replicates"]).astype(float)
    summary = {"study": plan["study"], "cases": n, "condition_episodes": len(rows),
               "split": "fresh assessment", "primary": "predictive minus frozen escape",
               "agents": {}, "contrasts": {}, "interventions": {},
               "interval_method": plan["binary_reporting"]["paired_interval"],
               "multiplicity": plan["multiplicity"], "bootstrap": plan["bootstrap"]}
    for variant, group in data.items():
        result = {"episodes": n, "invalid": sum(r["error"] is not None for r in group)}
        for name in ("escape", "death", "timeout", "door", "two_keys"):
            count = sum(event(r, name) for r in group)
            ci = binomtest(count, n).proportion_ci(method="wilson")
            result[name] = {"count": count, "rate": count/n, "ci95": [float(ci.low), float(ci.high)]}
        for name in ("task", "keys", "model_score", "combined_score", "steps", "seconds"):
            values = np.asarray([r[name] for r in group])
            result[name] = {"mean": float(values.mean()), **interval(weights @ values/n)}
        for name in ("steps", "seconds"):
            counts = np.asarray([r["reason"] == "escaped" for r in group], dtype=float)
            totals = np.asarray([r[name] for r in group])*counts
            result["successful_escape_"+name] = {"mean": float(totals.sum()/counts.sum()) if counts.sum() else None,
                                                  "episodes": int(counts.sum()), **interval(boot_ratio(weights, totals, counts))}
        result["runtime_total_seconds"] = sum(r["seconds"] for r in group)
        result["runtime_median_seconds"] = float(np.median([r["seconds"] for r in group]))
        result["runtime_p95_seconds"] = float(np.quantile([r["seconds"] for r in group], .95))
        for metric in sorted({k for r in group for k in r["stats"]}):
            totals, counts = sums(group, metric)
            result[metric] = {"value": float(totals.sum()/counts.sum()) if counts.sum() else None,
                              "targets": int(counts.sum()), **interval(boot_ratio(weights, totals, counts))}
        audits = [r["audit"] for r in group]
        summary["interventions"][variant] = {
            "parameters_exported_cases": sum(a["parameters_exported"] for a in audits),
            "parameters_constant_cases": sum(a["parameters_exported"] and a["parameters_constant"] for a in audits),
            "parameters_changed_cases": sum(a["parameters_exported"] and not a["parameters_constant"] for a in audits),
            "initial_parameter_vectors": sorted({json.dumps(a["first_parameters"]) for a in audits}),
            "localization_errors": sum(a["localization_errors"] for a in audits),
            "visible_terrain_errors": sum(a["visible_terrain_errors"] for a in audits),
            "visible_terrain_checks": sum(a["visible_terrain_checks"] for a in audits),
            "mapping_changed_cases": sum(a["distinct_terrain_states"] > 1 for a in audits),
            "localized_movement_cases": sum(a["distinct_positions"] > 1 for a in audits),
            "mean_label_updates": sum(r["learning"].get("updates", 0) for r in group)/n,
            "mean_parameter_update_steps": sum(r["learning"].get("parameter_updates", 0) for r in group)/n}
        summary["agents"][variant] = result
    for left, right in PAIRS:
        a, b = data[left], data[right]
        result = {"role": "primary" if right == "frozen" else "secondary",
                  "binary": {name: binary_pair(a, b, name) for name in ("escape", "death", "timeout", "door", "two_keys")},
                  "on_policy_prediction": {metric: pooled_pair(a, b, metric, weights) for metric in METRICS},
                  "continuous": {}}
        result["escape_discordances_involving_invalid"] = [x["case"] for x, y in zip(a, b)
            if event(x, "escape") != event(y, "escape") and (x["error"] or y["error"])]
        for name in ("task", "keys", "model_score", "combined_score", "steps", "seconds"):
            delta = np.asarray([x[name]-y[name] for x, y in zip(a, b)])
            result["continuous"][name] = {"difference": float(delta.mean()), **interval(weights @ delta/n)}
        summary["contrasts"][left+" minus "+right] = result
    secondary = [summary["contrasts"][a+" minus "+b]["binary"]["escape"] for a, b in PAIRS[1:]]
    for result, adjusted in zip(secondary, holm([r["mcnemar_exact_p"] for r in secondary])):
        result["holm_adjusted_p"] = adjusted
    a, b = data["no_planning"], data["frozen_no_planning"]
    trajectory_bad = [x["case"] for x, y in zip(a, b) if x["audit"]["trajectory_sha256"] != y["audit"]["trajectory_sha256"]]
    map_bad = [x["case"] for x, y in zip(a, b) if x["audit"]["map_position_sha256"] != y["audit"]["map_position_sha256"]]
    matched = not trajectory_bad and not map_bad
    summary["matched_prediction"] = {"verified": matched, "trajectory_mismatch_cases": trajectory_bad,
        "map_localization_mismatch_cases": map_bad,
        "binary": {name: binary_pair(a, b, name) for name in ("escape", "death", "timeout", "door", "two_keys")},
        "metrics": {metric: pooled_pair(a, b, metric, weights) for metric in METRICS},
        "interpretation": "Matched experience" if matched else "Trajectory checks failed: these are on-policy differences, not a matched learning effect"}
    ls, ln = sums(a, "brier_near")
    rs, rn = sums(b, "brier_near")
    episode_delta = ls/ln - rs/rn
    summary["matched_prediction"]["episode_near_loss_directions"] = {
        "improved": int((episode_delta < -1e-12).sum()),
        "tied": int((np.abs(episode_delta) <= 1e-12).sum()),
        "worsened": int((episode_delta > 1e-12).sum())}
    curve = []
    if matched:
        for i in range(8):
            result = pooled_pair(a, b, "brier_near", weights, i)
            if result is not None:
                curve.append({"start_step": i*25, "end_step": (i+1)*25-1, **result})
    summary["learning_curve"] = curve
    # The rule is fixed in the preregistration; seeds are not accessed here.
    strata = {"selected_only": [], "frozen_only": [], "both_fail": [], "both_escape": []}
    for x, y in zip(data["predictive"], data["frozen"]):
        key = ("both_escape" if event(y, "escape") else "selected_only") if event(x, "escape") else ("frozen_only" if event(y, "escape") else "both_fail")
        strata[key].append(x["case"])
    summary["example_selection"] = {k: {"eligible_cases": len(v), "case": v[0] if v else None} for k, v in strata.items()}
    summary["invalid_episodes"] = [{"case": r["case"], "variant": r["variant"], "error": r["error"]} for r in rows if r["error"]]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", help="Completed private assessment directory; omitted to reproduce from committed data")
    parser.add_argument("--data", default="artifacts/campaign-v2/assessment-1024")
    args = parser.parse_args()
    out = Path(args.data)
    plan = json.loads((out / "preregistration.json").read_text())
    if args.raw:
        if (out / "analysis-closed.json").exists():
            raise ValueError("Analysis already closed; reproduce from committed data without --raw")
        export_rows(Path(args.raw), out)
    with gzip.open(out / "episodes.jsonl.gz", "rt") as file:
        rows = [json.loads(line) for line in file]
    result = analyze(rows, plan)
    payload = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if not (out / "analysis-closed.json").exists():
        (out / "analysis.json").write_text(payload)
        record(out / "analysis-closed.json", {"closed_utc": datetime.now(timezone.utc).isoformat(),
               "preregistration_sha256": sha256(out / "preregistration.json"),
               "episodes_sha256": sha256(out / "episodes.jsonl.gz"),
               "analysis_sha256": sha256(out / "analysis.json"), "analysis_script_sha256": sha256(__file__),
               "cases": 1024, "condition_episodes": 6144,
               "example_case_indices": result["example_selection"]})
    else:
        closed = json.loads((out / "analysis-closed.json").read_text())
        assert closed["episodes_sha256"] == sha256(out / "episodes.jsonl.gz")
        assert closed["analysis_sha256"] == sha256(out / "analysis.json")
        assert closed["analysis_sha256"] == hashlib.sha256(payload.encode()).hexdigest(), "Recomputed analysis differs; original preserved"
    print(json.dumps({k: result[k] for k in ("cases", "condition_episodes", "contrasts", "matched_prediction", "example_selection")}, indent=2))


if __name__ == "__main__":
    main()
