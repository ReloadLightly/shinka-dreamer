"""Saved-data scientific export for RUN1; never imports or executes a candidate.

Native source/lineage/model-call archives belong to the separate native exporter.
This script publishes allowlisted episode data, task/forecast/cost summaries,
resource-aligned progress, structural source changes and descriptive paired
development comparisons. Private pool values, laws and observations stay private.
"""
import argparse
import ast
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
FORECASTS = ("brier_near", "brier_audit", "brier_destination", "brier_threat")
ENTRYPOINTS = ("world_model_step", "planner", "export_model")
SCOPES = {
    "task": "One selected development panel, repeatedly exposed to evolutionary selection; no held-out assessment or independent-search replication.",
    "prediction": "On-policy forecast losses; actions and target exposure differ between programs. This is not matched-experience prediction-learning evidence.",
    "control": "No selected-program freeze or prediction-use intervention was assessed in this RUN1 campaign. Predictive-adaptation control benefit remains untested.",
    "bootstrap": "1000 percentile bootstrap draws, fixed seed4137, resampling all8 layout cases with all6 conditions and both programs kept together. Nominal descriptive uncertainty; it does not remove winner-selection bias.",
    "cpu": "Candidate and evaluator episode-process CPU sums are disjoint; native host evaluation elapsed and summed episode wall time overlap and must not be added. Uncheckpointed work can be unavailable.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha(code):
    return hashlib.sha256(code.encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decoded(value, default):
    return json.loads(value) if isinstance(value, str) and value else (value if value is not None else default)


def epoch(value):
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() if value else None


def database_rows(database):
    if not database.exists():
        return []
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("select * from programs order by generation,timestamp,id")]
    for row in rows:
        row["metadata"] = decoded(row.get("metadata"), {})
        row["public_metrics"] = decoded(row.get("public_metrics"), {})
    return rows


def canonical_slots(rows):
    result = {}
    for row in rows:
        generation = int(row["generation"])
        previous = result.get(generation)
        if previous is None or (previous["metadata"].get("_is_island_copy") and not row["metadata"].get("_is_island_copy")):
            result[generation] = row
        elif generation != 0 and source_sha(previous["code"]) != source_sha(row["code"]):
            raise ValueError(f"Different native sources share slot {generation}")
    return result


def episode_rows(folder, case_map, conditions, expected_source, expected_evaluation):
    manifest_path = folder / "evaluation-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["program_sha256"] != expected_source or manifest["evaluation"] != expected_evaluation:
            raise ValueError("Episode evaluation/source identity changed")
    saved, inputs = {}, {}
    files = sorted((folder / "episode-checkpoints").glob("*.json"))
    if (folder / "episodes.json").exists():
        files.append(folder / "episodes.json")
    for path in files:
        raw = json.loads(path.read_text())
        inputs[str(path)] = sha(path)
        for row in raw if isinstance(raw, list) else [raw]:
            if row["seed"] not in case_map or row["regime"] not in conditions:
                raise ValueError("Episode outside registered development panel")
            key = case_map[row["seed"]], row["regime"]
            if key in saved and saved[key] != row:
                raise ValueError("Conflicting aggregate/checkpoint episode; preserve repeated execution separately")
            saved[key] = row
    return saved, inputs


def compact_episode(row, generation, case, source):
    # Allowlisting intentionally excludes raw candidate learning representations,
    # true laws, seed values, switch times, snapshots and observation traces.
    error_type = str(row.get("error", "")).split(":", 1)[0]
    if not error_type.isidentifier() or not error_type.endswith(("Error", "Exception")):
        error_type = "unclassified"
    result = {"generation": generation, "case": case, "condition": row["regime"],
              "program_sha256": source, "reason": row["reason"], "invalid": row.get("error") is not None,
              "error_type": error_type if row.get("error") else None,
              "error_sha256": source_sha(str(row["error"])) if row.get("error") else None,
              "task": row["combined_score"], "steps": row["steps"], "keys": row["keys"],
              "door": bool(row["door"]), "stats": row.get("stats", {}), "bins": row.get("bins", {}),
              "episode_wall_seconds": row.get("seconds"),
              "candidate_cpu_seconds": row.get("candidate_cpu_seconds"),
              "evaluator_cpu_seconds": row.get("evaluator_cpu_seconds"),
              "encountered_switch": bool(row.get("encountered_switch")),
              "replacements_encountered": sum(e.get("steps", 0) > 0 for e in row.get("switch_exposure", [])),
              "enemy_innovation_draws": row.get("enemy_innovation_draws"),
              "exposure": row.get("exposure", {}),
              "enemy_visibility_radius": row.get("enemy_visibility_radius")}
    learning = row.get("learning", {})
    result["candidate_reported_update_counts"] = {key: learning[key] for key in
        ("updates", "parameter_updates", "effective_updates")
        if isinstance(learning.get(key), (int, float)) and math.isfinite(learning[key])}
    return result


def pooled(rows, name):
    total = sum(row.get("stats", {}).get(name, [0., 0])[0] for row in rows)
    count = sum(row.get("stats", {}).get(name, [0., 0])[1] for row in rows)
    return total / count if count else None


def metrics(rows, expected):
    n = len(rows)
    if expected <= 0 or n > expected:
        raise ValueError("Invalid episode denominator")
    result = {"expected_episodes": expected, "recorded_episodes": n, "missing_episodes": expected - n,
              "task": sum(r["task"] for r in rows) / expected if n == expected else None,
              "task_lower_bound": sum(r["task"] for r in rows) / expected,
              "task_upper_bound": (sum(r["task"] for r in rows) + expected - n) / expected,
              "transitions_recorded": sum(r["steps"] for r in rows)}
    for event, reason in (("escape", "escaped"), ("death", "caught"), ("timeout", "timeout"), ("invalid", "invalid")):
        count = sum((r["invalid"] if event == "invalid" else r["reason"] == reason) for r in rows)
        result[event + "_count"] = count
        result[event + "_rate"] = count / expected if n == expected else None
        result[event + "_rate_lower_bound"] = count / expected
    for stat in FORECASTS:
        result[stat] = pooled(rows, stat)
        result[stat + "_targets"] = sum(r["stats"].get(stat, [0, 0])[1] for r in rows)
        result[stat + "_contributing_episodes"] = sum(r["stats"].get(stat, [0, 0])[1] > 0 for r in rows)
    for field in ("candidate_cpu_seconds", "evaluator_cpu_seconds", "episode_wall_seconds"):
        measured = [r[field] for r in rows if isinstance(r.get(field), (int, float)) and math.isfinite(r[field])]
        result[field + "_sum"] = sum(measured)
        result[field + "_measured_episodes"] = len(measured)
        result[field + "_unavailable_recorded_episodes"] = n - len(measured)
    result.update(encountered_switch_episodes=sum(r["encountered_switch"] for r in rows),
                  replacements_encountered=sum(r["replacements_encountered"] for r in rows),
                  post_switch_steps=sum(r["exposure"].get("post_switch_steps", 0) for r in rows),
                  post_switch_visible_enemy_steps=sum(r["exposure"].get("post_switch_visible_enemy_steps", 0) for r in rows))
    return result


class RemoveDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node


def ast_summary(code):
    try:
        tree = RemoveDocstrings().visit(ast.parse(code))
    except (SyntaxError, ValueError) as exc:
        return {"parse_error": type(exc).__name__, "source_sha256": source_sha(code)}, {}
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    named_calls = {name: sorted({node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)} & set(functions))
                   for name, function in functions.items()}
    reachable, queue = set(), [name for name in ENTRYPOINTS if name in functions]
    while queue:
        name = queue.pop()
        if name in reachable:
            continue
        reachable.add(name)
        queue.extend(named_calls[name])
    dumps = {name: ast.dump(function, include_attributes=False) for name, function in functions.items()}
    literals = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return {"source_sha256": source_sha(code), "source_lines": len(code.splitlines()),
            "ast_nodes": sum(1 for _ in ast.walk(tree)), "top_level_functions": sorted(functions),
            "direct_named_call_graph": named_calls,
            "functions_without_direct_named_path_from_entrypoints": sorted(set(functions) - reachable),
            "static_limit": "Named-call reachability is only a source-review hint; dynamic dispatch and callbacks may be missed. Structural change is not proof of predictive adaptation.",
            "observation_flags_referenced": [key for key in ("learn", "predictive_planning", "enemy_visibility_radius") if key in literals],
            "predictive_mechanism_status": "Not behaviorally verified in RUN1; exported update counters or flags alone are insufficient."}, dumps


def program_changes(row, by_id, manual_notes):
    result, functions = ast_summary(row["code"])
    parent = by_id.get(row.get("parent_id"))
    prior = ast_summary(parent["code"])[1] if parent else {}
    result.update(generation=row["generation"], native_id=row["id"], parent_id=row.get("parent_id"),
                  public_source=f"native/programs/generation-{row['generation']:03d}.py",
                  functions_added_vs_parent=sorted(set(functions) - set(prior)) if parent else [],
                  functions_removed_vs_parent=sorted(set(prior) - set(functions)),
                  functions_changed_vs_parent=sorted(name for name in set(functions) & set(prior) if functions[name] != prior[name]),
                  entrypoints_changed_vs_parent=[name for name in ENTRYPOINTS if name in set(functions) & set(prior) and functions[name] != prior[name]])
    note = manual_notes.get(result["source_sha256"])
    if note:
        result["source_review"] = note
    elif row["generation"] == 0:
        result["source_review"] = {"scope": "Static source inspection of immutable historical seed, not a new causal assessment.",
            "predictor": "Updates beta-like occupancy frequencies for blocked/occupied/neighbor/distant features from consecutive observations; exports per-cell occupancy heuristics.",
            "planner": "Risk-weighted graph search to keys, door, exit or exploration frontier; may wait if immediate risk difference exceeds a threshold.",
            "new_sensor_limitation": "Original seed does not read enemy_visibility_radius; outer-ring absence in late-sensor conditions can therefore be misinterpreted. No manual compatibility architecture was inserted."}
    return result


def call_snapshot(calls, cutoff):
    completed = [call for call in calls if epoch(call.get("ended_utc")) is not None and epoch(call["ended_utc"]) <= cutoff]
    usages = [c["usage"] for c in completed if isinstance(c.get("usage"), dict)]
    return {"completed_calls": len(completed),
            "uncached_input_plus_output_tokens": sum(u.get("inputTokens", 0) + u.get("outputTokens", 0) for u in usages),
            "cached_input_tokens": sum(u.get("cacheReadTokens", 0) for u in usages),
            "remote_elapsed_seconds": sum(c.get("elapsed_seconds", 0.) for c in completed),
            "usage_unavailable_completed_calls": sum(not c.get("usage") or c["usage"].get("usageStatus") != "reported" for c in completed),
            "readiness_fixture_calls": sum(c.get("role") == "readiness" for c in completed)}


def percentiles(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]


def paired_comparison(episodes, seed_generation, best_generation, conditions, cases=8):
    lookup = {(r["generation"], r["case"], r["condition"]): r for r in episodes}
    required = [(g, c, condition) for g in {seed_generation, best_generation} for c in range(cases) for condition in conditions]
    if any(key not in lookup for key in required):
        return {"available": False, "reason": "A required complete paired development panel is unavailable", "scope": SCOPES}
    rng = random.Random(4137)
    weights = [[rng.randrange(cases) for _ in range(cases)] for _ in range(1000)]
    groups = {}
    for name, subset in [("all", conditions)] + [(condition, [condition]) for condition in conditions]:
        def values(indices):
            rows = {g: [lookup[g, case, condition] for case in indices for condition in subset]
                    for g in {seed_generation, best_generation}}
            result = {}
            for metric, field, target in (("task", "task", None), ("escape", "reason", "escaped"),
                                          ("death", "reason", "caught"), ("timeout", "reason", "timeout"),
                                          ("invalid", "invalid", True), ("keys", "keys", None),
                                          ("door", "door", None), ("steps", "steps", None)):
                means = {g: sum((r[field] == target) if target is not None else r[field] for r in group) / len(group)
                         for g, group in rows.items()}
                result[metric] = (means[seed_generation], means[best_generation], means[best_generation] - means[seed_generation])
            for metric in FORECASTS:
                a, b = pooled(rows[seed_generation], metric), pooled(rows[best_generation], metric)
                result[metric] = (a, b, b - a if a is not None and b is not None else None)
            return result
        observed = values(range(cases))
        bootstrap = {key: [] for key in observed}
        for sampled in weights:
            for key, triple in values(sampled).items():
                bootstrap[key].append(triple[2])
        group = {"layout_cases": cases, "condition_episodes_per_program": cases * len(subset), "metrics": {}}
        for key, (a, b, diff) in observed.items():
            group["metrics"][key] = {"seed": a, "best": b, "difference_best_minus_seed": diff,
                                      "ci95_descriptive_layout_bootstrap": percentiles(bootstrap[key]),
                                      "bootstrap_draws_available": sum(x is not None for x in bootstrap[key])}
        groups[name] = group
    return {"available": True, "seed_generation": seed_generation, "best_generation": best_generation,
            "comparison_is_identity": seed_generation == best_generation,
            "cases": cases, "conditions": list(conditions), "bootstrap_draws": 1000, "bootstrap_seed": 4137,
            "groups": groups, "scope": SCOPES,
            "winner_selection_warning": "The winner was selected using these same development cases. Intervals describe observed case variability and are not confirmatory generalization or adaptation-control evidence."}


def generated_document(summary, candidates, paired, path):
    lines = ["# RUN1 saved development results", "", SCOPES["task"], "",
             f"Current export: **{summary['persisted_unique_slots']} persisted slots**, including the seed; "
             f"{summary['recorded_condition_episodes']} saved condition-episodes. "
             f"{summary['administrative_seed_copy_rows']} administrative seed-copy rows are excluded from slot counts.", "",
             "| Slot | Native valid | Recorded / expected | Mean task | Escape | Death | Timeout | Invalid | Missing |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for row in candidates:
        score = f"{row['task']:.5f}" if row["task"] is not None else "Incomplete"
        lines.append(f"| {row['generation']} | {row['native_correct']} | {row['recorded_episodes']}/{row['expected_episodes']} | {score} | {row['escape_count']} | {row['death_count']} | {row['timeout_count']} | {row['invalid_count']} | {row['missing_episodes']} |")
    lines += ["", "Native eligibility additionally requires every episode to validate. Failed candidates remain in the slot and evidence records; their episode means retain invalid executions with score zero. Missing episodes are unavailable, not silently counted as successes or observed failures.", ""]
    for row in candidates:
        if row.get("infrastructure_failure"):
            lines += [f"Slot{row['generation']} infrastructure finding: {row['infrastructure_failure']['interpretation']}", ""]
    if paired.get("available") and paired["best_generation"] == paired["seed_generation"]:
        lines += ["The seed remains the best eligible development program. The paired export is therefore an identity comparison, not evidence about a new program.", ""]
    elif paired.get("available"):
        best = paired["best_generation"]
        effect = paired["groups"]["all"]["metrics"]["escape"]
        lo, hi = effect["ci95_descriptive_layout_bootstrap"]
        lines += [f"The current eligible development winner is slot **{best}**. Its paired escape difference versus slot0 is "
                  f"{100*effect['difference_best_minus_seed']:+.2f} percentage points "
                  f"(descriptive layout-bootstrap interval {100*lo:+.2f} to {100*hi:+.2f}). "
                  "The same cases selected this winner; this interval does not establish out-of-sample improvement.", ""]
    lines += [SCOPES["prediction"], "", SCOPES["control"], "", SCOPES["bootstrap"], "", SCOPES["cpu"], "",
              "Program changes are reported from source hashes and structural AST comparisons. Named-call reachability flags possible unused helpers but does not prove inactivity or useful learning. Source-specific causal analysis remains separate.", "",
              "Reproduce this saved-data export without executing candidates, worlds or model calls:", "", "```bash",
              ".venv/bin/python scripts/run1_science_report.py", "```", "",
              "Compact data and the source-analysis record are under `artifacts/campaign-v4/run1/science`; the separate native export preserves all source branches and actual search mechanisms."]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="results/campaign-v4-run1")
    parser.add_argument("--out", default="artifacts/campaign-v4/run1/science")
    parser.add_argument("--notes", default="artifacts/campaign-v4/run1/program-review-notes.json")
    parser.add_argument("--infrastructure", default="artifacts/campaign-v4/run1/native/infrastructure-findings.json")
    parser.add_argument("--document", default="docs/run1-scientific-results.md")
    args = parser.parse_args()
    campaign, out = ROOT / args.campaign, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((campaign / "campaign-manifest.json").read_text())
    pool = Path(manifest["development_pool"]["path"])
    if not pool.is_absolute():
        pool = ROOT / pool
    if sha(pool) != manifest["development_pool"]["sha256"]:
        raise ValueError("Registered private development pool changed")
    seeds = json.loads(pool.read_text())
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("Expected eight distinct registered development layouts")
    case_map, conditions = {seed: i for i, seed in enumerate(seeds)}, manifest["regimes"]
    if len(conditions) != 6 or len(set(conditions)) != 6:
        raise ValueError("Expected six equally weighted conditions")
    expected = len(seeds) * len(conditions)
    native_rows = database_rows(campaign / "programs.sqlite")
    slots, by_id = canonical_slots(native_rows), {row["id"]: row for row in native_rows}
    generations = set(slots) | {int(path.name[4:]) for path in campaign.glob("gen_*") if path.name[4:].isdigit()}
    notes_path = ROOT / args.notes
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    infrastructure_path = ROOT / args.infrastructure
    infrastructure = {row["generation"]: {key: row[key] for key in ("classification", "interpretation")}
                      for row in json.loads(infrastructure_path.read_text())} if infrastructure_path.exists() else {}
    candidates, condition_metrics, compact, analyses, inputs, snapshots = [], [], [], [], {}, []
    for generation in sorted(generations):
        row = slots.get(generation)
        meta = row["metadata"] if row else {}
        source = source_sha(row["code"]) if row else (sha(campaign / f"gen_{generation}/main.py") if (campaign / f"gen_{generation}/main.py").exists() else None)
        folder = Path(meta.get("recovery_results_dir", campaign / f"gen_{generation}/results"))
        if not folder.is_absolute():
            folder = ROOT / folder
        raw, provenance = episode_rows(folder, case_map, conditions, source, manifest["evaluation"])
        inputs.update(provenance)
        clean = [compact_episode(r, generation, case, source) for (case, condition), r in sorted(raw.items())]
        compact.extend(clean)
        metric = {"generation": generation, "native_id": row["id"] if row else None,
                  "source_sha256": source, "native_correct": bool(row["correct"]) if row else None,
                  "native_score": row["combined_score"] if row else None,
                  "status": ("persisted_complete_panel" if len(clean) == expected else "persisted_no_panel" if not clean else "persisted_incomplete_panel") if row else "pending_native_slot",
                  "evaluation_started": (folder / "evaluation-manifest.json").exists(),
                  "infrastructure_failure": infrastructure.get(generation),
                  "native_evaluation_elapsed_seconds": meta.get("evaluation_seconds"), **metrics(clean, expected)}
        if row and len(clean) == expected and abs(metric["task"] - row["combined_score"]) > 1e-10:
            raise ValueError("Complete saved episode mean differs from native score")
        candidates.append(metric)
        for condition in conditions:
            subset = [r for r in clean if r["condition"] == condition]
            condition_metrics.append({"generation": generation, "condition": condition, **metrics(subset, len(seeds))})
        if row:
            analyses.append(program_changes(row, by_id, notes))
            snapshots.append((epoch(meta.get("evaluation_finished_at") or row["timestamp"]), metric))
    calls = [json.loads(path.read_text()) for path in sorted((campaign / "model-calls").glob("*.json"))]
    curve, best = [], None
    cumulative_cpu = {field: 0. for field in ("candidate_cpu_seconds", "evaluator_cpu_seconds")}
    first_evaluation = min((epoch(r["metadata"].get("evaluation_started_at")) for r in slots.values()
                            if epoch(r["metadata"].get("evaluation_started_at")) is not None), default=None)
    for cutoff, metric in sorted(snapshots, key=lambda pair: (pair[0], pair[1]["generation"])):
        if metric["native_correct"] and metric["recorded_episodes"] == expected:
            if best is None or (metric["native_score"], -metric["generation"]) > (best["native_score"], -best["generation"]):
                best = metric
        for field in cumulative_cpu:
            cumulative_cpu[field] += metric[field + "_sum"]
        curve.append({"generation": metric["generation"], "evaluation_finished_epoch": cutoff,
                      "elapsed_since_first_evaluation_seconds": cutoff - first_evaluation if first_evaluation is not None else None,
                      **{field + "_cumulative_recorded_sum": value for field, value in cumulative_cpu.items()},
                      "native_correct": metric["native_correct"], "task": metric["task"],
                      "bestsofar_generation": best["generation"] if best else None,
                      "bestsofar_task": best["task"] if best else None,
                      **call_snapshot(calls, cutoff)})
    paired = paired_comparison(compact, 0, best["generation"], conditions) if best and 0 in slots else {"available": False, "reason": "Seed and eligible complete winner not both available", "scope": SCOPES}
    summary = {"campaign": manifest["campaign"], "exported_utc": datetime.now(timezone.utc).isoformat(),
               "target_slot_cap": manifest["total_candidate_slots"], "persisted_unique_slots": len(slots),
               "administrative_seed_copy_rows": sum(bool(r["metadata"].get("_is_island_copy")) for r in native_rows),
               "pending_slot_directories": sorted(generations - set(slots)),
               "valid_persisted_slots": sum(bool(row["correct"]) for row in slots.values()),
               "failed_persisted_slots": sorted(g for g, row in slots.items() if not row["correct"]),
               "recorded_condition_episodes": len(compact), "recorded_physical_transitions": sum(r["steps"] for r in compact),
               "recorded_invalid_episodes": sum(r["invalid"] for r in compact),
               "best_development_generation": best["generation"] if best else None,
               "best_development_task": best["task"] if best else None,
               "source_review_notes_sha256": sha(notes_path) if notes_path.exists() else None,
               "infrastructure_findings_sha256": sha(infrastructure_path) if infrastructure_path.exists() else None,
               "campaign_manifest_sha256": sha(campaign / "campaign-manifest.json"),
               "private_pool_sha256": sha(pool), "reporter_sha256": sha(__file__),
               "input_episode_files": len(inputs),
               "input_episode_file_inventory_sha256": source_sha(json.dumps(inputs, sort_keys=True)),
               "scope": SCOPES, "new_world_episodes": 0, "new_candidate_executions": 0, "new_model_calls": 0}
    for field in ("candidate_cpu_seconds", "evaluator_cpu_seconds", "episode_wall_seconds"):
        summary[field + "_recorded_sum"] = sum(r.get(field) or 0. for r in compact)
        summary[field + "_unavailable_recorded_episodes"] = sum(r.get(field) is None for r in compact)
    write_json(out / "summary.json", summary)
    write_json(out / "candidate-metrics.json", candidates)
    write_csv(out / "candidate-metrics.csv", candidates)
    write_csv(out / "condition-metrics.csv", condition_metrics)
    write_json(out / "resource-curve.json", curve)
    write_csv(out / "resource-curve.csv", curve)
    write_json(out / "paired-development.json", paired)
    write_json(out / "program-analysis.json", analyses)
    with gzip.GzipFile(filename=str(out / "development-episodes.jsonl.gz"), mode="wb", mtime=0) as stream:
        for row in compact:
            stream.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    generated_document(summary, candidates, paired, ROOT / args.document)
    print(json.dumps({key: summary[key] for key in ("persisted_unique_slots", "recorded_condition_episodes", "best_development_generation", "recorded_invalid_episodes")}))


if __name__ == "__main__":
    main()
