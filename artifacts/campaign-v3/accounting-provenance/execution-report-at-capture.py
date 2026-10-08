"""Read-only accounting of v3 executions, separate from scientific outcomes.

No candidate, model, world, private pool or treatment-effect analysis is run.
While assessment is pending, only checkpoint filenames are inspected. Completed
assessment resources are read after its durable completion marker exists.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.provenance import sha256

ART = ROOT / "artifacts/campaign-v3"
SUPPLEMENT = ART / "execution-development-supplement.json"


def read(path):
    return json.loads(Path(path).read_text())


def evidence(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": sha256(path)}


def resources(parts, **extra):
    """Sum only caller-declared disjoint CPU scopes; null means unavailable."""
    return {"cpu_components_seconds": parts,
            "measured_nonoverlapping_cpu_seconds": sum(v for v in parts.values() if v is not None),
            "missing_cpu_components": [k for k, v in parts.items() if v is None], **extra}


def row_cost(rows):
    return resources({key: sum(r[key] for r in rows) for key in
                      ("candidate_cpu_seconds", "evaluator_cpu_seconds")},
                     summed_episode_wall_seconds=sum(r["seconds"] for r in rows),
                     coverage="Episode candidate/evaluator CPU; surrounding analysis/controller overhead unmeasured.")


def capture_development():
    """Archive compact cost/count evidence from otherwise ignored old records."""
    if SUPPLEMENT.exists():
        raise FileExistsError("Preserve the existing development accounting snapshot")
    ledger_path = ART / "development-ledger.json"
    ledger = read(ledger_path)
    units = {}
    objective = ART / "objective-check-episodes.json"
    rows = read(objective)
    units["objective_check"] = {"world_episodes": len(rows),
        "invalid_world_episodes": sum(r["error"] is not None for r in rows),
        "environment_steps": sum(r["steps"] for r in rows), "resources": row_cost(rows),
        "evidence": [evidence(objective)]}

    corpus_path = ROOT / "results/v3-comparator-fit/corpus-summary.json"
    corpus = read(corpus_path)
    rows = corpus["train"] + corpus["validation"]
    fit = ledger["comparator_fitting_corpus"]
    units["comparator_fitting_corpus"] = {"world_episodes": len(rows),
        "invalid_world_episodes": sum(r["reason"] == "invalid" for r in rows),
        "environment_steps": sum(r["steps"] for r in rows),
        "resources": resources({"whole_fit_cpu_seconds": None},
            elapsed_seconds=fit["resources_whole_fit_run"]["elapsed_seconds"],
            summed_corpus_episode_wall_seconds=sum(r["seconds"] for r in rows),
            coverage="264 corpus worlds, feature construction, fitting and 288 prediction passes. CPU unmeasured; corpus wall sum is a subset, not added to elapsed."),
        "evidence": [evidence(corpus_path), evidence(ROOT / "controls/v3/fit-results.json")]}

    diagnostic_path = ROOT / "controls/v3/development-diagnostic.json"
    diagnostic = read(diagnostic_path)
    rows = [r for case in diagnostic["cases"] for r in case["conditions"].values()]
    units["comparator_decision_diagnostic"] = {"world_episodes": len(rows),
        "invalid_world_episodes": sum(r["error"] is not None for r in rows),
        "environment_steps": sum(r["steps"] for r in rows),
        "resources": resources({key: diagnostic[key] for key in ("controller_cpu_seconds", "children_cpu_seconds")},
            elapsed_seconds=diagnostic["elapsed_seconds"],
            coverage="Whole diagnostic: 72 worlds plus 84 new in-process prediction passes. Child CPU and controller CPU are disjoint; per-world counters are subsets and are not added."),
        "evidence": [evidence(diagnostic_path)]}
    units["initial_non_test_smoke"] = {"world_episodes": ledger["non_test_smoke"]["condition_episodes"],
        "invalid_world_episodes": None, "environment_steps": None,
        "resources": resources({"smoke_cpu_seconds": None}, coverage="Four declared initial smoke worlds; detailed cost/invalid records not retained in the compact ledger."),
        "evidence": [evidence(ledger_path)], "description": ledger["non_test_smoke"]["description"]}

    smoke_path = ROOT / "results/v3-matched-smoke.json"
    smoke = read(smoke_path)
    rows = smoke["records"]
    units["matched_predictor_smoke"] = {"world_episodes": smoke["memory_environment_episodes"],
        "shadow_episode_passes": smoke["shadow_condition_episodes"],
        "invalid_world_episodes": sum(r["policy_result"]["error"] is not None for r in rows),
        "invalid_shadow_passes": sum(bool(s.get("error")) for r in rows for s in r["shadows"].values()),
        "environment_steps": sum(r["policy_result"]["steps"] for r in rows),
        "resources": resources({
            "policy_candidate_cpu_seconds": sum(r["policy_result"]["candidate_cpu_seconds"] for r in rows),
            "policy_evaluator_cpu_seconds": sum(r["policy_result"]["evaluator_cpu_seconds"] for r in rows),
            "shadow_candidate_cpu_seconds": sum(r["candidate_cpu_seconds"] for r in rows),
            "shadow_evaluator_cpu_seconds": sum(r["evaluator_cpu_seconds"] for r in rows)},
            coverage="Eight world runs and 24 isolated shadow passes; outer controller overhead excluded."),
        "evidence": [evidence(smoke_path)]}

    for key, path in (("old_matched_driver_integration", Path("/tmp/shinka-v3-matched-integration-development/execution-complete.json")),
                      ("five_shadow_integration", ART / "matched-integration-smoke/execution-complete.json")):
        complete = read(path)
        shadow = complete["matched"]
        worker_key = "assessment_worker_cpu_seconds" if "assessment_worker_cpu_seconds" in complete else "evaluator_cpu_seconds"
        units[key] = {"world_episodes": complete["condition_episodes"],
            "invalid_world_episodes": complete["invalid_condition_episodes"],
            "shadow_episode_passes": shadow["shadow_condition_episodes"],
            "resources": resources({"world_candidate_cpu_seconds": complete["candidate_cpu_seconds"],
                "world_worker_cpu_seconds": complete[worker_key],
                "shadow_candidate_cpu_seconds": shadow["candidate_cpu_seconds"],
                "shadow_evaluator_cpu_seconds": shadow["evaluator_cpu_seconds"]},
                world_worker_scope=worker_key,
                coverage="World and shadow intervals are disjoint. Assessment worker CPU includes evaluator CPU when present; evaluator is not added twice. Controller overhead excluded."),
            "evidence": [evidence(path)]}

    for key, path in (("gen1_mechanism_smoke", ART / "mechanism-gen1-smoke/summary.json"),
                      ("selected_transparency", ART / "selection/transparency.json"),
                      ("selected_mechanism", ART / "mechanism-selected/summary.json")):
        result = read(path)
        count = result.get("condition_episodes", result.get("manifest", {}).get("condition_episodes"))
        units[key] = {"world_episodes": count, "invalid_world_episodes": result["invalid_condition_episodes"],
            "environment_steps": result["environment_steps"],
            "resources": resources({k: result[k] for k in ("candidate_cpu_seconds", "evaluator_cpu_seconds")},
                elapsed_seconds=result["session_seconds"],
                coverage="World candidate/evaluator intervals; diagnostic compaction and surrounding driver CPU not measured."),
            "evidence": [evidence(path)]}
    if sum(u["world_episodes"] for u in units.values()) != 830:
        raise ValueError("Development counts differ from the individually reconciled 830 worlds")
    for key, item in ledger.items():
        if isinstance(item, dict) and isinstance(item.get("condition_episodes"), int):
            # Every declared world unit is represented; no replay-only unit is added.
            if key not in {"objective_check", "comparator_fitting_corpus", "comparator_decision_diagnostic",
                "non_test_smoke", "matched_predictor_smoke", "matched_driver_integration_smoke",
                "gen1_mechanism_smoke", "selected_instrumentation_transparency", "selected_mechanism_audit"}:
                raise ValueError(f"Unreviewed development unit: {key}")
    result = {"units": units, "development_world_episodes": 830,
        "in_process_prediction_episode_passes": ledger["completed_comparator_replay_accounting"]["in_process_prediction_episode_passes"],
        "feature_construction_episode_passes": ledger["completed_comparator_replay_accounting"]["in_process_feature_construction_episode_passes"],
        "isolated_shadow_episode_passes": sum(u.get("shadow_episode_passes", 0) for u in units.values()),
        "arithmetic_correction": "Earlier conversational totals617/930 were arithmetic errors, not missing executions. Named earlier units sum517; adding24 transparency+288 selected audit+1 five-shadow integration gives830.",
        "scope": "Scientific development runs only. Tests, source inspection, UI browsing and already-recorded replay rendering are not additional world episodes.",
        "capture_script_sha256": sha256(__file__)}
    SUPPLEMENT.write_text(json.dumps(result, indent=2) + "\n")


def elapsed_value(value):
    pieces = [float(p) for p in value.split(":")]
    return sum(v * 60**i for i, v in enumerate(reversed(pieces)))


def host_time(path):
    values = {}
    keys = {"User time (seconds)": "user_cpu_seconds", "System time (seconds)": "system_cpu_seconds",
            "Elapsed (wall clock) time (h:mm:ss or m:ss)": "elapsed_text", "Exit status": "exit_status"}
    for line in path.read_text().splitlines():
        for prefix, key in keys.items():
            if line.strip().startswith(prefix + ":"):
                value = line.strip()[len(prefix)+1:].strip()
                values[key] = value if key == "elapsed_text" else float(value)
    if "elapsed_text" in values:
        values["elapsed_seconds"] = elapsed_value(values["elapsed_text"])
    return {**values, "evidence": evidence(path)}


def assessment_account(raw, plan):
    expected = plan["counts"]
    result = {"planned_world_episodes": expected["condition_episodes"],
              "planned_shadow_passes": expected["passive_shadow_episode_passes"],
              "model_calls": 0, "new_worlds_from_shadows": 0}
    complete_path = raw / "execution-complete.json"
    if not complete_path.exists():
        # Filename-only progress, with no outcome or treatment-cost inspection.
        result.update(status="pending", completed_world_episodes=None, completed_shadow_passes=None,
            durable_world_checkpoint_files=sum(1 for _ in (raw / "episode-checkpoints").glob("*/*.json")),
            durable_paired_case_files=sum(1 for _ in (raw / "cases").glob("*.json")),
            recorded_policy_episode_files=sum(1 for _ in (raw / "matched").glob("*.json")),
            resource_status="Pending durable execution completion; no treatment aggregates read.")
        return result
    complete = read(complete_path)
    if complete["cases"] != plan["sample_size"] or complete["condition_episodes"] != expected["condition_episodes"]:
        raise ValueError("Assessment completion count does not match the frozen plan")
    matched = complete["matched"]
    result.update(status="execution-complete", evidence=evidence(complete_path),
        completed_world_episodes=complete["condition_episodes"],
        completed_world_attempts=complete["completed_episode_attempts"],
        interrupted_world_attempts_unmeasured=len(complete["interrupted_attempts_unmeasured"]),
        cpu_unavailable_world_attempts=complete["cpu_unavailable_attempts"],
        invalid_world_episodes=complete["invalid_condition_episodes"],
        completed_shadow_passes=matched["shadow_condition_episodes"],
        completed_shadow_attempts=matched["completed_shadow_attempts"],
        interrupted_shadow_attempt_groups_unmeasured=len(matched["interrupted_attempts_unmeasured"]),
        failed_recorded_episode_shadow_analyses=matched["failed_recorded_episode_analyses"],
        resources=resources({"world_candidate_cpu_seconds": complete["candidate_cpu_seconds"],
            "world_assessment_worker_cpu_seconds": complete["assessment_worker_cpu_seconds"],
            "shadow_candidate_cpu_seconds": matched["candidate_cpu_seconds"],
            "shadow_evaluator_cpu_seconds": matched["evaluator_cpu_seconds"]},
            evaluator_cpu_seconds_subset_not_added=complete["evaluator_cpu_seconds"],
            world_attempt_wall_seconds=complete["episode_wall_seconds"],
            shadow_attempt_wall_seconds=matched["wall_seconds"],
            coverage="Disjoint world candidate/worker and shadow candidate/evaluator scopes. World worker includes evaluator; complete controller overhead is additional and not silently inferred."))
    # Whole-command counters are alternatives to, not additions to, these scopes.
    result["gnu_time_invocations_alternative_not_added"] = [host_time(p) for p in sorted(raw.glob("host-time*.txt"))]
    result["driver_sessions"] = [read(p) for p in sorted(raw.glob("execution-*.finished.json"))]
    provenance = raw / "execution-provenance.json"
    if provenance.exists():
        result["archived_execution_provenance"] = {"evidence": evidence(provenance), "record": read(provenance)}
    result["clock_note"] = "Keep monotonic driver durations, UTC endpoints and GNU time separately. They may differ; no discrepancy is silently resolved and overlapping wall intervals are never added as campaign elapsed."
    return result


def build_report(raw):
    development = read(SUPPLEMENT)
    native_path = ART / "native-audit.json"
    native = read(native_path)
    search = read(ART / "search-summary.json")
    selection_path = ART / "selection/execution-provenance.json"
    selection = read(selection_path)
    plan_path = ART / "assessment/preregistration.json"
    plan = read(plan_path)
    public_assessment = ART / "assessment"
    if not (raw / "execution-complete.json").exists() and (public_assessment / "execution-complete.json").exists():
        raw = public_assessment
    assessment = assessment_account(raw, plan)
    pre = development["development_world_episodes"] + search["saved_evaluation_episodes"] + selection["condition_episodes"]
    selected = next(c for c in plan["conditions"] if c["name"] == "selected")
    campaign = read(ART / "campaign-manifest.json")
    roles = {c["name"]: {"path": c["program_path"], "sha256": c["program_sha256"]}
             for c in plan["conditions"] if c["name"] in ("memory", "seed", "v2_gen14", "fitted_online")}
    roles.update(evolution_start={"source": campaign["seed_source"], "sha256": campaign["seed_program_sha256"]},
                 selected_instrumented={"path": selected["program_path"], "sha256": selected["program_sha256"]},
                 selected_original=evidence(ART / "selection/selected.py"))
    return {"updated_utc": datetime.now(timezone.utc).isoformat(), "report_script_sha256": sha256(__file__),
        "scope": "Execution accounting only; no treatment effects. One v3 search, separate from preserved v2 science.",
        "development": {**development, "evidence": evidence(SUPPLEMENT)},
        "search": {**search, "invalid_world_episodes": native["invalid_saved_episodes"], "evidence": evidence(ART / "search-summary.json")},
        "selection": {"world_episodes": selection["condition_episodes"],
            "invalid_world_episodes": selection["resource_summary"]["invalid_condition_episodes"],
            "resources": selection["resource_summary"], "gnu_time_alternative_not_added": selection["host_time"],
            "elapsed_monotonic_seconds": selection["controller"]["elapsed_seconds"],
            "utc_end_minus_start_seconds": selection["utc_end_minus_start_seconds"],
            "gnu_elapsed_seconds": elapsed_value(selection["host_time"]["elapsed"]),
            "scope": selection["cpu_note"], "clock_note": selection["elapsed_note"], "evidence": evidence(selection_path)},
        "assessment": assessment,
        "counts": {"pre_assessment_world_episodes": pre,
            "planned_full_wave_world_episodes": pre + plan["counts"]["condition_episodes"],
            "completed_full_wave_world_episodes": pre + assessment["completed_world_episodes"] if assessment["completed_world_episodes"] is not None else None,
            "development_in_process_prediction_passes": development["in_process_prediction_episode_passes"],
            "development_feature_construction_passes": development["feature_construction_episode_passes"],
            "development_isolated_shadow_passes": development["isolated_shadow_episode_passes"],
            "planned_full_wave_isolated_shadow_passes": development["isolated_shadow_episode_passes"] + plan["counts"]["passive_shadow_episode_passes"],
            "new_worlds_from_all_replays": 0,
            "note": "World outcome counts retain invalids. Completed attempt counts and unmeasured interrupted attempts are separate; cases shared across regimes/conditions are dependent, not independent replicates."},
        "model_requests": {"native": native["native_model_requests"], "readiness_probes": len(native["probe_calls"]),
            "all_including_probes": native["all_model_requests_including_probes"],
            "native_by_role": native["model_requests_by_role"], "tokens": native["reported_tokens"],
            "token_scope": native["reported_token_scope_note"], "routes": native["billing_attempt_routes"],
            "patch_application_failures": native["patch_application_failures"], "patch_retries": native["patch_retry_attempts"],
            "model_nonzero_returncodes": native["model_nonzero_returncodes"],
            "effective_model": native["effective_model"], "effective_effort": native["effective_effort"],
            "native_call_elapsed_seconds": native["resources"]["model_elapsed_seconds"],
            "timing_scope": native["resource_scope_note"], "clock_records": native["controller_log_timing"],
            "upstream_revision": native["upstream_revision"], "cost_note": native["cost_note"], "evidence": evidence(native_path)},
        "source_roles": roles,
        "manual_engineering_note": "Directional comparator was manually engineered and fitted on development. Evolution began from the historical seed. Selected instrumentation adds diagnostics only and retains its separately archived original source.",
        "resource_limitations": [
            "No single total CPU or wall value is claimed: controller/whole-command and per-episode counters overlap.",
            "Fit CPU, four initial-smoke costs, some driver/analysis overhead, remote model CPU and full search-controller CPU are unavailable; null is not zero.",
            "Native model-call elapsed, summed episode wall, UTC spans and GNU/monotonic durations have different scopes. Selection GNU/UTC elapsed is about752.2s versus685.6s monotonic; cause unestablished.",
            "Actual subscription charges are unmeasured. Native API-list-price estimates are not charges.",
            "Original search parent-session exit code was lost; normal native finalization and released lock are recorded separately."],
        "frozen_plan": evidence(plan_path), "new_world_episodes_from_report": 0, "new_model_calls_from_report": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, default=ROOT / "results/campaign-v3-assessment")
    parser.add_argument("--out", type=Path, default=ART / "execution-ledger.json")
    parser.add_argument("--capture-development", action="store_true", help="Create immutable compact counts/cost snapshot from retained development records once")
    args = parser.parse_args()
    if args.capture_development:
        capture_development()
    result = build_report(args.assessment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"assessment_status": result["assessment"]["status"], **result["counts"]}))


if __name__ == "__main__":
    main()
