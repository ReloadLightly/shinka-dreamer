"""Summarize recorded development interventions without executing any worlds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.provenance import sha256
from v3_analysis import behavior_pair, paired_binary
from v3_assessment import atomic_create

REGIMES = ("uniform", "stationary", "switch")
VARIANTS = ("predictive", "frozen", "no_planning", "frozen_no_planning")


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    files = sorted((directory / "episodes").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in files]
    expected = {(case, regime, variant) for case in range(manifest["layouts"])
                for regime in REGIMES for variant in VARIANTS}
    by_key = {(r["case"], r["regime"], r["variant"]): r for r in rows}
    if (len(rows) != len(expected) or set(by_key) != expected or
            summary["manifest"] != manifest or
            any(r["program_sha256"] != manifest["program_sha256"] for r in rows)):
        raise ValueError("Require the complete, source-bound development grid")
    mismatch_keys = ("chosen_move_mismatches", "choice_score_mismatches",
                    "choice_hazard_field_mismatches", "planning_reconstruction_mismatches",
                    "export_reconstruction_mismatches", "observer_law_mismatches")
    count_keys = ("frames", "choice_frames", "fallback_frames", "choices",
                  "immediate_target_frames", "excluded_occupied_entry_frames")
    decision = {key: sum(r["decision_audit"].get(key, 0) for r in rows)
                for key in count_keys + mismatch_keys}
    decision["all_recorded_checks_pass"] = all(decision[key] == 0 for key in mismatch_keys)
    decision["by_variant"] = {}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        fields = ("planning_brier_destination", "exported_brier_destination", "planning_export_absolute_gap")
        decision["by_variant"][variant] = {
            key: [sum(r["decision_audit"][key][i] for r in subset) for i in (0, 1)]
            for key in fields}
        decision["by_variant"][variant]["planning_export_differing_frames"] = sum(
            r["decision_audit"]["planning_export_differing_frames"] for r in subset)
    groups = {}
    for regime in REGIMES:
        group = {variant: [by_key[case, regime, variant] for case in range(manifest["layouts"])]
                 for variant in VARIANTS}
        a, b = group["predictive"], group["frozen"]
        c, d = group["no_planning"], group["frozen_no_planning"]
        matched = {}
        if summary["matched_prediction_valid_by_regime"][regime]:
            for metric in ("brier_near", "brier_audit", "brier_destination"):
                sa, na = (sum(r["stats"].get(metric, [0, 0])[i] for r in c) for i in (0, 1))
                sb, nb = (sum(r["stats"].get(metric, [0, 0])[i] for r in d) for i in (0, 1))
                matched[metric] = {"online_sum_count": [sa, na], "frozen_sum_count": [sb, nb],
                                   "relative_reduction": 1 - (sa / na) / (sb / nb) if na and nb and sb else None}
        groups[regime] = {
            "outcomes": {variant: {reason: sum(r["reason"] == reason for r in values)
                                    for reason in ("escaped", "caught", "timeout", "invalid")}
                         for variant, values in group.items()},
            "predictive_minus_frozen_escape": paired_binary(
                [r["reason"] == "escaped" for r in a], [r["reason"] == "escaped" for r in b]),
            "predictive_versus_frozen_behavior": behavior_pair(a, b),
            "online_parameter_change_episodes": {variant: sum(r["audit"]["parameters_constant"] is False for r in group[variant])
                                                   for variant in ("predictive", "no_planning")},
            "frozen_parameter_constant_episodes": {variant: sum(r["audit"]["parameters_constant"] is True for r in group[variant])
                                                     for variant in ("frozen", "frozen_no_planning")},
            "map_grew_episodes": {variant: sum(r["audit"]["last_map_cells"] > r["audit"]["first_map_cells"] for r in values)
                                  for variant, values in group.items()},
            "localization_errors": sum(r["audit"]["localization_errors"] for values in group.values() for r in values),
            "matched_fixed_uniform_control_prediction": matched,
            "matched_valid": summary["matched_prediction_valid_by_regime"][regime]}
    return {
        "scope": "Reused development cases; source-specific mechanism validation, not fresh performance or reliable-discovery evidence.",
        "manifest_sha256": sha256(directory / "manifest.json"),
        "summary_sha256": sha256(directory / "summary.json"),
        "episode_files_sha256": hashlib.sha256("\n".join(f"{p.name}:{sha256(p)}" for p in files).encode()).hexdigest(),
        "report_script_sha256": sha256(__file__),
        "program_sha256": manifest["program_sha256"],
        "condition_episodes": len(rows), "decision_audit": decision, "regimes": groups,
        "resources": {key: summary[key] for key in ("candidate_cpu_seconds", "evaluator_cpu_seconds", "session_seconds", "environment_steps")},
        "limitations": [
            "Development outcome intervals describe paired variation but do not correct reuse by evolutionary selection.",
            "Matched prediction point estimates reuse whole-episode sums/counts; fresh assessment supplies registered episode-resampled uncertainty.",
            "Raw parameter changes can include forgetting without new information.",
            "The no_planning flag substitutes a uniform movement law inside the same predictive planner.",
            "Recorded stage-two/three hazards are unconditional before/after occupancy unions, not calibrated survival-conditioned path risk."],
        "additional_environment_episodes": 0, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="artifacts/campaign-v3/mechanism-selected")
    args = parser.parse_args()
    directory = Path(args.directory)
    result = summarize(directory)
    atomic_create(directory / "runtime-review.json", result)
    print(json.dumps({"condition_episodes": result["condition_episodes"],
                      "decision_checks_pass": result["decision_audit"]["all_recorded_checks_pass"]}))


if __name__ == "__main__":
    main()
