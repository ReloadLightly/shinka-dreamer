import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v3_execution_report", ROOT / "scripts/v3_execution_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def plan():
    return {"sample_size": 1, "counts": {"condition_episodes": 2, "passive_shadow_episode_passes": 5}}


def test_pending_only_counts_filenames(tmp_path):
    directory = tmp_path / "episode-checkpoints/000000"
    directory.mkdir(parents=True)
    (directory / "memory--uniform.json").write_text("not JSON: outcome content must not be read")
    result = report.assessment_account(tmp_path, plan())
    assert result["status"] == "pending"
    assert result["durable_world_checkpoint_files"] == 1
    assert result["completed_world_episodes"] is None
    assert result["full_assessment_complete"] is False
    assert "resources" not in result


def test_completed_costs_do_not_double_count_evaluator(tmp_path):
    complete = {"cases": 1, "condition_episodes": 2, "completed_episode_attempts": 2,
        "invalid_condition_episodes": 1, "candidate_cpu_seconds": 1,
        "assessment_worker_cpu_seconds": 2, "evaluator_cpu_seconds": .5,
        "episode_wall_seconds": 8, "interrupted_attempts_unmeasured": [],
        "cpu_unavailable_attempts": 0,
        "matched": {"shadow_condition_episodes": 5, "completed_shadow_attempts": 5,
            "candidate_cpu_seconds": 3, "evaluator_cpu_seconds": 4, "wall_seconds": 9,
            "interrupted_attempts_unmeasured": [], "failed_recorded_episode_analyses": 0}}
    (tmp_path / "execution-complete.json").write_text(json.dumps(complete))
    result = report.assessment_account(tmp_path, plan())
    assert result["resources"]["measured_nonoverlapping_cpu_seconds"] == 10
    assert result["resources"]["evaluator_cpu_seconds_subset_not_added"] == .5
    assert result["invalid_world_episodes"] == 1
    assert result["completed_world_episodes"] == 2
    assert result["full_assessment_complete"] is True
    complete["interrupted_attempts_unmeasured"] = ["world-started-no-end"]
    complete["matched"]["interrupted_attempts_unmeasured"] = ["shadow-group-started-no-end"]
    (tmp_path / "execution-complete.json").write_text(json.dumps(complete))
    result = report.assessment_account(tmp_path, plan())
    assert result["completed_world_episodes"] == 2
    assert result["unknown_interrupted_world_executions"] is None
    assert result["unknown_interrupted_shadow_passes"] is None
    assert result["resources"]["cpu_components_seconds"]["unknown_world_attempt_cpu_seconds"] is None
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "execution-complete.json").write_text(json.dumps(complete))
    (archive / "execution-provenance.json").write_text(json.dumps({"controller_cpu_seconds": 7}))
    result = report.assessment_account(tmp_path, plan(), analysis_directory=archive)
    assert result["archived_execution_provenance"]["record"]["controller_cpu_seconds"] == 7
    (archive / "execution-complete.json").write_text("{}")
    with pytest.raises(ValueError, match="provenance does not match"):
        report.assessment_account(tmp_path, plan(), analysis_directory=archive)


def test_missing_measurement_is_not_zero_cpu():
    result = report.resources({"measured": 5, "unknown": None})
    assert result["measured_nonoverlapping_cpu_seconds"] == 5
    assert result["missing_cpu_components"] == ["unknown"]
    assert result["cpu_components_seconds"]["unknown"] is None


def test_clock_formats_remain_independent(tmp_path):
    path = tmp_path / "host-time.txt"
    path.write_text("\tUser time (seconds): 3.4\n\tSystem time (seconds): 0.7\n"
                    "\tElapsed (wall clock) time (h:mm:ss or m:ss): 12:32.20\n\tExit status: 0\n")
    result = report.host_time(path)
    assert result["elapsed_seconds"] == 752.2
    assert result["user_cpu_seconds"] == 3.4
    assert result["status"] == "complete"
    assert report.elapsed_value("1:02:03.5") == 3723.5
    path.write_text("")
    assert report.host_time(path)["status"] == "pending"


def test_shadow_errors_require_hash_verified_closed_analysis(tmp_path):
    analysis = tmp_path / "matched-analysis.json"
    analysis.write_text(json.dumps({"regimes": {"uniform": {"audits": {
        "shadow_error_episodes": {"online": 2, "frozen": 1},
        "missing_records": [{"condition": "frozen", "case": 0}]}}}}))
    assert report.closed_shadow_errors(tmp_path, "plan")["individual_shadow_errors"] is None
    closure = {"status": "analysis-complete", "plan_sha256": "plan",
               "files": {"matched-analysis.json": report.sha256(analysis)}}
    (tmp_path / "analysis-closure.json").write_text(json.dumps(closure))
    result = report.closed_shadow_errors(tmp_path, "plan")
    assert result["individual_shadow_errors"] == 2
    assert result["missing_shadow_records"] == 1
    analysis.write_text(analysis.read_text() + "\n")
    with pytest.raises(ValueError, match="hash verification"):
        report.closed_shadow_errors(tmp_path, "plan")


def paused_fixture():
    plan = {"sample_size": 1536, "counts": {
        "condition_episodes": 46080, "passive_shadow_episode_passes": 23040}}
    checkpoint = {"status": "paused-by-user", "plan_sha256": "frozen-plan",
        "full_assessment_complete": False, "automatic_follow_through_stopped": True,
        "treatment_effects_aggregated": False, "new_world_episodes_from_checkpoint_closure": 0,
        "new_model_calls_from_checkpoint_closure": 0, "planned_cases": 1536,
        "fully_evaluated_cases": 140, "case_file_sha256": {str(i): "hash" for i in range(140)},
        "world_episodes": 4200, "shadow_episode_passes": 2100, "invalid_world_episodes": 27,
        "individual_shadow_errors": 0, "outer_matched_failures": 0,
        "unfinished_world_attempts": 0, "unfinished_matched_attempt_groups": 0,
        "unavailable_world_cpu_measurements": 0,
        "resources": {"world_candidate_cpu_seconds": 1, "world_assessment_worker_cpu_seconds": 2,
            "shadow_candidate_cpu_seconds": 3, "shadow_evaluator_cpu_seconds": 4,
            "controller_cpu_seconds": 5, "measured_nonoverlapping_cpu_seconds": 15,
            "world_evaluator_cpu_seconds_subset_not_added": .5, "driver_monotonic_seconds": 10,
            "wrapper_monotonic_seconds": 10.1, "gnu_time_alternative_not_added": {"elapsed_seconds": 11}},
        "driver_session_finished": {"completed_cases": 136},
        "launch_finished": {"exit_code": 130}, "resource_note": "Clocks differ; signal interrupted launch.",
        "resume_status": "No automatic resume."}
    return plan, checkpoint


def test_paused_checkpoint_counts_are_not_full_completion(tmp_path):
    frozen, checkpoint = paused_fixture()
    public = tmp_path / "public"
    public.mkdir()
    (public / "operator-checkpoint-complete.json").write_text(json.dumps(checkpoint))
    # Deliberately unreadable scientific records prove this path needs neither
    # private outcomes nor a scientific matched-analysis result.
    directory = tmp_path / "episode-checkpoints/000000"
    directory.mkdir(parents=True)
    (directory / "memory--uniform.json").write_text("private outcome must not be read")
    (public / "matched-analysis.json").write_text("scientific analysis must not be read")
    result = report.assessment_account(tmp_path, frozen, public, "frozen-plan")
    assert result["status"] == "paused-by-user"
    assert result["full_assessment_complete"] is False
    assert result["completed_world_episodes"] == 4200
    assert result["completed_shadow_passes"] == 2100
    assert result["invalid_world_episodes"] == 27
    assert result["individual_shadow_errors"] == 0
    assert result["unknown_interrupted_world_executions"] == 0
    assert result["unknown_interrupted_shadow_passes"] == 0
    assert result["resources"]["measured_nonoverlapping_cpu_seconds"] == 15
    assert result["resources"]["evaluator_cpu_seconds_subset_not_added"] == .5
    counts = report.execution_counts(5390, {"isolated_shadow_episode_passes": 32,
        "in_process_prediction_episode_passes": 372, "feature_construction_episode_passes": 264}, frozen, result)
    assert counts["executed_to_date_world_episodes"] == 9590
    assert counts["executed_to_date_isolated_shadow_passes"] == 2132
    assert counts["planned_full_wave_world_episodes"] == 51470
    assert counts["planned_full_wave_isolated_shadow_passes"] == 23072
    assert counts["full_wave_complete"] is False
    assert all(value is None for key, value in counts.items() if "completed_full_wave" in key)


@pytest.mark.parametrize("mutation,error", [
    ({"plan_sha256": "other-plan"}, "identity/status"),
    ({"full_assessment_complete": True}, "identity/status"),
    ({"treatment_effects_aggregated": True}, "identity/status"),
    ({"world_episodes": 4230}, "counts"),
])
def test_paused_checkpoint_rejects_mismatched_identity_or_counts(tmp_path, mutation, error):
    frozen, checkpoint = paused_fixture()
    checkpoint.update(mutation)
    path = tmp_path / "operator-checkpoint-complete.json"
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match=error):
        report.assessment_account(tmp_path, frozen, plan_sha256="frozen-plan")
