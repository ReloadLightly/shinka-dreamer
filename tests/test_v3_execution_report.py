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
