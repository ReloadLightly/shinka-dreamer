import json

import pytest

from scripts.run1_science_report import (
    ast_summary, call_snapshot, canonical_slots, compact_episode, episode_rows,
    metrics, paired_comparison, source_sha,
)


def episode(generation=0, case=0, condition="a", task=0., reason="invalid", count=1):
    return {"generation": generation, "case": case, "condition": condition,
            "task": task, "reason": reason, "invalid": reason == "invalid",
            "keys": 0, "door": False, "steps": 1,
            "stats": {"brier_near": [task * count, count]},
            "candidate_cpu_seconds": None, "evaluator_cpu_seconds": .1,
            "episode_wall_seconds": .2, "encountered_switch": False,
            "replacements_encountered": 0, "exposure": {}}


def test_all_invalid_stay_in_outcomes_but_missing_cpu_is_not_zero_measurement():
    result = metrics([episode(case=i) for i in range(8)], 8)
    assert result["task"] == 0 and result["invalid_rate"] == 1
    assert result["escape_rate"] == result["death_rate"] == result["timeout_rate"] == 0
    assert result["candidate_cpu_seconds_measured_episodes"] == 0
    assert result["candidate_cpu_seconds_unavailable_recorded_episodes"] == 8
    incomplete = metrics([episode(task=.8, reason="escaped")], 8)
    assert incomplete["task"] is None and incomplete["escape_rate"] is None
    assert incomplete["task_lower_bound"] == .1
    assert incomplete["task_upper_bound"] == pytest.approx(.975)
    assert incomplete["missing_episodes"] == 7


def test_public_episode_allowlist_excludes_private_seed_and_payloads():
    raw = {"seed": 9919199191, "regime": "a", "combined_score": 0.,
           "reason": "invalid", "error": "RuntimeError: secret seed 9919199191",
           "steps": 0, "keys": 0, "door": False,
           "learning": {"updates": 1, "secret": 9919199191},
           "switch_step": 49, "law": [9919199191], "trace": [9919199191],
           "switch_exposure": [{"steps": 0}, {"steps": 2}]}
    public = compact_episode(raw, 1, 0, "source")
    serialized = json.dumps(public)
    assert "9919199191" not in serialized
    assert public["error_type"] == "RuntimeError"
    assert public["replacements_encountered"] == 1
    assert public["candidate_reported_update_counts"] == {"updates": 1}


def test_checkpoint_and_aggregate_deduplicate_but_conflicts_are_not_retries(tmp_path):
    checkpoint = tmp_path / "episode-checkpoints"
    checkpoint.mkdir()
    row = {"seed": 123, "regime": "a", "value": 1}
    (checkpoint / "one.json").write_text(json.dumps(row))
    (tmp_path / "episodes.json").write_text(json.dumps([row]))
    rows, inventory = episode_rows(tmp_path, {123: 0}, ["a"], None, None)
    assert len(rows) == 1 and len(inventory) == 2
    (tmp_path / "episodes.json").write_text(json.dumps([{**row, "value": 2}]))
    with pytest.raises(ValueError, match="Conflicting"):
        episode_rows(tmp_path, {123: 0}, ["a"], None, None)


def test_whole_layout_bootstrap_keeps_opposite_conditions_together_and_recomputes_ratios():
    rows = []
    for case in range(8):
        value = case / 8
        for condition, effect in (("a", value), ("b", -value)):
            # Task effects cancel within each layout, so every whole-layout
            # resample must cancel, despite nonzero condition effects.
            rows.append(episode(0, case, condition, .5, "escaped", case + 1))
            rows.append(episode(1, case, condition, .5 + effect / 2, "escaped", 9 - case))
    result = paired_comparison(rows, 0, 1, ["a", "b"])
    task = result["groups"]["all"]["metrics"]["task"]
    assert task["difference_best_minus_seed"] == 0
    assert task["ci95_descriptive_layout_bootstrap"] == [0, 0]
    assert result["groups"]["a"]["metrics"]["task"]["difference_best_minus_seed"] > 0
    near = result["groups"]["a"]["metrics"]["brier_near"]
    expected = sum((.5 + case / 16) * (9 - case) for case in range(8)) / sum(9 - case for case in range(8))
    assert near["best"] == pytest.approx(expected)
    assert near["best"] != pytest.approx(sum(.5 + case / 16 for case in range(8)) / 8)
    assert result == paired_comparison(rows, 0, 1, ["a", "b"])
    unavailable = paired_comparison(rows[:-1], 0, 1, ["a", "b"])
    assert not unavailable["available"]


def test_seed_administrative_copies_do_not_become_new_candidate_slots():
    original = {"generation": 0, "metadata": {}, "code": "same", "id": "original"}
    clone = {**original, "metadata": {"_is_island_copy": True}, "id": "copy"}
    assert canonical_slots([clone, original]) == {0: original}
    with pytest.raises(ValueError, match="Different native sources"):
        canonical_slots([{**original, "generation": 1}, {**original, "generation": 1, "code": "different"}])


def test_resource_alignment_excludes_later_calls_and_does_not_double_count_reasoning():
    calls = [{"ended_utc": "2026-10-09T01:00:00Z", "role": "readiness", "elapsed_seconds": 4,
              "usage": {"inputTokens": 10, "outputTokens": 5, "reasoningOutputTokens": 3,
                        "cacheReadTokens": 20, "usageStatus": "reported"}},
             {"ended_utc": "2026-10-09T03:00:00Z", "role": "mutation", "usage": None}]
    result = call_snapshot(calls, 1791507601)
    assert result["completed_calls"] == result["readiness_fixture_calls"] == 1
    assert result["uncached_input_plus_output_tokens"] == 15
    assert result["cached_input_tokens"] == 20


def test_static_unreachable_helpers_are_hints_not_claims_of_inactivity():
    code = "def unused():\n return 1\ndef helper():\n return 2\ndef planner(m,o):\n return helper()\n"
    result, _ = ast_summary(code)
    assert result["functions_without_direct_named_path_from_entrypoints"] == ["unused"]
    assert "callbacks may be missed" in result["static_limit"]
    assert result["source_sha256"] == source_sha(code)
