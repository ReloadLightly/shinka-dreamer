"""Checks that protect paired intervention evidence and saved assessment cases."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_candidate import compact_trace
from assess_selected import save_case


def test_trace_hash_separates_predictions_from_executed_experience():
    replay = json.loads((ROOT / "artifacts/campaign-v2/mechanism-gen14/replay/replay.json").read_text())
    trace = replay["frames"][:2]
    left, right = copy.deepcopy(trace), copy.deepcopy(trace)
    for frame in left + right:
        frame["model"]["learning"]["transition_weights"] = [0., 0.]
    left[-1]["model"]["learning"]["transition_weights"] = [1., 0.]
    a, b = compact_trace({"trace": left})["audit"], compact_trace({"trace": right})["audit"]
    assert a["trajectory_sha256"] == b["trajectory_sha256"]
    assert a["map_position_sha256"] == b["map_position_sha256"]
    assert not a["parameters_constant"] and b["parameters_constant"]
    assert a["localization_errors"] == a["visible_terrain_errors"] == 0
    changed = copy.deepcopy(right)
    changed[0]["action"]["move"] = [0, 0]
    if right[0]["action"]["move"] == [0, 0]:
        changed[0]["action"]["move"] = [1, 0]
    assert compact_trace({"trace": changed})["audit"]["trajectory_sha256"] != b["trajectory_sha256"]


def test_atomic_case_never_replaces_a_saved_result(tmp_path):
    path = tmp_path / "0000.json"
    save_case(path, {"case": 0, "result": "original"})
    with pytest.raises(FileExistsError):
        save_case(path, {"case": 0, "result": "replacement"})
    assert json.loads(path.read_text())["result"] == "original"
