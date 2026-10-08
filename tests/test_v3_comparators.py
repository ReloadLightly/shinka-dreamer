"""Semantic checks for the manual comparator, independent of evolved programs."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v3_fit import fitting_arrays, load_module, objective, replay_predictor


def observation(enemy=(1, 1), step=0):
    terrain = [[0] * 5 for _ in range(5)]
    grid = copy.deepcopy(terrain)
    grid[enemy[1] + 2][enemy[0] + 2] = 5
    return {"terrain": terrain, "grid": grid, "keys": 0, "door_open": False,
            "step": step, "feedback": {"displacement": [0, 0]}}


def test_blocked_attempts_add_staying_mass_and_preserve_absolute_directions():
    agent = load_module()
    obs = observation()
    memory = agent.world_model_step(None, obs, None)
    # Enemy at origin; its east cell is blocked. This also blocks both east diagonals.
    memory["map"][(1, 0)] = 1
    memory["known_law"] = [.01, .02, .03, .04, .05, .4, .1, .15, .2]
    rows = agent._transition_rows(memory, (0, 0), (0, 0), (0, 0), {})
    masses = {p: probability for p, probability, _ in rows}
    assert sum(masses.values()) == pytest.approx(1.)
    assert masses[(0, 0)] == pytest.approx(.05 + .03 + .4 + .2)
    assert masses[(-1, 0)] == pytest.approx(.04)
    assert (1, 0) not in masses


def test_aggregated_direction_derivative_matches_finite_difference():
    agent = load_module()
    memory = agent.world_model_step(None, observation(), None)
    memory["map"][(1, 0)] = 1
    memory["theta"] = [-.2, .1, .3, .4, -.1, .2, .1, -.3, .6]
    rows = agent._transition_rows(memory, (0, 0), (0, 0), (0, 0), {})
    for destination, probability, derivative in rows:
        for j in range(9):
            changed = copy.deepcopy(memory)
            changed["theta"][j] += 1e-6
            updated = dict((p, v) for p, v, _ in agent._transition_rows(changed, (0, 0), (0, 0), (0, 0), {}))
            assert (updated[destination] - probability) / 1e-6 == pytest.approx(derivative[j], abs=2e-7)


def test_frozen_update_preserves_mapping_but_changes_no_transition_parameters():
    agent = load_module()
    online = agent.world_model_step(None, observation(), None)
    frozen = copy.deepcopy(online)
    obs = observation((1, 0), step=1)
    online = agent.world_model_step(online, obs, {"move": [0, 0], "interact": True})
    frozen = agent.world_model_step(frozen, dict(obs, learn=False), {"move": [0, 0], "interact": True})
    assert online["map"] == frozen["map"] and online["pos"] == frozen["pos"]
    assert online["enemies"] == frozen["enemies"]
    assert online["theta"] != agent._PRIOR
    assert frozen["theta"] == agent._PRIOR
    assert frozen["step"] == 1 and frozen["parameter_updates"] == 0
    assert online["parameter_updates"] == 1


def test_known_law_bypasses_parameter_estimation_without_hidden_state():
    agent = load_module()
    law = [0., 0., 0., 0., 0., 1., 0., 0., 0.]
    memory = agent.world_model_step(None, dict(observation(), known_law=law), None)
    assert agent._probabilities(memory) == law
    assert memory["theta"] == agent._PRIOR
    # Deterministic east attempt sends visible (1,1) to (2,1).
    assert memory["risk"][(2, 1)] == pytest.approx(1.)
    assert set(memory["terrain"]) == {(x, y) for x in range(-2, 3) for y in range(-2, 3)}


def test_fitting_objective_anonymous_union_gradient():
    np = pytest.importorskip("numpy")
    masks = np.zeros((2, 3, 9))
    masks[0, 0, [4, 5]] = 1  # A blocked attempt aliases with staying.
    masks[0, 1, 3] = 1
    masks[1, 0, 8] = 1
    labels = np.array([1., 0.])
    theta = np.arange(9) / 10.
    loss, gradient = objective(theta, masks, labels, .001)
    for j in range(9):
        changed = theta.copy()
        changed[j] += 1e-6
        assert (objective(changed, masks, labels, .001)[0] - loss) / 1e-6 == pytest.approx(gradient[j], abs=2e-7)


def _shadow_trace():
    first = {"obs": observation(step=0), "action": {"move": [1, 0], "interact": True},
             "world": {"agent": [5, 5], "origin": [5, 5], "enemies": [[6, 6]],
                       "grid": [[0]*15 for _ in range(15)], "keys": 0, "door_open": False},
             "next_enemies": [[6, 5]]}
    second = copy.deepcopy(first)
    second["obs"] = observation(step=1)
    second["obs"]["feedback"]["displacement"] = [1, 0]
    second["world"]["agent"] = [6, 5]
    second["action"]["move"] = [0, 0]
    return [first, second]


def test_matched_helper_forces_experience_keeps_terminal_and_private_law(monkeypatch):
    import v3_matched as matched
    seen = []
    class FakeLaw:
        switch_step = 2
        def __init__(self, *args, **kwargs):
            pass
        def law_for_transition(self, transition):
            return [1/9] * 9
    class FakeShadow:
        def __init__(self, *args):
            self.inputs = []
            seen.append(self.inputs)
        def query(self, request):
            self.inputs.append(copy.deepcopy(request))
            return {"passive_shadow": True,
                    "model": {"enemy": [[1, 0, .7]], "default_enemy": 0.,
                              "position": [request["obs"]["step"], 0], "terrain": [],
                              "learning": {"transition_weights": [0.]}}}
        def close(self):
            pass
    monkeypatch.setattr(matched, "UnknownDynamicsMaze", FakeLaw)
    monkeypatch.setattr(matched, "Shadow", FakeShadow)
    trace = _shadow_trace()
    untouched = copy.deepcopy(trace)
    result = matched.evaluate_trace(trace, 0, "switch", 0, ROOT / "controls/v3/directional.py", policy_reason="caught")
    assert trace == untouched
    assert result["terminal_transition_included"]
    assert result["new_environment_episodes"] == 0
    assert result["policy_trajectory_sha256"] == matched.digest([{k: f[k] for k in ("world", "action", "next_enemies")} for f in trace])
    for inputs in seen:
        assert inputs[0]["last_action"] is None
        assert inputs[1]["last_action"] == trace[0]["action"]
    assert all("known_law" not in r["obs"] for inputs in seen[:2] for r in inputs)
    assert all("known_law" in r["obs"] for r in seen[2])
    for row in result["shadows"].values():
        assert row["stats"]["brier_near"][1] == 18
        assert row["stats"]["brier_near_terminal"][1] == 9
        assert row["stats"]["brier_near_post_switch"][1] == 9
        assert row["stats"]["brier_destination"][0] == pytest.approx(.18)


def test_matched_failed_shadow_preserves_target_denominators(monkeypatch):
    import v3_matched as matched
    class FakeLaw:
        switch_step = None
        def __init__(self, *args, **kwargs):
            pass
        def law_for_transition(self, transition):
            return [1/9] * 9
    class FailedShadow:
        def __init__(self, *args):
            pass
        def query(self, request):
            raise TimeoutError("synthetic comparator failure")
        def close(self):
            pass
    monkeypatch.setattr(matched, "UnknownDynamicsMaze", FakeLaw)
    monkeypatch.setattr(matched, "Shadow", FailedShadow)
    result = matched.evaluate_trace(_shadow_trace(), 0, "uniform", 0, ROOT / "controls/v3/directional.py", policy_reason="invalid")
    assert not result["terminal_transition_included"]
    for row in result["shadows"].values():
        assert row["error"] and row["missing_forecast_frames"] == 2
        assert row["stats"]["brier_near"] == [4.5, 18]
        assert "brier_near_terminal" not in row["stats"]
