import copy
import pickle

import pytest

from dreamer.world_v3 import UnknownDynamicsMaze, _AttemptStream, UNIFORM
from scripts.run1_branches import observable_encounter, branch_return, summarize
from scripts.run1_state_digest import state_digest


def test_structurally_equal_set_orders_have_same_canonical_not_pickle_digest():
    first = {"enemies": set(i * 32 for i in range(3)), "state": ((1, 2), [0.0, -0.0])}
    second = {"enemies": set(reversed([i * 32 for i in range(3)])), "state": ((1, 2), [0.0, -0.0])}
    assert first == second
    assert pickle.dumps(first, protocol=4) != pickle.dumps(second, protocol=4)
    assert state_digest(first) == state_digest(second) == state_digest(copy.deepcopy(first))
    second["state"][1][0] = 1e-15
    assert state_digest(first) != state_digest(second)


def observation():
    obs = {"step": 10, "keys": 0, "door_open": False,
           "grid": [[1] * 5 for _ in range(5)],
           "terrain": [[1] * 5 for _ in range(5)]}
    obs["grid"][2][2] = obs["terrain"][2][2] = 0
    obs["grid"][0][0], obs["terrain"][0][0] = 5, 0
    return obs


def test_observable_encounter_rejects_single_destination_and_corner_cutting():
    obs = observation()
    assert not observable_encounter(obs)
    obs["grid"][1][1] = obs["terrain"][1][1] = 0
    assert not observable_encounter(obs)  # Diagonal blocked by cardinal walls.
    obs["grid"][2][1] = obs["terrain"][2][1] = 0
    assert observable_encounter(obs)
    obs["step"] = 9
    assert not observable_encounter(obs)


def test_observable_door_interaction_matches_current_keys():
    obs = observation()
    obs["grid"][2][1] = obs["terrain"][2][1] = 3
    assert not observable_encounter(obs)
    obs["keys"] = 2
    assert observable_encounter(obs)


def test_deepcopy_preserves_enemy_owner_and_pairs_innovations_without_a_world_episode():
    # Construct only the RNG ownership graph, not a maze or transition.
    env = UnknownDynamicsMaze.__new__(UnknownDynamicsMaze)
    env.initial_law, env.switch_step, env.steps = UNIFORM, None, 10
    env.enemy_rng = _AttemptStream(env, 19)
    first, second = copy.deepcopy(env), copy.deepcopy(env)
    assert first.enemy_rng.owner is first and second.enemy_rng.owner is second
    assert first.enemy_rng.owner is not env
    draws = [first.enemy_rng.rng.random() for _ in range(36)]
    assert draws == [second.enemy_rng.rng.random() for _ in range(36)]
    assert env.enemy_rng.draws == 0


def test_return_components_and_invalid_denominators_are_distinct():
    assert branch_return("caught", 0, False, 12) == pytest.approx(-1.12)
    assert branch_return("escaped", 1, True, 3) == pytest.approx(1.47)
    state = {"state": "s", "regime": "uniform", "case": 0,
             "stratum": "disagreement", "actions": {}}
    valid = {"labels": ["learned"], "valid": True, "collision": True,
             "escaped": False, "return": -1.12, "failure_or_invalid": True,
             "keys_delta": 0, "door_opened": False, "transitions": 12}
    invalid = {**valid, "labels": ["frozen_prior", "known_law"],
               "valid": False, "collision": False, "transitions": None}
    selection = {"states": [state], "quota_counts": {},
                 "reconstructed_transitions": 0, "candidate_operations": 1}
    results = [{"state": "s", "branches": [valid, invalid],
                "actual_world_transitions": 12, "candidate_operations": 1}]
    summary = summarize({"regimes": ["uniform"]}, selection, results)
    rows = summary["regimes"]["uniform"]["states"][0]["outcomes"]
    assert rows["frozen_prior"]["attempted_futures"] == 1
    assert rows["frozen_prior"]["invalid"] == 1
    assert rows["frozen_prior"]["collision"] == 0
    assert rows["frozen_prior"]["failure_or_invalid_rate"] == 1
    assert rows["learned"]["collision"] == 1
    assert summary["invalid_unique_action_branches"] == 1  # Duplicate labels alias.
