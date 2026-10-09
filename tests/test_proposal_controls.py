"""Source-specific intervention checks on synthetic observations, without worlds."""
import ast
from copy import deepcopy
import unittest

from proposal.study_controls import (
    MEMORY, SELECTED, SOURCE_HASHES, candidate_sources, digest,
)


def observation(pos=(0, 0), previous=(0, 0), cells=None, enemies=(), step=0,
                keys=0, opened=False, collected=False):
    cells = cells or {}
    terrain = [[cells.get((pos[0] + dx, pos[1] + dy), 0)
                for dx in range(-2, 3)] for dy in range(-2, 3)]
    grid = [[5 if (pos[0] + dx, pos[1] + dy) in enemies else terrain[dy + 2][dx + 2]
             for dx in range(-2, 3)] for dy in range(-2, 3)]
    return dict(terrain=terrain, grid=grid, step=step, keys=keys, door_open=opened,
                feedback=dict(displacement=[pos[0] - previous[0], pos[1] - previous[1]],
                              collected=collected))


def load(source):
    namespace = {"__name__": "synthetic_candidate"}
    exec(compile(source, "synthetic_candidate.py", "exec"), namespace)
    return namespace


def function_ast(source, name):
    return ast.dump(next(node for node in ast.parse(source).body
                         if isinstance(node, ast.FunctionDef) and node.name == name))


class ProposalControlTests(unittest.TestCase):
    def setUp(self):
        self.sources = candidate_sources()
        self.selected_source = SELECTED.read_text()
        self.selected = load(self.selected_source)

    def test_frozen_ancestry_and_untouched_functions(self):
        self.assertEqual(digest(SELECTED.read_bytes()), SOURCE_HASHES["selected"])
        self.assertEqual(digest(MEMORY.read_bytes()), SOURCE_HASHES["memory"])
        self.assertTrue(self.sources["memory"].startswith(MEMORY.read_text()))
        self.assertTrue(self.sources["local_map_only"].startswith(self.selected_source))
        self.assertEqual(function_ast(self.sources["no_future_risk"], "world_model_step"),
                         function_ast(self.selected_source, "world_model_step"))
        self.assertEqual(function_ast(self.sources["local_map_only"], "planner"),
                         function_ast(self.selected_source, "planner"))

    def test_memory_exports_remembered_terrain_and_only_current_occupancy(self):
        agent = load(self.sources["memory"])
        obs = observation(cells={(-2, -2): -1, (2, 0): 2}, enemies={(1, 0)})
        memory = agent["world_model_step"](None, obs, None)
        rates = deepcopy(memory["rates"])
        self.assertNotIn((-2, -2), memory["believed_map"])
        self.assertEqual(memory["believed_map"][1, 0], 5)
        obs = observation(pos=(-1, 0), previous=(0, 0), step=1)
        memory = agent["world_model_step"](memory, obs, None)
        self.assertEqual(memory["believed_map"][2, 0], 2)
        self.assertEqual(memory["believed_map"][1, 0], 0)
        self.assertEqual(memory["rates"], rates)
        self.assertEqual(memory["updates"], 0)
        action = agent["planner"](memory, dict(obs, predictive_planning=True))
        memory["risk"] = {p: 1. for p in memory["map"]}
        self.assertEqual(agent["planner"](memory, obs), action)

    def test_memory_repairs_key_underfoot_and_closed_door_entry(self):
        agent = load(self.sources["memory"])
        obs = observation(cells={(0, 0): 2})
        memory = agent["world_model_step"](None, obs, None)
        self.assertEqual(agent["planner"](memory, obs), {"move": [0, 0], "interact": True})
        obs = observation(cells={(1, 1): 3}, keys=2)
        memory = agent["world_model_step"](None, obs, None)
        self.assertFalse(agent["_legal"](memory, (0, 0), (1, 1)))
        self.assertTrue(agent["_legal"](memory, (0, 1), (1, 0)))
        self.assertNotEqual(agent["planner"](memory, obs)["move"], [1, 1])
        memory["door_open"] = True
        self.assertTrue(agent["_legal"](memory, (0, 0), (1, 1)))

    def test_local_map_removes_all_old_spatial_fields_but_keeps_other_memory(self):
        agent = load(self.sources["local_map_only"])
        obs = observation(cells={(2, 1): 1, (2, -1): -1}, collected=True, keys=1)
        memory = agent["world_model_step"](None, obs, None)
        obs = observation(cells={(2, 1): 0, (2, -1): -1}, step=1, keys=1)
        memory = agent["world_model_step"](memory, obs, None)
        self.assertIn((2, 1), memory["unstable"])
        self.assertIn((2, -1), memory["outside"])
        previous = (0, 0)
        for tick, pos in enumerate(((-1, 0), (-2, 0), (-3, 0), (-4, 0)), 2):
            obs = observation(pos=pos, previous=previous, step=tick, keys=1)
            memory = agent["world_model_step"](memory, obs, None)
            previous = pos
        expected_fields = set(self.selected["world_model_step"](None, observation(), None))
        self.assertEqual(set(memory), expected_fields)  # No private old-map copy.
        visible = {(x, y) for x in range(-6, -1) for y in range(-2, 3)}
        for field in ("terrain_map", "believed_map", "seen_at", "enemy_sightings", "unstable", "outside"):
            self.assertLessEqual(set(memory[field]), visible, field)
        self.assertEqual(memory["agent_pos"], (-4, 0))
        self.assertEqual(memory["keys"], 1)
        self.assertIn((0, 0), memory["keys_collected"])
        self.assertEqual(memory["visits"][0, 0], 2)

    def test_local_map_changes_action_when_goal_leaves_window(self):
        local = load(self.sources["local_map_only"])
        actions = []
        for agent in (self.selected, local):
            obs = observation(cells={(2, 0): 2})
            memory = agent["world_model_step"](None, obs, None)
            obs = observation(pos=(-1, 0), step=1, cells={(2, 0): 2})
            memory = agent["world_model_step"](memory, obs, None)
            actions.append(agent["planner"](memory, obs))
        self.assertEqual(actions[0]["move"], [1, -1])
        self.assertEqual(actions[1]["move"], [-1, -1])

    def test_no_future_risk_preserves_updater_and_changes_prediction_driven_action(self):
        no_risk = load(self.sources["no_future_risk"])
        obs = observation(cells={(1, 0): 2}, enemies={(2, 1)})
        original_memory = self.selected["world_model_step"](None, obs, None)
        intervened_memory = no_risk["world_model_step"](None, obs, None)
        self.assertEqual(original_memory, intervened_memory)
        self.assertEqual(self.selected["planner"](original_memory, obs)["move"], [0, -1])
        self.assertEqual(no_risk["planner"](intervened_memory, obs)["move"], [1, 0])
        self.assertEqual(original_memory, intervened_memory)  # Planning has no extra state.
        # A directly occupied objective remains excluded despite removing future risk.
        obs = observation(cells={(1, 0): 2}, enemies={(1, 0)}, step=1)
        original_memory = self.selected["world_model_step"](original_memory, obs, None)
        intervened_memory = no_risk["world_model_step"](intervened_memory, obs, None)
        self.assertEqual(original_memory, intervened_memory)
        self.assertNotEqual(no_risk["planner"](intervened_memory, obs)["move"], [1, 0])


if __name__ == "__main__":
    unittest.main()
