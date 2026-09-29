import importlib.util
from pathlib import Path
import tempfile

from dreamer.evaluation import Candidate, run_episode
from dreamer.world import DynamicMaze, valid_action

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed", ROOT / "initial.py")
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)


def test_generation_reset_and_objective_feasibility():
    for case in range(100):
        env = DynamicMaze(case)
        snapshot = env.snapshot()
        assert all(x not in (0, 14) and y not in (0, 14) for x, y in env.dynamic_walls)
        # Cardinal routes in the all-closed state work in every later wall phase.
        seen, frontier = {env.agent_pos}, [env.agent_pos]
        while frontier:
            p = frontier.pop()
            for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                q = p[0]+d[0], p[1]+d[1]
                if q not in seen and env.can_move(p, d):
                    seen.add(q)
                    frontier.append(q)
        assert env.exit_pos not in seen
        assert sum(env.grid[y][x] == 2 for x, y in seen) == 2
        assert any(abs(x-env.door_pos[0])+abs(y-env.door_pos[1]) == 1 for x, y in seen)
        env.step({"move": [0, 0], "interact": True})
        env.reset()
        assert snapshot == env.snapshot()


def test_actions_corners_door_and_collision_order():
    assert not valid_action({"move": [2, 0], "interact": False})
    assert not valid_action({"move": [True, 0], "interact": False})
    assert not valid_action({"move": [0, 0], "interact": 1})
    env = DynamicMaze(4)
    env.agent_pos, env.enemies = (5, 5), [(6, 5)]
    for y in range(4, 8):
        for x in range(4, 8):
            env.grid[y][x] = 0
    env.grid[6][5] = 1
    assert not env.can_move((5, 5), (1, 1))
    env.step({"move": [1, 0], "interact": False})
    assert env.reason == "caught"  # Enemy cannot move away to undo collision.
    env = DynamicMaze(4)
    env.enemies = []
    env.door_pos, env.exit_pos, env.agent_pos = (6, 5), (7, 5), (5, 5)
    env.grid[5][5:8] = [0, 3, 4]
    env.step({"move": [1, 0], "interact": True})
    assert env.agent_pos == (5, 5) and env.feedback["displacement"] == [0, 0]
    env.keys_collected = 2
    env.step({"move": [1, 0], "interact": True})
    assert env.door_open and env.agent_pos == (6, 5)
    env.step({"move": [1, 0], "interact": True})
    assert env.reason == "escaped"


def test_dynamic_occupied_closure_skips_until_next_tick():
    env = DynamicMaze(7)
    p = next(iter(env.dynamic_walls))
    env.grid[p[1]][p[0]], env.agent_pos, env.enemies = 0, p, []
    env.steps = 24
    env.step({"move": [0, 0], "interact": False})
    assert env.grid[p[1]][p[0]] == 0
    env.agent_pos = env.origin
    env.step({"move": [0, 0], "interact": False})
    assert env.grid[p[1]][p[0]] == 0


def test_freeze_keeps_localization_mapping_and_inventory():
    env = DynamicMaze(10000)
    learned, frozen = None, None
    for _ in range(35):
        obs = env.observe()
        learned = seed.world_model_step(learned, obs, None)
        obs["learn"] = False
        frozen = seed.world_model_step(frozen, obs, None)
        assert learned["pos"] == frozen["pos"]
        assert learned["map"] == frozen["map"]
        assert learned["keys"] == frozen["keys"]
        env.step(seed.planner(learned, obs))
        if env.done:
            break
    assert learned["updates"] > 0 and frozen["updates"] == 0
    assert learned["rates"] != frozen["rates"]
    assert len(frozen["map"]) > 25


def test_os_boundary_blocks_private_files_processes_and_network(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("withheld")
    candidate = tmp_path / "probe.py"
    candidate.write_text('''
import os, socket
def world_model_step(m, o, a):
    outcomes = []
    for f in [lambda: open(%r).read(), lambda: open('/proc/self/environ').read(),
              lambda: socket.socket(), lambda: os.fork()]:
        try:
            f()
            outcomes.append(False)
        except PermissionError:
            outcomes.append(True)
    return outcomes
def planner(m, o):
    return {'move': [0, 0] if all(m) else [9, 9], 'interact': False}
''' % str(secret))
    with Candidate(candidate) as worker:
        assert worker.query({"obs": {}})["action"]["move"] == [0, 0]


def test_preoutcome_scoring_and_controlled_rng(tmp_path):
    candidate = tmp_path / "constant.py"
    candidate.write_text('''
def world_model_step(m,o,a): return None
def planner(m,o): return {'move':[0,0], 'interact':False}
def export_model(m,o): return {'default_enemy':0.25}
''')
    result = run_episode(candidate, 10000)
    # For p=.25, squared error equals .0625 + .5*y exactly.
    for group in ("near", "audit"):
        b, n = result["stats"]["brier_"+group]
        positives, n2 = result["stats"]["prevalence_"+group]
        assert n == n2 and abs(b - (.0625*n + .5*positives)) < 1e-8
    a = run_episode(ROOT / "initial.py", 10000, "no_planning")
    b = run_episode(ROOT / "initial.py", 10000, "frozen_no_planning")
    assert (a["reason"], a["steps"], a["keys"]) == (b["reason"], b["steps"], b["keys"])
    assert a["stats"]["prevalence_near"] == b["stats"]["prevalence_near"]
