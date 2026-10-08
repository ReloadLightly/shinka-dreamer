"""Focused transition and information-boundary checks for the new experiment."""
import math
import copy
from pathlib import Path
import random

from dreamer.evaluation_v3 import Candidate, aggregate, run_episode
from dreamer.world import DynamicMaze, MOVES, stream_seed
from dreamer.world_v3 import REGIMES, UNIFORM, UnknownDynamicsMaze

ROOT = Path(__file__).resolve().parents[1]
WAIT = {"move": [0, 0], "interact": False}


def test_layout_laws_and_switch_are_independent_tagged_streams():
    for seed in range(30):
        original = DynamicMaze(seed)
        worlds = [UnknownDynamicsMaze(seed, regime=r) for r in REGIMES]
        assert all(w.snapshot() == original.snapshot() for w in worlds)
        stationary, switching = worlds[1:]
        assert stationary.initial_law == switching.initial_law
        assert worlds[0].initial_law == UNIFORM
        assert 25 <= switching.switch_step <= 75
        assert switching.switch_step == random.Random(stream_seed(seed, 'v3:switch-time')).randint(25, 75)
        for law in (switching.initial_law, switching.replacement_law):
            assert abs(sum(law) - 1) < 1e-12
            assert max(law) <= .6
            assert -sum(p * math.log(p) for p in law) >= 1.2
        assert sum(abs(a-b) for a, b in zip(switching.initial_law, switching.replacement_law))/2 >= .3
        assert switching.law_for_transition(switching.switch_step-1) == switching.initial_law
        assert switching.law_for_transition(switching.switch_step) == switching.replacement_law
        assert len({stream_seed(seed, tag) for tag in ('layout', 'v3:law:initial', 'v3:law:replacement',
                    'v3:switch-time', 'v3:enemy-innovations', 'v3:forecast-targets', 'v3:candidate')}) == 7


def test_blocked_probability_stays_and_fixed_draw_count():
    env = UnknownDynamicsMaze(1)
    env.agent_pos = (4, 4)
    env.enemies = [(7, 7), (8, 7), (9, 7)]
    env.dynamic_walls = set()
    for y in range(3, 10):
        for x in range(3, 12):
            env.grid[y][x] = 0
    for x in (7, 8, 9):
        env.grid[8][x] = 1
    env.initial_law = tuple(float(m == (0, 1)) for m in MOVES)
    expected = list(env.enemies)
    for _ in range(4):
        env.step(WAIT)
        assert env.enemies == expected  # No redistribution onto legal attempts.
    assert env.enemy_rng.draws == 12
    env.agent_pos = env.enemies[0]
    env.step(WAIT)
    assert env.reason == 'caught'
    assert env.enemy_rng.draws == 15  # Including terminal pre-movement collision.


def test_switch_uses_replacement_on_recorded_transition_and_reset():
    env = UnknownDynamicsMaze(82, regime='switch')
    snapshot, law, switch = copy.deepcopy(env.snapshot()), env.initial_law, env.switch_step
    env.agent_pos, env.enemies, env.dynamic_walls = (3, 3), [(7, 7)]*3, set()
    for y in range(2, 10):
        for x in range(2, 10):
            env.grid[y][x] = 0
    env.initial_law = tuple(float(m == (0, 0)) for m in MOVES)
    env.replacement_law = tuple(float(m == (1, 0)) for m in MOVES)
    env.steps = switch - 2
    env.step(WAIT)
    assert env.enemies == [(7, 7)]*3
    assert env.current_law == env.replacement_law
    env.step(WAIT)
    assert env.steps == switch and env.enemies == [(8, 7)]*3
    env.reset()
    assert env.snapshot() == snapshot and env.initial_law == law and env.switch_step == switch


def test_candidate_boundary_and_tagged_randomness(tmp_path):
    probe = tmp_path / 'probe.py'
    probe.write_text('''
import random, sys
INITIAL = random.random()
def world_model_step(m,o,a): return None
def planner(m,o): return {'move':[0,0], 'interact':False}
def export_model(m,o):
    return {'default_enemy':0.25, 'keys':sorted(o), 'argv':sys.argv, 'initial':INITIAL}
''')
    r = run_episode(probe, 2014727, replay=True, regime='switch')
    public_keys = {'grid','terrain','step','health','keys','door_open','feedback','learn','predictive_planning'}
    assert all(set(t['model']['keys']) == public_keys for t in r['trace'])
    assert all(t['model']['argv'] == ['candidate.py'] for t in r['trace'])
    for group in ('near', 'audit'):
        total, n = r['stats']['brier_' + group]
        positive, n2 = r['stats']['prevalence_' + group]
        assert n == n2 and abs(total - (.0625*n + .5*positive)) < 1e-8
    assert r['enemy_innovation_draws'] == 3*r['steps']
    assert r['combined_score'] == r['task']
    known = run_episode(probe, 2014727, variant='known_law', replay=True, regime='switch')
    assert all(set(t['model']['keys']) == public_keys | {'known_law'} for t in known['trace'])
    initial = []
    for candidate_seed in (1, 1, 2):
        with Candidate(probe, candidate_seed) as worker:
            initial.append(worker.query({'obs': {}})['model']['initial'])
    assert initial[0] == initial[1] != initial[2]


def test_agent_random_consumption_cannot_shift_world_or_audit(tmp_path):
    programs = []
    for extra in (0, 57):
        path = tmp_path / f'consume{extra}.py'
        path.write_text(f'''
import random
def world_model_step(m,o,a):
    for _ in range({extra}): random.random()
    return None
def planner(m,o): return {{'move':[0,0], 'interact':False}}
def export_model(m,o): return {{'default_enemy':0.1}}
''')
        programs.append(path)
    a, b = [run_episode(p, 2014728, replay=True, regime='switch') for p in programs]
    assert a['stats'] == b['stats'] and a['trace'] == b['trace']


def test_existing_seed_executes_all_regimes_and_invalid_is_zero(tmp_path):
    rows = [run_episode(ROOT / 'initial.py', 872634, regime=regime) for regime in REGIMES]
    assert all(r['error'] is None for r in rows)
    result = aggregate(rows)
    assert result['public']['episodes'] == 3
    assert set(result['public']['regimes']) == set(REGIMES)
    assert isinstance(result['text_feedback'], str)
    assert result['combined_score'] == sum(r['task'] for r in rows)/3
    bad = tmp_path / 'bad.py'
    bad.write_text("raise RuntimeError('invalid candidate')")
    invalid = run_episode(bad, 872634)
    assert invalid['reason'] == 'invalid' and invalid['combined_score'] == 0
    assert aggregate([invalid])['public']['episodes'] == 1
