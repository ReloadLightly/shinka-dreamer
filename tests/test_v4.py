"""Focused physical/sensor/RNG boundary checks; no scored candidate episodes."""
from dreamer.world import MOVES, EMPTY, WALL
from dreamer.world_v3 import UnknownDynamicsMaze
from dreamer.world_v4 import FactorMaze, CONDITIONS
from dreamer.evaluation_v4 import aggregate


def test_invalid_episode_stays_in_all_task_and_outcome_denominators():
    valid = {"regime": "stationary-full", "reason": "escaped", "error": None,
             "combined_score": .9, "task": .9, "keys": 2, "door": True,
             "steps": 50, "model_score": .8, "seconds": 1.,
             "candidate_cpu_seconds": .5, "evaluator_cpu_seconds": .1,
             "stats": {}, "encountered_switch": False,
             "exposure": {"post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}}
    invalid = {**valid, "reason": "invalid", "error": "failed after partial progress",
               "combined_score": 0., "task": .3}
    result = aggregate([valid, invalid])
    assert result["combined_score"] == .45
    assert result["public"]["task"] == .45
    assert result["public"]["episodes"] == 2
    assert result["public"]["invalid"] == 1
    assert result["public"]["escape"] == .5
    assert result["public"]["regimes"]["stationary-full"]["task"] == .45


def test_reference_continuity_and_hidden_boundary():
    for regime in ('uniform','stationary','switch'):
        old=UnknownDynamicsMaze(731,regime=regime)
        new=FactorMaze(731,condition=regime+'-full')
        assert new.snapshot()==old.snapshot()
        for t in (1,25,26,50,51,200):
            assert new.law_for_transition(t)==old.law_for_transition(t)
        obs=new.observe();radius=obs.pop('enemy_visibility_radius')
        assert radius==2 and obs==old.observe()
        assert not {'seed','law','known_law','enemies','switch_step'} & obs.keys()


def test_late_sensor_changes_only_enemy_overlay():
    full=FactorMaze(732,condition='stationary-full')
    late=FactorMaze(732,condition='stationary-late')
    for world in (full,late):
        world.agent_pos=(7,7);world.enemies=[(5,7),(7,8),(9,9)]
    a,b=full.observe(),late.observe()
    assert a['terrain']==b['terrain']
    assert b['grid'][3][2]==5
    assert a['grid'][2][0]==5 and b['grid'][2][0]==b['terrain'][2][0]
    assert a['grid'][4][4]==5 and b['grid'][4][4]==b['terrain'][4][4]
    assert full.initial_law==late.initial_law
    assert full.enemy_rng.rng.getstate()==late.enemy_rng.rng.getstate()


def test_repeat_law_boundaries_and_pairing():
    full=FactorMaze(733,condition='repeat25-full')
    late=FactorMaze(733,condition='repeat25-late')
    assert full.laws==late.laws
    assert full.switch_transitions==late.switch_transitions
    prior=1
    for t in full.switch_transitions:
        assert 20<=t-prior<=30
        prior=t
    first=full.switch_transitions[0]
    assert full.law_for_transition(first-1)==full.laws[0]
    assert full.law_for_transition(first)==full.laws[1]
    for i in range(1,len(full.laws)):
        assert sum(abs(a-b) for a,b in zip(full.laws[i-1],full.laws[i]))/2>=.3
    # Law construction must never consume enemy innovation randomness.
    ordinary=UnknownDynamicsMaze(733,regime='stationary')
    assert full.enemy_rng.rng.getstate()==ordinary.enemy_rng.rng.getstate()


def test_blocked_attempts_stay_and_entry_contact_precedes_innovations():
    world=FactorMaze(734,condition='repeat25-full')
    world.grid=[[EMPTY]*15 for _ in range(15)]
    world.agent_pos=(7,7);world.enemies=[(5,5),(9,9),(10,10)]
    # Exactly one attempted direction, with a blocking wall at its destination.
    world.grid[5][6]=WALL
    east=MOVES.index((1,0))
    world.laws=[tuple(float(i==east) for i in range(9))]*8
    world.step({'move':[0,0],'interact':False})
    assert world.enemies[0]==(5,5) and world.enemy_rng.draws==3
    world.agent_pos=(4,5)
    world.step({'move':[1,0],'interact':False})
    assert world.reason=='caught' and world.enemy_rng.draws==6
