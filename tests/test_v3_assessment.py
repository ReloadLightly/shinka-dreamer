import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from dreamer.evaluation_v3 import evaluation_identity
from dreamer.provenance import sha256
from scripts import v3_assessment as assessment
from scripts.v3_select import candidates_after_completion, immutable_source


def database(path, rows):
    with sqlite3.connect(path) as db:
        db.execute('create table programs (id text, generation integer, code text, correct integer, combined_score real)')
        db.executemany('insert into programs values (?,?,?,?,?)', rows)


def test_selection_requires_complete_slots_counts_seed_clones_and_deduplicates(tmp_path):
    path = tmp_path / 'programs.sqlite'
    rows = [('seed0', 0, '#seed', 1, .2), ('seed1', 0, '#seed', 1, .2),
            ('a', 1, '#a', 1, .9), ('b', 2, '#a', 1, .9), ('c', 3, '#c', 0, 1.)]
    database(path, rows)
    with pytest.raises(ValueError, match='all 50'):
        candidates_after_completion(path)
    selected, inventory = candidates_after_completion(path, total_slots=4)
    assert [r['generation'] for r in selected] == [1, 0]
    assert inventory['total_slots'] == 4 and inventory['valid_slots'] == 3 and inventory['failed_slots'] == 1
    assert inventory['native_rows_including_seed_clones'] == 5
    frozen = tmp_path / 'selected.py'
    immutable_source(frozen, '#a')
    immutable_source(frozen, '#a')
    with pytest.raises(ValueError, match='Frozen source'):
        immutable_source(frozen, '#c')


def test_arbitrary_nested_parameter_audit_missing_is_not_frozen():
    def row():
        return {'trace': [{'world': {'agent': [2, 3], 'origin': [1, 1]}, 'action': {}, 'next_enemies': [],
                          'obs': {}, 'model': {'learning': {'posterior': {'alpha': i}},
                                              'position': [1, 2], 'terrain': []}} for i in (1, 2)]}
    adaptive = assessment.compact_trace(row(), ['learning', 'posterior'])['audit']
    assert adaptive['parameters_exported'] and not adaptive['parameters_constant']
    assert adaptive['parameter_change_steps'] == 1 and adaptive['localization_errors'] == 0
    absent = assessment.compact_trace(row(), 'weights_that_are_not_exported')['audit']
    assert not absent['parameters_exported'] and absent['parameters_constant'] is None
    assert assessment.compact_trace(row())['audit']['parameters_constant'] is None
    privileged = row()
    privileged.update(variant='known_law', learning={'attempt_probabilities': [1/9]*9})
    compact = assessment.compact_trace(privileged, ['learning', 'posterior'])
    assert 'attempt_probabilities' not in compact['learning']
    assert compact['audit']['first_parameters'] is None


def plan_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(assessment, 'ROOT', tmp_path)
    private = tmp_path / 'results/private'
    private.mkdir(parents=True)
    source = tmp_path / 'source.py'
    source.write_text('# frozen test source')
    selection = tmp_path / 'selection.json'
    selection.write_text('{}')
    design = tmp_path / 'design.json'
    design.write_text('{}')
    analysis = tmp_path / 'analysis.py'
    analysis.write_text('# analysis')
    size = tmp_path / 'size.json'
    size.write_text('{"chosen":3}')
    ref = lambda p: {'path': str(p), 'sha256': sha256(p)}
    plan = {'protocol': assessment.PROTOCOL, 'evaluation': evaluation_identity(), 'sample_size': 3,
            'regimes': list(assessment.REGIMES), 'driver_sha256': sha256(assessment.__file__),
            'selection': ref(selection), 'sample_size_justification': ref(size),
            'analysis_files': {str(analysis): sha256(analysis)}, 'design_files': {str(design): sha256(design)},
            'conditions': [{'name': 'selected', 'program_path': str(source), 'program_sha256': sha256(source), 'variant': 'predictive'}],
            'pool_path': str(private / 'fresh-assessment-seeds.json')}
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    return plan, plan_path, source


def test_pool_draw_follows_freeze_and_resume_rejects_drift(tmp_path, monkeypatch):
    plan, plan_path, source = plan_fixture(tmp_path, monkeypatch)
    out = tmp_path / 'assessment'
    with pytest.raises(ValueError, match='--reserve-pool'):
        assessment.reserve_pool(plan, plan_path, out)
    assert not Path(plan['pool_path']).exists()
    seeds = iter([3, 3, 5, 7])
    monkeypatch.setattr(assessment.secrets, 'randbits', lambda _: next(seeds))
    result, info = assessment.reserve_pool(plan, plan_path, out, reserve=True)
    assert result == [3, 5, 7]
    assert assessment.reserve_pool(plan, plan_path, out)[0] == result
    source.write_text('# changed')
    with pytest.raises(ValueError, match='condition source changed'):
        assessment.reserve_pool(plan, plan_path, out, reserve=True)


def test_pool_reservation_recovers_crash_after_pool_write(tmp_path, monkeypatch):
    plan, plan_path, _ = plan_fixture(tmp_path, monkeypatch)
    out = tmp_path / 'assessment'
    seeds = iter([23, 25, 27])
    monkeypatch.setattr(assessment.secrets, 'randbits', lambda _: next(seeds))
    original_record = assessment.atomic_create
    def fail_manifest(path, value):
        if Path(path).name == 'pool-manifest.json':
            raise RuntimeError('injected power loss after pool write')
        return original_record(path, value)
    monkeypatch.setattr(assessment, 'atomic_create', fail_manifest)
    with pytest.raises(RuntimeError, match='power loss'):
        assessment.reserve_pool(plan, plan_path, out, reserve=True)
    monkeypatch.setattr(assessment, 'atomic_create', original_record)
    assert assessment.reserve_pool(plan, plan_path, out)[0] == [23, 25, 27]


def test_completed_condition_survives_partial_case_and_failed_run_counts(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('#source')
    conditions = [{'name': 'a', 'variant': 'predictive', 'program_path': str(source), 'program_sha256': sha256(source)},
                  {'name': 'b', 'variant': 'frozen', 'program_path': str(source), 'program_sha256': sha256(source)}]
    called = []
    def failing(program, seed, variant, **kwargs):
        called.append(variant)
        if variant == 'frozen':
            raise KeyboardInterrupt('injected interruption')
        row = assessment.failure_row(conditions[0], 'stationary', RuntimeError('bad agent'), .01)
        row['trace'] = []
        return row
    monkeypatch.setattr(assessment, 'run_episode', failing)
    job = (0, 99, conditions, ['stationary'], str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        assessment.evaluate_case(job)
    assert (tmp_path / 'episode-checkpoints/000000/a--stationary.json').exists()
    def resumed(program, seed, variant, **kwargs):
        called.append(variant)
        raise RuntimeError('candidate invalid')
    monkeypatch.setattr(assessment, 'run_episode', resumed)
    result = assessment.evaluate_case(job)
    assert called == ['predictive', 'frozen', 'frozen']
    assert len(result['conditions']) == 2
    assert all(r['reason'] == 'invalid' and r['combined_score'] == 0 for r in result['conditions'])
    assert all('seed' not in r for r in result['conditions'])
    assert len(list((tmp_path / 'attempts').glob('*.started.json'))) == 3
    assert len(list((tmp_path / 'attempts').glob('*.completed.json'))) == 2


def test_physical_observation_hash_ignores_intervention_instructions():
    import copy
    frame = {'world': {'origin': [0, 0], 'agent': [0, 0]}, 'action': {}, 'next_enemies': [],
             'obs': {'step': 0, 'terrain': [], 'learn': True, 'predictive_planning': True},
             'model': {}}
    first = {'trace': [copy.deepcopy(frame)]}
    frame['obs'].update(learn=False, known_law=[1/9]*9)
    second = {'trace': [frame]}
    a, b = assessment.compact_trace(first)['audit'], assessment.compact_trace(second)['audit']
    assert a['observations_sha256'] == b['observations_sha256']
    assert a['observation_interventions_sha256'] != b['observation_interventions_sha256']


def test_shadow_interruption_keeps_outcome_and_private_trace_without_world_rerun(tmp_path, monkeypatch):
    from scripts import v3_matched
    source = tmp_path / 'source.py'
    source.write_text('# source')
    condition = {'name': 'memory', 'variant': 'memory', 'program_path': str(source), 'program_sha256': sha256(source)}
    spec = {'memory_condition': 'memory', 'source': str(source), 'source_sha256': sha256(source)}
    trace = [{'world': {'origin': [0, 0], 'agent': [0, 0]}, 'action': {},
              'next_enemies': [], 'obs': {}, 'model': {}}]
    calls = []
    def world(*args, **kwargs):
        calls.append('world')
        row = assessment.failure_row(condition, 'stationary', RuntimeError('fixture'), .01)
        row.update(reason='timeout', error=None, trace=trace)
        return row
    def shadow_stop(*args, **kwargs):
        raise KeyboardInterrupt('interrupted shadow')
    monkeypatch.setattr(assessment, 'run_episode', world)
    monkeypatch.setattr(v3_matched, 'evaluate_trace', shadow_stop)
    job = (0, 77, [condition], ['stationary'], str(tmp_path), spec)
    with pytest.raises(KeyboardInterrupt):
        assessment.evaluate_case(job)
    assert (tmp_path / 'episode-checkpoints/000000/memory--stationary.json').exists()
    assert (tmp_path / 'matched-private-traces/000000--stationary.json.gz').exists()
    def completed(trace, seed, regime, case, source, **kwargs):
        assert kwargs['policy_reason'] == 'timeout'
        return {'case': case, 'regime': regime,
                'policy_trajectory_sha256': assessment.digest([{k: f[k] for k in ('world', 'action', 'next_enemies')} for f in trace]),
                'shadows': {}, 'condition_episodes': 3, 'seconds': .1,
                'candidate_cpu_seconds': .2, 'evaluator_cpu_seconds': .03}
    monkeypatch.setattr(v3_matched, 'evaluate_trace', completed)
    result = assessment.evaluate_case(job)
    assert calls == ['world']
    assert result['conditions'][0]['reason'] == 'timeout'
    assert (tmp_path / 'matched/000000--stationary.json').exists()
    assert not (tmp_path / 'matched-private-traces/000000--stationary.json.gz').exists()


def opportunity_frame(step, enemies, displacement=(0, 0), walls=(), parameter=0):
    terrain = [[0]*5 for _ in range(5)]
    for x, y in walls:
        terrain[y+2][x+2] = 1
    grid = [row[:] for row in terrain]
    for x, y in enemies:
        grid[y+2][x+2] = 5
    return {'obs': {'step': step, 'grid': grid, 'terrain': terrain, 'keys': 0, 'door_open': False,
                    'feedback': {'displacement': list(displacement)}},
            'action': {'move': list(displacement), 'interact': False},
            'world': {'origin': [0, 0], 'agent': [0, 0]}, 'next_enemies': [],
            'model': {'learning': {'theta': parameter}}}


def test_blocked_all_stay_is_not_a_direction_information_opportunity():
    walls = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
    trace = [opportunity_frame(0, [(0, 0)], walls=walls),
             opportunity_frame(1, [(0, 0)], walls=walls, parameter=1)]
    result, by_frame = assessment.observed_transition_opportunities(trace)
    counts = result['all']
    assert counts['known_attempt_terrain_source_opportunities'] == 1
    assert counts['multi_destination_source_opportunities'] == 0
    assert counts['fully_observed_transition_opportunities'] == 0 and by_frame == [0, 0]
    compact = assessment.compact_trace({'trace': trace}, 'theta')
    assert compact['audit']['parameter_change_steps'] == 1
    assert compact['audit']['observed_transition_opportunities']['parameter_changes_without_counted_contrast'] == 1


def test_partially_observed_destinations_are_censored_not_full_opportunities():
    # Previous enemy at origin-relative (1,0); agent then moves left. The enemy
    # remains visible at local (2,0), but possible destination (2,0) is out of view.
    trace = [opportunity_frame(24, [(1, 0)]), opportunity_frame(25, [(2, 0)], displacement=(-1, 0))]
    result, by_frame = assessment.observed_transition_opportunities(trace, switch_step=25)
    counts = result['all']
    assert counts['observed_destination_contrast_opportunities'] == 1
    assert counts['fully_observed_transition_opportunities'] == 0
    assert counts['censored_transition_opportunities'] == 1 and by_frame == [0, 1]
    assert result['post_switch']['observed_destination_contrast_opportunities'] == 1
    earlier, _ = assessment.observed_transition_opportunities(trace, switch_step=26)
    assert earlier['post_switch']['observed_destination_contrast_opportunities'] == 0


def test_anonymous_sources_do_not_imply_identified_motion_or_distinct_enemies():
    # The next occupied cell is a possible destination of both prior occupied
    # cells. Count two source-cell opportunities, without assigning correspondence.
    trace = [opportunity_frame(0, [(0, 0), (1, 0)]), opportunity_frame(1, [(0, 0)])]
    result, _ = assessment.observed_transition_opportunities(trace)
    counts = result['all']
    assert counts['prior_visible_source_cells'] == counts['distinct_prior_source_cells'] == 2
    assert counts['observed_destination_contrast_opportunities'] == 2
    assert counts['fully_observed_transition_opportunities'] == 2
    assert counts['observation_steps_with_contrast'] == 1
    assert 'not enemy identity' in result['unit']
    # No consecutive observation means no information proxy for a final frame.
    final_only, _ = assessment.observed_transition_opportunities(trace[:1])
    assert final_only['all']['consecutive_observation_pairs'] == 0


def test_private_pool_complete_partial_is_recovered_without_second_random_draw(tmp_path, monkeypatch):
    plan, plan_path, _ = plan_fixture(tmp_path, monkeypatch)
    out = tmp_path / 'assessment'
    assessment.atomic_create(out / 'frozen-plan.json', plan)
    assessment.atomic_create(out / 'pool-reservation.json', {
        'frozen_plan_sha256': sha256(out / 'frozen-plan.json'), 'original_plan_sha256': sha256(plan_path),
        'pool_path': plan['pool_path'], 'sample_size': plan['sample_size']})
    target = Path(plan['pool_path'])
    target.with_name(target.name + '.partial').write_text('[31, 33, 35]')
    def forbidden_draw(_):
        raise AssertionError('A completed private pool must not be drawn twice')
    monkeypatch.setattr(assessment.secrets, 'randbits', forbidden_draw)
    assert assessment.reserve_pool(plan, plan_path, out)[0] == [31, 33, 35]
    assert not target.with_name(target.name + '.partial').exists()


def test_registered_replays_are_retained_passively_and_required_on_resume(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('# fixture')
    condition = {'name': 'selected', 'variant': 'predictive', 'program_path': str(source), 'program_sha256': sha256(source)}
    spec = {'conditions': ['selected'], 'first_cases': 1}
    calls = []
    def world(*args, **kwargs):
        calls.append('world')
        row = assessment.failure_row(condition, 'switch', RuntimeError('fixture'), .01)
        row.update(reason='timeout', error=None, trace=[opportunity_frame(0, [(0, 0)])])
        return row
    monkeypatch.setattr(assessment, 'run_episode', world)
    job = (0, 77, [condition], ['switch'], str(tmp_path), None, spec)
    row = assessment.evaluate_case(job)['conditions'][0]
    retained = row['audit']['retained_replay']
    replay = tmp_path / retained['path']
    assert retained['path'] == 'replays/000000--switch--selected.json.gz'
    assert retained['sha256'] == sha256(replay)
    assert assessment.evaluate_case(job)['conditions'][0] == row and calls == ['world']
    assessment.evaluate_case((1, 78, [condition], ['switch'], str(tmp_path), None, spec))
    assert calls == ['world', 'world'] and len(list((tmp_path / 'replays').glob('*.json.gz'))) == 1
    replay.unlink()
    with pytest.raises(ValueError, match='Registered passive replay'):
        assessment.evaluate_case(job)
    assert calls == ['world', 'world']


def test_replay_prefix_must_be_registered_with_known_conditions(tmp_path, monkeypatch):
    plan, _, _ = plan_fixture(tmp_path, monkeypatch)
    plan['replays'] = {'conditions': ['selected'], 'first_cases': 4}
    with pytest.raises(ValueError, match='bounded prefix'):
        assessment.verify_plan(plan)
    plan['replays'] = {'conditions': ['missing'], 'first_cases': 1}
    with pytest.raises(ValueError, match='bounded prefix'):
        assessment.verify_plan(plan)
