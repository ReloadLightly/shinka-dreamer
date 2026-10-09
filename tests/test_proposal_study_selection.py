"""Nomination decisions use source identities, fixed ties and actual call prefixes."""
import json

from proposal.study_selection import EVALUATOR_SOURCES, ROOT, nominate, read_run, sha


def candidate(generation, source, fitness, task, escaped, calls):
    return {'generation': generation, 'source_sha256': source,
            'all_role_calls_at_acceptance': calls,
            'training_metrics': {'combined_score': fitness, 'task': task, 'escaped': escaped}}


SEED = candidate(0, 'seed', .5, .2, 0, 0)


def test_failed_search_and_empty_prefix_retain_seed():
    assert nominate([], SEED)['nominations'][0]['source_sha256'] == 'seed'
    proposed = candidate(1, 'child', .9, .9, 5, 4)
    prefix = nominate([proposed], SEED, call_prefix=3)
    assert prefix['seed_fallback']
    assert prefix['nominations'][0]['generation'] == 0


def test_metric_winners_deduplicate_shared_source_and_combine_reasons():
    first = candidate(1, 'same-code', .91, .9, 4, 2)
    duplicate = candidate(3, 'same-code', .91, .9, 4, 7)
    other = candidate(2, 'more-escapes', .8, .8, 5, 4)
    chosen = nominate([duplicate, other, first], SEED)
    assert chosen['eligible_distinct_descendants'] == 2
    assert [(row['generation'], row['source_sha256']) for row in chosen['nominations']] == [(1, 'same-code'), (2, 'more-escapes')]
    assert chosen['nominations'][0]['nomination_reasons'] == ['highest training combined_score', 'highest training task']


def test_all_metric_ties_choose_earlier_generation_independent_of_input_order():
    earlier = candidate(2, 'earlier', .9, .8, 4, 6)
    later = candidate(5, 'later', .9, .8, 4, 8)
    selected = nominate([later, earlier], SEED)
    assert len(selected['nominations']) == 1
    assert selected['nominations'][0]['source_sha256'] == 'earlier'


def test_call_prefix_includes_boundary_and_excludes_later_superior_candidate():
    boundary = candidate(2, 'at-prefix', .7, .6, 2, 5)
    later = candidate(3, 'after-prefix', .99, .99, 5, 6)
    full = nominate([later, boundary], SEED)
    prefix = nominate([later, boundary], SEED, call_prefix=5)
    assert full['nominations'][0]['source_sha256'] == 'after-prefix'
    assert prefix['nominations'][0]['source_sha256'] == 'at-prefix'
    assert prefix['eligible_distinct_descendants'] == 1


def test_terminal_failure_before_seed_evaluation_is_counted_without_invented_metrics(tmp_path):
    spec = {'run_id': 'full-r1', 'arm': 'full', 'native_seed': 7109260, 'replicate': 1}
    limits = {'all_role_provider_calls': 24, 'provider_seconds': 5400,
              'wall_minutes': 120, 'total_candidate_slots': 25}
    config = {'run_id': spec['run_id'], 'arm': spec['arm'],
              'native_sampling_seed': spec['native_seed'],
              'evaluator': 'namazu-proposal-reconstruction-v1', 'search_cases': list(range(5)),
              'limits': {'calls': 24, 'remote_elapsed_seconds': 5400}, 'wall_minutes': 120,
              'evolution': {'num_generations': 25},
              'source_sha256': {name: sha(ROOT / name) for name in EVALUATOR_SOURCES}}
    (tmp_path / 'campaign-manifest.json').write_text(json.dumps(config))
    identity = sha(tmp_path / 'campaign-manifest.json')
    (tmp_path / 'execution-grant.json').write_text(json.dumps({'manifest_sha256': identity}))
    (tmp_path / 'execution-resources.json').write_text(json.dumps({
        'manifest_sha256': identity, 'ended_utc': '2026-10-09T12:00:00+00:00',
        'status': 'interrupted_or_failed', 'returncode': 1, 'stop_reason': 'fixture route unavailable'}))
    record = read_run(tmp_path, spec, {'search_design': {'per_run_limits': limits}}, {})
    selected = nominate(record['valid_descendants'], record['seed'])
    assert record['actual_all_role_calls'] == 0
    assert record['worker_returncode'] == 1
    assert selected['seed_fallback']
    assert selected['nominations'][0]['training_metrics'] is None
