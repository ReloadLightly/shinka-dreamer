import numpy as np
from scripts.v3_analysis import paired_binary, forecast_pair, holm


def test_invalid_remains_a_non_escape_in_paired_denominator():
    result=paired_binary([True,False,False,True],[False,True,False,True])
    assert result['cases']==4 and result['left_only']==result['right_only']==1
    assert result['difference']==0 and result['mcnemar_exact_p']==1


def test_pooled_episode_resampling_does_not_average_episode_ratios():
    left=[{'stats':{'brier_near':[1,2]}},{'stats':{'brier_near':[1,10]}}]
    right=[{'stats':{'brier_near':[2,2]}},{'stats':{'brier_near':[2,10]}}]
    weights=np.array([[1,1],[2,0],[0,2]],float)
    result=forecast_pair(left,right,'brier_near',weights)
    assert abs(result['left_loss']-1/6)<1e-12
    assert result['relative_reduction']==.5
    assert result['relative_reduction_ci95']==[.5,.5]


def test_holm_preserves_original_order():
    assert holm([.04,.001])==[.04,.002]


def test_paired_binary_confidence_level_is_explicit_and_widens():
    left = [True]*4 + [False]*16
    right = [False]*3 + [True]*2 + [False]*15
    usual = paired_binary(left, right)
    wider = paired_binary(left, right, alpha=.025)
    assert usual['ci95'] == usual['confidence_interval']
    assert wider['confidence_level'] == .975 and 'ci95' not in wider
    assert wider['confidence_interval'][0] <= usual['ci95'][0]
    assert wider['confidence_interval'][1] >= usual['ci95'][1]


def test_regime_interaction_preserves_shared_cases_and_marks_degenerate_bootstrap():
    from scripts.v3_analysis import regime_interactions
    plan = {'regime_interactions': [{'left': 'selected', 'right': 'frozen',
                                   'regime_a': 'switch', 'regime_b': 'uniform'}]}
    def rows(values):
        return [{'reason': 'escaped' if value else 'invalid'} for value in values]
    data = {(regime, condition): rows(values)
            for regime in ('switch', 'uniform')
            for condition, values in [('selected', [1, 0, 0]), ('frozen', [0, 1, 0])]}
    weights = np.array([[1, 1, 1], [3, 0, 0], [0, 3, 0]], float)
    key = 'switch-minus-uniform/selected-minus-frozen'
    result = regime_interactions(data, plan, weights)[key]
    assert result['difference'] == 0. and result['cases'] == 3
    assert result['ci95_shared_case_bootstrap'] == [0., 0.]
    assert result['ci95_conservative'][0] < 0. < result['ci95_conservative'][1]
    assert result['sparse_discordance_warning'] and result['bootstrap_degenerate']
    assert result['cases_with_nonzero_interaction'] == 0
    # A single switched-regime rescue changes the paired interaction by 1/3;
    # the invalid case remains in both regimes' denominator.
    data['switch', 'selected'][2]['reason'] = 'escaped'
    changed = regime_interactions(data, plan, weights)[key]
    assert changed['difference'] == 1/3 and changed['cases_with_nonzero_interaction'] == 1
    assert not changed['bootstrap_degenerate']
    first = changed['component_effects']['switch']['confidence_interval']
    second = changed['component_effects']['uniform']['confidence_interval']
    assert changed['ci95_conservative'] == [first[0]-second[1], first[1]-second[0]]


def matched_fixture():
    import copy
    names = ('fitted_frozen', 'fitted_online', 'known_law')
    plan = {'study': 'fixture', 'sample_size': 2, 'regimes': ['stationary', 'switch'],
            'matched': {'memory_condition': 'memory'}}
    records, policy = [], []
    for regime in plan['regimes']:
        for case, count in enumerate((2, 10)):
            trace_hash = f'{regime}-trajectory-{case}'
            record = {'case': case, 'regime': regime, 'policy_trajectory_sha256': trace_hash,
                      'policy_frames': 2, 'encountered_switch': regime == 'switch' and case == 1,
                      'terminal_transition_included': True, 'shadows': {}}
            for name, scale in zip(names, (1., .5, .25)):
                stats = {'brier_near': [2*scale, count]}
                if case == 1 and regime == 'switch':
                    stats['brier_near_post_switch'] = [2*scale, count]
                record['shadows'][name] = {'case': case, 'regime': regime, 'error': None,
                    'missing_forecast_frames': 0, 'frames': 2, 'stats': stats,
                    'bins': {str(case): {'brier': [2*scale, count]}},
                    'audit': {'observations_sha256': f'observations-{case}', 'map_position_sha256': f'map-{case}',
                              'parameters_exported': True, 'parameters_constant': name != 'fitted_online',
                              'parameter_change_steps': int(name == 'fitted_online'), 'first_parameters': [0.]}}
            records.append(record)
            policy.append({'case': case, 'regime': regime, 'condition': 'memory',
                           'audit': {'trajectory_sha256': trace_hash}})
    return records, plan, policy


def test_matched_bootstrap_pools_counts_and_preserves_cross_regime_resamples(monkeypatch):
    from scripts import v3_analysis as analysis
    monkeypatch.setattr(analysis, 'BOOTSTRAP_REPLICATES', 200)
    rows, plan, policy = matched_fixture()
    result = analysis.analyze_matched(rows, plan, policy)
    for regime in plan['regimes']:
        entry = result['regimes'][regime]
        pair = entry['pairs']['fitted_online-minus-fitted_frozen']['forecasts']['brier_near']
        assert abs(pair['left_loss'] - 1/6) < 1e-12
        assert pair['relative_reduction'] == .5 and pair['relative_reduction_ci95'] == [.5, .5]
        assert pair['left_contributing_episodes'] == 2
        assert entry['audits']['policy_trajectory_verified_against_outcomes'] == 2
        assert entry['audits']['frozen_parameters_constant_episodes'] == 2
        later = entry['bins'][1]['conditions']['fitted_online']
        assert later['contributing_episodes'] == 1 and later['targets'] == 10
        assert 0 < later['bootstrap_retained'] < 200
        assert 'Survivor-conditioned' in entry['bins'][1]['scope']
    assert result['regimes']['stationary']['bins'] == result['regimes']['switch']['bins']
    post = result['regimes']['switch']['forecasts']['fitted_online']['brier_near_post_switch']
    assert post['contributing_episodes'] == 1 and post['targets'] == 10


def test_matched_rejects_wrong_trajectory_and_does_not_hide_missing_or_failed_shadows(monkeypatch):
    import copy
    import pytest
    from scripts import v3_analysis as analysis
    monkeypatch.setattr(analysis, 'BOOTSTRAP_REPLICATES', 20)
    rows, plan, policy = matched_fixture()
    wrong = copy.deepcopy(policy)
    wrong[0]['audit']['trajectory_sha256'] = 'different experience'
    with pytest.raises(ValueError, match='disagrees'):
        analysis.analyze_matched(rows, plan, wrong)
    failed = copy.deepcopy(rows)
    failed[0]['shadows']['fitted_online']['error'] = 'Worker failed; .5 fallback already scored'
    failed[0]['shadows']['fitted_online']['missing_forecast_frames'] = 2
    result = analysis.analyze_matched(failed, plan, policy)['regimes']['stationary']
    assert result['audits']['shadow_error_episodes']['fitted_online'] == 1
    assert result['pairs']['fitted_online-minus-fitted_frozen']['forecasts']['brier_near']['left_targets'] == 12
    missing = copy.deepcopy(rows)
    del missing[0]['shadows']['fitted_online']
    result = analysis.analyze_matched(missing, plan, policy)['regimes']['stationary']
    assert not result['complete_matched_prediction_evidence'] and result['pairs'] == {}
    assert len(result['audits']['missing_records']) == 1


def test_selected_matched_requires_every_case_match_without_selecting_subset():
    import copy
    from scripts.v3_analysis import selected_matched
    plan = {'sample_size': 2, 'regimes': ['switch'], 'conditions': [
        {'name': 'selected_fixed_risk', 'variant': 'no_planning'},
        {'name': 'selected_frozen_fixed_risk', 'variant': 'frozen_no_planning'}]}
    left = [{'case': i, 'stats': {'brier_near': [1., 4]}, 'bins': {}, 'error': None,
             'audit': {'trajectory_sha256': str(i), 'observations_sha256': str(i), 'map_position_sha256': str(i)}}
            for i in range(2)]
    right = copy.deepcopy(left)
    data = {('switch', plan['conditions'][0]['name']): left, ('switch', plan['conditions'][1]['name']): right}
    weights = np.array([[1., 1.], [2., 0.], [0., 2.]])
    assert selected_matched(data, plan, weights)['switch']['available']
    right[1]['audit']['trajectory_sha256'] = 'changed action'
    result = selected_matched(data, plan, weights)['switch']
    assert not result['available'] and result['trajectory_matched_episodes'] == 1
    assert 'forecasts' not in result


def test_export_verifies_completion_hashes_and_deterministic_private_free_gzip(tmp_path):
    import gzip
    import json
    import pytest
    from scripts.v3_analysis import export_compact
    from dreamer.provenance import sha256
    raw = tmp_path / 'raw'
    (raw / 'cases').mkdir(parents=True)
    (raw / 'matched').mkdir()
    plan = {'study': 'fixture', 'sample_size': 1, 'regimes': ['switch'],
        'conditions': [{'name': 'memory', 'program_sha256': 'memory-source', 'variant': 'memory'}],
        'matched': {'memory_condition': 'memory', 'source_sha256': 'predictor-source'}}
    def write(path, value):
        path.write_text(json.dumps(value))
    plan_path = tmp_path / 'plan.json'
    write(plan_path, plan)
    write(raw / 'frozen-plan.json', plan)
    pool = {'sha256': 'private-pool-hash', 'episodes': 1}
    write(raw / 'manifest.json', {'plan_sha256': sha256(plan_path), 'episode_pool': pool,
                                'conditions': plan['conditions'], 'regimes': plan['regimes']})
    write(raw / 'pool-manifest.json', {'original_plan_sha256': sha256(plan_path),
        'frozen_plan_sha256': sha256(raw / 'frozen-plan.json'), 'pool': pool})
    case = raw / 'cases/000000.json'
    write(case, {'case': 0, 'conditions': [{'case': 0, 'regime': 'switch', 'condition': 'memory',
        'variant': 'memory', 'program_sha256': 'memory-source', 'reason': 'invalid'}]})
    matched = raw / 'matched/000000--switch.json'
    write(matched, {'case': 0, 'regime': 'switch', 'source_sha256': 'predictor-source'})
    write(raw / 'execution-complete.json', {'cases': 1, 'condition_episodes': 1,
        'case_file_sha256': {case.name: sha256(case)}, 'matched': {'file_sha256': {matched.name: sha256(matched)}}})
    episodes, matches = tmp_path / 'episodes.jsonl.gz', tmp_path / 'matched.jsonl.gz'
    export_compact(raw, plan_path, episodes, matches)
    first = episodes.read_bytes()
    export_compact(raw, plan_path, episodes, matches)
    assert episodes.read_bytes() == first
    with gzip.open(episodes, 'rt') as handle:
        published = json.loads(next(handle))
    assert published['reason'] == 'invalid' and 'seed' not in published and 'trace' not in published
    case.write_text(case.read_text() + ' ')
    with pytest.raises(ValueError, match='checkpoint changed'):
        export_compact(raw, plan_path, episodes, matches)


def test_all_invalid_outcomes_keep_full_denominator_and_no_forecast_is_invented(monkeypatch):
    import copy
    import pytest
    from scripts import v3_analysis as analysis
    monkeypatch.setattr(analysis, 'BOOTSTRAP_REPLICATES', 20)
    plan = {'study': 'invalid-fixture', 'sample_size': 3, 'regimes': ['switch'],
            'conditions': [{'name': 'selected', 'variant': 'predictive'}], 'contrasts': []}
    rows = [{'case': i, 'regime': 'switch', 'condition': 'selected', 'variant': 'predictive',
             'reason': 'invalid', 'error': 'worker failed', 'stats': {}, 'keys': 0, 'door': False,
             'steps': 0, 'task': 0., 'combined_score': 0.} for i in range(3)]
    result = analysis.analyze(rows, plan)['outcomes']['switch/selected']
    assert result['episodes'] == 3 and result['invalid']['count'] == 3
    assert result['escape']['rate'] == 0. and result['combined_score']['mean'] == 0.
    assert result['forecasts']['brier_near'] is None
    extra = copy.deepcopy(rows)
    extra.append(dict(rows[0], condition='undeclared'))
    with pytest.raises(ValueError, match='undeclared outcome'):
        analysis.analyze(extra, plan)


def test_matched_requires_identical_target_counts_and_nonempty_experience(monkeypatch):
    import copy
    import pytest
    from scripts import v3_analysis as analysis
    monkeypatch.setattr(analysis, 'BOOTSTRAP_REPLICATES', 20)
    rows, plan, policy = matched_fixture()
    inconsistent = copy.deepcopy(rows)
    inconsistent[0]['shadows']['fitted_online']['stats']['brier_near'][1] -= 1
    with pytest.raises(ValueError, match='different target counts'):
        analysis.analyze_matched(inconsistent, plan, policy)
    empty = copy.deepcopy(rows)
    for record in empty:
        record['policy_frames'] = 0
        for shadow in record['shadows'].values():
            shadow['stats'], shadow['bins'] = {}, {}
    result = analysis.analyze_matched(empty, plan, policy)['regimes']['switch']
    assert not result['complete_matched_prediction_evidence'] and not result['pairs']
    assert result['empty_policy_episode_cases'] == [0, 1]
    assert result['episodes_with_forecast_targets']['fitted_online'] == 0


def test_analysis_closure_binds_both_outputs_before_example_selection(tmp_path):
    import json
    import pytest
    from scripts.v3_analysis import close_analysis
    from dreamer.provenance import sha256
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps({'sample_size': 2}))
    paths = [tmp_path / name for name in ('episodes.jsonl.gz', 'matched-episodes.jsonl.gz', 'analysis.json', 'matched-analysis.json')]
    for path in paths:
        path.write_text('{}')
    closure_path = tmp_path / 'analysis-closure.json'
    result = close_analysis(plan_path, paths, closure_path, {'raw_completion_sha256': 'frozen-completion'}, 12, 6)
    assert result['status'] == 'analysis-complete' and result['analysis_precedes_example_selection']
    assert result['files']['analysis.json'] == sha256(paths[2])
    assert result['files']['matched-analysis.json'] == sha256(paths[3])
    assert close_analysis(plan_path, paths, closure_path, {'raw_completion_sha256': 'frozen-completion'}, 12, 6) == result
    paths[2].write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Checkpoint drift'):
        close_analysis(plan_path, paths, closure_path, {'raw_completion_sha256': 'frozen-completion'}, 12, 6)
