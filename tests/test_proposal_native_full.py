"""Offline checks for bounded original-task adapters; no worlds or provider calls."""
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from proposal import evaluate_full
from dreamer.native_run1 import CALL_ROLE, CallBudgetExceeded, Run1Runner
from proposal.native_full import FullCallBudget, FullRunner


LIMITS = {'calls': 30, 'remote_elapsed_seconds': 1200,
          'request_timeout_seconds': 250,
          'role_caps': {'mutation_repair': 30, 'novelty': 30, 'summary': 30,
                        'global_insight': 30, 'recommendation': 30,
                        'prompt_mutation': 30, 'readiness': 0}}


def begin_call(budget, role='mutation'):
    token = CALL_ROLE.set(role)
    try:
        return budget.begin({'model': SimpleNamespace(agent='codex', agent_model='gpt-6.1-sol', effort='high'),
                             'command': ['offline-fixture-never-executed']})
    finally:
        CALL_ROLE.reset(token)


@pytest.mark.parametrize('usage', [None, {'usageStatus': 'unavailable'}])
def test_missing_usage_remains_unknown_and_allows_next_authorized_call(tmp_path, usage):
    budget = FullCallBudget(tmp_path, LIMITS)
    stdout = 'Synthetic reply without usage' if usage is None else json.dumps({'usage': usage})
    result = SimpleNamespace(returncode=0, stdout=stdout, stderr='')
    budget.finish(begin_call(budget), result=result)
    assert budget.calls[0]['usage_missing'] is True
    assert budget.reason('novelty') is None
    restored = FullCallBudget(tmp_path, LIMITS)
    assert restored.reason('novelty') is None
    assert restored.totals()['usage_unavailable_calls'] == 1
    restored.finish(begin_call(restored, 'novelty'), result=result)
    assert restored.totals()['calls'] == 2
    assert restored.totals()['roles'] == {'mutation': 1, 'novelty': 1}


def test_two_provider_timeouts_count_and_block_further_dispatch(tmp_path):
    budget = FullCallBudget(tmp_path, LIMITS)
    budget.finish(begin_call(budget), error=TimeoutError('Synthetic transport timeout'))
    assert budget.reason('repair') is None
    budget.finish(begin_call(budget, 'repair'), error=TimeoutError('Second synthetic timeout'))
    assert budget.totals()['calls'] == 2
    assert budget.totals()['usage_unavailable_calls'] == 2
    with pytest.raises(CallBudgetExceeded, match='Two consecutive'):
        begin_call(budget)
    assert FullCallBudget(tmp_path, LIMITS).reason('mutation') == 'Two consecutive provider transport failures'


@pytest.mark.parametrize('ceiling', ['deadline', 'provider_time', 'readiness'])
def test_call_admission_reserves_full_request_and_disallows_probes(tmp_path, ceiling):
    deadline = None
    limits = {**LIMITS}
    if ceiling == 'deadline':
        deadline = (datetime.now(timezone.utc) + timedelta(seconds=200)).isoformat()
    if ceiling == 'provider_time':
        limits['remote_elapsed_seconds'] = 249
    budget = FullCallBudget(tmp_path, limits, deadline_utc=deadline)
    with pytest.raises(CallBudgetExceeded):
        begin_call(budget, 'readiness' if ceiling == 'readiness' else 'mutation')
    assert budget.totals()['calls'] == 0


def test_unresolved_prior_request_still_requires_reconciliation(tmp_path):
    budget = FullCallBudget(tmp_path, LIMITS)
    begin_call(budget)
    restored = FullCallBudget(tmp_path, LIMITS)
    with pytest.raises(CallBudgetExceeded, match='Unresolved'):
        begin_call(restored)


@pytest.mark.parametrize('fault', [None, 'missing_resource', 'native_fallback', 'episode_error'])
def test_seed_requires_real_valid_artifacts_before_mutation(tmp_path, monkeypatch, fault):
    async def native_setup(self, code):
        return

    monkeypatch.setattr(Run1Runner, '_setup_initial_program', native_setup)
    output = tmp_path / 'gen_0' / 'results'
    output.mkdir(parents=True)
    (output / 'correct.json').write_text(json.dumps({'correct': True}))
    (output / 'metrics.json').write_text(json.dumps({'combined_score': .4, 'text_feedback': 'Fixture only.'}))
    rows = [{'case': case, 'error': None} for case in range(5)]
    if fault == 'episode_error':
        rows[0]['error'] = 'Synthetic failure hidden by native correct flag'
    (output / 'episodes.json').write_text(json.dumps(rows))
    if fault != 'missing_resource':
        (output / 'resource.json').write_text(json.dumps({'completed': True}))
    runner = object.__new__(FullRunner)
    runner.results_dir = tmp_path
    program = SimpleNamespace(metadata={'evaluation_failed': fault == 'native_fallback'})
    runner.db = SimpleNamespace(get_programs_by_generation=lambda generation: [program])
    if fault is None:
        asyncio.run(runner._setup_initial_program('Offline seed fixture'))
    else:
        with pytest.raises((RuntimeError, FileNotFoundError)):
            asyncio.run(runner._setup_initial_program('Offline seed fixture'))


def frozen_manifest(tmp_path, monkeypatch):
    root = tmp_path / 'repository'
    sources = {}
    for name in evaluate_full.REQUIRED_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'Frozen offline fixture: {name}\n')
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / 'campaign-manifest.json'
    manifest.write_text(json.dumps({'source_sha256': sources}))
    monkeypatch.setattr(evaluate_full, 'ROOT', root)
    return manifest, root


@pytest.mark.parametrize('failure', ['missing_source', 'changed_source'])
def test_frozen_source_checks_precede_original_evaluator(tmp_path, monkeypatch, failure):
    from proposal import evaluate
    manifest, root = frozen_manifest(tmp_path, monkeypatch)
    if failure == 'missing_source':
        data = json.loads(manifest.read_text())
        del data['source_sha256']['dreamer/isolation.py']
        manifest.write_text(json.dumps(data))
    else:
        (root / 'dreamer/world.py').write_text('Changed fixture')
    out = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['evaluate_full.py', '--campaign-manifest', str(manifest),
                                    '--results_dir', str(out)])
    monkeypatch.setattr(evaluate, 'main', lambda: pytest.fail('Evaluator called before source validation'))
    with pytest.raises(ValueError, match='frozen evaluator|SHA256 mismatch'):
        evaluate_full.main()
    saved = json.loads((out / 'resource.json').read_text())
    assert saved['completed'] is False
    assert saved['candidate_cpu_seconds'] >= 0
    assert not (out / 'metrics.json').exists()


@pytest.mark.parametrize('episode_error', [None, 'TimeoutError: Candidate exceeded 3 s response deadline'])
@pytest.mark.parametrize('record_first', ['yes', 'no'])
def test_wrapper_preserves_metrics_and_forwards_actual_failures(tmp_path, monkeypatch, episode_error, record_first):
    from proposal import evaluate
    manifest, _ = frozen_manifest(tmp_path, monkeypatch)
    out = tmp_path / 'output'
    original = ['evaluate_full.py', '--campaign-manifest', str(manifest),
                '--record-first', record_first,
                '--results_dir', str(out), '--episodes', '5', '--seed-start', '0']
    monkeypatch.setattr(sys, 'argv', original)
    metrics = {'combined_score': .4, 'public': {'invalid': int(episode_error is not None)},
               'text_feedback': 'Original fixture feedback.'}

    def fake_evaluate():
        assert sys.argv == [original[0], *original[5:], *(['--replay-first'] if record_first == 'yes' else [])]
        out.mkdir()
        (out / 'metrics.json').write_text(json.dumps(metrics))
        (out / 'correct.json').write_text(json.dumps({'correct': episode_error is None,
                                                    'error': [episode_error] if episode_error else None}))

    monkeypatch.setattr(evaluate, 'main', fake_evaluate)
    evaluate_full.main()
    assert sys.argv is original
    saved_metrics = json.loads((out / 'metrics.json').read_text())
    assert saved_metrics['combined_score'] == metrics['combined_score']
    assert saved_metrics['public'] == metrics['public']
    if episode_error:
        assert episode_error in saved_metrics['text_feedback']
    else:
        assert saved_metrics == metrics
    saved = json.loads((out / 'resource.json').read_text())
    assert saved['completed'] is True
    assert saved['error'] is None
    assert saved['elapsed_seconds'] >= 0
    assert saved['total_cpu_seconds'] == pytest.approx(
        saved['candidate_cpu_seconds'] + saved['evaluator_cpu_seconds'])


def test_wrapper_records_resource_and_restores_argv_when_evaluator_raises(tmp_path, monkeypatch):
    from proposal import evaluate
    manifest, _ = frozen_manifest(tmp_path, monkeypatch)
    out = tmp_path / 'output'
    original = ['evaluate_full.py', '--campaign-manifest', str(manifest), '--results_dir', str(out)]
    monkeypatch.setattr(sys, 'argv', original)

    def broken_evaluate():
        raise RuntimeError('Synthetic evaluator failure')

    monkeypatch.setattr(evaluate, 'main', broken_evaluate)
    with pytest.raises(RuntimeError, match='Synthetic evaluator failure'):
        evaluate_full.main()
    assert sys.argv is original
    saved = json.loads((out / 'resource.json').read_text())
    assert not saved['completed']
    assert saved['error'] == 'RuntimeError: Synthetic evaluator failure'
