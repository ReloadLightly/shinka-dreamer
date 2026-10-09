"""Offline admission/configuration checks; no providers, embeddings or worlds."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from dreamer.native_run1 import CALL_ROLE
from proposal import evolve_study
from proposal.study_native import IndependentRewriteRunner, StudyCallBudget, StudyRunner
from proposal.native_full import FullRunner


def config(treatment='full'):
    return evolve_study.build_configuration(SimpleNamespace(
        results=str(evolve_study.ROOT/'results/offline-study-fixture'),
        run_id='offline-fixture', native_seed=129, slots=6, calls=10,
        provider_seconds=1000., minutes=12., treatment=treatment))


def write_audit(budget, state, violation=None):
    directory = budget.root/'codex-event-audit'
    directory.mkdir(exist_ok=True)
    raw = directory/(state[0]['id']+'.events.jsonl')
    raw.write_text('{"type":"turn.completed"}\n')
    (directory/(state[0]['id']+'.summary.json')).write_text(json.dumps({
        'unauthorized_tool_event': violation, 'raw_events_file': raw.name,
        'raw_events_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}))


def test_configuration_preserves_native_mechanisms_and_original_panel():
    value = config()
    assert value['evolution']['llm_models'] == evolve_study.MODELS
    assert value['evolution']['evolve_prompts']
    assert value['evolution']['novelty_llm_models']
    assert value['evolution']['embedding_model'].startswith('local/')
    assert value['database']['num_islands'] == 4
    assert value['database']['num_archive_inspirations'] == 1
    assert value['job']['extra_cmd_args']['episodes'] == 5
    assert value['search_cases'] == list(range(5))
    assert value['headless_capacity_retries'] == 0
    assert value['limits']['role_caps']['readiness'] == 0
    assert value['limits']['request_timeout_seconds'] == 500
    assert value['child_timeout_seconds'] == 480


def test_prepared_configuration_cannot_change_or_consume_existing_directory(tmp_path):
    value = config()
    value['evolution']['results_dir'] = str(tmp_path/'fresh')
    value['mutation_context']['work_dir'] = str(tmp_path/'model-context')
    evolve_study.freeze_configuration(value)
    evolve_study.freeze_configuration(value)
    value['limits']['calls'] += 1
    with pytest.raises(ValueError, match='Frozen study configuration changed'):
        evolve_study.freeze_configuration(value)
    value['evolution']['results_dir'] = str(tmp_path/'historical')
    historical = tmp_path/'historical'
    historical.mkdir()
    (historical/'existing').write_text('preserve')
    with pytest.raises(ValueError, match='existing nonempty'):
        evolve_study.freeze_configuration(value)


def test_prior_execution_grant_blocks_relaunch_before_subprocess(tmp_path, monkeypatch):
    value = config()
    value['evolution']['results_dir'] = str(tmp_path)
    (tmp_path/'execution-grant.json').write_text('{}')
    monkeypatch.setattr(evolve_study, 'runtime_lock', lambda: nullcontext())
    def forbidden(*args, **kwargs):
        raise AssertionError('Subprocess execution is forbidden in this test')
    monkeypatch.setattr(evolve_study.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='one-shot execution grant'):
        evolve_study.supervise(value)


@pytest.mark.parametrize('role', ['mutation', 'repair', 'novelty', 'summary',
                                'global_insight', 'recommendation', 'prompt_mutation'])
def test_every_role_requires_read_only_and_counts_one_real_dispatch(tmp_path, role):
    budget = StudyCallBudget(tmp_path/role, limits=config()['limits'])
    token = CALL_ROLE.set(role)
    try:
        state = budget.begin({'command': ['offline-fixture', '--allow', 'read-only'],
            'model': SimpleNamespace(agent='codex', agent_model='gpt-6.1-sol', effort='high')})
        write_audit(budget, state)
        budget.finish(state, result=SimpleNamespace(returncode=0, stdout='', stderr=''))
    finally:
        CALL_ROLE.reset(token)
    assert budget.totals()['roles'] == {role: 1}
    assert budget.calls[0]['usage_missing']


@pytest.mark.parametrize('command', [['fixture'], ['fixture', '--allow', 'yolo'],
    ['fixture', '--allow'], ['fixture', '--allow', 'read-only', '--allow', 'yolo']])
def test_unsafe_or_ambiguous_commands_fail_before_call_admission(tmp_path, command):
    budget = StudyCallBudget(tmp_path, limits=config()['limits'])
    with pytest.raises(RuntimeError, match='read-only sandbox'):
        budget.begin({'command': command})
    assert budget.totals()['calls'] == 0


def test_rewrite_is_native_seed_only_without_auxiliary_mechanisms():
    value = config('rewrite')
    assert value['database']['parent_selection_strategy'] == 'best_of_n'
    assert value['database']['num_islands'] == 1
    assert value['database']['num_archive_inspirations'] == 0
    assert value['database']['num_top_k_inspirations'] == 0
    assert value['evolution']['patch_types'] == ['full']
    assert value['evolution']['max_patch_attempts'] == 1
    for key in ('llm_dynamic_selection', 'meta_rec_interval', 'embedding_model', 'novelty_llm_models'):
        assert value['evolution'][key] is None
    assert not value['evolution']['evolve_prompts']
    assert value['allowed_roles'] == ['mutation']
    assert all(cap == 0 for role, cap in value['limits']['role_caps'].items() if role != 'mutation_repair')
    assert value['limits']['request_timeout_seconds'] == 500
    assert value['child_timeout_seconds'] == 480


def test_rewrite_rejects_repair_before_actual_dispatch(tmp_path):
    value = config('rewrite')
    budget = StudyCallBudget(tmp_path, limits=value['limits'], allowed_roles=value['allowed_roles'])
    token = CALL_ROLE.set('repair')
    try:
        with pytest.raises(RuntimeError, match='Role forbidden'):
            budget.begin({'command': ['fixture', '--allow', 'read-only']})
    finally:
        CALL_ROLE.reset(token)
    assert budget.totals()['calls'] == 0


def test_rewrite_enforces_original_parent_and_alternates_models(monkeypatch):
    runner = object.__new__(IndependentRewriteRunner)
    runner.rewrite_seed_sha256 = hashlib.sha256(b'original seed\n').hexdigest()
    runner.model_offset = 1
    runner.evo_config = SimpleNamespace(llm_models=['model-a', 'model-b'])
    parent = SimpleNamespace(generation=0, code='original seed\n')
    seen = []
    async def mock_patch(self, parent, archive, top, generation, meta_recs, **kwargs):
        seen.append(kwargs['model_sample_probs'])
    monkeypatch.setattr(StudyRunner, '_run_patch_async', mock_patch)
    asyncio.run(runner._run_patch_async(parent, [], [], 1))
    asyncio.run(runner._run_patch_async(parent, [], [], 2))
    assert seen == [[0., 1.], [1., 0.]]
    for altered, archive, meta in [(SimpleNamespace(generation=1, code=parent.code), [], None),
            (SimpleNamespace(generation=0, code='different'), [], None),
            (parent, [parent], None), (parent, [], 'prior discovery')]:
        with pytest.raises(RuntimeError, match='accumulated search context'):
            asyncio.run(runner._run_patch_async(altered, archive, [], 3, meta))
    with pytest.raises(RuntimeError, match='does not repair'):
        asyncio.run(runner._run_fix_patch_async())
    assert asyncio.run(runner._get_code_embedding_async('not-read')) == ([], 0.)


def test_acceptance_records_actual_all_role_prefix_before_evaluation(tmp_path, monkeypatch):
    directory = tmp_path/'gen_3'
    directory.mkdir()
    runner = object.__new__(StudyRunner)
    runner.call_budget = SimpleNamespace(totals=lambda: {'calls': 7, 'remote_elapsed_seconds': 419.})
    seen = []
    runner.event = lambda name, **data: seen.append((name, data))
    async def mock_submit(self, exec_fname, results_dir, sampling_worker_id):
        evidence = json.loads((directory/'acceptance-budget.json').read_text())
        assert evidence['all_role_calls_at_acceptance'] == 7
        return 'synthetic-job'
    monkeypatch.setattr(FullRunner, '_submit_evaluation_job_with_slot', mock_submit)
    value = asyncio.run(runner._submit_evaluation_job_with_slot(str(directory/'main.py'), str(directory/'results')))
    assert value == 'synthetic-job'
    assert seen[0][0] == 'proposal_acceptance_budget'


@pytest.mark.parametrize('tool_event', [False, True])
def test_codex_shim_preserves_boundaries_and_rejects_tool_events(tmp_path, tool_event):
    fake = tmp_path/'fake-codex'
    event = {'type': 'item.started', 'item': {'type': 'command_execution'}} if tool_event else {
        'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Synthetic fixture'}}
    fake.write_text('#!/usr/bin/env python3\nimport json,sys\n'
        +f'open({str(tmp_path/"args.json")!r},"w").write(json.dumps(sys.argv[1:]))\n'
        +f'print({json.dumps(event)!r},flush=True)\n')
    fake.chmod(0o700)
    env = {**os.environ, 'SHINKA_STUDY_CODEX_BIN': str(fake),
           'SHINKA_STUDY_TOOL_AUDIT_DIR': str(tmp_path/'audit')}
    result = subprocess.run([sys.executable, str(evolve_study.ROOT/'scripts/study_codex_bin/codex'),
        '--sandbox', 'read-only', '--ask-for-approval', 'never', '--search', 'exec', '--json', '-'],
        input='', text=True, capture_output=True, env=env, timeout=10)
    assert result.returncode == (78 if tool_event else 0), result.stderr
    args = json.loads((tmp_path/'args.json').read_text())
    assert '--search' not in args
    assert args[args.index('--sandbox')+1] == 'read-only'
    assert args[args.index('--ask-for-approval')+1] == 'never'
    assert 'project_doc_max_bytes=0' in args
    assert 'features.shell_tool=false' in args
    summary = json.loads(next((tmp_path/'audit').glob('*.summary.json')).read_text())
    assert bool(summary['unauthorized_tool_event']) == tool_event


def test_global_mcp_configuration_is_blocked_without_logging_values(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_HOME', str(tmp_path))
    (tmp_path/'config.toml').write_text('[mcp_servers.fixture]\ncommand="PRIVATE_VALUE"\n')
    with pytest.raises(ValueError, match='no global MCP') as error:
        evolve_study.check_global_config()
    assert 'PRIVATE_VALUE' not in str(error.value)


@pytest.mark.parametrize('problem', ['missing', 'tool_event', 'changed_raw'])
def test_context_audit_failure_stops_run_and_quarantines_response(tmp_path, problem):
    budget = StudyCallBudget(tmp_path, limits=config()['limits'])
    token = CALL_ROLE.set('mutation')
    try:
        state = budget.begin({'command': ['fixture', '--allow', 'read-only'],
            'model': SimpleNamespace(agent='codex', agent_model='gpt-6.1-sol', effort='high')})
        if problem != 'missing':
            write_audit(budget, state, {'item_type': 'command_execution'} if problem == 'tool_event' else None)
        if problem == 'changed_raw':
            (budget.root/'codex-event-audit'/(state[0]['id']+'.events.jsonl')).write_text('altered')
        with pytest.raises(RuntimeError, match='Quarantined model response'):
            budget.finish(state, result=SimpleNamespace(returncode=0, stdout='untrusted reply', stderr=''))
        assert budget.reason('mutation').startswith('Codex context audit failed:')
        assert budget.totals()['calls'] == 1
        assert state[0]['mutation_context_validated'] is False
        with pytest.raises(RuntimeError):
            budget.begin({'command': ['fixture', '--allow', 'read-only'],
                'model': SimpleNamespace(agent='codex', agent_model='gpt-6.1-sol', effort='high')})
        assert budget.totals()['calls'] == 1
    finally:
        CALL_ROLE.reset(token)
