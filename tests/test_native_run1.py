"""Deterministic adapter checks: no model requests or scored maze episodes."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dreamer.native_run1 import (CALL_ROLE, DEFAULT_LIMITS, CallBudgetExceeded,
                                Run1CallBudget, Run1Runner)


def call_kwargs(tmp_path, model='gpt-6.1-sol'):
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('Synthetic transport fixture.')
    return {'model': SimpleNamespace(agent='codex', agent_model=model, effort='high'),
            'command': ['subscription-wrapper', 'codex', '--prompt-file', str(prompt)]}


def response(tokens=12, route='subscription', model='gpt-6.1-sol'):
    usage = {'model': model, 'inputTokens': tokens, 'outputTokens': 3,
             'cacheReadTokens': 100, 'reasoningOutputTokens': 2,
             'totalTokens': tokens + 103, 'usageStatus': 'reported',
             'billing': {'attempts': [{'route': route}]}}
    return SimpleNamespace(returncode=0, stdout='OK\n' + json.dumps({'usage': usage}), stderr='')


def begin(budget, kwargs, role='mutation'):
    token = CALL_ROLE.set(role)
    try:
        return budget.begin(kwargs)
    finally:
        CALL_ROLE.reset(token)


def test_budget_counts_uncached_and_output_once(tmp_path):
    budget = Run1CallBudget(tmp_path)
    budget.finish(begin(budget, call_kwargs(tmp_path)), result=response())
    assert budget.totals()['uncached_input_plus_output_tokens'] == 15
    assert budget.totals()['cached_input_tokens'] == 100
    assert budget.totals()['calls'] == 1
    assert budget.totals()['role_groups'] == {'mutation_repair': 1}
    assert len(list((tmp_path / 'model-calls').glob('*.prompt.md'))) == 1


def test_token_boundary_overshoot_blocks_next_call(tmp_path):
    limits = deepcopy(DEFAULT_LIMITS)
    limits['uncached_input_plus_output_tokens'] = 10
    budget = Run1CallBudget(tmp_path, limits)
    budget.finish(begin(budget, call_kwargs(tmp_path)), result=response())
    assert budget.totals()['uncached_input_plus_output_tokens'] == 15
    with pytest.raises(CallBudgetExceeded):
        begin(budget, call_kwargs(tmp_path))
    assert budget.totals()['calls'] == 1


def test_total_and_role_caps_include_readiness(tmp_path):
    limits = deepcopy(DEFAULT_LIMITS)
    limits['calls'] = 2
    limits['role_caps']['readiness'] = 1
    budget = Run1CallBudget(tmp_path, limits)
    budget.finish(begin(budget, call_kwargs(tmp_path), 'readiness'), result=response())
    assert budget.reason('readiness') == 'readiness call cap reached'
    assert budget.reason('mutation') is None
    budget.finish(begin(budget, call_kwargs(tmp_path), 'repair'), result=response())
    assert 'calls budget reached' in budget.reason()


@pytest.mark.parametrize('bad', ['missing', 'billing', 'model'])
def test_unknown_usage_and_route_fail_closed_across_resume(tmp_path, bad):
    budget = Run1CallBudget(tmp_path)
    result = response(route='paid' if bad == 'billing' else 'subscription',
                      model='other' if bad == 'model' else 'gpt-6.1-sol')
    if bad == 'missing':
        result.stdout = 'failed before usage'
    budget.finish(begin(budget, call_kwargs(tmp_path)), result=result)
    restored = Run1CallBudget(tmp_path)
    with pytest.raises(CallBudgetExceeded):
        begin(restored, call_kwargs(tmp_path))
    assert restored.totals()['calls'] == 1


def test_unresolved_started_request_blocks_resume(tmp_path):
    budget = Run1CallBudget(tmp_path)
    begin(budget, call_kwargs(tmp_path))
    restored = Run1CallBudget(tmp_path)
    assert 'Unresolved' in restored.reason()


def test_limit_changes_rejected(tmp_path):
    Run1CallBudget(tmp_path)
    limits = deepcopy(DEFAULT_LIMITS)
    limits['calls'] += 1
    with pytest.raises(ValueError, match='limits changed'):
        Run1CallBudget(tmp_path, limits)


def test_prompt_seed_archive_is_not_duplicated_on_resume(tmp_path):
    from shinka.core import EvolutionConfig
    runner = object.__new__(Run1Runner)
    runner.results_dir = tmp_path
    runner.evo_config = EvolutionConfig(task_sys_msg='Synthetic prompt without model execution.')
    runner.prompt_llm = None
    runner.verbose = False
    asyncio.run(runner._setup_prompt_evolution())
    first = runner.prompt_db.get_archive()[0].id
    assert runner.prompt_db._count_prompts_in_db() == 1
    asyncio.run(runner._setup_prompt_evolution())
    assert runner.prompt_db._count_prompts_in_db() == 1
    assert runner.prompt_db.get_archive()[0].id == first


def test_native_bandit_rng_round_trip_and_zero_cost(tmp_path):
    from shinka.llm import AsymmetricUCB
    runner = object.__new__(Run1Runner)
    runner.results_dir = tmp_path
    runner._authorized_total_slots = 32
    runner._completed_side_effects = {'program-fixture'}
    runner._saturation_reviews = set()
    runner.prompt_evolution_counter = 3
    runner.prompt_percentile_recompute_counter = 7
    runner.prompt_api_cost = 0
    runner.meta_summarizer = None
    runner.llm_selection = AsymmetricUCB(arm_names=['a', 'b'], seed=23, cost_aware_coef=0)
    for arm, reward in [('a', .7), ('b', .8)]:
        runner.llm_selection.update_submitted(arm)
        runner.llm_selection.update_cost(arm, 0)
        runner.llm_selection.update(arm, reward, baseline=.5)
    runner._save_state('fixture')
    state = json.loads((tmp_path / 'run1-native-state.json').read_text())
    assert state['prompt_evolution_counter'] == 3
    expected = [runner.llm_selection.select_llm()[0].tolist() for _ in range(12)]
    runner.llm_selection = AsymmetricUCB(arm_names=['a', 'b'], seed=999, cost_aware_coef=0)
    runner._saved_state = state
    runner._load_bandit_state()
    assert [runner.llm_selection.select_llm()[0].tolist() for _ in range(12)] == expected
    assert np.isfinite(runner.llm_selection.posterior()).all()


@pytest.mark.parametrize('fixture_mode', ['accepted', 'rejected', 'budget_held'])
def test_native_fixture_executes_all_auxiliary_roles_without_remote_calls(tmp_path, monkeypatch, fixture_mode):
    """Full native plumbing on a synthetic constant; not a scientific search."""
    import sys
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    from shinka.core import async_runner
    from shinka.llm.providers import headless
    from dreamer.native_run1 import install_run1_call_audit
    from dreamer.headless_transport import install

    monkeypatch.setenv('SHINKA_PRICING_MODE', 'offline')
    monkeypatch.setenv('SHINKA_HEADLESS_COMMAND', '/synthetic/no-subprocess-is-executed')
    monkeypatch.setattr(async_runner, '_validate_evo_config_model_env_access', lambda config: None)

    class FixtureEmbedding:
        def __init__(self, model_name):
            pass
        async def embed_async(self, text):
            return [1., 0., 0., 0.], 0.
    monkeypatch.setattr(async_runner, 'AsyncEmbeddingClient', FixtureEmbedding)
    monkeypatch.setattr(headless, '_run1_budget_installed', False, raising=False)
    async def fake_command(**kwargs):
        role = CALL_ROLE.get()
        if role == 'mutation':
            content = '<NAME>synthetic</NAME><DESCRIPTION>Fixture only.</DESCRIPTION><CODE>\n```python\n# EVOLVE-BLOCK-START\nVALUE = 2\n# EVOLVE-BLOCK-END\n```\n</CODE>'
        elif role == 'novelty':
            content = ('NOT NOVEL' if fixture_mode == 'rejected' else 'NOVEL') + ': synthetic pipeline fixture only.'
        elif role == 'prompt_mutation':
            content = '<NAME>fixture_prompt</NAME><DESCRIPTION>Fixture.</DESCRIPTION><PROMPT>A synthetic system instruction long enough for native prompt parsing. Improve the declared constant for this test only.</PROMPT>'
        else:
            content = '• Preserve the synthetic fixture and use its deterministic feedback.'
        result = response(model=kwargs['model'].agent_model)
        result.stdout = content + '\n' + result.stdout.split('\n')[-1]
        return result
    monkeypatch.setattr(headless, '_run_headless_command_async', fake_command)
    monkeypatch.setattr(headless, '_run_headless_command_sync', lambda **k: pytest.fail('unexpected sync model request'))
    # Restore mutable provider functions when this test ends.
    monkeypatch.setattr(headless, '_parse_stdout', headless._parse_stdout)
    install()
    seed = tmp_path / 'seed.py'
    seed.write_text('# EVOLVE-BLOCK-START\nVALUE = 1\n# EVOLVE-BLOCK-END\n')
    evaluator = tmp_path / 'evaluate.py'
    evaluator.write_text('''import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--program_path');p.add_argument('--results_dir');a=p.parse_args()
r=pathlib.Path(a.results_dir);r.mkdir(parents=True,exist_ok=True)
score=.2 if 'VALUE = 2' in pathlib.Path(a.program_path).read_text() else .1
(r/'metrics.json').write_text(json.dumps({'combined_score':score,'public':{'fixture':True},'private':{},'text_feedback':'Synthetic fixture; zero maze episodes.'}))
(r/'correct.json').write_text(json.dumps({'correct':True,'error':None}))
''')
    results = tmp_path / 'native'
    limits = deepcopy(DEFAULT_LIMITS)
    if fixture_mode == 'budget_held':
        limits['role_caps']['novelty'] = 0
    budget = Run1CallBudget(results, limits)
    install_run1_call_audit(budget)
    model = 'headless/codex@gpt-6.1-sol?effort=high'
    evo = EvolutionConfig(results_dir=str(results), init_program_path=str(seed), num_generations=2,
        task_sys_msg='Synthetic adapter plumbing fixture, not a research experiment.',
        llm_models=[model], llm_dynamic_selection='ucb', llm_dynamic_selection_kwargs={'cost_aware_coef':0.,'seed':5},
        llm_kwargs={}, patch_types=['full'], patch_type_probs=[1.], max_patch_attempts=1,
        max_patch_resamples=1, max_novelty_attempts=1,
        embedding_model='local/fixture@http://127.0.0.1:9/v1', code_embed_sim_threshold=.9,
        novelty_llm_models=[model], meta_llm_models=[model], meta_rec_interval=1,
        evolve_prompts=True, prompt_llm_models=[model], prompt_evolution_interval=1, use_text_feedback=True)
    job = LocalJobConfig(eval_program_path=str(evaluator), python_executable=sys.executable, time='00:00:15')
    runner = Run1Runner(evo_config=evo, job_config=job,
        db_config=DatabaseConfig(num_islands=1), call_budget=budget,
        max_proposal_jobs=1, max_evaluation_jobs=1, max_db_workers=1, verbose=False)
    async def run_with_heartbeat():
        async def heartbeat():
            while True:
                await asyncio.sleep(.05)
        beat = asyncio.create_task(heartbeat())
        try:
            await asyncio.wait_for(runner.run_async(), timeout=60)
        finally:
            # Keep the heartbeat alive through thread-executor shutdown; the
            # sandbox can otherwise defer its completion notification.
            await asyncio.get_running_loop().shutdown_default_executor()
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)
    asyncio.run(run_with_heartbeat())
    roles = set(budget.totals()['roles'])
    if fixture_mode != 'budget_held':
        assert {'mutation','novelty','prompt_mutation','summary','global_insight','recommendation'} <= roles
    saved = json.loads((results / 'run1-native-state.json').read_text())
    assert saved['boundary'] == 'controller_finalization'
    if fixture_mode == 'accepted':
        accepted = json.loads((results / 'gen_1/accepted-proposal.json').read_text())
        assert accepted['stage'] == 'accepted_after_native_novelty'
        assert accepted['novelty_evidence']['accepted']
    else:
        assert not (results / 'gen_1/accepted-proposal.json').exists()
    import sqlite3
    with sqlite3.connect(results / 'programs.sqlite') as db:
        rows = db.execute('select generation,correct from programs order by generation').fetchall()
    assert rows == ([(0, 1)] if fixture_mode == 'budget_held' else [(0, 1), (1, int(fixture_mode == 'accepted'))])
    if fixture_mode == 'budget_held':
        held = json.loads((results / 'gen_1/held-proposal.json').read_text())
        assert held['stage'] == 'generated_before_novelty'
        assert not held['counts_as_completed_slot']
        assert 'novelty' not in roles
    assert (results / 'gen_1/proposal-sources/novelty-1-resample-1.py').exists()
    assert all(c['usage']['billing']['attempts'][0]['route'] == 'subscription' for c in budget.calls)
    if fixture_mode == 'accepted':
        # The same native database resumes without any repeated calls, source
        # evaluation, prompt seed insertion or bandit random-stream reset.
        count_before = len(budget.calls)
        bandit_rng_before = saved['bandit_rng']
        with sqlite3.connect(results / 'prompts.sqlite') as db:
            prompts_before = db.execute('select count(*) from system_prompts').fetchone()[0]
        runner = Run1Runner(evo_config=evo, job_config=job,
            db_config=DatabaseConfig(num_islands=1), call_budget=budget,
            max_proposal_jobs=1, max_evaluation_jobs=1, max_db_workers=1, verbose=False)
        asyncio.run(run_with_heartbeat())
        restored = json.loads((results / 'run1-native-state.json').read_text())
        assert restored['bandit_rng'] == bandit_rng_before
        assert restored['prompt_evolution_counter'] == saved['prompt_evolution_counter']
        assert len(budget.calls) == count_before
        with sqlite3.connect(results / 'prompts.sqlite') as db:
            assert db.execute('select count(*) from system_prompts').fetchone()[0] == prompts_before


def test_saturation_requires_three_perfect_correct_and_tightly_scored_rows():
    from dreamer.native_run1 import saturation_condition
    records = [{'generation':g, 'correct':True, 'escape':1., 'score':.98} for g in (5,6,7)]
    assert saturation_condition(records)
    assert not saturation_condition(records[:2])
    changed = deepcopy(records)
    changed[0]['correct'] = False
    assert not saturation_condition(changed)
    changed[0]['correct'] = True
    changed[0]['escape'] = .999
    assert not saturation_condition(changed)
    changed[0]['escape'] = 1.
    changed[0]['score'] = .977
    assert not saturation_condition(changed)
