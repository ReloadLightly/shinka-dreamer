"""RUN1-local audit, call budgets and checkpoint fidelity for pinned Shinka.

Selection, mutation, crossover, novelty comparisons, prompt fitness and migration
remain native. This adapter records actual calls and makes transport/budget
failures and pre-evaluation failed slots explicit. It never edits upstream.
"""
import asyncio
from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import uuid

import numpy as np

from dreamer.native import CheckpointRunner
from dreamer.native_v3 import atomic_json
from dreamer.provenance import sha256


CALL_ROLE = ContextVar('run1_call_role', default='unknown')
DEFAULT_LIMITS = {
    'uncached_input_plus_output_tokens': 1_500_000,
    'calls': 120,
    'remote_elapsed_seconds': 7200,
    'request_timeout_seconds': 600,
    'role_caps': {'mutation_repair': 64, 'novelty': 16, 'summary': 32,
                  'global_insight': 8, 'recommendation': 8,
                  'prompt_mutation': 16, 'readiness': 4},
}
IMMUTABLE_BOUNDARY = (
    'NON-EVOLVING EXPERIMENT BOUNDARY: Preserve observation-only '
    'world_model_step/planner/export_model interface and whole-program '
    'search. Hidden laws, identities, seeds, future innovations and private '
    'case pools are inaccessible; do not attempt to inspect them. Evaluator, '
    'score, 15x15 maze, candidate isolation and resource limits are fixed '
    'outside candidate scope. All tools/network/model calls by candidate code '
    'are forbidden. Improve absolute task performance; never weaken a '
    'disabled/frozen counterpart to manufacture an adaptation effect. '
    'Prediction diagnostics are feedback, not selection fitness.'
)


def now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


def usage_from_stdout(stdout):
    for line in reversed((stdout or '').splitlines()):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get('usage'), dict):
            return obj['usage']
    return None


class CallBudgetExceeded(RuntimeError):
    pass


class EmbeddingUnavailable(RuntimeError):
    pass


class ProposalHeld(BaseException):
    """Escape native retry handlers without discarding a budget-held proposal."""


class Run1CallBudget:
    """One ledger across native mutation, auxiliary clients and readiness.

    Admission is serialized by native Headless's process-wide CLI lock. Tokens
    and elapsed time are observed after each completed call; any final-call
    overshoot is retained. Missing usage stops further admissions.
    """
    def __init__(self, results, limits=None, deadline_utc=None):
        self.root = Path(results).resolve()
        self.directory = self.root / 'model-calls'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limits = json.loads(json.dumps(limits or DEFAULT_LIMITS))
        self.deadline = (datetime.fromisoformat(deadline_utc.replace('Z', '+00:00'))
                         if deadline_utc else None)
        self.runner = None
        self.blocked_reason = None
        ledger = self.root / 'call-budget-ledger.json'
        if ledger.exists():
            previous = json.loads(ledger.read_text())
            if previous['limits'] != self.limits:
                raise ValueError('RUN1 call limits changed; an explicit separately recorded amendment is required')
        self._load()

    def _load(self):
        self.calls = [json.loads(p.read_text()) for p in sorted(self.directory.glob('*.json'))]
        unresolved = [x['id'] for x in self.calls if x.get('status') == 'started']
        if unresolved:
            self.blocked_reason = 'Unresolved prior model requests: ' + ','.join(unresolved)
        elif any(c.get('billing_violation') or c.get('model_mismatch') for c in self.calls):
            self.blocked_reason = 'Prior billing route or effective model mismatch requires review'
        elif any(not c.get('usage') or c['usage'].get('usageStatus') != 'reported' for c in self.calls):
            self.blocked_reason = 'Prior reported usage unavailable; token budget cannot be verified'
        self.write_ledger()

    def totals(self):
        usages = [c['usage'] for c in self.calls if isinstance(c.get('usage'), dict)]
        result = {'calls': len(self.calls),
                  'roles': dict(Counter(c['role'] for c in self.calls)),
                  'remote_elapsed_seconds': sum(c.get('elapsed_seconds', 0) for c in self.calls),
                  'uncached_input_plus_output_tokens': sum(
                      u.get('inputTokens', 0) + u.get('outputTokens', 0) for u in usages),
                  'cached_input_tokens': sum(u.get('cacheReadTokens', 0) for u in usages),
                  'cache_write_tokens': sum(u.get('cacheWriteTokens', 0) for u in usages),
                  'reported_total_tokens': sum(u.get('totalTokens', 0) for u in usages),
                  'usage_unavailable_calls': sum(c.get('status') != 'started' and
                      (not c.get('usage') or c['usage'].get('usageStatus') != 'reported')
                      for c in self.calls)}
        result['role_groups'] = dict(Counter(
            'mutation_repair' if c['role'] in ('mutation', 'repair') else c['role']
            for c in self.calls))
        return result

    def write_ledger(self):
        atomic_json(self.root / 'call-budget-ledger.json', {
            'updated_utc': now(), 'limits': self.limits, 'totals': self.totals(),
            'deadline_utc': self.deadline.isoformat() if self.deadline else None,
            'blocked_reason': self.blocked_reason,
            'token_definition': 'Headless reported inputTokens (uncached) + outputTokens; cached input is separate; reasoning output is included in output.',
            'elapsed_definition': 'Sum of monotonic elapsed around actual Headless requests; local embedding CPU is separate.',
            'dollars': 'Native dollar estimates are API list-price estimates, never subscription charges.',
            'admission': 'No new call after a limit is observed; the last completed call may overshoot tokens/time and remains counted.'})

    def reason(self, role=None):
        if self.blocked_reason:
            return self.blocked_reason
        if self.deadline and datetime.now(timezone.utc) >= self.deadline:
            return 'RUN1 call-admission deadline reached'
        total = self.totals()
        for key in ('calls', 'uncached_input_plus_output_tokens', 'remote_elapsed_seconds'):
            if total[key] >= self.limits[key]:
                return f'{key} budget reached ({total[key]} >= {self.limits[key]})'
        if role is not None:
            group = 'mutation_repair' if role in ('mutation', 'repair') else role
            if group not in self.limits['role_caps']:
                return f'Unclassified or unauthorized model role: {role}'
            if total['role_groups'].get(group, 0) >= self.limits['role_caps'][group]:
                return f'{group} call cap reached'
        return None

    def stop(self, reason):
        self.blocked_reason = self.blocked_reason or reason
        self.write_ledger()
        if self.runner is not None:
            self.runner.request_checkpoint(reason)

    def begin(self, kwargs):
        role = CALL_ROLE.get()
        reason = self.reason(role)
        if reason:
            self.stop(reason)
            raise CallBudgetExceeded(reason)
        command = kwargs['command']
        model = kwargs.get('model')
        if model is None or model.agent != 'codex':
            self.stop('Only configured Codex subscription routes are authorized')
            raise CallBudgetExceeded(self.blocked_reason)
        prompt = Path(command[command.index('--prompt-file') + 1]) if '--prompt-file' in command else None
        call_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]
        record = {'id': call_id, 'role': role, 'started_utc': now(),
                  'command': command, 'requested_model': model.agent_model,
                  'requested_effort': model.effort, 'billing_required': 'subscription',
                  'status': 'started', 'prompt_sha256': sha256(prompt) if prompt else None}
        if prompt:
            shutil.copyfile(prompt, self.directory / (call_id + '.prompt.md'))
        self.calls.append(record)
        atomic_json(self.directory / (call_id + '.json'), record)
        self.write_ledger()
        return record, time.monotonic()

    def finish(self, state, result=None, error=None):
        record, start = state
        record.update(elapsed_seconds=time.monotonic() - start, ended_utc=now(),
                      status='error' if error is not None else 'completed')
        if error is not None:
            record['error'] = f'{type(error).__name__}: {error}'
        if result is not None:
            record['returncode'] = result.returncode
            (self.directory / (record['id'] + '.stdout')).write_text(result.stdout or '')
            (self.directory / (record['id'] + '.stderr')).write_text(result.stderr or '')
            usage = usage_from_stdout(result.stdout)
            if usage:
                record['usage'] = usage
                routes = [a.get('route') for a in usage.get('billing', {}).get('attempts', [])]
                if not routes or any(route != 'subscription' for route in routes):
                    record['billing_violation'] = True
                if usage.get('model') != record['requested_model']:
                    record['model_mismatch'] = True
        atomic_json(self.directory / (record['id'] + '.json'), record)
        if record.get('billing_violation') or record.get('model_mismatch'):
            self.stop('Observed billing route or effective model mismatch')
        elif not record.get('usage') or record['usage'].get('usageStatus') != 'reported':
            self.stop('Reported usage unavailable; token budget cannot be verified')
        elif self.reason():
            self.stop(self.reason())
        else:
            self.write_ledger()

    async def probe(self, model_name, prompt='Reply only READY. Do not call tools.'):
        """An explicitly budgeted readiness call; launcher alone decides to use it."""
        from shinka.llm import AsyncLLMClient
        client = AsyncLLMClient(model_names=[model_name], headless_work_dir=str(self.root))
        token = CALL_ROLE.set('readiness')
        try:
            return await client.query(msg=prompt, system_msg='Subscription readiness check.')
        finally:
            CALL_ROLE.reset(token)


def install_run1_call_audit(budget):
    """Cover every native Headless role at the actual subprocess boundary."""
    from shinka.llm.providers import headless
    if getattr(headless, '_run1_budget_installed', False):
        raise RuntimeError('RUN1 call auditor is already installed in this process')
    original_async = headless._run_headless_command_async
    original_sync = headless._run_headless_command_sync

    async def audited_async(**kwargs):
        state = budget.begin(kwargs)
        try:
            result = await original_async(**kwargs)
        except BaseException as exc:
            budget.finish(state, error=exc)
            raise
        budget.finish(state, result=result)
        if state[0].get('billing_violation') or state[0].get('model_mismatch'):
            raise RuntimeError('Quarantined response with billing route or effective model mismatch')
        return result

    def audited_sync(**kwargs):
        state = budget.begin(kwargs)
        try:
            result = original_sync(**kwargs)
        except BaseException as exc:
            budget.finish(state, error=exc)
            raise
        budget.finish(state, result=result)
        if state[0].get('billing_violation') or state[0].get('model_mismatch'):
            raise RuntimeError('Quarantined response with billing route or effective model mismatch')
        return result

    headless._run_headless_command_async = audited_async
    headless._run_headless_command_sync = audited_sync
    headless._run1_budget_installed = True


def tag_client(client, role):
    original = client.query

    async def query(*args, **kwargs):
        resolved_role = role(kwargs) if callable(role) else role
        token = CALL_ROLE.set(resolved_role)
        try:
            return await original(*args, **kwargs)
        finally:
            CALL_ROLE.reset(token)
    client.query = query


class Run1Runner(CheckpointRunner):
    """Pinned native runner with explicit, auditable RUN1 compatibility hooks."""
    def __init__(self, *args, call_budget, immutable_prompt_boundary=IMMUTABLE_BOUNDARY, **kwargs):
        self.call_budget = call_budget
        self.immutable_prompt_boundary = immutable_prompt_boundary
        self._saved_state = None
        self._last_sample = {}
        self._last_patch = {}
        self._embedding_evidence = {}
        self._novelty_evidence = {}
        self._checkpoint_reasons = []
        self._completed_side_effects = set()
        self._saturation_reviews = set()
        self._terminal_persist_in_progress = set()
        super().__init__(*args, **kwargs)
        if self.max_proposal_jobs != 1 or self.max_evaluation_jobs != 1:
            raise ValueError('RUN1 adapter is scoped to one proposal and one evaluation worker')
        self._authorized_total_slots = self.evo_config.num_generations
        call_budget.runner = self
        self._install_role_tags()
        self._install_novelty_audit()
        # Add boundaries after native format/recommendation composition, including fixes.
        for name in ('sample', 'sample_fix'):
            original = getattr(self.prompt_sampler, name)
            def bounded(*args, _original=original, **kwargs):
                system, user, operator = _original(*args, **kwargs)
                return system + '\n\n' + self.immutable_prompt_boundary, user, operator
            setattr(self.prompt_sampler, name, bounded)

    def event(self, kind, **data):
        path = Path(self.results_dir) / 'native-events.jsonl'
        with path.open('a') as handle:
            handle.write(json.dumps(jsonable({'utc': now(), 'kind': kind, **data})) + '\n')

    def _install_role_tags(self):
        from shinka.prompts.prompts_meta import META_STEP1_SYSTEM_MSG, META_STEP2_SYSTEM_MSG, META_STEP3_SYSTEM_MSG
        tag_client(self.llm, lambda k: 'repair' if k.get('msg_history') else 'mutation')
        if self.prompt_llm:
            tag_client(self.prompt_llm, 'prompt_mutation')
        if self.novelty_judge:
            tag_client(self.novelty_judge.async_llm_client, 'novelty')
        if self.meta_summarizer:
            roles = {META_STEP1_SYSTEM_MSG: 'summary', META_STEP2_SYSTEM_MSG: 'global_insight',
                     META_STEP3_SYSTEM_MSG: 'recommendation'}
            client = self.meta_summarizer.async_llm_client
            tag_client(client, lambda k: roles.get(k.get('system_msg'), 'unknown'))
            original = client.batch_kwargs_query
            async def complete_batch(*args, **kwargs):
                # Native batch dispatch bypasses client.query, so its context
                # must be tagged at the batch boundary as well.
                token = CALL_ROLE.set(roles.get(kwargs.get('system_msg'), 'unknown'))
                try:
                    result = await original(*args, **kwargs)
                finally:
                    CALL_ROLE.reset(token)
                expected = kwargs.get('num_samples', args[0] if args else None)
                if (not result or len(result) != expected or
                        any(r is None or not getattr(r, 'content', None) for r in result)):
                    self.event('partial_meta_batch_rejected', expected=kwargs.get('num_samples'),
                               returned=len(result) if result else 0)
                    return None
                return result
            client.batch_kwargs_query = complete_batch

    def _install_novelty_audit(self):
        if not self.novelty_judge:
            return
        original = self.novelty_judge.assess_novelty_with_rejection_sampling_async
        async def assess(exec_fname, code_embedding, parent_program, db):
            accepted, metadata = await original(exec_fname, code_embedding, parent_program, db)
            explanation = metadata.get('novelty_explanation', '')
            if ('similarity_scores' not in metadata or
                    explanation.startswith(('Error in novelty', 'LLM response was empty'))):
                if self.call_budget.reason('novelty'):
                    raise ProposalHeld(self.call_budget.reason('novelty'))
                accepted = False
                metadata['run1_fail_closed'] = True
                self.request_checkpoint('Novelty judge unavailable; no silent acceptance')
            evidence = {'source_sha256': sha256(exec_fname), 'parent_id': parent_program.id,
                        'accepted': accepted, 'metadata': metadata}
            self._novelty_evidence[str(Path(exec_fname).resolve())] = evidence
            self.event('novelty_decision', **evidence)
            return accepted, metadata
        self.novelty_judge.assess_novelty_with_rejection_sampling_async = assess
        original_check = self.novelty_judge._check_llm_novelty_async
        async def check(*args, **kwargs):
            reason = self.call_budget.reason('novelty')
            if reason:
                self.call_budget.stop(reason)
                raise ProposalHeld(reason)
            return await original_check(*args, **kwargs)
        self.novelty_judge._check_llm_novelty_async = check

    def request_checkpoint(self, reason):
        if reason in self._checkpoint_reasons:
            return
        self._checkpoint_reasons.append(reason)
        # Native increments next_generation_to_submit BEFORE scheduling the task.
        active = [int(t.get_name().split('_')[1]) + 1
                  for t in self.active_proposal_tasks.values()
                  if t.get_name().startswith('proposal_')]
        running = [j.generation + 1 for j in self.running_jobs]
        stop = max([self.completed_generations, self.next_generation_to_submit, *active, *running])
        self.evo_config.num_generations = min(self._authorized_total_slots, stop)
        self.event('checkpoint_requested', reason=reason, drain_through_total_slots=self.evo_config.num_generations,
                   authorized_total_slots=self._authorized_total_slots)
        self.slot_available.set()

    def _save_state(self, boundary):
        rng = np.random.get_state()
        bandit = self.llm_selection
        state = {'version': 1, 'saved_utc': now(), 'boundary': boundary,
                 'authorized_total_slots': self._authorized_total_slots,
                 'python_rng': random.getstate(), 'numpy_rng': jsonable(rng),
                 'bandit_state': jsonable(bandit.get_state()) if bandit else None,
                 'bandit_rng': jsonable(bandit.rng.bit_generator.state) if bandit else None,
                 'prompt_evolution_counter': self.prompt_evolution_counter,
                 'prompt_percentile_recompute_counter': self.prompt_percentile_recompute_counter,
                 'prompt_api_cost': self.prompt_api_cost,
                 'saturation_reviews': sorted(self._saturation_reviews),
                 'side_effects_completed_program_ids': sorted(self._completed_side_effects)}
        if self.meta_summarizer:
            path = Path(self.results_dir) / 'run1-meta-state.json'
            self.meta_summarizer.save_meta_state(str(path))
            state['meta_state_sha256'] = sha256(path)
        atomic_json(Path(self.results_dir) / 'run1-native-state.json', state)

    def _load_bandit_state(self):
        if self._saved_state and self.llm_selection:
            self.llm_selection.set_state(self._saved_state['bandit_state'])
            self.llm_selection.rng.bit_generator.state = self._saved_state['bandit_rng']
        else:
            super()._load_bandit_state()

    async def _setup_prompt_evolution(self):
        # Upstream mistakes a nonempty generation-zero archive for a new one.
        from shinka.database.prompt_dbase import SystemPromptConfig, SystemPromptDatabase, create_system_prompt
        from shinka.core.prompt_evolver import SystemPromptSampler, AsyncSystemPromptEvolver
        path = Path(self.results_dir) / 'prompts.sqlite'
        config = SystemPromptConfig(db_path=str(path), archive_size=self.evo_config.prompt_archive_size,
                                    ucb_exploration_constant=self.evo_config.prompt_ucb_exploration_constant,
                                    epsilon=self.evo_config.prompt_epsilon)
        self.prompt_db = SystemPromptDatabase(config)
        if self.prompt_db._count_prompts_in_db() == 0:
            self.prompt_db.add(create_system_prompt(
                prompt_text=self.evo_config.task_sys_msg or 'You are an expert software engineer.',
                generation=0, patch_type='init', metadata={'source': 'initial_config'},
                name='initial_system_prompt', description='Initial system prompt provided by the user.'), verbose=self.verbose)
        self.prompt_sampler_evo = SystemPromptSampler(self.prompt_db,
            exploration_constant=self.evo_config.prompt_ucb_exploration_constant, epsilon=self.evo_config.prompt_epsilon)
        self.prompt_evolver = AsyncSystemPromptEvolver(self.prompt_llm,
            patch_types=self.evo_config.prompt_patch_types,
            patch_type_probs=self.evo_config.prompt_patch_type_probs,
            llm_kwargs=self.evo_config.prompt_llm_kwargs)

    async def _setup_async(self):
        path = Path(self.results_dir) / 'run1-native-state.json'
        if path.exists():
            self._saved_state = json.loads(path.read_text())
            state = self._saved_state
            if state['authorized_total_slots'] != self._authorized_total_slots:
                raise ValueError('Authorized total slot budget changed on resume')
            random.setstate(tuples(state['python_rng']))
            nr = state['numpy_rng']
            np.random.set_state((nr[0], np.array(nr[1], dtype='uint32'), nr[2], nr[3], nr[4]))
            self.prompt_evolution_counter = state['prompt_evolution_counter']
            self.prompt_percentile_recompute_counter = state['prompt_percentile_recompute_counter']
            self.prompt_api_cost = state['prompt_api_cost']
            self._completed_side_effects = set(state['side_effects_completed_program_ids'])
            self._saturation_reviews = set(state.get('saturation_reviews', []))
        await super()._setup_async()
        if self._saved_state:
            self._load_bandit_state()  # Also handles completed seed / generation-zero resume.
            if self.meta_summarizer:
                meta = Path(self.results_dir) / 'run1-meta-state.json'
                if sha256(meta) != self._saved_state['meta_state_sha256']:
                    raise ValueError('Native meta checkpoint hash mismatch')
                if not self.meta_summarizer.load_meta_state(str(meta)):
                    raise ValueError('Native meta checkpoint could not be restored')
            self.event('native_state_restored', source_sha256=sha256(path),
                       prompt_evolution_counter=self.prompt_evolution_counter,
                       recommendation_history_count=len(self.meta_summarizer.meta_recommendations_history) if self.meta_summarizer else 0,
                       bandit_rng_restored=self.llm_selection is not None)
        await self._recover_accepted_proposals()
        self._save_state('setup_complete')

    async def _recover_accepted_proposals(self):
        from shinka.core.async_runner import AsyncRunningJob
        root = Path(self.results_dir)
        for source in sorted(root.glob('gen_*/main.py'), key=lambda p: int(p.parent.name[4:])):
            generation = int(source.parent.name[4:])
            if self.db.get_programs_by_generation(generation):
                continue
            checkpoint = source.parent / 'accepted-proposal.json'
            if not checkpoint.exists():
                raise ValueError(f'Unaccepted saved proposal retained at {source}; resume requires completing its saved novelty stage, not regenerating or bypassing it')
            value = json.loads(checkpoint.read_text())
            if sha256(source) != value['source_sha256']:
                raise ValueError('Accepted proposal source changed')
            if generation != self.next_generation_to_submit:
                raise ValueError('Accepted proposal is not the next native slot')
            data = value['job']
            for key in ('parent_id',):
                if not self.db.get(data[key]):
                    raise ValueError('Accepted proposal parent absent')
            output = source.parent / ('recovery-evaluation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            job_id, worker, submitted, started, running = await self._submit_evaluation_job_with_slot(str(source), str(output), None)
            data.update(job_id=job_id, results_dir=str(output), evaluation_worker_id=worker,
                        evaluation_submitted_at=submitted, evaluation_started_at=started,
                        running_eval_jobs_at_submit=running, start_time=time.time())
            data['meta_patch_data']['recovery_results_dir'] = str(output)
            data['meta_patch_data']['recovered_after_native_novelty_acceptance'] = True
            job = AsyncRunningJob(**data)
            while await self.scheduler.check_job_status_async(job):
                if time.time() - data['start_time'] > self._get_evaluation_runtime_limit_seconds():
                    await self.scheduler.cancel_job_async(job_id)
                    raise RuntimeError('Recovered evaluation exceeded frozen runtime limit')
                await asyncio.sleep(1)
            job.completion_detected_at = time.time()
            if not await self._process_single_job_safely(job):
                raise RuntimeError('Recovered proposal did not persist')
            await self._restore_resume_progress()
            self.event('accepted_proposal_recovered', generation=generation, source_sha256=value['source_sha256'])

    def _sample_record(self, parent, archive, top_k, generation, kwargs, meta=None):
        data = {'generation': generation, 'parent_id': parent.id, 'parent_sha256': hashlib.sha256(parent.code.encode()).hexdigest(),
                'parent_island': parent.island_idx, 'archive_inspiration_ids': [p.id for p in archive],
                'top_k_inspiration_ids': [p.id for p in top_k], 'parent_text_feedback': parent.text_feedback,
                'parent_feedback_is_string': isinstance(parent.text_feedback, str), 'recommendation': meta,
                'novelty_attempt': kwargs.get('novelty_attempt', 1), 'resample_attempt': kwargs.get('resample_attempt', 1)}
        self._last_sample[generation] = data
        self.event('sampling', **data)
        self._save_state('parent_sampled')
        return data

    def _save_generated(self, generation, sample, result):
        if not result or not result[2]:
            return
        directory = Path(self.results_dir) / f'gen_{generation}'
        source = directory / 'main.py'
        attempts = directory / 'proposal-sources'
        attempts.mkdir(exist_ok=True)
        stem = f"novelty-{sample['novelty_attempt']}-resample-{sample['resample_attempt']}"
        shutil.copyfile(source, attempts / (stem + '.py'))
        atomic_json(attempts / (stem + '.json'), {'stage': 'generated_before_novelty',
            'source_sha256': sha256(source), 'sampling': sample, 'code_diff': result[0], 'patch_metadata': result[1]})
        self._last_patch[generation] = {'code_diff': result[0], 'patch_metadata': result[1]}
        self._save_state('generated_before_novelty')

    async def _run_patch_async(self, parent_program, archive_programs, top_k_programs, generation, meta_recs=None, **kwargs):
        if self.call_budget.reason('mutation'):
            raise ProposalHeld(self.call_budget.reason('mutation'))
        sample = self._sample_record(parent_program, archive_programs, top_k_programs, generation, kwargs, meta_recs)
        result = await super()._run_patch_async(parent_program, archive_programs, top_k_programs, generation, meta_recs, **kwargs)
        self._save_generated(generation, sample, result)
        if (not result or not result[2]) and self.call_budget.reason('mutation'):
            raise ProposalHeld(self.call_budget.reason('mutation'))
        return result

    async def _run_fix_patch_async(self, incorrect_program, ancestor_inspirations, generation, **kwargs):
        if self.call_budget.reason('repair'):
            raise ProposalHeld(self.call_budget.reason('repair'))
        sample = self._sample_record(incorrect_program, ancestor_inspirations, [], generation, kwargs)
        result = await super()._run_fix_patch_async(incorrect_program, ancestor_inspirations, generation, **kwargs)
        self._save_generated(generation, sample, result)
        if (not result or not result[2]) and self.call_budget.reason('repair'):
            raise ProposalHeld(self.call_budget.reason('repair'))
        return result

    async def _get_code_embedding_async(self, exec_fname):
        started = time.monotonic()
        embedding, cost = await super()._get_code_embedding_async(exec_fname)
        source = Path(exec_fname).read_text()
        evidence = {'source_sha256': sha256(exec_fname), 'embedded_prefix_sha256': hashlib.sha256(source[:10000].encode()).hexdigest(),
                    'source_characters': len(source), 'embedded_characters': min(10000, len(source)),
                    'dimensions': len(embedding) if embedding else 0, 'elapsed_seconds': time.monotonic() - started,
                    'native_estimated_api_cost': cost}
        self.event('local_embedding', **evidence)
        self._embedding_evidence[str(Path(exec_fname).resolve())] = evidence
        if not embedding or not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
            self.request_checkpoint('Local embedding unavailable; native novelty cannot be skipped')
            raise EmbeddingUnavailable('Missing, zero or nonfinite local embedding')
        return embedding, cost

    async def _generate_evolved_proposal(self, generation, task_id, exec_fname, results_dir,
                                        meta_recs, meta_summary, meta_scratch, proposal_started_at,
                                        sampling_worker_id, active_proposals_at_start):
        try:
            job = await super()._generate_evolved_proposal(generation, task_id, exec_fname, results_dir,
                meta_recs, meta_summary, meta_scratch, proposal_started_at, sampling_worker_id, active_proposals_at_start)
        except ProposalHeld as exc:
            self.call_budget.stop(str(exc))
            # This reserved generation has no accepted/evaluated outcome yet.
            # Drain preceding work and leave all attempt/source files intact.
            self.evo_config.num_generations = min(self.evo_config.num_generations, generation)
            atomic_json(Path(exec_fname).parent / 'held-proposal.json', {
                'generation': generation, 'reason': str(exc), 'held_utc': now(),
                'stage': 'generated_before_novelty' if Path(exec_fname).exists() else 'before_valid_proposal',
                'source_sha256': sha256(exec_fname) if Path(exec_fname).exists() else None,
                'sampling': self._last_sample.get(generation),
                'authorized_total_slots': self._authorized_total_slots,
                'counts_as_completed_slot': False,
                'counts_as_reserved_slot': True})
            self.event('proposal_held', generation=generation, reason=str(exc))
            self._save_state('proposal_held')
            self.slot_available.set()
            return None
        except EmbeddingUnavailable as exc:
            sample = self._last_sample[generation]
            patch = self._last_patch.get(generation, {})
            metadata = patch.get('patch_metadata', {'patch_name': 'embedding_unavailable'})
            await self._record_terminal_failed_proposal(generation=generation, exec_fname=exec_fname,
                proposal_started_at=proposal_started_at, sampling_worker_id=sampling_worker_id,
                active_proposals_at_start=active_proposals_at_start, parent_program=self.db.get(sample['parent_id']),
                archive_programs=[self.db.get(i) for i in sample['archive_inspiration_ids']],
                top_k_programs=[self.db.get(i) for i in sample['top_k_inspiration_ids']],
                code_diff=patch.get('code_diff'), meta_patch_data=metadata, code_embedding=None,
                embed_cost=0, novelty_cost=0, api_costs=metadata.get('api_costs', 0),
                failure_stage='embedding', failure_reason=str(exc))
            return None
        if job:
            from dataclasses import fields
            # Local job_id is ProcessWithLogging, containing unpicklable locks.
            # Its old PID is provenance only; recovery submits a new process.
            data = {field.name: getattr(job, field.name) for field in fields(job) if field.name != 'job_id'}
            data['job_id'] = str(getattr(job.job_id, 'pid', job.job_id))
            atomic_json(Path(exec_fname).parent / 'accepted-proposal.json', {
                'stage': 'accepted_after_native_novelty', 'source_sha256': sha256(exec_fname), 'job': data,
                'embedding_evidence': self._embedding_evidence.get(str(Path(exec_fname).resolve())),
                'novelty_evidence': self._novelty_evidence.get(str(Path(exec_fname).resolve()))})
            self._save_state('accepted_proposal_submitted')
        return job

    async def _record_terminal_failed_proposal(self, **kwargs):
        # Native terminal proposal failures otherwise live only in attempt_log.
        generation = kwargs['generation']
        self._terminal_persist_in_progress.add(generation)
        try:
            await super()._record_terminal_failed_proposal(**kwargs)
            program = await self._persist_failed_generation(**kwargs)
            if program is None:
                raise RuntimeError('Terminal proposal failure could not be counted in the native database')
            await self._review_saturation()
            from shinka.core.async_runner import AsyncRunningJob, PersistedProgramEvent
            timestamp = time.time()
            job = AsyncRunningJob(job_id=f"failed:{generation}", exec_fname=kwargs['exec_fname'],
                results_dir=str(Path(kwargs['exec_fname']).parent / 'results'), start_time=timestamp,
                proposal_started_at=kwargs['proposal_started_at'], evaluation_submitted_at=timestamp,
                generation=generation, parent_id=program.parent_id,
                meta_patch_data=kwargs['meta_patch_data'] or {})
            # Use the same native background queue as evaluated programs. Keep
            # this generation out of the completed counter until its side
            # effects are queued; otherwise native can cancel the active
            # proposal immediately at the final slot and lose its credit.
            await self._enqueue_background_side_effects([
                PersistedProgramEvent(job, program, timestamp, timestamp, timestamp)])
            self.event('failed_slot_counted', generation=generation, program_id=program.id,
                       failure_stage=kwargs['failure_stage'])
        finally:
            self._terminal_persist_in_progress.discard(generation)
        await self._update_completed_generations()

    async def _count_completed_generations_from_db(self):
        count = await super()._count_completed_generations_from_db()
        return max(0, count - len(self._terminal_persist_in_progress))

    async def _apply_persisted_program_side_effects(self, event):
        if event.program.id in self._completed_side_effects:
            return
        await super()._apply_persisted_program_side_effects(event)
        self._completed_side_effects.add(event.program.id)
        self._save_state('program_side_effects_complete')

    async def _persist_completed_job(self, job):
        result = await super()._persist_completed_job(job)
        if result.success:
            await self._review_saturation()
        return result

    async def _review_saturation(self):
        """Prospective development-only ceiling review; never an assessment rule."""
        generations = sorted(set(await self.async_db.get_persisted_generation_ids_async()))
        slots = len(generations)
        if slots not in (8, 12, 16, 20, 24, 28) or slots in self._saturation_reviews:
            return
        self._saturation_reviews.add(slots)
        latest = [g for g in generations if g > 0][-3:]
        records = []
        for generation in latest:
            programs = await self.async_db.get_programs_by_generation_async(generation)
            if len(programs) != 1:
                raise ValueError('Non-seed generation has multiple native rows during saturation review')
            p = programs[0]
            records.append({'generation': generation, 'correct': p.correct,
                            'score': p.combined_score, 'escape': p.public_metrics.get('escape')})
        saturated = saturation_condition(records)
        self.event('development_saturation_review', completed_slots=slots,
                   latest_nonseed=records, saturated=saturated,
                   criterion='three correct rows, each escape == 1.0, task score range <= .002')
        if saturated:
            self.request_checkpoint('prospective development saturation rule')

    async def _cleanup_async(self):
        self._save_state('controller_finalization')
        await super()._cleanup_async()


def saturation_condition(records):
    if len(records) != 3 or not all(r['correct'] and r['escape'] == 1.0 for r in records):
        return False
    scores = [r['score'] for r in records]
    return all(isinstance(s, (int, float)) and np.isfinite(s) for s in scores) and max(scores) - min(scores) <= .002
