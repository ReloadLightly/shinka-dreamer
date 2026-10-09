"""Original-task reuse of native Shinka adapters with call/time primary budgets."""
from datetime import datetime, timezone
from contextvars import ContextVar
import json
from pathlib import Path
import time

from dreamer.native_run1 import (Run1CallBudget, Run1Runner, now,
    usage_from_stdout, ProposalHeld)
from dreamer.native_v3 import atomic_json
from dreamer.native_run1_runtime_fix import install_scheduler_timeout

LOCK_WAIT = ContextVar('proposal_headless_lock_wait', default=0.)


def install_lock_wait_audit():
    from shinka.llm.providers import headless
    original = headless._acquire_cli_lock_async
    async def acquire():
        started = time.monotonic()
        await original()
        LOCK_WAIT.set(time.monotonic()-started)
    headless._acquire_cli_lock_async = acquire

BOUNDARY = (
    'NON-EVOLVING ORIGINAL-PROPOSAL BOUNDARY: Jointly evolve world_model_step '
    'and planner, including helpers and arbitrary internal representations. '
    'Export current memory["believed_map"] in initial-relative coordinates. '
    'The fixed 15x15 maze, local observations, 200-step horizon, evaluator, '
    'isolation and resource rules cannot be edited. Selection is the original '
    '0.6 task reward + 0.4 current-map accuracy, not a forecast objective. '
    'Do not inspect files, tools, seeds, hidden state or assessment pools. '
    'All candidate network/process/model calls are forbidden.'
)


class FullCallBudget(Run1CallBudget):
    """Unknown tokens remain unknown; actual calls and request time bound work."""
    def _load(self):
        self.calls = [json.loads(p.read_text()) for p in sorted(self.directory.glob('*.json'))]
        if any(c.get('status') == 'started' for c in self.calls):
            self.blocked_reason = 'Unresolved prior request; reconcile before continuation'
        elif any(c.get('billing_violation') or c.get('model_mismatch') for c in self.calls):
            self.blocked_reason = 'Prior billing or effective model mismatch requires review'
        self.write_ledger()

    def begin(self, kwargs):
        state = super().begin(kwargs)
        state[0]['cli_lock_wait_seconds'] = LOCK_WAIT.get()
        return state

    def reason(self, role=None):
        if self.blocked_reason:
            return self.blocked_reason
        reserve = self.limits['request_timeout_seconds'] if role is not None else 0
        if self.deadline and (self.deadline-datetime.now(timezone.utc)).total_seconds() <= reserve:
            return 'Execution deadline/reserved request duration reached'
        totals = self.totals()
        if totals['calls'] >= self.limits['calls']:
            return 'All-role call cap reached'
        if totals['remote_elapsed_seconds'] + reserve >= self.limits['remote_elapsed_seconds']:
            return 'Provider-time cap/reserved request duration reached'
        recent = self.calls[-2:]
        if len(recent) == 2 and all(c.get('status') == 'error' or
                (c.get('returncode') is not None and c['returncode'] != 0) for c in recent):
            return 'Two consecutive provider transport failures'
        if role is not None:
            group = 'mutation_repair' if role in ('mutation','repair') else role
            cap = self.limits['role_caps'].get(group, 0)
            if totals['role_groups'].get(group, 0) >= cap:
                return f'Unauthorized role or role cap: {role}'
        return None

    def finish(self, state, result=None, error=None):
        record, started = state
        record.update(elapsed_seconds=time.monotonic()-started, ended_utc=now(),
                      status='error' if error is not None else 'completed')
        if error is not None:
            record['error'] = f'{type(error).__name__}: {error}'
        if result is not None:
            record['returncode'] = result.returncode
            (self.directory/(record['id']+'.stdout')).write_text(result.stdout or '')
            (self.directory/(record['id']+'.stderr')).write_text(result.stderr or '')
            usage = usage_from_stdout(result.stdout)
            if usage:
                record['usage'] = usage
                routes = [a.get('route') for a in usage.get('billing',{}).get('attempts',[])]
                if (routes and any(r != 'subscription' for r in routes)) or (usage.get('usageStatus')=='reported' and not routes):
                    record['billing_violation'] = True
                if usage.get('model') is not None and usage['model'] != record['requested_model']:
                    record['model_mismatch'] = True
        record['usage_missing'] = (not record.get('usage') or
                                   record['usage'].get('usageStatus') != 'reported')
        atomic_json(self.directory/(record['id']+'.json'), record)
        if record.get('billing_violation') or record.get('model_mismatch'):
            self.stop('Observed billing or effective model mismatch')
        elif self.reason():
            self.stop(self.reason())
        else:
            self.write_ledger()

    def write_ledger(self):
        super().write_ledger()
        path = self.root/'call-budget-ledger.json'
        value = json.loads(path.read_text())
        value.update(primary_budgets=['actual_all_role_calls','provider_elapsed_seconds','deadline'],
                     token_budget_enforced=False,
                     cli_lock_wait_seconds=sum(c.get('cli_lock_wait_seconds',0) for c in self.calls),
                     admission='Reserve maximum request duration before actual dispatch; unknown tokens are not zero and do not alone stop work.')
        atomic_json(path,value)


class FullRunner(Run1Runner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,immutable_prompt_boundary=BOUNDARY,**kwargs)
        install_scheduler_timeout(self.scheduler)

    async def _review_saturation(self):
        # The predictive task-only escape rule is inapplicable to this objective.
        return

    async def _recover_accepted_proposals(self):
        from shinka.core.async_runner import AsyncRunningJob
        from dreamer.provenance import sha256
        for checkpoint in sorted(Path(self.results_dir).glob('gen_*/accepted-proposal.json')):
            generation = int(checkpoint.parent.name[4:])
            if self.db.get_programs_by_generation(generation):
                continue
            accepted = json.loads(checkpoint.read_text())
            data = accepted['job']
            output = Path(data['results_dir'])
            if not (output/'episodes.json').exists():
                continue
            if sha256(checkpoint.parent/'main.py') != accepted['source_sha256']:
                raise ValueError('Saved accepted source changed')
            if not all((output/f).exists() for f in ('metrics.json','correct.json','resource.json')):
                raise ValueError('Partial evaluation artifacts require reconciliation, not a duplicate run')
            data.update(job_id=f'saved:{generation}',evaluation_slot_released=True,
                        completion_detected_at=time.time())
            if not await self._process_single_job_safely(AsyncRunningJob(**data)):
                raise RuntimeError('Saved complete evaluation could not persist')
            await self._restore_resume_progress()
            self.event('completed_evaluation_recovered_without_rerun',generation=generation)
        await super()._recover_accepted_proposals()

    async def _setup_initial_program(self, code):
        await super()._setup_initial_program(code)
        output = Path(self.results_dir)/'gen_0'/'results'
        correct = json.loads((output/'correct.json').read_text())
        metrics = json.loads((output/'metrics.json').read_text())
        rows = json.loads((output/'episodes.json').read_text())
        resources = json.loads((output/'resource.json').read_text())
        programs = self.db.get_programs_by_generation(0)
        if (not correct.get('correct') or len(rows) != 5 or any(r.get('error') for r in rows)
                or not resources.get('completed') or any(p.metadata.get('evaluation_failed') for p in programs)
                or not isinstance(metrics.get('text_feedback'),str)):
            raise RuntimeError('Seed lacks five valid real evaluation artifacts and string feedback')

    async def _submit_evaluation_job_with_slot(self,exec_fname,results_dir,sampling_worker_id=None):
        # Stop rather than duplicate a completed evaluation on recovery.
        if (Path(results_dir)/'episodes.json').exists():
            raise RuntimeError('Completed evaluation exists; reconcile persistence instead of rerunning')
        return await super()._submit_evaluation_job_with_slot(exec_fname,results_dir,sampling_worker_id)
