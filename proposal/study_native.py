"""Additive all-role route checks for new independent original-task searches."""
import hashlib
import json
import os
from pathlib import Path

from dreamer.native_run1 import CALL_ROLE
from dreamer.native_v3 import atomic_json
from proposal.native_full import FullCallBudget, FullRunner


class StudyCallBudget(FullCallBudget):
    def __init__(self, *args, allowed_roles=None, **kwargs):
        self.allowed_roles = allowed_roles
        super().__init__(*args, **kwargs)

    def begin(self, kwargs):
        if self.allowed_roles is not None and CALL_ROLE.get() not in self.allowed_roles:
            self.stop(f'Role forbidden for this treatment: {CALL_ROLE.get()}')
            raise RuntimeError(self.blocked_reason)
        command = kwargs.get('command', [])
        if (command.count('--allow') != 1 or
                command[command.index('--allow') + 1:command.index('--allow') + 2] != ['read-only']):
            self.stop('Headless request lacks the required read-only sandbox route')
            raise RuntimeError(self.blocked_reason)
        state = super().begin(kwargs)
        os.environ['SHINKA_STUDY_TOOL_AUDIT_ID'] = state[0]['id']
        state[0]['codex_tool_audit'] = f"codex-event-audit/{state[0]['id']}.summary.json"
        atomic_json(self.directory/(state[0]['id']+'.json'), state[0])
        return state

    def finish(self, state, result=None, error=None):
        record, _ = state
        directory = self.root/'codex-event-audit'
        summary_path = directory/(record['id']+'.summary.json')
        raw_path = directory/(record['id']+'.events.jsonl')
        problem = None
        try:
            summary = json.loads(summary_path.read_text())
            if not isinstance(summary, dict):
                raise ValueError('Tool audit summary must be an object')
            if summary.get('unauthorized_tool_event'):
                problem = 'unauthorized_tool_event'
            elif (summary.get('raw_events_file') != raw_path.name or
                    summary.get('raw_events_sha256') != hashlib.sha256(raw_path.read_bytes()).hexdigest()):
                problem = 'raw_event_audit_mismatch'
            record['codex_tool_audit_outcome'] = summary
        except (OSError, ValueError, TypeError) as exc:
            problem = f'missing_or_invalid_audit:{type(exc).__name__}'
        record['mutation_context_validated'] = problem is None
        record['context_audit_failure'] = problem
        if problem:
            # Includes failures before Codex starts: counted requests, never a
            # context-validated response or permission for a second dispatch.
            self.stop('Codex context audit failed: '+problem)
        super().finish(state, result=result, error=error)
        if problem:
            raise RuntimeError('Quarantined model response: Codex context audit failed: '+problem)


class StudyRunner(FullRunner):
    """Record a common all-role call prefix at native proposal acceptance."""
    async def _setup_initial_program(self, code):
        await super()._setup_initial_program(code)
        evidence = {'generation': 0, 'all_role_calls_at_acceptance': 0,
            'provider_elapsed_seconds_at_acceptance': 0., 'definition': 'Original seed before any search-model request.'}
        atomic_json(Path(self.results_dir)/'gen_0'/'acceptance-budget.json', evidence)
        self.event('proposal_acceptance_budget', **evidence)

    async def _submit_evaluation_job_with_slot(self, exec_fname, results_dir, sampling_worker_id=None):
        source = Path(exec_fname)
        generation = int(source.parent.name.removeprefix('gen_'))
        totals = self.call_budget.totals()
        evidence = {'generation': generation,
            'all_role_calls_at_acceptance': totals['calls'],
            'provider_elapsed_seconds_at_acceptance': totals['remote_elapsed_seconds'],
            'definition': 'All actually dispatched role calls before evaluation submission, including failures and any auxiliary request already in flight.'}
        atomic_json(source.parent/'acceptance-budget.json', evidence)
        self.event('proposal_acceptance_budget', **evidence)
        return await super()._submit_evaluation_job_with_slot(exec_fname, results_dir, sampling_worker_id)

    async def _generate_evolved_proposal(self, generation, *args, **kwargs):
        job = await super()._generate_evolved_proposal(generation, *args, **kwargs)
        if job is not None:
            directory = Path(job.exec_fname).parent
            evidence = json.loads((directory/'acceptance-budget.json').read_text())
            job.meta_patch_data.update(evidence)
            accepted = json.loads((directory/'accepted-proposal.json').read_text())
            accepted['job']['meta_patch_data'].update(evidence)
            atomic_json(directory/'accepted-proposal.json', accepted)
        return job


class IndependentRewriteRunner(StudyRunner):
    """Native best-of-N with fail-closed seed-only independent rewrite context."""
    def __init__(self, *args, rewrite_seed_sha256, model_offset, **kwargs):
        self.rewrite_seed_sha256 = rewrite_seed_sha256
        self.model_offset = model_offset
        super().__init__(*args, **kwargs)

    async def _run_patch_async(self, parent_program, archive_programs, top_k_programs,
                               generation, meta_recs=None, **kwargs):
        if (parent_program.generation != 0 or
                hashlib.sha256(parent_program.code.encode()).hexdigest() != self.rewrite_seed_sha256 or
                archive_programs or top_k_programs or meta_recs):
            raise RuntimeError('Independent rewrite received non-seed or accumulated search context')
        index = (generation-1+self.model_offset) % len(self.evo_config.llm_models)
        kwargs['model_sample_probs'] = [float(i == index) for i in range(len(self.evo_config.llm_models))]
        kwargs['model_posterior'] = None
        return await super()._run_patch_async(parent_program, [], [], generation, None, **kwargs)

    async def _run_fix_patch_async(self, *args, **kwargs):
        raise RuntimeError('Independent rewrite does not repair previous candidate programs')

    async def _get_code_embedding_async(self, exec_fname):
        # Explicit treatment removal, not a failed/missing full-arm embedding.
        return [], 0.
