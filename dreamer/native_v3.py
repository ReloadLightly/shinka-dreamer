"""Additive v3 audit/recovery hooks; native selection and mutations stay upstream."""
from datetime import datetime, timezone
import hashlib
import json
import random
from pathlib import Path
import time
import uuid

from dreamer.native import CheckpointRunner
from dreamer.provenance import record, sha256
from scripts.recover_campaign import recovery_runner


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def install_call_audit(root):
    """Preserve every native mutation/repair/meta CLI result, including billing."""
    from shinka.llm.providers import headless
    directory = Path(root) / 'model-calls'
    directory.mkdir(exist_ok=True)
    original_async = headless._run_headless_command_async
    original_sync = headless._run_headless_command_sync

    def begin(kwargs):
        call_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]
        command = kwargs['command']
        prompt = Path(command[command.index('--prompt-file') + 1]) if '--prompt-file' in command else None
        value = {'id': call_id, 'started_utc': datetime.now(timezone.utc).isoformat(),
                 'command': command, 'prompt_sha256': sha256(prompt) if prompt else None,
                 'status': 'started', 'billing_required': 'subscription'}
        atomic_json(directory / (call_id + '.json'), value)
        return call_id, time.monotonic(), value

    def finish(state, result=None, error=None):
        call_id, start, value = state
        value['elapsed_seconds'] = time.monotonic() - start
        value['status'] = 'error' if error is not None else 'completed'
        if result is not None:
            value['returncode'] = result.returncode
            (directory / (call_id + '.stdout')).write_text(result.stdout or '')
            (directory / (call_id + '.stderr')).write_text(result.stderr or '')
            for line in reversed((result.stdout or '').splitlines()):
                try:
                    usage = json.loads(line).get('usage')
                except (ValueError, AttributeError):
                    continue
                if usage:
                    value['usage'] = usage
                    attempts = usage.get('billing', {}).get('attempts', [])
                    if any(item.get('route') != 'subscription' for item in attempts):
                        value['billing_violation'] = True
                    break
        if error is not None:
            value['error'] = f'{type(error).__name__}: {error}'
        atomic_json(directory / (call_id + '.json'), value)
        if value.get('billing_violation'):
            raise RuntimeError('Observed non-subscription billing route; stop campaign')

    async def audited_async(**kwargs):
        state = begin(kwargs)
        try:
            result = await original_async(**kwargs)
        except BaseException as error:
            finish(state, error=error)
            raise
        finish(state, result=result)
        return result

    def audited_sync(**kwargs):
        state = begin(kwargs)
        try:
            result = original_sync(**kwargs)
        except BaseException as error:
            finish(state, error=error)
            raise
        finish(state, result=result)
        return result

    headless._run_headless_command_async = audited_async
    headless._run_headless_command_sync = audited_sync


class V3Runner(recovery_runner(CheckpointRunner)):
    """Keep sampling evidence before evaluations, then use existing native recovery."""

    async def _setup_async(self):
        root = Path(self.results_dir)
        contexts = []
        for path in sorted(root.glob('gen_*/proposal-checkpoint.json')):
            contexts.append(json.loads(path.read_text()))
        if contexts:
            atomic_json(root / 'recovery-jobs.json', contexts)
        state_path = root / 'native-rng-state.json'
        if state_path.exists():
            import numpy as np
            state = json.loads(state_path.read_text())
            def tuples(value):
                return tuple(map(tuples, value)) if isinstance(value, list) else value
            random.setstate(tuples(state['python']))
            ns = state['numpy']
            np.random.set_state((ns[0], np.array(ns[1], dtype='uint32'), ns[2], ns[3], ns[4]))
        await super()._setup_async()

    def _save_rng(self):
        import numpy as np
        state = np.random.get_state()
        atomic_json(Path(self.results_dir) / 'native-rng-state.json', {
            'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'note': 'Native sampling checkpoint; LLM sampling is not deterministic'})

    async def _process_single_job_safely(self, job):
        result = await super()._process_single_job_safely(job)
        self._save_rng()
        return result

    def _sampling_record(self, parent, archive, top_k, generation, mode, kwargs, meta=None):
        directory = Path(self.results_dir) / f'gen_{generation}'
        attempt = f"{mode}-{kwargs.get('novelty_attempt', 1)}-{kwargs.get('resample_attempt', 1)}"
        data = {'generation': generation, 'parent_id': parent.id,
                'parent_sha256': hashlib.sha256(parent.code.encode()).hexdigest(),
                'parent_island': parent.island_idx,
                'archive_inspiration_ids': [p.id for p in archive],
                'top_k_inspiration_ids': [p.id for p in top_k],
                'parent_feedback_is_string': isinstance(parent.text_feedback, str),
                'parent_text_feedback': parent.text_feedback,
                'mode': mode, 'recommendation': meta}
        record(directory / f'sampling-{attempt}.json', data)
        self._save_rng()
        return data

    def _checkpoint(self, generation, sampled, result):
        if not result or not result[2]:
            return
        directory = Path(self.results_dir) / f'gen_{generation}'
        source = directory / 'main.py'
        evidence = {}
        for path in [directory / 'original.py', directory / 'edit.diff']:
            if path.exists():
                evidence[str(path.relative_to(self.results_dir))] = sha256(path)
        data = {'generation': generation, 'source_sha256': sha256(source),
                'parent_id': sampled['parent_id'],
                'archive_inspiration_ids': sampled['archive_inspiration_ids'],
                'top_k_inspiration_ids': sampled['top_k_inspiration_ids'],
                'evidence_sha256': evidence, 'patch_metadata': result[1]}
        record(directory / 'proposal-checkpoint.json', data)
        self._save_rng()

    async def _run_patch_async(self, parent_program, archive_programs, top_k_programs,
                               generation, meta_recs=None, **kwargs):
        sampled = self._sampling_record(parent_program, archive_programs, top_k_programs,
                                        generation, 'mutation', kwargs, meta_recs)
        result = await super()._run_patch_async(parent_program, archive_programs,
                                               top_k_programs, generation, meta_recs, **kwargs)
        self._checkpoint(generation, sampled, result)
        return result

    async def _run_fix_patch_async(self, incorrect_program, ancestor_inspirations,
                                   generation, **kwargs):
        sampled = self._sampling_record(incorrect_program, ancestor_inspirations, [],
                                        generation, 'fix', kwargs)
        result = await super()._run_fix_patch_async(incorrect_program, ancestor_inspirations,
                                                   generation, **kwargs)
        self._checkpoint(generation, sampled, result)
        return result
