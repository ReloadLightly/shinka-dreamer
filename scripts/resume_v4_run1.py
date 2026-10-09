"""Resume the same RUN1 with recorded local-timeout compatibility fixes only."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dreamer import native_run1
    from dreamer.native_run1_runtime_fix import (CorrectedRunner,
        INFRASTRUCTURE_FEEDBACK, install_headless_child_timeout)
    from dreamer.provenance import record, sha256
    from scripts import evolve_v4
    from scripts.recover_campaign import campaign_lock
    from shinka.launch.scheduler import JobScheduler
    from shinka.llm.providers import headless

    campaign = ROOT / 'results/campaign-v4-run1'
    manifest_path = campaign / 'campaign-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['driver_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Original frozen source changed: ' + name)
    if sha256(campaign / 'dreamer-resolved.json') != manifest['resolved_sha256']:
        raise ValueError('Original resolved configuration changed')
    original_hashes = {name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        for name, fn in [('JobScheduler.check_job_status', JobScheduler.check_job_status),
                         ('headless._build_headless_command', headless._build_headless_command)]}
    added_files = ('dreamer/native_run1_runtime_fix.py', 'scripts/resume_v4_run1.py',
                   'tests/test_native_run1_runtime_fix.py')
    amendment_path = campaign / 'runtime-amendment.json'
    # A competing controller is rejected before any amendment or monkeypatch.
    with campaign_lock(campaign):
        if amendment_path.exists():
            amendment = json.loads(amendment_path.read_text())
            if amendment['campaign_manifest_sha256'] != sha256(manifest_path):
                raise ValueError('Runtime amendment belongs to another campaign')
            for name, expected in amendment['added_source_sha256'].items():
                if sha256(ROOT / name) != expected:
                    raise ValueError('Recorded compatibility source changed: ' + name)
            if original_hashes != amendment['upstream_function_sha256']:
                raise ValueError('Pinned upstream timeout/command function changed')
        else:
            state_path = campaign / 'run1-native-state.json'
            state = json.loads(state_path.read_text())
            if state['boundary'] != 'launcher_finally':
                raise ValueError('Original controller has not completed a normal checkpoint')
            calls = [json.loads(p.read_text()) for p in (campaign / 'model-calls').glob('*.json')]
            if any(c['status'] == 'started' for c in calls):
                raise ValueError('An actual model request remains unresolved')
            with sqlite3.connect(f'file:{campaign / "programs.sqlite"}?mode=ro', uri=True) as db:
                failed = db.execute('select id,correct,combined_score,code from programs where generation=1').fetchall()
                slots = [r[0] for r in db.execute('select distinct generation from programs order by generation')]
            if len(failed) != 1 or failed[0][1] or failed[0][2] != 0:
                raise ValueError('Expected preserved generation-1 infrastructure failure is absent')
            checkpoint = campaign / 'checkpoints/before-runtime-fix'
            checkpoint.mkdir(parents=True, exist_ok=True)
            checkpoint_files = {}
            for name in ('run1-native-state.json', 'run1-meta-state.json', 'call-budget-ledger.json',
                         'native-events.jsonl', 'controller.log'):
                source, target = campaign / name, checkpoint / name
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise ValueError('Original runtime-fix checkpoint already exists with different bytes')
                target.write_bytes(source.read_bytes())
                checkpoint_files[name] = {'path': str(target.relative_to(ROOT)), 'sha256': sha256(target)}
            for name in ('programs.sqlite', 'prompts.sqlite'):
                target = checkpoint / name
                if target.exists():
                    raise ValueError('Unexpected preexisting checkpoint database: ' + str(target))
                with sqlite3.connect(f'file:{campaign / name}?mode=ro', uri=True) as source_db:
                    with sqlite3.connect(target) as target_db:
                        source_db.backup(target_db)
                checkpoint_files[name] = {'path': str(target.relative_to(ROOT)), 'sha256': sha256(target)}
            amendment = {
                'type': 'runtime compatibility bug fixes; no candidate/evaluator/selection change',
                'recorded_utc': datetime.now(timezone.utc).isoformat(),
                'campaign_manifest_sha256': sha256(manifest_path),
                'original_resolved_sha256': manifest['resolved_sha256'],
                'original_driver_sha256': manifest['driver_sha256'],
                'added_source_sha256': {name: sha256(ROOT / name) for name in added_files},
                'upstream_function_sha256': original_hashes,
                'checkpoint_native_state_sha256': sha256(state_path),
                'checkpoint_meta_state_sha256': sha256(campaign / 'run1-meta-state.json'),
                'checkpoint_call_ledger_sha256': sha256(campaign / 'call-budget-ledger.json'),
                'preserved_checkpoint_files': checkpoint_files,
                'persisted_slots_before_resume': slots,
                'generation_1_preserved': {'id': failed[0][0], 'correct': False, 'score': 0,
                    'source_sha256': hashlib.sha256(failed[0][3].encode()).hexdigest(),
                    'accepted_proposal_sha256': sha256(campaign / 'gen_1/accepted-proposal.json'),
                    'failure': 'Native scheduler charged 526.33s proposal/embedding time against480s evaluation deadline, killed evaluator after approximately.03s before episode output.',
                    'policy': 'Keep original failure row, code, reward credit and consumed calls. No automatic retry or reclassification as an empirical algorithm failure.'},
                'fixes': [
                    {'old': 'Local JobScheduler.check_job_status compared now minus job.start_time (proposal start) to480s.',
                     'new': 'Only the scheduler receives dataclasses.replace(job,start_time=job.evaluation_started_at), falling back to evaluation_submitted_at. Original job and pipeline metrics remain unchanged.',
                     'scope': 'The RUN1 runner instance local scheduler only; candidate CPU/reply/memory limits and480s evaluation allowance unchanged.'},
                    {'old': 'Only provider600s outer timeout; SIGKILL of Headless could leave Codex child alive.',
                     'new': 'Append native supported --timeout590 to every RUN1 Headless command. Headless owns a child process group and performs graceful termination/escalation before the outer600s ceiling.',
                     'scope': 'All actual calls remain audited subscription calls. No authentication, approval, sandbox, model, effort or fallback setting changed.'}],
                'manual_infrastructure_feedback': {'text': INFRASTRUCTURE_FEEDBACK,
                    'sha256': hashlib.sha256(INFRASTRUCTURE_FEEDBACK.encode()).hexdigest(),
                    'placement': 'Appended after native mutation/fix operator-specific prompt composition, alongside the non-evolving boundary. Existing DB feedback, scores and saved meta text remain untouched.'},
                'unchanged': ['original manifest/config/source files', 'native installed revision/files',
                    'objective', 'environment', 'case panel', 'seed', 'candidate resource boundaries',
                    'total32slot ceiling', 'all-call token/time/role budgets', 'deadline',
                    'operators', 'parent/inspiration sampling', 'islands/migration', 'bandit',
                    'novelty', 'prompt coevolution', 'meta recommendations', 'saved native state'],
                'resume_limit': 'Previously accepted pending proposals recover natively. Generated but not novelty-accepted held proposals remain preserved and require explicit stage recovery; do not blindly regenerate.'}
            record(amendment_path, amendment)
            record(ROOT / 'artifacts/campaign-v4/run1/runtime-amendment.json', amendment)

    native_run1.Run1Runner = CorrectedRunner
    install_headless_child_timeout()
    started = datetime.now(timezone.utc)
    execution = {'started_utc': started.isoformat(), 'amendment_sha256': sha256(amendment_path),
                 'campaign_manifest_sha256': sha256(manifest_path),
                 'mode': 'run' if '--run' in sys.argv else 'prepare-only'}
    stem = 'runtime-fix-execution-' + started.strftime('%Y%m%dT%H%M%S%fZ')
    record(campaign / (stem + '-start.json'), execution)
    try:
        evolve_v4.main()
    except BaseException as exc:
        record(campaign / (stem + '-end.json'), execution | {
            'ended_utc': datetime.now(timezone.utc).isoformat(), 'status': 'error',
            'error': type(exc).__name__ + ': ' + str(exc)})
        raise
    else:
        record(campaign / (stem + '-end.json'), execution | {
            'ended_utc': datetime.now(timezone.utc).isoformat(), 'status': 'returned_normally'})


if __name__ == '__main__':
    main()

