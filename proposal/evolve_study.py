"""Prepare, then explicitly execute one fresh independent native search.

Preparation makes no model calls, embeddings or world episodes. All budget flags
are explicit; --run accepts only the same frozen configuration. Executions are
one-shot: a stopped repetition cannot silently acquire another allowance.
"""
import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from proposal.evolve_full import MODELS, UPSTREAM

HEADLESS_REVISION = '93cd9b06b85f848af1308c41e018991b33907c5e'
REQUEST_TIMEOUT_SECONDS = 500
CHILD_TIMEOUT_SECONDS = 480
CHECKPOINT_RESERVE_SECONDS = 30
FILES = [
    'proposal/initial.py', 'proposal/prompt.md', 'proposal/worker.py',
    'proposal/evaluation.py', 'proposal/evaluate.py', 'proposal/evaluate_full.py',
    'proposal/native_full.py', 'proposal/evolve_full.py', 'proposal/study_native.py',
    'proposal/evolve_study.py', 'dreamer/world.py', 'dreamer/evaluation.py',
    'dreamer/isolation.py', 'dreamer/native.py', 'dreamer/native_run1.py',
    'dreamer/native_run1_runtime_fix.py', 'dreamer/native_v3.py',
    'dreamer/headless_transport.py', 'dreamer/provenance.py',
    'scripts/subscription_headless.sh', 'scripts/run1_embedding_server.py',
    'scripts/recover_campaign.py', 'scripts/study_codex_bin/codex',
]


def utcnow():
    return datetime.now(timezone.utc)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_environment():
    shim = ROOT/'scripts/study_codex_bin'
    if not os.environ.get('SHINKA_STUDY_CODEX_BIN'):
        real = shutil.which('codex')
        if not real or Path(real).resolve() == (shim/'codex').resolve():
            raise ValueError('Cannot resolve the original Codex executable before installing the task shim')
        os.environ['SHINKA_STUDY_CODEX_BIN'] = str(Path(real).absolute())
    if str(shim) not in os.environ.get('PATH', '').split(os.pathsep):
        os.environ['PATH'] = str(shim)+os.pathsep+os.environ.get('PATH', '')
    os.environ.update(HEADLESS_BILLING='subscription', HEADLESS_CAPACITY_RETRIES='0',
        SHINKA_LLM_MAX_RETRIES='1', SHINKA_HEADLESS_TIMEOUT=str(REQUEST_TIMEOUT_SECONDS),
        SHINKA_HEADLESS_COMMAND=str(ROOT/'scripts/subscription_headless.sh'),
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
    for name in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL',
                 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY',
                 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
                 'AWS_SESSION_TOKEN', 'OPENROUTER_API_KEY'):
        os.environ.pop(name, None)


def check_global_config():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    location = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))/'config.toml'
    content = location.read_bytes() if location.exists() else b''
    parsed = tomllib.loads(content.decode())
    if any(parsed.get(key) for key in ('mcp_servers', 'developer_instructions', 'model_instructions_file',
                                      'experimental_instructions_file', 'instructions')):
        raise ValueError('Study requires no global MCP servers or custom instruction fields; configuration was not changed')
    return hashlib.sha256(content).hexdigest()


def check_runtime():
    check_global_config()
    direct = json.loads(importlib.metadata.distribution('shinka-evolve').read_text('direct_url.json'))
    if direct.get('vcs_info', {}).get('commit_id') != UPSTREAM:
        raise ValueError('Installed Shinka upstream revision differs from the study')
    revision = subprocess.check_output(['git', '-C', str(ROOT/'.runtime/headless'),
                                        'rev-parse', 'HEAD'], text=True).strip()
    if revision != HEADLESS_REVISION:
        raise ValueError('Installed Headless revision differs from the study')


def build_configuration(args):
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    results = Path(args.results).resolve()
    if not results.is_relative_to(ROOT/'results') or results == ROOT/'results':
        raise ValueError('Use a new results subdirectory inside this repository')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', args.run_id):
        raise ValueError('run-id must be a short alphanumeric identifier')
    if (args.slots < 2 or args.calls < 1 or not math.isfinite(args.provider_seconds)
            or not math.isfinite(args.minutes) or args.provider_seconds <= REQUEST_TIMEOUT_SECONDS
            or args.minutes*60 <= REQUEST_TIMEOUT_SECONDS+CHECKPOINT_RESERVE_SECONDS):
        raise ValueError('Require slots>=2, calls>=1, provider-seconds>500 and enough wall time for one request plus checkpoint reserve')
    if not 0 <= args.native_seed < 2**32 - 1:
        raise ValueError('native-seed must fit the NumPy seed range with room for bandit seed+1')
    model_work_dir = str(Path('/tmp')/f'shinka-original-study-{args.run_id}')
    role_kwargs = {'headless_work_dir': model_work_dir}
    evo = EvolutionConfig(task_sys_msg=(ROOT/'proposal/prompt.md').read_text(),
        init_program_path=str(ROOT/'proposal/initial.py'), results_dir=str(results),
        num_generations=args.slots, llm_models=MODELS, llm_dynamic_selection='ucb',
        llm_dynamic_selection_kwargs={'cost_aware_coef': 0., 'seed': args.native_seed+1},
        llm_kwargs=dict(role_kwargs), meta_rec_interval=4, meta_llm_models=[MODELS[1]], meta_llm_kwargs=dict(role_kwargs),
        embedding_model='local/bge-small-code-chunks-v1@http://127.0.0.1:8771/v1',
        code_embed_sim_threshold=.95, max_novelty_attempts=2,
        novelty_llm_models=[MODELS[1]], novelty_llm_kwargs=dict(role_kwargs), use_text_feedback=True,
        evolve_prompts=True, prompt_llm_models=[MODELS[1]], prompt_llm_kwargs=dict(role_kwargs),
        prompt_evolution_interval=4, prompt_archive_size=10,
        prompt_percentile_recompute_interval=4, prompt_patch_types=['diff', 'full'],
        prompt_patch_type_probs=[.7, .3], patch_types=['diff', 'full', 'cross'],
        patch_type_probs=[.4, .3, .3], max_patch_resamples=1, max_patch_attempts=2,
        enable_controlled_oversubscription=False)
    db = DatabaseConfig(num_islands=4, migration_interval=4, migration_rate=.25,
        archive_size=40, num_archive_inspirations=1, num_top_k_inspirations=1,
        parent_selection_strategy='weighted', parent_selection_lambda=10.,
        island_selection_strategy='uniform', island_elitism=True,
        enforce_island_separation=True, archive_selection_strategy='fitness')
    job = LocalJobConfig(eval_program_path=str(ROOT/'proposal/evaluate_full.py'),
        python_executable=str(ROOT/'.venv/bin/python'), time='00:02:00',
        numeric_threads_per_job=1,
        extra_cmd_args={'episodes': 5, 'seed-start': 0,
            'campaign-manifest': str(results/'campaign-manifest.json'), 'record-first': 'yes'})
    limits = {'calls': args.calls, 'remote_elapsed_seconds': args.provider_seconds,
        'request_timeout_seconds': REQUEST_TIMEOUT_SECONDS,
        'role_caps': {role: args.calls for role in ('mutation_repair', 'novelty',
            'summary', 'global_insight', 'recommendation', 'prompt_mutation')}}
    limits['role_caps']['readiness'] = 0
    treatment = getattr(args, 'treatment', 'full')
    if treatment not in ('full', 'rewrite'):
        raise ValueError('Unknown study treatment')
    if treatment == 'rewrite':
        evo.llm_dynamic_selection = None
        evo.llm_dynamic_selection_kwargs = {}
        evo.meta_rec_interval = None
        evo.meta_llm_models = None
        evo.evolve_prompts = False
        evo.prompt_llm_models = None
        evo.embedding_model = None
        evo.novelty_llm_models = None
        evo.max_novelty_attempts = 1
        evo.max_patch_attempts = 1
        evo.patch_types = ['full']
        evo.patch_type_probs = [1.]
        db.num_islands = 1
        db.migration_rate = 0.
        db.parent_selection_strategy = 'best_of_n'
        db.num_archive_inspirations = 0
        db.num_top_k_inspirations = 0
        for role in limits['role_caps']:
            if role != 'mutation_repair':
                limits['role_caps'][role] = 0
    config = {'schema': 1, 'treatment': f'original-proposal-independent-{treatment}-v1',
        'arm': treatment, 'allowed_roles': ['mutation'] if treatment == 'rewrite' else None,
        'mutation_context': {'work_dir': model_work_dir, 'project_doc_max_bytes': 0,
            'global_config_sha256': check_global_config(), 'tool_event_policy': 'Reject and terminate on an observed tool event.',
            'configuration_reference': 'https://learn.chatgpt.com/docs/config-file/config-reference',
            'limitation': 'Restrictive per-invocation flags and raw event auditing, not an adversarial security guarantee.'},
        'rewrite_model_offset': args.native_seed % len(MODELS),
        'run_id': args.run_id, 'evaluator': 'namazu-proposal-reconstruction-v1',
        'evolution': asdict(evo), 'database': asdict(db), 'job': asdict(job),
        'limits': limits, 'wall_minutes': args.minutes, 'checkpoint_reserve_seconds': CHECKPOINT_RESERVE_SECONDS,
        'native_sampling_seed': args.native_seed, 'bandit_seed': args.native_seed+1,
        'candidate_rng_seed': 712934, 'search_cases': [0, 1, 2, 3, 4],
        'rng_limitation': 'Native SQLite RANDOM() and remote sampling are not seed-controlled.',
        'shinka_revision': UPSTREAM, 'headless_revision': HEADLESS_REVISION,
        'headless_capacity_retries': 0, 'provider_retries': 1,
        'child_timeout_seconds': CHILD_TIMEOUT_SECONDS, 'billing': 'subscription only; no paid fallback',
        'concurrency': {'proposals': 1, 'evaluations': 1, 'database_workers': 1, 'numeric_threads': 1},
        'episode_cap': args.slots*5, 'transition_cap': args.slots*5*200,
        'candidate_cpu_seconds_cap': args.slots*5*10,
        'combined_rss_bytes_cap': 2*1024**3,
        'source_sha256': {name: sha(ROOT/name) for name in FILES},
        'missing_usage_policy': 'Unknown tokens remain unknown; call/time/deadline budgets are authoritative.',
        'unsupported_not_forwarded': ['temperature', 'max_tokens'],
        'resume_policy': 'No automatic or blind resumption; every directory admits one execution only.'}
    return config


def freeze_configuration(config):
    results = Path(config['evolution']['results_dir'])
    marker = results/'campaign-manifest.json'
    if results.exists() and not marker.exists() and any(results.iterdir()):
        raise ValueError('Refusing an existing nonempty campaign directory')
    results.mkdir(parents=True, exist_ok=True)
    for name in ('campaign-manifest.json', 'resolved-config.json'):
        path = results/name
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError('Frozen study configuration changed; use a new independent directory')
        else:
            with path.open('x') as handle:
                handle.write(json.dumps(config, indent=2)+'\n')
    (results/'AGENTS.md').write_text('Return only the requested code or analysis. Do not use tools or inspect files. The supplied prompt contains the complete task.\n')
    context = Path(config['mutation_context']['work_dir'])
    if context.is_symlink():
        raise ValueError('Model context directory must not be a symlink')
    if not context.exists():
        context.mkdir(mode=0o700)
    marker = context/'study-context.json'
    identity = {'run_id': config['run_id'], 'results': str(results)}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('External model context belongs to another study')
    elif any(context.iterdir()):
        raise ValueError('External model context is not empty')
    else:
        marker.write_text(json.dumps(identity)+'\n')


def execute_worker(results):
    """Only the supervising process invokes this after creating the one-shot grant."""
    import numpy as np
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    from shinka.llm.providers import headless
    from dreamer.headless_transport import install
    from dreamer.native_run1 import install_run1_call_audit
    from dreamer.native_v3 import atomic_json
    from proposal.native_full import install_lock_wait_audit
    from proposal.study_native import StudyCallBudget, StudyRunner, IndependentRewriteRunner
    from proposal.evaluate_full import validate_sources
    config = validate_sources(results/'campaign-manifest.json')
    if check_global_config() != config['mutation_context']['global_config_sha256']:
        raise ValueError('Global Codex configuration changed since preparation; no settings were changed')
    os.environ['SHINKA_STUDY_TOOL_AUDIT_DIR'] = str(results/'codex-event-audit')
    grant = json.loads((results/'execution-grant.json').read_text())
    if grant['manifest_sha256'] != sha(results/'campaign-manifest.json'):
        raise ValueError('Execution grant differs from the frozen configuration')
    if grant['supervisor_pid'] != os.getppid():
        raise ValueError('Only the live supervising process may consume this grant')
    with (results/'worker-started.json').open('x') as handle:
        handle.write(json.dumps({'pid': os.getpid(), 'started_utc': utcnow().isoformat()})+'\n')
    budget = StudyCallBudget(results, limits=config['limits'], deadline_utc=grant['admission_end_utc'],
        allowed_roles=config['allowed_roles'])
    if budget.reason('mutation'):
        raise RuntimeError(budget.reason('mutation'))
    install()
    original_command = headless._build_headless_command
    headless._build_headless_command = lambda **kwargs: original_command(**kwargs)+['--timeout', str(config['child_timeout_seconds'])]
    install_run1_call_audit(budget)
    install_lock_wait_audit()
    random.seed(config['native_sampling_seed'])
    np.random.seed(config['native_sampling_seed'])
    with (results/'embedding-service.log').open('a') as output:
        server = None
        try:
            if config['arm'] == 'full':
                server = subprocess.Popen([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run1_embedding_server.py'),
                    '--results', str(results), '--port', '8771'], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                for _ in range(80):
                    if server.poll() is not None:
                        raise RuntimeError('Local embedding service exited; inspect its log')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8771/health', timeout=.5) as response:
                            if json.load(response).get('model') == 'bge-small-code-chunks-v1':
                                break
                    except OSError:
                        time.sleep(.25)
                else:
                    raise RuntimeError('Local embedding service did not become ready')

            async def run():
                runner_type = StudyRunner if config['arm'] == 'full' else IndependentRewriteRunner
                extra = {} if config['arm'] == 'full' else {
                    'rewrite_seed_sha256': config['source_sha256']['proposal/initial.py'],
                    'model_offset': config['rewrite_model_offset']}
                runner = runner_type(evo_config=EvolutionConfig(**config['evolution']),
                    db_config=DatabaseConfig(**config['database']), job_config=LocalJobConfig(**config['job']),
                    max_evaluation_jobs=1, max_proposal_jobs=1, max_db_workers=1, call_budget=budget, **extra)
                loop = asyncio.get_running_loop()
                def pulse():
                    loop.call_later(.05, pulse)
                pulse()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, runner.request_checkpoint, f'signal:{sig.name}')
                async def monitor():
                    while True:
                        await asyncio.sleep(1)
                        reason = budget.reason()
                        if reason:
                            runner.request_checkpoint(reason)
                watcher = asyncio.create_task(monitor())
                try:
                    await runner.run_async()
                finally:
                    watcher.cancel()
                    try:
                        await watcher
                    except asyncio.CancelledError:
                        pass
                    runner._save_state('independent_study_finally')
                    budget.write_ledger()
            asyncio.run(run())
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            atomic_json(results/'worker-ended.json', {'ended_utc': utcnow().isoformat(), 'all_role_calls': budget.totals()})


@contextmanager
def runtime_lock():
    """The native adapter and local encoder share one fixed loopback port."""
    import fcntl
    with (ROOT/'results/proposal-study-runtime.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another independent-study controller is active') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def supervise(config):
    import psutil
    from dreamer.native_v3 import atomic_json
    from scripts.recover_campaign import campaign_lock
    results = Path(config['evolution']['results_dir'])
    with runtime_lock(), campaign_lock(results):
        if (results/'execution-grant.json').exists():
            raise ValueError('This repetition already consumed its one-shot execution grant; inspect its checkpoint')
        now = utcnow()
        hard_end = now+timedelta(minutes=config['wall_minutes'])
        grant = {'started_utc': now.isoformat(), 'hard_end_utc': hard_end.isoformat(),
            'admission_end_utc': (hard_end-timedelta(seconds=config['checkpoint_reserve_seconds'])).isoformat(),
            'manifest_sha256': sha(results/'campaign-manifest.json'), 'automatic_continuation': False,
            'supervisor_pid': os.getpid()}
        with (results/'execution-grant.json').open('x') as handle:
            handle.write(json.dumps(grant, indent=2)+'\n')
        stop_reason = None
        descendants = {}
        peak_rss = 0
        started = time.monotonic()
        previous_handlers = {}
        def stop(signum, _frame):
            nonlocal stop_reason
            stop_reason = stop_reason or f'supervisor_signal:{signal.Signals(signum).name}'
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, stop)
        process = None
        try:
            with (results/'controller.log').open('a') as output:
                process = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-m', 'proposal.evolve_study',
                    '--_worker', str(results)], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True)
                root_process = psutil.Process(process.pid)
                while process.poll() is None:
                    try:
                        family = [root_process, *root_process.children(recursive=True)]
                    except psutil.NoSuchProcess:
                        break
                    rss = 0
                    for member in family:
                        try:
                            descendants[(member.pid, member.create_time())] = member
                            rss += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak_rss = max(peak_rss, rss)
                    if rss > config['combined_rss_bytes_cap']:
                        stop_reason = stop_reason or 'combined_rss_ceiling'
                    if utcnow() >= hard_end:
                        stop_reason = stop_reason or 'hard_execution_deadline'
                    if stop_reason:
                        break
                    time.sleep(.25)
        except BaseException as exc:
            stop_reason = stop_reason or f'supervisor_error:{type(exc).__name__}:{exc}'
            raise
        finally:
            # Capture descendants before reparenting. Headless can create its own
            # process groups, so killing just the controller group is insufficient.
            if process is not None:
                try:
                    for member in psutil.Process(process.pid).children(recursive=True):
                        descendants[(member.pid, member.create_time())] = member
                except psutil.NoSuchProcess:
                    pass
                active = [p for p in descendants.values() if p.is_running()]
                for member in active:
                    try:
                        member.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(active, timeout=5)
                for member in alive:
                    try:
                        member.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=5)
                process.wait(timeout=5)
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)
            atomic_json(results/'execution-resources.json', {
                **grant, 'ended_utc': utcnow().isoformat(),
                'supervisor_wall_seconds': time.monotonic()-started,
                'combined_rss_bytes': peak_rss, 'stop_reason': stop_reason,
                'returncode': process.returncode if process else None,
                'status': 'worker_returned' if process and process.returncode == 0 and not stop_reason else 'interrupted_or_failed',
                'automatic_continuation': False,
                'supervising_assistant_usage': 'Additional; not measurable by this experiment ledger.'})
        return process.returncode if not stop_reason else 1


def main():
    configure_environment()
    if len(sys.argv) == 3 and sys.argv[1] == '--_worker':
        check_runtime()
        execute_worker(Path(sys.argv[2]).resolve())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--treatment', choices=['full', 'rewrite'], default='full')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--native-seed', type=int, required=True)
    parser.add_argument('--slots', type=int, required=True)
    parser.add_argument('--calls', type=int, required=True)
    parser.add_argument('--provider-seconds', type=float, required=True)
    parser.add_argument('--minutes', type=float, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    check_runtime()
    config = build_configuration(args)
    freeze_configuration(config)
    if args.run:
        raise SystemExit(supervise(config))
    print(json.dumps({'prepared': True, 'model_calls': 0, 'episodes': 0,
        'results': config['evolution']['results_dir'], 'run_id': args.run_id,
        'total_slots': args.slots, 'limits': config['limits']}))


if __name__ == '__main__':
    main()
