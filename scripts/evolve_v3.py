"""One subscription-only native v3 controller, 50 total slots including failures."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPSTREAM = '9912af12d423504b8d580f4179fd15f5f88b8c50'
HEADLESS = '93cd9b06b85f848af1308c41e018991b33907c5e'


def tree_hash(directory, suffix):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob('*' + suffix)):
        digest.update(str(path.relative_to(directory)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/campaign-v3')
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--episodes', type=int, default=24, help='Layout cases; each runs all three regimes')
    parser.add_argument('--seed-file', default='results/private/v3-development-seeds.json')
    parser.add_argument('--model', default='headless/codex@gpt-6-astra?effort=high')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 2 <= args.generations <= 50:
        parser.error('This authorized wave contains 50 total slots; stop must be 2..50')
    if not args.model.startswith('headless/codex@'):
        parser.error('Only the subscription-backed Headless Codex route is authorized')
    os.environ['HEADLESS_BILLING'] = 'subscription'
    os.environ['SHINKA_HEADLESS_COMMAND'] = str(ROOT / 'scripts/subscription_headless.sh')
    os.environ['SHINKA_HEADLESS_TIMEOUT'] = '600'
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    for name in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL', 'ANTHROPIC_API_KEY',
                 'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'AWS_ACCESS_KEY_ID',
                 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'):
        os.environ.pop(name, None)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'

    import numpy as np
    import shinka
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    from shinka.llm.providers.headless import parse_headless_model
    from dreamer.native_v3 import V3Runner, install_call_audit
    from dreamer.evaluation_v3 import evaluation_identity
    from dreamer.provenance import control_path, pool, record, sha256
    from scripts.recover_campaign import campaign_lock

    results = Path(args.results).resolve()
    if 'v3' not in results.name:
        raise ValueError('Use a separately versioned v3 campaign; never append to v2')
    results.mkdir(parents=True, exist_ok=True)
    route = parse_headless_model(args.model)
    seed_file = Path(args.seed_file).resolve()
    _, pool_info = pool(seed_file, args.episodes)
    protocol_file = ROOT / 'artifacts/campaign-v3/protocol.json'
    protocol = json.loads(protocol_file.read_text())
    if protocol['evaluation'] != evaluation_identity() or protocol['total_candidate_slots'] != 50:
        raise ValueError('Protocol evaluator identity or authorized total slot budget differs')
    installed = json.loads(importlib.metadata.distribution('shinka-evolve').read_text('direct_url.json'))
    if installed.get('vcs_info', {}).get('commit_id') != UPSTREAM:
        raise ValueError('Installed Shinka revision differs from frozen upstream')
    if subprocess.check_output(['git', '-C', str(ROOT / '.runtime/headless'), 'rev-parse', 'HEAD'], text=True).strip() != HEADLESS:
        raise ValueError('Headless revision differs from frozen subscription route')
    evo = EvolutionConfig(
        task_sys_msg=(ROOT / 'docs/mutation-prompt-v3.md').read_text(),
        num_generations=args.generations, init_program_path=str(control_path('predictive')),
        results_dir=str(results), llm_models=[args.model], llm_dynamic_selection=None,
        llm_kwargs={'temperatures': [1.0], 'max_tokens': 24000},
        meta_rec_interval=10, meta_llm_models=[args.model], meta_llm_kwargs={},
        novelty_llm_models=None, embedding_model=None, max_novelty_attempts=1,
        use_text_feedback=True, evolve_prompts=False, prompt_llm_models=None,
        patch_types=['diff', 'full'], patch_type_probs=[.6, .4],
        max_patch_resamples=2, max_patch_attempts=2,
        enable_controlled_oversubscription=False)
    database = DatabaseConfig(num_islands=4, migration_interval=10, migration_rate=.1,
                              archive_size=40, num_archive_inspirations=1, num_top_k_inspirations=1)
    job = LocalJobConfig(eval_program_path=str(ROOT / 'evaluate_v3.py'),
                         python_executable=str(ROOT / '.venv/bin/python'),
                         extra_cmd_args={'episodes': args.episodes, 'seed_file': str(seed_file),
                                         'campaign_manifest': str(results / 'campaign-manifest.json')},
                         time='00:15:00', numeric_threads_per_job=1)
    resolved = {'evolution': asdict(evo), 'database': asdict(database), 'job': asdict(job),
                'max_evaluation_jobs': 1, 'max_proposal_jobs': 1, 'max_db_workers': 1,
                'billing': 'subscription', 'upstream': UPSTREAM, 'headless': HEADLESS,
                'native_sampling_seed': 3181473,
                'native_sampling_resume': 'Python/NumPy checkpoint restored before setup/recovery; saved proposals and native recommendations restored; LLM sampling not reproducible',
                'roles': {'mutation_fix': args.model, 'meta_recommendation': args.model,
                          'novelty': None, 'embedding': None, 'evaluator': 'local Python; no LLM'},
                'effective_codex': {'model': route.agent_model, 'effort': route.effort,
                                    'timeout_seconds': 600,
                                    'not_forwarded': ['temperature', 'max_tokens']}}
    resolved['evolution']['num_generations'] = 50
    with campaign_lock(results):
        record(results / 'dreamer-resolved.json', resolved)
        manifest = {'campaign': 'namazu-unknown-dynamics-v3', 'target_generation_slots': 50,
                    'evaluation': evaluation_identity(), 'development_pool': pool_info,
                    'regimes': ['uniform', 'stationary', 'switch'],
                    'candidate_episodes': args.episodes * 3,
                    'seed_program_sha256': sha256(control_path('predictive')),
                    'seed_source': 'immutable original predictive seed; no manually fitted comparator incorporated',
                    'protocol_sha256': sha256(protocol_file),
                    'resolved_sha256': sha256(results / 'dreamer-resolved.json'),
                    'driver_sha256': {p: sha256(ROOT / p) for p in ('scripts/evolve_v3.py', 'dreamer/native_v3.py',
                        'scripts/recover_campaign.py', 'dreamer/native.py', 'scripts/subscription_headless.sh', 'docs/mutation-prompt-v3.md')},
                    'installed': {'shinka_revision': UPSTREAM, 'shinka_version': importlib.metadata.version('shinka-evolve'),
                                  'shinka_python_sha256': tree_hash(Path(shinka.__file__).parent, '.py'),
                                  'headless_revision': HEADLESS,
                                  'headless_js_sha256': tree_hash(ROOT / '.runtime/headless/dist', '.js'),
                                  'codex': subprocess.check_output(['codex', '--version'], text=True).strip()}}
        record(results / 'campaign-manifest.json', manifest)
        if args.prepare_only:
            print(json.dumps({'manifest': str(results / 'campaign-manifest.json'), 'candidate_episodes': args.episodes * 3}))
            return
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record(results / f'execution-{stamp}.json', {'generation_stop': args.generations,
               'includes_seed_and_failures': True, 'campaign_sha256': sha256(results / 'campaign-manifest.json')})
        command = [str(ROOT / 'scripts/subscription_headless.sh'), 'codex', '--allow', 'read-only',
                   '--model', route.agent_model, '--reasoning-effort', route.effort or 'high',
                   '--timeout', '60', '--usage', '--prompt', 'Reply only READY. Do not call tools.']
        probe = subprocess.run(command, capture_output=True, text=True, timeout=75)
        log = results / f'subscription-probe-{stamp}.log'
        log.write_text(probe.stdout + probe.stderr)
        if probe.returncode:
            raise RuntimeError('Subscription startup failed; no native retries. See ' + str(log))
        random.seed(3181473)
        np.random.seed(3181473)
        install_call_audit(results)
        async def run():
            loop = asyncio.get_running_loop()
            def pulse():
                loop.call_later(.05, pulse)
            pulse()
            runner = V3Runner(evo_config=evo, db_config=database, job_config=job,
                              max_evaluation_jobs=1, max_proposal_jobs=1, max_db_workers=1)
            try:
                await runner.run_async()
            finally:
                runner._save_rng()
        asyncio.run(run())


if __name__ == '__main__':
    main()
