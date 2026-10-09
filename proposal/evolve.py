"""Prepare or explicitly launch native Shinka for the original proposal only."""
import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/proposal-reconstruction')
    parser.add_argument('--slots', type=int, default=100,
                        help='Total slots including seed; original proposal target100')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--model', action='append', required=True,
                        help='Explicit subscription Headless model; repeat for native UCB')
    parser.add_argument('--run', action='store_true', help='Without this flag, prepare only; zero calls')
    args = parser.parse_args()
    if args.slots < 1 or args.episodes < 1:
        parser.error('slots and episodes must be positive')
    if any(not model.startswith('headless/codex@') for model in args.model):
        parser.error('Only the authorized subscription Codex route is supported')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['HEADLESS_BILLING'] = 'subscription'
    os.environ['SHINKA_HEADLESS_COMMAND'] = str(ROOT/'scripts/subscription_headless.sh')
    os.environ['SHINKA_HEADLESS_TIMEOUT'] = '300'
    os.environ['MPLCONFIGDIR'] = str(ROOT/'.cache/matplotlib')
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    from dreamer.native import CheckpointRunner
    from dreamer.headless_transport import install
    from shinka.llm.providers import headless
    for model in args.model:
        headless.parse_headless_model(model)
    out = Path(args.results).resolve()
    if (out/'programs.sqlite').exists() and not (out/'proposal-config.json').exists():
        raise ValueError('Refusing to append proposal scores to another experiment')
    out.mkdir(parents=True, exist_ok=True)
    evo = EvolutionConfig(task_sys_msg=(ROOT/'proposal/prompt.md').read_text(),
        init_program_path=str(ROOT/'proposal/initial.py'), results_dir=str(out),
        num_generations=args.slots, llm_models=args.model,
        llm_dynamic_selection='ucb' if len(set(args.model)) > 1 else None,
        llm_kwargs={}, meta_llm_models=[args.model[0]], meta_llm_kwargs={}, meta_rec_interval=10,
        novelty_llm_models=None, embedding_model=None, max_novelty_attempts=1,
        use_text_feedback=True, evolve_prompts=False, prompt_llm_models=None,
        patch_types=['diff', 'full', 'cross'], patch_type_probs=[.4, .3, .3],
        max_patch_resamples=1, max_patch_attempts=1, enable_controlled_oversubscription=False)
    db = DatabaseConfig(num_islands=4, migration_interval=10, migration_rate=.1,
        archive_size=40, num_archive_inspirations=1, num_top_k_inspirations=1)
    job = LocalJobConfig(eval_program_path=str(ROOT/'proposal/evaluate.py'),
        python_executable=str(ROOT/'.venv/bin/python'),
        extra_cmd_args={'episodes': args.episodes}, time='00:08:00', numeric_threads_per_job=1)
    source_files = ['proposal/initial.py', 'proposal/prompt.md', 'proposal/evaluation.py',
        'proposal/worker.py', 'proposal/evaluate.py', 'proposal/evolve.py',
        'dreamer/world.py', 'dreamer/evaluation.py', 'dreamer/isolation.py',
        'dreamer/native.py', 'dreamer/headless_transport.py', 'scripts/subscription_headless.sh']
    config = {'identity': 'namazu-proposal-reconstruction-v1',
        'shinka_revision': '9912af12d423504b8d580f4179fd15f5f88b8c50',
        'installed_version': importlib.metadata.version('shinka-evolve'),
        'evolution': asdict(evo), 'database': asdict(db), 'job': asdict(job),
        'concurrency': {'proposals': 1, 'evaluations': 1, 'episode_workers': 1},
        'sources': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files},
        'billing': 'subscription-only; no paid fallback',
        'auxiliary_roles': 'meta uses first model; embeddings, novelty judge and prompt evolution are disabled',
        'timeouts': '290s Headless child /300s provider; evaluation480s from its own launch',
        'unsupported_not_forwarded': ['temperature', 'max_tokens']}
    config_path = out/'proposal-config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration changed; use a separate campaign directory')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    (out/'AGENTS.md').write_text('Return only the requested candidate code or analysis. '
        'Do not use tools or inspect files. The supplied prompt is the complete task.\n')
    if not args.run:
        print(json.dumps({'prepared': str(config_path), 'model_calls': 0, 'candidate_evaluations': 0}))
        return
    install()
    original_command = headless._build_headless_command
    headless._build_headless_command = lambda **kwargs: original_command(**kwargs) + ['--timeout', '290']
    async def run():
        runner = CheckpointRunner(evo_config=evo, db_config=db, job_config=job,
            max_evaluation_jobs=1, max_proposal_jobs=1, max_db_workers=1)
        original_check = runner.scheduler.check_job_status
        def check(job):
            started = job.evaluation_started_at or job.evaluation_submitted_at
            return original_check(replace(job, start_time=started))
        runner.scheduler.check_job_status = check
        loop = asyncio.get_running_loop()
        def pulse():
            loop.call_later(.05, pulse)
        pulse()
        await runner.run_async()
    asyncio.run(run())


if __name__ == '__main__':
    main()
