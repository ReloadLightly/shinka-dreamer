"""Resume the frozen v3 campaign with the recorded prompt-description correction."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/campaign-v3')
    parser.add_argument('--generations', type=int, default=50)
    args = parser.parse_args()
    if args.generations != 50:
        parser.error('The authorized v3 unit closes at 50 total candidate slots')
    results = Path(args.results).resolve()
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
    import importlib.metadata
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    from shinka.llm.providers.headless import parse_headless_model
    from dreamer.native_v3 import V3Runner, install_call_audit
    from dreamer.evaluation_v3 import evaluation_identity
    from dreamer.provenance import record, sha256, pool
    from scripts.recover_campaign import campaign_lock
    from scripts.evolve_v3 import tree_hash

    with campaign_lock(results):
        manifest = json.loads((results / 'campaign-manifest.json').read_text())
        resolved = json.loads((results / 'dreamer-resolved.json').read_text())
        if manifest['evaluation'] != evaluation_identity():
            raise ValueError('Frozen v3 evaluator changed')
        if manifest['resolved_sha256'] != sha256(results / 'dreamer-resolved.json'):
            raise ValueError('Original native settings changed')
        for name, expected in manifest['driver_sha256'].items():
            if sha256(ROOT / name) != expected:
                raise ValueError('Frozen original driver changed: ' + name)
        if sha256(ROOT / 'artifacts/campaign-v3/protocol.json') != manifest['protocol_sha256']:
            raise ValueError('Frozen protocol changed')
        _, pool_info = pool(manifest['development_pool']['path'])
        if pool_info != manifest['development_pool']:
            raise ValueError('Development pool changed')
        installed = manifest['installed']
        if tree_hash(Path(shinka.__file__).parent, '.py') != installed['shinka_python_sha256']:
            raise ValueError('Installed upstream source changed')
        if tree_hash(ROOT / '.runtime/headless/dist', '.js') != installed['headless_js_sha256']:
            raise ValueError('Installed subscription backend changed')
        if subprocess.check_output(['codex', '--version'], text=True).strip() != installed['codex']:
            raise ValueError('Codex runtime changed')
        old_prompt = resolved['evolution']['task_sys_msg']
        old_phrase = 'where keys is the fraction of the two collected keys.'
        new_phrase = 'where keys is the raw number of collected keys (0, 1, or 2).'
        if old_prompt.count(old_phrase) != 1:
            raise ValueError('Prompt amendment must have one exact expected old phrase')
        new_prompt = old_prompt.replace(old_phrase, new_phrase)
        prompt_path = ROOT / 'docs/mutation-prompt-v3-corrected.md'
        if prompt_path.read_text() != new_prompt:
            raise ValueError('Corrected prompt differs beyond the approved descriptive phrase')
        instruction_path = ROOT / 'docs/v3-mutation-worker-instructions.md'
        if (results / 'AGENTS.md').read_text() != instruction_path.read_text():
            raise ValueError('Campaign-local mutation instructions changed')
        amendment = {'campaign_sha256': sha256(results / 'campaign-manifest.json'),
                     'reason': 'Correct a mistaken description of the existing keys score term; evaluator always used raw count.',
                     'old_text': old_phrase, 'new_text': new_phrase,
                     'old_prompt_sha256': sha256(ROOT / 'docs/mutation-prompt-v3.md'),
                     'new_prompt_sha256': sha256(prompt_path),
                     'evaluation_unchanged': True, 'search_mechanisms_unchanged': True,
                     'first_controller_stop': json.loads((results / 'prompt-correction-stop.json').read_text()),
                     'new_resume_driver_sha256': sha256(__file__),
                     'local_worker_instructions_sha256': sha256(instruction_path),
                     'local_worker_instructions': instruction_path.read_text(),
                     'instruction_reason': 'Prevent inherited repository implementation workflow from triggering outside-context reads; no global authentication/settings/sandbox changed.',
                     'observed_first_request_context': 'Native tool telemetry shows pwd plus CODEX_TASK.md/docs/design.md reads; no v3 private validation/fitting/assessment data read.',
                     'preservation': 'Original manifest/resolved/prompt/driver and every call/proposal remain intact.'}
        record(results / 'prompt-amendment.json', amendment)
        record(ROOT / 'artifacts/campaign-v3/prompt-amendment.json', amendment)
        evodata = dict(resolved['evolution'])
        evodata['task_sys_msg'] = new_prompt
        evodata['num_generations'] = args.generations
        evo = EvolutionConfig(**evodata)
        database = DatabaseConfig(**resolved['database'])
        job = LocalJobConfig(**resolved['job'])
        record(results / 'amended-resolved.json', {**resolved, 'evolution': asdict(evo),
               'amendment_sha256': sha256(results / 'prompt-amendment.json')})
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record(results / f'resume-{stamp}.json', {'generation_stop': args.generations,
               'campaign_sha256': sha256(results / 'campaign-manifest.json'),
               'amendment_sha256': sha256(results / 'prompt-amendment.json'),
               'resume_driver_sha256': sha256(__file__)})
        route = parse_headless_model(evo.llm_models[0])
        probe = subprocess.run([str(ROOT / 'scripts/subscription_headless.sh'), 'codex',
                   '--allow', 'read-only', '--model', route.agent_model,
                   '--reasoning-effort', route.effort, '--timeout', '60', '--usage',
                   '--prompt', 'Reply only READY. Do not call tools.'], capture_output=True, text=True, timeout=75)
        log = results / f'subscription-probe-{stamp}.log'
        log.write_text(probe.stdout + probe.stderr)
        if probe.returncode:
            raise RuntimeError('Subscription probe failed; no mutation retries: ' + str(log))
        random.seed(resolved['native_sampling_seed'])
        np.random.seed(resolved['native_sampling_seed'])
        install_call_audit(results)
        async def run():
            loop = asyncio.get_running_loop()
            def pulse():
                loop.call_later(.05, pulse)
            pulse()
            runner = V3Runner(evo_config=evo, db_config=database, job_config=job,
                              max_evaluation_jobs=1, max_proposal_jobs=1, max_db_workers=1)
            def request_checkpoint():
                stop = max(runner.next_generation_to_submit, runner.completed_generations, 1)
                runner.evo_config.num_generations = min(stop, args.generations)
                record(results / f'checkpoint-request-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}.json', {
                    'authorized_total_slots': args.generations, 'invocation_stop_after_pending_slots': stop,
                    'reason': 'SIGUSR1 requested a normal native checkpoint; resume retains total budget'})
                runner.slot_available.set()
            loop.add_signal_handler(signal.SIGUSR1, request_checkpoint)
            try:
                await runner.run_async()
            finally:
                runner._save_rng()
                loop.remove_signal_handler(signal.SIGUSR1)
        asyncio.run(run())


if __name__ == '__main__':
    main()
