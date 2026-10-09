"""Bounded Stage 1, original objective, complete native Shinka mechanisms."""
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
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RESULTS=ROOT/'results/proposal-full-native-01'
UPSTREAM='9912af12d423504b8d580f4179fd15f5f88b8c50'
START='2026-10-09T02:45:07+00:00'
EXECUTION_END='2026-10-09T03:14:07+00:00'
HARD_END='2026-10-09T03:25:07+00:00'
MODELS=['headless/codex@gpt-6-astra?effort=high','headless/codex@gpt-6.1-sol?effort=high']
LIMITS={'calls':30,'remote_elapsed_seconds':1200,'request_timeout_seconds':250,
        'role_caps':{'mutation_repair':30,'novelty':30,'summary':30,
                     'global_insight':30,'recommendation':30,'prompt_mutation':30,'readiness':0}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_frozen(path,value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Frozen configuration changed: {path}')
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Explicitly execute within the saved Stage 1 grant')
    args=parser.parse_args()
    os.environ.update(HEADLESS_BILLING='subscription',SHINKA_LLM_MAX_RETRIES='1',
        SHINKA_HEADLESS_COMMAND=str(ROOT/'scripts/subscription_headless.sh'),
        SHINKA_HEADLESS_TIMEOUT='250',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
    for name in ('OPENAI_API_KEY','CODEX_API_KEY','OPENAI_BASE_URL','ANTHROPIC_API_KEY',
                 'GEMINI_API_KEY','GOOGLE_API_KEY','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY',
                 'AWS_SESSION_TOKEN','OPENROUTER_API_KEY'):
        os.environ.pop(name,None)
    import numpy as np
    import psutil
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    from shinka.llm.providers import headless
    from dreamer.headless_transport import install
    from dreamer.native_run1 import install_run1_call_audit
    from dreamer.native_v3 import atomic_json
    from proposal.native_full import FullCallBudget,FullRunner,install_lock_wait_audit
    from scripts.recover_campaign import campaign_lock
    installed=json.loads(importlib.metadata.distribution('shinka-evolve').read_text('direct_url.json'))
    if installed.get('vcs_info',{}).get('commit_id') != UPSTREAM:
        raise ValueError('Installed upstream revision differs from freeze')
    headless_revision=subprocess.check_output(['git','-C',str(ROOT/'.runtime/headless'),'rev-parse','HEAD'],text=True).strip()
    if headless_revision != '93cd9b06b85f848af1308c41e018991b33907c5e':
        raise ValueError('Headless revision changed')
    RESULTS.mkdir(parents=True,exist_ok=True)
    with campaign_lock(RESULTS):
        evo=EvolutionConfig(task_sys_msg=(ROOT/'proposal/prompt.md').read_text(),
            init_program_path=str(ROOT/'proposal/initial.py'),results_dir=str(RESULTS),num_generations=6,
            llm_models=MODELS,llm_dynamic_selection='ucb',llm_dynamic_selection_kwargs={'cost_aware_coef':0.,'seed':619047},llm_kwargs={},
            meta_rec_interval=4,meta_llm_models=[MODELS[1]],meta_llm_kwargs={},
            embedding_model='local/bge-small-code-chunks-v1@http://127.0.0.1:8771/v1',
            code_embed_sim_threshold=.95,max_novelty_attempts=2,novelty_llm_models=[MODELS[1]],novelty_llm_kwargs={},
            use_text_feedback=True,evolve_prompts=True,prompt_llm_models=[MODELS[1]],prompt_llm_kwargs={},
            prompt_evolution_interval=4,prompt_archive_size=10,prompt_percentile_recompute_interval=4,
            prompt_patch_types=['diff','full'],prompt_patch_type_probs=[.7,.3],
            patch_types=['diff','full','cross'],patch_type_probs=[.4,.3,.3],
            max_patch_resamples=1,max_patch_attempts=2,enable_controlled_oversubscription=False)
        db=DatabaseConfig(num_islands=4,migration_interval=4,migration_rate=.25,archive_size=40,
            num_archive_inspirations=1,num_top_k_inspirations=1,parent_selection_strategy='weighted',
            parent_selection_lambda=10.,island_selection_strategy='uniform',island_elitism=True,
            enforce_island_separation=True,archive_selection_strategy='fitness')
        job=LocalJobConfig(eval_program_path=str(ROOT/'proposal/evaluate_full.py'),
            python_executable=str(ROOT/'.venv/bin/python'),time='00:02:00',numeric_threads_per_job=1,
            extra_cmd_args={'episodes':5,'campaign-manifest':str(RESULTS/'campaign-manifest.json'),'record-first':'yes'})
        files=['proposal/initial.py','proposal/prompt.md','proposal/worker.py','proposal/evaluation.py',
               'proposal/evaluate.py','proposal/evaluate_full.py','proposal/native_full.py','proposal/evolve_full.py',
               'dreamer/world.py','dreamer/evaluation.py','dreamer/isolation.py','dreamer/native.py',
               'dreamer/native_run1.py','dreamer/native_run1_runtime_fix.py','dreamer/native_v3.py',
               'dreamer/headless_transport.py','scripts/subscription_headless.sh','scripts/run1_embedding_server.py']
        resolved={'treatment':'original-proposal-full-native-01','evaluator':'namazu-proposal-reconstruction-v1',
            'evolution':asdict(evo),'database':asdict(db),'job':asdict(job),'limits':LIMITS,
            'stage_start_utc':START,'execution_end_utc':EXECUTION_END,'hard_checkpoint_utc':HARD_END,
            'native_sampling_seed':619046,'shinka_revision':UPSTREAM,'headless_revision':headless_revision,
            'concurrency':{'proposals':1,'evaluations':1,'database_workers':1,'numeric_threads':1},
            'episode_cap':30,'transition_cap':6000,'candidate_cpu_seconds_cap':300,
            'combined_rss_bytes_cap':2*1024**3,'candidate_rss_limit_bytes':192*1024**2,
            'billing':'subscription only; all model roles; no paid fallback',
            'missing_usage_policy':'unknown tokens; call/time budgets remain authoritative',
            'unsupported_not_forwarded':['temperature','max_tokens'],
            'source_sha256':{p:sha(ROOT/p) for p in files}}
        write_frozen(RESULTS/'campaign-manifest.json',resolved)
        write_frozen(RESULTS/'resolved-config.json',resolved)
        instructions=RESULTS/'AGENTS.md'
        instructions.write_text('Return only the requested code or analysis. Do not use tools or inspect files. The supplied prompt contains the complete task.\n')
        if not args.run:
            print(json.dumps({'prepared':True,'model_calls':0,'episode_cap':30,'total_slots':6}))
            return
        budget=FullCallBudget(RESULTS,limits=LIMITS,deadline_utc=EXECUTION_END)
        if budget.reason('mutation'):
            raise RuntimeError(budget.reason('mutation'))
        install()
        original_command=headless._build_headless_command
        headless._build_headless_command=lambda **kwargs: original_command(**kwargs)+['--timeout','240']
        install_run1_call_audit(budget)
        install_lock_wait_audit()
        random.seed(619046);np.random.seed(619046)
        server_log=(RESULTS/'embedding-service.log').open('a')
        server=subprocess.Popen([str(ROOT/'.venv/bin/python'),'scripts/run1_embedding_server.py',
            '--results',str(RESULTS),'--port','8771'],cwd=ROOT,stdout=server_log,stderr=subprocess.STDOUT)
        execution_started=time.monotonic()
        resource_peaks={'combined_rss_bytes':0,'controller_cpu_seconds':0,'embedding_cpu_seconds':0}
        try:
            for _ in range(40):
                if server.poll() is not None:
                    raise RuntimeError('Local embedding service exited; inspect log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8771/health',timeout=.5) as response:
                        if json.load(response).get('model')=='bge-small-code-chunks-v1':break
                except OSError:time.sleep(.25)
            else:raise RuntimeError('Local embedding service did not become ready')
            async def run():
                runner=FullRunner(evo_config=evo,db_config=db,job_config=job,max_evaluation_jobs=1,
                    max_proposal_jobs=1,max_db_workers=1,call_budget=budget)
                loop=asyncio.get_running_loop()
                def pulse():loop.call_later(.05,pulse)
                pulse()
                for sig in (signal.SIGINT,signal.SIGTERM):
                    loop.add_signal_handler(sig,runner.request_checkpoint,f'signal:{sig.name}')
                async def monitor():
                    while True:
                        await asyncio.sleep(1)
                        me=psutil.Process()
                        family=[me,*me.children(recursive=True)]
                        rss=0
                        for p in family:
                            try:rss+=p.memory_info().rss
                            except psutil.NoSuchProcess:pass
                        resource_peaks['combined_rss_bytes']=max(resource_peaks['combined_rss_bytes'],rss)
                        resource_peaks['controller_cpu_seconds']=sum(me.cpu_times()[:2])
                        try:resource_peaks['embedding_cpu_seconds']=sum(psutil.Process(server.pid).cpu_times()[:2])
                        except psutil.NoSuchProcess:pass
                        if rss>resolved['combined_rss_bytes_cap']:
                            budget.stop('Combined resident-memory ceiling exceeded')
                        elif budget.reason():runner.request_checkpoint(budget.reason())
                task=asyncio.create_task(monitor())
                try:await runner.run_async()
                finally:
                    task.cancel()
                    try:await task
                    except asyncio.CancelledError:pass
                    runner._save_state('original_stage1_finally')
                    budget.write_ledger()
            asyncio.run(run())
        finally:
            server.terminate()
            try:server.wait(timeout=5)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            server_log.close()
            atomic_json(RESULTS/'execution-resources.json',{'ended_utc':datetime.now(timezone.utc).isoformat(),
                'controller_execution_wall_seconds':time.monotonic()-execution_started,**resource_peaks,
                'all_role_calls':budget.totals(),'automatic_continuation':False,
                'supervising_assistant_usage':'additional; not measurable as subscription allowance here'})


if __name__=='__main__':main()
