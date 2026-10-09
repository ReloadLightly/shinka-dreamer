"""One bounded full-machinery native Shinka RUN 1; no automatic next wave."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run1_readiness import configure, MODELS
from scripts.evolve_v3 import tree_hash, UPSTREAM, HEADLESS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare_only == args.run:
        parser.error('Specify exactly one of --prepare-only or --run')
    configure()
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    import numpy as np
    import shinka
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    from dreamer.native_run1 import Run1Runner,Run1CallBudget,install_run1_call_audit,DEFAULT_LIMITS
    from dreamer.provenance import control_path,pool,record,sha256
    from dreamer.evaluation_v4 import evaluation_identity
    from dreamer.world_v4 import CONDITIONS
    from scripts.recover_campaign import campaign_lock
    results=ROOT/'results/campaign-v4-run1'
    results.mkdir(parents=True,exist_ok=True)
    seed_file=ROOT/'results/private/v4-run1-development-seeds.json'
    _,pool_info=pool(seed_file,8)
    installed=json.loads(importlib.metadata.distribution('shinka-evolve').read_text('direct_url.json'))
    if installed.get('vcs_info',{}).get('commit_id') != UPSTREAM:
        raise ValueError('Installed native revision changed')
    if subprocess.check_output(['git','-C',str(ROOT/'.runtime/headless'),'rev-parse','HEAD'],text=True).strip()!=HEADLESS:
        raise ValueError('Headless route revision changed')
    readiness=json.loads((results/'subscription-readiness.json').read_text())
    if readiness['status']!='verified' or readiness['model_count']!=2:
        raise ValueError('Two distinct subscription models have not been verified')
    embedding=json.loads((results/'embedding-identity.json').read_text())
    import urllib.request
    with urllib.request.urlopen('http://127.0.0.1:8771/health',timeout=3) as response:
        if json.load(response).get('model')!=embedding['endpoint_model']:
            raise ValueError('Unexpected local embedding identity')
    evo=EvolutionConfig(
        task_sys_msg=(ROOT/'docs/mutation-prompt-v4.md').read_text(),
        num_generations=32,init_program_path=str(control_path('predictive')),
        results_dir=str(results),llm_models=MODELS,llm_dynamic_selection='ucb',
        llm_dynamic_selection_kwargs={'cost_aware_coef':0.0},llm_kwargs={},
        meta_rec_interval=4,meta_llm_models=[MODELS[1]],meta_llm_kwargs={},
        meta_max_recommendations=5,sample_single_meta_rec=True,
        embedding_model='local/bge-small-code-chunks-v1@http://127.0.0.1:8771/v1',
        code_embed_sim_threshold=.95,max_novelty_attempts=3,
        novelty_llm_models=[MODELS[1]],novelty_llm_kwargs={},
        use_text_feedback=True,evolve_prompts=True,prompt_llm_models=[MODELS[1]],
        prompt_llm_kwargs={},prompt_evolution_interval=4,prompt_archive_size=8,
        prompt_percentile_recompute_interval=4,prompt_evo_top_k_programs=3,
        patch_types=['diff','full','cross'],patch_type_probs=[.4,.3,.3],
        max_patch_resamples=2,max_patch_attempts=2,
        enable_controlled_oversubscription=False)
    database=DatabaseConfig(num_islands=4,migration_interval=4,migration_rate=.25,
        archive_size=32,num_archive_inspirations=1,num_top_k_inspirations=1,
        parent_selection_strategy='weighted',parent_selection_lambda=10.0,
        island_selection_strategy='uniform',enforce_island_separation=True,
        island_elitism=True,archive_selection_strategy='fitness')
    job=LocalJobConfig(eval_program_path=str(ROOT/'evaluate_v4.py'),
        python_executable=str(ROOT/'.venv/bin/python'),
        extra_cmd_args={'episodes':8,'seed_file':str(seed_file),
                        'campaign_manifest':str(results/'campaign-manifest.json')},
        time='00:08:00',numeric_threads_per_job=1)
    resolved={'evolution':asdict(evo),'database':asdict(database),'job':asdict(job),
        'max_evaluation_jobs':1,'max_proposal_jobs':1,'max_db_workers':1,
        'episode_workers':2,'budget':DEFAULT_LIMITS,'billing':'subscription',
        'deadline_utc':'2026-10-09T04:24:23Z','hard_checkpoint_utc':'2026-10-09T04:54:23Z',
        'native_sampling_seed':4017391,'upstream':UPSTREAM,'headless':HEADLESS,
        'effective_models':[{'model':'gpt-6-astra','effort':'high'},{'model':'gpt-6.1-sol','effort':'high'}],
        'unsupported_not_forwarded':['temperature','max_tokens'],
        'roles':{'mutation_repair':MODELS,'summary_insight_recommendation':MODELS[1],
                 'novelty':MODELS[1],'prompt_mutation':MODELS[1],
                 'embedding':'CPU-only local quantized BGE chunks; no model API','evaluator':'local Python; no LLM'}}
    with campaign_lock(results):
        record(results/'dreamer-resolved.json',resolved)
        manifest={'campaign':'namazu-persistence-information-v4-run1','total_candidate_slots':32,
            'primary_budget':DEFAULT_LIMITS,'evaluation':evaluation_identity(),
            'development_pool':pool_info,'regimes':list(CONDITIONS),'candidate_episodes':48,
            'seed_program_sha256':sha256(control_path('predictive')),
            'seed_source':'immutable original predictive seed; no hand-engineered comparator or v3 champion incorporated',
            'protocol_sha256':sha256(ROOT/'docs/run1-protocol.md'),
            'resolved_sha256':sha256(results/'dreamer-resolved.json'),
            'embedding':embedding,'worker_instruction_sha256':sha256(results/'AGENTS.md'),
            'subscription_readiness_sha256':sha256(results/'subscription-readiness.json'),
            'driver_sha256':{p:sha256(ROOT/p) for p in ('scripts/evolve_v4.py','scripts/run1_readiness.py',
                'dreamer/native_run1.py','dreamer/native.py','dreamer/headless_transport.py',
                'scripts/subscription_headless.sh','scripts/run1_embedding_server.py','docs/mutation-prompt-v4.md')},
            'installed':{'shinka_revision':UPSTREAM,'shinka_version':importlib.metadata.version('shinka-evolve'),
                'shinka_python_sha256':tree_hash(Path(shinka.__file__).parent,'.py'),
                'headless_revision':HEADLESS,'headless_js_sha256':tree_hash(ROOT/'.runtime/headless/dist','.js'),
                'codex':subprocess.check_output(['codex','--version'],text=True).strip()}}
        record(results/'campaign-manifest.json',manifest)
        public=dict(manifest)
        record(ROOT/'artifacts/campaign-v4/run1/protocol.json',public)
        if args.prepare_only:
            print(json.dumps({'prepared':True,'slots':32,'episodes_per_slot':48,'manifest_sha256':sha256(results/'campaign-manifest.json')}))
            return
        budget=Run1CallBudget(results,deadline_utc=resolved['deadline_utc'])
        if budget.reason():raise RuntimeError(budget.reason())
        install_run1_call_audit(budget)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record(results/f'execution-{stamp}.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
            'manifest_sha256':sha256(results/'campaign-manifest.json'),'total_slots_including_seed_and_failures':32})
        random.seed(resolved['native_sampling_seed']);np.random.seed(resolved['native_sampling_seed'])
        async def run():
            loop=asyncio.get_running_loop()
            def pulse():loop.call_later(.05,pulse)
            pulse()
            runner=Run1Runner(evo_config=evo,db_config=database,job_config=job,
                max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1,call_budget=budget)
            for sig in (signal.SIGINT,signal.SIGTERM):
                loop.add_signal_handler(sig,runner.request_checkpoint,f'signal:{sig.name}')
            async def deadline_monitor():
                while True:
                    await asyncio.sleep(1)
                    if budget.reason():
                        runner.request_checkpoint(budget.reason())
                        return
            monitor=asyncio.create_task(deadline_monitor())
            try:await runner.run_async()
            finally:
                monitor.cancel()
                try:await monitor
                except asyncio.CancelledError:pass
                runner._save_state('launcher_finally')
                budget.write_ledger()
        asyncio.run(run())

if __name__=='__main__':main()
