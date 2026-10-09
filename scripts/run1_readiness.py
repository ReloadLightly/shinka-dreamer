"""Exactly one budgeted subscription check per requested RUN1 model."""
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MODELS=['headless/codex@gpt-6-astra?effort=high','headless/codex@gpt-6.1-sol?effort=high']

def configure():
    os.environ['HEADLESS_BILLING']='subscription'
    os.environ['SHINKA_HEADLESS_COMMAND']=str(ROOT/'scripts/subscription_headless.sh')
    os.environ['SHINKA_HEADLESS_TIMEOUT']='600'
    for name in ('OPENAI_API_KEY','CODEX_API_KEY','OPENAI_BASE_URL','ANTHROPIC_API_KEY',
                 'GEMINI_API_KEY','GOOGLE_API_KEY','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY',
                 'AWS_SESSION_TOKEN','OPENROUTER_API_KEY'):
        os.environ.pop(name,None)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    from dreamer.headless_transport import install
    install()

async def main():
    configure()
    from dreamer.native_run1 import Run1CallBudget,install_run1_call_audit
    root=ROOT/'results/campaign-v4-run1'
    marker=root/'subscription-readiness.json'
    if marker.exists():
        raise SystemExit('Readiness already attempted; inspect its checkpoint, do not repeat probes')
    marker.write_text(json.dumps({'status':'started','models':MODELS},indent=2))
    budget=Run1CallBudget(root,deadline_utc='2026-10-09T04:24:23Z')
    install_run1_call_audit(budget)
    completed=[]
    try:
        for model in MODELS:
            result=await budget.probe(model)
            if budget.reason():raise RuntimeError(budget.reason())
            completed.append({'model':model,'reply':str(result)[:400]})
            marker.write_text(json.dumps({'status':'running','completed':completed},indent=2))
        marker.write_text(json.dumps({'status':'verified','completed':completed,'model_count':2},indent=2))
        print(json.dumps({'status':'verified','model_count':2,'budget':budget.totals()}),flush=True)
    except BaseException as exc:
        marker.write_text(json.dumps({'status':'blocked','completed':completed,'error':str(exc)},indent=2))
        raise

if __name__=='__main__':asyncio.run(main())
