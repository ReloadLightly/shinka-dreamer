"""Finish authorized saved-data reporting only after the frozen run closes."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl, hashlib, json, os, subprocess, sys, time
ROOT=Path('/home/roland/projects/shinka-dreamer')
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from scripts.v3_assessment import verify_plan, atomic_create
from scripts.v3_execution_report import host_time
raw=ROOT/'results/campaign-v3-assessment'
public=ROOT/'artifacts/campaign-v3/assessment'
lock=(raw/'report-follow-through.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=public/'preregistration.json'
expected='2b292e2373cc1d5b3c558a5339ed1588323272e2dbc606ead130640bfd1aae7d'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def emit(**kw):print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**kw}),flush=True)
emit(status='waiting-for-frozen-assessment',pid=os.getpid(),model_calls=0,new_world_episodes=0)
while not (raw/'execution-complete.json').exists():
 finished=list(raw.glob('launch-*.finished.json'))
 if finished and any(read(p).get('exit_code') for p in finished):
  raise RuntimeError('Assessment invocation failed; retain evidence and review before resuming')
 time.sleep(30)
while not list(raw.glob('launch-*.finished.json')):time.sleep(2)
assert sha(plan)==expected
verify_plan(read(plan))
complete=read(raw/'execution-complete.json')
assert complete['cases']==1536 and complete['condition_episodes']==46080
launches=[read(p) for p in sorted(raw.glob('launch-*.finished.json'))]
assert all(p['exit_code']==0 for p in launches)
for name in ('execution-complete.json','manifest.json','pool-manifest.json'):
 atomic_create(public/name,read(raw/name))
# GNU time command lines contain frozen public paths only, no case values.
host=[]
for p in sorted(raw.glob('host-time*.txt')):
 target=public/p.name
 if target.exists():assert target.read_bytes()==p.read_bytes()
 else:target.write_bytes(p.read_bytes())
 host.append(host_time(target))
provenance={'status':'execution-complete','freeze_commit':launches[0]['commit'],
 'plan_sha256':expected,'launches':launches,
 'driver_sessions':[read(p) for p in sorted(raw.glob('execution-*.finished.json'))],
 'host_time':host,'raw_completion_sha256':sha(raw/'execution-complete.json'),
 'model_calls':0,'resource_note':'Whole-command CPU overlaps episode/worker/shadow components; do not add. UTC, monotonic and GNU wall measures remain separate.',
 'pool_note':'Only path, count, hash and exclusion metadata are public; case values remain private.',
 'automation_source_sha256':sha(__file__)}
atomic_create(public/'execution-provenance.json',provenance)
emit(status='execution-verified-starting-frozen-analysis')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
commands=[['.venv/bin/python','scripts/v3_analysis.py','--raw','results/campaign-v3-assessment'],
 ['.venv/bin/python','scripts/v3_figures.py'],
 ['.venv/bin/python','scripts/v3_examples.py','--raw','results/campaign-v3-assessment','--gif'],
 ['.venv/bin/python','scripts/v3_execution_report.py']]
records=[]
for command in commands:
 start=time.monotonic();emit(status='running',command=command)
 result=subprocess.run(command,env=env)
 records.append({'command':command,'exit_code':result.returncode,'monotonic_seconds':time.monotonic()-start})
 atomic_create(raw/f'report-stage-{len(records)}.json',records[-1])
 if result.returncode:raise RuntimeError('Reporting command failed: '+str(command))
atomic_create(raw/'report-follow-through-complete.json',{'commands':records,'new_world_episodes':0,'model_calls':0})
emit(status='saved-data-reports-complete-human-review-remains')
