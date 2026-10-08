from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, time, hashlib
root=Path('/home/roland/projects/shinka-dreamer')
out=root/'results/campaign-v3-assessment'
out.mkdir(parents=True,exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
record={'started_utc':datetime.now(timezone.utc).isoformat(),'wrapper_pid':os.getpid(),
 'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'plan_sha256':hashlib.sha256((root/'artifacts/campaign-v3/assessment/preregistration.json').read_bytes()).hexdigest(),
 'command':['.venv/bin/python','-u','scripts/v3_assessment.py','--plan','artifacts/campaign-v3/assessment/preregistration.json','--out','results/campaign-v3-assessment','--workers','4','--reserve-pool'],
 'host_time_file':'host-time-'+stamp+'.txt','log_file':'controller-'+stamp+'.log','model_calls':0}
log=(out/record['log_file']).open('xb')
start=time.monotonic()
process=subprocess.Popen(['/usr/bin/time','-v','-o',str(out/record['host_time_file']),*record['command']],cwd=root,stdout=log,stderr=subprocess.STDOUT)
record['timed_process_pid']=process.pid
(out/('launch-'+stamp+'.started.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
code=process.wait();log.close()
record.update(finished_utc=datetime.now(timezone.utc).isoformat(),wrapper_monotonic_elapsed_seconds=time.monotonic()-start,exit_code=code)
(out/('launch-'+stamp+'.finished.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(code)
