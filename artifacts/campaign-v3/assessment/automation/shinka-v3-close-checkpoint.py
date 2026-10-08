"""Close an operator checkpoint without running programs or analyzing effects."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib,sys
ROOT=Path('/home/roland/projects/shinka-dreamer');sys.path.insert(0,str(ROOT))
from scripts.v3_assessment import verify_plan,verify_retained_replay,atomic_create
from scripts.v3_execution_report import host_time
raw=ROOT/'results/campaign-v3-assessment';public=ROOT/'artifacts/campaign-v3/assessment'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(public/'preregistration.json');verify_plan(plan)
conditions=plan['conditions'];regimes=plan['regimes'];order={(c['name'],r):i for i,(r,c) in enumerate((r,c) for r in regimes for c in conditions)}
expected={(c['name'],r) for r in regimes for c in conditions}
# Assemble the final drained case files from already durable, source-checked rows.
assembled=[];all_rows=[]
for folder in sorted((raw/'episode-checkpoints').iterdir()):
 if not folder.is_dir():continue
 i=int(folder.name);rows=[read(p) for p in folder.glob('*.json')]
 assert len(rows)==len(expected) and {(r['condition'],r['regime']) for r in rows}==expected
 for row in rows:
  c=next(c for c in conditions if c['name']==row['condition'])
  assert row['case']==i and row['program_sha256']==c['program_sha256'] and row['variant']==c['variant']
  verify_retained_replay(raw,i,row['regime'],row['condition'],plan['replays'],row)
 for regime in regimes:
  assert (raw/'matched'/f'{i:06d}--{regime}.json').is_file()
 rows.sort(key=lambda r:order[r['condition'],r['regime']]);case={'case':i,'conditions':rows}
 target=raw/'cases'/f'{i:06d}.json'
 if target.exists():assert read(target)==case
 else:atomic_create(target,case);assembled.append(i)
 all_rows.extend(rows)
world=[read(p) for p in (raw/'attempts').glob('*.completed.json')]
shadow=[read(p) for p in (raw/'matched-attempts').glob('*.completed.json')]
matched=[read(p) for p in (raw/'matched').glob('*.json')]
for folder,records in [('attempts',world),('matched-attempts',shadow)]:
 assert {read(p)['attempt'] for p in (raw/folder).glob('*.started.json')}=={r['attempt'] for r in records}
assert len(world)==len(all_rows)==4200
assert len({(r['case'],r['condition'],r['regime']) for r in world})==len(world)
assert sum(r['condition_episodes'] for r in shadow)==2100
assert len(matched)==420
launch=read(next(raw.glob('launch-*.finished.json')))
assert launch['exit_code']==130 and not (raw/'execution-complete.json').exists()
host_path=raw/launch['host_time_file'];(public/host_path.name).write_bytes(host_path.read_bytes())
session=read(next(raw.glob('execution-*.finished.json')))
invalid=Counter(f"{r['regime']}/{r['condition']}" for r in world if r['invalid'])
resources={'world_candidate_cpu_seconds':sum(r['candidate_cpu_seconds'] for r in world),'world_assessment_worker_cpu_seconds':sum(r['assessment_cpu_seconds'] for r in world),'world_evaluator_cpu_seconds_subset_not_added':sum(r['evaluator_cpu_seconds'] for r in world),'shadow_candidate_cpu_seconds':sum(r['candidate_cpu_seconds'] for r in shadow),'shadow_evaluator_cpu_seconds':sum(r['evaluator_cpu_seconds'] for r in shadow),'controller_cpu_seconds':session['controller_cpu_seconds'],'driver_monotonic_seconds':session['wall_seconds'],'wrapper_monotonic_seconds':launch['wrapper_monotonic_elapsed_seconds'],'gnu_time_alternative_not_added':host_time(public/host_path.name)}
resources['measured_nonoverlapping_cpu_seconds']=sum(resources[k] for k in ['world_candidate_cpu_seconds','world_assessment_worker_cpu_seconds','shadow_candidate_cpu_seconds','shadow_evaluator_cpu_seconds','controller_cpu_seconds'])
record={'status':'paused-by-user','recorded_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(public/'preregistration.json'),'planned_cases':plan['sample_size'],'fully_evaluated_cases':140,'stitched_cases_before_reconstruction':136,'reconstructed_cases_from_existing_checkpoints':assembled,'world_episodes':len(world),'shadow_episode_passes':sum(r['condition_episodes'] for r in shadow),'recorded_policy_episodes':len(matched),'invalid_world_episodes':sum(invalid.values()),'invalid_world_episodes_by_regime_condition':dict(sorted(invalid.items())),'individual_shadow_errors':sum(bool(s.get('error')) for r in matched for s in r['shadows'].values()),'outer_matched_failures':sum(bool(r.get('error')) for r in matched),'unfinished_world_attempts':0,'unfinished_matched_attempt_groups':0,'unavailable_world_cpu_measurements':sum(bool(r.get('cpu_measurement_unavailable')) for r in world),'case_file_sha256':{p.name:sha(p) for p in sorted((raw/'cases').glob('*.json'))},'world_attempt_manifest_sha256':sha(Path('/tmp/shinka-v3-close-checkpoint.py')),'resources':resources,'launch_finished':launch,'driver_session_finished':session,'stop_reason':read(raw/'operator-checkpoint-request.json')['reason'],'full_assessment_complete':False,'treatment_effects_aggregated':False,'automatic_follow_through_stopped':True,'new_world_episodes_from_checkpoint_closure':0,'new_model_calls_from_checkpoint_closure':0,'resume_command':'.venv/bin/python scripts/v3_assessment.py --plan artifacts/campaign-v3/assessment/preregistration.json --out results/campaign-v3-assessment --workers 4','resume_status':'Do not run until the user decides the remaining scope. No automatic resume.','resource_note':'GNU time terminated-by-signal2 is authoritative alongside launcher exit130; its printed Exit status0 is not a successful experimental completion. Whole-command CPU overlaps measured components; evaluator is included in workerCPU. Monotonic and UTC/GNU elapsed differ and remain separate.'}
record['checkpoint_closure_source_sha256']=record.pop('world_attempt_manifest_sha256')
atomic_create(raw/'operator-checkpoint-complete.json',record)
atomic_create(public/'operator-checkpoint-complete.json',record)
atomic_create(public/'operator-checkpoint-request.json',read(raw/'operator-checkpoint-request.json'))
print(json.dumps({k:record[k] for k in ['status','fully_evaluated_cases','world_episodes','shadow_episode_passes','invalid_world_episodes','individual_shadow_errors','unfinished_world_attempts','reconstructed_cases_from_existing_checkpoints']}))
print(json.dumps(resources))
