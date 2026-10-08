"""Bounded development replay diagnostic. No worlds or LLM calls are executed."""
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.world import stream_seed
from dreamer.world_v3 import draw_law, UNIFORM

OUT = ROOT / 'artifacts/campaign-v3/development-diagnostic20'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def laws(seed, regime):
    """Only tagged dynamics sampling: no maze construction or transition."""
    initial = draw_law(random.Random(stream_seed(seed, 'v3:law:initial')))
    rng = random.Random(stream_seed(seed, 'v3:law:replacement'))
    replacement = draw_law(rng)
    while sum(abs(a-b) for a,b in zip(initial,replacement))/2 < .3:
        replacement = draw_law(rng)
    switch = random.Random(stream_seed(seed,'v3:switch-time')).randint(25,75) if regime == 'switch' else None
    return UNIFORM if regime == 'uniform' else initial, replacement, switch


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    private = ROOT/'results/v3-development-diagnostic20'
    private.mkdir(parents=True, exist_ok=True)
    lock = (private/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT/'execution-started.json').exists():
        raise RuntimeError('Already attempted: preserve all slots; no automatic rerun')
    now = datetime.now(timezone.utc)
    absolute_deadline = datetime.fromisoformat(protocol['execution_deadline_utc'])
    remaining = min(protocol['maximum_execution_seconds'], (absolute_deadline-now).total_seconds())
    if remaining <= 0:
        raise RuntimeError('Recorded execution deadline expired; no execution authorized')
    started, cpu = time.monotonic(), time.process_time()
    deadline = started + remaining
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    seeds_path = ROOT/'results/private/v3-fit-validation-seeds.json'
    if sha(seeds_path) != protocol['private_pool_sha256']:
        raise ValueError('Development pool changed')
    seeds = json.loads(seeds_path.read_text())
    sources = protocol['sources']
    for source in sources.values():
        if sha(ROOT/source['path']) != source['sha256']:
            raise ValueError('Frozen source changed')
    jobs = []
    for regime in protocol['regimes']:
        for case in protocol['cases_per_regime']:
            name = f'{regime}-{case:04d}.json.gz'
            path = ROOT/protocol['corpus_directory']/name
            if sha(path) != protocol['corpus_sha256'][name]:
                raise ValueError('Development recording changed')
            with gzip.open(path,'rt') as handle:
                record = json.load(handle)
            if record['case'] != case or record['regime'] != regime:
                raise ValueError('Recording identity mismatch')
            frames = record['frames'][:protocol['frames_per_pass_cap']]
            initial,replacement,switch = laws(seeds[case],regime)
            for kind in ('selected','comparator'):
                for profiled in (False,True):
                    requests, last = [], None
                    for frame in frames:
                        request = {'obs':frame['obs'], 'last_action':last, 'recorded_action':frame['action']}
                        if kind == 'comparator':
                            request['known_law'] = list(replacement if switch is not None and frame['obs']['step']+1 >= switch else initial)
                        requests.append(request)
                        last = frame['action']
                    jobs.append((regime,case,kind,profiled,requests,switch))
    if len(jobs) != 48 or sum(len(job[4]) for job in jobs) > 3840:
        raise ValueError('Recorded replay budget exceeded')
    scripts = ('v3_diagnostic20.py','v3_diagnostic_worker.py','v3_diagnostic_observer.py')
    save(OUT/'execution-started.json', {
        'started_utc':now.isoformat(), 'protocol_sha256':sha(OUT/'protocol.json'),
        'script_sha256':{name:sha(ROOT/'scripts'/name) for name in scripts},
        'attempt_slots':len(jobs), 'requested_frames':sum(len(job[4]) for job in jobs),
        'remaining_execution_seconds':remaining, 'workers':2, 'new_world_episodes':0,
        'experiment_model_calls':0, 'original_assessment_resumed':False})

    def run(job):
        regime,case,kind,profiled,requests,switch = job
        name = f'{regime}-{case:04d}-{kind}-'+('profiled' if profiled else 'plain')
        start = time.monotonic()
        identity = {'case':case,'regime':regime,'kind':kind,'profiled':profiled,
                    'requested_frames':len(requests),'source_sha256':sources[kind]['sha256']}
        if start >= deadline:
            result = {**identity,'status':'not_started_deadline','frames':[]}
        else:
            save(OUT/'attempts'/f'{name}.started.json',identity)
            cmd = ['/usr/bin/python3','-s','-S',str(ROOT/'scripts/v3_diagnostic_worker.py'),
                   str(ROOT/sources[kind]['path']),sources[kind]['sha256'],kind,str(int(profiled))]
            payload = ''.join(json.dumps(request)+'\n' for request in requests)+'{"finish":true}\n'
            proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                    cwd='/tmp',env={'PATH':'/usr/bin:/bin','PYTHONHASHSEED':'0'})
            timed_out = False
            try:
                output, errors = proc.communicate(payload.encode(),timeout=min(35,max(.01,deadline-start)))
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                output, errors = proc.communicate()
            records = [json.loads(line) for line in output.decode().splitlines() if line.strip()]
            finishes = [r for r in records if r.get('finished')]
            frames = [r for r in records if 'frame' in r]
            result = {**identity,'status':'complete' if proc.returncode == 0 and len(finishes)==1 and len(frames)==len(requests) else 'failed',
                      'returncode':proc.returncode,'timeout':timed_out,'frames':frames,
                      'finish':finishes[0] if finishes else None,
                      'elapsed_seconds':time.monotonic()-start,
                      'post_switch_recorded_frames':sum(r['obs']['step'] >= switch for r in requests) if switch is not None else 0}
            # Keep full stderr private; public status/error class suffices.
            if errors:
                (private/f'{name}.stderr.txt').write_bytes(errors)
                result['stderr_present'] = True
        save(OUT/'passes'/f'{name}.json',result)
        return result

    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,job) for job in jobs]):
            result=future.result();results.append(result)
            print(json.dumps({'completed_slots':len(results),'of':len(jobs),'status':result['status'],
                              'kind':result['kind'],'regime':result['regime'],'case':result['case']}),flush=True)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    save(OUT/'execution-finished.json', {
        'finished_utc':datetime.now(timezone.utc).isoformat(), 'elapsed_seconds':time.monotonic()-started,
        'controller_cpu_seconds':time.process_time()-cpu,
        'worker_cpu_seconds':child.ru_utime+child.ru_stime-child_before.ru_utime-child_before.ru_stime,
        'complete_passes':sum(r['status']=='complete' for r in results),
        'failed_passes':sum(r['status']=='failed' for r in results),
        'unstarted_passes':sum(r['status']=='not_started_deadline' for r in results),
        'returned_frames':sum(len(r['frames']) for r in results),
        'new_world_episodes':0,'experiment_model_calls':0,'original_assessment_resumed':False,
        'pass_file_sha256':{p.name:sha(p) for p in sorted((OUT/'passes').glob('*.json'))}})


if __name__ == '__main__':
    main()
