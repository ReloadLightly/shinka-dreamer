"""Run a frozen paired case panel through the unchanged original evaluator.

Private world seeds stay in results/private; public rows use ordinal case IDs.
Each episode is checkpointed. No model clients or search controller are imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import secrets
import signal
import time

from proposal.evaluate_full import validate_sources
from proposal.evaluation import aggregate, run_episode

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
    temp.replace(path)


def cpu():
    own, child = (resource.getrusage(who) for who in
                  (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    return own.ru_utime + own.ru_stime, child.ru_utime + child.ru_stime


def private_pool(protocol, protocol_path, out):
    target = (ROOT / protocol['pool_path']).resolve()
    if not target.is_relative_to(ROOT / 'results/private'):
        raise ValueError('Pool must stay under ignored results/private')
    excludes, sources = set(), {}
    for path in sorted((ROOT / 'results/private').rglob('*seed*.json')):
        if path.resolve() == target:
            continue
        values = read(path)
        if not isinstance(values, list) or not all(type(v) is int for v in values):
            raise ValueError('Noninteger private seed pool: ' + path.name)
        excludes.update(values)
        sources[str(path.relative_to(ROOT))] = sha(path)
    n = protocol['case_count']
    binding = out / 'pool-manifest.json'
    if not binding.exists():
        if target.exists():
            raise ValueError('Unbound existing pool; do not redraw or overwrite')
        seeds = []
        while len(seeds) < n:
            value = secrets.randbits(63)
            if value >= 1_000_000 and value not in excludes and value not in seeds:
                seeds.append(value)
        write(target, seeds)
        target.chmod(0o600)
        write(binding, {'protocol_sha256': sha(protocol_path), 'pool_sha256': sha(target),
                        'case_count': n, 'excluded_ranges': [[0, 1_000_000]],
                        'excluded_pool_sha256': sources,
                        'public_ids': 'ordinal indices only; private seeds never exported'})
    manifest = read(binding)
    if manifest['protocol_sha256'] != sha(protocol_path) or manifest['pool_sha256'] != sha(target):
        raise ValueError('Frozen protocol or private pool changed')
    seeds = read(target)
    if len(seeds) != n or len(set(seeds)) != n or any(s < 1_000_000 or s in excludes for s in seeds):
        raise ValueError('Invalid or overlapping case pool')
    return seeds


def alarm(_signum, _frame):
    raise TimeoutError('Declared panel episode wall limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    protocol = validate_sources(args.protocol)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(protocol['budget']['execution_stop_utc'].replace('Z', '+00:00'))
    if datetime.now(timezone.utc) >= deadline:
        raise RuntimeError('Execution deadline reached; no automatic budget extension')
    seeds = private_pool(protocol, args.protocol, out)
    signal.signal(signal.SIGALRM, alarm)
    arms = protocol['programs']
    start = time.monotonic()
    start_cpu = cpu()
    started = datetime.now(timezone.utc).isoformat()
    measured = {arm: [] for arm in arms}
    stop_reason = 'complete'
    for index, seed in enumerate(seeds):
        for arm, spec in arms.items():
            path = out / arm / f'case-{index:03d}.json'
            if path.exists():
                measured[arm].append(read(path))
                continue
            used = sum(r['resource']['total_cpu_seconds'] for rows in measured.values() for r in rows)
            if used + 20 > protocol['budget']['total_evaluator_cpu_seconds'] or datetime.now(timezone.utc) >= deadline:
                stop_reason = 'resource checkpoint'
                break
            intent = path.with_suffix('.pending.json')
            if intent.exists():
                raise RuntimeError('Unfinished episode intent; preserve it for explicit recovery')
            write(intent, {'case_index': index, 'arm': arm, 'started_utc': datetime.now(timezone.utc).isoformat()})
            before, began = cpu(), time.monotonic()
            signal.setitimer(signal.ITIMER_REAL, protocol['budget']['episode_wall_seconds'])
            try:
                row = run_episode(ROOT / spec['path'], seed)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            after = cpu()
            row['resource'] = {'elapsed_seconds': time.monotonic() - began,
                'evaluator_cpu_seconds': after[0] - before[0],
                'candidate_cpu_seconds': after[1] - before[1],
                'total_cpu_seconds': sum(after) - sum(before)}
            write(path, row)
            intent.unlink()
            measured[arm].append(row)
        if stop_reason != 'complete':
            break
        if (index + 1) % 16 == 0:
            print(json.dumps({'paired_cases_completed': index + 1}), flush=True)
    public = ROOT / protocol['public_output']
    for arm, rows in measured.items():
        exported = []
        for index, row in enumerate(rows):
            item = dict(row)
            item['case'] = index
            item.pop('resource', None)
            exported.append(item)
        write(public / arm / 'episodes.json', exported)
        if rows:
            write(public / arm / 'metrics.json', aggregate(rows))
        write(public / arm / 'resource.json', {
            key: sum(r['resource'][key] for r in rows)
            for key in ['elapsed_seconds', 'evaluator_cpu_seconds', 'candidate_cpu_seconds', 'total_cpu_seconds']})
        write(public / arm / 'manifest.json', {'evaluator': protocol['evaluator'],
            'candidate_sha256': sha(ROOT / arms[arm]['path']), 'episodes': len(rows),
            'case_ids': 'private pool ordinal indices', 'pool_sha256': read(out / 'pool-manifest.json')['pool_sha256'],
            'source_sha256': protocol['source_sha256'], 'purpose': 'Selection-validation; selection-biased, not final assessment'})
    write(public / 'pool-manifest.json', read(out / 'pool-manifest.json'))
    end_cpu = cpu()
    write(public / 'execution.json', {'started_utc': started,
        'ended_utc': datetime.now(timezone.utc).isoformat(), 'invocation_elapsed_seconds': time.monotonic() - start,
        'status': stop_reason, 'episodes': sum(map(len, measured.values())),
        'transitions': sum(r['steps'] for rows in measured.values() for r in rows),
        'model_calls': 0, 'new_candidate_sources': 0,
        'evaluator_cpu_seconds': end_cpu[0] - start_cpu[0],
        'candidate_cpu_seconds': end_cpu[1] - start_cpu[1],
        'total_cpu_seconds': sum(end_cpu) - sum(start_cpu),
        'definition': 'Per-episode CPU covers evaluator plus reaped candidate child. Driver export/validation overhead is outside summed episode CPU.'})
    print(json.dumps(read(public / 'execution.json')), flush=True)


if __name__ == '__main__':
    main()
