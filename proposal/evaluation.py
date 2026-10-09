"""Original task and current-map accuracy; hidden world stays evaluator-owned."""
import json
from pathlib import Path
import selectors
import subprocess
import time

from dreamer.evaluation import Candidate as Transport
from dreamer.world import DynamicMaze, ENEMY

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = 'namazu-proposal-reconstruction-v1'


class Candidate(Transport):
    # Reuse only the established bounded JSON transport and cleanup methods.
    def __init__(self, path):
        self.process = subprocess.Popen(
            ['/usr/bin/python3', '-s', '-S', str(ROOT/'proposal/worker.py'), str(Path(path).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={'PATH': '/usr/bin:/bin', 'PYTHONHASHSEED': '0'}, cwd='/tmp', bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b''


def task_score(steps, keys, door_open, reason):
    # Literal source reward, including its positive survival/step term.
    raw = .1 * steps + 20 * keys + 30 * door_open
    raw += 100 * (reason == 'escaped') - 50 * (reason == 'caught')
    return min(1., max(0., (raw + 50) / 250))


def audit_map(belief, env):
    """Source known-cell denominator; privately repair relative coordinates."""
    if not isinstance(belief, list):
        raise ValueError('believed_map must be a coordinate-to-cell dictionary')
    correct = total = 0
    unique = set()
    for row in belief:
        if (not isinstance(row, list) or len(row) != 3
                or any(type(v) is not int for v in row) or row[2] not in range(6)):
            raise ValueError('Invalid believed-map entry')
        x, y, cell = row
        if (x, y) in unique:
            raise ValueError('Duplicate believed-map coordinate')
        unique.add((x, y))
        gx, gy = x + env.origin[0], y + env.origin[1]
        if 0 <= gx < env.size and 0 <= gy < env.size:
            actual = ENEMY if (gx, gy) in env.enemies else env.grid[gy][gx]
            correct += cell == actual
            total += 1
    return correct, total


def run_episode(candidate_path, seed, replay=False):
    env = DynamicMaze(seed)
    correct = total = 0
    timeline, trace = [], []
    error = None
    start = time.monotonic()
    with Candidate(candidate_path) as candidate:
        while not env.done:
            try:
                request = {'obs': env.observe()}
                reply = candidate.query(request)
                c, n = audit_map(reply['belief'], env)
                correct += c
                total += n
                timeline.append({'step': env.steps, 'correct': c, 'audited': n,
                                 'coverage': n / env.size**2})
                if replay:
                    # Copy before transition; all privileged state stays here.
                    trace.append(json.loads(json.dumps({'world': env.snapshot(),
                        'belief': reply['belief'], 'action': reply['action']})))
                env.step(reply['action'])
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                env.done, env.reason = True, 'invalid'
    task = task_score(env.steps, env.keys_collected, env.door_open, env.reason)
    accuracy = correct / total if total else 0.
    result = {'case': seed, 'reason': env.reason, 'steps': env.steps,
              'keys': env.keys_collected, 'door_open': env.door_open,
              'task': task, 'model_accuracy': accuracy,
              'combined_score': .6 * task + .4 * accuracy if error is None else 0.,
              'map_correct': correct, 'map_audited': total,
              'final_coverage': timeline[-1]['coverage'] if timeline else 0.,
              'timeline': timeline, 'error': error, 'seconds': time.monotonic() - start}
    if replay:
        result['trace'] = trace
    return result


def aggregate(rows):
    if not rows:
        raise ValueError('No episodes')
    mean = lambda key: sum(row[key] for row in rows) / len(rows)
    public = {'episodes': len(rows), 'mean_task_score': mean('task'),
              'mean_model_accuracy': mean('model_accuracy'), 'avg_steps': mean('steps'),
              'avg_keys': mean('keys'), 'mean_map_coverage': mean('final_coverage'),
              **{name: sum(r['reason'] == name for r in rows)
                 for name in ('escaped', 'caught', 'timeout', 'invalid')}}
    return {'combined_score': mean('combined_score'), 'public': public,
            'text_feedback': f'{IDENTITY}. Original .6 task + .4 current-map accuracy. '
            f'{json.dumps(public)}. Improve both map updating and planning. '
            'This is reconstruction after observation, not future prediction. '
            'Accuracy uses reported in-bounds map cells; coverage is diagnostic. '
            'Inspect escapes and failures separately from the weighted score.'}
