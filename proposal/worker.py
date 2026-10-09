"""Fixed observation/action/map boundary; no forecasting contract."""
import sys
sys.path[:] = ['/usr/lib/python3.10', '/usr/lib/python3.10/lib-dynload']
import contextlib
import json
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dreamer'))
from isolation import restrict
sys.path.pop(0)


def main():
    with open(sys.argv[1]) as source_file:
        source = source_file.read(512 * 1024 + 1)
    if len(source) > 512 * 1024:
        raise ValueError('Candidate exceeds 512 KiB')
    restrict()
    random.seed(712934)  # fixed candidate stream, independent of hidden world RNGs
    namespace = {'__name__': 'candidate'}
    memory, last = None, None
    for index, line in enumerate(sys.stdin):
        request = json.loads(line)
        if index == 0:
            with contextlib.redirect_stdout(sys.stderr):
                exec(compile(source, 'candidate.py', 'exec'), namespace)
        with contextlib.redirect_stdout(sys.stderr):
            obs = request['obs']
            memory = namespace['world_model_step'](memory, obs, last)
            action = namespace['planner'](memory, obs)
            belief = [[x, y, cell] for (x, y), cell in memory['believed_map'].items()]
        print(json.dumps({'action': action, 'belief': belief}, allow_nan=False), flush=True)
        last = action


if __name__ == '__main__':
    main()
