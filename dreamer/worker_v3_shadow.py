"""Isolated passive replay of source-reviewed v3 predictors.

No planner runs and no action is executed. Only the current public observation
and the preceding recorded action cross the ordinary-shadow boundary. This
adapters are specific to the reviewed directional comparator and selected gen4;
they are not interfaces imposed on evolved candidates.
"""
import sys
sys.path[:] = ["/usr/lib/python3.10", "/usr/lib/python3.10/lib-dynload"]
import contextlib
import json
import hashlib
import os
import random
sys.path.insert(0, os.path.dirname(__file__))
from isolation import restrict
sys.path.pop(0)


def main():
    with open(sys.argv[1]) as handle:
        source = handle.read(512 * 1024 + 1)
    if len(source) > 512 * 1024:
        raise ValueError("Comparator exceeds source limit")
    candidate_seed = int(sys.argv[2])
    adapter, expected_hash = sys.argv[3:5]
    if adapter not in ('directional-v3-no-planner', 'selected-gen4-no-planner'):
        raise ValueError('Unreviewed passive adapter')
    if hashlib.sha256(source.encode()).hexdigest() != expected_hash:
        raise ValueError('Passive source changed before isolated execution')
    restrict()
    sys.argv[:] = ["comparator.py"]
    random.seed(candidate_seed)
    namespace = {"__name__": "comparator"}
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(source, "comparator.py", "exec"), namespace)
    memory = None
    for line in sys.stdin:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            observation = request["obs"]
            memory = namespace["world_model_step"](memory, observation, request["last_action"])
            # Both source-reviewed world-model updates compute forecasts before
            # their planners. The memory policy always interacts before moving.
            # Gen4's planner only writes target/target_kind, unused by its model.
            model = namespace["export_model"](memory, observation)
        print(json.dumps({"action": {"move": [0, 0], "interact": True},
                          "model": model, "passive_shadow": True}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
