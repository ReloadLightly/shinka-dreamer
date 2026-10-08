"""Isolated passive replay of the reviewed direction-aware v3 comparator.

No planner runs and no action is executed. Only the current public observation
and the preceding recorded action cross the ordinary-shadow boundary. This
adapter is specific to the manually reviewed direction-aware comparator; it is
not an interface imposed on evolved candidates.
"""
import sys
sys.path[:] = ["/usr/lib/python3.10", "/usr/lib/python3.10/lib-dynload"]
import contextlib
import json
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
            # world_model_step records the comparator's one-tick forecast. Its
            # transition probabilities do not depend on a hypothetical action;
            # the shared memory policy always interacts before moving.
            model = namespace["export_model"](memory, observation)
        print(json.dumps({"action": {"move": [0, 0], "interact": True},
                          "model": model, "passive_shadow": True}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
