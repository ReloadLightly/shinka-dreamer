"""Source-bound selected-agent adapter for a bounded development branch study.

The evaluator sends public observations and actions only. At an encounter it may
also send current movement probabilities exclusively for a discarded known-law
planning copy. Physical snapshots and future randomness never cross this pipe.
"""
import sys
sys.path[:] = ["/usr/lib/python3.10", "/usr/lib/python3.10/lib-dynload"]
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run1_state_digest import state_digest
sys.path.pop(0)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dreamer"))
from isolation import restrict
sys.path.pop(0)

SOURCE_SHA256 = "aafa35fd866c353c83d0b661d79f2060d678fd2060ff50a1b10ea2f0e3ef5d3c"


def digest(value):
    return state_digest(value)


def main():
    if len(sys.argv) != 3 or sys.argv[2] != SOURCE_SHA256:
        raise ValueError("Expected reviewed selected source and exact SHA256")
    with open(sys.argv[1], "rb") as handle:
        source = handle.read(512 * 1024 + 1)
    if len(source) > 512 * 1024 or hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Selected branch source changed")
    restrict()
    sys.argv[:] = ["candidate.py"]
    random.seed(0)
    namespace = {"__name__": "candidate"}
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(source.decode(), "candidate.py", "exec"), namespace)
    memory = checkpoint = None
    frames = proposals = restores = 0
    start_cpu, start_wall = time.process_time(), time.monotonic()
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("finish"):
            print(json.dumps({"finished": True, "frames": frames,
                              "proposal_sets": proposals, "restores": restores,
                              "cpu_seconds": time.process_time() - start_cpu,
                              "wall_seconds": time.monotonic() - start_wall}), flush=True)
            return
        if request.get("restore"):
            if checkpoint is None:
                raise ValueError("No observed checkpoint to restore")
            memory = copy.deepcopy(checkpoint)
            if memory != checkpoint:
                raise ValueError("Restored selected built-in memory values differ")
            restores += 1
            print(json.dumps({"restored": True, "memory_sha256": digest(memory)}), flush=True)
            continue
        if frames >= 128:
            raise ValueError("Branch worker public-observation cap exceeded")
        obs = dict(request["obs"])
        if "known_law" in obs:
            raise ValueError("Privileged law in ordinary observation")
        obs["learn"], obs["predictive_planning"] = True, True
        with contextlib.redirect_stdout(sys.stderr):
            memory = namespace["world_model_step"](memory, obs, request.get("last_action"))
            alternatives = None
            if request.get("propose"):
                law = request["known_law"]
                if (not isinstance(law, list) or len(law) != 9 or
                        any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in law)
                        or abs(sum(law) - 1.) > 1e-9):
                    raise ValueError("Invalid privileged current law")
                base_before = digest(memory)
                alternatives = {}
                for name, replacement in (("frozen_prior", namespace["UNIFORM"]),
                                          ("known_law", law)):
                    isolated = copy.deepcopy(memory)
                    forecast, hazards, default = namespace["_forecast"](isolated, replacement)
                    isolated["risk"], isolated["hazards"] = forecast[1], hazards
                    isolated["default_enemy"] = default
                    alternatives[name] = namespace["planner"](isolated, obs)
                if digest(memory) != base_before:
                    raise ValueError("Law replacement mutated ordinary learned memory")
                proposals += 1
            action = namespace["planner"](memory, obs)
            reply = {"action": action, "frame": frames,
                     "memory_sha256": digest(memory),
                     "learning_updates": memory["updates"],
                     "effective_learning_updates": memory["effective_updates"]}
            if alternatives is not None:
                alternatives["learned"] = action
                checkpoint = copy.deepcopy(memory)
                reply.update(actions=alternatives, checkpoint_sha256=digest(checkpoint),
                             intervention="Only first-action recommendation changes; every branch restores the same learned planner-memory checkpoint.")
            elif request.get("save_checkpoint"):
                checkpoint = copy.deepcopy(memory)
                reply["checkpoint_sha256"] = digest(checkpoint)
        print(json.dumps(reply, allow_nan=False), flush=True)
        frames += 1
    raise RuntimeError("Branch worker ended without explicit finish")


if __name__ == "__main__":
    main()
