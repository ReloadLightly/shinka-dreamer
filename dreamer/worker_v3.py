"""JSON-lines worker; source loaded as text before restricting execution."""
import sys
# -s -S honors the explicit hash seed but loads neither user nor system sites.
# Remove script/cwd/import environment paths before importing anything else.
sys.path[:] = ["/usr/lib/python3.10", "/usr/lib/python3.10/lib-dynload"]
import contextlib
import json
import os
import random

# Explicitly import only the trusted boundary, then remove its path.
sys.path.insert(0, os.path.dirname(__file__))
from isolation import restrict
sys.path.pop(0)


def main():
    with open(sys.argv[1]) as f:
        source = f.read(512 * 1024 + 1)
    if len(source) > 512 * 1024:
        raise ValueError("Candidate exceeds 512 KiB")
    restrict()
    # Independent agent stream, initialized BEFORE any candidate top-level code.
    # The one-way tagged candidate stream is paired across programs/regimes.
    # Clear argv before source execution; no world seed crosses the boundary.
    candidate_seed = int(sys.argv[2])
    sys.argv[:] = ["candidate.py"]
    random.seed(candidate_seed)
    namespace = {"__name__": "candidate"}
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(source, "candidate.py", "exec"), namespace)
    memory, last = None, None
    for line in sys.stdin:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            obs = request["obs"]
            memory = namespace["world_model_step"](memory, obs, last)
            action = namespace["planner"](memory, obs)
            # Export after planner but BEFORE environment transition.
            exported = namespace.get("export_model", lambda m, o: {})(memory, obs)
        print(json.dumps({"action": action, "model": exported}, allow_nan=False), flush=True)
        last = action


if __name__ == "__main__":
    main()
