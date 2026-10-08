"""Bounded development-only call profiling; never executes a world transition.

CLI: SOURCE SHA256 selected|comparator 0|1. JSONL requests contain public obs,
the preceding recorded action, and the current recorded action. Comparator
requests additionally supply a current known law for an isolated planning clone.
The ordinary memory never receives that law. End with {"finish": true}.
"""
import sys

sys.path[:] = ["/usr/lib/python3.10", "/usr/lib/python3.10/lib-dynload"]
import contextlib
import copy
import cProfile
import hashlib
import json
import math
import os
import pstats  # Preload profiler support before the filesystem restriction.
import random
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from v3_diagnostic_observer import ranked_plan
sys.path.pop(0)
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "dreamer"))
from isolation import restrict
sys.path.pop(0)

PHASES = ("world_model_step", "planner", "export_model")
COMPARATOR_SHA = "a8dec56f17a61a05981d619b075475068ab8469de871fd44de76ac276c2eca78"
SELECTED_SHA = "aafa35fd866c353c83d0b661d79f2060d678fd2060ff50a1b10ea2f0e3ef5d3c"


def canonical_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def profile_rows(profile):
    profile.create_stats()
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, callers) in profile.stats.items():
        rows.append({"file": os.path.basename(filename), "line": line,
                     "function": name, "primitive_calls": primitive,
                     "calls": calls, "self_seconds": own,
                     "cumulative_seconds": cumulative})
    return sorted(rows, key=lambda r: (-r["cumulative_seconds"],
                                      -r["self_seconds"], r["function"]))[:15]


def main():
    if len(sys.argv) != 5:
        raise ValueError("Expected SOURCE SHA256 selected|comparator 0|1")
    source_path, expected_sha, kind, profiling = sys.argv[1:]
    if kind not in ("selected", "comparator") or profiling not in ("0", "1"):
        raise ValueError("Unreviewed diagnostic adapter")
    with open(source_path, "rb") as handle:
        source_bytes = handle.read(512 * 1024 + 1)
    if len(source_bytes) > 512 * 1024:
        raise ValueError("Candidate exceeds source limit")
    actual_sha = hashlib.sha256(source_bytes).hexdigest()
    reviewed_sha = COMPARATOR_SHA if kind == "comparator" else SELECTED_SHA
    if actual_sha != expected_sha or actual_sha != reviewed_sha:
        raise ValueError("Diagnostic source differs from reviewed immutable source")
    source = source_bytes.decode("utf-8")
    profiled = profiling == "1"
    profiles = {phase: cProfile.Profile() for phase in PHASES} if profiled else {}
    totals = {phase: {"calls": 0, "wall_seconds": 0., "process_seconds": 0.}
              for phase in PHASES}
    restrict()  # Unmodified 192 MiB / 10 CPU second execution boundary.
    sys.argv[:] = ["candidate.py"]
    random.seed(0)  # Fixed diagnostic stream; no environment seed is supplied.
    namespace = {"__name__": "candidate", "__diagnostic_source_sha256__": actual_sha}
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(source, "candidate.py", "exec"), namespace)

    def measured(phase, function, *args):
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        if profiled:
            profiles[phase].enable()
        try:
            return function(*args)
        finally:
            if profiled:
                profiles[phase].disable()
            totals[phase]["calls"] += 1
            totals[phase]["wall_seconds"] += time.perf_counter() - start_wall
            totals[phase]["process_seconds"] += time.process_time() - start_cpu

    memory, frames, recorded_waits = None, 0, 0
    worker_start_wall, worker_start_cpu = time.perf_counter(), time.process_time()
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("finish") is True:
            output = {"finished": True, "frames": frames, "source_sha256": actual_sha,
                      "kind": kind, "profiled": profiled, "phase_totals": totals,
                      "profiles": {phase: profile_rows(profile)
                                   for phase, profile in profiles.items()},
                      "worker_wall_seconds": time.perf_counter() - worker_start_wall,
                      "worker_process_seconds": time.process_time() - worker_start_cpu,
                      "timing_scope": "Main WM/planner/export only; excludes known-law clone and recorded-action alignment. Unprofiled comparator planner includes ranking trace overhead.",
                      "adapter": "Recorded public observations and preceding recorded actions; predictive_planning enabled for both sources. Comparator fitted parameters frozen, waits and forecast reset after export to the recorded action. Known-law clone discarded each frame; selected predictive learning enabled."}
            print(json.dumps(output, allow_nan=False), flush=True)
            return
        obs = dict(request["obs"])
        if "known_law" in obs:
            raise ValueError("Ordinary public observation contains privileged law")
        if kind == "selected" and "known_law" in request:
            raise ValueError("Selected adapter must not receive a privileged law")
        obs["learn"] = kind == "selected"
        # The corpus was recorded under the memory policy, whose observation
        # flag disables predictive planning. This diagnostic explicitly tests
        # each source's predictive planner on the same physical observations.
        obs["predictive_planning"] = True
        last_action = request["last_action"]
        recorded_action = request["recorded_action"]
        known_action, ranking = None, None
        with contextlib.redirect_stdout(sys.stderr):
            memory = measured("world_model_step", namespace["world_model_step"],
                              memory, obs, last_action)
            known_memory = None
            if kind == "comparator":
                law = request["known_law"]
                if (not isinstance(law, list) or len(law) != 9 or
                        any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0
                            for p in law) or abs(sum(law) - 1.) > 1e-9):
                    raise ValueError("Invalid current-law diagnostic input")
                known_memory = copy.deepcopy(memory)
                known_memory["known_law"] = list(law)
                namespace["_record_forecast"](known_memory, known_memory["pos"])
            if kind == "comparator" and not profiled:
                action, baseline_ranking = measured("planner", ranked_plan,
                                                    namespace, memory, obs)
                known_action, known_ranking = ranked_plan(namespace, known_memory, obs)
                ranking = {"baseline": baseline_ranking, "known_law": known_ranking}
            else:
                action = measured("planner", namespace["planner"], memory, obs)
                if known_memory is not None:
                    known_action = namespace["planner"](known_memory, obs)
            exported = measured("export_model", namespace["export_model"], memory, obs)
            output_hash = canonical_digest({"action": action, "model": exported})
            if kind == "comparator":
                recorded_waits = recorded_waits + 1 if recorded_action["move"] == [0, 0] else 0
                memory["waits"] = recorded_waits
                destination = namespace["_add"](memory["pos"], recorded_action["move"])
                namespace["_record_forecast"](memory, destination)
                if memory.get("known_law") is not None:
                    raise ValueError("Privileged clone leaked into ordinary memory")
        output = {"frame": frames, "action_model_sha256": output_hash,
                  "action": action, "known_action": known_action}
        if ranking is not None:
            output["ranking"] = ranking
        print(json.dumps(output, allow_nan=False), flush=True)
        frames += 1
    raise RuntimeError("Input ended before explicit finish request")


if __name__ == "__main__":
    main()
