"""Same-experience prediction assessment for the reviewed v3 comparators.

Final assessment calls evaluate_trace on an existing immutable-memory trace,
inside its frozen driver. This CLI only runs a small development smoke test.
Neither this helper nor its worker draws or reads final-assessment pools.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import selectors
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.evaluation_v3 import Candidate, add_stat, decode, run_episode
from dreamer.provenance import sha256
from dreamer.world import stream_seed
from dreamer.world_v3 import UnknownDynamicsMaze

SHADOW_WORKER = ROOT / "dreamer/worker_v3_shadow.py"
CONDITIONS = ("fitted_frozen", "fitted_online", "known_law")
ADAPTER = "directional-v3-no-planner"
APPROVED_COMPARATOR_SHA256 = "a8dec56f17a61a05981d619b075475068ab8469de871fd44de76ac276c2eca78"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Shadow(Candidate):
    def __init__(self, path, candidate_seed):
        self.process = subprocess.Popen(
            ["/usr/bin/python3", "-s", "-S", str(SHADOW_WORKER), str(Path(path).resolve()), str(candidate_seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"}, cwd="/tmp", bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""


def evaluate_trace(trace, seed, regime, case, source, expected_source_sha256=None, adapter=ADAPTER, policy_reason=None):
    """Score three isolated predictors on the exact supplied memory trajectory.

    Score horizon-one union enemy occupancy in coordinates relative to the
    episode origin. Near targets are the PRE-action agent-centered 3x3; audit
    targets use the evaluator's independent tagged stream; destination is the
    actual post-action position. Destination occupancy excludes entry contact,
    so is not total collision risk. All transitions, including terminal ones,
    are scored from evaluator-owned labels. Those labels never enter a worker;
    learning uses only the next ordinary observation if one exists.

    Calls are isolated per shadow. If one shadow fails, it retains its episode
    and every target with a declared .5 missing-forecast fallback. Its error and
    missing counts are explicit. The caller's policy row is never changed.
    """
    if adapter != ADAPTER:
        raise ValueError("Selected evolved programs need an inspected, separately frozen replay adapter")
    source = Path(source)
    source_sha = sha256(source)
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        raise ValueError("Frozen comparator source changed")
    # This helper deliberately does not pretend to replay an arbitrary evolved
    # planner. Check the precise already-reviewed comparator, even if copied.
    if source_sha != APPROVED_COMPARATOR_SHA256:
        raise ValueError("This adapter is only approved for the manually engineered directional comparator")
    if any(frame["action"].get("interact") is not True for frame in trace):
        raise ValueError("Comparator door conditioning requires the recorded policy to interact")
    started, cpu_started = time.monotonic(), time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    law_reference = UnknownDynamicsMaze(seed, regime=regime)
    audit_rng = random.Random(stream_seed(seed, "v3:forecast-targets"))
    # Targets remain in the evaluator, independent of predictions and laws.
    targets = []
    for index, frame in enumerate(trace):
        ax, ay = frame["world"]["agent"]
        near = [(ax + dx, ay + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        audit = [(audit_rng.randrange(1, 14), audit_rng.randrange(1, 14)) for _ in range(12)]
        if index + 1 < len(trace):
            destination = tuple(trace[index + 1]["world"]["agent"])
        else:
            # Derive the final destination from the recorded snapshot using
            # the same blocked/door/corner rules. No random rollout is added.
            world = frame["world"]
            agent_pos = tuple(world["agent"])
            dx, dy = frame["action"]["move"]
            grid = world["grid"]
            keys, door_open = world["keys"], world["door_open"]
            def blocked(p):
                x, y = p
                if not (0 <= x < 15 and 0 <= y < 15):
                    return True
                cell = grid[y][x]
                if cell == 3 and not door_open:
                    opens = keys == 2 and abs(x-agent_pos[0]) + abs(y-agent_pos[1]) == 1
                    return not opens
                return cell == 1
            destination = (agent_pos[0] + dx, agent_pos[1] + dy)
            if blocked(destination) or (dx and dy and
                    (blocked((agent_pos[0] + dx, agent_pos[1])) or blocked((agent_pos[0], agent_pos[1] + dy)))):
                destination = agent_pos
        danger = any(max(abs(ex-ax), abs(ey-ay)) <= 2 for ex, ey in frame["world"]["enemies"])
        post = law_reference.switch_step is not None and frame["obs"]["step"] + 1 >= law_reference.switch_step
        targets.append((near, audit, destination, danger, post))
    shadows = {}
    for condition in CONDITIONS:
        condition_started = time.monotonic()
        stats, bins, error, error_step = {}, {}, None, None
        parameters, map_positions, observations = [], [], []
        last_action, missing, processed = None, 0, 0
        worker = None
        try:
            worker = Shadow(source, stream_seed(seed, "v3:candidate"))
        except Exception as exc:
            error, error_step = f"{type(exc).__name__}: {exc}", 0
        for index, (frame, target) in enumerate(zip(trace, targets)):
            observation = dict(frame["obs"], learn=condition != "fitted_frozen", predictive_planning=True)
            observation.pop("known_law", None)
            if condition == "known_law":
                observation["known_law"] = list(law_reference.law_for_transition(observation["step"] + 1))
            forecast, default, model = {}, .5, {}
            if error is None:
                try:
                    reply = worker.query({"obs": observation, "last_action": last_action})
                    if reply.get("passive_shadow") is not True:
                        raise ValueError("Not a passive replay worker")
                    model = reply["model"]
                    forecast, terrain, default = decode(model)
                    processed += 1
                except Exception as exc:
                    error, error_step = f"{type(exc).__name__}: {exc}", observation["step"]
            if error is not None:
                missing += 1
                forecast, default = {}, .5
            # No hidden or privileged content is included in public audit hashes
            # of ordinary-observation identity; strip known-law and control flags.
            observations.append({k: v for k, v in observation.items() if k not in ("known_law", "learn", "predictive_planning")})
            parameters.append(model.get("learning", {}).get("transition_weights"))
            map_positions.append({"position": model.get("position"), "terrain": sorted(model.get("terrain", []))})
            near, audit, destination, danger, post = target
            enemies = {tuple(p) for p in frame["next_enemies"]}
            ox, oy = frame["world"]["origin"]
            for group, points in (("near", near), ("audit", audit), ("destination", [destination])):
                for x, y in points:
                    probability = forecast.get((x-ox, y-oy), default)
                    label = float((x, y) in enemies)
                    loss = (probability-label)**2
                    add_stat(stats, "brier_" + group, loss)
                    add_stat(stats, "prevalence_" + group, label)
                    add_stat(stats, "brier_" + group + ("_post_switch" if post else "_pre_switch"), loss)
                    if group == "near":
                        add_stat(bins.setdefault(str(frame["obs"]["step"] // 25), {}), "brier", loss)
                        if danger:
                            add_stat(stats, "brier_threat", loss)
                    if index == len(trace)-1:
                        add_stat(stats, "brier_" + group + "_final_recorded", loss)
                        if policy_reason in ("escaped", "caught", "timeout"):
                            add_stat(stats, "brier_" + group + "_terminal", loss)
                    if error is None:
                        add_stat(stats, "valid_brier_" + group, loss)
            last_action = frame["action"]
        if worker is not None:
            worker.close()
        exported = bool(parameters) and all(p is not None for p in parameters)
        shadows[condition] = {
            "condition": condition, "case": case, "regime": regime,
            "stats": stats, "bins": bins, "error": error, "error_step": error_step,
            "frames": len(trace), "scored_frames": len(trace), "successful_forecast_frames": processed,
            "missing_forecast_frames": missing, "seconds": time.monotonic()-condition_started,
            "audit": {"observations_sha256": digest(observations), "map_position_sha256": digest(map_positions),
                "parameters_exported": exported,
                "parameters_constant": all(p == parameters[0] for p in parameters) if exported else None,
                "parameter_change_steps": sum(a != b for a, b in zip(parameters, parameters[1:])) if exported else None,
                "first_parameters": parameters[0] if exported else None,
                "final_parameters": parameters[-1] if exported else None}}
    return {"case": case, "regime": regime, "adapter": adapter,
            "program_sha256": source_sha, "policy_frames": len(trace),
            "policy_trajectory_sha256": digest([{k: f[k] for k in ("world", "action", "next_enemies")} for f in trace]),
            "shadows": shadows, "condition_episodes": len(CONDITIONS), "new_environment_episodes": 0,
            "model_calls": 0, "terminal_transition_included": policy_reason in ("escaped", "caught", "timeout"),
            "policy_reason": policy_reason, "missing_forecast_probability": .5,
            "post_switch_scope": "survivor-conditioned: only recorded transitions reaching the private switch",
            "switch_step": law_reference.switch_step,
            "encountered_switch": any(t[-1] for t in targets),
            "target_contract": "horizon1 union enemy occupancy; origin-relative coordinates; near previous3x3, audit12, actual destination",
            "label_visibility": "near targets become visible in the next observation if nonterminal; terminal and nonlocal audit labels are evaluator-only",
            "seconds": time.monotonic()-started, "evaluator_cpu_seconds": time.process_time()-cpu_started,
            "candidate_cpu_seconds": ((resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime-child_before.ru_utime)
                                     +(resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime-child_before.ru_stime))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", action="store_true", required=True)
    parser.add_argument("--seeds", default="results/private/v3-development-seeds.json")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--regime", default="switch", choices=("uniform", "stationary", "switch"))
    parser.add_argument("--program", default="controls/v3/directional.py")
    parser.add_argument("--out", default="results/v3-matched-smoke.json")
    args = parser.parse_args()
    if not 1 <= args.limit <= 8:
        raise ValueError("This CLI is only a development smoke test of at most eight episodes")
    seeds = json.loads(Path(args.seeds).read_text())[:args.limit]
    records = []
    for case, seed in enumerate(seeds):
        policy = run_episode(ROOT / "controls/v1/memory.py", seed, "memory", replay=True, regime=args.regime)
        result = evaluate_trace(policy.pop("trace"), seed, args.regime, case, args.program, policy_reason=policy["reason"])
        result["policy_result"] = {k: policy[k] for k in ("reason", "steps", "error", "seconds", "candidate_cpu_seconds", "evaluator_cpu_seconds")}
        records.append(result)
    output = {"development_only": True, "driver_sha256": sha256(__file__), "worker_sha256": sha256(SHADOW_WORKER),
              "pool_sha256": sha256(args.seeds), "memory_environment_episodes": len(records),
              "shadow_condition_episodes": len(records)*3, "model_calls": 0, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"memory_environment_episodes":len(records),"shadow_condition_episodes":len(records)*3,
                      "invalid_shadows":sum(r["error"] is not None for c in records for r in c["shadows"].values()),
                      "driver_sha256": output["driver_sha256"], "worker_sha256": output["worker_sha256"]}))


if __name__ == "__main__":
    main()
