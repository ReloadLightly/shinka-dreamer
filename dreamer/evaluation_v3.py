"""Trusted, streaming evaluator. Candidate metrics are never accepted."""
import json
import math
import os
from pathlib import Path
import random
import resource
import platform
import selectors
import subprocess
import time

from .world import stream_seed, valid_action
from .world_v3 import UnknownDynamicsMaze, REGIMES
from .provenance import sha256, control_path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "namazu-unknown-dynamics-v3"
OBJECTIVE = "absolute-task-v1"


class Candidate:
    def __init__(self, path, candidate_seed=712934):
        self.process = subprocess.Popen(
            ["/usr/bin/python3", "-s", "-S", str(ROOT / "dreamer/worker_v3.py"), str(Path(path).resolve()), str(candidate_seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"}, cwd="/tmp", bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""

    def query(self, request):
        self.process.stdin.write((json.dumps(request, allow_nan=False) + "\n").encode())
        deadline = time.monotonic() + 3
        while b"\n" not in self.buffer:
            if not self.selector.select(max(0, deadline-time.monotonic())):
                raise TimeoutError("Candidate exceeded 3 s response deadline")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"Candidate worker exited ({self.process.poll()})")
            self.buffer += chunk
            if len(self.buffer) > 2 * 1024**2:
                raise ValueError("Candidate response exceeds 2 MiB")
        line, self.buffer = self.buffer.split(b"\n", 1)
        result = json.loads(line)
        if not isinstance(result, dict) or not valid_action(result.get("action")):
            raise ValueError("Malformed candidate action")
        return result

    def close(self):
        self.process.kill() if self.process.poll() is None else None
        self.process.wait()
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def probability(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid forecast probability")
    return float(value)


def decode(model):
    if not isinstance(model, dict):
        raise ValueError("Model export must be a dict")
    enemy, terrain = {}, {}
    for row in model.get("enemy", []):
        x, y, p = row
        if type(x) is not int or type(y) is not int or abs(x) > 30 or abs(y) > 30:
            raise ValueError("Invalid model coordinate")
        enemy[x, y] = probability(p)
    for row in model.get("terrain", []):
        x, y, cell, step = row
        if any(type(v) is not int for v in row) or cell not in range(-1, 5):
            raise ValueError("Invalid terrain export")
        terrain[x, y] = (cell, step)
    default = probability(model.get("default_enemy", .5))
    return enemy, terrain, default


def run_episode(program_path, seed, variant="predictive", replay=False, interval=25, regime="stationary"):
    # These controls are original programs, never flags applied to a descendant.
    if variant in ("memory", "original_predictive"):
        from .provenance import control_path
        program_path = control_path("memory" if variant == "memory" else "predictive")
    env = UnknownDynamicsMaze(seed, interval=interval, regime=regime)
    audit = random.Random(stream_seed(seed, "v3:forecast-targets"))
    trace, bins, stats = [], {}, {}
    latest_model = {}
    error = None
    started = time.monotonic()
    cpu_started = time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    exposure = {"visible_enemy_steps": 0, "visible_enemy_cells": 0,
                "consecutive_visible_steps": 0, "post_switch_steps": 0,
                "post_switch_visible_enemy_steps": 0, "post_switch_consecutive_visible_steps": 0}
    previous_visible = False
    # The worker initializes its independent stream before executing the program.
    with Candidate(program_path, stream_seed(seed, "v3:candidate")) as candidate:
        while not env.done:
            obs = env.observe()
            if variant == "known_law":
                obs["known_law"] = list(env.current_law)
            post_switch = env.switch_step is not None and env.steps + 1 >= env.switch_step
            visible_cells = sum(cell == 5 for row in obs["grid"] for cell in row)
            obs["learn"] = variant not in ("frozen", "memory", "frozen_no_planning")
            obs["predictive_planning"] = variant not in ("no_planning", "memory", "frozen_no_planning")
            try:
                request = {"obs": obs}
                reply = candidate.query(request)
                latest_model = reply.get("model", {})
                forecast, terrain, default = decode(latest_model)
                action = reply["action"]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                env.done, env.reason = True, "invalid"
                break
            # Counts successful action opportunities, excluding failed worker queries.
            exposure["visible_enemy_steps"] += bool(visible_cells)
            exposure["visible_enemy_cells"] += visible_cells
            exposure["consecutive_visible_steps"] += bool(visible_cells and previous_visible)
            if post_switch:
                exposure["post_switch_steps"] += 1
                exposure["post_switch_visible_enemy_steps"] += bool(visible_cells)
                # At the first replacement transition no new-law outcome exists yet.
                if env.steps >= env.switch_step:
                    exposure["post_switch_consecutive_visible_steps"] += bool(visible_cells and previous_visible)
            previous_visible = bool(visible_cells)
            ox, oy = env.origin
            ax, ay = env.agent_pos
            # The selected targets never cross the candidate boundary.
            near = [(ax+dx, ay+dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            uniform = [(audit.randrange(1, 14), audit.randrange(1, 14)) for _ in range(12)]
            dangerous = any(max(abs(ex-ax), abs(ey-ay)) <= 2 for ex, ey in env.enemies)
            observations = {}
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    observations[ax+dx, ay+dy] = obs["grid"][dy+2][dx+2] == 5
            # Reconstruction targets/ages chosen and tracked by evaluator.
            if env.steps == 0:
                seen = {}
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    p = ax+dx, ay+dy
                    if 0 <= p[0] < 15 and 0 <= p[1] < 15:
                        seen[p] = env.steps
            for p in uniform:
                belief = terrain.get((p[0]-ox, p[1]-oy))
                add_stat(stats, "coverage", float(belief is not None))
                add_stat(stats, "reconstruction", float(belief is not None and belief[0] == env.grid[p[1]][p[0]]))
                category = "unseen" if p not in seen else "visible" if p in observations else "stale"
                add_stat(stats, "reconstruction_" + category, float(belief is not None and belief[0] == env.grid[p[1]][p[0]]))
                if p in seen:
                    add_stat(stats, "map_age", env.steps-seen[p])
            saved = None
            if replay:
                saved = json.loads(json.dumps({"world": env.snapshot(), "obs": obs, "model": latest_model, "action": action}))
            time_bin = str(env.steps // 25)
            env.step(action)
            # Separate action-dependent diagnostic: post-transition occupancy at the
            # actual destination. This excludes pre-move contact and is NOT a
            # calibrated estimate of total collision/death probability.
            destination_pred = forecast.get((env.agent_pos[0]-ox, env.agent_pos[1]-oy), default)
            destination_label = float(env.agent_pos in env.enemies)
            add_stat(stats, "brier_destination", (destination_pred-destination_label)**2)
            add_stat(stats, "prevalence_destination", destination_label)
            for group, targets in (("near", near), ("audit", uniform)):
                for p in targets:
                    label = float(p in env.enemies)
                    relative = p[0]-ox, p[1]-oy
                    pred = forecast.get(relative, default)
                    # Matched persistence comparator: 0/1 if visible, base .02 otherwise.
                    persistence = float(observations[p]) if p in observations else .02
                    if variant == "memory":
                        pred = persistence
                    brier = (pred-label)**2
                    add_stat(stats, "brier_" + group, brier)
                    add_stat(stats, "brier_" + group + ("_post_switch" if post_switch else "_pre_switch"), brier)
                    add_stat(stats, "persistence_" + group, (persistence-label)**2)
                    add_stat(stats, "base_" + group, (.02-label)**2)
                    add_stat(stats, "prevalence_" + group, label)
                    if group == "near" and dangerous:
                        add_stat(stats, "brier_threat", brier)
                        add_stat(stats, "persistence_threat", (persistence-label)**2)
                    if group == "near":
                        add_stat(bins.setdefault(time_bin, {}), "brier", brier)
                        add_stat(bins[time_bin], "persistence", (persistence-label)**2)
            if saved is not None:
                saved["next_enemies"] = env.enemies
                trace.append(saved)
    # Escape dominates; no positive reward for simply waiting. All terms [0,1].
    task = (.65 * (env.reason == "escaped") + .10 * env.keys_collected
            + .10 * env.door_open + .05 * (env.reason == "escaped") * (1-env.steps/env.max_steps))
    model_score = 1 - (mean_stat(stats, "brier_near", .25) + mean_stat(stats, "brier_audit", .25)) / 2
    result = {"seed": seed, "regime": regime, "variant": variant, "reason": env.reason, "steps": env.steps,
              "keys": env.keys_collected, "door": env.door_open, "task": task,
              "model_score": model_score, "combined_score": task,
              "stats": stats, "bins": bins, "error": error,
              "learning": latest_model.get("learning", {}), "seconds": time.monotonic()-started,
              "evaluator_cpu_seconds": time.process_time()-cpu_started,
              "candidate_cpu_seconds": ((resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - child_before.ru_utime)
                                        + (resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime - child_before.ru_stime)),
              "switch_step": env.switch_step,
              "encountered_switch": env.switch_step is not None and env.steps >= env.switch_step,
              "enemy_innovation_draws": env.enemy_rng.draws, "exposure": exposure}
    if error:
        result["combined_score"] = 0.
    if replay:
        result["trace"] = trace
    return result


def add_stat(stats, name, value):
    acc = stats.setdefault(name, [0., 0])
    acc[0] += value
    acc[1] += 1


def mean_stat(stats, name, default=None):
    total, n = stats.get(name, (0, 0))
    return total / n if n else default


def aggregate(rows):
    if not rows:
        raise ValueError("No episodes to aggregate")
    n = len(rows)
    avg = lambda name: sum(row[name] for row in rows) / n
    stats = {name: sum(r["stats"].get(name, [0, 0])[0] for r in rows) /
             max(1, sum(r["stats"].get(name, [0, 0])[1] for r in rows))
             for name in sorted({k for r in rows for k in r["stats"]})}
    public = {"episodes": n, "escape": sum(r["reason"] == "escaped" for r in rows)/n,
              "death": sum(r["reason"] == "caught" for r in rows)/n,
              "timeout": sum(r["reason"] == "timeout" for r in rows)/n,
              "invalid": sum(r["error"] is not None for r in rows),
              "task": avg("task"), "keys": avg("keys"), "door": avg("door"),
              "steps": avg("steps"), "model_score": avg("model_score"),
              "seconds": avg("seconds"), "candidate_cpu_seconds": avg("candidate_cpu_seconds"),
              "evaluator_cpu_seconds": avg("evaluator_cpu_seconds"), **stats}
    escaped = [r["steps"] for r in rows if r["reason"] == "escaped"]
    public["escape_steps"] = sum(escaped)/len(escaped) if escaped else None
    public["regimes"] = {}
    for regime in REGIMES:
        subset = [r for r in rows if r["regime"] == regime]
        if subset:
            public["regimes"][regime] = {
                "episodes": len(subset),
                "escape": sum(r["reason"] == "escaped" for r in subset)/len(subset),
                "task": sum(r["combined_score"] for r in subset)/len(subset),
                "encountered_switch": sum(r["encountered_switch"] for r in subset),
                "post_switch_steps": sum(r["exposure"]["post_switch_steps"] for r in subset),
                "post_switch_visible_enemy_steps": sum(r["exposure"]["post_switch_visible_enemy_steps"] for r in subset)}
    return {"combined_score": avg("combined_score"), "public": public,
            "text_feedback": (f"{PROTOCOL}; {OBJECTIVE}. {n} paired development condition-episodes. "
                              f"Absolute task fitness; prediction losses are diagnostics: {json.dumps(public)}. "
                              "Improve escape/keys/door and escape efficiency under unknown stationary and switching "
                              "attempted movement laws; blocked enemy attempts stay. Predictive feedback is horizon-1 "
                              "occupancy at fixed cells in starting-agent coordinates: near targets surround the OLD "
                              "agent position, audit targets sample interior cells; predictions precede outcomes. "
                              "Threat losses concern currently visible nearby enemies; audit prevalence is sparse. "
                              "All representations, memory, updating, helpers and planning may evolve. "
                              "Do not infer adaptive prediction from map accumulation, or control benefit from lower Brier.")}


def evaluation_identity():
    paths = ["evaluate_v3.py", "dreamer/evaluation_v3.py", "dreamer/world_v3.py", "dreamer/world.py",
             "dreamer/worker_v3.py", "dreamer/isolation.py", "dreamer/bridge_v3.py", "dreamer/provenance.py"]
    return {"protocol": PROTOCOL, "objective": OBJECTIVE,
            "objective_formula": ".65*escape+.10*keys+.10*door+.05*escape*(1-steps/200); invalid=0",
            "regimes": list(REGIMES), "regime_weights": [1/3]*3,
            "law_distribution": {"dirichlet_alpha": [1]*9, "entropy_min_nats": 1.2,
                                 "max_component": .6, "switch_min_tv": .3,
                                 "switch_first_transition_inclusive": [25, 75]},
            "rng": {"layout": "layout (unchanged v2 generation)",
                    "initial_law": "v3:law:initial", "replacement_law": "v3:law:replacement",
                    "switch_time": "v3:switch-time", "enemy_innovations": "v3:enemy-innovations",
                    "forecast_targets": "v3:forecast-targets", "candidate": "v3:candidate",
                    "enemy_draws_per_tick": 3, "sampler": "one uniform innovation, inverse CDF per enemy"},
            "source_sha256": {p: sha256(ROOT / p) for p in paths},
            "controls": {n: sha256(control_path(n)) for n in ("predictive", "memory")},
            "runtime": {"evaluator_python": platform.python_version(),
                        "candidate_executable": "/usr/bin/python3",
                        "candidate_executable_sha256": sha256("/usr/bin/python3"),
                        "candidate_args": ["-s", "-S"], "hash_seed": 0,
                        "agent_random_seed_before_exec": "stream_seed(case, 'v3:candidate')",
                        "limits": {"address_space_mib": 192, "cpu_seconds": 10,
                                   "reply_seconds": 3, "source_kib": 512, "reply_mib": 2,
                                   "file_descriptors": 64},
                        "isolation": "Landlock + seccomp, unchanged from v2"}}
