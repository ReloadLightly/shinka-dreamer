"""Trusted, streaming evaluator. Candidate metrics are never accepted."""
import json
import math
import os
from pathlib import Path
import random
import selectors
import subprocess
import time

from .world import DynamicMaze, stream_seed, valid_action

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "namazu-repair-predictive-v1"
OBJECTIVE = "task06-forecast04-v1"


class Candidate:
    def __init__(self, path):
        self.process = subprocess.Popen(
            ["/usr/bin/python3", "-I", str(ROOT / "dreamer/worker.py"), str(Path(path).resolve())],
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


def run_episode(program_path, seed, variant="predictive", replay=False, interval=25):
    env = DynamicMaze(seed, interval=interval)
    audit = random.Random(stream_seed(seed, "audit"))
    trace, bins, stats = [], {}, {}
    latest_model = {}
    error = None
    started = time.monotonic()
    # Separate agent RNG: index is assigned by the experiment, not world state.
    # Its constant reproducible stream cannot reveal an environment seed.
    with Candidate(program_path) as candidate:
        while not env.done:
            obs = env.observe()
            obs["learn"] = variant not in ("frozen", "memory", "frozen_no_planning")
            obs["predictive_planning"] = variant not in ("no_planning", "memory", "frozen_no_planning")
            try:
                request = {"obs": obs}
                if env.steps == 0:
                    request["seed"] = 712934
                reply = candidate.query(request)
                latest_model = reply.get("model", {})
                forecast, terrain, default = decode(latest_model)
                action = reply["action"]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                env.done, env.reason = True, "invalid"
                break
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
                saved = json.loads(json.dumps({"world": env.snapshot(), "model": latest_model, "action": action}))
            time_bin = str(env.steps // 25)
            env.step(action)
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
    result = {"seed": seed, "variant": variant, "reason": env.reason, "steps": env.steps,
              "keys": env.keys_collected, "door": env.door_open, "task": task,
              "model_score": model_score, "combined_score": .6*task + .4*model_score,
              "stats": stats, "bins": bins, "error": error,
              "learning": latest_model.get("learning", {}), "seconds": time.monotonic()-started}
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
              "steps": avg("steps"), "model_score": avg("model_score"), **stats}
    escaped = [r["steps"] for r in rows if r["reason"] == "escaped"]
    public["escape_steps"] = sum(escaped)/len(escaped) if escaped else None
    return {"combined_score": avg("combined_score"), "public": public,
            "text_feedback": (f"{PROTOCOL}; {OBJECTIVE}. {n} paired development episodes. "
                              f"Outcomes and pre-outcome horizon-1 enemy occupancy errors: {json.dumps(public)}. "
                              "Improve both representation/learning and planning; all helpers may evolve. "
                              "Do not infer learning from map accumulation or scalar fitness alone.")}
