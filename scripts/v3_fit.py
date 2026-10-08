"""Fit the manually engineered direction-aware v3 comparator on development only.

The saved corpus contains candidate-visible observations and executed actions.
Training never reads hidden laws, identities, positions, or assessment cases.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.evaluation_v3 import run_episode

REGIMES = ("uniform", "stationary", "switch")
REGULARIZATION = (0.0, 0.001, 0.01)
LEARNING_RATES = (0.5, 2.0, 8.0)
TEMPLATE = ROOT / "controls/v3/directional_template.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text)
    temporary.replace(path)


def load_module(path=TEMPLATE):
    spec = importlib.util.spec_from_file_location("directional_comparator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_one(job):
    path, seed, regime, index = job
    path = Path(path)
    if path.exists():
        with gzip.open(path, "rt") as handle:
            row = json.load(handle)
        assert row["case"] == index and row["regime"] == regime
        return row["summary"]
    row = run_episode(ROOT / "controls/v1/memory.py", seed, "memory", replay=True, regime=regime)
    if row["error"]:
        raise RuntimeError(f"Fixed corpus policy failed: {row['error']}")
    corpus = {"case": index, "regime": regime,
              "frames": [{"obs": f["obs"], "action": f["action"]} for f in row["trace"]],
              "summary": {"case": index, "regime": regime, "reason": row["reason"],
                          "steps": row["steps"], "seconds": row["seconds"]}}
    temporary = path.with_suffix(".partial")
    with gzip.open(temporary, "wt") as handle:
        json.dump(corpus, handle, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)
    return corpus["summary"]


def corpus_rows(directory):
    for path in sorted(Path(directory).glob("*.json.gz")):
        with gzip.open(path, "rt") as handle:
            yield json.load(handle)


def visible_next(frame, following, position):
    obs = following["obs"]
    dx, dy = obs["feedback"]["displacement"]
    center = (position[0] + dx, position[1] + dy)
    terrain, enemies = {}, set()
    for y in range(5):
        for x in range(5):
            p = (center[0] + x - 2, center[1] + y - 2)
            terrain[p] = obs["terrain"][y][x]
            if obs["grid"][y][x] == 5:
                enemies.add(p)
    return terrain, enemies


def fitting_arrays(directory):
    import numpy as np
    agent = load_module()
    masks, labels = [], []
    counts = {"episodes": 0, "transitions": 0, "targets": 0, "informative_targets": 0}
    for row in corpus_rows(directory):
        memory, last = None, None
        counts["episodes"] += 1
        for frame, following in zip(row["frames"], row["frames"][1:]):
            obs = dict(frame["obs"], learn=False)
            memory = agent.world_model_step(memory, obs, last)
            last = frame["action"]
            terrain, enemies = visible_next(frame, following, memory["pos"])
            origins = sorted(memory["enemies"])
            destinations = [agent._attempt_destinations(memory, e) for e in origins]
            counts["transitions"] += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = (memory["pos"][0] + dx, memory["pos"][1] + dy)
                    if (p not in terrain or terrain[p] in (-1, 1, 3) or
                            memory["terrain"].get(p, 1) in (-1, 1, 3)):
                        continue
                    mask = [[float(q == p) for q in ds] for ds in destinations]
                    mask += [[0.] * 9 for _ in range(3 - len(mask))]
                    masks.append(mask)
                    labels.append(float(p in enemies))
                    counts["targets"] += 1
                    counts["informative_targets"] += any(any(a) for a in mask)
    return np.asarray(masks, dtype=float), np.asarray(labels, dtype=float), counts


def objective(theta, masks, labels, regularization):
    import numpy as np
    probability = np.exp(theta - np.max(theta))
    probability /= probability.sum()
    branch = masks @ probability
    absence = 1. - branch
    prediction = 1. - absence.prod(axis=1)
    error = prediction - labels
    derivative_probability = np.zeros_like(masks[:, 0, :])
    for j in range(3):
        other = absence[:, [k for k in range(3) if k != j]].prod(axis=1)
        derivative_probability += other[:, None] * masks[:, j, :]
    gradient_probability = (2. * error[:, None] * derivative_probability).mean(axis=0)
    gradient = probability * (gradient_probability - gradient_probability @ probability)
    loss = float((error ** 2).mean()) + regularization * float((theta ** 2).mean())
    gradient += 2. * regularization * theta / 9.
    return loss, gradient


def replay_predictor(agent, row, learn, learning_rate=None, with_frames=False):
    """Identical observed experience; forecasts are before each next observation.

    The action is the recorded policy action. Current visible observations and
    prior predicted occupancy continue to update state in BOTH frozen and online
    conditions; only the transition-logit gradient update is disabled.
    Terminal transitions lack a subsequent observation and are excluded here.
    """
    if learning_rate is not None:
        agent._LEARNING_RATE = learning_rate
    memory, last = None, None
    total, count, informative = 0., 0, 0
    parameter_changes, frames = 0, []
    for frame, following in zip(row["frames"], row["frames"][1:]):
        obs = dict(frame["obs"], learn=learn)
        memory = agent.world_model_step(memory, obs, last)
        action = frame["action"]
        destination = agent._add(memory["pos"], action["move"])
        agent._record_forecast(memory, destination)
        terrain, enemies = visible_next(frame, following, memory["pos"])
        loss, n = 0., 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p = (memory["pos"][0] + dx, memory["pos"][1] + dy)
                if p not in terrain:
                    continue
                loss += (memory["risk"].get(p, .02) - float(p in enemies)) ** 2
                n += 1
        informative += bool(memory["enemies"])
        total += loss
        count += n
        parameter_changes += memory["theta"] != list(agent._PRIOR)
        if with_frames:
            frames.append({"step": obs["step"], "loss_sum": loss, "targets": n,
                           "parameters": list(memory["theta"]),
                           "position": list(memory["pos"]),
                           "map": sorted((p, c) for p, c in memory["map"].items())})
        last = action
    result = {"case": row["case"], "regime": row["regime"], "loss_sum": total,
              "targets": count, "visible_enemy_frames": informative,
              "changed_parameter_frames": parameter_changes,
              "parameter_updates": memory["parameter_updates"] if memory else 0,
              "first_parameters": list(agent._PRIOR),
              "final_parameters": memory["theta"] if memory else list(agent._PRIOR)}
    if with_frames:
        result["frames"] = frames
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seeds", default="results/private/v3-fit-seeds.json")
    parser.add_argument("--validation-seeds", default="results/private/v3-fit-validation-seeds.json")
    parser.add_argument("--out", default="results/v3-comparator-fit")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--source-out", default="controls/v3/directional.py")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_seeds = json.loads(Path(args.train_seeds).read_text())
    validation_seeds = json.loads(Path(args.validation_seeds).read_text())
    assert not set(train_seeds) & set(validation_seeds)
    plan = {"purpose": "manually engineered ordinary comparator; not evolution seed",
            "training_layouts": len(train_seeds), "fit_subvalidation_layouts": len(validation_seeds),
            "regimes": REGIMES, "regime_weights": [1/3] * 3,
            "train_pool_sha256": sha(args.train_seeds),
            "fit_subvalidation_pool_sha256": sha(args.validation_seeds),
            "corpus_policy_sha256": sha(ROOT / "controls/v1/memory.py"),
            "template_sha256": sha(TEMPLATE), "driver_sha256": sha(__file__),
            "fit": "nine absolute attempted-direction logits; blocked attempts stay; anonymous occupancy Brier",
            "regularization_candidates": REGULARIZATION, "online_rate_candidates": LEARNING_RATES,
            "optimizer": "L-BFGS-B, maxiter 150, theta bounds [-5,5], zero start",
            "selection": "lowest subvalidation proper Brier, ties prefer smaller regularization/rate",
            "fit_targets": "next visible labels at previous inner 3x3 traversable cells; no terminal labels",
            "online_rate_targets": "same observation-only replay, previous inner 3x3 including walls",
            "model_calls": 0}
    plan_path = out / "fit-plan.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == json.loads(json.dumps(plan))
    else:
        save(plan_path, plan)
    started = time.monotonic()
    collection = {}
    for split, seeds in (("train", train_seeds), ("validation", validation_seeds)):
        directory = out / split
        directory.mkdir(exist_ok=True)
        jobs = [(str(directory / f"{regime}-{i:04d}.json.gz"), seed, regime, i)
                for regime in REGIMES for i, seed in enumerate(seeds)]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(collect_one, jobs):
                rows.append(row)
                if len(rows) % 24 == 0 or len(rows) == len(jobs):
                    print(json.dumps({"stage": "corpus", "split": split, "episodes": len(rows),
                                      "of": len(jobs), "elapsed_seconds": time.monotonic()-started}), flush=True)
        collection[split] = rows
    import numpy as np
    from scipy.optimize import minimize
    x, y, train_counts = fitting_arrays(out / "train")
    vx, vy, validation_counts = fitting_arrays(out / "validation")
    fits = []
    for regularization in REGULARIZATION:
        result = minimize(objective, np.zeros(9), args=(x, y, regularization), jac=True,
                          method="L-BFGS-B", bounds=[(-5., 5.)] * 9,
                          options={"maxiter": 150, "ftol": 1e-13, "gtol": 1e-9})
        theta = result.x - result.x.mean()
        fits.append({"regularization": regularization, "theta": theta.tolist(),
                     "train_brier": objective(theta, x, y, 0.)[0],
                     "validation_brier": objective(theta, vx, vy, 0.)[0],
                     "iterations": int(result.nit), "success": bool(result.success)})
    best = min(fits, key=lambda f: (f["validation_brier"], f["regularization"]))
    agent = load_module()
    agent._PRIOR = best["theta"]
    replays = list(corpus_rows(out / "validation"))
    online = []
    matched = {}
    for rate in (0.,) + LEARNING_RATES:
        rows = [replay_predictor(agent, row, rate != 0., rate) for row in replays]
        matched[str(rate)] = rows
        online.append({"rate": rate, "loss_sum": sum(r["loss_sum"] for r in rows),
                       "targets": sum(r["targets"] for r in rows),
                       "brier": sum(r["loss_sum"] for r in rows) / sum(r["targets"] for r in rows)})
        print(json.dumps({"stage": "online_subvalidation", **online[-1]}), flush=True)
    best_rate = min((r for r in online if r["rate"] > 0.), key=lambda r: (r["brier"], r["rate"]))
    source = TEMPLATE.read_text().replace("_PRIOR = [0.0] * 9  # FITTED_PRIOR",
                 "_PRIOR = " + repr(best["theta"]) + "  # FITTED_PRIOR")
    source = source.replace("_LEARNING_RATE = 2.0  # FITTED_RATE",
                            "_LEARNING_RATE = " + repr(best_rate["rate"]) + "  # FITTED_RATE")
    Path(args.source_out).write_text(source)
    summary = {"plan": plan, "train_counts": train_counts, "validation_counts": validation_counts,
               "frozen_fits": fits, "selected_regularization": best["regularization"],
               "selected_prior": best["theta"], "online_rate_results": online,
               "selected_online_rate": best_rate["rate"], "program_sha256": sha(args.source_out),
               "elapsed_seconds": time.monotonic()-started,
               "corpus_condition_episodes": len(train_seeds)*3 + len(validation_seeds)*3,
               "corpus_steps": sum(r["steps"] for rows in collection.values() for r in rows),
               "corpus_episode_seconds": sum(r["seconds"] for rows in collection.values() for r in rows),
               "limitations": ["anonymous occupancy approximation cannot identify overlapping enemies",
                  "unseen terrain uses observed map and gen14 occupancy belief; not an exact POMDP filter",
                  "fit subvalidation is development and is selection-biased",
                  "online updates use Brier gradients, not privileged movement labels"]}
    save(out / "fit-summary.json", summary)
    save(out / "matched-validation.json", matched)
    save(out / "corpus-summary.json", collection)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
