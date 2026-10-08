"""Development-only comparator behavior and same-experience prediction checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.evaluation_v3 import run_episode, aggregate
from dreamer.world_v3 import UnknownDynamicsMaze
from v3_fit import REGIMES, corpus_rows, load_module, replay_predictor, save, sha


def matched_known_law(source, row, seed):
    """Privileged current probabilities; never unseen map, occupancy, or identity."""
    agent = load_module(source)
    environment = UnknownDynamicsMaze(seed, regime=row["regime"])
    original = agent.world_model_step
    def step(memory, observation, last):
        observation = dict(observation,
                           known_law=list(environment.law_for_transition(observation["step"] + 1)))
        return original(memory, observation, last)
    agent.world_model_step = step
    return replay_predictor(agent, row, learn=False)


def episode(job):
    source, seed, case, regime = job
    rows = {}
    actions = {}
    for variant in ("frozen", "predictive", "known_law"):
        row = run_episode(source, seed, variant, replay=True, regime=regime)
        if row["error"]:
            raise RuntimeError(row["error"])
        trace = row.pop("trace")
        parameters = [f["model"]["learning"]["transition_weights"] for f in trace]
        actions[variant] = [f["action"] for f in trace]
        row.pop("seed")
        row["case"] = case
        row["parameter_frames_changed_from_first"] = sum(p != parameters[0] for p in parameters)
        # Never publish true-law model exports from the privileged reference.
        if variant == "known_law":
            row["learning"].pop("attempt_probabilities", None)
        rows[variant] = row
    differences = {}
    for left in ("predictive", "known_law"):
        a, b = actions[left], actions["frozen"]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        differences[left + "_versus_frozen"] = {"first_action_difference_step": first,
                                                "compared_prefix_frames": min(len(a), len(b))}
    return {"case": case, "regime": regime, "conditions": rows, "decisions": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", default="controls/v3/directional.py")
    parser.add_argument("--fit-out", default="results/v3-comparator-fit")
    parser.add_argument("--development-seeds", default="results/private/v3-development-seeds.json")
    parser.add_argument("--fit-validation-seeds", default="results/private/v3-fit-validation-seeds.json")
    parser.add_argument("--out", default="controls/v3/development-diagnostic.json")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    cpu = time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    seeds = json.loads(Path(args.development_seeds).read_text())[:8]
    fitting_seeds = json.loads(Path(args.fit_validation_seeds).read_text())
    if isinstance(seeds, dict):
        seeds = seeds["seeds"][:8]
    jobs = [(args.program, seed, i, regime) for regime in REGIMES for i, seed in enumerate(seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cases = list(pool.map(episode, jobs))
    source = Path(args.program)
    matched = {}
    rates = json.loads((Path(args.fit_out) / "matched-validation.json").read_text())
    fitting = json.loads((Path(args.fit_out) / "fit-summary.json").read_text())
    matched["frozen"] = rates["0.0"]
    matched["predictive"] = rates[str(fitting["selected_online_rate"])]
    matched["known_law"] = [matched_known_law(source, row, fitting_seeds[row["case"]])
                            for row in corpus_rows(Path(args.fit_out) / "validation")]
    # Verify intervention on actual recorded observation/action streams, with
    # state inference and mapping running in both versions. Selection uses no audit.
    audits = []
    for row in list(corpus_rows(Path(args.fit_out) / "validation"))[:6]:
        agent = load_module(source)
        online = replay_predictor(agent, row, True, with_frames=True)
        frozen = replay_predictor(agent, row, False, with_frames=True)
        pairs = list(zip(online["frames"], frozen["frames"]))
        audits.append({"case": row["case"], "regime": row["regime"],
                       "frames": len(pairs),
                       "same_map_and_localization": all(a["position"] == b["position"] and a["map"] == b["map"] for a, b in pairs),
                       "frozen_parameters_constant": all(a["parameters"] == agent._PRIOR for a in frozen["frames"]),
                       "online_changed_parameter_frames": online["changed_parameter_frames"],
                       "online_parameter_updates": online["parameter_updates"]})
    summaries = {regime: {variant: aggregate([c["conditions"][variant] for c in cases if c["regime"] == regime])["public"]
                          for variant in ("frozen", "predictive", "known_law")}
                 for regime in REGIMES}
    result = {"development_only": True, "program_sha256": sha(source),
              "condition_episodes": len(cases) * 3, "paired_cases_per_regime": len(seeds),
              "development_pool_sha256": sha(args.development_seeds),
              "matched_fit_subvalidation_pool_sha256": sha(args.fit_validation_seeds),
              "cases": cases, "summary": summaries, "matched_predictions": matched,
              "intervention_audits": audits,
              "decision_changes": {key: sum(c["decisions"][key]["first_action_difference_step"] is not None for c in cases)
                                   for key in ("predictive_versus_frozen", "known_law_versus_frozen")},
              "elapsed_seconds": time.monotonic() - started,
              "controller_cpu_seconds": time.process_time() - cpu,
              "children_cpu_seconds": ((resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - before.ru_utime)
                                       + (resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime - before.ru_stime)),
              "scope": "short development diagnostic, not assessment or optimal-policy bound",
              "matched_target": "one-tick union enemy occupancy, previous inner3, observed next labels; terminal transitions excluded",
              "matched_policy": "immutable memory control; recorded actions forced in predictor replay",
              "matched_selection_bias": "online rate and prior selected on these fitting-subvalidation cases",
              "ordinary_parameter_freeze": "only nine transition logits; map, localization, occupancy-state inference still update",
              "planning_cost": "gen14 one-tick occupancy and survival-conditioned two-tick collision heuristic; future path costs heuristic"}
    save(args.out, result)
    print(json.dumps({"summary": summaries, "decision_changes": result["decision_changes"],
                      "intervention_audits": audits, "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
