"""Source-reviewed development interventions; never champion selection.

A review file binds the inspected adaptive-state key, intervention semantics,
forecast contract, and any planning-risk adapter to an immutable source hash.
Different evolved representations require a fresh source review, not a generic
assumption that a learn flag freezes predictive adaptation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.evaluation_v3 import aggregate, run_episode
from dreamer.provenance import sha256
from dreamer.world_v3 import REGIMES
from v3_assessment import atomic_create, compact_trace

VARIANTS = ("predictive", "frozen", "no_planning", "frozen_no_planning")
MOVES = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))
GEN1_SHA256 = "ffd4065f8468273cc9df39176294606b21e9f76e6afcfd08d5282dc166a33286"
GEN4_INSTRUMENTED_SHA256 = "5861f5e973b98bb48c878cc6f780af8d919f2b413775642aa5b6d640c36aba3c"


def gen4_decision_audit(trace, variant, review):
    """Check the exact values captured after gen4's original action choice.

    Immediate occupancy is independently reconstructed from visible sources,
    attempted probabilities and public terrain. All predecessors of a legal
    immediate destination are visible, so hidden intensity cannot arrive there.
    Later hazards have a different event/horizon and are not scored as occupancy.
    """
    if review["program_sha256"] != GEN4_INSTRUMENTED_SHA256:
        raise ValueError("Gen4 observer audit requires its exact reviewed source")
    visits = {}
    counts = {key: 0 for key in ("frames", "choice_frames", "fallback_frames", "choices",
        "immediate_target_frames", "excluded_occupied_entry_frames",
        "chosen_move_mismatches", "choice_score_mismatches", "choice_hazard_field_mismatches",
        "planning_reconstruction_mismatches", "export_reconstruction_mismatches",
        "observer_law_mismatches", "planning_export_differing_frames")}
    planning_sum = exported_sum = gap_sum = gap_max = 0.
    examples = []
    for frame in trace:
        model, obs, action = frame["model"], frame["obs"], frame["action"]
        d = model.get("diagnostic_planning")
        if not isinstance(d, dict) or d.get("adapter") != "gen4-planner-observer-v1":
            raise ValueError("Missing actual planner diagnostic")
        pos = tuple(model["position"])
        visits[pos] = visits.get(pos, 0) + 1
        target = tuple(pos[i] + action["move"][i] for i in range(2))
        terrain = {(x, y): cell for x, y, cell, age in model["terrain"]}
        enemies = {(pos[0]+x-2, pos[1]+y-2) for y in range(5) for x in range(5) if obs["grid"][y][x] == 5}
        opened = {tuple(pos[i]+step[i] for i in range(2)) for step in ((1,0),(-1,0),(0,1),(0,-1))
                  if obs["keys"] >= 2 and terrain.get(tuple(pos[i]+step[i] for i in range(2))) == 3}
        def wall(p):
            return terrain.get(p) in (-1, 1) or (terrain.get(p) == 3 and not obs["door_open"] and p not in opened)
        def probability(law):
            empty = 1.
            for ex, ey in sorted(enemies):
                mass = 0.
                for (dx, dy), weight in zip(MOVES, law):
                    q = ex+dx, ey+dy
                    if wall(q) or (dx and dy and (wall((ex+dx, ey)) or wall((ex, ey+dy)))):
                        q = ex, ey
                    if q == target:
                        mass += weight
                empty *= 1.-mass
            return min(1., max(0., 1.-empty))
        learned_law = model["learning"]["attempted_movement_law"]
        used_law = [1/9]*9 if variant in ("no_planning", "frozen_no_planning") else learned_law
        counts["observer_law_mismatches"] += any(abs(a-b)>1e-12 for a,b in zip(used_law,d["law"]))
        hazard1 = {(x,y):p for x,y,p in d["collision_hazards_local"][0]}
        occupancy1 = {(x,y):p for x,y,p in d["occupancy_horizon1_local"]}
        exported = {(x,y):p for x,y,p in model["enemy"]}.get(target,model["default_enemy"])
        used = hazard1.get(target, 0.)
        choices = d["choices"]
        counts["frames"] += 1
        counts["choices"] += len(choices)
        counts["chosen_move_mismatches"] += d["move"] != action["move"]
        if choices:
            counts["choice_frames"] += 1
            best = min(choices, key=lambda c: (math.inf if c["score"] is None else c["score"],
                c["hazard"], math.inf if c["potential"] is None else c["potential"], tuple(c["move"])))
            counts["chosen_move_mismatches"] += best["move"] != action["move"]
            for choice in choices:
                q = tuple(choice["destination"])
                hazard = hazard1.get(q, 0.)
                counts["choice_hazard_field_mismatches"] += abs(hazard-choice["hazard"])>1e-12
                continuation = math.inf if choice["continuation"] is None else choice["continuation"]
                reconstructed = (1.+(.16 if choice["move"] == [0,0] else 0.)
                    +.02*min(10,visits.get(q,0)) +d["risk_weight"]*(-math.log(max(1e-7,1.-hazard)))+continuation)
                observed = math.inf if choice["score"] is None else choice["score"]
                counts["choice_score_mismatches"] += not (reconstructed == observed or abs(reconstructed-observed)<=1e-10)
        else:
            counts["fallback_frames"] += 1
        # The chosen action's pre-contact contribution is zero when the planner
        # admits it. Retain fallbacks separately rather than assume that event.
        if target in enemies:
            counts["excluded_occupied_entry_frames"] += 1
            continue
        counts["immediate_target_frames"] += 1
        counts["planning_reconstruction_mismatches"] += (abs(used-probability(used_law))>1e-12 or abs(used-occupancy1.get(target,0.))>1e-12)
        counts["export_reconstruction_mismatches"] += abs(exported-probability(learned_law))>1e-12
        ox, oy = frame["world"]["origin"]
        label = float((target[0]+ox,target[1]+oy) in {tuple(p) for p in frame["next_enemies"]})
        planning_sum += (used-label)**2
        exported_sum += (exported-label)**2
        gap = abs(used-exported)
        gap_sum += gap
        gap_max = max(gap_max,gap)
        counts["planning_export_differing_frames"] += gap>1e-12
        if gap>1e-12 and len(examples)<3:
            examples.append({"step":obs["step"],"move":action["move"],"destination_relative":list(target),
                "exported_occupancy":exported,"used_immediate_hazard":used})
    scored = counts["immediate_target_frames"]
    return {"available":True,**counts,"planning_brier_destination":[planning_sum,scored],
        "exported_brier_destination":[exported_sum,scored],"planning_export_absolute_gap":[gap_sum,scored],
        "max_planning_export_gap":gap_max,"examples":examples,
        "target":"Immediate next-tick occupancy at chosen legal destination with no observed pre-contact; source-specific equality to hazard1, not general death probability.",
        "future_hazard_contract":"Actual captured stage2/3 hazards are unconditional unions of occupancy before/after that stage, not horizon1 occupancy or path-conditioned death risk.",
        "raw_fields_retained_in_private_replay":"diagnostic_planning has actual candidate scores/continuations, learned-or-uniform law and local horizon1 occupancy/stage1..3 hazards."}


def reviewed_gen1_decision_risk(frame, variant):
    """Reconstruct the reviewed immediate planning risk from public exports.

    This executes no candidate source and reads no hidden state. Gen1 uses the
    same forecast dict as its exporter in predictive mode; its intervention
    recomputes this law-independent geometry with uniform attempted movements.
    At any candidate immediate destination (distance<=1), its horizon1 hidden
    background is exactly zero. Later planning terms are explicitly heuristic.
    """
    model, obs, action = frame["model"], frame["obs"], frame["action"]
    pos = tuple(model["position"])
    target = tuple(pos[i] + action["move"][i] for i in range(2))
    terrain = {(x, y): cell for x, y, cell, age in model["terrain"]}
    enemies = {(pos[0]+x-2, pos[1]+y-2) for y in range(5) for x in range(5) if obs["grid"][y][x] == 5}
    opened = obs["door_open"] or (obs["keys"] >= 2 and any(
        terrain.get((pos[0]+dx, pos[1]+dy)) == 3 for dx, dy in ((1,0),(-1,0),(0,1),(0,-1))))
    def wall(p):
        c = terrain.get(p)
        return c in (-1, 1) or (c == 3 and not opened)
    def probability(law):
        empty = 1.
        for ex, ey in sorted(enemies):
            mass = 0.
            for (dx, dy), weight in zip(MOVES, law):
                q = ex+dx, ey+dy
                if wall(q) or (dx and dy and (wall((ex+dx, ey)) or wall((ex, ey+dy)))):
                    q = ex, ey
                if q == target:
                    mass += weight
            empty *= 1. - min(1., mass)
        return min(1., max(0., 1.-empty))
    law = model["learning"]["attempt_probabilities"]
    predicted = probability(law)
    used = probability([1/9]*9) if variant in ("no_planning", "frozen_no_planning") else predicted
    exports = {(x, y): p for x, y, p in model["enemy"]}
    return {"target_relative": list(target), "exported_probability": exports.get(target, model["default_enemy"]),
            "reconstructed_export": predicted, "planning_probability": used}


def decision_audit(trace, variant, review):
    if review.get("risk_adapter") == "gen4_planner_observer":
        return gen4_decision_audit(trace, variant, review)
    if review.get("risk_adapter") != "gen1_attempt_law":
        return {"available": False, "reason": "No source-reviewed decision-risk adapter"}
    if review["program_sha256"] != GEN1_SHA256:
        raise ValueError("The gen1 decision-risk reconstruction is bound to its inspected source")
    valid, mismatch, planning_sum, exported_sum, gap_sum, gap_max, differs = 0, 0, 0., 0., 0., 0., 0
    examples = []
    for frame in trace:
        risk = reviewed_gen1_decision_risk(frame, variant)
        ox, oy = frame["world"]["origin"]
        target = risk["target_relative"]
        label = float((target[0]+ox, target[1]+oy) in {tuple(p) for p in frame["next_enemies"]})
        mismatch += abs(risk["exported_probability"]-risk["reconstructed_export"]) > 1e-12
        planning_sum += (risk["planning_probability"]-label)**2
        exported_sum += (risk["exported_probability"]-label)**2
        gap = abs(risk["planning_probability"]-risk["exported_probability"])
        gap_sum += gap
        gap_max = max(gap_max, gap)
        differs += gap > 1e-12
        valid += 1
        if gap > 1e-12 and len(examples) < 3:
            # Only probabilities/relative coordinates, never hidden positions or laws.
            examples.append({"step": frame["obs"]["step"], "attempt": frame["action"]["move"], **risk})
    return {"available": True, "frames": valid, "reconstructed_export_mismatches": mismatch,
            "planning_brier_destination": [planning_sum, valid],
            "exported_brier_destination": [exported_sum, valid],
            "planning_export_absolute_gap": [gap_sum, valid], "max_planning_export_gap": gap_max,
            "planning_export_differing_frames": differs, "examples": examples,
            "target": "next-tick union occupancy at attempted legal destination; excludes entry collision",
            "risk_source": "uniform attempted-law occupancy" if variant in ("no_planning", "frozen_no_planning") else "same learned horizon1 occupancy as exported"}


def run_one(job):
    case, seed, regime, program, review, out = job
    rows = []
    for variant in VARIANTS:
        target = Path(out) / "episodes" / f"{case:04d}--{regime}--{variant}.json"
        if target.exists():
            row = json.loads(target.read_text())
            if row["program_sha256"] != review["program_sha256"]:
                raise ValueError("Mechanism checkpoint source drift")
            rows.append(row)
            continue
        row = run_episode(program, seed, variant, replay=True, regime=regime)
        audit = decision_audit(row["trace"], variant, review)
        row = compact_trace(row, review["parameter_key"])
        row.pop("seed")
        row.update(case=case, program_sha256=review["program_sha256"], decision_audit=audit)
        atomic_create(target, row)
        rows.append(row)
    return rows


def analyze(rows):
    by_key = {(r["case"], r["regime"], r["variant"]): r for r in rows}
    comparisons = []
    for case, regime in sorted({(r["case"], r["regime"]) for r in rows}):
        a, b = by_key[case, regime, "no_planning"], by_key[case, regime, "frozen_no_planning"]
        frozen, frozen_fixed = by_key[case, regime, "frozen"], b
        comparisons.append({"case": case, "regime": regime,
            "fixed_prediction_use_both_valid": a["error"] is None and b["error"] is None,
            "fixed_prediction_use_same_trajectory": a["audit"]["trajectory_sha256"] == b["audit"]["trajectory_sha256"],
            "fixed_prediction_use_same_physical_states": a["audit"]["recorded_physical_states_sha256"] == b["audit"]["recorded_physical_states_sha256"],
            "fixed_prediction_use_same_observations": a["audit"]["observations_sha256"] == b["audit"]["observations_sha256"],
            "fixed_prediction_use_same_map_localization": a["audit"]["map_position_sha256"] == b["audit"]["map_position_sha256"],
            "fixed_prediction_use_same_actions": a["audit"]["actions_sha256"] == b["audit"]["actions_sha256"],
            "frozen_versus_frozen_fixed_same_trajectory": frozen["audit"]["trajectory_sha256"] == frozen_fixed["audit"]["trajectory_sha256"],
            "online_parameters_changed": a["audit"]["parameters_constant"] is False,
            "frozen_parameters_constant": frozen["audit"]["parameters_constant"] is True and b["audit"]["parameters_constant"] is True,
            "matched_online_stats": a["stats"], "matched_frozen_stats": b["stats"]})
    return {"variants": {regime: {variant: aggregate([r for r in rows if r["regime"] == regime and r["variant"] == variant])["public"]
                                     for variant in VARIANTS} for regime in REGIMES},
            "matched_episodes": comparisons,
            "matched_prediction_interpretation": "Identical experience only for pairs whose full trajectory and map/action checks pass; episode sums/counts retained",
            "all_fixed_prediction_use_trajectories_match": all(r["fixed_prediction_use_same_trajectory"] for r in comparisons),
            "all_fixed_prediction_use_maps_match": all(r["fixed_prediction_use_same_map_localization"] for r in comparisons),
            "all_fixed_prediction_use_observations_match": all(r["fixed_prediction_use_same_observations"] for r in comparisons),
            "matched_prediction_valid_by_regime": {regime: all(all(r[k] for k in (
                "fixed_prediction_use_both_valid", "fixed_prediction_use_same_trajectory",
                "fixed_prediction_use_same_physical_states", "fixed_prediction_use_same_observations",
                "fixed_prediction_use_same_map_localization", "fixed_prediction_use_same_actions"))
                for r in comparisons if r["regime"]==regime) for regime in REGIMES},
            "all_frozen_parameters_constant": all(r["frozen_parameters_constant"] is True for r in comparisons),
            "online_parameter_change_episodes": sum(r["online_parameters_changed"] for r in comparisons)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="results/private/v3-development-seeds.json")
    parser.add_argument("--smoke", action="store_true", help="First four development layouts x three regimes x four interventions")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    review = json.loads(Path(args.review).read_text())
    if review["program_sha256"] != sha256(args.program):
        raise ValueError("Program changed since its manual intervention review")
    seeds = json.loads(Path(args.seeds).read_text())
    if args.smoke:
        seeds = seeds[:4]
    elif len(seeds) != 24:
        raise ValueError("Full development mechanism audit uses the frozen 24-layout development pool")
    out = Path(args.out)
    manifest = {"development_only": True, "champion_selection": False,
                "scope": "gen1 interface smoke" if args.smoke else "selected-program development mechanism audit",
                "program_sha256": sha256(args.program), "review_sha256": sha256(args.review),
                "driver_sha256": sha256(__file__), "pool_sha256": sha256(args.seeds),
                "layouts": len(seeds), "regimes": list(REGIMES), "variants": list(VARIANTS),
                "condition_episodes": len(seeds)*3*4, "model_calls": 0,
                "intervention_names": review["interventions"]}
    atomic_create(out / "manifest.json", manifest)
    started = time.monotonic()
    jobs = [(case, seed, regime, str(Path(args.program).resolve()), review, str(out))
            for regime in REGIMES for case, seed in enumerate(seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for group in pool.map(run_one, jobs):
            rows.extend(group)
            print(json.dumps({"completed_condition_episodes": len(rows), "total": len(jobs)*4}), flush=True)
    summary = {"manifest": manifest, **analyze(rows), "session_seconds": time.monotonic()-started,
               "episode_seconds": sum(r["seconds"] for r in rows),
               "candidate_cpu_seconds": sum(r["candidate_cpu_seconds"] for r in rows),
               "evaluator_cpu_seconds": sum(r["evaluator_cpu_seconds"] for r in rows),
               "environment_steps": sum(r["steps"] for r in rows),
               "invalid_condition_episodes": sum(r["error"] is not None for r in rows)}
    # Repeated analysis preserves observations and records each analysis execution.
    target = out / ("analysis-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    atomic_create(target, summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k: summary[k] for k in ("all_fixed_prediction_use_trajectories_match", "all_fixed_prediction_use_maps_match", "all_frozen_parameters_constant", "online_parameter_change_episodes", "environment_steps", "session_seconds")}), flush=True)


if __name__ == "__main__":
    main()
