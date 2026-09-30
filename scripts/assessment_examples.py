"""Expose rule-selected assessment examples only after the numerical analysis closes."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_candidate import compact_trace
from dreamer.evaluation import run_episode
from dreamer.provenance import sha256, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/campaign-v2/assessment-1024")
    args = parser.parse_args()
    data = Path(args.data)
    closed = json.loads((data / "analysis-closed.json").read_text())
    assert sha256(data / "analysis.json") == closed["analysis_sha256"]
    assert sha256(data / "episodes.jsonl.gz") == closed["episodes_sha256"]
    if (data / "behavior-examples.json").exists():
        print("Examples already recorded; preserving them.")
        return
    analysis = json.loads((data / "analysis.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    pool = Path(manifest["episode_pool"]["path"])
    assert sha256(pool) == manifest["episode_pool"]["sha256"]
    seeds = json.loads(pool.read_text())
    program = ROOT / "artifacts/campaign-v2/completed-50/selected.py"
    assert sha256(program) == manifest["programs"]["predictive"]
    with gzip.open(data / "episodes.jsonl.gz", "rt") as file:
        saved = {(r["case"], r["variant"]): r for r in map(json.loads, file)}
    examples = []
    for stratum, selection in analysis["example_selection"].items():
        index = selection["case"]
        if index is None:
            continue
        traces, agents = {}, {}
        for variant in ("predictive", "frozen"):
            result = run_episode(program, seeds[index], variant, replay=True)
            trace = result["trace"]
            compact_trace(result)
            original = saved[index, variant]
            fields = [k for k in original if k not in ("case", "seconds")]
            mismatches = [k for k in fields if original[k] != result[k]]
            traces[variant] = trace
            agents[variant] = {
                "original": {k: original[k] for k in ("reason", "steps", "keys", "door", "error")},
                "replay": {k: result[k] for k in ("reason", "steps", "keys", "door", "error")},
                "reproduces_original": not mismatches, "mismatched_fields": mismatches,
                "trajectory_sha256": result["audit"]["trajectory_sha256"],
                "path": [f["world"]["agent"] for f in trace]}
        common = min(len(t) for t in traces.values())
        divergence = next((i for i in range(common) if traces["predictive"][i]["action"] != traces["frozen"][i]["action"]), None)
        frame_index = divergence if divergence is not None else max(0, common-1)
        for variant in agents:
            agents[variant]["display_frame"] = traces[variant][frame_index] if traces[variant] else None
        examples.append({"stratum": stratum, "case": index, "exposed_seed": seeds[index],
                         "first_action_divergence_step": divergence,
                         "display_step": frame_index, "agents": agents})
    result = {"published_after_analysis_closed": True, "analysis_closed_sha256": sha256(data / "analysis-closed.json"),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "selection_rule": "Lowest case index in each prespecified selected/frozen escape stratum",
              "display_rule": "First differing action; last shared frame if actions never differ",
              "exposure_warning": "These cases are now exposed and must never be reused as a future fresh test.",
              "replays_are_additional_diagnostics": True, "inferential_episode_count_unchanged": 6144,
              "examples": examples}
    record(data / "behavior-examples.json", result)
    record(data / "exposed-cases.json", {"analysis_closed_sha256": sha256(data / "analysis-closed.json"),
           "cases": [{"case": x["case"], "seed": x["exposed_seed"]} for x in examples],
           "status": "Exposed after analysis; exclude from all future fresh assessments"})
    print(json.dumps({x["stratum"]: {v: a["reproduces_original"] for v, a in x["agents"].items()} for x in examples}))


if __name__ == "__main__":
    main()
