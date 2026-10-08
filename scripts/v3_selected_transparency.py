"""Compare immutable selected gen4 with its passive diagnostic instrumentation.

Exactly one registered development layout, three regimes and four conditions
are executed for each source (24 worlds). No selection or assessment cases.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.evaluation_v3 import run_episode
from dreamer.provenance import sha256
from v3_assessment import atomic_create, compact_trace, digest
from v3_instrument_selected import ORIGINAL_SHA256
from v3_mechanism import GEN4_INSTRUMENTED_SHA256, VARIANTS, gen4_decision_audit

SOURCES = {"original": ROOT/"artifacts/campaign-v3/selection/selected.py",
           "instrumented": ROOT/"artifacts/campaign-v3/selection/selected-instrumented.py"}
HASHES = {"original": ORIGINAL_SHA256, "instrumented": GEN4_INSTRUMENTED_SHA256}


def original_fields(model):
    result = {k:v for k,v in model.items() if k != "diagnostic_planning"}
    result["learning"] = {k:v for k,v in result["learning"].items() if k != "raw_predictive_state"}
    return result


def run_pair(job):
    seed, regime, variant, out = job
    rows = {}
    for name, path in SOURCES.items():
        target = Path(out)/"episodes"/f"{name}--{regime}--{variant}.json"
        if target.exists():
            row = json.loads(target.read_text())
            if row["program_sha256"] != HASHES[name]:
                raise ValueError("Transparency checkpoint source drift")
            rows[name] = row
            continue
        row = run_episode(path, seed, variant, replay=True, regime=regime)
        trace = row["trace"]
        row["original_export_fields_sha256"] = digest([original_fields(f["model"]) for f in trace])
        if name == "instrumented":
            row["decision_audit"] = gen4_decision_audit(trace, variant, {"program_sha256":HASHES[name]})
        if regime == "switch" and variant == "predictive":
            replay = Path(out)/"replays"/f"{name}--switch--predictive.json.gz"
            replay.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(replay,"wt") as handle:
                json.dump(trace,handle,separators=(",",":"),allow_nan=False)
            row["private_replay"] = {"path":str(replay),"sha256":sha256(replay)}
        row = compact_trace(row, "model.learning.raw_predictive_state" if name=="instrumented" else None)
        row.pop("seed")
        row.update(case=0,source_name=name,program_sha256=HASHES[name])
        atomic_create(target,row)
        rows[name] = row
    a,b = rows["original"],rows["instrumented"]
    return {"case":0,"regime":regime,"variant":variant,
        "both_valid":a["error"] is None and b["error"] is None,
        "same_original_export_fields":a["original_export_fields_sha256"]==b["original_export_fields_sha256"],
        "same_actions":a["audit"]["actions_sha256"]==b["audit"]["actions_sha256"],
        "same_observations":a["audit"]["observations_sha256"]==b["audit"]["observations_sha256"],
        "same_physical_trajectory":a["audit"]["trajectory_sha256"]==b["audit"]["trajectory_sha256"],
        "same_map_localization":a["audit"]["map_position_sha256"]==b["audit"]["map_position_sha256"],
        "same_forecast_statistics":a["stats"]==b["stats"],
        "same_outcome":all(a[k]==b[k] for k in ("reason","steps","task","keys","door")),
        "original_frames":a["audit"]["frames"],"instrumented_frames":b["audit"]["frames"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="results/v3-selected-transparency")
    parser.add_argument("--summary",default="artifacts/campaign-v3/selection/transparency.json")
    parser.add_argument("--seeds",default="results/private/v3-development-seeds.json")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    for name,path in SOURCES.items():
        if sha256(path)!=HASHES[name]:
            raise ValueError("Source differs from inspected instrumentation")
    seeds=json.loads(Path(args.seeds).read_text())
    protocol=json.loads((ROOT/"artifacts/campaign-v3/protocol.json").read_text())
    if sha256(args.seeds)!=protocol["splits"]["development"]["pool_sha256"]:
        raise ValueError("Use registered development pool")
    out=Path(args.out)
    manifest={"development_only":True,"cases":[0],"regimes":["uniform","stationary","switch"],
        "variants":list(VARIANTS),"sources":{k:{"path":str(v),"sha256":HASHES[k]} for k,v in SOURCES.items()},
        "pool_sha256":sha256(args.seeds),"driver_sha256":sha256(__file__),
        "mechanism_driver_sha256":sha256(ROOT/"scripts/v3_mechanism.py"),"condition_episodes":24,"model_calls":0}
    atomic_create(out/"manifest.json",manifest)
    started=time.monotonic()
    jobs=[(seeds[0],regime,variant,str(out)) for regime in manifest["regimes"] for variant in VARIANTS]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pairs=[]
        for pair in executor.map(run_pair,jobs):
            pairs.append(pair)
            print(json.dumps({"transparency_pairs_complete":len(pairs),"of":12}),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted((out/"episodes").glob("*.json"))]
    checks=["both_valid","same_original_export_fields","same_actions","same_observations",
        "same_physical_trajectory","same_map_localization","same_forecast_statistics","same_outcome"]
    summary={"manifest":manifest,"pairs":pairs,"all_checks_pass":all(p[k] for p in pairs for k in checks),
        "condition_episodes":len(rows),"environment_steps":sum(r["steps"] for r in rows),
        "invalid_condition_episodes":sum(r["error"] is not None for r in rows),
        "session_seconds":time.monotonic()-started,
        "episode_seconds":sum(r["seconds"] for r in rows),
        "candidate_cpu_seconds":sum(r["candidate_cpu_seconds"] for r in rows),
        "evaluator_cpu_seconds":sum(r["evaluator_cpu_seconds"] for r in rows),
        "decision_audits":[r["decision_audit"] for r in rows if "decision_audit" in r],
        "cpu_by_source":{name:sum(r["candidate_cpu_seconds"] for r in rows if r["source_name"]==name) for name in SOURCES},
        "limitation":"Paired runtime transparency on one reused development layout per regime, plus exact original-algorithm AST preservation; instrumentation still adds measured runtime and output overhead.",
        "model_calls":0}
    atomic_create(Path(args.summary),summary)
    print(json.dumps({k:summary[k] for k in ("all_checks_pass","condition_episodes","environment_steps","session_seconds")}),flush=True)
    if not summary["all_checks_pass"]:
        raise RuntimeError("Instrumentation transparency did not pass; preserve all observations")


if __name__ == "__main__":
    main()
