"""Fixed, preregistered six-condition assessment; no mutation or model calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_candidate import compact_trace
from dreamer.evaluation import run_episode
from dreamer.provenance import assessment_pool, control_path, evaluation_identity, record, sha256

SELECTED = "588eeb7c10b978fe86c7e7b572f177e8b755c9c25699f3c75bfadd41192ec5e0"
CONDITIONS = ("memory", "original_predictive", "predictive", "frozen", "no_planning", "frozen_no_planning")


def evaluate_case(job):
    index, seed, programs = job
    rows = []
    # Rotate order by case to distribute machine-load/launch-order effects.
    for variant in CONDITIONS[index % 6:] + CONDITIONS[:index % 6]:
        row = compact_trace(run_episode(programs[variant], seed, variant, replay=True))
        row.pop("seed")  # Private pool supplies the index-to-seed mapping.
        row["case"] = index
        rows.append(row)
    rows.sort(key=lambda r: CONDITIONS.index(r["variant"]))
    return {"case": index, "conditions": rows}


def save_case(path, data):
    """Publish one complete paired case atomically, without replacing any record."""
    temporary = path.with_suffix(".partial")
    with temporary.open("w") as handle:
        json.dump(data, handle, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)  # Fails if a saved result already exists.
    temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/campaign-v2-assessment-1024")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = ROOT / "artifacts/campaign-v2/assessment-1024/preregistration.json"
        plan = json.loads(plan_path.read_text())
        assert plan["sample_size"] == 1024 and plan["conditions"] == list(CONDITIONS)
        assert sha256(__file__) == plan["assessment_driver_sha256"]
        assert sha256(ROOT / "scripts/audit_candidate.py") == plan["audit_helper_sha256"]
        program = ROOT / "artifacts/campaign-v2/completed-50/selected.py"
        assert sha256(program) == SELECTED == plan["selected_source_sha256"]
        assert evaluation_identity() == plan["evaluation"]
        campaign = ROOT / "results/campaign-v2"
        seed_path = ROOT / "results/private/campaign-v2-assessment-seeds.json"
        # An existing registered assessment must agree; never replace its pool.
        seeds, info = assessment_pool(campaign, seed_path, program, 1024, reserve=True)
        programs = {v: str(control_path("memory" if v == "memory" else "predictive"))
                    if v in ("memory", "original_predictive") else str(program) for v in CONDITIONS}
        manifest = {"preregistration_sha256": sha256(plan_path), "episode_pool": info,
                    "conditions": list(CONDITIONS), "episodes_per_condition": 1024,
                    "programs": {v: sha256(p) for v, p in programs.items()},
                    "evaluation": evaluation_identity(), "driver_sha256": sha256(__file__),
                    "audit_helper_sha256": sha256(ROOT / "scripts/audit_candidate.py"),
                    "model_calls": 0, "condition_order": "rotate by case index modulo six"}
        record(out / "manifest.json", manifest)
        cases = out / "cases"
        cases.mkdir(exist_ok=True)
        # A crash between fsync and linking can leave a complete saved case.
        # Preserve incomplete bytes and explicitly record any required rerun.
        for partial in cases.glob("*.partial"):
            try:
                data = json.loads(partial.read_text())
                assert data["case"] == int(partial.stem)
                assert [r["variant"] for r in data["conditions"]] == list(CONDITIONS)
            except (ValueError, KeyError, AssertionError):
                archived = partial.with_name(partial.name + ".interrupted-" + str(time.time_ns()))
                partial.rename(archived)
                print("Preserved incomplete checkpoint; repeating case", partial.stem, str(archived), flush=True)
                continue
            target = partial.with_suffix(".json")
            if target.exists():
                assert json.loads(target.read_text()) == data
            else:
                os.link(partial, target)
            partial.unlink()
        existing = {}
        for path in cases.glob("*.json"):
            data = json.loads(path.read_text())
            assert data["case"] == int(path.stem) and 0 <= data["case"] < 1024
            assert [r["variant"] for r in data["conditions"]] == list(CONDITIONS)
            assert all(r["case"] == data["case"] for r in data["conditions"])
            existing[data["case"]] = data
        record(out / ("execution-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"),
               {"workers": args.workers, "existing_cases": len(existing), "sample_size": 1024})
        pending = [i for i in range(1024) if i not in existing]
        started = time.monotonic()
        finished_now = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for offset in range(0, len(pending), args.workers):
                batch = pending[offset:offset+args.workers]
                for data in executor.map(evaluate_case, [(i, seeds[i], programs) for i in batch]):
                    save_case(cases / f"{data['case']:04d}.json", data)
                    existing[data["case"]] = data
                    finished_now += 1
                elapsed = time.monotonic() - started
                if len(existing) % 16 == 0 or len(existing) == 1024:
                    rows = [r for c in existing.values() for r in c["conditions"]]
                    print(json.dumps({"completed_cases": len(existing), "condition_episodes": len(rows),
                          "elapsed_seconds": round(elapsed, 1),
                          "estimated_remaining_seconds": round((1024-len(existing))*elapsed/finished_now, 1),
                          "escapes": {v: sum(r["reason"] == "escaped" for r in rows if r["variant"] == v) for v in CONDITIONS},
                          "invalid": sum(r["error"] is not None for r in rows)}), flush=True)
        record(out / "execution-complete.json", {"cases": 1024, "condition_episodes": 6144,
               "case_file_sha256": {p.name: sha256(p) for p in sorted(cases.glob("*.json"))},
               "manifest_sha256": sha256(out / "manifest.json")})


if __name__ == "__main__":
    main()
