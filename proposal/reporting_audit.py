"""Bounded passive map-reporting audit of one existing generation-2 replay."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource
import signal
import time

from dreamer.world import DynamicMaze
from proposal.evaluation import Candidate, audit_map
from proposal.study_controls import ROOT, SELECTED, SOURCE_HASHES


REPLAY = ROOT / "artifacts/proposal/full-native-01/representative-replays.json.gz"
REPLAY_SHA256 = "83bd8d22e3581aa98215e0909c8651f8f6e410250ffaab6d0b56b72c53db0dae"
VARIANT = ROOT / "proposal/study/reporting_retention.py"
OUT = ROOT / "artifacts/proposal/broader-study/reporting-audit"
WITHDRAWAL = '''    if epoch != memory.get("map_epoch", 0):
        for pos in unstable:
            believed.pop(pos, None)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Refusing changed overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def prepare():
    if sha(SELECTED) != SOURCE_HASHES["selected"] or sha(REPLAY) != REPLAY_SHA256:
        raise ValueError("Frozen selected source or historical replay changed")
    source = SELECTED.read_text()
    if source.count(WITHDRAWAL) != 1:
        raise ValueError("Expected exactly one epoch-based reporting withdrawal")
    variant = source.replace(WITHDRAWAL,
        "    # Reporting-only intervention: retain previously reported unstable cells.\n", 1)
    write_once(VARIANT, variant)
    protocol = {
        "identity": "proposal-passive-reporting-retention-v1",
        "question": "On this fixed historical trajectory, does epoch-based withdrawal alter map-reporting accuracy or coverage?",
        "replay": str(REPLAY.relative_to(ROOT)), "replay_sha256": REPLAY_SHA256,
        "record_selector": {"generation": 2, "case": 0}, "frames": 79,
        "selected": str(SELECTED.relative_to(ROOT)), "selected_sha256": SOURCE_HASHES["selected"],
        "variant": str(VARIANT.relative_to(ROOT)), "variant_sha256": sha(VARIANT),
        "variant_derivation": "One exact source splice removes only the epoch-change for-loop calling believed.pop(pos, None); all terrain, unstable flags and planning stay unchanged.",
        "driver_sha256": sha(Path(__file__)),
        "evaluator_sha256": {name: sha(ROOT / name) for name in
            ("proposal/evaluation.py", "proposal/worker.py", "dreamer/world.py", "dreamer/isolation.py")},
        "observation_reconstruction": "object.__new__(DynamicMaze), saved snapshot fields and previous-snapshot observable displacement/key/door differences, followed by unchanged observe(); no initialization, transitions or RNG draws.",
        "validation": "Every selected-program action and ordered belief list must exactly match the saved frame; otherwise stop before the variant pass.",
        "candidate_boundary": "Existing restricted proposal.evaluation.Candidate subprocess; observations only.",
        "limits": {"max_replies": 158, "candidate_cpu_seconds_each": 10,
                   "candidate_memory_mib": 192, "execution_wall_seconds": 60,
                   "model_calls": 0, "new_world_episodes": 0, "new_transitions": 0},
        "analysis": "Pooled reported-cell accuracy, per-frame coverage, extra claims and accuracy of extra claims; actions compared on identical observations. No variant action is followed.",
        "inference": "One exposed historical representative trajectory; point diagnostic only, no population uncertainty or new performance estimate.",
    }
    write_once(OUT / "protocol.json", protocol)
    return protocol


def snapshot_observation(snapshot, previous):
    """Use exact evaluator observation semantics without constructing a world."""
    env = object.__new__(DynamicMaze)
    env.grid = snapshot["grid"]
    env.enemies = [tuple(p) for p in snapshot["enemies"]]
    env.agent_pos = tuple(snapshot["agent"])
    env.origin = tuple(snapshot["origin"])
    env.steps = snapshot["step"]
    env.keys_collected = snapshot["keys"]
    env.door_open = snapshot["door_open"]
    env.feedback = {
        "displacement": [a - b for a, b in zip(snapshot["agent"], previous["agent"])] if previous else [0, 0],
        "collected": snapshot["keys"] > previous["keys"] if previous else False,
        "opened": snapshot["door_open"] and not previous["door_open"] if previous else False,
    }
    return env, env.observe()


def run():
    if (OUT / "summary.json").exists():
        raise ValueError("This audit is complete or stopped; do not rerun it")
    if not (OUT / "protocol.json").exists():
        raise ValueError("Freeze audit scope with --prepare before --run")
    protocol = json.loads((OUT / "protocol.json").read_text())
    # Read-back preparation checks all source/replay identities against the freeze.
    if protocol != prepare():
        raise ValueError("Frozen audit specification changed")
    records = json.loads(gzip.decompress(REPLAY.read_bytes()))
    selected = [r for r in records if r["generation"] == 2 and r["case"] == 0]
    if len(selected) != 1 or len(selected[0]["trace"]) != 79:
        raise ValueError("Frozen replay selector/frame count differs")
    frames = selected[0]["trace"]
    if [f["world"]["step"] for f in frames] != list(range(79)):
        raise ValueError("Replay does not contain all successive pre-action frames")
    started = time.monotonic()
    cpu_started = resource.getrusage(resource.RUSAGE_CHILDREN)
    replies, rows = 0, []
    baseline = []
    error = None
    started_utc = datetime.now(timezone.utc).isoformat()

    def timeout(signum, frame):
        raise TimeoutError("Frozen passive audit exceeded 60-second wall boundary")

    def query(candidate, obs):
        nonlocal replies
        if replies >= 158 or time.monotonic() - started >= 60:
            raise TimeoutError("Frozen passive audit reply/wall boundary")
        replies += 1
        return candidate.query({"obs": obs})

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    try:
        with Candidate(SELECTED) as candidate:
            previous = None
            for index, frame in enumerate(frames):
                env, obs = snapshot_observation(frame["world"], previous)
                reply = query(candidate, obs)
                if reply["action"] != frame["action"] or reply["belief"] != frame["belief"]:
                    raise ValueError(f"Historical selected-program reproduction failed at frame {index}")
                correct, count = audit_map(reply["belief"], env)
                saved = selected[0]["timeline"][index]
                if (correct, count) != (saved["correct"], saved["audited"]):
                    raise ValueError(f"Historical evaluator metrics differ at frame {index}")
                baseline.append(reply)
                previous = frame["world"]
        with Candidate(VARIANT) as candidate:
            previous = None
            for index, frame in enumerate(frames):
                env, obs = snapshot_observation(frame["world"], previous)
                reply = query(candidate, obs)
                old_correct, old_count = audit_map(baseline[index]["belief"], env)
                correct, count = audit_map(reply["belief"], env)
                old_map = {(x, y): cell for x, y, cell in baseline[index]["belief"]}
                new_map = {(x, y): cell for x, y, cell in reply["belief"]}
                if any(new_map.get(pos) != cell for pos, cell in old_map.items()):
                    raise ValueError(f"Retention changed an existing claim at frame {index}")
                extra = [[x, y, cell] for (x, y), cell in new_map.items() if (x, y) not in old_map]
                extra_correct, extra_count = audit_map(extra, env)
                rows.append({"frame": index, "step": env.steps,
                    "selected_correct": old_correct, "selected_audited": old_count,
                    "selected_accuracy": old_correct / old_count,
                    "selected_coverage": old_count / env.size ** 2,
                    "retained_correct": correct, "retained_audited": count,
                    "retained_accuracy": correct / count,
                    "retained_coverage": count / env.size ** 2,
                    "extra_correct": extra_correct, "extra_audited": extra_count,
                    "action_equal": reply["action"] == baseline[index]["action"]})
                previous = frame["world"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    cpu_ended = resource.getrusage(resource.RUSAGE_CHILDREN)
    conditions = {}
    for name in ("selected", "retained"):
        correct = sum(row[f"{name}_correct"] for row in rows)
        count = sum(row[f"{name}_audited"] for row in rows)
        conditions[name] = {"correct": correct, "audited": count,
            "pooled_accuracy": correct / count if count else None,
            "mean_frame_coverage": sum(row[f"{name}_coverage"] for row in rows) / len(rows) if rows else None,
            "final_coverage": rows[-1][f"{name}_coverage"] if rows else None}
    extra_correct = sum(row["extra_correct"] for row in rows)
    extra_count = sum(row["extra_audited"] for row in rows)
    summary = {
        "status": "complete" if error is None else "stopped", "error": error,
        "protocol_sha256": sha(OUT / "protocol.json"), "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "matched_recorded_frames": len(baseline), "compared_frames": len(rows),
        "candidate_replies": replies, "candidate_processes": 2 if len(baseline) == 79 else 1,
        "new_world_episodes": 0, "new_transitions": 0, "model_calls": 0,
        "seconds": time.monotonic() - started,
        "candidate_cpu_seconds": cpu_ended.ru_utime + cpu_ended.ru_stime - cpu_started.ru_utime - cpu_started.ru_stime,
        "conditions": conditions,
        "extra_claims": {"correct": extra_correct, "audited": extra_count,
                         "accuracy": extra_correct / extra_count if extra_count else None},
        "frames_with_extra_claims": sum(row["extra_audited"] > 0 for row in rows),
        "changed_actions": sum(not row["action_equal"] for row in rows),
        "claim_limit": protocol["inference"],
        "uncertainty": None,
    }
    write_once(OUT / "frames.json", rows)
    write_once(OUT / "summary.json", summary)
    if error:
        raise RuntimeError(error)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.prepare else run(), indent=2))


if __name__ == "__main__":
    main()
