"""One-shot, capped development first-action branch experiment.

Only the evaluator constructs/copies physical worlds or owns future RNGs. The
source-bound worker receives observations, recorded actions and, during state
selection only, current law probabilities for a discarded planning copy.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import resource
import selectors
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.world import stream_seed, valid_action
from dreamer.world_v3 import UnknownDynamicsMaze

LABELS = ("learned", "frozen_prior", "known_law")
PHYSICAL_FIELDS = ("grid", "terrain", "step", "health", "keys", "door_open", "feedback")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserving existing checkpoint: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Preserving interrupted checkpoint: {temporary}")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def physical(obs):
    return {key: obs[key] for key in PHYSICAL_FIELDS}


def observable_encounter(obs):
    """Eligibility uses only the supplied local terrain and occupancy."""
    if obs["step"] < 10 or not any(5 in row for row in obs["grid"]):
        return False
    terrain, occupied = obs["terrain"], obs["grid"]
    def blocked(dx, dy):
        cell = terrain[dy + 2][dx + 2]
        return cell in (-1, 1) or (cell == 3 and not obs["door_open"] and
                                  not (obs["keys"] == 2 and abs(dx) + abs(dy) == 1))
    possible = set()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if occupied[dy + 2][dx + 2] == 5 or blocked(dx, dy):
                continue
            if dx and dy and (blocked(dx, 0) or blocked(0, dy)):
                continue
            possible.add((dx, dy))
    return len(possible) >= 2


class Worker:
    def __init__(self, protocol, deadline, private_log):
        self.deadline = deadline
        self.log = Path(private_log).open("xb")
        source = protocol["source"]
        self.process = subprocess.Popen(
            ["/usr/bin/python3", "-s", "-S", str(ROOT / "scripts/run1_branch_worker.py"),
             str(ROOT / source["path"]), source["sha256"]],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            cwd="/tmp", env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"}, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        self.public_frames = self.extra_planners = 0

    def query(self, request):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Registered experiment deadline reached")
        if "obs" in request:
            self.public_frames += 1
            self.extra_planners += 2 * bool(request.get("propose"))
        self.process.stdin.write((json.dumps(request, allow_nan=False) + "\n").encode())
        end = min(self.deadline, time.monotonic() + 3.)
        while b"\n" not in self.buffer:
            if not self.selector.select(max(0., end - time.monotonic())):
                raise TimeoutError("Branch worker exceeded three-second response deadline")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError(f"Branch worker exited ({self.process.poll()})")
            self.buffer += data
            if len(self.buffer) > 2 * 1024**2:
                raise ValueError("Branch worker response too large")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def load_record(protocol, regime, case):
    name = f"{regime}-{case:04d}.json.gz"
    path = ROOT / protocol["corpus_directory"] / name
    if sha(path) != protocol["corpus_sha256"][name]:
        raise ValueError("Registered development recording changed")
    with gzip.open(path, "rt") as handle:
        record = json.load(handle)
    if (record["regime"], record["case"]) != (regime, case):
        raise ValueError("Development recording identity differs")
    return name, record["frames"][:protocol["frames_per_recording_cap"]]


def select_states(protocol, out, private, deadline, seeds):
    states, attempts = [], []
    operations = reconstructed = 0
    for regime in protocol["regimes"]:
        quota = Counter()
        for case in protocol["cases"]:
            identity = f"{regime}-{case:04d}"
            if time.monotonic() >= deadline:
                attempts.append({"regime": regime, "case": case, "status": "unstarted_deadline"})
                continue
            name, frames = load_record(protocol, regime, case)
            env = UnknownDynamicsMaze(seeds[case], regime=regime)
            local, last_selected, last_action = 0, -100, None
            worker = Worker(protocol, deadline, private / f"select-{identity}.stderr")
            error = None
            started = time.monotonic()
            returned = 0
            try:
                for index, frame in enumerate(frames):
                    if physical(env.observe()) != physical(frame["obs"]):
                        raise ValueError("Reconstructed public observation differs from saved recording")
                    eligible = (local < 2 and index - last_selected >= 8 and
                                observable_encounter(frame["obs"]) and
                                (quota["agreement"] < 4 or quota["disagreement"] < 4))
                    cost = 3 if eligible else 1
                    if operations + cost > protocol["candidate_operation_cap"]:
                        raise RuntimeError("Registered candidate-operation cap reached")
                    operations += cost
                    request = {"obs": physical(frame["obs"]), "last_action": last_action}
                    if eligible:
                        request.update(propose=True, known_law=list(env.current_law))
                    reply = worker.query(request)
                    returned += 1
                    if not valid_action(reply.get("action")):
                        raise ValueError("Invalid selected recommendation")
                    if eligible:
                        actions = reply["actions"]
                        if set(actions) != set(LABELS) or not all(valid_action(a) for a in actions.values()):
                            raise ValueError("Invalid replacement-law recommendation")
                        category = "agreement" if len({digest(a) for a in actions.values()}) == 1 else "disagreement"
                        if quota[category] < 4:
                            state_id = f"{regime}-{case:04d}-{index:03d}"
                            snapshot_path = private / "states" / f"{state_id}.pickle.gz"
                            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                            with gzip.open(snapshot_path, "xb") as handle:
                                pickle.dump(copy.deepcopy(env), handle, protocol=4)
                            state = {"state": state_id, "regime": regime, "case": case,
                                     "frame_index": index, "observation_step": frame["obs"]["step"],
                                     "stratum": category, "actions": actions,
                                     "checkpoint_sha256": reply["checkpoint_sha256"],
                                     "observable_enemy_cells": sum(row.count(5) for row in frame["obs"]["grid"]),
                                     "learning_updates": reply["learning_updates"],
                                     "effective_learning_updates": reply["effective_learning_updates"],
                                     "replacement_outcome_observed": env.switch_step is not None and env.steps >= env.switch_step,
                                     "corpus_sha256": protocol["corpus_sha256"][name],
                                     "private_snapshot_sha256": sha(snapshot_path)}
                            states.append(state)
                            save(out / "states" / f"{state_id}.json", state)
                            local += 1
                            quota[category] += 1
                            last_selected = index
                    # One exact recorded transition, no novel policy rollout.
                    env.step(frame["action"])
                    reconstructed += 1
                    last_action = frame["action"]
                    if local == 2 or (quota["agreement"] == 4 and quota["disagreement"] == 4) or env.done:
                        break
                finish = worker.query({"finish": True})
            except Exception as exc:
                error, finish = f"{type(exc).__name__}: {exc}", None
            finally:
                worker.close()
            attempts.append({"regime": regime, "case": case, "status": "failed" if error else "complete",
                             "returned_frames": returned, "candidate_operations": worker.public_frames + worker.extra_planners,
                             "selected_states": local, "error": error, "finish": finish,
                             "elapsed_seconds": time.monotonic() - started})
            save(out / "selection-attempts" / f"{identity}.json", attempts[-1])
    selection = {"states": states, "attempts": attempts, "candidate_operations": operations,
                 "reconstructed_transitions": reconstructed, "recording_attempts": len(attempts),
                 "quota_counts": {r: {s: sum(x["regime"] == r and x["stratum"] == s for x in states)
                                      for s in ("agreement", "disagreement")} for r in protocol["regimes"]}}
    save(out / "selection.json", selection)
    return selection


def branch_return(reason, keys_delta, opened, transitions):
    return (-1. if reason == "caught" else 0.) + (1. if reason == "escaped" else 0.) + .25 * keys_delta + .25 * opened - .01 * transitions


def execute_future(protocol, state, replication, out, private, deadline):
    key = f"{state['state']}--{replication:02d}"
    started = time.monotonic()
    identity = {"state": state["state"], "regime": state["regime"], "case": state["case"],
                "stratum": state["stratum"], "replication": replication}
    if time.monotonic() >= deadline:
        result = {**identity, "status": "unstarted_deadline", "branches": [],
                  "error": None, "finish": None, "actual_world_transitions": 0,
                  "candidate_operations": 0, "elapsed_seconds": 0.}
        save(out / "futures" / f"{key}.json", result)
        return result
    save(out / "attempts" / f"{key}.started.json", identity)
    _, frames = load_record(protocol, state["regime"], state["case"])
    snapshot_path = private / "states" / f"{state['state']}.pickle.gz"
    if sha(snapshot_path) != state["private_snapshot_sha256"]:
        raise ValueError("Private physical checkpoint changed")
    with gzip.open(snapshot_path, "rb") as handle:
        original = pickle.load(handle)
    groups = {}
    for label in LABELS:
        action = state["actions"][label]
        groups.setdefault(digest(action), {"action": action, "labels": []})["labels"].append(label)
    worker = Worker(protocol, deadline, private / f"branch-{key}.stderr")
    rows, error, finish = [], None, None
    transitions = 0
    try:
        last = None
        for index, frame in enumerate(frames[:state["frame_index"] + 1]):
            response = worker.query({"obs": physical(frame["obs"]), "last_action": last,
                                     "save_checkpoint": index == state["frame_index"]})
            last = frame["action"]
        if response["checkpoint_sha256"] != state["checkpoint_sha256"]:
            raise ValueError("Warmup did not reproduce the identical learned checkpoint")
        for group in groups.values():
            reset = worker.query({"restore": True})
            if reset["memory_sha256"] != state["checkpoint_sha256"]:
                raise ValueError("Branch did not restore common learned memory")
            env = copy.deepcopy(original)
            innovation_seed = stream_seed(protocol["future_stream_key"], f"run1:branch:{state['state']}:{replication}")
            env.enemy_rng.rng = random.Random(innovation_seed)
            env.enemy_rng.draws = 0
            first_keys, first_door = env.keys_collected, env.door_open
            action, count, actions = group["action"], 0, []
            for step in range(protocol["branch_horizon"]):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Registered branch deadline reached")
                env.step(action)
                count += 1
                transitions += 1
                actions.append(action)
                if env.done or step + 1 == protocol["branch_horizon"]:
                    break
                reply = worker.query({"obs": env.observe(), "last_action": action})
                action = reply["action"]
                if not valid_action(action):
                    raise ValueError("Invalid learned continuation action")
            reason = env.reason if env.done else "window_complete"
            row = {"labels": group["labels"], "first_action": group["action"],
                   "valid": True, "error": None, "reason": reason, "transitions": count,
                   "collision": reason == "caught", "escaped": reason == "escaped",
                   "failure_or_invalid": reason == "caught", "keys_delta": env.keys_collected - first_keys,
                   "door_opened": bool(env.door_open and not first_door),
                   "action_sequence_sha256": digest(actions), "enemy_draws": env.enemy_rng.draws}
            row["return"] = branch_return(reason, row["keys_delta"], row["door_opened"], count)
            if env.enemy_rng.draws != 3 * count:
                raise ValueError("Enemy innovation consumption differs from fixed three draws per transition")
            rows.append(row)
            save(out / "branches" / f"{key}--{group['labels'][0]}.json", {**identity, **row})
        finish = worker.query({"finish": True})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        worker.close()
    completed = {label for row in rows for label in row["labels"]}
    for group in groups.values():
        if any(label not in completed for label in group["labels"]):
            rows.append({"labels": group["labels"], "first_action": group["action"],
                         "valid": False, "error": error or "Unavailable branch", "reason": "invalid",
                         "transitions": None, "collision": False, "escaped": False,
                         "failure_or_invalid": True, "keys_delta": None, "door_opened": None,
                         "return": -1.12})
    result = {**identity, "status": "failed" if error else "complete", "branches": rows,
              "error": error, "finish": finish, "actual_world_transitions": transitions,
              "candidate_operations": worker.public_frames + worker.extra_planners,
              "elapsed_seconds": time.monotonic() - started}
    save(out / "futures" / f"{key}.json", result)
    return result


def summarize(protocol, selection, results):
    regimes = {}
    for regime in protocol["regimes"]:
        state_results = []
        for state in [s for s in selection["states"] if s["regime"] == regime]:
            futures = [r for r in results if r["state"] == state["state"]]
            rows = {label: [] for label in LABELS}
            for future in futures:
                for row in future["branches"]:
                    for label in row["labels"]:
                        rows[label].append(row)
            outcomes = {}
            for label, records in rows.items():
                n = len(records)
                outcomes[label] = {"attempted_futures": n, "invalid": sum(not r["valid"] for r in records),
                                   "collision": sum(r["collision"] for r in records),
                                   "escaped": sum(r["escaped"] for r in records),
                                   "return_mean": sum(r["return"] for r in records) / n if n else None,
                                   "failure_or_invalid_rate": sum(r["failure_or_invalid"] for r in records) / n if n else None,
                                   "keys_gained_sum": sum(r["keys_delta"] or 0 for r in records),
                                   "doors_opened": sum(bool(r["door_opened"]) for r in records),
                                   "executed_transitions": sum(r["transitions"] or 0 for r in records)}
            pairs = {}
            for left, right in (("learned", "frozen_prior"), ("known_law", "learned"), ("known_law", "frozen_prior")):
                a, b = outcomes[left], outcomes[right]
                if a["attempted_futures"] and b["attempted_futures"]:
                    pairs[f"{left}_minus_{right}"] = {"return": a["return_mean"] - b["return_mean"],
                        "failure_or_invalid": a["failure_or_invalid_rate"] - b["failure_or_invalid_rate"]}
            state_results.append({"state": state["state"], "case": state["case"], "stratum": state["stratum"],
                                  "actions": state["actions"], "outcomes": outcomes, "paired_effects": pairs})
        aggregate = {}
        for pair in ("learned_minus_frozen_prior", "known_law_minus_learned", "known_law_minus_frozen_prior"):
            for stratum in ("agreement", "disagreement", "all"):
                eligible = [r for r in state_results if pair in r["paired_effects"] and (stratum == "all" or r["stratum"] == stratum)]
                aggregate[f"{pair}/{stratum}"] = {"states": len(eligible),
                    "precursor_recordings": len({r["case"] for r in eligible}),
                    "mean_return_effect": sum(r["paired_effects"][pair]["return"] for r in eligible) / len(eligible) if eligible else None,
                    "mean_failure_or_invalid_effect": sum(r["paired_effects"][pair]["failure_or_invalid"] for r in eligible) / len(eligible) if eligible else None}
                entry = aggregate[f"{pair}/{stratum}"]
                cases = sorted({r["case"] for r in eligible})
                if len(cases) >= 2:
                    rng = random.Random(stream_seed(20261009, f"run1:branch-bootstrap:{regime}:{pair}:{stratum}"))
                    draws = []
                    for _ in range(2000):
                        resampled = [r for case in rng.choices(cases, k=len(cases))
                                     for r in eligible if r["case"] == case]
                        draws.append(sum(r["paired_effects"][pair]["return"] for r in resampled) / len(resampled))
                    draws.sort()
                    entry["return_cluster_bootstrap_percentile95"] = [draws[49], draws[1949]]
                else:
                    entry["return_cluster_bootstrap_percentile95"] = None
                entry["uncertainty_scope"] = "Descriptive recording-cluster bootstrap conditional on selected encounter states; at most4 clusters/regime. Sparse/zero differences can give degenerate intervals and do not prove equivalence."
        regimes[regime] = {"states": state_results, "paired_effects": aggregate}
    return {"status": "bounded development action-consequence diagnostic", "regimes": regimes,
            "scope": "Selected visible encounter states on reused development experience; finite12-step first-action effect under a common learned continuation policy. Not whole-episode learning benefit or independent discovery.",
            "invalid_policy": "Invalid branch retained with return -1.12 and failure_or_invalid=1; invalid is separately reported and never called observed collision.",
            "quota_counts": selection["quota_counts"], "future_attempts": len(results),
            "unique_action_branches": sum(len(r["branches"]) for r in results),
            "invalid_unique_action_branches": sum(not b["valid"] for r in results for b in r["branches"]),
            "completed_branch_transitions": sum(r["actual_world_transitions"] for r in results),
            "reconstructed_transitions": selection["reconstructed_transitions"],
            "candidate_operations": selection["candidate_operations"] + sum(r["candidate_operations"] for r in results),
            "model_calls": 0, "assessment_cases_used": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="artifacts/run1/branches/protocol.json")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text())
    if (protocol["regimes"] != ["uniform", "stationary", "switch"] or
            protocol["cases"] != [0, 1, 2, 3] or protocol["frames_per_recording_cap"] != 80 or
            protocol["future_replicates"] != 8 or protocol["branch_horizon"] != 12 or
            protocol["candidate_operation_cap"] > 22272 or protocol["branch_transition_cap"] > 6912 or
            protocol["reconstruction_transition_cap"] > 960 or protocol["workers"] > 2):
        raise ValueError("Protocol exceeds the explicitly authorized bounded branch design")
    out, private = ROOT / protocol["public_output"], ROOT / protocol["private_output"]
    out.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    lock = (private / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "execution-started.json").exists():
        raise RuntimeError("Already attempted; no automatic rerun or budget extension")
    for relative, expected in protocol["bound_files"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Registered source changed: {relative}")
    if sha(ROOT / protocol["private_pool"]) != protocol["private_pool_sha256"]:
        raise ValueError("Registered development pool changed")
    seeds = json.loads((ROOT / protocol["private_pool"]).read_text())
    started_utc = datetime.now(timezone.utc)
    remaining = (datetime.fromisoformat(protocol["execution_deadline_utc"]) - started_utc).total_seconds()
    if remaining <= 0:
        raise RuntimeError("Registered execution deadline expired")
    start, cpu = time.monotonic(), time.process_time()
    deadline = start + min(remaining, protocol["maximum_execution_seconds"])
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    save(out / "execution-started.json", {"started_utc": started_utc.isoformat(),
        "protocol_sha256": sha(protocol_path), "bound_files": protocol["bound_files"],
        "remaining_execution_seconds": deadline - start, "model_calls": 0})
    selection = select_states(protocol, out, private, deadline, seeds)
    jobs, unstarted, reserved = [], [], selection["candidate_operations"]
    for state in selection["states"]:
        unique = len({digest(a) for a in state["actions"].values()})
        cost = state["frame_index"] + 1 + unique * (protocol["branch_horizon"] - 1)
        for replication in range(protocol["future_replicates"]):
            if reserved + cost > protocol["candidate_operation_cap"]:
                unstarted.append({"state": state["state"], "replication": replication, "reason": "candidate_operation_cap"})
            else:
                jobs.append((state, replication))
                reserved += cost
    results = []
    with ThreadPoolExecutor(max_workers=min(2, protocol["workers"])) as pool:
        futures = {pool.submit(execute_future, protocol, state, replication, out, private, deadline): (state, replication)
                   for state, replication in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"future_attempts_complete": len(results), "planned": len(jobs),
                              "status": result["status"], "state": result["state"]}), flush=True)
    summary = summarize(protocol, selection, results)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    summary.update(unstarted_future_attempts=unstarted,
                   elapsed_seconds=time.monotonic() - start,
                   controller_cpu_seconds=time.process_time() - cpu,
                   worker_cpu_seconds=child.ru_utime + child.ru_stime - child_before.ru_utime - child_before.ru_stime,
                   finished_utc=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=sha(protocol_path))
    if (summary["completed_branch_transitions"] > protocol["branch_transition_cap"] or
            summary["reconstructed_transitions"] > protocol["reconstruction_transition_cap"] or
            summary["candidate_operations"] > protocol["candidate_operation_cap"]):
        raise RuntimeError("Recorded budget invariant violated")
    save(out / "summary.json", summary)
    save(out / "execution-finished.json", {k: v for k, v in summary.items() if k != "regimes"})


if __name__ == "__main__":
    main()
