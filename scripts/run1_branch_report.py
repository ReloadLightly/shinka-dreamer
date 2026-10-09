"""Saved-data-only review of completed RUN1 branches and validator missingness."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/run1/branches"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    started = json.loads((OUT / "execution-started.json").read_text())
    if started["protocol_sha256"] != sha(OUT / "protocol.json"):
        raise ValueError("Executed protocol changed")
    if sha(OUT / "executed-worker.py") != protocol["bound_files"]["scripts/run1_branch_worker.py"]:
        raise ValueError("Executed worker archive changed")
    selection = json.loads((OUT / "selection.json").read_text())
    futures = [json.loads(p.read_text()) for p in sorted((OUT / "futures").glob("*.json"))]
    pairs, regimes = {}, {}
    for regime in protocol["regimes"]:
        states = [s for s in selection["states"] if s["regime"] == regime]
        complete = {s["state"] for s in states if all(
            f["status"] == "complete" for f in futures if f["state"] == s["state"])}
        regimes[regime] = {"selected_states": len(states), "complete_states": len(complete),
            "selected_disagreement_states": sum(s["stratum"] == "disagreement" for s in states),
            "complete_disagreement_states": sum(s["stratum"] == "disagreement" and s["state"] in complete for s in states),
            "failed_state_ids": [s["state"] for s in states if s["state"] not in complete]}
        for left, right in (("learned", "frozen_prior"), ("known_law", "learned"), ("known_law", "frozen_prior")):
            attempted, valid, changed = [], [], []
            for future in futures:
                if future["regime"] != regime:
                    continue
                by = {label: row for row in future["branches"] for label in row["labels"]}
                if left not in by or right not in by:
                    continue
                a, b = by[left], by[right]
                attempted.append((a, b))
                if a["valid"] and b["valid"]:
                    valid.append((a, b))
                    if a["first_action"] != b["first_action"]:
                        changed.append((a, b))
            n = len(changed)
            pairs[f"{regime}/{left}_minus_{right}"] = {
                "attempted_label_pairs": len(attempted), "invalid_label_pairs": len(attempted) - len(valid),
                "valid_label_pairs": len(valid), "valid_changed_action_pairs": n,
                "return_discordant": sum(a["return"] != b["return"] for a, b in changed),
                "left_better_return": sum(a["return"] > b["return"] for a, b in changed),
                "left_worse_return": sum(a["return"] < b["return"] for a, b in changed),
                "collision_discordant": sum(a["collision"] != b["collision"] for a, b in changed),
                "escape_discordant": sum(a["escaped"] != b["escaped"] for a, b in changed),
                "changed_action_return_difference_sum": sum(a["return"] - b["return"] for a, b in changed),
                "changed_action_return_difference_mean": sum(a["return"] - b["return"] for a, b in changed) / n if n else None,
                "scope": "Conditional on valid paired executions with different first actions; missing states are not random and are not evidence of zero effect."}
    result = {"status": "completed bounded run with substantial validator missingness",
        "input_sha256": {"protocol": sha(OUT / "protocol.json"), "selection": sha(OUT / "selection.json"),
                         "execution_finished": sha(OUT / "execution-finished.json"), "executed_worker": sha(OUT / "executed-worker.py")},
        "regimes": regimes, "pairs": pairs,
        "future_status": dict(Counter(f["status"] for f in futures)),
        "unique_action_branch_outcomes": dict(Counter(b["reason"] for f in futures for b in f["branches"])),
        "failure_cause": "56 future workers rejected a restored-memory pickle-byte digest. Equal built-in values can serialize differently because set iteration/memo representation differs; a synthetic test demonstrates this. Failed candidate states were not retained, so actual structural equality for those failed states is unverified. No failed branch is retroactively validated.",
        "interpretation": "No paired collision/escape advantage observed among the6complete disagreement states. One uniform future has one fewer collected key after the learned first action. Small selected development sample,12-step horizon and5missing disagreement states limit the finding. Agreement effects are zero by action aliasing.",
        "no_reruns": True, "new_world_transitions": 0, "new_candidate_executions": 0, "new_model_calls": 0}
    path = OUT / "reviewed-summary.json"
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError("Preserving previously published saved-data review")
    path.write_text(text)
    print(json.dumps({"regimes": regimes, "future_status": result["future_status"],
                      "unique_action_branch_outcomes": result["unique_action_branch_outcomes"]}))


if __name__ == "__main__":
    main()
