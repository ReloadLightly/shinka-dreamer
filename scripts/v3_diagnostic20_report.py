"""Summarize completed development replay diagnostics; never executes agents."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
from visual_theme import apply_theme, save_figure, COBALT, MAGENTA, ORANGE, SECONDARY
import matplotlib.pyplot as plt

PHASES = ("world_model_step", "planner", "export_model")
REGIMES = ("uniform", "stationary", "switch")
EPS = 1e-12


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decisions(rows, post_only=False):
    totals = Counter()
    gaps, risks = [], []
    for row in rows:
        changed_episode = False
        threshold = row["requested_frames"] - row.get("post_switch_recorded_frames", 0)
        for frame in row["frames"]:
            if post_only and frame["frame"] < threshold:
                continue
            totals["frames"] += 1
            changed = frame["action"] != frame["known_action"]
            changed_episode |= changed
            totals["changed_chosen_actions"] += changed
            rank = frame.get("ranking")
            if not rank:
                totals["ranking_unavailable_frames"] += 1
                continue
            a, b = rank["baseline"], rank["known_law"]
            same_target = a["target"] == b["target"]
            totals["same_navigation_target" if same_target else "different_navigation_target"] += 1
            totals["changed_action_same_target" if same_target else "changed_action_different_target"] += changed
            left = [tuple(c["move"]) for c in a["choices"]]
            right = [tuple(c["move"]) for c in b["choices"]]
            totals["full_rank_order_changes"] += left != right
            totals["candidate_set_changes"] += set(left) != set(right)
            totals["predictive_flag_disabled_frames"] += not (a["predictive_planning_enabled"] and b["predictive_planning_enabled"])
            known = {tuple(c["move"]): c for c in b["choices"]}
            base = known.get(tuple(frame["action"]["move"]))
            chosen = known.get(tuple(frame["known_action"]["move"]))
            if not base or not chosen or any(c[k] is None for c in (base, chosen) for k in ("score", "immediate")):
                totals["known_score_comparison_unavailable"] += 1
                continue
            gap, risk = base["score"] - chosen["score"], base["immediate"] - chosen["immediate"]
            gaps.append(gap); risks.append(risk)
            for name, value in (("known_score_gap", gap), ("known_immediate_risk_delta", risk)):
                sign = "positive" if value > EPS else "negative" if value < -EPS else "zero"
                totals[name + "_" + sign] += 1
                if changed:
                    totals["changed_action_" + name + "_" + sign] += 1
            both_zero = abs(base["immediate"]) <= EPS and abs(chosen["immediate"]) <= EPS
            totals["both_chosen_immediate_risks_zero"] += both_zero
            totals["changed_action_both_immediate_risks_zero"] += changed and both_zero
        totals["episodes_with_action_changes"] += changed_episode
    for key in ("frames", "changed_chosen_actions", "episodes_with_action_changes", "same_navigation_target",
                "different_navigation_target", "full_rank_order_changes", "candidate_set_changes",
                "known_score_comparison_unavailable", "both_chosen_immediate_risks_zero",
                "changed_action_both_immediate_risks_zero", "changed_action_same_target", "changed_action_different_target"):
        totals.setdefault(key, 0)
    for prefix in ("known_score_gap", "known_immediate_risk_delta", "changed_action_known_score_gap", "changed_action_known_immediate_risk_delta"):
        for sign in ("positive", "zero", "negative"):
            totals.setdefault(prefix + "_" + sign, 0)
    return {**totals, "recorded_episode_passes": len(rows),
            "complete_episode_passes": sum(r["status"] == "complete" for r in rows),
            "known_score_gap_sum": sum(gaps), "known_score_gap_mean": sum(gaps) / len(gaps) if gaps else None,
            "known_immediate_risk_delta_sum": sum(risks),
            "known_immediate_risk_delta_mean": sum(risks) / len(risks) if risks else None}


def summarize(directory):
    finish_path = directory / "execution-finished.json"
    if not finish_path.exists():
        raise RuntimeError("Execution is still pending; this report cannot start or resume it")
    finish, start, protocol = read(finish_path), read(directory / "execution-started.json"), read(directory / "protocol.json")
    if start["protocol_sha256"] != sha(directory / "protocol.json"):
        raise ValueError("Diagnostic protocol hash mismatch")
    rows = []
    for name, digest in finish["pass_file_sha256"].items():
        path = directory / "passes" / name
        if sha(path) != digest:
            raise ValueError("Saved pass hash mismatch: " + name)
        rows.append(read(path))
    paired, by = [], {(r["kind"], r["regime"], r["case"], r["profiled"]): r for r in rows}
    for kind, regime, case in sorted({k[:3] for k in by}):
        a, b = by.get((kind, regime, case, False)), by.get((kind, regime, case, True))
        pairs = list(zip(a["frames"], b["frames"])) if a and b else []
        main = sum(x["action_model_sha256"] == y["action_model_sha256"] for x, y in pairs)
        known = sum(x["known_action"] == y["known_action"] for x, y in pairs) if kind == "comparator" else None
        complete = bool(a and b and a["status"] == b["status"] == "complete")
        paired.append({"kind": kind, "regime": regime, "case": case, "common_frames": len(pairs),
                       "both_passes_complete": complete, "identical_main_action_export_frames": main,
                       "identical_known_action_frames": known,
                       "equivalent": complete and main == len(pairs) and (known is None or known == len(pairs))})
    profiles = {}
    for kind in ("selected", "comparator"):
        group = [r for r in rows if r["kind"] == kind and r["profiled"] and r.get("finish")]
        profiles[kind] = {"profiled_passes_with_finish": len(group), "phases": {}}
        for phase in PHASES:
            funcs = defaultdict(lambda: Counter())
            for row in group:
                for f in row["finish"]["profiles"][phase]:
                    key = (f["file"], f["line"], f["function"])
                    funcs[key].update({k: f[k] for k in ("calls", "primitive_calls", "self_seconds", "cumulative_seconds")})
            top = [{"file": k[0], "line": k[1], "function": k[2], **v} for k, v in funcs.items()]
            top.sort(key=lambda r: -r["cumulative_seconds"])
            profiles[kind]["phases"][phase] = {
                "process_seconds": sum(r["finish"]["phase_totals"][phase]["process_seconds"] for r in group),
                "calls": sum(r["finish"]["phase_totals"][phase]["calls"] for r in group),
                "top_functions_from_retained_rows": top[:8]}
    plain = [r for r in rows if r["kind"] == "comparator" and not r["profiled"]]
    complete_profiled = [r for r in rows if r["kind"] == "comparator" and r["profiled"] and r["status"] == "complete"]
    action_changes = {}
    for regime in REGIMES:
        group = [r for r in complete_profiled if r["regime"] == regime]
        action_changes[regime] = {"frames": sum(len(r["frames"]) for r in group),
            "changed_chosen_actions": sum(f["action"] != f["known_action"] for r in group for f in r["frames"]),
            "recordings": len(group), "recordings_with_changes": sum(any(f["action"] != f["known_action"] for f in r["frames"]) for r in group)}
    post_actions = decisions([r for r in complete_profiled if r["regime"] == "switch"], True)
    return {"scope": "Development-only saved memory-policy experience; diagnostic actions were not executed. No control-performance inference.",
        "execution": finish, "protocol_sha256": sha(directory / "protocol.json"), "report_source_sha256": sha(Path(__file__)),
        "attempt_status_counts": dict(Counter(r["status"] for r in rows)), "requested_passes": start["attempt_slots"],
        "requested_frames": start["requested_frames"], "returned_frames": sum(len(r["frames"]) for r in rows),
        "unreturned_frames": start["requested_frames"] - sum(len(r["frames"]) for r in rows),
        "failed_passes": [{k: r.get(k) for k in ("kind", "regime", "case", "profiled", "status", "returncode", "requested_frames")} | {"returned_frames": len(r["frames"])} for r in rows if r["status"] != "complete"],
        "returned_frames_by_kind_and_mode": {kind+"/"+mode: sum(len(r["frames"]) for r in rows if r["kind"] == kind and r["profiled"] == flag) for kind in ("selected", "comparator") for mode, flag in (("plain", False), ("profiled", True))},
        "unique_recordings": len({(r["regime"], r["case"]) for r in rows}), "paired_instrumentation": paired,
        "all_instrumentation_pairs_complete_and_equal": all(r["equivalent"] for r in paired),
        "instrumentation_complete_pairs": sum(r["both_passes_complete"] for r in paired),
        "all_shared_prefix_main_hashes_equal": all(r["identical_main_action_export_frames"] == r["common_frames"] for r in paired),
        "all_shared_prefix_known_actions_equal": all(r["identical_known_action_frames"] == r["common_frames"] for r in paired if r["kind"] == "comparator"),
        "action_changes": action_changes,
        "complete_profiled_post_switch_action_changes": {k: post_actions[k] for k in ("frames", "changed_chosen_actions", "episodes_with_action_changes", "recorded_episode_passes")},
        "action_change_scope": "Complete profiled comparator passes provide all recorded-prefix action recommendations. Rank/margin diagnostics separately use returned plain-pass frames, including partial prefixes from CPU-limit failures; missing ranks are not imputed.",
        "profiles": profiles, "decisions": {regime: decisions([r for r in plain if r["regime"] == regime]) for regime in REGIMES},
        "post_switch_observation_frames": decisions([r for r in plain if r["regime"] == "switch"], True),
        "known_law_planner_calls_observed_in_returned_frames": sum(len(r["frames"]) for r in rows if r["kind"] == "comparator"),
        "profile_scope": "Only profiled main WM/planner/export calls; known-law clone, copying and recorded-action resets are excluded. Each pass retained only its top15 functions by cumulative time per phase. Aggregated function rows are truncated evidence, not full function self-time totals; nested cumulative times must not be added.",
        "timing_scope": "Instrumentation costs, not an uninstrumented speed benchmark. Plain comparator includes ranking-trace overhead; profiled passes include cProfile overhead. Worker and phase CPU overlap and are not added.",
        "decision_scope": "Known-law minus fitted dynamics at the same fitted hidden-occupancy/map/location/history state. Known-score gap=score(baseline choice)-score(known choice), both under known-law scores. Immediate delta likewise compares both choices under known-law occupancy risk, not calibrated death risk. Navigation targets may change. Rank changes compare ordered eligible moves, not merely changed score values.",
        "post_switch_scope": "Uses the final logged post-switch-observation frames of each requested prefix (obs.step>=switch). The first forecast governed by the replacement law occurs one observation earlier; this diagnostic does not relabel that boundary. Failed prefixes contribute only returned frames meeting that cutoff.",
        "numerical_zero_tolerance": EPS, "missing_profile_scope": "Failed passes without finish retain returned decisions but have no complete phase/function timing; do not treat missing timing as zero.",
        "new_world_episodes": 0, "experiment_model_calls": 0, "assessment_resumed": False,
        "limitations": [protocol["corpus"], "Four recordings per regime, capped at80 observations each; dependent frames are not independent samples. No significance tests.", "Improving the planner's own score is mechanically expected after recomputing its optimum and does not prove actual safety or escape benefit."]}


def render(result, directory):
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": .40})
    fig.subplots_adjust(top=.78, bottom=.28)
    for index, kind in enumerate(("selected", "comparator")):
        left = 0
        for phase, color in zip(PHASES, (COBALT, MAGENTA, ORANGE)):
            value = result["profiles"][kind]["phases"][phase]["process_seconds"]
            axes[0].barh(index, value, left=left, color=color, label=phase.replace("_", " ") if index == 0 else None)
            left += value
    axes[0].set(yticks=[0, 1], yticklabels=["Selected program", "Fitted comparator"], xlabel="Profiled main-phase CPU seconds", title="Instrumented cost by phase")
    axes[0].invert_yaxis()
    for index, regime in enumerate(REGIMES):
        row = result["action_changes"][regime]; value = 100 * row["changed_chosen_actions"] / row["frames"] if row["frames"] else 0
        axes[1].barh(index, value, color=(SECONDARY, COBALT, MAGENTA)[index])
        axes[1].text(value+.2, index, f"{row['changed_chosen_actions']}/{row['frames']}", va="center", fontsize=10)
    axes[1].set(yticks=range(3), yticklabels=[r.title() for r in REGIMES], xlabel="Recorded states with changed choice (%)", title="Known law in the same fitted state")
    axes[1].invert_yaxis(); axes[1].margins(x=.35)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.90), ncol=3, fontsize=9)
    fig.suptitle("Bounded development diagnostic · saved experience only", weight="bold", fontsize=15, y=1.01)
    fig.text(.125,.025, f"{result['execution']['complete_passes']}/{result['requested_passes']} passes complete · {result['execution']['failed_passes']} partial ranking passes · {result['unreturned_frames']} rank frames unavailable\n{result['unique_recordings']} reused recordings · {result['returned_frames']:,} returned pass-frames · zero new worlds or experiment-model calls\nProfiling overhead is included; changed recommendations are not measured control gains.", fontsize=9, color=SECONDARY)
    save_figure(fig, directory / "summary-figure")
    lines = ["# Bounded development replay diagnostic", "", result["scope"], "",
        f"Completed passes: {result['execution']['complete_passes']}/{result['requested_passes']}; failed: {result['execution']['failed_passes']}; unstarted: {result['execution']['unstarted_passes']}. Returned frames: {result['returned_frames']}/{result['requested_frames']}; missing: {result['unreturned_frames']}. Complete plain/profiled pairs: {result['instrumentation_complete_pairs']}/{len(result['paired_instrumentation'])}. Shared-prefix main action/export hashes equal: {result['all_shared_prefix_main_hashes_equal']}; comparator known actions equal: {result['all_shared_prefix_known_actions_equal']}.", "", result["action_change_scope"], "",
        "| Regime | Changed choices / complete profiled frames | Returned plain rank frames | Changed choices / rank frames | Recordings with changed choices | Same / different target | Changed full rank order | Changed choices with both immediate risks zero |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for regime, r in result["decisions"].items():
        full=result["action_changes"][regime]
        lines.append(f"| {regime} | {full['changed_chosen_actions']}/{full['frames']} | {r['frames']} | {r['changed_chosen_actions']}/{r['frames']} | {r['episodes_with_action_changes']}/{r['recorded_episode_passes']} | {r['same_navigation_target']} / {r['different_navigation_target']} | {r['full_rank_order_changes']} | {r['changed_action_both_immediate_risks_zero']} |")
    lines += ["", result["decision_scope"], "", "| Regime | Known-score gap + / 0 / − | Immediate-risk delta + / 0 / − | Mean known-score gap |", "|:--|--:|--:|--:|"]
    for regime, r in result["decisions"].items():
        signs = lambda key: " / ".join(str(r[key+"_"+s]) for s in ("positive", "zero", "negative"))
        lines.append(f"| {regime} | {signs('known_score_gap')} | {signs('known_immediate_risk_delta')} | {r['known_score_gap_mean']} |")
    lines += ["", result["profile_scope"], "", result["timing_scope"], "", "| Source | Phase | Calls | Profiled CPU s | Largest retained cumulative-time functions |", "|:--|:--|--:|--:|:--|"]
    for kind, group in result["profiles"].items():
        for phase, r in group["phases"].items():
            top = "; ".join(f"{f['function']} ({f['cumulative_seconds']:.3f}s)" for f in r["top_functions_from_retained_rows"][:4])
            lines.append(f"| {kind} | {phase} | {r['calls']} | {r['process_seconds']:.3f} | {top} |")
    lines += ["", result["post_switch_scope"], "", *result["limitations"]]
    (directory / "table.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/campaign-v3/development-diagnostic20")
    directory = parser.parse_args().directory
    result = summarize(directory)
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    render(result, directory)
    print(json.dumps({"complete_passes": result["execution"]["complete_passes"], "returned_frames": result["returned_frames"], "instrumentation_equal": result["all_instrumentation_pairs_complete_and_equal"], "decisions": result["decisions"]}))


if __name__ == "__main__":
    main()
