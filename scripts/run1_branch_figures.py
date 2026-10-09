"""Chromatic Field figures from closed RUN1 branch checkpoints; no execution."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chromatic_fields import (apply_theme, save_figure, BACKGROUND, TEXT,
                             SECONDARY, COBALT, MAGENTA, ORANGE, RULE, MUTED)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DATA = ROOT / "artifacts/run1/branches"
OUT = DATA / "figures"
REGIMES = ("uniform", "stationary", "switch")
PAIRS = (("learned", "frozen_prior"), ("known_law", "learned"), ("known_law", "frozen_prior"))
NAMES = {"learned": "Learned", "frozen_prior": "Prior", "known_law": "Known"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table_rows(selection, futures):
    states, components = [], []
    for state in selection["states"]:
        group = [f for f in futures if f["state"] == state["state"]]
        for left, right in PAIRS:
            matched, attempted = [], []
            for future in group:
                by = {label: row for row in future["branches"] for label in row["labels"]}
                if left not in by or right not in by:
                    continue
                a, b = by[left], by[right]
                attempted.append((a, b))
                if a["valid"] and b["valid"]:
                    matched.append((a, b))
            n = len(matched)
            fields = {"regime": state["regime"], "case": state["case"], "state": state["state"],
                      "step": state["observation_step"], "stratum": state["stratum"],
                      "left": left, "right": right,
                      "different_first_actions": state["actions"][left] != state["actions"][right],
                      "attempted_paired_futures": len(attempted), "valid_paired_futures": n,
                      "invalid_paired_futures": len(attempted) - n}
            for name, field in (("return", "return"), ("collision", "collision"),
                                ("escape", "escaped"), ("keys", "keys_delta"),
                                ("door", "door_opened"), ("transitions", "transitions")):
                fields[f"mean_{name}_difference_valid_pairs"] = sum(a[field] - b[field] for a, b in matched) / n if n else None
            fields["scope"] = "State mean over valid paired futures only; unavailable is not zero; cases shared across states/regimes."
            states.append(fields)
    for regime in REGIMES:
        for left, right in PAIRS:
            rows = []
            invalid = 0
            for future in futures:
                if future["regime"] != regime:
                    continue
                by = {label: row for row in future["branches"] for label in row["labels"]}
                if left not in by or right not in by:
                    continue
                a, b = by[left], by[right]
                if not a["valid"] or not b["valid"]:
                    invalid += 1
                elif a["first_action"] != b["first_action"]:
                    rows.append((a, b))
            row = {"regime": regime, "left": left, "right": right,
                   "valid_changed_action_future_pairs": len(rows),
                   "invalid_future_pairs_all_selected_states": invalid}
            for name, field, benefit in (("collision", "collision", -1), ("escape", "escaped", 1),
                                         ("keys", "keys_delta", 1), ("door", "door_opened", 1),
                                         ("return", "return", 1)):
                row[f"{name}_left_better"] = sum(benefit * (a[field] - b[field]) > 0 for a, b in rows)
                row[f"{name}_left_worse"] = sum(benefit * (a[field] - b[field]) < 0 for a, b in rows)
                row[f"{name}_equal"] = sum(a[field] == b[field] for a, b in rows)
            row["scope"] = "Exact descriptive counts on valid changed-action pairs; futures are dependent within states/cases."
            components.append(row)
    cases = []
    for regime in REGIMES:
        for case in range(4):
            for left, right in PAIRS:
                group = [r for r in states if (r["regime"], r["case"], r["left"], r["right"]) == (regime, case, left, right)]
                valid = [r for r in group if r["valid_paired_futures"] == 8]
                cases.append({"regime": regime, "case": case, "left": left, "right": right,
                              "selected_states": len(group), "complete_states": len(valid),
                              "unavailable_states": len(group) - len(valid),
                              "mean_return_difference_complete_states": sum(r["mean_return_difference_valid_pairs"] for r in valid) / len(valid) if valid else None,
                              "scope": "Equal state weights within precursor case; complete states only, missingness explicit; no independent-future uncertainty."})
    return states, cases, components


def render(selection, state_rows, components):
    apply_theme()
    fig = plt.figure(figsize=(12.6, 9.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.07, 1.], height_ratios=[1., 1.06],
                           left=.055, right=.96, top=.855, bottom=.19, wspace=.57, hspace=.53)
    coverage, effects, counts = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[:, 1]), fig.add_subplot(grid[1, 0])
    fig.suptitle("First-action consequences on selected development states", x=.055, ha="left", y=.975,
                 fontsize=16, fontweight="bold")
    fig.text(.055, .937, "23 selected states · 16 complete · 7 validator rejections · original v3 development data",
             fontsize=11, color=SECONDARY)

    lookup = {(r["state"], r["left"], r["right"]): r for r in state_rows}
    y, labels = 0, []
    for regime in REGIMES:
        for stratum in ("agreement", "disagreement"):
            selected = [s for s in selection["states"] if s["regime"] == regime and s["stratum"] == stratum]
            complete = sum(lookup[s["state"], "learned", "frozen_prior"]["valid_paired_futures"] == 8 for s in selected)
            failed, unfilled = len(selected) - complete, 4 - len(selected)
            coverage.barh(y, complete, height=.63, color=COBALT)
            coverage.barh(y, failed, left=complete, height=.63, color=MUTED, edgecolor=SECONDARY, hatch="////", linewidth=.5)
            coverage.barh(y, unfilled, left=complete + failed, height=.63, color=BACKGROUND, edgecolor=RULE, linewidth=1.)
            for start, value, color in ((0, complete, BACKGROUND), (complete, failed, TEXT), (complete + failed, unfilled, SECONDARY)):
                if value:
                    coverage.text(start + value / 2, y, str(value), color=color, va="center", ha="center", fontsize=10, fontweight="bold")
            labels.append(regime.title() + (" · agree" if stratum == "agreement" else " · differ"))
            y += 1
        y += .24
    positions = [0, 1, 2.24, 3.24, 4.48, 5.48]
    coverage.set(yticks=positions, yticklabels=labels, xlim=(0, 4), xticks=range(5), xlabel="Selected states (registered quota: four / group)")
    coverage.invert_yaxis()
    coverage.set_title("A   Coverage and unavailable consequences", loc="left", pad=36, fontsize=12, fontweight="bold")
    coverage.legend(handles=[Patch(color=COBALT, label="Complete"),
                             Patch(facecolor=MUTED, edgecolor=SECONDARY, hatch="////", label="Validator failure"),
                             Patch(facecolor=BACKGROUND, edgecolor=RULE, label="Quota unfilled")],
                    frameon=False, loc="lower left", bbox_to_anchor=(-.02, 1.01), ncol=3, fontsize=8.8,
                    handlelength=1.3, columnspacing=1.1)
    coverage.grid(axis="x", alpha=.35)
    coverage.set_axisbelow(True)
    coverage.tick_params(axis="y", labelsize=9)

    state_order = sorted(selection["states"], key=lambda s: (REGIMES.index(s["regime"]), s["case"], s["frame_index"]))
    labels, last_regime = [], None
    for index, state in enumerate(state_order):
        if last_regime is not None and state["regime"] != last_regime:
            effects.axhline(index - .5, color=RULE, linewidth=1.2)
        last_regime = state["regime"]
        short = {"uniform": "U", "stationary": "S", "switch": "W"}[state["regime"]]
        labels.append(f"{short} · case {state['case']:02d} · t{state['observation_step']:02d}" + ("  =" if state["stratum"] == "agreement" else ""))
        row = lookup[state["state"], "learned", "frozen_prior"]
        if row["valid_paired_futures"] != 8:
            effects.axhspan(index - .42, index + .42, color=MUTED, zorder=0)
            effects.text(.5, index, "unavailable", transform=effects.get_yaxis_transform(),
                         ha="center", va="center", fontsize=8.5, color=SECONDARY)
            continue
        for left, right, offset, color, marker in (("learned", "frozen_prior", -.13, COBALT, "o"),
                                                   ("known_law", "learned", .13, ORANGE, "^")):
            row = lookup[state["state"], left, right]
            value = row["mean_return_difference_valid_pairs"]
            face = BACKGROUND if state["stratum"] == "agreement" else color
            effects.plot(value, index + offset, marker=marker, markersize=5.7,
                         markeredgecolor=color, markerfacecolor=face, linestyle="none", zorder=3)
    effects.axvline(0, color=RULE, linewidth=1., zorder=0)
    effects.set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels) - .4, -.6),
                xlim=(-.044, .044), xticks=[-.04, -.02, 0., .02, .04], xlabel="Mean paired return difference / state\nEight common future streams · no independent-future CI")
    effects.tick_params(axis="y", labelsize=8.7)
    effects.tick_params(axis="x", labelsize=9)
    effects.set_title("B   Valid state effects; missing values stay missing", loc="left", fontsize=12, fontweight="bold", pad=36)
    effects.legend(handles=[Line2D([], [], color=COBALT, marker="o", ls="none", label="Learned − prior"),
                            Line2D([], [], color=ORANGE, marker="^", ls="none", label="Known − learned")],
                   loc="lower left", bbox_to_anchor=(-.02, 1.01), frameon=False, ncol=2, fontsize=9,
                   handletextpad=.45, columnspacing=1.)

    counts.set_title("C   Components on valid changed-action pairs", loc="left", fontsize=12, fontweight="bold", pad=18)
    counts.axis("off")
    text_rows = []
    for row in components:
        pair = f"{NAMES[row['left']]} − {NAMES[row['right']]}"
        component = lambda name: (f"{row[name + '_left_better']} / {row[name + '_left_worse']}"
                                  if row["valid_changed_action_future_pairs"] else "—")
        text_rows.append([row["regime"].title() + "\n" + pair,
                          str(row["valid_changed_action_future_pairs"]),
                          component("collision"), component("keys"), component("escape")])
    table = counts.table(cellText=text_rows,
                         colLabels=["Paired contrast", "Valid\npairs", "Collision\n+ / −", "Keys\n+ / −", "Escape\n+ / −"],
                         colWidths=[.44, .13, .15, .13, .15], cellLoc="center", bbox=[-.02, -.04, 1.04, 1.04])
    table.auto_set_font_size(False)
    table.set_fontsize(9.)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(.4)
        cell.set_facecolor(BACKGROUND if row else MUTED)
        if row == 0:
            cell.get_text().set_fontweight("bold")
        if col == 0:
            cell.get_text().set_ha("left")
    for index, row in enumerate(components, 1):
        if row["keys_left_better"] or row["keys_left_worse"]:
            table[index, 3].get_text().set_color(MAGENTA)
            table[index, 3].get_text().set_fontweight("bold")

    fig.text(.055, .079, "U = uniform · S = stationary · W = switch.  = / open markers: agreement controls; identical actions share one execution.",
             fontsize=9.2, color=SECONDARY)
    fig.text(.055, .054, "Components: + / − counts futures favoring / disfavoring the left action (fewer collisions, more keys/escapes). Cases and futures are shared.",
             fontsize=9.2, color=SECONDARY)
    fig.text(.055, .029, "Five of 11 disagreement states are unavailable. Validator rejections are not observed collisions; a small 12-step sample cannot establish equivalence.",
             fontsize=9.2, color=MAGENTA)
    save_figure(fig, OUT / "branch-consequences")
    fig.savefig("/tmp/run1-branch-preview.png", dpi=72, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)


def main():
    selection = json.loads((DATA / "selection.json").read_text())
    futures = [json.loads(path.read_text()) for path in sorted((DATA / "futures").glob("*.json"))]
    review = json.loads((DATA / "reviewed-summary.json").read_text())
    if review["input_sha256"]["selection"] != sha(DATA / "selection.json"):
        raise ValueError("Reviewed state selection changed")
    OUT.mkdir(parents=True, exist_ok=True)
    states, cases, components = table_rows(selection, futures)
    write_csv(OUT / "state-paired-effects.csv", states)
    write_csv(OUT / "case-paired-effects.csv", cases)
    write_csv(OUT / "component-pair-counts.csv", components)
    render(selection, states, components)
    manifest = {"renderer_sha256": sha(__file__), "theme_sha256": sha(ROOT / "scripts/chromatic_fields.py"),
                "selection_sha256": sha(DATA / "selection.json"), "reviewed_summary_sha256": sha(DATA / "reviewed-summary.json"),
                "future_files_sha256": hashlib.sha256(json.dumps({p.name: sha(p) for p in sorted((DATA / "futures").glob("*.json"))}, sort_keys=True).encode()).hexdigest(),
                "new_candidate_executions": 0, "new_world_transitions": 0, "new_model_calls": 0,
                "statistical_scope": "Exact descriptive state means and pair counts; no independently resampled future confidence intervals; unavailable states remain unavailable."}
    (OUT / "figure-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved three-panel branch figure and three CSV tables to {OUT}")


if __name__ == "__main__":
    main()
