"""Regenerate the research overview from committed evidence; runs no episodes.

Usage: python scripts/research_figure.py
Requires numpy and matplotlib. Paired 95% percentile intervals use the same
5,000 episode resamples and RNG seed as scripts/report.py. Figures describe
one selected campaign on its reused development cases, not generalization.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/campaign-v2"
OUT = BASE / "research-review"
SOURCE_PATHS = [
    "completed-50/generation-metrics.json",
    "completed-50/summary.json",
    "completed-50/episodes.csv",
    "controls/episodes.csv",
    "mechanism-gen14/episodes.csv",
    "mechanism-gen14/summary.json",
    "mechanism-gen14/audit-summary.json",
]


def load_json(path):
    return json.loads((BASE / path).read_text())


def groups(path, key):
    out = {}
    with (BASE / path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row[key], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda row: int(row["episode"]))
        assert [int(row["episode"]) for row in rows] == list(range(64))
    return out


def paired(left, right, metric):
    """Paired episode bootstrap, pooling target losses within each resample."""
    assert [r["episode"] for r in left] == [r["episode"] for r in right]
    indices = np.random.default_rng(382940).integers(0, len(left), (5000, len(left)))

    def estimate(rows):
        if metric.startswith("brier_"):
            values = np.array([[float(r[metric + "_sum"]), int(r[metric + "_n"])] for r in rows])
            samples = values[indices].sum(axis=1)
            assert (samples[:, 1] > 0).all()
            totals = values.sum(axis=0)
            return totals[0] / totals[1], samples[:, 0] / samples[:, 1]
        values = np.array([r["reason"] == "escaped" if metric == "escape" else float(r[metric]) for r in rows])
        return values.mean(), values[indices].mean(axis=1)

    left_mean, left_samples = estimate(left)
    right_mean, right_samples = estimate(right)
    return {
        "left": float(left_mean),
        "right": float(right_mean),
        "difference": float(left_mean - right_mean),
        "ci95": np.quantile(left_samples - right_samples, [.025, .975]).tolist(),
    }


def discordance(left, right):
    cells = {name: [] for name in ("both_escape", "left_only_escape", "right_only_escape", "neither_escape")}
    for a, b in zip(left, right):
        assert a["episode"] == b["episode"]
        names = {(True, True): "both_escape", (True, False): "left_only_escape",
                 (False, True): "right_only_escape", (False, False): "neither_escape"}
        cells[names[(a["reason"] == "escaped", b["reason"] == "escaped")]].append(int(a["episode"]))
    return {
        "counts": {key: len(value) for key, value in cells.items()},
        "episode_ids": cells,
        "total_steps_left_minus_right": sum(int(a["steps"]) - int(b["steps"]) for a, b in zip(left, right)),
        "escape_effect": paired(left, right, "escape"),
    }


def analysis():
    metrics = load_json("completed-50/generation-metrics.json")
    by_gen = {row["generation"]: row for row in metrics}
    campaign = groups("completed-50/episodes.csv", "generation")
    controls = groups("controls/episodes.csv", "agent")
    mechanisms = groups("mechanism-gen14/episodes.csv", "agent")
    seed, early, selected = (by_gen[g] for g in (0, 5, 14))
    valid = [row for row in metrics if row["correct"]]
    best_task = max(valid, key=lambda row: row["task"])
    best_model = max(valid, key=lambda row: row["model_score"])
    assert selected == max(valid, key=lambda row: row["combined_score"])
    for row in valid:
        saved = campaign[str(row["generation"])]
        for field in ("task", "model_score", "combined_score"):
            assert np.isclose(np.mean([float(r[field]) for r in saved]), row[field], rtol=0, atol=1e-12)

    effects = {
        "selected_minus_memory": paired(campaign["14"], controls["memory"], "escape"),
        "predictive_minus_frozen": paired(mechanisms["predictive"], mechanisms["frozen"], "escape"),
        "predictive_minus_fixed_risk": paired(mechanisms["predictive"], mechanisms["no_planning"], "escape"),
    }
    matched = {metric: paired(mechanisms["no_planning"], mechanisms["frozen_no_planning"], metric)
               for metric in ("brier_near", "brier_threat")}
    for effect in matched.values():
        effect["relative_loss_reduction"] = -effect["difference"] / effect["right"]

    # Confirm regenerated intervals agree with the already published summaries.
    summaries = [load_json("completed-50/summary.json"), load_json("mechanism-gen14/summary.json")]
    references = [summaries[0]["paired"]["selected minus memory"]["escape"],
                  summaries[1]["paired"]["predictive minus frozen"]["escape"],
                  summaries[1]["paired"]["predictive minus no_planning"]["escape"]]
    for effect, reference in zip(effects.values(), references):
        assert np.allclose([effect["difference"], *effect["ci95"]], [reference["difference"], *reference["ci95"]], rtol=0, atol=1e-12)
    for metric, effect in matched.items():
        reference = summaries[1]["paired"]["no_planning minus frozen_no_planning"][metric]
        assert np.allclose([effect["difference"], *effect["ci95"]], [reference["difference"], *reference["ci95"]], rtol=0, atol=1e-12)
    audit = load_json("mechanism-gen14/audit-summary.json")
    assert audit["matched_fixed_risk_trajectories"] and audit["matched_fixed_risk_maps_and_localization"]

    fields = ("generation", "combined_score", "task", "model_score", "escape", "brier_near", "brier_threat")
    total_gain = selected["combined_score"] - seed["combined_score"]
    result = {
        "sources": {str((BASE / p).relative_to(ROOT)): hashlib.sha256((BASE / p).read_bytes()).hexdigest() for p in SOURCE_PATHS},
        "scope": {
            "split": "development", "paired_cases": 64, "campaigns": 1,
            "interval": "95% paired percentile bootstrap; 5000 episode resamples; NumPy default_rng seed 382940",
            "brier_aggregation": "sum of squared errors / target count; resampling paired episodes, not individual cells",
            "boundaries": [
                "Development cases were reused in selection; intervals do not account for selection bias.",
                "No new evaluation or model call is performed by this analysis.",
                "Between-candidate prediction scores are measured on different policy-induced trajectories.",
                "Matched learning contrasts use identical fixed-risk trajectories with mapping/localization retained.",
                "Forecast improvement does not establish an escape benefit, held-out generalization or search repeatability.",
            ],
        },
        "milestones": [{field: by_gen[g][field] for field in fields} for g in (0, 2, 5, 14)],
        "seed_to_selected_score_decomposition": {
            "combined_gain": total_gain,
            "task_gain": selected["task"] - seed["task"],
            "model_gain": selected["model_score"] - seed["model_score"],
            "weighted_task_contribution": .6 * (selected["task"] - seed["task"]),
            "weighted_model_contribution": .4 * (selected["model_score"] - seed["model_score"]),
        },
        "generation_5_to_14": {
            "combined_gain": selected["combined_score"] - early["combined_score"],
            "fraction_of_final_score_gain_already_at_generation_5": (early["combined_score"] - seed["combined_score"]) / total_gain,
            "weighted_task_contribution": .6 * (selected["task"] - early["task"]),
            "weighted_model_contribution": .4 * (selected["model_score"] - early["model_score"]),
        },
        "best_model_score_candidate": {field: best_model[field] for field in fields},
        "best_task_score_candidate": {field: best_task[field] for field in fields},
        "paired_outcomes": {
            "generation_14_minus_seed": discordance(campaign["14"], campaign["0"]),
            "generation_14_minus_generation_5": discordance(campaign["14"], campaign["5"]),
        },
        "escape_effects": effects,
        "matched_learning_effects": matched,
    }
    d = result["seed_to_selected_score_decomposition"]
    assert np.isclose(d["combined_gain"], d["weighted_task_contribution"] + d["weighted_model_contribution"], rtol=0, atol=1e-12)
    return metrics, result


def draw(metrics, result):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/shinkadreamer-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    ink, muted, teal, violet, gray = "#172C3B", "#52616E", "#007F80", "#7253A8", "#B2BDC5"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": ink,
        "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted,
        "axes.edgecolor": "#C5CED4", "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "shinkadreamer-research-50",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(12, 9.5))
    fig.text(.07, .965, "ShinkaDreamer  /  What evolution actually changed", fontsize=20, weight="bold", va="top")
    fig.text(.07, .925, "One campaign · 50 slots · 64 reused development mazes · selected agent: generation 14", fontsize=11.5, color=muted)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.02, 1], left=.075, right=.965,
                           bottom=.17, top=.84, wspace=.29, hspace=.70)
    ax = fig.add_subplot(grid[0, :])
    ax.set_title("A   Almost all score improvement arrived by generation 5", loc="left", pad=19, fontsize=14, weight="bold")
    valid = [r for r in metrics if r["correct"]]
    ax.scatter([r["generation"] for r in valid], [r["combined_score"] for r in valid],
               c=gray, s=29, edgecolors="white", linewidths=.5, zorder=3, label="Valid candidate")
    values = np.array([r["combined_score"] if r["correct"] else -np.inf for r in metrics])
    ax.step(range(50), np.maximum.accumulate(values), where="post", color=teal, lw=2.3, label="Best so far")
    milestones = result["milestones"]
    ax.plot([r["generation"] for r in milestones], [r["combined_score"] for r in milestones],
            color=teal, ls="none", marker="o", ms=6, zorder=4)
    offsets = {0: (6, -19), 2: (-10, 10), 5: (3, 10), 14: (-13, 10)}
    for row in milestones:
        ax.annotate(str(row["generation"]), (row["generation"], row["combined_score"]),
                    xytext=offsets[row["generation"]], textcoords="offset points", color=teal, weight="bold", fontsize=10)
    ax.text(27, .983, "No later candidate exceeded generation 14", fontsize=10.5, color=teal)
    margin = result["generation_5_to_14"]["combined_gain"]
    ax.text(26, .847, f"Generation 14 − generation 5: {margin:+.6f}\nSame 62 successes; 17 fewer total steps", fontsize=10.5,
            color=ink, bbox={"facecolor": "#F2F6F7", "edgecolor": "none", "pad": 8})
    ax.set(xlim=(-.8, 49.8), ylim=(.827, 1.001), ylabel="Selection score", xlabel="Generation / candidate slot")
    ax.set_xticks([0, 5, 14, 25, 35, 49])
    ax.set_yticks([.85, .90, .95, 1.00])
    ax.grid(axis="y", color="#E8EDF0", linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=10, ncol=2)

    ax = fig.add_subplot(grid[1, 0])
    ax.set_title("B   Escape gains remain uncertain", loc="left", pad=31, fontsize=14, weight="bold")
    ax.text(0, 1.045, "Paired differences in escape rate", transform=ax.transAxes, fontsize=10.5, color=muted)
    ax.axvline(0, color="#A8B5BD", ls=(0, (3, 3)), lw=1)
    labels = ["Selected − memory baseline", "Learned − frozen parameters", "Predictive − fixed-risk planning"]
    for y, label, effect in zip([2, 1, 0], labels, result["escape_effects"].values()):
        x = 100 * effect["difference"]
        lo, hi = np.array(effect["ci95"]) * 100
        ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt="o", color=violet, lw=2, ms=6, capsize=4)
        ax.text(-6.5, y+.31, label, fontsize=10.5, color=ink)
        ax.text(20.5, y-.04, f"{x:+.1f}", ha="right", fontsize=11, color=violet, weight="bold")
    ax.set(xlim=(-6.5, 21), ylim=(-.47, 2.62), yticks=[], xlabel="Escape difference (percentage points)")
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.spines["left"].set_visible(False)
    ax.text(0, -.26, "Frozen: mapping retained, weights fixed.\nFixed-risk: learned forecast use removed from planning.",
            transform=ax.transAxes, fontsize=9.4, color=muted, va="top", linespacing=1.5)

    ax = fig.add_subplot(grid[1, 1])
    ax.set_title("C   Learning improves forecasts", loc="left", pad=31, fontsize=14, weight="bold")
    ax.text(0, 1.045, "Same actions, worlds, maps and localization", transform=ax.transAxes, fontsize=10.5, color=muted)
    ax.axvline(0, color="#A8B5BD", ls=(0, (3, 3)), lw=1)
    for y, label, effect in zip([1.6, .25], ["Near-cell Brier loss", "Threat-cell Brier loss"], result["matched_learning_effects"].values()):
        x = effect["difference"] * 1000
        lo, hi = np.array(effect["ci95"]) * 1000
        ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt="o", color=teal, lw=2, ms=6, capsize=4)
        ax.text(-1.68, y+.4, label, fontsize=11, color=ink)
        ax.text(-1.68, y+.17, f"{effect['relative_loss_reduction']*100:.1f}% lower than frozen", fontsize=10, color=teal)
    ax.set(xlim=(-1.68, .16), ylim=(-.47, 2.62), yticks=[], xlabel="Learned − frozen Brier loss (× 10⁻³)")
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.spines["left"].set_visible(False)
    ax.text(0, -.26, "Both variants use the same fixed-risk planner.\nNegative values indicate better learned forecasts.",
            transform=ax.transAxes, fontsize=9.4, color=muted, va="top", linespacing=1.5)

    fig.text(.075, .06, "Intervals: 95% paired episode bootstrap. Development evidence only; selection bias is not corrected.", fontsize=10, color=muted)
    fig.text(.075, .036, "Better prediction on matched experience is demonstrated here; a general escape benefit is still unestablished.", fontsize=10, color=muted)
    fig.savefig(OUT / "research-overview.svg", metadata={"Date": None, "Creator": "ShinkaDreamer scripts/research_figure.py"})
    svg = OUT / "research-overview.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUT / "research-overview.png", dpi=180, metadata={"Software": "ShinkaDreamer scripts/research_figure.py"})
    plt.close(fig)


def main():
    metrics, result = analysis()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    draw(metrics, result)
    print(f"Regenerated committed-data analysis and figures in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
