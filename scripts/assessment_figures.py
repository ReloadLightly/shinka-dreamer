"""Publication figures made only from committed assessment and campaign data."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, TEAL, GRAY, RED = "#0072B2", "#D55E00", "#009E73", "#777777", "#CC6677"
NAMES = {"memory": "Original memory", "original_predictive": "Original predictive seed",
         "predictive": "Selected generation 14", "frozen": "Selected · frozen weights",
         "no_planning": "Selected · fixed-risk", "frozen_no_planning": "Selected · frozen + fixed-risk"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "savefig.facecolor": "white", "axes.labelcolor": "#222222",
                     "text.color": "#222222", "axes.edgecolor": "#666666"})


def save(fig, out, name):
    for extension in ("svg", "pdf", "png"):
        fig.savefig(out / f"{name}.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def evolution(out):
    metrics = json.loads((ROOT / "artifacts/campaign-v2/completed-50/generation-metrics.json").read_text())
    valid = [m for m in metrics if m["correct"]]
    x = [m["generation"] for m in valid]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), layout="constrained")
    for ax, key, label in zip(axes, ("combined_score", "task", "model_score"),
                              ("Selection score", "Mean task score", "Mean model score")):
        values = [m[key] for m in valid]
        ax.scatter(x, values, c=BLUE, s=23, alpha=.7, zorder=3)
        selected = next(m for m in valid if m["generation"] == 14)
        ax.scatter([14], [selected[key]], marker="*", s=155, color=ORANGE, edgecolor="white", linewidth=.6, zorder=5)
        for m in metrics:
            if not m["correct"]:
                ax.axvline(m["generation"], color="#dddddd", lw=1, zorder=0)
        ax.set(xlabel="Candidate slot (seed = 0)", ylabel=label, xlim=(-1, 50))
        ax.grid(axis="y", alpha=.18)
    axes[0].step(x, np.maximum.accumulate([m["combined_score"] for m in valid]), where="post", color=ORANGE, lw=1.6, label="Best so far")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_title("a  Joint selection objective")
    axes[1].set_title("b  Task component (weight 0.6)")
    axes[2].set_title("c  Model component (weight 0.4)")
    fig.suptitle("One 50-slot search · 64 reused development mazes", fontsize=13)
    save(fig, out, "evolution-components")


def outcomes(summary, out):
    order = list(NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), gridspec_kw={"width_ratios": [1.08, 1]}, layout="constrained")
    y = np.arange(len(order))
    ax = axes[0]
    offset = np.zeros(len(order))
    for name, color, label in (("escape", TEAL, "Escape"), ("death", RED, "Death"), ("timeout", GRAY, "Timeout"), ("invalid", "#DDCC77", "Invalid")):
        values = np.array([summary["agents"][v][name]["rate"] if name != "invalid" else summary["agents"][v][name]/summary["cases"] for v in order])*100
        ax.barh(y, values, left=offset, color=color, label=label, height=.66)
        offset += values
    for i, variant in enumerate(order):
        rate = summary["agents"][variant]["escape"]
        lo, hi = np.array(rate["ci95"])*100
        ax.errorbar(rate["rate"]*100, i, xerr=[[rate["rate"]*100-lo], [hi-rate["rate"]*100]], fmt="none", ecolor="#222222", capsize=3, lw=1.2)
        ax.text(1.02, i, f"{rate['count']}/1024", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=[NAMES[v] for v in order], xlabel="Fraction of fresh cases (%)", xlim=(0, 100), title="a  Outcomes and Wilson escape intervals")
    ax.invert_yaxis()
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.30), ncol=4, frameon=False, fontsize=9)
    ax = axes[1]
    keys = ["predictive minus "+v for v in ("frozen", "memory", "original_predictive", "no_planning")]
    labels = ["Selected − frozen (primary)", "Selected − memory", "Selected − original seed", "Selected − fixed-risk"]
    limits = []
    for i, key in enumerate(keys):
        result = summary["contrasts"][key]["binary"]["escape"]
        point = result["difference"]*100
        lo, hi = np.array(result["ci95"])*100
        ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt="o", color=BLUE if i == 0 else GRAY, markersize=7, capsize=4, lw=1.8)
        p = result.get("holm_adjusted_p", result["mcnemar_exact_p"])
        ax.text(1.02, i, f"p{' (Holm)' if i else ''}={p:.3g}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        limits.extend([lo, hi])
    ax.axvline(0, color="#999999", ls="--", lw=1)
    pad = max(1, (max(limits)-min(limits))*.12)
    ax.set(yticks=range(4), yticklabels=labels, xlim=(min(limits)-pad, max(limits)+pad), xlabel="Paired escape difference (percentage points)", title="b  Paired effects · conservative 95% intervals")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.18)
    fig.suptitle("Fresh assessment · 1,024 paired mazes · 6,144 condition-episodes", fontsize=13)
    save(fig, out, "assessment-outcomes-effects")


def learning(summary, out):
    if not summary["matched_prediction"]["verified"]:
        raise ValueError("Cannot plot a matched learning claim: trajectory checks failed")
    curve = summary["learning_curve"]
    x = np.array([p["start_step"]+12.5 for p in curve])
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4), layout="constrained")
    for side, label, color in (("left", "Online updates", BLUE), ("right", "Frozen weights", ORANGE)):
        values = np.array([p[side+"_loss"] for p in curve])
        ci = np.array([p[side+"_interval"]["ci95"] for p in curve])
        axes[0].plot(x, values, "o-", color=color, label=label, markersize=4)
        axes[0].fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=.14)
    axes[0].set(ylabel="Pooled near-cell Brier loss", title="a  Identical fixed-risk experience")
    axes[0].legend(frameon=False, fontsize=9)
    delta = np.array([p["difference"] for p in curve])
    ci = np.array([p["ci95"] for p in curve])
    axes[1].errorbar(x, delta, yerr=np.maximum(0, np.array([delta-ci[:, 0], ci[:, 1]-delta])), fmt="o-", color=BLUE, capsize=3)
    axes[1].axhline(0, color=GRAY, lw=1, ls="--")
    axes[1].set(ylabel="Brier difference (learned − frozen)", title="b  Paired whole-episode bootstrap")
    counts = [p["left_episodes"] for p in curve]
    axes[2].bar(x, counts, width=17, color="#999999")
    for xx, count in zip(x, counts):
        axes[2].text(xx, count+15, str(count), ha="center", fontsize=9)
    axes[2].set(ylabel="Episodes contributing targets", ylim=(0, max(counts)*1.14), title="c  Survivor-conditioned sample size")
    for ax in axes:
        ax.set(xlabel="Episode step (25-step bins)", xticks=[0, 50, 100, 150, 200])
        ax.grid(axis="y", alpha=.16)
    fig.suptitle("Prediction updates on fresh mazes · shaded/error-bar intervals are pointwise 95%", fontsize=13)
    save(fig, out, "matched-learning")


def behavior(record, out):
    examples = record["examples"]
    nrows = (len(examples)+1)//2
    fig, axes = plt.subplots(nrows, 4, figsize=(13.3, nrows*3.7), squeeze=False)
    palette = ListedColormap(["#faf8f0", "#334653", "#E8B730", "#AA88BB", "#55AA88"])
    norm = BoundaryNorm(np.arange(-.5, 5.5), palette.N)
    strata = {"selected_only": "Selected escapes; frozen fails", "frozen_only": "Frozen escapes; selected fails",
              "both_fail": "Both fail", "both_escape": "Both escape"}
    for j, example in enumerate(examples):
        for k, (variant, label, color) in enumerate((("predictive", "Online updates", BLUE), ("frozen", "Frozen weights", ORANGE))):
            ax = axes[j//2, 2*(j % 2)+k]
            agent = example["agents"][variant]
            frame = agent["display_frame"]
            if frame is None:
                ax.text(.5, .5, "No successful action frame", ha="center")
                ax.axis("off")
                continue
            world, model = frame["world"], frame["model"]
            ax.pcolormesh(np.arange(16)-.5, np.arange(16)-.5, np.array(world["grid"]), cmap=palette, norm=norm, shading="flat", rasterized=False)
            path = np.array(agent["path"][:example["display_step"]+1])
            if len(path):
                ax.plot(path[:, 0], path[:, 1], color=color, lw=1.4, alpha=.65)
                ax.scatter(*path[0], marker="s", color=color, s=17, edgecolor="white", linewidth=.3)
            x, y = world["agent"]
            ax.add_patch(Rectangle((x-2.5, y-2.5), 5, 5, fill=False, edgecolor=color, lw=1.2))
            ax.scatter(x, y, s=45, color=color, edgecolor="white", linewidth=.6, zorder=5)
            dx, dy = frame["action"]["move"]
            if dx or dy:
                ax.arrow(x, y, dx*.85, dy*.85, width=.07, head_width=.35, color=color, zorder=6)
            else:
                ax.scatter(x, y, s=125, facecolor="none", edgecolor=color, lw=1.6, zorder=6)
            for ex, ey in world["enemies"]:
                ax.scatter(ex, ey, s=24, color=RED, edgecolor="white", linewidth=.5)
            for ex, ey in frame["next_enemies"]:
                ax.scatter(ex, ey, s=45, facecolor="none", edgecolor="#8f3344", linewidth=.9)
            ox, oy = world["origin"]
            forecast = {(xx+ox, yy+oy): p for xx, yy, p in model.get("enemy", [])}
            p = forecast.get((x+dx, y+dy), model.get("default_enemy", .5))
            result = agent["original"]
            status = "" if agent["reproduces_original"] else " · rerun differs"
            ax.set_title(f"Case {example['case']:04d} · {label}\n{result['reason']}, {result['steps']} steps{status}", fontsize=9.5, color=color)
            ax.set(xlim=(-.5, 14.5), ylim=(14.5, -.5), aspect="equal", xticks=[], yticks=[])
            ax.set_xlabel(f"Step {world['step']} · next-cell forecast {p:.3f}", fontsize=9)
        left_ax = axes[j//2, 2*(j % 2)]
        left_ax.text(1.08, 1.22, strata[example["stratum"]], transform=left_ax.transAxes, ha="center", fontsize=10, weight="bold")
    for empty in range(len(examples)*2, nrows*4):
        axes.flat[empty].axis("off")
    fig.suptitle("Rule-selected paired behavior · exposed assessment cases", fontsize=14, y=.99)
    fig.text(.5, .01, "Grid at the first differing action (last shared frame if identical). Lines show preceding paths; arrows show chosen actions.\nSquares: starts · outlined view: observed 5×5 · red dots: current enemies · hollow red rings: next enemies, shown retrospectively.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.09, hspace=.48, wspace=.20)
    save(fig, out, "paired-behavior")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/campaign-v2/assessment-1024")
    args = parser.parse_args()
    data = Path(args.data)
    out = data / "figures"
    out.mkdir(exist_ok=True)
    summary = json.loads((data / "analysis.json").read_text())
    evolution(out)
    outcomes(summary, out)
    learning(summary, out)
    if (data / "behavior-examples.json").exists():
        behavior(json.loads((data / "behavior-examples.json").read_text()), out)
    print("Wrote SVG, PDF and PNG figures from committed data.")


if __name__ == "__main__":
    main()
