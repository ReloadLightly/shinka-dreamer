"""Render evaluator-owned true world, exported map and BEFORE-outcome forecast."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
from dreamer.evaluation import run_episode
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--out", default="artifacts/replay")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = run_episode(ROOT / "initial.py", args.seed, replay=True)
    trace = result.pop("trace")
    (out / "replay.json").write_text(json.dumps({"episode": result, "frames": trace}, separators=(",", ":")))
    cmap = ListedColormap(["#c4cdd5", "#1b2934", "#f3efe5", "#243846", "#f2bc39", "#8862a8", "#53a477"])
    norm = BoundaryNorm(np.arange(-2.5, 5.5), cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.4))
    frames = []
    selected = next((i for i, f in enumerate(trace) if f["world"]["step"] >= 25 and any(
        max(abs(x-f["world"]["agent"][0]), abs(y-f["world"]["agent"][1])) <= 2
        for x, y in f["world"]["enemies"])), min(25, len(trace)-1))
    for i, frame in enumerate(trace):
        world, model = frame["world"], frame["model"]
        ox, oy = world["origin"]
        ax, ay = world["agent"]
        belief, risk = np.full((15, 15), -2), np.full((15, 15), model.get("default_enemy", .5))
        for x, y, cell, seen in model.get("terrain", []):
            if 0 <= x+ox < 15 and 0 <= y+oy < 15:
                belief[y+oy, x+ox] = cell
        for x, y, probability in model.get("enemy", []):
            if 0 <= x+ox < 15 and 0 <= y+oy < 15:
                risk[y+oy, x+ox] = probability
        titles = ["Hidden world at t", "Agent's remembered terrain", "Forecast: enemy occupancy at t+1"]
        for j, axis in enumerate(axes):
            axis.clear()
            if j == 2:
                axis.imshow(risk, cmap="YlOrRd", vmin=0, vmax=.4)
                for x, y in frame["next_enemies"]:
                    axis.scatter(x, y, s=65, facecolors="none", edgecolors="#70201d", linewidths=1.5)
            else:
                axis.imshow(world["grid"] if j == 0 else belief, cmap=cmap, norm=norm)
                for x, y in world["enemies"]:
                    if j == 0 or max(abs(x-ax), abs(y-ay)) <= 2:
                        axis.scatter(x, y, s=45, c="#c52b36", edgecolors="white", linewidths=.5)
            axis.scatter(ax, ay, s=55, c="#248bb8", edgecolors="white", zorder=5)
            axis.add_patch(Rectangle((ax-2.5, ay-2.5), 5, 5, fill=False, edgecolor="#248bb8", linewidth=1.4))
            dx, dy = frame["action"]["move"]
            if dx or dy:
                axis.arrow(ax, ay, dx*.85, dy*.85, head_width=.3, width=.08, color="#124e6d", zorder=6)
            axis.set(title=titles[j], xticks=[], yticks=[], xlim=(-.5, 14.5), ylim=(14.5, -.5))
        learning = model.get("learning", {})
        rates = learning.get("rates", {})
        fig.suptitle(f"Predictive seed · maze {args.seed} · step {world['step']} → {world['step']+1} · keys {world['keys']}/2", fontsize=13)
        caption = (f"Action {frame['action']['move']}  |  predictive updates {learning.get('updates',0)}  |  "
                   f"P(enemy next | nearby) {rates.get('near',0):.3f}\n"
                   "Blue: agent/view/action · red: enemy · yellow: key · purple: door · green: exit\n"
                   "Forecast scale: pale=0, red≥0.4 · hollow rings: next enemies (shown for retrospective audit)")
        if len(fig.texts) > 1:
            fig.texts[1].set_text(caption)
        else:
            fig.text(.5, .025, caption, ha="center", fontsize=9, linespacing=1.5)
        fig.subplots_adjust(left=.02, right=.98, top=.84, bottom=.23, wspace=.08)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        if i == selected:
            fig.savefig(out / "frame.png", dpi=150)
    frames[0].save(out / "replay.gif", save_all=True, append_images=frames[1:], duration=180, loop=0)
    plt.close(fig)
    print(args.seed, result["reason"], result["steps"], "frames", len(frames))


if __name__ == "__main__":
    main()
