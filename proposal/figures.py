"""Render saved original-proposal evidence; never execute agents or worlds."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import chromatic_fields as style

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

OUTCOMES = (
    ('escaped', 'Escape', 'escape', 'o'),
    ('caught', 'Death', 'death', 's'),
    ('timeout', 'Timeout', 'timeout', '^'),
    ('invalid', 'Invalid', 'invalid', 'X'),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(count, total):
    z = 1.959963984540054
    p = count / total
    denominator = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return max(0., middle - half), min(1., middle + half)


def outcomes_figure(rows, out):
    n = len(rows)
    counts = Counter(row['reason'] for row in rows)
    if set(counts) - {item[0] for item in OUTCOMES}:
        raise ValueError('Unrecognized episode outcome')
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5),
                             gridspec_kw={'width_ratios': [1, 1.35, 1.4]})
    fig.subplots_adjust(left=.085, right=.985, top=.72, bottom=.24, wspace=.64)
    fig.suptitle('Original-proposal seed: initial evaluation', x=.04, ha='left', y=.98)
    fig.text(.04, .89, f'{n} development episodes · fixed cases {rows[0]["case"]}–{rows[-1]["case"]}'
             ' · no evolved descendants', color=style.SECONDARY)

    ax = axes[0]
    ax.set_title('A  Outcomes', loc='left')
    for index, (reason, label, role, marker) in enumerate(OUTCOMES):
        count = counts[reason]
        low, high = wilson(count, n)
        ax.plot([100 * low, 100 * high], [index, index], color=style.OUTCOMES[role], lw=2)
        ax.scatter(100 * count / n, index, color=style.OUTCOMES[role], marker=marker,
                   s=42, zorder=3)
        ax.text(103, index, f'{count}/{n}', va='center', fontsize=10)
    ax.set(yticks=range(4), yticklabels=[item[1] for item in OUTCOMES],
           ylim=(3.5, -.5), xlim=(-4, 128), xticks=[0, 50, 100], xlabel='Episodes (%)')
    ax.grid(axis='x')
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.set_title('B  Original objective', loc='left')
    fields = [('task', 'Task'), ('model_accuracy', 'Map\naccuracy'),
              ('combined_score', 'Combined')]
    for index, (field, label) in enumerate(fields):
        values = [row[field] for row in rows]
        # Deterministic display offsets preserve each recorded observation.
        offsets = np.linspace(-.13, .13, n)
        ax.scatter(values, index + offsets, s=14, color=style.SECONDARY,
                   alpha=.55, linewidth=0, zorder=2)
        mean = sum(values) / n
        ax.scatter([mean], [index], s=50, marker='D', color=style.COBALT,
                   edgecolor=style.BACKGROUND, linewidth=.7, zorder=3)
        ax.text(1.035, index, f'{mean:.3f}', va='center', fontsize=10)
    ax.set(yticks=range(3), yticklabels=[item[1] for item in fields],
           ylim=(2.5, -.5), xlim=(-.03, 1.25), xticks=[0, .5, 1], xlabel='Score (0–1)')
    ax.grid(axis='x')
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.set_title('C  Map reconstruction', loc='left')
    for reason, label, role, marker in OUTCOMES:
        subset = [row for row in rows if row['reason'] == reason]
        if subset:
            ax.scatter([100 * row['final_coverage'] for row in subset],
                       [100 * row['model_accuracy'] for row in subset],
                       color=style.OUTCOMES[role], marker=marker, s=38,
                       edgecolor=style.BACKGROUND, linewidth=.6, alpha=.8)
    ax.set(xlim=(-3, 103), ylim=(-3, 103), xticks=[0, 50, 100], yticks=[0, 50, 100],
           xlabel='Last-record coverage (%)', ylabel='Current-map accuracy (%)')
    ax.grid()
    ax.set_axisbelow(True)
    fig.text(.04, .105, 'A: descriptive 95% Wilson intervals.  B: dots = episodes; diamonds = means.',
             color=style.SECONDARY, fontsize=10)
    fig.text(.04, .045, 'Combined = 0.6 × task + 0.4 × current-map accuracy; invalid episodes score 0.',
             color=style.SECONDARY, fontsize=10)
    style.save_figure(fig, out/'initial-outcomes')


def translated_belief(frame):
    world = frame['world']
    terrain = np.asarray(world['grid'])
    belief = np.full(terrain.shape, -2, dtype=int)
    ox, oy = world['origin']
    for rx, ry, cell in frame['belief']:
        x, y = rx + ox, ry + oy
        if 0 <= y < terrain.shape[0] and 0 <= x < terrain.shape[1]:
            belief[y, x] = cell
    actual = terrain.copy()
    for x, y in world['enemies']:
        actual[y, x] = 5
    known = belief != -2
    return belief, actual, known


def draw_map(ax, terrain, enemies, agent, *, mismatches=None):
    display = np.asarray(terrain).copy()
    # An occupied belief cell says nothing about its underlying terrain.
    display[display == 5] = -2
    cmap, norm = style.terrain_cmap()
    ax.imshow(display, cmap=cmap, norm=norm, interpolation='nearest')
    style.terrain_symbols(ax, display, size=34)
    if enemies:
        ax.scatter(*zip(*enemies), marker='x', color=style.MAGENTA, s=43,
                   linewidth=1.5, zorder=4)
    ax.scatter(*agent, marker='o', facecolor=style.BACKGROUND, edgecolor=style.TEXT,
               s=60, linewidth=1.4, zorder=5)
    if mismatches is not None:
        for y, x in zip(*np.where(mismatches)):
            ax.add_patch(Rectangle((x - .47, y - .47), .94, .94,
                                   fill=False, edgecolor=style.ORANGE, lw=1.1, zorder=3))
    height, width = display.shape
    ax.set(xticks=[0, width // 2, width - 1], yticks=[0, height // 2, height - 1],
           xlabel='x (world coordinate)', ylabel='y')
    ax.set_xticks(np.arange(-.5, width), minor=True)
    ax.set_yticks(np.arange(-.5, height), minor=True)
    ax.grid(which='minor', color=style.RULE, linewidth=.3)
    ax.tick_params(which='minor', length=0)


def replay_figure(row, out):
    trace = row['trace']
    if not trace:
        raise ValueError('The first episode has no recorded frames')
    indices = [0, len(trace) - 1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.7))
    fig.subplots_adjust(left=.10, right=.90, top=.85, bottom=.23, hspace=.49, wspace=.30)
    fig.suptitle(f'Original-proposal seed: recorded episode {row["case"]}',
                 x=.04, ha='left', y=.985)
    fig.text(.04, .938, f'Fixed first / last recorded decisions · outcome: {row["reason"]}'
             f' · {row["steps"]} transitions · {row["keys"]}/2 keys', color=style.SECONDARY)
    facts = []
    for pair, frame_index in enumerate(indices):
        frame = trace[frame_index]
        world = frame['world']
        belief, actual, known = translated_belief(frame)
        correct = int(np.sum(known & (belief == actual)))
        total = int(known.sum())
        expected = row['timeline'][frame_index]
        if (correct, total) != (expected['correct'], expected['audited']):
            raise ValueError('Replay coordinates disagree with recorded audit')
        position = 'First' if pair == 0 else 'Last'
        title = f'{position} record · step {world["step"]}'
        draw_map(axes[pair, 0], world['grid'], world['enemies'], world['agent'])
        axes[pair, 0].set_title(f'{title}\nHidden world (retrospective)', loc='left')
        yy, xx = np.where(belief == 5)
        enemies = list(zip(xx.tolist(), yy.tolist()))
        draw_map(axes[pair, 1], belief, enemies, world['agent'],
                 mismatches=known & (belief != actual))
        axes[pair, 1].set_title(f'{title}\nBelief: {correct}/{total} reported cells correct', loc='left')
        facts.append({'trace_index': frame_index, 'step': world['step'],
                      'correct': correct, 'audited': total, 'world_cells': int(belief.size),
                      'origin': world['origin'], 'agent': world['agent']})
    handles = [
        Patch(facecolor=style.MUTED, edgecolor=style.RULE, label='Unknown'),
        Patch(facecolor=style.BACKGROUND, edgecolor=style.RULE, label='Empty'),
        Patch(facecolor=style.WALL, label='Wall'),
        *[Line2D([], [], marker=marker, color=color, linestyle='', label=label,
                  markersize=8) for marker, color, label in style.TERRAIN_GLYPHS.values()],
        Line2D([], [], marker='o', markerfacecolor=style.BACKGROUND,
               color=style.TEXT, linestyle='', label='Agent', markersize=7),
        Line2D([], [], marker='x', color=style.MAGENTA, linestyle='',
               label='Enemy / remembered enemy', markersize=7),
        Patch(facecolor=style.BACKGROUND, edgecolor=style.ORANGE, label='Incorrect belief'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .075),
               ncol=3, columnspacing=2, handlelength=1.4)
    fig.text(.04, .032, 'Both records precede the chosen action. Belief coordinates are translated by the recorded origin.',
             color=style.SECONDARY, fontsize=10)
    style.save_figure(fig, out/'initial-replay')
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'artifacts/proposal/initial-evaluation')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/proposal/figures')
    args = parser.parse_args()
    paths = {name: args.input/f'{name}.json' for name in ('episodes', 'metrics', 'manifest')}
    inputs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    rows = inputs['episodes']
    if not rows or inputs['metrics']['public']['episodes'] != len(rows):
        raise ValueError('Missing episodes or inconsistent episode count')
    if inputs['manifest']['episodes'] != len(rows):
        raise ValueError('Manifest episode count differs from evidence')
    style.apply_theme()
    args.out.mkdir(parents=True, exist_ok=True)
    outcomes_figure(rows, args.out)
    frames = replay_figure(rows[0], args.out)
    n = len(rows)
    counts = {reason: sum(row['reason'] == reason for row in rows) for reason, *_ in OUTCOMES}
    captions = {
        'initial-outcomes': (
            f'Figure 1. Initial original-proposal seed on {n} fixed development cases '
            f'({rows[0]["case"]}–{rows[-1]["case"]}); no evolutionary search. '
            'A: episode outcomes with descriptive nominal 95% Wilson intervals on these development cases; '
            'the intervals do not represent a held-out assessment. '
            'B: individual episode scores and their arithmetic means; the original objective is '
            '0.6 × normalized task reward + 0.4 × current-map accuracy, with invalid episodes scoring zero. '
            'C: episode accuracy (sum of correct reported cells / sum of audited reported cells over '
            'all pre-action records) versus coverage at the last record (reported in-bounds cells / '
            'all world cells). Unknown cells are excluded from accuracy. Repeated audits of a cell '
            'are included. These are observation-updated reconstruction measurements; they do not '
            'measure predictive learning, generalization, or evolution.'),
        'initial-replay': (
            f'Figure 2. Episode {rows[0]["case"]}, chosen prospectively as the first case. '
            f'The first and last of its {len(rows[0]["trace"])} saved pre-action records are shown '
            f'(steps {frames[0]["step"]} and {frames[-1]["step"]}). '
            'Left: retrospective hidden world, unavailable to the candidate. Right: the candidate’s '
            'reported map after the local observation, translated into world coordinates using the '
            'recorded origin. Enemy crosses are anonymous; on the belief map they can be stale sightings '
            'and specify occupancy without underlying terrain. Pale cells are unknown. Orange outlines '
            'mark reported cells that disagree with the current hidden world. '
            'Counts in panel titles use reported in-bounds cells at that record. '
            'The last record precedes the terminal action and is not a terminal-state snapshot.'),
    }
    (args.out/'captions.json').write_text(json.dumps(captions, indent=2, ensure_ascii=False) + '\n')
    outputs = [args.out/f'{stem}.{ext}' for stem in captions for ext in ('svg', 'pdf', 'png')]
    outputs.append(args.out/'captions.json')
    sources = [ROOT/'proposal/figures.py', ROOT/'scripts/chromatic_fields.py', ROOT/'scripts/visual_theme.py']
    manifest = {
        'purpose': 'Saved-data rendering of the initial original-proposal seed only',
        'style': style.PRESENTATION_VERSION,
        'command': '.venv/bin/python -m proposal.figures',
        'inputs_sha256': {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path)
                          for path in paths.values()},
        'sources_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        'outputs_sha256': {path.name: sha256(path) for path in outputs},
        'episodes': n, 'outcomes': counts, 'replay_case': rows[0]['case'], 'replay_records': frames,
        'execution': 'No agents, environments, model calls, resampling, or evolutionary search executed',
    }
    (args.out/'rendering-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'episodes': n, 'outcomes': counts, 'replay_records': frames}, indent=2))


if __name__ == '__main__':
    main()
