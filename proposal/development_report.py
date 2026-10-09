"""Analyse saved Stage 3 development episodes; never execute candidates or models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts import chromatic_fields as style

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/proposal/stage3-development'
BOOTSTRAP_SEED = 310932
BOOTSTRAP_DRAWS = 10000
METRICS = ('combined_score', 'task', 'model_accuracy', 'final_coverage',
           'escaped', 'caught', 'timeout', 'invalid', 'keys', 'door_open', 'steps')
LABELS = {'combined_score': 'Combined F', 'task': 'Task S',
          'model_accuracy': 'Map accuracy A', 'final_coverage': 'Map coverage',
          'escaped': 'Escape fraction'}
OUTCOMES = {'escaped': ('Escape', style.COBALT, 'o'),
            'caught': ('Death', style.MAGENTA, 's'),
            'timeout': ('Timeout', style.ORANGE, '^'),
            'invalid': ('Invalid', style.SECONDARY, 'X')}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., center - half), min(1., center + half)]


def validate(rows):
    """Check saved arithmetic without importing the scientific execution path."""
    if len(rows) != 32 or sorted(r['case'] for r in rows) != list(range(32)):
        raise ValueError('Expected exactly cases 0–31, each appearing once')
    for r in rows:
        if r['reason'] not in OUTCOMES:
            raise ValueError(f'Unknown outcome in case {r["case"]}')
        a = r['map_correct'] / r['map_audited'] if r['map_audited'] else 0.
        raw = (.1 * r['steps'] + 20 * r['keys'] + 30 * r['door_open']
               + 100 * (r['reason'] == 'escaped') - 50 * (r['reason'] == 'caught'))
        s = min(1., max(0., (raw + 50) / 250))
        f = 0. if r['error'] is not None else .6 * s + .4 * a
        for key, actual in [('model_accuracy', a), ('task', s), ('combined_score', f)]:
            if not math.isclose(r[key], actual, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Incorrect saved {key}: case {r["case"]}')


def matrix(rows):
    values = []
    for row in rows:
        r = dict(row)
        # Original A is one pooled-cell ratio within each episode, then an
        # equal-weight mean of episodes. Do not pool cells across episodes.
        r['model_accuracy'] = r['map_correct'] / r['map_audited'] if r['map_audited'] else 0.
        r.update({outcome: float(r['reason'] == outcome) for outcome in OUTCOMES})
        values.append([float(r[key]) for key in METRICS])
    return np.asarray(values)


def describe(seed, evolved, rng):
    a, b = matrix(seed), matrix(evolved)
    n = len(seed)
    indices = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    # Whole paired episodes, with the same draws for every metric and arm.
    sampled_a, sampled_b = a[indices].mean(axis=1), b[indices].mean(axis=1)
    result = {'episodes': n, 'cases': [r['case'] for r in seed], 'metrics': {}}
    for i, key in enumerate(METRICS):
        result['metrics'][key] = {
            'seed_mean': float(a[:, i].mean()),
            'generation_2_mean': float(b[:, i].mean()),
            'difference': float((b[:, i] - a[:, i]).mean()),
            'difference_ci95': np.quantile(sampled_b[:, i] - sampled_a[:, i], [.025, .975]).tolist(),
            'seed_ci95': np.quantile(sampled_a[:, i], [.025, .975]).tolist(),
            'generation_2_ci95': np.quantile(sampled_b[:, i], [.025, .975]).tolist(),
        }
    result['outcomes'] = {
        name: {reason: sum(row['reason'] == reason for row in rows) for reason in OUTCOMES}
        for name, rows in [('seed', seed), ('generation_2', evolved)]
    }
    result['transitions'] = {'seed': sum(r['steps'] for r in seed),
                             'generation_2': sum(r['steps'] for r in evolved)}
    result['post_hoc_escape_boundary_diagnostic'] = {
        'method': '95% Wilson interval for each arm separately, not a paired effect interval',
        'seed': wilson(result['outcomes']['seed']['escaped'], n),
        'generation_2': wilson(result['outcomes']['generation_2']['escaped'], n),
        'reason': 'Added after observing constant escape outcomes; illustrates uncertainty hidden by a degenerate empirical bootstrap.'}
    result['objective_contributions'] = {
        'task_difference_times_0_6': .6 * result['metrics']['task']['difference'],
        'accuracy_difference_times_0_4': .4 * result['metrics']['model_accuracy']['difference'],
        'note': 'Contributions sum to F difference only when neither arm has invalid executions.'}
    result['outcome_pairs'] = {
        f'{first} -> {second}': sum(a['reason'] == first and b['reason'] == second
                                   for a, b in zip(seed, evolved, strict=True))
        for first in OUTCOMES for second in OUTCOMES
    }
    return result


def effects_figure(summary, out):
    fields = list(LABELS)
    metrics = summary['all_32_primary']['metrics']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.1),
                             gridspec_kw={'width_ratios': [1.05, 1.3]})
    fig.subplots_adjust(left=.18, right=.96, top=.75, bottom=.25, wspace=.48)
    fig.suptitle('Generation 2 on the original development panel', x=.035, y=.98, ha='left')
    fig.text(.035, .89, '32 paired cases · existing exposed development data · selected after 5 search cases',
             color=style.SECONDARY, fontsize=10.5)
    for i, field in enumerate(fields):
        r = metrics[field]
        axes[0].plot([r['seed_mean'], r['generation_2_mean']], [i, i], color=style.RULE, lw=2)
        for name, offset, color, marker in [('seed', -.11, style.SECONDARY, 'D'),
                                           ('generation_2', .11, style.COBALT, 'o')]:
            lo, hi = r[name + '_ci95']
            if field == 'escaped':
                lo, hi = summary['all_32_primary']['post_hoc_escape_boundary_diagnostic'][name]
            axes[0].plot([lo, hi], [i + offset] * 2, color=color, lw=1.5)
            axes[0].scatter(r[name + '_mean'], i + offset, color=color, marker=marker, s=34, zorder=3)
        lo, hi = r['difference_ci95']
        axes[1].plot([lo, hi], [i, i], color=style.COBALT, lw=2)
        axes[1].scatter(r['difference'], i, color=style.COBALT, s=34, zorder=3)
        axes[1].text(1.025, i, f'{r["difference"]:+.3f}\n[{lo:+.3f}, {hi:+.3f}]',
                     transform=axes[1].get_yaxis_transform(), va='center', fontsize=9)
    axes[0].set_title('A  Program means', loc='left')
    axes[1].set_title('B  Paired differences', loc='left')
    axes[0].set(yticks=range(len(fields)), yticklabels=[LABELS[k] for k in fields],
                xlim=(-.04, 1.05), xticks=[0, .5, 1], xlabel='Score or fraction')
    axes[1].set(yticks=range(len(fields)), yticklabels=[], xlim=(-.09, 1.04),
                xticks=[0, .25, .5, .75, 1], xlabel='Generation 2 − seed')
    axes[1].axvline(0, color=style.SECONDARY, lw=.8)
    for ax in axes:
        ax.set_ylim(len(fields) - .5, -.5)
        ax.grid(axis='x')
        ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=style.SECONDARY, marker='D', ls='', label='Original seed'),
                        Line2D([], [], color=style.COBALT, marker='o', ls='', label='Generation 2')],
               loc='upper left', bbox_to_anchor=(.17, .855), ncol=2, fontsize=9)
    fig.text(.035, .12, '95% intervals: paired episode bootstrap; panel A escape uses separate-arm Wilson intervals (post hoc).',
             color=style.SECONDARY, fontsize=9.5)
    fig.text(.035, .075, 'Panel B escape: constant outcomes give a degenerate empirical interval, not certainty about population effects.',
             color=style.SECONDARY, fontsize=9.5)
    fig.text(.035, .035, 'Map accuracy reconstructs the current world on different trajectories; it is not a future-prediction comparison.',
             color=style.SECONDARY, fontsize=9.5)
    style.save_figure(fig, out / 'development-effects')


def cases_figure(seed, evolved, summary, out):
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(left=.095, right=.97, top=.76, bottom=.20, hspace=.38)
    fig.suptitle('Every development case, including failures', x=.035, y=.98, ha='left')
    fig.text(.035, .89, 'Cases 0–4 appeared in search; cases 5–31 extend this program’s evaluation on an existing development panel.',
             color=style.SECONDARY, fontsize=10)
    delta = [b['combined_score'] - a['combined_score'] for a, b in zip(seed, evolved, strict=True)]
    cases = [r['case'] for r in seed]
    for ax in axes:
        ax.axvspan(-.5, 4.5, color=style.WASH, zorder=0)
        ax.axvline(4.5, color=style.SECONDARY, lw=.8, ls=':')
        ax.grid(axis='y')
        ax.set_axisbelow(True)
    axes[0].axhline(0, color=style.SECONDARY, lw=.8)
    axes[0].vlines(cases, 0, delta, color=style.RULE, lw=1.5)
    for outcome, (label, color, marker) in OUTCOMES.items():
        idx = [i for i, r in enumerate(evolved) if r['reason'] == outcome]
        axes[0].scatter([cases[i] for i in idx], [delta[i] for i in idx],
                        color=color, marker=marker, s=32, zorder=3)
        for y, rows in enumerate([seed, evolved]):
            subset = [r['case'] for r in rows if r['reason'] == outcome]
            axes[1].scatter(subset, [y] * len(subset), color=color, marker=marker, s=43, zorder=3)
    axes[0].set_title('A  Paired combined-score difference', loc='left')
    axes[0].set_ylabel('Generation 2 − seed')
    lo, hi = min(0, min(delta)), max(delta)
    axes[0].set_ylim(lo - .055, hi + .13)
    for cases_range, key, center in [(range(5), 'search_cases_0_4', 2),
                                     (range(5, 32), 'additional_cases_5_31', 18)]:
        mean = summary[key]['metrics']['combined_score']['difference']
        axes[0].plot([min(cases_range) - .4, max(cases_range) + .4], [mean] * 2,
                     color=style.COBALT, lw=1, ls='--')
        axes[0].text(center, hi + .075, f'n = {len(cases_range)} · mean ΔF = {mean:+.3f}',
                     ha='center', fontsize=9, color=style.SECONDARY)
    axes[1].set_title('B  Observed outcome', loc='left')
    axes[1].set(yticks=[0, 1], yticklabels=['Original seed', 'Generation 2'],
                ylim=(1.55, -.55), xlim=(-.6, 31.6), xticks=[0, 4, 8, 12, 16, 20, 24, 28, 31],
                xlabel='Case identifier')
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, ls='', label=label)
                        for label, color, marker in OUTCOMES.values()], loc='lower center',
               bbox_to_anchor=(.5, .06), ncol=4)
    fig.text(.035, .025, 'Both panels use the same paired cases. Failed and invalid executions remain in denominators; no cases are held out.',
             color=style.SECONDARY, fontsize=9.5)
    style.save_figure(fig, out / 'development-cases')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    seed_dir = ROOT / 'artifacts/proposal/initial-evaluation'
    evolved_dir = out / 'gen2'
    inputs = [out / 'protocol.json']
    inputs += [directory / filename for directory in [seed_dir, evolved_dir]
               for filename in ['episodes.json', 'metrics.json', 'manifest.json']]
    inputs.append(evolved_dir / 'resource.json')
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    seed, evolved = (sorted(read(directory / 'episodes.json'), key=lambda r: r['case'])
                     for directory in [seed_dir, evolved_dir])
    validate(seed)
    validate(evolved)
    manifests = [read(directory / 'manifest.json') for directory in [seed_dir, evolved_dir]]
    if manifests[0]['evaluator'] != manifests[1]['evaluator']:
        raise ValueError('Different evaluator identities')
    for name, old_hash in manifests[0]['source_sha256'].items():
        if manifests[1]['source_sha256'].get(name) != old_hash:
            raise ValueError(f'Changed scientific dependency: {name}')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {
        'scope': 'Exposed development panel, not held-out assessment. Generation 2 was selected using cases 0–4.',
        'estimands': 'Equal episode means; generation 2 minus original seed on paired cases. Map accuracy is on-policy current-map reconstruction, not prediction.',
        'uncertainty': {'method': 'whole-episode paired percentile bootstrap', 'draws': BOOTSTRAP_DRAWS,
                        'rng_seed': BOOTSTRAP_SEED, 'level': .95,
                        'limitations': 'Descriptive intervals do not remove selection bias or development exposure; constant binary samples give degenerate percentile intervals.'},
        'all_32_primary': describe(seed, evolved, rng),
        'search_cases_0_4': describe(seed[:5], evolved[:5], rng),
        'additional_cases_5_31': describe(seed[5:], evolved[5:], rng),
        'new_execution': {'episodes': 32, 'model_calls': 0, 'new_sources': 0,
                          'transitions': sum(r['steps'] for r in evolved),
                          'resource': read(evolved_dir / 'resource.json')},
        'reused_seed': {'episodes': 32, 'transitions': sum(r['steps'] for r in seed),
                        'summed_episode_wall_seconds': sum(r['seconds'] for r in seed),
                        'cpu_seconds': None, 'note': 'Saved episodes reused unchanged; no new seed executions.'},
        'identities': {'seed': manifests[0], 'generation_2': manifests[1]},
    }
    write(out / 'summary.json', summary)
    rows = []
    for a, b in zip(seed, evolved, strict=True):
        row = {'case': a['case'], 'subset': 'search' if a['case'] < 5 else 'additional_development',
               'seed_outcome': a['reason'], 'generation_2_outcome': b['reason']}
        for key in METRICS:
            av, bv = matrix([a])[0, METRICS.index(key)], matrix([b])[0, METRICS.index(key)]
            row.update({f'seed_{key}': av, f'generation_2_{key}': bv, f'difference_{key}': bv - av})
        rows.append(row)
    with (out / 'paired-episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    style.apply_theme()
    figures = out / 'figures'
    effects_figure(summary, figures)
    cases_figure(seed, evolved, summary, figures)
    (out / 'captions.md').write_text(
        '# Stage 3 development figures\n\n'
        '**development-effects.** Means and paired generation-2-minus-seed differences on '
        '32 existing development cases, with 95% percentile intervals from 10,000 whole-episode '
        'paired bootstrap draws (seed 310932), except panel A escape whiskers use separate-arm '
        '95% Wilson intervals, added post hoc to expose the uncertainty concealed by constant '
        'binary observations. These are not paired effect intervals. All bootstrap metrics and both arms share resampled case '
        'indices. Accuracy is the equal-weight mean of each episode’s correct/audited-cell '
        'ratio; coverage is the last-record fraction of map cells. Failed executions remain '
        'in denominators. These intervals are descriptive and do not correct selection bias. '
        'A constant observed binary outcome yields a degenerate percentile interval; that '
        'does not establish a population probability of zero or one.\n\n'
        '**development-cases.** Every case’s paired combined-score difference and both '
        'outcomes. Shading marks the five cases already used by evolutionary selection. '
        'Dashed lines show descriptive subset mean differences; cases 5–31 are additional '
        'evaluations of this candidate on an existing exposed development panel. Outcome '
        'colors and symbols in panel A refer to generation 2. Neither panel establishes '
        'held-out performance, prediction learning, or a causal adaptation benefit.\n')
    outputs = [out / 'summary.json', out / 'paired-episodes.csv', out / 'captions.md']
    outputs += [figures / f'{stem}.{extension}' for stem in ['development-effects', 'development-cases']
                for extension in ['svg', 'pdf', 'png']]
    sources = [Path(__file__), ROOT / 'scripts/chromatic_fields.py', ROOT / 'scripts/visual_theme.py']
    relative = lambda path: str(path.relative_to(ROOT))
    write(out / 'rendering-manifest.json', {
        'style': style.PRESENTATION_VERSION,
        'reproduction': '.venv/bin/python -m proposal.development_report',
        'scientific_execution': 'None: saved-data analysis and rendering only.',
        'input_sha256': {relative(p): sha256(p) for p in inputs},
        'source_sha256': {relative(p): sha256(p) for p in sources},
        'output_sha256': {relative(p): sha256(p) for p in outputs},
    })
    print(json.dumps({'output_dir': str(out), 'all_32': summary['all_32_primary'],
                      'new_execution': summary['new_execution']}, indent=2))


if __name__ == '__main__':
    main()
