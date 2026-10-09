"""Analyse the frozen Stage 4 selection panel; never run candidates or models."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from proposal.development_report import METRICS, OUTCOMES, matrix, read, sha256, wilson, write
from scripts import chromatic_fields as style

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/proposal/stage4-selection'
ARMS = ('seed', 'generation_1', 'generation_2')
ARM_STYLE = {'seed': ('Original seed', style.SECONDARY, 'D'),
             'generation_1': ('Generation 1', style.ORANGE, '^'),
             'generation_2': ('Generation 2', style.COBALT, 'o')}
PAIRS = (('generation_1', 'seed', style.ORANGE),
         ('generation_2', 'seed', style.COBALT),
         ('generation_2', 'generation_1', style.MAGENTA))
FIELDS = {'combined_score': 'Combined F', 'task': 'Task S',
          'model_accuracy': 'Map accuracy A', 'final_coverage': 'Map coverage',
          'escaped': 'Escape fraction'}
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 410964, 10000


def validate(rows):
    if len(rows) != 64 or [r['case'] for r in rows] != list(range(64)):
        raise ValueError('Expected exactly 64 paired ordinal cases, 0–63')
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
                raise ValueError(f'Incorrect saved {key}: ordinal case {r["case"]}')


def analyse(rows, resources):
    values = {arm: matrix(rows[arm]) for arm in ARMS}
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 64, (BOOTSTRAP_DRAWS, 64))
    sampled = {arm: values[arm][indices].mean(axis=1) for arm in ARMS}
    summary = {
        'scope': 'Selection-validation panel, disjoint from search and development. This panel selects the descendant; its selected result is selection-biased and is not final assessment.',
        'estimands': 'Equal episode means on 64 paired cases. Accuracy is the mean of within-episode correct/audited-cell ratios on each policy trajectory; it is current-map reconstruction, not prediction.',
        'selection_rule': 'Highest mean combined F among generation 1 and generation 2, then mean task S, then earlier generation. The original seed is a reference, not an eligible descendant.',
        'uncertainty': {'method': 'whole-episode paired percentile bootstrap', 'draws': BOOTSTRAP_DRAWS,
                        'rng_seed': BOOTSTRAP_SEED, 'level': .95,
                        'shared_resampling': 'All arms, metrics and pairwise contrasts use identical case draws.',
                        'escape_arm_intervals': 'Prospectively specified separate-arm 95% Wilson intervals.',
                        'limitations': 'Descriptive, unadjusted intervals do not remove selection bias. Constant binary observations give degenerate paired percentile intervals, not certainty.'},
        'arms': {}, 'pairwise': {},
    }
    for arm in ARMS:
        outcomes = {reason: sum(r['reason'] == reason for r in rows[arm]) for reason in OUTCOMES}
        summary['arms'][arm] = {
            'episodes': 64, 'transitions': sum(r['steps'] for r in rows[arm]),
            'outcomes': outcomes, 'escape_wilson_ci95': wilson(outcomes['escaped'], 64),
            'metrics': {key: {'mean': float(values[arm][:, i].mean()),
                              'ci95': np.quantile(sampled[arm][:, i], [.025, .975]).tolist()}
                        for i, key in enumerate(METRICS)},
            'resource': resources[arm],
        }
    for first, second, _ in PAIRS:
        metrics = {key: {'difference': float((values[first][:, i] - values[second][:, i]).mean()),
                         'ci95': np.quantile(sampled[first][:, i] - sampled[second][:, i], [.025, .975]).tolist()}
                   for i, key in enumerate(METRICS)}
        for metric in metrics.values():
            metric['degenerate_ci'] = math.isclose(*metric['ci95'], rel_tol=0., abs_tol=1e-14)
        summary['pairwise'][f'{first}_minus_{second}'] = {
            'first': first, 'second': second, 'metrics': metrics,
            'objective_contributions': {'task_difference_times_0_6': .6 * metrics['task']['difference'],
                                        'accuracy_difference_times_0_4': .4 * metrics['model_accuracy']['difference'],
                                        'note': 'Contributions sum to F difference only when neither arm has invalid executions.'},
            'outcome_pairs': {f'{a} -> {b}': sum(x['reason'] == a and y['reason'] == b
                                                 for x, y in zip(rows[second], rows[first], strict=True))
                              for a in OUTCOMES for b in OUTCOMES},
        }
    rank = sorted(ARMS[1:], key=lambda arm: (-summary['arms'][arm]['metrics']['combined_score']['mean'],
                                           -summary['arms'][arm]['metrics']['task']['mean'], int(arm[-1])))
    summary.update({'selection_ranking': rank, 'selected_arm': rank[0], 'selected_generation': int(rank[0][-1]),
                    'execution': {'episodes': 192, 'transitions': sum(r['steps'] for arm in ARMS for r in rows[arm]),
                                  'experiment_model_calls': 0, 'new_sources': 0,
                                  'resource_by_arm': resources}})
    return summary, values


def effects_figure(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.7), gridspec_kw={'width_ratios': [1, 1.35]})
    fig.subplots_adjust(left=.17, right=.96, top=.72, bottom=.24, wspace=.25)
    fig.suptitle('Selecting an existing program on 64 disjoint cases', x=.03, y=.98, ha='left')
    fig.text(.03, .895, f'Prespecified rule selected generation {summary["selected_generation"]} · this panel supports selection, not final assessment',
             color=style.SECONDARY, fontsize=10.5)
    for y, field in enumerate(FIELDS):
        for j, arm in enumerate(ARMS):
            _, color, marker = ARM_STYLE[arm]
            m = summary['arms'][arm]['metrics'][field]
            lo, hi = summary['arms'][arm]['escape_wilson_ci95'] if field == 'escaped' else m['ci95']
            yy = y + (j - 1) * .20
            axes[0].plot([lo, hi], [yy, yy], color=color, lw=1.5)
            axes[0].scatter(m['mean'], yy, color=color, marker=marker, s=34, zorder=3)
        for j, (first, second, color) in enumerate(PAIRS):
            m = summary['pairwise'][f'{first}_minus_{second}']['metrics'][field]
            yy = y + (j - 1) * .20
            axes[1].plot(m['ci95'], [yy, yy], color=color, lw=1.7)
            axes[1].scatter(m['difference'], yy, color=color, marker=['^', 'o', 's'][j], s=32, zorder=3)
    axes[0].set_title('A  Program means', loc='left')
    axes[0].set(yticks=range(len(FIELDS)), yticklabels=list(FIELDS.values()),
                xlim=(-.04, 1.04), xticks=[0, .25, .5, .75, 1], xlabel='Score or fraction')
    axes[1].set_title('B  Paired differences', loc='left')
    axes[1].set(yticks=range(len(FIELDS)), yticklabels=[], xlabel='First program − second program')
    axes[1].axvline(0, color=style.SECONDARY, lw=.8)
    for ax in axes:
        ax.set_ylim(4.5, -.5)
        ax.grid(axis='x')
        ax.set_axisbelow(True)
    axes[0].legend(handles=[Line2D([], [], color=c, marker=m, ls='', label=label)
                           for label, c, m in ARM_STYLE.values()], loc='lower left', bbox_to_anchor=(-.06, 1.15),
                   ncol=1, fontsize=9, handletextpad=.4, labelspacing=.25)
    pair_labels = ['Generation 1 − seed', 'Generation 2 − seed', 'Generation 2 − generation 1']
    axes[1].legend(handles=[Line2D([], [], color=c, marker=m, ls='', label=label)
                           for (_, _, c), m, label in zip(PAIRS, ['^', 'o', 's'], pair_labels, strict=True)],
                   loc='lower left', bbox_to_anchor=(-.02, 1.15), ncol=1, fontsize=9, handletextpad=.4, labelspacing=.25)
    fig.text(.03, .13, '95% intervals: paired whole-case bootstrap; panel A escape uses prospectively specified separate-arm Wilson intervals.',
             color=style.SECONDARY, fontsize=9)
    fig.text(.03, .085, 'The combined objective is 0.6 task + 0.4 reconstruction accuracy. All outcomes, including invalid executions, retain their denominator.',
             color=style.SECONDARY, fontsize=9)
    fig.text(.03, .04, 'Accuracy measures the current map on different trajectories. These results do not identify prediction learning or its causal benefit.',
             color=style.SECONDARY, fontsize=9)
    style.save_figure(fig, out / 'selection-effects')


def cases_figure(rows, values, summary, out):
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 5.9), sharex=True, gridspec_kw={'height_ratios': [1.7, 1]})
    fig.subplots_adjust(left=.12, right=.97, top=.80, bottom=.22, hspace=.45)
    fig.suptitle('Every selection case: program gains and failures', x=.03, y=.98, ha='left')
    fig.text(.03, .895, '64 paired case ordinals · all cases remain visible · environment seeds stay in the private evaluator pool',
             color=style.SECONDARY, fontsize=10.5)
    x = np.arange(64)
    for arm in ARMS[1:]:
        label, color, marker = ARM_STYLE[arm]
        delta = values[arm][:, 0] - values['seed'][:, 0]
        axes[0].plot(x, delta, color=color, marker=marker, ms=3.2, lw=.8, alpha=.9, label=label)
    axes[0].axhline(0, color=style.SECONDARY, lw=.8)
    axes[0].set_title('A  Paired combined-score differences', loc='left')
    axes[0].set_ylabel('Descendant − seed')
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + .13)
    axes[0].legend(loc='upper right', ncol=2, fontsize=9)
    for y, arm in enumerate(ARMS):
        for reason, (_, color, marker) in OUTCOMES.items():
            cases = [r['case'] for r in rows[arm] if r['reason'] == reason]
            axes[1].scatter(cases, [y] * len(cases), color=color, marker=marker, s=28, zorder=3)
    axes[1].set_title('B  Observed outcomes', loc='left')
    axes[1].set(yticks=[0, 1, 2], yticklabels=[ARM_STYLE[arm][0] for arm in ARMS], ylim=(2.6, -.6),
                xlim=(-.7, 63.7), xticks=[0, 8, 16, 24, 32, 40, 48, 56, 63], xlabel='Private-pool case ordinal')
    for ax in axes:
        ax.grid(axis='y')
        ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, ls='', label=label)
                        for label, color, marker in OUTCOMES.values()], loc='lower center', bbox_to_anchor=(.5, .075), ncol=4)
    fig.text(.03, .035, f'192 episodes · {summary["execution"]["transitions"]:,} transitions · zero experiment-model calls · selection-biased evidence, not final assessment',
             color=style.SECONDARY, fontsize=9.5)
    style.save_figure(fig, out / 'selection-cases')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    inputs = [out / name for name in ['protocol.json', 'pool-manifest.json', 'execution.json']]
    inputs += [out / arm / name for arm in ARMS for name in ['episodes.json', 'resource.json', 'manifest.json']]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = read(out / 'protocol.json')
    if (protocol['case_count'] != 64 or protocol['analysis']['bootstrap_replicates'] != BOOTSTRAP_DRAWS
            or protocol['analysis']['bootstrap_seed'] != BOOTSTRAP_SEED):
        raise ValueError('Report constants differ from the frozen selection protocol')
    pool = read(out / 'pool-manifest.json')
    if pool['protocol_sha256'] != sha256(out / 'protocol.json') or pool['case_count'] != 64:
        raise ValueError('Pool manifest does not match the frozen protocol')
    manifests = {arm: read(out / arm / 'manifest.json') for arm in ARMS}
    for arm, manifest in manifests.items():
        candidate_path = protocol['programs'][arm]['path']
        if (manifest['evaluator'] != protocol['evaluator'] or manifest['pool_sha256'] != pool['pool_sha256']
                or manifest['source_sha256'] != protocol['source_sha256'] or manifest['episodes'] != 64
                or manifest['candidate_sha256'] != protocol['source_sha256'][candidate_path]):
            raise ValueError(f'Incompatible arm evaluator, pool or source identity: {arm}')
    rows = {arm: sorted(read(out / arm / 'episodes.json'), key=lambda r: r['case']) for arm in ARMS}
    for arm in ARMS:
        validate(rows[arm])
    resources = {arm: read(out / arm / 'resource.json') for arm in ARMS}
    summary, values = analyse(rows, resources)
    execution = read(out / 'execution.json')
    if (execution['status'] != 'complete' or execution['episodes'] != 192
            or execution['transitions'] != summary['execution']['transitions']):
        raise ValueError('Execution record does not match complete episode data')
    summary['execution']['controller'] = execution
    summary['identities'] = {'evaluator': protocol['evaluator'], 'pool_sha256': pool['pool_sha256'],
                            'source_sha256': protocol['source_sha256'],
                            'candidate_sha256': {arm: m['candidate_sha256'] for arm, m in manifests.items()}}
    summary['protocol_sha256'] = sha256(out / 'protocol.json')
    selected = protocol['programs'][summary['selected_arm']]
    summary['selected_program'] = {**selected, 'sha256': protocol['source_sha256'][selected['path']]}
    write(out / 'summary.json', summary)
    table = []
    for i in range(64):
        r = {'case_ordinal': i}
        for arm in ARMS:
            r[f'{arm}_outcome'] = rows[arm][i]['reason']
            r.update({f'{arm}_{key}': values[arm][i, j] for j, key in enumerate(METRICS)})
        for first, second, _ in PAIRS:
            r.update({f'{first}_minus_{second}_{key}': values[first][i, j] - values[second][i, j]
                      for j, key in enumerate(METRICS)})
        table.append(r)
    with (out / 'paired-episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    style.apply_theme()
    figures = out / 'figures'
    effects_figure(summary, figures)
    cases_figure(rows, values, summary, figures)
    (out / 'captions.md').write_text(
        '# Stage 4 selection-validation figures\n\n'
        '**selection-effects.** Three frozen programs on 64 paired cases disjoint from search and development. '
        'Generation 1 and generation 2 were nominated before drawing the private panel. Selection uses mean '
        'F, then S, then earlier generation; the original seed is a reference. Panel A reports program means '
        'and panel B reports all three paired contrasts. Intervals are descriptive 95% percentiles from '
        '10,000 whole-case bootstrap draws (seed 410964), sharing indices across every arm and metric. '
        'Panel A escape intervals use prospectively declared separate-arm Wilson intervals; they are not '
        'paired-effect intervals. Accuracy is the mean of each episode’s pooled correct/audited-cell ratio, '
        'not a cell-weighted pool across episodes. Coverage is the last recorded map fraction. Invalid '
        'executions remain in denominators with zero combined score. Constant empirical binary samples '
        'can yield degenerate bootstrap intervals; this is not population certainty. Selection bias and '
        'multiple comparisons are not removed by these intervals.\n\n'
        '**selection-cases.** Every ordinal case’s descendant-minus-seed combined score and all three '
        'programs’ outcomes. Actual environment seeds remain private. Connecting lines aid tracing and '
        'do not imply temporal order or a causal relation between cases. All failures are retained. '
        'Both figures measure task performance and current-map reconstruction on different policy '
        'trajectories; neither measures matched prediction learning or a causal adaptation benefit. '
        'This panel is used to select a program and therefore is not a final frozen-program assessment.\n')
    outputs = [out / name for name in ['summary.json', 'paired-episodes.csv', 'captions.md']]
    outputs += [figures / f'{stem}.{ext}' for stem in ['selection-effects', 'selection-cases'] for ext in ['svg', 'pdf', 'png']]
    sources = [Path(__file__), ROOT / 'proposal/development_report.py', ROOT / 'scripts/chromatic_fields.py', ROOT / 'scripts/visual_theme.py']
    rel = lambda p: str(p.relative_to(ROOT))
    write(out / 'rendering-manifest.json', {'style': style.PRESENTATION_VERSION,
          'reproduction': 'OPENBLAS_NUM_THREADS=1 .venv/bin/python -m proposal.selection_report',
          'scientific_execution': 'None: saved public episode analysis and rendering only.',
          'input_sha256': {rel(p): sha256(p) for p in inputs}, 'source_sha256': {rel(p): sha256(p) for p in sources},
          'output_sha256': {rel(p): sha256(p) for p in outputs}})
    print(f'Selected generation {summary["selected_generation"]}; 192 episodes, {summary["execution"]["transitions"]} transitions')
    for arm in ARMS:
        a = summary['arms'][arm]
        print(f'{arm}: F={a["metrics"]["combined_score"]["mean"]:.6f}; outcomes={a["outcomes"]}')
    for key, contrast in summary['pairwise'].items():
        m = contrast['metrics']['combined_score']
        print(f'{key}: ΔF={m["difference"]:.6f}; CI={m["ci95"]}')


if __name__ == '__main__':
    main()
