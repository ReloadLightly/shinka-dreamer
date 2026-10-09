"""Analyse the frozen Stage 5 assessment; never access pools or run programs."""
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
DEFAULT = ROOT / 'artifacts/proposal/stage5-assessment'
FREEZE = ROOT / 'artifacts/proposal/stage4-selection/assessment-freeze.json'
ARMS = ('seed', 'generation_2')
ARM_STYLE = {'seed': ('Original seed', style.SECONDARY, 'D'),
             'generation_2': ('Selected generation 2', style.COBALT, 'o')}
FIELDS = {'combined_score': 'Combined F', 'task': 'Task S',
          'model_accuracy': 'Map accuracy A', 'final_coverage': 'Map coverage',
          'escaped': 'Escape fraction'}
N, DRAWS, RNG_SEED = 256, 10000, 5109256


def validate(rows):
    if len(rows) != N or [r['case'] for r in rows] != list(range(N)):
        raise ValueError('Expected 256 unique paired case ordinals, 0–255')
    for r in rows:
        if r['reason'] not in OUTCOMES:
            raise ValueError(f'Unknown outcome in ordinal case {r["case"]}')
        accuracy = r['map_correct'] / r['map_audited'] if r['map_audited'] else 0.
        raw = (.1 * r['steps'] + 20 * r['keys'] + 30 * r['door_open']
               + 100 * (r['reason'] == 'escaped') - 50 * (r['reason'] == 'caught'))
        task = min(1., max(0., (raw + 50) / 250))
        combined = 0. if r['error'] is not None else .6 * task + .4 * accuracy
        for key, actual in [('model_accuracy', accuracy), ('task', task), ('combined_score', combined)]:
            if not math.isclose(r[key], actual, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Incorrect {key}: ordinal case {r["case"]}')


def analyse(rows, resources):
    values = {arm: matrix(rows[arm]) for arm in ARMS}
    # One index matrix preserves all within-case dependencies. Apply it to one
    # metric at a time: ~20 MiB temporary, rather than ~450 MiB for both arms.
    indices = np.random.default_rng(RNG_SEED).integers(0, N, (DRAWS, N))
    sampled = {arm: np.empty((DRAWS, len(METRICS))) for arm in ARMS}
    for arm in ARMS:
        for j in range(len(METRICS)):
            sampled[arm][:, j] = values[arm][:, j][indices].mean(axis=1)
    summary = {
        'scope': 'Fresh held-out assessment of one previously selected and frozen program against the original seed. The programs, contrast, sample size and analysis were frozen before the pool was drawn.',
        'estimands': 'Equal episode means on 256 paired cases. A is the mean of within-episode correct/audited-cell ratios; current-map reconstruction on different policy trajectories is not prediction learning.',
        'uncertainty': {'method': 'whole-episode paired percentile bootstrap', 'draws': DRAWS,
                        'rng_seed': RNG_SEED, 'level': .95,
                        'shared_resampling': 'Identical paired case draws for both arms and every metric.',
                        'escape_arm_intervals': 'Prospectively specified separate-arm Wilson 95% intervals.',
                        'limitations': 'Secondary intervals are unadjusted. A degenerate empirical binary bootstrap is not population certainty.'},
        'arms': {}, 'contrast': {},
        'claim_limits': ['Original seed is a weak greedy/random baseline, not a competent handcrafted pathfinder.',
                         'One selected program from one short search does not establish reliable discovery or the causal value of native search mechanisms.',
                         'No causal intervention on memory, planning or learning; no matched-experience prediction comparison.'],
    }
    for arm in ARMS:
        outcomes = {reason: sum(r['reason'] == reason for r in rows[arm]) for reason in OUTCOMES}
        summary['arms'][arm] = {
            'episodes': N, 'transitions': sum(r['steps'] for r in rows[arm]),
            'outcomes': outcomes, 'escape_wilson_ci95': wilson(outcomes['escaped'], N),
            'metrics': {key: {'mean': float(values[arm][:, j].mean()),
                              'ci95': np.quantile(sampled[arm][:, j], [.025, .975]).tolist()}
                        for j, key in enumerate(METRICS)},
            'resource': resources[arm],
            'mean_episode_wall_seconds': sum(r['seconds'] for r in rows[arm]) / N,
            'memory_updates': 'Ordinary within-episode map-memory updates remain active; frozen source is not a frozen-learning intervention.',
        }
    metrics = {}
    for j, key in enumerate(METRICS):
        lo, hi = np.quantile(sampled['generation_2'][:, j] - sampled['seed'][:, j], [.025, .975])
        metrics[key] = {'difference': float((values['generation_2'][:, j] - values['seed'][:, j]).mean()),
                        'ci95': [float(lo), float(hi)],
                        'degenerate_ci': math.isclose(lo, hi, rel_tol=0., abs_tol=1e-14)}
    summary['contrast'] = {
        'first': 'generation_2', 'second': 'seed', 'metrics': metrics,
        'primary': {'metric': 'combined_score', 'meaningful_difference': .05,
                    'lower_ci_exceeds_meaningful_difference': metrics['combined_score']['ci95'][0] > .05},
        'objective_contributions': {'task_difference_times_0_6': .6 * metrics['task']['difference'],
                                    'accuracy_difference_times_0_4': .4 * metrics['model_accuracy']['difference'],
                                    'invalid_execution_correction': metrics['combined_score']['difference'] - .6 * metrics['task']['difference'] - .4 * metrics['model_accuracy']['difference']},
        'outcome_pairs': {f'{a} -> {b}': sum(x['reason'] == a and y['reason'] == b
                                            for x, y in zip(rows['seed'], rows['generation_2'], strict=True))
                          for a in OUTCOMES for b in OUTCOMES},
        'escape_discordance': {
            'generation_2_only': sum(a['reason'] != 'escaped' and b['reason'] == 'escaped' for a, b in zip(rows['seed'], rows['generation_2'], strict=True)),
            'seed_only': sum(a['reason'] == 'escaped' and b['reason'] != 'escaped' for a, b in zip(rows['seed'], rows['generation_2'], strict=True))},
    }
    summary['execution'] = {'episodes': 2 * N, 'transitions': sum(r['steps'] for arm in ARMS for r in rows[arm]),
                            'experiment_model_calls': 0, 'new_sources': 0, 'resource_by_arm': resources}
    return summary, values


def effects_figure(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.7), gridspec_kw={'width_ratios': [1, 1.25]})
    fig.subplots_adjust(left=.16, right=.80, top=.76, bottom=.25, wspace=.36)
    fig.suptitle('A frozen program on 256 fresh paired cases', x=.03, y=.98, ha='left')
    fig.text(.03, .895, 'Original maze and objective · generation 2 selected before assessment · one prespecified primary contrast',
             color=style.SECONDARY, fontsize=10.5)
    for y, field in enumerate(FIELDS):
        for j, arm in enumerate(ARMS):
            _, color, marker = ARM_STYLE[arm]
            m = summary['arms'][arm]['metrics'][field]
            lo, hi = summary['arms'][arm]['escape_wilson_ci95'] if field == 'escaped' else m['ci95']
            yy = y + (j - .5) * .25
            axes[0].plot([lo, hi], [yy, yy], color=color, lw=1.7)
            axes[0].scatter(m['mean'], yy, color=color, marker=marker, s=35, zorder=3)
        m = summary['contrast']['metrics'][field]
        axes[1].plot(m['ci95'], [y, y], color=style.COBALT, lw=2)
        axes[1].scatter(m['difference'], y, color=style.COBALT, s=35, zorder=3)
        axes[1].text(1.06, y, f'{m["difference"]:+.3f}\n[{m["ci95"][0]:+.3f}, {m["ci95"][1]:+.3f}]',
                     transform=axes[1].get_yaxis_transform(), va='center', fontsize=9.5)
    axes[0].set_title('A  Program means', loc='left')
    axes[0].set(yticks=range(5), yticklabels=list(FIELDS.values()), xlim=(-.035, 1.04),
                xticks=[0, .5, 1], xlabel='Score or fraction')
    axes[1].set_title('B  Paired effects and 95% intervals', loc='left')
    axes[1].set(yticks=range(5), yticklabels=[], xlim=(-.06, 1.02),
                xticks=[0, .25, .5, .75, 1], xlabel='Generation 2 − seed')
    axes[1].axvline(0, color=style.SECONDARY, lw=.8)
    # The threshold belongs to F alone; a short tick avoids suggesting that
    # the same scientific threshold applies to all secondary quantities.
    axes[1].plot([.05, .05], [-.27, .27], color=style.ORANGE, lw=2)
    for ax in axes:
        ax.set_ylim(4.5, -.5)
        ax.grid(axis='x')
        ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=c, marker=m, ls='', label=label)
                        for label, c, m in ARM_STYLE.values()], loc='upper left',
               bbox_to_anchor=(.155, .87), ncol=2, fontsize=9.5)
    fig.text(.03, .15, 'Orange tick: prespecified meaningful ΔF = 0.05. The original seed is a weak baseline, not a strong pathfinding comparator.',
             color=style.SECONDARY, fontsize=9.5)
    fig.text(.03, .095, '95% intervals: whole paired episodes; panel A escape uses separate-arm Wilson intervals. Secondary intervals are unadjusted.',
             color=style.SECONDARY, fontsize=9.5)
    fig.text(.03, .04, 'A measures current-map reconstruction under different actions. These results do not isolate learning or establish discovery reliability.',
             color=style.SECONDARY, fontsize=9.5)
    style.save_figure(fig, out / 'assessment-effects')


def cases_figure(rows, values, summary, out):
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.3), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(left=.13, right=.97, top=.80, bottom=.25, hspace=.48)
    fig.suptitle('Every assessment case: gains, failures and outcomes', x=.03, y=.98, ha='left')
    fig.text(.03, .90, '256 paired ordinal cases · environment seeds remain private · failed executions remain in all denominators',
             color=style.SECONDARY, fontsize=10.5)
    x = np.arange(N)
    difference = values['generation_2'][:, 0] - values['seed'][:, 0]
    axes[0].vlines(x, 0, difference, color=style.RULE, lw=.7)
    for reason, (label, color, marker) in OUTCOMES.items():
        cases = [r['case'] for r in rows['generation_2'] if r['reason'] == reason]
        axes[0].scatter(cases, difference[cases], color=color, marker=marker, s=16, zorder=3)
        for y, arm in enumerate(ARMS):
            ids = [r['case'] for r in rows[arm] if r['reason'] == reason]
            axes[1].scatter(ids, [y] * len(ids), color=color, marker=marker, s=8, zorder=3)
    # Highlight the rare seed successes after every outcome layer so that
    # adjacent dense death/timeout symbols cannot obscure observed escapes.
    seed_escapes = [r['case'] for r in rows['seed'] if r['reason'] == 'escaped']
    axes[1].scatter(seed_escapes, [0] * len(seed_escapes), color=style.COBALT,
                    edgecolor=style.BACKGROUND, linewidth=.9, marker='o', s=42, zorder=6)
    for case in seed_escapes:
        axes[1].annotate(f'Seed escape {case}', (case, 0), xytext=(0, 17),
                         textcoords='offset points', ha='center', fontsize=8,
                         color=style.COBALT, arrowprops={'arrowstyle': '-', 'color': style.COBALT, 'lw': .6})
    axes[0].axhline(0, color=style.SECONDARY, lw=.8)
    axes[0].axhline(summary['contrast']['metrics']['combined_score']['difference'],
                    color=style.COBALT, lw=1., ls='--')
    axes[0].set_title('A  Combined-score gain; symbols show generation 2 outcome', loc='left')
    axes[0].set_ylabel('Generation 2 − seed')
    axes[1].set_title('B  Both programs’ outcomes', loc='left')
    axes[1].set(yticks=[0, 1], yticklabels=['Original seed', 'Generation 2'], ylim=(1.6, -.6),
                xlim=(-2, 257), xticks=[0, 32, 64, 96, 128, 160, 192, 224, 255], xlabel='Private-pool case ordinal')
    for ax in axes:
        ax.grid(axis='y')
        ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, ls='', label=label)
                        for label, color, marker in OUTCOMES.values()], loc='lower center',
               bbox_to_anchor=(.5, .13), ncol=4)
    counts = summary['arms']['generation_2']['outcomes']
    timeout_label = 'timeout' if counts['timeout'] == 1 else 'timeouts'
    fig.text(.03, .085, f'Generation 2: {counts["escaped"]} escapes, {counts["caught"]} captures, {counts["timeout"]} {timeout_label}, {counts["invalid"]} invalid. Dashed line: mean paired ΔF.',
             color=style.SECONDARY, fontsize=9.5)
    cpu = sum(summary['arms'][arm]['resource']['total_cpu_seconds'] for arm in ARMS)
    fig.text(.03, .04, f'512 episodes · {summary["execution"]["transitions"]:,} transitions · {cpu:.2f} episode-worker CPU-seconds · zero experiment-model calls',
             color=style.SECONDARY, fontsize=9.5)
    style.save_figure(fig, out / 'assessment-cases')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    out = parser.parse_args().output_dir.resolve()
    inputs = [out / name for name in ['protocol.json', 'pool-manifest.json', 'execution.json']]
    inputs += [out / arm / name for arm in ARMS for name in ['episodes.json', 'metrics.json', 'resource.json', 'manifest.json']]
    inputs.append(FREEZE)
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    protocol, freeze, pool = read(out / 'protocol.json'), read(FREEZE), read(out / 'pool-manifest.json')
    for spec in (protocol, freeze):
        if (spec['case_count'] != N or spec['analysis']['bootstrap_replicates'] != DRAWS
                or spec['analysis']['bootstrap_seed'] != RNG_SEED):
            raise ValueError('Report constants differ from frozen assessment specification')
    if (pool['protocol_sha256'] != sha256(out / 'protocol.json') or pool['case_count'] != N
            or protocol['evaluator'] != freeze['evaluator']
            or protocol['assessment_freeze']['sha256'] != sha256(FREEZE)
            or protocol['analysis'] != freeze['analysis']
            or protocol['primary_contrast'] != freeze['primary_contrast']
            or any(protocol['source_sha256'].get(path) != digest
                   for path, digest in freeze['source_sha256'].items())):
        raise ValueError('Assessment freeze, protocol and pool identities differ')
    manifests = {arm: read(out / arm / 'manifest.json') for arm in ARMS}
    for arm, manifest in manifests.items():
        candidate = freeze['control' if arm == 'seed' else 'selected_program']
        if (manifest['evaluator'] != protocol['evaluator'] or manifest['pool_sha256'] != pool['pool_sha256']
                or manifest['source_sha256'] != protocol['source_sha256'] or manifest['episodes'] != N
                or manifest['candidate_sha256'] != candidate['sha256']
                or manifest['assessment_freeze_sha256'] != sha256(FREEZE)
                or protocol['programs'][arm]['path'] != candidate['path']):
            raise ValueError(f'Incompatible candidate, evaluator or pool: {arm}')
    rows = {arm: sorted(read(out / arm / 'episodes.json'), key=lambda r: r['case']) for arm in ARMS}
    for arm in ARMS:
        validate(rows[arm])
    resources = {arm: read(out / arm / 'resource.json') for arm in ARMS}
    summary, values = analyse(rows, resources)
    for arm in ARMS:
        saved = read(out / arm / 'metrics.json')
        expected = summary['arms'][arm]['metrics']
        if not math.isclose(saved['combined_score'], expected['combined_score']['mean'], abs_tol=1e-12):
            raise ValueError(f'Saved combined-score aggregate mismatch: {arm}')
        for key, metric in [('mean_task_score', 'task'), ('mean_model_accuracy', 'model_accuracy'),
                            ('mean_map_coverage', 'final_coverage'), ('avg_steps', 'steps'), ('avg_keys', 'keys')]:
            if not math.isclose(saved['public'][key], expected[metric]['mean'], abs_tol=1e-12):
                raise ValueError(f'Saved aggregate mismatch: {arm}/{key}')
        for outcome, count in summary['arms'][arm]['outcomes'].items():
            if saved['public'][outcome] != count:
                raise ValueError(f'Saved outcome mismatch: {arm}/{outcome}')
    execution = read(out / 'execution.json')
    if (execution['status'] != 'complete' or execution['episodes'] != 2 * N
            or execution['transitions'] != summary['execution']['transitions']):
        raise ValueError('Execution record disagrees with complete episode data')
    summary['execution']['controller'] = execution
    summary['identities'] = {'evaluator': protocol['evaluator'], 'pool_sha256': pool['pool_sha256'],
                            'source_sha256': protocol['source_sha256'],
                            'candidate_sha256': {arm: m['candidate_sha256'] for arm, m in manifests.items()}}
    summary['protocol_sha256'] = sha256(out / 'protocol.json')
    summary['assessment_freeze_sha256'] = sha256(FREEZE)
    write(out / 'summary.json', summary)
    table = []
    for i in range(N):
        row = {'case_ordinal': i}
        for arm in ARMS:
            row[f'{arm}_outcome'] = rows[arm][i]['reason']
            row[f'{arm}_wall_seconds'] = rows[arm][i]['seconds']
            row.update({f'{arm}_{key}': values[arm][i, j] for j, key in enumerate(METRICS)})
        row.update({f'generation_2_minus_seed_{key}': values['generation_2'][i, j] - values['seed'][i, j]
                    for j, key in enumerate(METRICS)})
        table.append(row)
    with (out / 'paired-episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    style.apply_theme()
    figures = out / 'figures'
    effects_figure(summary, figures)
    cases_figure(rows, values, summary, figures)
    (out / 'captions.md').write_text(
        '# Stage 5 frozen-program assessment figures\n\n'
        '**assessment-effects.** The previously selected generation 2 and the original seed on 256 fresh '
        'paired cases. Sources, analysis and sample size were frozen before drawing the assessment pool. '
        'Panel A gives program means and panel B gives generation 2 minus seed. The sole primary '
        'contrast is mean combined F; the short orange tick marks its prospectively declared meaningful '
        'difference of 0.05. Every interval is 95%; effects and nonbinary arm means use 10,000 whole-case '
        'paired bootstrap draws (seed 5109256), shared across all arms and metrics. Separate-arm escape '
        'intervals are Wilson intervals, as prospectively specified; they are not paired-effect intervals. '
        'Secondary intervals are unadjusted. Degenerate binary bootstrap intervals are flagged in the '
        'summary and must not be interpreted as population certainty. Accuracy is the equal-episode '
        'mean of within-episode correct/audited-cell ratios, not accuracy pooled across episodes. '
        'Coverage is the final reported fraction. Invalid executions retain their denominator with F=0.\n\n'
        '**assessment-cases.** All 256 ordinal cases, including every failure, with paired F differences '
        'and both programs’ outcomes. The dashed line is the mean paired difference. Case order has no '
        'temporal meaning; environment seeds remain private. The cost caption reports measured evaluator '
        'and candidate episode-worker CPU, excluding controller overhead, supervising-assistant usage '
        'and saved-data analysis. Frozen refers to source code; ordinary map-memory updates remain active '
        'within each episode, and this is not a frozen-learning intervention. Both figures '
        'assess the frozen program on the declared original task distribution. The control is a weak '
        'original seed, not a competent handwritten pathfinder. These data do not isolate adaptation, '
        'compare prediction on identical experience, or establish reliable evolutionary discovery.\n')
    outputs = [out / name for name in ['summary.json', 'paired-episodes.csv', 'captions.md']]
    outputs += [figures / f'{stem}.{ext}' for stem in ['assessment-effects', 'assessment-cases'] for ext in ['svg', 'pdf', 'png']]
    sources = [Path(__file__), ROOT / 'proposal/development_report.py', ROOT / 'scripts/chromatic_fields.py', ROOT / 'scripts/visual_theme.py']
    rel = lambda p: str(p.relative_to(ROOT))
    write(out / 'rendering-manifest.json', {'style': style.PRESENTATION_VERSION,
          'reproduction': 'OPENBLAS_NUM_THREADS=1 .venv/bin/python -m proposal.assessment_report',
          'scientific_execution': 'No candidate/model execution or private-pool access; saved public episodes only.',
          'input_sha256': {rel(p): sha256(p) for p in inputs}, 'source_sha256': {rel(p): sha256(p) for p in sources},
          'output_sha256': {rel(p): sha256(p) for p in outputs}})
    for arm in ARMS:
        a = summary['arms'][arm]
        print(f'{arm}: F={a["metrics"]["combined_score"]["mean"]:.6f}; outcomes={a["outcomes"]}')
    print('Paired F:', summary['contrast']['metrics']['combined_score'])
    print('Episodes/transitions:', 2 * N, summary['execution']['transitions'])


if __name__ == '__main__':
    main()
