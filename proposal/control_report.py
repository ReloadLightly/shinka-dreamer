"""Analyse four frozen original-task controls using public saved episodes only."""
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
DEFAULT = ROOT / 'artifacts/proposal/broader-study/controls-assessment'
ARMS = ('generation_2', 'memory_pathfinder', 'local_map', 'no_risk')
CONTROLS = ARMS[1:]
LABELS = {'generation_2': 'Generation 2', 'memory_pathfinder': 'Memory pathfinder',
          'local_map': 'Local map for planning', 'no_risk': 'No next-step risk'}
COLORS = (style.COBALT, style.SECONDARY, style.MAGENTA, style.ORANGE)
N, DRAWS, RNG_SEED = 256, 10000, 6109256
PRIMARY = {'memory_pathfinder': ('combined_score', .05),
           'local_map': ('escaped', 0.), 'no_risk': ('escaped', 0.)}
FAMILY_SIZE = len(PRIMARY)
FAMILY_LEVEL = 1 - .05 / FAMILY_SIZE


def validate(rows):
    """Verify original arithmetic and retain failures in the declared panel."""
    if len(rows) != N or [r['case'] for r in rows] != list(range(N)):
        raise ValueError('Expected exactly 256 unique case ordinals, 0–255')
    for row in rows:
        if row['reason'] not in OUTCOMES:
            raise ValueError(f'Unknown outcome: {row["case"]}')
        if (row['reason'] == 'invalid') != (row['error'] is not None):
            raise ValueError(f'Inconsistent invalid execution: {row["case"]}')
        if not 0 <= row['map_correct'] <= row['map_audited']:
            raise ValueError(f'Invalid map denominator: {row["case"]}')
        accuracy = row['map_correct'] / row['map_audited'] if row['map_audited'] else 0.
        raw = (.1 * row['steps'] + 20 * row['keys'] + 30 * row['door_open']
               + 100 * (row['reason'] == 'escaped') - 50 * (row['reason'] == 'caught'))
        task = min(1., max(0., (raw + 50) / 250))
        fitness = 0. if row['error'] is not None else .6 * task + .4 * accuracy
        for key, actual in [('model_accuracy', accuracy), ('task', task), ('combined_score', fitness)]:
            if not math.isclose(row[key], actual, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Incorrect saved {key}: {row["case"]}')


def paired_binary_p(first_only, second_only):
    """Exact two-sided conditional test; no discordant cases gives p=1."""
    n = first_only + second_only
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(first_only, second_only) + 1)) / 2 ** n)


def analyse(rows, resources):
    values = {arm: matrix(rows[arm]) for arm in ARMS}
    indices = np.random.default_rng(RNG_SEED).integers(0, N, (DRAWS, N))
    sampled = {arm: np.empty((DRAWS, len(METRICS))) for arm in ARMS}
    # Reuse one index matrix across arms and metrics without allocating a
    # draws-by-cases-by-metrics tensor for each arm.
    for arm in ARMS:
        for j in range(len(METRICS)):
            sampled[arm][:, j] = values[arm][:, j][indices].mean(axis=1)
    summary = {
        'scope': 'Fresh paired assessment of frozen generation 2, a manually adapted historical pathfinder, and two source-specific planning interventions.',
        'estimands': 'Equal episode means on 256 common cases. Current-map accuracy is the equal-weight mean of within-episode correct/audited-cell ratios, with zero for no reported cells. It is not pooled across episodes or a future-prediction metric.',
        'uncertainty': {'method': 'whole-paired-episode percentile bootstrap', 'draws': DRAWS,
                        'rng_seed': RNG_SEED, 'marginal_level': .95,
                        'primary_family_size': FAMILY_SIZE, 'primary_interval_level': FAMILY_LEVEL,
                        'primary_family_method': 'Bonferroni: two-sided 98.333333% bootstrap intervals for the three prespecified endpoints; nominal family coverage 95%.',
                        'shared_resampling': 'All arms and every metric use identical resampled case indices.',
                        'binary_arm_intervals': 'Separate-arm 95% Wilson intervals accompany all outcome proportions.',
                        'limitations': 'Bootstrap family coverage is approximate; secondary intervals are unadjusted. Degenerate empirical intervals do not establish population certainty.'},
        'arms': {}, 'contrasts': {},
        'claim_limits': ['The two interventions change specific planning inputs, not all memory or all learning.',
                         'The risk calculation assumes uniform attempted enemy moves; this is not learned dynamics.',
                         'Policies induce different trajectories, so their map accuracies are not matched-experience comparisons.',
                         'This assessment does not establish independent evolutionary replication or a causal advantage of native Shinka mechanisms.'],
    }
    for arm in ARMS:
        outcomes = {outcome: sum(row['reason'] == outcome for row in rows[arm]) for outcome in OUTCOMES}
        summary['arms'][arm] = {
            'episodes': N, 'transitions': sum(row['steps'] for row in rows[arm]),
            'outcomes': outcomes,
            'outcome_wilson_ci95': {outcome: wilson(count, N) for outcome, count in outcomes.items()},
            'metrics': {key: {'mean': float(values[arm][:, j].mean()),
                              'ci95': np.quantile(sampled[arm][:, j], [.025, .975]).tolist()}
                        for j, key in enumerate(METRICS)},
            'resource': resources[arm],
            'mean_episode_wall_seconds': sum(row['seconds'] for row in rows[arm]) / N,
        }
    for control in CONTROLS:
        delta = sampled['generation_2'] - sampled[control]
        metrics = {}
        for j, key in enumerate(METRICS):
            lo, hi = np.quantile(delta[:, j], [.025, .975])
            metrics[key] = {'difference': float((values['generation_2'][:, j] - values[control][:, j]).mean()),
                            'ci95': [float(lo), float(hi)],
                            'degenerate_ci': math.isclose(lo, hi, rel_tol=0., abs_tol=1e-14)}
        metric, meaningful = PRIMARY[control]
        tails = (.05 / (2 * FAMILY_SIZE), 1 - .05 / (2 * FAMILY_SIZE))
        family_ci = np.quantile(delta[:, METRICS.index(metric)], tails).tolist()
        discordance = {}
        for outcome in OUTCOMES:
            discordance[outcome] = {
                'generation_2_only': sum(a['reason'] == outcome and b['reason'] != outcome
                                         for a, b in zip(rows['generation_2'], rows[control], strict=True)),
                'control_only': sum(a['reason'] != outcome and b['reason'] == outcome
                                   for a, b in zip(rows['generation_2'], rows[control], strict=True)),
            }
        primary = {'metric': metric, 'meaningful_difference': meaningful,
                   'difference': metrics[metric]['difference'], 'ci_family': family_ci,
                   'interval_level': FAMILY_LEVEL,
                   'lower_family_ci_exceeds_zero': family_ci[0] > 0,
                   'lower_family_ci_exceeds_meaningful_difference': family_ci[0] > meaningful}
        if metric == 'escaped':
            p = paired_binary_p(**dict(first_only=discordance['escaped']['generation_2_only'],
                                      second_only=discordance['escaped']['control_only']))
            primary['exact_paired_binary_test'] = {
                'method': 'Two-sided exact binomial test conditional on discordant pairs, null probability 0.5.',
                'p_two_sided': p, 'p_bonferroni_three_endpoint_family': min(1., FAMILY_SIZE * p)}
        summary['contrasts'][control] = {
            'first': 'generation_2', 'second': control, 'metrics': metrics, 'primary': primary,
            'objective_contributions': {
                'task_difference_times_0_6': .6 * metrics['task']['difference'],
                'accuracy_difference_times_0_4': .4 * metrics['model_accuracy']['difference'],
                'invalid_execution_correction': metrics['combined_score']['difference']
                    - .6 * metrics['task']['difference'] - .4 * metrics['model_accuracy']['difference']},
            'outcome_discordance': discordance,
            'outcome_pairs_control_to_generation_2': {
                f'{a} -> {b}': sum(x['reason'] == a and y['reason'] == b
                                   for x, y in zip(rows[control], rows['generation_2'], strict=True))
                for a in OUTCOMES for b in OUTCOMES},
        }
    summary['execution'] = {'episodes': len(ARMS) * N,
                            'transitions': sum(row['steps'] for arm in ARMS for row in rows[arm]),
                            'experiment_model_calls': 0, 'resource_by_arm': resources}
    return summary, values


def effects_figure(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), gridspec_kw={'width_ratios': [1., 1.18]})
    fig.subplots_adjust(left=.18, right=.96, top=.77, bottom=.26, wspace=.70)
    fig.suptitle('What supports the evolved program’s performance?', x=.025, y=.98, ha='left')
    fig.text(.025, .90, '256 fresh paired cases · four frozen programs · three prespecified primary comparisons',
             color=style.SECONDARY, fontsize=10.5)
    for y, (arm, color) in enumerate(zip(ARMS, COLORS, strict=True)):
        result = summary['arms'][arm]
        lo, hi = result['outcome_wilson_ci95']['escaped']
        mean = result['metrics']['escaped']['mean']
        axes[0].plot([lo, hi], [y, y], color=color, lw=2)
        axes[0].scatter(mean, y, color=color, s=36, zorder=3)
        axes[0].text(.015, y + .30, f'{result["outcomes"]["escaped"]}/{N} escapes',
                     transform=axes[0].get_yaxis_transform(), fontsize=9, color=style.SECONDARY)
    axes[0].set_title('A  Escape fractions', loc='left')
    axes[0].set(yticks=range(4), yticklabels=[LABELS[a] for a in ARMS], ylim=(3.6, -.6),
                xlim=(-.025, 1.04), xticks=[0, .25, .5, .75, 1], xlabel='Fraction (separate-arm 95% Wilson interval)')
    intervals = []
    for y, control in enumerate(CONTROLS):
        primary = summary['contrasts'][control]['primary']
        marginal = summary['contrasts'][control]['metrics'][primary['metric']]['ci95']
        lo, hi = primary['ci_family']
        intervals += [lo, hi]
        color = COLORS[y + 1]
        axes[1].plot([lo, hi], [y, y], color=color, lw=1.3)
        axes[1].plot(marginal, [y, y], color=color, lw=4)
        axes[1].scatter(primary['difference'], y, color=color, s=38, zorder=3)
        axes[1].text(.02, y + .32,
                     f'{primary["difference"]:+.3f}  [{lo:+.3f}, {hi:+.3f}]',
                     transform=axes[1].get_yaxis_transform(), fontsize=9.5, color=style.SECONDARY)
    axes[1].set_title('B  Generation 2 − control', loc='left')
    labels = ['ΔF vs pathfinder', 'ΔEscape vs local map', 'ΔEscape vs no risk']
    span = max(.15, max(intervals) - min(0., min(intervals)))
    axes[1].set(yticks=range(3), yticklabels=labels, ylim=(2.65, -.6),
                xlim=(min(0., min(intervals)) - .10 * span, max(.05, max(intervals)) + .12 * span),
                xlabel='Difference in F or escape fraction')
    axes[1].axvline(0, color=style.SECONDARY, lw=.8)
    axes[1].plot([.05, .05], [-.16, .16], color=style.ORANGE, lw=2)
    for ax in axes:
        ax.grid(axis='x')
        ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=style.SECONDARY, lw=4, label='Marginal 95% interval'),
                        Line2D([], [], color=style.SECONDARY, lw=1.3, label='98⅓% interval; three-endpoint family')],
               loc='lower center', bbox_to_anchor=(.55, .12), ncol=2, fontsize=9)
    fig.text(.025, .085, 'Paired effects use 10,000 shared whole-case bootstrap draws. Orange tick: meaningful ΔF = 0.05 for the pathfinder comparison.',
             color=style.SECONDARY, fontsize=9)
    fig.text(.025, .04, 'Local map limits past map access during planning; no risk removes next-step probabilities. Neither is a general no-learning intervention.',
             color=style.SECONDARY, fontsize=9)
    style.save_figure(fig, out / 'control-effects')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    out = parser.parse_args().output_dir.resolve()
    inputs = [out / name for name in ['protocol.json', 'pool-manifest.json', 'execution.json']]
    inputs += [out / arm / name for arm in ARMS for name in ['episodes.json', 'metrics.json', 'resource.json', 'manifest.json']]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    protocol, pool = read(out / 'protocol.json'), read(out / 'pool-manifest.json')
    if (protocol['case_count'] != N or set(protocol['programs']) != set(ARMS)
            or protocol['evaluator'] != 'namazu-proposal-reconstruction-v1'
            or protocol['analysis']['bootstrap_replicates'] != DRAWS
            or protocol['analysis']['bootstrap_seed'] != RNG_SEED
            or pool['protocol_sha256'] != sha256(out / 'protocol.json') or pool['case_count'] != N):
        raise ValueError('Report constants, protocol or pool identities differ')
    manifests = {arm: read(out / arm / 'manifest.json') for arm in ARMS}
    for arm, manifest in manifests.items():
        if (manifest['evaluator'] != protocol['evaluator'] or manifest['episodes'] != N
                or manifest['pool_sha256'] != pool['pool_sha256']
                or manifest['source_sha256'] != protocol['source_sha256']
                or manifest['candidate_sha256'] != protocol['programs'][arm]['sha256']):
            raise ValueError(f'Candidate, source or pool identity differs: {arm}')
    rows = {arm: sorted(read(out / arm / 'episodes.json'), key=lambda row: row['case']) for arm in ARMS}
    for arm in ARMS:
        validate(rows[arm])
    summary, values = analyse(rows, {arm: read(out / arm / 'resource.json') for arm in ARMS})
    for arm in ARMS:
        saved, expected = read(out / arm / 'metrics.json'), summary['arms'][arm]
        if not math.isclose(saved['combined_score'], expected['metrics']['combined_score']['mean'], abs_tol=1e-12):
            raise ValueError(f'Aggregate fitness mismatch: {arm}')
        for key, metric in [('mean_task_score', 'task'), ('mean_model_accuracy', 'model_accuracy'),
                            ('mean_map_coverage', 'final_coverage'), ('avg_steps', 'steps'), ('avg_keys', 'keys')]:
            if not math.isclose(saved['public'][key], expected['metrics'][metric]['mean'], abs_tol=1e-12):
                raise ValueError(f'Aggregate metric mismatch: {arm}/{key}')
        if saved['public']['episodes'] != N or any(saved['public'][key] != count for key, count in expected['outcomes'].items()):
            raise ValueError(f'Aggregate outcome denominator mismatch: {arm}')
    execution = read(out / 'execution.json')
    if (execution['status'] != 'complete' or execution['episodes'] != len(ARMS) * N
            or execution['transitions'] != summary['execution']['transitions']):
        raise ValueError('Execution record disagrees with complete episode data')
    summary['execution']['controller'] = execution
    summary['identities'] = {'evaluator': protocol['evaluator'], 'pool_sha256': pool['pool_sha256'],
                            'source_sha256': protocol['source_sha256'],
                            'candidate_sha256': {arm: manifest['candidate_sha256'] for arm, manifest in manifests.items()}}
    summary['protocol_sha256'] = sha256(out / 'protocol.json')
    write(out / 'summary.json', summary)
    table = []
    for i in range(N):
        row = {'case_ordinal': i}
        for arm in ARMS:
            row.update({f'{arm}_outcome': rows[arm][i]['reason'], f'{arm}_wall_seconds': rows[arm][i]['seconds']})
            row.update({f'{arm}_{key}': values[arm][i, j] for j, key in enumerate(METRICS)})
        for control in CONTROLS:
            row.update({f'generation_2_minus_{control}_{key}': values['generation_2'][i, j] - values[control][i, j]
                        for j, key in enumerate(METRICS)})
        table.append(row)
    with (out / 'paired-episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    style.apply_theme()
    effects_figure(summary, out / 'figures')
    (out / 'captions.md').write_text(
        '**control-effects.** Four frozen programs on 256 fresh paired cases. Panel A shows escape proportions '
        'with separate-arm 95% Wilson intervals; these are not paired-effect intervals. Panel B shows three '
        'prespecified primary contrasts: generation 2 minus the memory pathfinder for original combined fitness F, '
        'and generation 2 minus each planning intervention for escape fraction. Thick intervals are marginal 95%; '
        'thin intervals are Bonferroni 98⅓% intervals for nominal 95% coverage of the three-endpoint family. '
        'Both use 10,000 whole-paired-case bootstrap draws (seed 6109256), shared across all arms and metrics. '
        'The orange tick marks meaningful ΔF=0.05 for the pathfinder comparison only. Failed executions remain '
        'in all denominators with F=0; accuracy is the equal-episode mean of each episode’s correct/audited ratio. '
        'Secondary intervals in the saved summary are unadjusted. Degenerate empirical intervals are flagged '
        'and must not be interpreted as population certainty. Local map removes access to past terrain/objective '
        'map entries during planning while retaining other specified state. No risk removes probabilistic '
        'next-enemy-move evaluation while retaining occupied-cell avoidance. Neither intervention disables all '
        'memory or isolates learned dynamics. These episodes do not replicate the evolutionary search.\n')
    outputs = [out / name for name in ['summary.json', 'paired-episodes.csv', 'captions.md']]
    outputs += [out / 'figures' / f'control-effects.{ext}' for ext in ['svg', 'pdf', 'png']]
    sources = [Path(__file__), ROOT / 'proposal/development_report.py', ROOT / 'scripts/chromatic_fields.py', ROOT / 'scripts/visual_theme.py']
    relative = lambda path: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    write(out / 'rendering-manifest.json', {
        'style': style.PRESENTATION_VERSION,
        'reproduction': 'OPENBLAS_NUM_THREADS=1 .venv/bin/python -m proposal.control_report',
        'scientific_execution': 'No candidates, experiment models or private pools accessed; public saved-data analysis only.',
        'input_sha256': {relative(path): sha256(path) for path in inputs},
        'source_sha256': {relative(path): sha256(path) for path in sources},
        'output_sha256': {relative(path): sha256(path) for path in outputs}})
    for arm in ARMS:
        print(arm, summary['arms'][arm]['outcomes'])
    for control in CONTROLS:
        print(control, summary['contrasts'][control]['primary'])


if __name__ == '__main__':
    main()
