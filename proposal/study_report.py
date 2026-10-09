"""Report a complete frozen search assessment from saved public episodes only."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scripts import chromatic_fields as style

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from proposal.development_report import METRICS, OUTCOMES, matrix, read, sha256, wilson, write
from proposal.study_panels import selection_means, verify_hashes
from proposal.study_statistics import crossed_intervals

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/proposal/broader-study/search-assessment'
GROUPS = ('primary', 'common_prefix_secondary')
N, DRAWS, RNG_SEED = 512, 10000, 7109512


def paired_values(winners, values, group):
    """Preserve repetition identity even when multiple winners alias one source."""
    if len(winners) != 6 or group not in GROUPS:
        raise ValueError('Expected six declared run winners and a frozen comparison group')
    pairs, differences = [], []
    for repetition in (1, 2, 3):
        members = [winner for winner in winners if winner['replicate'] == repetition]
        if len(members) != 2 or {winner['arm'] for winner in members} != {'full', 'rewrite'}:
            raise ValueError('Each repetition must contain exactly full and rewrite winners')
        by_arm = {winner['arm']: winner for winner in members}
        keys = {arm: winner[group]['panel_program'] for arm, winner in by_arm.items()}
        if any(key not in values for key in keys.values()):
            raise ValueError('Frozen winner is absent from assessed programs')
        left, right = values[keys['full']], values[keys['rewrite']]
        if left.shape != right.shape:
            raise ValueError('Paired winners have different episode/metric shapes')
        differences.append(left - right)
        pairs.append({'replicate': repetition,
                      'full_run': by_arm['full']['run_id'], 'rewrite_run': by_arm['rewrite']['run_id'],
                      'full_program': keys['full'], 'rewrite_program': keys['rewrite'],
                      'full_mean': left.mean(axis=0).tolist(), 'rewrite_mean': right.mean(axis=0).tolist(),
                      'full_calls': by_arm['full']['actual_all_role_calls'],
                      'rewrite_calls': by_arm['rewrite']['actual_all_role_calls'],
                      'common_call_prefix': by_arm['full']['common_call_prefix']})
    return np.stack(differences), pairs


def analyse(rows, freeze, resources):
    values = {key: matrix(episodes) for key, episodes in rows.items()}
    differences, pairs = {}, {}
    for group in GROUPS:
        differences[group], pairs[group] = paired_values(freeze['winners'], values, group)
    # One invocation shares both resampling axes across all metrics and both
    # groups. Aliased sources remain one evaluated program, not extra cases.
    combined = np.concatenate([differences[group] for group in GROUPS], axis=2)
    estimates = crossed_intervals(combined, draws=DRAWS, seed=RNG_SEED)
    summary = {'scope': 'Frozen exploratory comparison of full native Shinka and independent original-seed rewrites; no assessment-based selection.',
               'cases_per_program': N, 'independent_repetition_pairs': 3,
               'metric_order': list(METRICS), 'groups': {}, 'programs': {}, 'controls': {},
               'uncertainty': {key: value for key, value in estimates.items() if key not in
                               ('mean_difference', 'crossed_interval', 'individual_mean_differences', 'individual_paired_case_intervals')},
               'estimands': 'Full minus rewrite, equal-weight means over three paired repetitions and 512 shared cases. A averages within-episode correct/audited ratios; invalid episodes retain F=0 and their denominator.',
               'claim_limits': ['Three paired repetitions give exploratory uncertainty about search strategies.',
                                'Source aliases are evaluated once and do not create independent episode samples.',
                                'Common-prefix comparisons are conditional on realized budgets and are secondary.',
                                'Whole-strategy differences do not isolate individual Shinka mechanisms.',
                                'Current-map reconstruction on different trajectories is not learned predictive accuracy.']}
    for group_index, group in enumerate(GROUPS):
        start = group_index * len(METRICS)
        metrics = {}
        for j, key in enumerate(METRICS):
            i = start + j
            interval = estimates['crossed_interval'][i]
            metrics[key] = {'difference': estimates['mean_difference'][i], 'crossed_ci95': interval,
                            'degenerate_ci': math.isclose(interval[0], interval[1], abs_tol=1e-14, rel_tol=0.),
                            'individual_run_differences': [row[i] for row in estimates['individual_mean_differences']],
                            'individual_paired_case_ci95': [row[i] for row in estimates['individual_paired_case_intervals']]}
        summary['groups'][group] = {'pairs': pairs[group], 'metrics': metrics,
            'objective_contributions': {
                'task_difference_times_0_6': .6 * metrics['task']['difference'],
                'accuracy_difference_times_0_4': .4 * metrics['model_accuracy']['difference'],
                'invalid_execution_correction': metrics['combined_score']['difference']
                    - .6 * metrics['task']['difference'] - .4 * metrics['model_accuracy']['difference']}}
        if group == 'primary':
            metric = metrics['combined_score']
            summary['groups'][group]['primary_endpoint'] = {
                'metric': 'combined_score', 'meaningful_difference': .05,
                'lower_ci_exceeds_zero': metric['crossed_ci95'][0] > 0,
                'lower_ci_exceeds_meaningful_difference': metric['crossed_ci95'][0] > .05}
    for key, episodes in rows.items():
        outcomes = {outcome: sum(row['reason'] == outcome for row in episodes) for outcome in OUTCOMES}
        summary['programs'][key] = {
            'source_sha256': freeze['programs'][key]['sha256'], 'episodes': N,
            'means': {name: float(values[key][:, j].mean()) for j, name in enumerate(METRICS)},
            'outcomes': outcomes, 'outcome_wilson_ci95': {name: wilson(count, N) for name, count in outcomes.items()},
            'resource': resources[key]}
    for label, spec in freeze['controls'].items():
        summary['controls'][label] = {'panel_program': spec['panel_program'], **summary['programs'][spec['panel_program']]}
    summary['winners'] = freeze['winners']
    summary['execution'] = {'distinct_programs': len(rows), 'episodes': len(rows) * N,
                            'transitions': sum(row['steps'] for episodes in rows.values() for row in episodes),
                            'assessment_model_calls': 0, 'search_calls_by_run': {
                                winner['run_id']: winner['actual_all_role_calls'] for winner in freeze['winners']}}
    return summary, values


def comparison_figure(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5), gridspec_kw={'width_ratios': [1., 1.15]})
    fig.subplots_adjust(left=.12, right=.96, top=.77, bottom=.25, wspace=.55)
    fig.suptitle('Independent search attempts on a shared assessment', x=.025, y=.98, ha='left')
    fig.text(.025, .89, '512 unseen paired cases · three independent repetition pairs · frozen full-budget winners', color=style.SECONDARY)
    primary = summary['groups']['primary']
    for index, pair in enumerate(primary['pairs']):
        a, b = pair['full_mean'][0], pair['rewrite_mean'][0]
        axes[0].plot([b, a], [index, index], color=style.RULE, lw=2)
        axes[0].scatter(a, index, color=style.COBALT, marker='o', s=45, zorder=3)
        axes[0].scatter(b, index, color=style.MAGENTA, marker='s', s=40, zorder=3)
        axes[0].text(.01, index + .28, f'Calls: full {pair["full_calls"]}; rewrite {pair["rewrite_calls"]}',
                     transform=axes[0].get_yaxis_transform(), fontsize=9, color=style.SECONDARY)
    axes[0].set(title='A  Mean original fitness F', yticks=range(3), yticklabels=['Pair 1', 'Pair 2', 'Pair 3'],
                xlabel='F = 0.6 task + 0.4 current-map accuracy', xlim=(-.025, 1.025), ylim=(2.7, -.6))
    metric = primary['metrics']['combined_score']
    effects = list(zip(metric['individual_run_differences'], metric['individual_paired_case_ci95']))
    effects += [(metric['difference'], metric['crossed_ci95'])]
    limits = [0., .05]
    for index, (mean, interval) in enumerate(effects):
        color = style.COBALT if index == 3 else style.SECONDARY
        axes[1].plot(interval, [index, index], color=color, lw=2)
        axes[1].scatter(mean, index, marker='D' if index == 3 else 'o', color=color, s=48, zorder=3)
        axes[1].text(.015, index + .27, f'{mean:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]',
                     transform=axes[1].get_yaxis_transform(), fontsize=9, color=style.SECONDARY)
        limits.extend(interval)
    span = max(.15, max(limits) - min(limits))
    axes[1].set(title='B  Full − rewrite; 95% intervals', yticks=range(4),
                yticklabels=['Pair 1', 'Pair 2', 'Pair 3', 'Across pairs'], xlabel='Paired difference in F',
                xlim=(min(limits) - .12 * span, max(limits) + .12 * span), ylim=(3.65, -.6))
    axes[1].axvline(0, color=style.SECONDARY, lw=.8)
    axes[1].plot([.05, .05], [2.85, 3.15], color=style.ORANGE, lw=2)
    for ax in axes:
        ax.grid(axis='x'); ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color=style.COBALT, marker='o', ls='', label='Full native Shinka'),
                        Line2D([], [], color=style.MAGENTA, marker='s', ls='', label='Independent rewrites')],
               loc='lower center', bbox_to_anchor=(.52, .115), ncol=2)
    fig.text(.025, .075, 'Individual intervals resample paired cases. Across-pair interval independently resamples paired runs and shared cases.',
             color=style.SECONDARY, fontsize=9)
    fig.text(.025, .035, '10,000 shared draws; only three run pairs. Orange tick: meaningful ΔF=0.05. Equal call ceilings can yield different realized costs.',
             color=style.SECONDARY, fontsize=9)
    style.save_figure(fig, out / 'search-comparison')


def outcomes_figure(summary, out):
    entries = [(winner['run_id'], winner['primary']['panel_program'])
               for winner in sorted(summary['winners'], key=lambda row: (row['arm'] != 'full', row['replicate']))]
    entries += [('Original seed', summary['controls']['original_seed']['panel_program']),
                ('Memory pathfinder', summary['controls']['memory_pathfinder']['panel_program'])]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.3), gridspec_kw={'width_ratios': [1.15, 1.]})
    fig.subplots_adjust(left=.19, right=.97, top=.80, bottom=.24, wspace=.24)
    fig.suptitle('Outcomes, task reward and reported maps', x=.025, y=.98, ha='left')
    fig.text(.025, .90, 'Full-budget winners and fixed controls · each row uses 512 shared cases · aliased sources appear under their run identities',
             color=style.SECONDARY, fontsize=10)
    colors = (style.COBALT, style.MAGENTA, style.ORANGE, style.SECONDARY)
    diagnostics = [('task', 'Task S', style.COBALT, 'o'), ('model_accuracy', 'Map accuracy A', style.MAGENTA, 's'),
                   ('final_coverage', 'Map coverage', style.ORANGE, '^')]
    for y, (label, key) in enumerate(entries):
        program = summary['programs'][key]
        left = 0.
        for outcome, color in zip(OUTCOMES, colors, strict=True):
            value = program['outcomes'][outcome] / N
            axes[0].barh(y, value, left=left, color=color, height=.62)
            left += value
        for j, (metric, _, color, marker) in enumerate(diagnostics):
            axes[1].scatter(program['means'][metric], y + (j - 1) * .17, color=color, marker=marker, s=30)
    axes[0].set(title='A  Every outcome remains counted', yticks=range(len(entries)), yticklabels=[row[0] for row in entries],
                xlabel='Episode fraction', xlim=(0, 1), ylim=(len(entries) - .35, -.65))
    axes[1].set(title='B  Separate score and coverage means', yticks=range(len(entries)), yticklabels=[],
                xlabel='Score or reported-cell fraction', xlim=(-.025, 1.025), ylim=(len(entries) - .35, -.65))
    axes[1].grid(axis='x'); axes[1].set_axisbelow(True)
    axes[0].legend(handles=[Line2D([], [], color=color, lw=6, label=label) for label, color in
                           zip(('Escape', 'Capture', 'Timeout', 'Invalid'), colors, strict=True)],
                   loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=9)
    axes[1].legend(handles=[Line2D([], [], color=color, marker=marker, ls='', label=label)
                           for _, label, color, marker in diagnostics], loc='upper center',
                   bbox_to_anchor=(.5, -.13), ncol=2, fontsize=9)
    fig.text(.025, .055, 'A is reported current-map accuracy on each policy’s trajectory; greater accuracy with less coverage need not mean a better world model.',
             color=style.SECONDARY, fontsize=9)
    fig.text(.025, .02, 'Separate-arm Wilson intervals and all paired effects are in summary.json. Repeated source identities do not add independent evaluations.',
             color=style.SECONDARY, fontsize=9)
    style.save_figure(fig, out / 'search-outcomes-models')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    out = parser.parse_args().output_dir.resolve()
    protocol_path = out / 'protocol.json'
    protocol = read(protocol_path)
    freeze_path = ROOT / protocol['assessment_freeze']['path']
    if sha256(freeze_path) != protocol['assessment_freeze']['sha256']:
        raise ValueError('Winner freeze changed')
    freeze = read(freeze_path)
    if (protocol['phase'] != 'assessment' or protocol['case_count'] != N
            or protocol['evaluator'] != 'namazu-proposal-reconstruction-v1'
            or any(protocol[key] != freeze[key] for key in ('programs', 'analysis', 'winners', 'controls', 'source_sha256'))
            or freeze['analysis']['bootstrap_replicates'] != DRAWS or freeze['analysis']['bootstrap_seed'] != RNG_SEED
            or freeze['analysis']['metrics'] != list(METRICS) or freeze['analysis']['repetitions'] != 3
            or freeze['analysis']['cases'] != N or freeze['analysis']['level'] != .95):
        raise ValueError('Assessment report and frozen design differ')
    verify_hashes(freeze['input_sha256'])
    verify_hashes(protocol['source_sha256'])
    pool, execution = read(out / 'pool-manifest.json'), read(out / 'execution.json')
    if (pool['protocol_sha256'] != sha256(protocol_path) or pool['case_count'] != N
            or execution['status'] != 'complete' or execution['episodes'] != N * len(protocol['programs'])):
        raise ValueError('Assessment panel is not complete or pool binding differs')
    inputs = [protocol_path, freeze_path, out / 'pool-manifest.json', out / 'execution.json']
    rows, resources = {}, {}
    for key, spec in protocol['programs'].items():
        folder = out / key
        rows[key] = read(folder / 'episodes.json')
        selected = selection_means(rows[key], count=N)
        manifest, metrics = read(folder / 'manifest.json'), read(folder / 'metrics.json')
        if (manifest['evaluator'] != protocol['evaluator'] or manifest['episodes'] != N
                or manifest['candidate_sha256'] != spec['sha256'] or manifest['pool_sha256'] != pool['pool_sha256']
                or manifest['source_sha256'] != protocol['source_sha256']
                or manifest.get('protocol_sha256') != sha256(protocol_path)):
            raise ValueError('Frozen candidate/evaluator/case identity differs')
        if (not math.isclose(metrics['combined_score'], selected['combined_score'], abs_tol=1e-12)
                or not math.isclose(metrics['public']['mean_task_score'], selected['task'], abs_tol=1e-12)
                or metrics['public']['episodes'] != N):
            raise ValueError('Saved aggregate differs from episode data')
        for public_key, row_key in [('mean_model_accuracy', 'model_accuracy'), ('mean_map_coverage', 'final_coverage'),
                                    ('avg_keys', 'keys'), ('avg_steps', 'steps')]:
            if not math.isclose(metrics['public'][public_key], sum(row[row_key] for row in rows[key]) / N, abs_tol=1e-12):
                raise ValueError('Saved secondary aggregate differs from episode data')
        for outcome in OUTCOMES:
            if metrics['public'][outcome] != sum(row['reason'] == outcome for row in rows[key]):
                raise ValueError('Saved outcome count differs')
        resources[key] = read(folder / 'resource.json')
        inputs += [folder / name for name in ('episodes.json', 'metrics.json', 'manifest.json', 'resource.json')]
    summary, values = analyse(rows, freeze, resources)
    if summary['execution']['transitions'] != execution['transitions']:
        raise ValueError('Execution transitions differ from saved episodes')
    summary['execution']['controller'] = execution
    summary['identities'] = {'protocol_sha256': sha256(protocol_path), 'assessment_freeze_sha256': sha256(freeze_path),
                             'pool_sha256': pool['pool_sha256'], 'source_sha256': protocol['source_sha256']}
    write(out / 'summary.json', summary)
    table = []
    for group in GROUPS:
        for pair in summary['groups'][group]['pairs']:
            for case in range(N):
                row = {'group': group, 'replicate': pair['replicate'], 'case_ordinal': case,
                       'full_run': pair['full_run'], 'rewrite_run': pair['rewrite_run'],
                       'full_program': pair['full_program'], 'rewrite_program': pair['rewrite_program']}
                for arm in ('full', 'rewrite'):
                    key = pair[arm + '_program']
                    row[arm + '_outcome'] = rows[key][case]['reason']
                    row.update({arm + '_' + metric: values[key][case, j] for j, metric in enumerate(METRICS)})
                row.update({'difference_' + metric: row['full_' + metric] - row['rewrite_' + metric] for metric in METRICS})
                table.append(row)
    with (out / 'paired-episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(table)
    style.apply_theme()
    comparison_figure(summary, out / 'figures')
    outcomes_figure(summary, out / 'figures')
    (out / 'captions.md').write_text(
        '**search-comparison.** Frozen full-budget winners from three independent paired search repetitions '
        'assessed on the same 512 fresh cases. Left: mean original F by run and actual all-role provider-call counts. '
        'Right: full-native minus independent-rewrite differences. Individual intervals resample common case IDs; '
        'the across-pair 95% interval independently resamples paired repetitions and shared cases. All 11 metrics '
        'and both primary/common-prefix groups share 10,000 draws (seed 7109512), generated by study_statistics.py. '
        'The orange tick marks meaningful ΔF=.05. Only F in the full-budget group is primary; all other intervals '
        'are marginal exploratory summaries without multiplicity correction. Three repetitions limit precision; '
        'a degenerate bootstrap interval is not population certainty. Realized common-prefix contrasts are secondary. '
        'Full-budget denotes all candidates from the realized attempt, including attempts that stopped below a ceiling.\n\n'
        '**search-outcomes-models.** Outcomes, task S, current-map accuracy A and final coverage for all six '
        'full-budget winner identities and the original seed/handcrafted comparator. Identical program hashes '
        'are executed once and reused, without creating independent cases. All invalid outcomes stay in denominators '
        'with F=0. A is the equal-episode mean of within-episode correct/audited ratios. Different policies induce '
        'different experience, so reported-map accuracy does not measure predictive learning. Controls and all '
        'program outcomes have separate-arm 95% Wilson intervals in summary.json. No assessment outcomes select '
        'programs or change the frozen sample size.\n')
    outputs = [out / name for name in ('summary.json', 'paired-episodes.csv', 'captions.md')]
    outputs += [out / 'figures' / f'{stem}.{ext}' for stem in ('search-comparison', 'search-outcomes-models') for ext in ('png', 'svg', 'pdf')]
    sources = [Path(__file__), ROOT / 'proposal/study_statistics.py', ROOT / 'proposal/study_panels.py',
               ROOT / 'proposal/development_report.py', ROOT / 'scripts/chromatic_fields.py', ROOT / 'scripts/visual_theme.py']
    rel = lambda path: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    write(out / 'rendering-manifest.json', {'style': style.PRESENTATION_VERSION,
          'scientific_execution': 'None: saved public episode analysis only; no candidate/model execution or private pool access.',
          'reproduction': 'OPENBLAS_NUM_THREADS=1 .venv/bin/python -m proposal.study_report',
          'input_sha256': {rel(path): sha256(path) for path in inputs},
          'source_sha256': {rel(path): sha256(path) for path in sources},
          'output_sha256': {rel(path): sha256(path) for path in outputs}})
    print(summary['groups']['primary']['metrics']['combined_score'])


if __name__ == '__main__':
    main()
