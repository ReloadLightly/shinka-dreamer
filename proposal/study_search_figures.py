"""Render saved independent-search checkpoints; never run candidates or models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from scripts import chromatic_fields as style


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(directory):
    directory = Path(directory)
    paths = {name: directory / (name + '.json') for name in ('summary', 'slots', 'model-calls', 'config')}
    summary, slots, calls, config = (read(paths[name]) for name in paths)
    if not summary['execution_status'].startswith(('worker_returned', 'interrupted_or_failed')):
        raise ValueError('Render the checkpoint only after its execution has terminated')
    style.apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.3), gridspec_kw={'width_ratios': [1.35, 1]})
    fig.subplots_adjust(left=.075, right=.97, top=.78, bottom=.30, wspace=.42)
    fig.suptitle(f'{summary["run_id"]} · independent search checkpoint', x=.03, y=.98, ha='left')
    fig.text(.03, .90, 'Five fixed development cases · every model role and failed request counts toward the same allowance',
             fontsize=10.5, color=style.SECONDARY)
    accepted = sorted((s for s in slots if s['status'] == 'valid'
                       and s['acceptance_budget'].get('all_role_calls_at_acceptance') is not None),
                      key=lambda s: (s['acceptance_budget']['all_role_calls_at_acceptance'], s['generation']))
    xs, best_values, incumbent = [], [], None
    for slot in accepted:
        x, y = slot['acceptance_budget']['all_role_calls_at_acceptance'], slot['fitness']
        incumbent = y if incumbent is None else max(incumbent, y)
        xs.append(x)
        best_values.append(incumbent)
        axes[0].scatter(x, y, marker='D' if slot['generation'] == 0 else 'o',
                        color=style.SECONDARY if slot['generation'] == 0 else style.COBALT, s=30, zorder=3)
        axes[0].annotate(str(slot['generation']), (x, y), xytext=(0, -14 if slot['generation'] % 2 else 7),
                         textcoords='offset points', ha='center', fontsize=8)
    if accepted:
        axes[0].step(xs + [len(calls)], best_values + [incumbent], where='post', color=style.COBALT, lw=1.4)
    else:
        axes[0].text(.5, .5, 'No valid evaluated source', transform=axes[0].transAxes, ha='center')
    for slot in slots:
        prefix = slot['acceptance_budget'].get('all_role_calls_at_acceptance')
        if slot['status'] == 'failed' and prefix is not None:
            axes[0].scatter(prefix, slot.get('fitness') or 0., marker='x', color=style.MAGENTA, s=40, zorder=3)
    cap = config['limits']['calls']
    axes[0].axvline(len(calls), color=style.SECONDARY, ls=':', lw=.9)
    axes[0].set(xlim=(-.5, cap+.5), ylim=(-.04, 1.04), xlabel='All-role calls at source acceptance', ylabel='Mean search fitness F')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axes[0].set_title('A  Evaluated sources and incumbent', loc='left')
    axes[0].grid(axis='y')
    axes[0].legend(handles=[Line2D([], [], color=style.COBALT, marker='o', label='Valid source / best so far'),
                           Line2D([], [], color=style.MAGENTA, marker='x', ls='', label='Failed evaluation')],
                   loc='lower right', fontsize=8)
    role_order = ['mutation', 'repair', 'novelty', 'summary', 'global_insight', 'recommendation', 'prompt_mutation', 'readiness']
    roles = [r for r in role_order if any(c['role'] == r for c in calls)]
    roles += sorted({c['role'] for c in calls} - set(role_order))
    colors = {'gpt-6.1-sol': style.COBALT, 'gpt-6-astra': style.MAGENTA}
    counts = Counter((c['role'], c.get('requested_model', 'unknown')) for c in calls)
    models = sorted({c.get('requested_model', 'unknown') for c in calls})
    left = [0] * len(roles)
    for model in models:
        values = [counts[(role, model)] for role in roles]
        axes[1].barh(range(len(roles)), values, left=left, color=colors.get(model, style.ORANGE), height=.6)
        left = [a+b for a, b in zip(left, values)]
    for index, total in enumerate(left):
        axes[1].text(total+.1, index, str(total), va='center', fontsize=9)
    if not roles:
        axes[1].text(.5, .5, 'No model dispatches', transform=axes[1].transAxes, ha='center')
    axes[1].set(yticks=range(len(roles)), yticklabels=[r.replace('_', ' ') for r in roles],
                xlim=(0, max(left, default=1)+1.3), xlabel='Dispatched requests')
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    axes[1].set_title('B  Actual request allocation', loc='left')
    axes[1].grid(axis='x')
    axes[1].set_axisbelow(True)
    if models:
        axes[1].legend(handles=[Patch(color=colors.get(m, style.ORANGE), label=m) for m in models],
                       loc='upper left', bbox_to_anchor=(0, -.15), fontsize=8)
    failed = sum(c.get('has_error') or c.get('returncode') not in (0, None) for c in calls)
    seconds = sum(c.get('elapsed_seconds', 0) for c in calls)
    fig.text(.03, .19, f'{len(calls)}/{cap} requests · {failed} failed requests · {seconds/60:.1f} provider-minutes · '
             f'{summary["valid_descendants"]} valid descendants · {summary["saved_episodes"]} training episodes',
             fontsize=10, color=style.SECONDARY)
    fig.text(.03, .12, 'Numbers at source points are generation IDs. Dotted line marks actual request consumption; unused allowance stays visible.',
             fontsize=9.5, color=style.SECONDARY)
    fig.text(.03, .055, 'Search fitness is selection-biased. No fresh assessment, repeatability claim or causal attribution follows from this curve.',
             fontsize=9.5, color=style.SECONDARY)
    out = directory / 'figures'
    out.mkdir(exist_ok=True)
    style.save_figure(fig, out / 'search-checkpoint')
    files = sorted(out.glob('search-checkpoint.*'))
    (out / 'rendering-manifest.json').write_text(json.dumps({
        'inputs': {str(p.relative_to(directory)): sha(p) for p in paths.values()},
        'renderer_sha256': sha(__file__), 'theme_sha256': sha(Path(style.__file__)),
        'outputs': {p.name: sha(p) for p in files}, 'new_episodes': 0, 'model_calls': 0,
    }, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    render(parser.parse_args().input)


if __name__ == '__main__':
    main()
