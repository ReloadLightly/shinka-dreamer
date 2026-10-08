"""Small development-only objective diagnostic, completed before v3 mutation."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.evaluation_v3 import run_episode, aggregate, evaluation_identity
from dreamer.provenance import record, sha256

CONDITIONS = {
    'original_memory': ('controls/v1/memory.py', 'memory'),
    'evolution_seed': ('controls/v1/predictive.py', 'predictive'),
    'v2_gen14': ('artifacts/campaign-v2/completed-50/selected.py', 'predictive'),
    'v2_gen14_frozen': ('artifacts/campaign-v2/completed-50/selected.py', 'frozen'),
    'v2_gen14_fixed_risk': ('artifacts/campaign-v2/completed-50/selected.py', 'no_planning'),
}


def one(job):
    name, index, seed, regime = job
    path, variant = CONDITIONS[name]
    row = run_episode(ROOT / path, seed, variant=variant, regime=regime, replay=True)
    trace = row.pop('trace')
    row['actions_sha256'] = hashlib.sha256(json.dumps([f['action'] for f in trace]).encode()).hexdigest()
    row['condition'], row['case'] = name, index
    row.pop('seed')
    return row


def main():
    out = ROOT / 'artifacts/campaign-v3'
    seeds = json.loads((ROOT / 'results/private/v3-development-seeds.json').read_text())[:8]
    plan = {'split': 'first 8 development cases; reused by search', 'case_count': 8,
            'regimes': ['uniform', 'stationary', 'switch'], 'conditions': CONDITIONS,
            'purpose': 'score scale/ranking/sparsity/decision relevance; no weight sweep',
            'proposed_objective': 'absolute-task-v1',
            'programs': {n: sha256(ROOT / p) for n, (p, _) in CONDITIONS.items()}}
    record(out / 'objective-check-plan.json', plan)
    with ProcessPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(one, [(name, i, seed, regime) for name in CONDITIONS
                           for i, seed in enumerate(seeds) for regime in plan['regimes']]))
    groups = {name: [r for r in rows if r['condition'] == name] for name in CONDITIONS}
    metrics = {name: aggregate(group)['public'] for name, group in groups.items()}
    for name, group in groups.items():
        metrics[name]['absolute_task'] = sum(r['combined_score'] for r in group) / len(group)
        metrics[name]['historical_composite_diagnostic'] = sum(
            .6*r['task']+.4*r['model_score'] if not r['error'] else 0 for r in group)/len(group)
    task = [m['absolute_task'] for m in metrics.values()]
    model = [m['model_score'] for m in metrics.values()]
    spans = {'task': max(task)-min(task), 'model_score': max(model)-min(model),
             'historical_weighted_task': .6*(max(task)-min(task)),
             'historical_weighted_model': .4*(max(model)-min(model))}
    paired = {}
    for right in ('v2_gen14_frozen', 'v2_gen14_fixed_risk'):
        pairs = list(zip(groups['v2_gen14'], groups[right]))
        paired[right] = {'action_sequence_changed_cases': sum(a['actions_sha256'] != b['actions_sha256'] for a,b in pairs),
                        'escape_wins': sum(a['reason']=='escaped' and b['reason']!='escaped' for a,b in pairs),
                        'escape_losses': sum(a['reason']!='escaped' and b['reason']=='escaped' for a,b in pairs)}
    report = {'plan': plan, 'evaluation': evaluation_identity(), 'episodes': len(rows),
              'metrics': metrics, 'spans': spans, 'decision_diagnostic': paired,
              'task_ranking': sorted(metrics,key=lambda n: -metrics[n]['absolute_task']),
              'historical_composite_ranking': sorted(metrics,key=lambda n: -metrics[n]['historical_composite_diagnostic']),
              'decision': 'Freeze absolute task; diagnostic forecasts do not add selection pressure from mostly empty cells.',
              'limitations': 'Five existing programs/variants on 24 development condition-cases; not a power study or search-discovery evidence.'}
    record(out / 'objective-check.json', report)
    record(out / 'objective-check-episodes.json', rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
