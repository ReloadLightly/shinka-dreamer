"""Freeze study nominations from six stopped searches; never execute experiments."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, closing
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / 'artifacts/proposal/broader-study/study-plan.json'
DEFAULT_OUTPUT = ROOT / 'artifacts/proposal/broader-study/search-selection'
OUTCOMES = ('escaped', 'caught', 'timeout', 'invalid')
EVALUATOR_SOURCES = ('proposal/initial.py', 'proposal/worker.py', 'proposal/evaluation.py',
                     'proposal/evaluate.py', 'proposal/evaluate_full.py', 'dreamer/world.py',
                     'dreamer/evaluation.py', 'dreamer/isolation.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def nominate(candidates, seed, call_prefix=None):
    """One source per hash; metric ties choose its earliest eligible generation."""
    eligible = [candidate for candidate in candidates if candidate['generation'] > 0
                and (call_prefix is None or candidate['all_role_calls_at_acceptance'] <= call_prefix)]
    unique = {}
    for candidate in sorted(eligible, key=lambda item: item['generation']):
        unique.setdefault(candidate['source_sha256'], candidate)
    if not unique:
        return {'seed_fallback': True, 'eligible_distinct_descendants': 0,
                'nominations': [dict(seed, nomination_reasons=['no eligible valid descendant'])]}
    chosen = {}
    for metric in ('combined_score', 'task', 'escaped'):
        best = min(unique.values(), key=lambda item: (-item['training_metrics'][metric], item['generation']))
        nomination = chosen.setdefault(best['source_sha256'], dict(best, nomination_reasons=[]))
        nomination['nomination_reasons'].append('highest training ' + metric)
    return {'seed_fallback': False, 'eligible_distinct_descendants': len(unique),
            'nominations': list(chosen.values())}


def training_metrics(rows):
    """Validate five exposed cases without importing the evaluator or candidates."""
    if len(rows) != 5 or [row['case'] for row in rows] != list(range(5)):
        raise ValueError('Expected five training cases 0–4')
    for row in rows:
        if row['error'] is not None or row['reason'] not in OUTCOMES or row['reason'] == 'invalid':
            raise ValueError('Correct nominee has invalid execution evidence')
        if not 0 <= row['map_correct'] <= row['map_audited']:
            raise ValueError('Invalid map denominator')
        accuracy = row['map_correct'] / row['map_audited'] if row['map_audited'] else 0.
        task = min(1., max(0., (.1 * row['steps'] + 20 * row['keys'] + 30 * row['door_open']
                  + 100 * (row['reason'] == 'escaped') - 50 * (row['reason'] == 'caught') + 50) / 250))
        for key, value in [('task', task), ('model_accuracy', accuracy), ('combined_score', .6 * task + .4 * accuracy)]:
            if not math.isclose(row[key], value, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f'Inconsistent training {key}')
    result = {key: sum(row[key] for row in rows) / 5 for key in
              ('combined_score', 'task', 'model_accuracy', 'final_coverage')}
    result.update({outcome: sum(row['reason'] == outcome for row in rows) for outcome in OUTCOMES})
    result['episodes'] = 5
    return result


def read_run(directory, spec, plan, inputs):
    """Read only terminal native evidence; incomplete proposals cannot qualify."""
    directory = Path(directory).resolve()

    def evidence(path):
        inputs[relative(path)] = sha(path)
        return read(path)

    config = evidence(directory / 'campaign-manifest.json')
    execution = evidence(directory / 'execution-resources.json')
    grant = evidence(directory / 'execution-grant.json')
    limits = plan['search_design']['per_run_limits']
    if (config['run_id'] != spec['run_id'] or config['arm'] != spec['arm']
            or config['native_sampling_seed'] != spec['native_seed']
            or config['evaluator'] != 'namazu-proposal-reconstruction-v1'
            or config['search_cases'] != [0, 1, 2, 3, 4]
            or config['limits']['calls'] != limits['all_role_provider_calls']
            or config['limits']['remote_elapsed_seconds'] != limits['provider_seconds']
            or config['wall_minutes'] != limits['wall_minutes']
            or config['evolution']['num_generations'] != limits['total_candidate_slots']):
        raise ValueError(f'Search treatment or budget differs from study plan: {spec["run_id"]}')
    if (not execution.get('ended_utc') or execution.get('status') not in ('worker_returned', 'interrupted_or_failed')
            or execution.get('returncode') is None
            or grant['manifest_sha256'] != sha(directory / 'campaign-manifest.json')
            or execution['manifest_sha256'] != grant['manifest_sha256']):
        raise ValueError(f'Search attempt is not terminal or bound to its configuration: {spec["run_id"]}')
    ledger_path = directory / 'call-budget-ledger.json'
    ledger = None
    calls = [evidence(path) for path in sorted((directory / 'model-calls').glob('*.json'))]
    if any(call.get('status') == 'started' for call in calls):
        raise ValueError('Unresolved dispatched request requires explicit accounting before nomination')
    if any(call.get('billing_violation') or call.get('model_mismatch') for call in calls):
        raise ValueError('Billing/model violations require scientific adjudication before nomination')
    elapsed = sum(call.get('elapsed_seconds', 0.) for call in calls)
    if calls or ledger_path.exists():
        ledger = evidence(ledger_path)
        if (ledger['limits'] != config['limits'] or ledger['totals']['calls'] != len(calls)
                or not math.isclose(ledger['totals']['remote_elapsed_seconds'], elapsed, abs_tol=1e-9)
                or ledger['totals']['roles'] != dict(Counter(call['role'] for call in calls))):
            raise ValueError('Dispatch ledger disagrees with actual request records')
    elif execution['status'] != 'interrupted_or_failed':
        raise ValueError('Successful search is missing its request ledger')
    if len(calls) > limits['all_role_provider_calls']:
        raise ValueError('Actual dispatches exceed declared call cap')
    if spec['arm'] == 'rewrite' and any(call['role'] != 'mutation' for call in calls):
        raise ValueError('Independent rewrite dispatched a forbidden auxiliary/repair role')
    source = ROOT / 'proposal/initial.py'
    seed_hash = config['source_sha256']['proposal/initial.py']
    if sha(source) != seed_hash:
        raise ValueError('Frozen original seed changed')
    inputs[relative(source)] = seed_hash
    seed = {'generation': 0, 'source': relative(source), 'source_sha256': seed_hash,
            'all_role_calls_at_acceptance': 0, 'training_metrics': None,
            'evaluation_status': 'not evaluated in this failed run'}
    database = directory / 'programs.sqlite'
    candidates, slots, canonical = [], [], {}
    if database.exists():
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            native = [dict(row) for row in connection.execute('SELECT * FROM programs ORDER BY generation, id')]
        inputs[relative(database)] = sha(database)
        wal = Path(str(database) + '-wal')
        if wal.exists():
            inputs[relative(wal)] = sha(wal)
        for row in native:
            metadata = json.loads(row.get('metadata') or '{}')
            if metadata.get('_is_island_copy'):
                continue
            generation = row['generation']
            if generation in canonical:
                raise ValueError('Multiple canonical native sources in one generation')
            canonical[generation] = row
            slots.append({'generation': generation, 'correct': bool(row['correct']),
                          'failure_stage': metadata.get('failure_stage'), 'failure_reason': metadata.get('failure_reason')})
            if not row['correct']:
                continue
            folder = directory / f'gen_{generation}'
            program = folder / 'main.py'
            digest = sha(program)
            if hashlib.sha256(row['code'].encode()).hexdigest() != digest:
                raise ValueError('Native code and saved executable differ')
            inputs[relative(program)] = digest
            output = folder / 'results'
            correct = evidence(output / 'correct.json')
            resource = evidence(output / 'resource.json')
            manifest = evidence(output / 'manifest.json')
            metrics = evidence(output / 'metrics.json')
            rows = evidence(output / 'episodes.json')
            if (not correct.get('correct') or not resource.get('completed')
                    or manifest['candidate_sha256'] != digest or manifest['episodes'] != 5
                    or manifest['evaluator'] != config['evaluator']
                    or manifest.get('seed_start') != 0
                    or any(config['source_sha256'].get(name) != value for name, value in manifest['source_sha256'].items())):
                raise ValueError('Candidate evaluator identities or completion evidence differ')
            measured = training_metrics(rows)
            for observed in (metrics['combined_score'], row['combined_score']):
                if not math.isclose(observed, measured['combined_score'], abs_tol=1e-12):
                    raise ValueError('Native/saved fitness differs from episode arithmetic')
            if (not math.isclose(metrics['public']['mean_task_score'], measured['task'], abs_tol=1e-12)
                    or metrics['public']['escaped'] != measured['escaped']):
                raise ValueError('Saved nomination metrics differ from episodes')
            acceptance = evidence(folder / 'acceptance-budget.json')
            prefix = acceptance['all_role_calls_at_acceptance']
            if type(prefix) is not int or not 0 <= prefix <= len(calls) or acceptance['generation'] != generation:
                raise ValueError('Invalid acceptance call prefix')
            if generation:
                accepted = evidence(folder / 'accepted-proposal.json')
                if (accepted['source_sha256'] != digest
                        or accepted['job']['meta_patch_data']['all_role_calls_at_acceptance'] != prefix
                        or metadata['all_role_calls_at_acceptance'] != prefix):
                    raise ValueError('Accepted and persisted proposal identities/prefix differ')
            elif digest != seed_hash or prefix != 0:
                raise ValueError('Original seed identity/prefix differs')
            candidate = {'generation': generation, 'source': relative(program), 'source_sha256': digest,
                         'native_program_id': row['id'], 'all_role_calls_at_acceptance': prefix,
                         'training_metrics': measured, 'evaluation_status': 'five valid evaluated training cases'}
            if generation == 0:
                seed = candidate
            else:
                candidates.append(candidate)
    unfinished = []
    for path in sorted(directory.glob('gen_*/held-proposal.json')):
        held = evidence(path)
        unfinished.append({'generation': int(path.parent.name[4:]), 'evidence': relative(path),
                           'stage': held.get('stage'), 'reason': held.get('reason')})
    for path in sorted(directory.glob('gen_*/accepted-proposal.json')):
        generation = int(path.parent.name[4:])
        if generation in canonical:
            continue
        accepted = evidence(path)
        output = path.parent / 'results'
        if ((output / 'correct.json').exists() and (output / 'resource.json').exists()
                and read(output / 'correct.json').get('correct')
                and read(output / 'resource.json').get('completed')):
            raise ValueError('A completed accepted evaluation lacks native persistence; reconcile saved evidence before nomination')
        unfinished.append({'generation': generation, 'evidence': relative(path),
                           'stage': accepted.get('stage'), 'reason': 'accepted proposal lacks completed native evaluation'})
    return {'run_id': spec['run_id'], 'arm': spec['arm'], 'replicate': spec['replicate'],
            'directory': relative(directory), 'native_seed': spec['native_seed'],
            'execution_status': execution['status'], 'stop_reason': execution.get('stop_reason'),
            'call_budget_blocked_reason': ledger.get('blocked_reason') if ledger else None,
            'worker_returncode': execution['returncode'],
            'actual_all_role_calls': len(calls), 'provider_seconds': elapsed,
            'call_roles': dict(Counter(call['role'] for call in calls)),
            'failed_provider_requests': sum(call.get('status') == 'error' or call.get('returncode') not in (None, 0) for call in calls),
            'usage_unavailable_calls': sum(call.get('usage_missing', not bool(call.get('usage'))) for call in calls),
            'completed_native_slots': slots, 'unfinished_proposals': unfinished,
            'evaluator_source_sha256': {name: config['source_sha256'][name] for name in EVALUATOR_SOURCES},
            'seed': seed, 'valid_descendants': candidates}


def freeze(plan_path, run_directories, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = read(plan_path)
    order = plan['search_design']['order']
    required = {spec['run_id'] for spec in order}
    if (len(order) != 6 or len(required) != 6 or set(run_directories) != required
            or plan['search_design']['repetitions_per_treatment'] != 3):
        raise ValueError('Nomination requires exactly the six declared independent search attempts')
    if not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Publish compact nominations inside repository artifacts/')
    inputs = {relative(plan_path): sha(plan_path)}
    with ExitStack() as stack:
        # Keep all stopped campaigns locked throughout read and freeze. These
        # read-only handles do not overwrite their existing controller records.
        for run_id in sorted(required):
            handle = stack.enter_context((Path(run_directories[run_id]) / 'controller.lock').open('r'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f'Search controller remains active: {run_id}') from exc
        runs = {spec['run_id']: read_run(run_directories[spec['run_id']], spec, plan, inputs) for spec in order}
        evaluator_sources = next(iter(runs.values()))['evaluator_source_sha256']
        if any(run['evaluator_source_sha256'] != evaluator_sources for run in runs.values()):
            raise ValueError('Independent searches used different seed/evaluator dependencies')
        prefixes = {}
        for replicate in range(1, 4):
            pair = [run for run in runs.values() if run['replicate'] == replicate]
            if len(pair) != 2 or {run['arm'] for run in pair} != {'full', 'rewrite'}:
                raise ValueError('Every repetition needs one full and one rewrite attempt')
            prefixes[str(replicate)] = min(run['actual_all_role_calls'] for run in pair)
        copies = {}
        for run in runs.values():
            prefix = prefixes[str(run['replicate'])]
            run['common_call_prefix'] = prefix
            run['primary'] = nominate(run['valid_descendants'], run['seed'])
            run['common_prefix_secondary'] = nominate(run['valid_descendants'], run['seed'], prefix)
            for group in ('primary', 'common_prefix_secondary'):
                for nominee in run[group]['nominations']:
                    path = ROOT / nominee['source']
                    destination = output / 'programs' / (nominee['source_sha256'] + '.py')
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != nominee['source_sha256']:
                        raise ValueError('Source changed while freezing nominations')
                    copies[destination] = data
                    nominee['frozen_source'] = relative(destination)
        manifest = {'status': 'nominations frozen; selection pool undrawn',
                    'study': plan['study'], 'plan_sha256': sha(plan_path),
                    'tool_sha256': sha(__file__), 'evaluator': 'namazu-proposal-reconstruction-v1',
                    'nomination_rule': plan['selection']['nominate_per_run'],
                    'selection_rule': plan['selection']['choose'],
                    'selection_case_count': plan['selection']['cases'],
                    'secondary_rule': plan['selection']['call_prefix_secondary'],
                    'common_call_prefix_by_replicate': prefixes, 'runs': list(runs.values()),
                    'input_sha256': inputs,
                    'frozen_program_sha256': {relative(path): hashlib.sha256(data).hexdigest() for path, data in copies.items()},
                    'new_episodes': 0, 'model_calls': 0, 'pools_drawn': 0}
        # Existing pools mean this is no longer a prospective nomination freeze.
        for key in ('selection', 'assessment'):
            if (ROOT / plan[key]['pool']).exists():
                raise ValueError('Selection/assessment pool already exists; refuse a new nomination freeze')
        payload = json.dumps(manifest, indent=2, allow_nan=False).encode() + b'\n'
        outputs = {**copies, output / 'nominations.json': payload}
        for path, data in outputs.items():
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f'Refusing to change frozen nomination evidence: {path}')
        for path, data in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open('xb') as handle:
                    handle.write(data)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['nominate'])
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', action='append', required=True, metavar='RUN_ID=DIRECTORY')
    args = parser.parse_args()
    directories = {}
    for argument in args.run:
        run_id, separator, directory = argument.partition('=')
        if not separator or not run_id or not directory or run_id in directories:
            parser.error('Each --run must uniquely specify RUN_ID=DIRECTORY')
        directories[run_id] = directory
    result = freeze(args.plan, directories, args.output)
    print(json.dumps({'status': result['status'], 'runs': len(result['runs']),
                      'distinct_frozen_sources': len(result['frozen_program_sha256']),
                      'common_call_prefix_by_replicate': result['common_call_prefix_by_replicate'],
                      'new_episodes': 0, 'model_calls': 0, 'pools_drawn': 0}, indent=2))


if __name__ == '__main__':
    main()
