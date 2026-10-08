"""Hash-frozen, resumable paired assessment of the separately versioned v3 wave.

No pool is generated without --reserve-pool and a complete verified plan. The
case checkpoint contains every regime/condition; immutable episode checkpoints
preserve completed work if a worker/controller stops partway through a case.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import gzip
import json
import os
from pathlib import Path
import secrets
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.evaluation_v3 import PROTOCOL, evaluation_identity, run_episode
from dreamer.provenance import pool, record, sha256
from dreamer.world_v3 import REGIMES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def recover_json_checkpoint(path):
    """Promote a complete interrupted write; retain incomplete bytes for audit."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    temporary = path.with_name(path.name + '.partial')
    if not temporary.exists():
        return None
    try:
        value = json.loads(temporary.read_text())
    except ValueError:
        temporary.rename(temporary.with_name(temporary.name + '.interrupted-' + uuid.uuid4().hex))
        return None
    os.link(temporary, path)
    temporary.unlink()
    return value


def atomic_create(path, value):
    """Atomic, immutable publication; recover complete temporary writes first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = recover_json_checkpoint(path)
    if path.exists():
        if existing != value:
            raise ValueError(f'Checkpoint drift: {path}')
        return
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False, separators=(',', ':'))
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def observed_transition_opportunities(trace, switch_step=None):
    """Conservative observation-only opportunities, never inferred enemy IDs.

    A source is one previously observed occupied cell. Its nine attempted moves
    are mapped using terrain observed in that same frame (including known door
    interaction and blocked/diagonal stays). All nine destination cells and
    diagonal-side cells must be observed. A contrast opportunity has at least
    two distinct possible destinations, of which the next observation reveals
    at least one occupied and at least one empty. Other destinations may be
    censored. A fully observed opportunity additionally reveals every possible
    destination. Overlapping/multiple enemies make these occupancy contrasts,
    not verified correspondence, independent samples, or quantified information.
    """
    keys = ('consecutive_observation_pairs', 'prior_visible_source_cells',
            'known_attempt_terrain_source_opportunities', 'multi_destination_source_opportunities',
            'observed_destination_contrast_opportunities', 'fully_observed_transition_opportunities',
            'censored_transition_opportunities', 'observation_steps_with_contrast')
    all_counts, post_counts = ({key: 0 for key in keys} for _ in range(2))
    all_sources, post_sources, contrast_sources, post_contrast_sources = set(), set(), set(), set()
    positions, position = [], (0, 0)
    for frame in trace:
        feedback = frame.get('obs', {}).get('feedback', {}).get('displacement', [0, 0])
        position = position[0]+feedback[0], position[1]+feedback[1]
        positions.append(position)
    contrast_by_frame = [0]*len(trace)
    for index in range(1, len(trace)):
        before, after = trace[index-1], trace[index]
        old, new = before.get('obs', {}), after.get('obs', {})
        if not all(key in old and key in new for key in ('step', 'grid', 'terrain')):
            continue
        if new['step'] != old['step']+1:
            continue
        ax, ay = positions[index-1]
        bx, by = positions[index]
        terrain = {(ax+x-2, ay+y-2): cell for y, row in enumerate(old['terrain']) for x, cell in enumerate(row)}
        sources = {(ax+x-2, ay+y-2) for y, row in enumerate(old['grid']) for x, cell in enumerate(row) if cell == 5}
        next_occupancy = {(bx+x-2, by+y-2): cell == 5 for y, row in enumerate(new['grid']) for x, cell in enumerate(row)}
        post = switch_step is not None and old['step']+1 >= switch_step
        selected_counts = [all_counts, post_counts] if post else [all_counts]
        for counts in selected_counts:
            counts['consecutive_observation_pairs'] += 1
            counts['prior_visible_source_cells'] += len(sources)
        all_sources.update(sources)
        if post:
            post_sources.update(sources)
        def blocked(point):
            cell = terrain[point]
            if cell in (-1, 1):
                return True
            if cell == 3 and not old.get('door_open', False):
                opens = (before.get('action', {}).get('interact') is True and old.get('keys') == 2
                         and abs(point[0]-ax)+abs(point[1]-ay) == 1)
                return not opens
            return False
        for source in sources:
            destinations = set()
            if any((source[0]+dx, source[1]+dy) not in terrain
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                continue
            for counts in selected_counts:
                counts['known_attempt_terrain_source_opportunities'] += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    target = source[0]+dx, source[1]+dy
                    illegal = blocked(target) or (dx and dy and
                        (blocked((source[0]+dx, source[1])) or blocked((source[0], source[1]+dy))))
                    destinations.add(source if illegal else target)
            if len(destinations) < 2:
                continue
            for counts in selected_counts:
                counts['multi_destination_source_opportunities'] += 1
            labels = {next_occupancy[p] for p in destinations if p in next_occupancy}
            if labels != {False, True}:
                continue
            full = destinations <= next_occupancy.keys()
            contrast_by_frame[index] += 1
            contrast_sources.add(source)
            if post:
                post_contrast_sources.add(source)
            for counts in selected_counts:
                counts['observed_destination_contrast_opportunities'] += 1
                counts['fully_observed_transition_opportunities'] += full
                counts['censored_transition_opportunities'] += not full
        if contrast_by_frame[index]:
            for counts in selected_counts:
                counts['observation_steps_with_contrast'] += 1
    all_counts.update(distinct_prior_source_cells=len(all_sources), distinct_contrast_source_cells=len(contrast_sources))
    post_counts.update(distinct_prior_source_cells=len(post_sources), distinct_contrast_source_cells=len(post_contrast_sources))
    return {'all': all_counts, 'post_switch': post_counts,
        'definition': 'Consecutive observations; a prior occupied cell with all nine attempt terrain cells observed, at least two distinct blocked-stay/legal destinations, and next-visible occupied versus empty destinations. Fully observed additionally sees every possible destination.',
        'unit': 'anonymous occupied-source-cell transition opportunity, not enemy identity or independent sample',
        'limitations': 'A conservative occupancy-contrast proxy, not measured information gain or confirmed motion correspondence; other enemies and unobserved destinations can confound it. Final transitions without a next observation are excluded.',
        'post_switch_scope': 'The observed outcome followed a replacement-law transition; conditional on surviving to its next observation.'}, contrast_by_frame


def compact_trace(row, parameter_key=None):
    """Hash arbitrary exported adaptive state, without assuming its architecture.

    parameter_key selects a key in model.learning, or a dotted path beginning
    model. No key means no claimed parameter audit. A missing export is recorded
    as unavailable, never interpreted as a successful freeze.
    """
    trace = row.pop('trace', [])
    keys = (parameter_key if isinstance(parameter_key, list) else parameter_key.split('.')[1:] if parameter_key and parameter_key.startswith('model.')
            else ['learning', parameter_key] if parameter_key else None)
    parameters = []
    for frame in trace:
        value = frame.get('model', {})
        for key in keys or []:
            value = value.get(key) if isinstance(value, dict) else None
        parameters.append(value if keys else None)
    exported = bool(keys and parameters) and all(p is not None for p in parameters)
    opportunities, contrast_by_frame = observed_transition_opportunities(trace, row.get('switch_step'))
    opportunities['parameter_changes_without_counted_contrast'] = (sum(parameters[i] != parameters[i-1] and not contrast_by_frame[i]
        for i in range(1, len(parameters))) if exported else None)
    opportunities['parameter_change_note'] = 'Exported parameter changes are separate from observed transition opportunities; forgetting or regularization can change parameters without a newly counted occupancy contrast.'
    positions = [f.get('model', {}).get('position') for f in trace]
    maps = [sorted(f.get('model', {}).get('terrain', [])) for f in trace]
    localization_checks = localization_errors = 0
    for frame, position in zip(trace, positions):
        if isinstance(position, (list, tuple)) and len(position) == 2:
            localization_checks += 1
            world = frame['world']
            localization_errors += [position[i]+world['origin'][i] for i in range(2)] != list(world['agent'])
    row['audit'] = {
        'trajectory_sha256': digest([{k: f[k] for k in ('world', 'action', 'next_enemies')} for f in trace]),
        'actions_sha256': digest([f['action'] for f in trace]),
        'observations_sha256': digest([{k: v for k, v in f.get('obs', {}).items()
            if k not in ('learn', 'predictive_planning', 'known_law')} for f in trace]),
        'observation_interventions_sha256': digest([{k: f.get('obs', {}).get(k)
            for k in ('learn', 'predictive_planning')} for f in trace]),
        'map_position_sha256': digest(list(zip(positions, maps))),
        'forecast_sha256': digest([{k: f.get('model', {}).get(k) for k in ('enemy', 'default_enemy')} for f in trace]),
        'observed_transition_opportunities': opportunities,
        'parameter_key': parameter_key, 'parameters_exported': exported,
        'parameters_constant': all(p == parameters[0] for p in parameters) if exported else None,
        'parameter_change_steps': sum(a != b for a, b in zip(parameters, parameters[1:])) if exported else None,
        'first_parameters': parameters[0] if exported else None,
        'final_parameters': parameters[-1] if exported else None,
        'parameter_trace_sha256': digest(parameters) if exported else None,
        'frames': len(trace), 'localization_checks': localization_checks,
        'localization_errors': localization_errors if localization_checks else None,
        'distinct_positions': len({digest(p) for p in positions if p is not None}),
        'first_map_cells': len(maps[0]) if maps else 0,
        'last_map_cells': len(maps[-1]) if maps else 0}
    if row.get('variant') == 'known_law':
        # Privileged probabilities belong to private execution, not compact evidence.
        row.get('learning', {}).pop('attempt_probabilities', None)
        row['audit']['first_parameters'] = None
        row['audit']['final_parameters'] = None
        row['audit']['privileged_parameter_values_redacted'] = True
    return row


def failure_row(condition, regime, error, seconds):
    return {'variant': condition['variant'], 'regime': regime, 'reason': 'invalid',
            'steps': 0, 'keys': 0, 'door': False, 'task': 0., 'combined_score': 0.,
            'model_score': .75, 'stats': {}, 'bins': {}, 'learning': {},
            'seconds': seconds, 'candidate_cpu_seconds': 0., 'evaluator_cpu_seconds': 0.,
            'cpu_measurement_unavailable': True,
            'error': f'{type(error).__name__}: {error}', 'switch_step': None,
            'encountered_switch': False, 'enemy_innovation_draws': 0,
            'exposure': {k: 0 for k in ('visible_enemy_steps', 'visible_enemy_cells',
                'consecutive_visible_steps', 'post_switch_steps', 'post_switch_visible_enemy_steps',
                'post_switch_consecutive_visible_steps')}}


def preserve_matched_trace(path, trace):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with gzip.open(path, 'rt') as handle:
            if json.load(handle) != trace:
                raise ValueError('Saved matched-experience trace drift')
        return
    temporary = path.with_name(path.name + '.partial')
    with gzip.open(temporary, 'wt') as handle:
        json.dump(trace, handle, allow_nan=False, separators=(',', ':'))
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def complete_matched(root, case, seed, regime, spec, row):
    if not spec:
        return
    root = Path(root)
    path = root / 'matched' / f'{case:06d}--{regime}.json'
    trace_path = root / 'matched-private-traces' / f'{case:06d}--{regime}.json.gz'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('policy_trajectory_sha256') != row.get('audit', {}).get('trajectory_sha256'):
            raise ValueError('Matched checkpoint trajectory mismatch')
        if trace_path.exists():
            trace_path.unlink()
        return
    if not trace_path.exists():
        raise ValueError('Missing recorded experience for matched analysis; refusing an unaccounted world rerun')
    with gzip.open(trace_path, 'rt') as handle:
        trace = json.load(handle)
    trajectory_hash = digest([{k: f[k] for k in ('world', 'action', 'next_enemies')} for f in trace])
    if trajectory_hash != row.get('audit', {}).get('trajectory_sha256'):
        raise ValueError('Private trace does not match saved outcome trajectory')
    from scripts.v3_matched import evaluate_trace
    attempt = uuid.uuid4().hex
    started = time.monotonic()
    record_start = {'attempt': attempt, 'case': case, 'regime': regime,
                    'started_utc': datetime.now(timezone.utc).isoformat(),
                    'policy_trajectory_sha256': trajectory_hash, 'trace_sha256': sha256(trace_path)}
    atomic_create(root / 'matched-attempts' / (attempt + '.started.json'), record_start)
    try:
        result = evaluate_trace(trace, seed, regime, case, str(resolve(spec['source'])),
                                expected_source_sha256=spec['source_sha256'], policy_reason=row['reason'])
    except Exception as error:
        # World outcomes are already durable. Failed diagnostic execution is reported.
        result = {'case': case, 'regime': regime, 'policy_trajectory_sha256': trajectory_hash,
                  'shadows': {}, 'error': f'{type(error).__name__}: {error}', 'condition_episodes': 0,
                  'seconds': time.monotonic()-started, 'candidate_cpu_seconds': 0.,
                  'evaluator_cpu_seconds': 0., 'cpu_measurement_unavailable': True}
    result['recorded_trace_sha256'] = record_start['trace_sha256']
    result['source_sha256'] = spec['source_sha256']
    result['policy_reason'] = row['reason']
    record_start.update(seconds=result['seconds'], candidate_cpu_seconds=result['candidate_cpu_seconds'],
                        evaluator_cpu_seconds=result['evaluator_cpu_seconds'],
                        condition_episodes=result['condition_episodes'], error=result.get('error'))
    atomic_create(root / 'matched-attempts' / (attempt + '.completed.json'), record_start)
    atomic_create(path, result)
    # Keep private evidence on diagnostic failure for an explicitly recorded retry.
    if not result.get('error'):
        trace_path.unlink()


def retain_replay(root, case, regime, condition, trace):
    relative = Path('replays') / f'{case:06d}--{regime}--{condition}.json.gz'
    path = Path(root) / relative
    preserve_matched_trace(path, trace)
    return {'path': relative.as_posix(), 'sha256': sha256(path),
            'trajectory_sha256': digest([{k: f[k] for k in ('world', 'action', 'next_enemies')} for f in trace])}


def replay_is_registered(spec, case, condition):
    return bool(spec and case < spec['first_cases'] and condition in spec['conditions'])


def verify_retained_replay(root, case, regime, condition, spec, row):
    if not replay_is_registered(spec, case, condition):
        return
    expected_path = f'replays/{case:06d}--{regime}--{condition}.json.gz'
    saved = row.get('audit', {}).get('retained_replay', {})
    path = Path(root) / expected_path
    if (saved.get('path') != expected_path or not path.exists() or saved.get('sha256') != sha256(path)
            or saved.get('trajectory_sha256') != row.get('audit', {}).get('trajectory_sha256')):
        raise ValueError('Registered passive replay is missing or changed; refusing an unaccounted world rerun')


def evaluate_case(job):
    index, seed, conditions, regimes, checkpoint_root, *options = job
    matched_spec = options[0] if options else None
    replay_spec = options[1] if len(options) > 1 else None
    root = Path(checkpoint_root)
    rows = []
    combinations = [(condition, regime) for regime in regimes for condition in conditions]
    offset = index % len(combinations)
    ordered = combinations[offset:] + combinations[:offset]
    for condition, regime in ordered:
        path = root / 'episode-checkpoints' / f'{index:06d}' / f'{condition["name"]}--{regime}.json'
        if not path.exists() and path.with_name(path.name + '.partial').exists():
            partial = path.with_name(path.name + '.partial')
            try:
                json.loads(partial.read_text())
                os.link(partial, path)
                partial.unlink()
            except ValueError:
                partial.rename(partial.with_name(partial.name + '.interrupted-' + uuid.uuid4().hex))
        condition_matched = matched_spec if matched_spec and condition['name'] == matched_spec['memory_condition'] else None
        if path.exists():
            row = json.loads(path.read_text())
            if row['program_sha256'] != condition['program_sha256'] or row['variant'] != condition['variant']:
                raise ValueError('Episode checkpoint source/variant mismatch')
            if (row['case'], row['condition'], row['regime']) != (index, condition['name'], regime):
                raise ValueError('Episode checkpoint identity mismatch')
            verify_retained_replay(root, index, regime, condition['name'], replay_spec, row)
            complete_matched(root, index, seed, regime, condition_matched, row)
            rows.append(row)
            continue
        if sha256(condition['program_path']) != condition['program_sha256']:
            raise ValueError('Frozen source changed during execution')
        attempt = uuid.uuid4().hex
        started = time.monotonic()
        worker_cpu_started = time.process_time()
        cost = {'attempt': attempt, 'case': index, 'condition': condition['name'], 'regime': regime,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'worker_pid': os.getpid()}
        atomic_create(root / 'attempts' / (attempt + '.started.json'), cost)
        try:
            row = run_episode(condition['program_path'], seed, variant=condition['variant'],
                              replay=True, regime=regime)
        except Exception as error:
            row = failure_row(condition, regime, error, time.monotonic()-started)
            row['trace'] = []
        retained_replay = (retain_replay(root, index, regime, condition['name'], row['trace'])
                           if replay_is_registered(replay_spec, index, condition['name']) else None)
        if condition_matched:
            preserve_matched_trace(root / 'matched-private-traces' / f'{index:06d}--{regime}.json.gz', row['trace'])
        try:
            row = compact_trace(row, condition.get('parameter_key'))
        except Exception as error:
            # Diagnostic export errors never rewrite evaluator-owned task outcomes.
            row.pop('trace', None)
            row['audit'] = {'error': f'{type(error).__name__}: {error}'}
        if retained_replay:
            row['audit']['retained_replay'] = retained_replay
            row['audit'].setdefault('trajectory_sha256', retained_replay['trajectory_sha256'])
        row.pop('seed', None)
        row.update(case=index, condition=condition['name'], program_sha256=condition['program_sha256'])
        cost.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started,
                    assessment_cpu_seconds=time.process_time()-worker_cpu_started,
                    candidate_cpu_seconds=row['candidate_cpu_seconds'], evaluator_cpu_seconds=row['evaluator_cpu_seconds'],
                    invalid=row['reason'] == 'invalid', cpu_measurement_unavailable=row.get('cpu_measurement_unavailable', False))
        atomic_create(root / 'attempts' / (attempt + '.completed.json'), cost)
        atomic_create(path, row)
        complete_matched(root, index, seed, regime, condition_matched, row)
        rows.append(row)
    order = {(c['name'], regime): i for i, (c, regime) in enumerate(combinations)}
    rows.sort(key=lambda r: order[r['condition'], r['regime']])
    return {'case': index, 'conditions': rows}


def check_reference(reference, label):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError(f'Missing frozen {label} reference')
    if sha256(resolve(reference['path'])) != reference['sha256']:
        raise ValueError(f'Frozen {label} changed')


def verify_plan(plan, check_driver=True):
    if plan.get('protocol') != PROTOCOL or plan.get('evaluation') != evaluation_identity():
        raise ValueError('Plan evaluator identity mismatch')
    if type(plan.get('sample_size')) is not int or plan['sample_size'] <= 0:
        raise ValueError('Freeze a positive sample size')
    if plan.get('regimes') != list(REGIMES):
        raise ValueError('All three frozen regimes are required')
    if check_driver and plan.get('driver_sha256') != sha256(__file__):
        raise ValueError('Assessment driver changed after freezing')
    check_reference(plan.get('selection'), 'selection')
    check_reference(plan.get('sample_size_justification'), 'sample-size justification')
    for category in ('analysis_files', 'design_files'):
        if not plan.get(category):
            raise ValueError(f'Freeze {category} before drawing cases')
        for path, expected in plan[category].items():
            if sha256(resolve(path)) != expected:
                raise ValueError(f'Frozen file changed: {path}')
    conditions = plan.get('conditions', [])
    names = [c['name'] for c in conditions]
    if not names or len(set(names)) != len(names):
        raise ValueError('Unique named conditions are required')
    for condition in conditions:
        if not condition['name'].replace('_', '').replace('-', '').isalnum():
            raise ValueError('Condition name must be safe for checkpoint paths')
        if sha256(resolve(condition['program_path'])) != condition['program_sha256']:
            raise ValueError(f'Frozen condition source changed: {condition["name"]}')
        if condition.get('variant') not in {'predictive', 'frozen', 'no_planning', 'frozen_no_planning',
                                          'known_law', 'memory', 'original_predictive'}:
            raise ValueError('Each condition needs a supported declared variant')
    replays = plan.get('replays')
    if replays:
        if (type(replays.get('first_cases')) is not int or not 1 <= replays['first_cases'] <= plan['sample_size']
                or not replays.get('conditions') or len(set(replays['conditions'])) != len(replays['conditions'])
                or not set(replays['conditions']) <= set(names)):
            raise ValueError('Freeze a bounded prefix and named conditions for passive replay retention')
    matched = plan.get('matched')
    if matched:
        if matched.get('memory_condition') not in names:
            raise ValueError('Matched experience needs a named assessment policy condition')
        if sha256(resolve(matched['source'])) != matched['source_sha256']:
            raise ValueError('Frozen matched-predictor source changed')
        if not matched.get('helper_files'):
            raise ValueError('Freeze matched helper and worker files before sampling')
        for path, expected in matched['helper_files'].items():
            if sha256(resolve(path)) != expected:
                raise ValueError('Frozen matched helper changed: ' + path)
    private = (ROOT / 'results/private').resolve()
    target = resolve(plan['pool_path'])
    if private not in target.parents:
        raise ValueError('Assessment seed pool must remain in ignored results/private')
    return [dict(c, program_path=str(resolve(c['program_path']))) for c in conditions]


def reserve_pool(plan, plan_path, out, reserve=False):
    """Verify every frozen artifact BEFORE sampling and bind plan+pool immutably."""
    verify_plan(plan)
    out = Path(out)
    atomic_create(out / 'frozen-plan.json', plan)
    target = resolve(plan['pool_path'])
    pool_manifest = out / 'pool-manifest.json'
    recover_json_checkpoint(pool_manifest)
    recover_json_checkpoint(out / 'pool-reservation.json')
    recover_json_checkpoint(target)
    if not pool_manifest.exists():
        intent_path = out / 'pool-reservation.json'
        if not intent_path.exists() and (not reserve or target.exists()):
            raise ValueError('Fresh pool requires --reserve-pool and a new private pool path')
        intent = {'frozen_plan_sha256': sha256(out / 'frozen-plan.json'),
                  'original_plan_sha256': sha256(plan_path), 'pool_path': str(target),
                  'sample_size': plan['sample_size']}
        atomic_create(intent_path, intent)
        excluded = set(range(10000, 100000))
        explicit = [resolve(p) for p in plan.get('excluded_pool_paths', [])]
        # Retire every locally retained historical/development/selection pool.
        private_pools = list((ROOT / 'results/private').rglob('*seed*.json'))
        for path in set(explicit + private_pools):
            if path == target:
                continue
            values = json.loads(path.read_text())
            if isinstance(values, list) and all(type(x) is int for x in values):
                excluded.update(values)
            elif path in explicit:
                raise ValueError(f'Cannot parse explicit excluded pool: {path}')
        excluded_count = len(excluded)
        if target.exists():
            seeds, _ = pool(target, plan['sample_size'])
            if set(seeds) & excluded:
                raise ValueError('Reserved pool overlaps excluded cases')
        else:
            seeds = []
            while len(seeds) < plan['sample_size']:
                candidate = secrets.randbits(63)
                if candidate not in excluded:
                    seeds.append(candidate)
                    excluded.add(candidate)
            atomic_create(target, seeds)
        _, info = pool(target, plan['sample_size'])
        atomic_create(pool_manifest, {'pool': info, 'frozen_plan_sha256': sha256(out / 'frozen-plan.json'),
                              'original_plan_sha256': sha256(plan_path), 'excluded_seed_count': excluded_count})
    saved = json.loads(pool_manifest.read_text())
    seeds, info = pool(target, plan['sample_size'])
    if (saved['pool'] != info or saved['frozen_plan_sha256'] != sha256(out / 'frozen-plan.json')
            or saved['original_plan_sha256'] != sha256(plan_path)):
        raise ValueError('Assessment pool or frozen plan drift')
    return seeds, info


def run_cases(out, seeds, conditions, regimes, workers=4, matched=None, replays=None):
    if not 1 <= workers <= 4:
        raise ValueError('Use one through four independent process workers')
    out = Path(out)
    cases = out / 'cases'
    cases.mkdir(parents=True, exist_ok=True)
    expected = {(c['name'], r) for r in regimes for c in conditions}
    existing = {}
    for path in cases.glob('*.json'):
        case = json.loads(path.read_text())
        i = case['case']
        rows = case['conditions']
        if not 0 <= i < len(seeds) or path.name != f'{i:06d}.json':
            raise ValueError('Unexpected case checkpoint')
        if len(rows) != len(expected) or {(r['condition'], r['regime']) for r in rows} != expected:
            raise ValueError('Incomplete paired case checkpoint')
        for row in rows:
            verify_retained_replay(out, i, row['regime'], row['condition'], replays, row)
        existing[i] = case
    pending = [i for i in range(len(seeds)) if i not in existing]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session = {'started_utc': stamp, 'workers': workers, 'existing_cases': len(existing),
               'requested_cases': len(seeds), 'model_calls': 0}
    atomic_create(out / ('execution-' + stamp + '.started.json'), session)
    started, cpu = time.monotonic(), time.process_time()
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(pending), workers):
                jobs = [(i, seeds[i], conditions, regimes, str(out), matched, replays) for i in pending[offset:offset+workers]]
                for case in executor.map(evaluate_case, jobs):
                    atomic_create(cases / f'{case["case"]:06d}.json', case)
                    existing[case['case']] = case
                print(json.dumps({'completed_cases': len(existing), 'requested_cases': len(seeds),
                                  'condition_episodes': len(existing)*len(expected),
                                  'session_elapsed_seconds': round(time.monotonic()-started, 2)}), flush=True)
    finally:
        session.update(completed_cases=len(existing), wall_seconds=time.monotonic()-started,
                       controller_cpu_seconds=time.process_time()-cpu)
        atomic_create(out / ('execution-' + stamp + '.finished.json'), session)
    if len(existing) != len(seeds):
        raise RuntimeError('Assessment incomplete; resume the same command')
    rows = [r for i in sorted(existing) for r in existing[i]['conditions']]
    attempts = [json.loads(p.read_text()) for p in (out / 'attempts').glob('*.completed.json')]
    completed_ids = {r['attempt'] for r in attempts}
    unknown = [p.stem.removesuffix('.started') for p in (out / 'attempts').glob('*.started.json')
               if p.stem.removesuffix('.started') not in completed_ids]
    completion = {'cases': len(seeds), 'condition_episodes': len(rows),
        'invalid_condition_episodes': sum(r['reason'] == 'invalid' for r in rows),
        'candidate_cpu_seconds': sum(r['candidate_cpu_seconds'] for r in attempts),
        'evaluator_cpu_seconds': sum(r['evaluator_cpu_seconds'] for r in attempts),
        'assessment_worker_cpu_seconds': sum(r.get('assessment_cpu_seconds', r['evaluator_cpu_seconds']) for r in attempts),
        'assessment_worker_cpu_scope': 'Includes evaluator CPU plus assessment audit/retention; do not add evaluator_cpu_seconds again.',
        'episode_wall_seconds': sum(r['wall_seconds'] for r in attempts),
        'completed_episode_attempts': len(attempts), 'interrupted_attempts_unmeasured': unknown,
        'cpu_unavailable_attempts': sum(r['cpu_measurement_unavailable'] for r in attempts),
        'case_file_sha256': {p.name: sha256(p) for p in sorted(cases.glob('*.json'))}}
    replay_paths = sorted((out / 'replays').glob('*.json.gz'))
    expected_replays = ({f'{i:06d}--{regime}--{name}.json.gz' for i in range(replays['first_cases'])
                         for regime in regimes for name in replays['conditions']} if replays else set())
    if {p.name for p in replay_paths} != expected_replays:
        raise ValueError('Saved passive replay set differs from the frozen retention prefix')
    completion['passive_replays'] = {'files': len(replay_paths),
        'retention_specification': replays,
        'file_sha256': {p.name: sha256(p) for p in replay_paths},
        'bytes': sum(p.stat().st_size for p in replay_paths), 'new_environment_episodes': 0}
    matched_rows = [json.loads(p.read_text()) for p in (out / 'matched').glob('*.json')]
    matched_attempts = [json.loads(p.read_text()) for p in (out / 'matched-attempts').glob('*.completed.json')]
    completed_matched_ids = {r['attempt'] for r in matched_attempts}
    completion['matched'] = {'recorded_policy_episodes': len(matched_rows),
        'shadow_condition_episodes': sum(r['condition_episodes'] for r in matched_rows),
        'candidate_cpu_seconds': sum(r['candidate_cpu_seconds'] for r in matched_attempts),
        'evaluator_cpu_seconds': sum(r['evaluator_cpu_seconds'] for r in matched_attempts),
        'wall_seconds': sum(r['seconds'] for r in matched_attempts),
        'completed_shadow_attempts': sum(r['condition_episodes'] for r in matched_attempts),
        'interrupted_attempts_unmeasured': [p.stem.removesuffix('.started') for p in
            (out / 'matched-attempts').glob('*.started.json')
            if p.stem.removesuffix('.started') not in completed_matched_ids],
        'failed_recorded_episode_analyses': sum(bool(r.get('error')) for r in matched_rows),
        'file_sha256': {p.name: sha256(p) for p in sorted((out / 'matched').glob('*.json'))}}
    atomic_create(out / 'execution-complete.json', completion)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reserve-pool', action='store_true')
    args = parser.parse_args()
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = resolve(args.plan)
        plan = json.loads(plan_path.read_text())
        conditions = verify_plan(plan)
        seeds, info = reserve_pool(plan, plan_path, out, args.reserve_pool)
        atomic_create(out / 'manifest.json', {'plan_sha256': sha256(plan_path), 'episode_pool': info,
            'evaluation': evaluation_identity(), 'conditions': plan['conditions'], 'regimes': plan['regimes'],
            'condition_order': 'rotate regime-major conditions by case index', 'model_calls': 0})
        run_cases(out, seeds, conditions, plan['regimes'], args.workers, plan.get('matched'), plan.get('replays'))


if __name__ == '__main__':
    main()
