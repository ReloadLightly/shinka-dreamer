"""Prepare and freeze original-study panels; execution requires explicit `run`."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

from proposal.study_selection import DEFAULT_PLAN, EVALUATOR_SOURCES, ROOT, read, relative, sha

IDENTITY = 'namazu-proposal-reconstruction-v1'
NOMINATIONS = ROOT / 'artifacts/proposal/broader-study/search-selection/nominations.json'
CONTROL_PROTOCOL = ROOT / 'artifacts/proposal/broader-study/controls-assessment/protocol.json'
PHASE_PURPOSE = {
    'selection': 'Shared 64-case selection validation; winner selection is biased, not final assessment.',
    'assessment': 'Fresh 512-case assessment of frozen independent-search winners and controls; no selection.',
}


def verify_hashes(expected):
    for name, digest in expected.items():
        path = ROOT / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'Frozen input hash differs: {name}')


def save_once(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f'Refusing to overwrite frozen evidence: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(payload)


def absent_pool(plan, phase):
    if (ROOT / plan[phase]['pool']).exists():
        raise ValueError(f'{phase.capitalize()} pool already exists; a prospective freeze cannot be changed')


def checked_nominations(path, plan_path):
    nominations, plan = read(path), read(plan_path)
    if (nominations['plan_sha256'] != sha(plan_path) or nominations['evaluator'] != IDENTITY
            or nominations['selection_case_count'] != 64 or plan['selection']['cases'] != 64
            or plan['assessment']['cases'] != 512):
        raise ValueError('Nomination and declared study identities differ')
    verify_hashes(nominations['input_sha256'])
    verify_hashes(nominations['frozen_program_sha256'])
    if nominations['tool_sha256'] != sha(ROOT / 'proposal/study_selection.py'):
        raise ValueError('Nomination tool differs from its frozen identity')
    expected = {(run['run_id'], run['arm'], run['replicate']) for run in plan['search_design']['order']}
    actual = {(run['run_id'], run['arm'], run['replicate']) for run in nominations['runs']}
    if len(nominations['runs']) != 6 or actual != expected:
        raise ValueError('All six declared search attempts must be nominated')
    return nominations, plan


def program_key(digest):
    return 'source_' + digest


def programs_from_nominees(nominations):
    programs = {}
    for run in nominations['runs']:
        for group in ('primary', 'common_prefix_secondary'):
            for candidate in run[group]['nominations']:
                digest = candidate['source_sha256']
                path = candidate['frozen_source']
                if nominations['frozen_program_sha256'].get(path) != digest:
                    raise ValueError('Nominee is absent from the frozen source manifest')
                programs.setdefault(program_key(digest), {'path': path, 'sha256': digest})
    if not programs:
        raise ValueError('Every search must supply nominations or seed fallback')
    return programs


def panel_sources(programs, original_sources):
    sources = {name: original_sources[name] for name in EVALUATOR_SOURCES}
    verify_hashes(sources)
    for name in ('proposal/panel.py', 'proposal/study_panels.py', 'proposal/study_statistics.py',
                 'proposal/study_report.py', 'proposal/development_report.py'):
        sources[name] = sha(ROOT / name)
    for spec in programs.values():
        path = (ROOT / spec['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Candidate source must remain inside this repository')
        sources[spec['path']] = spec['sha256']
    verify_hashes(sources)
    return sources


def bind_protocol(phase, plan_path, plan, public, private, programs, sources,
                  minutes, cpu_seconds, evidence):
    public, private = Path(public).resolve(), Path(private).resolve()
    if not public.is_relative_to(ROOT / 'artifacts') or not private.is_relative_to(ROOT / 'results'):
        raise ValueError('Public artifacts and private results must stay in their repository roots')
    if (not math.isfinite(minutes) or not 5 < minutes <= 120
            or not math.isfinite(cpu_seconds) or cpu_seconds < 20):
        raise ValueError('Require 5 < minutes <= 120 and an explicit finite CPU ceiling >=20 seconds')
    protocol_path = public / 'protocol.json'
    if protocol_path.exists():
        raise ValueError('Execution protocol already frozen; do not extend its deadline')
    if private.exists() and any(private.iterdir()):
        raise ValueError('Use a fresh private execution directory')
    absent_pool(plan, phase)
    now = datetime.now(timezone.utc)
    count = plan[phase]['cases']
    protocol = {'stage': f'broader-original-study-search-{phase}-v1', 'phase': phase,
                'created_utc': now.isoformat(), 'evaluator': IDENTITY, 'purpose': PHASE_PURPOSE[phase],
                'study_plan': {'path': relative(plan_path), 'sha256': sha(plan_path)},
                'case_count': count, 'pool_path': plan[phase]['pool'],
                'public_output': relative(public), 'private_output': relative(private),
                'programs': programs, 'source_sha256': sources, **evidence,
                'budget': {'stage_start_utc': now.isoformat(),
                           'execution_stop_utc': (now + timedelta(minutes=minutes - 5)).isoformat(),
                           'checkpoint_deadline_utc': (now + timedelta(minutes=minutes)).isoformat(),
                           'stage_elapsed_minutes': minutes, 'reporting_reserve_minutes': 5,
                           'total_evaluator_cpu_seconds': cpu_seconds, 'experiment_model_calls': 0,
                           'episodes': count * len(programs), 'transitions': count * len(programs) * 200,
                           'workers': 1, 'candidate_memory_mib': 192,
                           'candidate_cpu_seconds_per_episode': 10, 'episode_wall_seconds': 20},
                'case_separation': 'Private draws reject all integers below1000000 and every earlier private seed pool. Public IDs are ordinals; no panel evidence enters search prompts.',
                'execution': 'Unchanged proposal.panel scoring, candidate isolation and per-episode checkpoints; no automatic retries or deadline extensions.',
                'automatic_continuation': False}
    save_once(protocol_path, protocol)
    return protocol


def prepare_selection(nomination_path, plan_path, public, private, minutes, cpu_seconds):
    nominations, plan = checked_nominations(nomination_path, plan_path)
    absent_pool(plan, 'selection')
    absent_pool(plan, 'assessment')
    programs = programs_from_nominees(nominations)
    sources = panel_sources(programs, nominations['runs'][0]['evaluator_source_sha256'])
    return bind_protocol('selection', plan_path, plan, public, private, programs, sources, minutes, cpu_seconds,
                         {'nominations': {'path': relative(nomination_path), 'sha256': sha(nomination_path)},
                          'selection_rule': plan['selection']['choose'],
                          'common_call_prefix_by_replicate': nominations['common_call_prefix_by_replicate']})


def selection_means(rows, count=64):
    if len(rows) != count or [row['case'] for row in rows] != list(range(count)):
        raise ValueError('Selection requires the complete ordered case panel, including invalids')
    for row in rows:
        if row['reason'] not in ('escaped', 'caught', 'timeout', 'invalid'):
            raise ValueError('Unknown saved outcome')
        if (row['reason'] == 'invalid') != (row['error'] is not None):
            raise ValueError('Invalid-outcome metadata differs')
        if not 0 <= row['map_correct'] <= row['map_audited']:
            raise ValueError('Invalid reported-map denominator')
        accuracy = row['map_correct'] / row['map_audited'] if row['map_audited'] else 0.
        task = min(1., max(0., (.1 * row['steps'] + 20 * row['keys'] + 30 * row['door_open']
                  + 100 * (row['reason'] == 'escaped') - 50 * (row['reason'] == 'caught') + 50) / 250))
        fitness = .6 * task + .4 * accuracy if row['error'] is None else 0.
        for key, value in [('task', task), ('model_accuracy', accuracy), ('combined_score', fitness)]:
            if not math.isclose(row[key], value, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f'Saved selection {key} arithmetic differs')
    return {'combined_score': sum(row['combined_score'] for row in rows) / count,
            'task': sum(row['task'] for row in rows) / count,
            'escaped': sum(row['reason'] == 'escaped' for row in rows),
            'invalid': sum(row['reason'] == 'invalid' for row in rows), 'episodes': count}


def choose_winner(nominees, means):
    """Frozen lexicographic rule; training scores never break selection ties."""
    if not nominees:
        raise ValueError('Nomination group must contain a candidate or seed fallback')
    winner = min(nominees, key=lambda candidate: (
        -means[program_key(candidate['source_sha256'])]['combined_score'],
        -means[program_key(candidate['source_sha256'])]['task'], candidate['generation']))
    return dict(winner, panel_program=program_key(winner['source_sha256']),
                selection_metrics=means[program_key(winner['source_sha256'])])


def freeze_winners(selection_dir, plan_path, output):
    selection_dir, output = Path(selection_dir).resolve(), Path(output).resolve()
    if not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Winner freeze belongs in public repository artifacts')
    protocol_path = selection_dir / 'protocol.json'
    protocol, plan = read(protocol_path), read(plan_path)
    absent_pool(plan, 'assessment')
    if (protocol['phase'] != 'selection' or protocol['case_count'] != 64
            or protocol['study_plan']['sha256'] != sha(plan_path)
            or protocol['evaluator'] != IDENTITY):
        raise ValueError('Expected the frozen 64-case selection protocol')
    nomination_path = ROOT / protocol['nominations']['path']
    if sha(nomination_path) != protocol['nominations']['sha256']:
        raise ValueError('Nomination freeze changed after panel preparation')
    nominations, _ = checked_nominations(nomination_path, plan_path)
    verify_hashes(protocol['source_sha256'])
    if protocol['programs'] != programs_from_nominees(nominations):
        raise ValueError('Selection programs differ from the deduplicated nominations')
    pool, execution = read(selection_dir / 'pool-manifest.json'), read(selection_dir / 'execution.json')
    if (pool['protocol_sha256'] != sha(protocol_path) or pool['case_count'] != 64
            or execution['status'] != 'complete' or execution['episodes'] != 64 * len(protocol['programs'])):
        raise ValueError('Selection execution is incomplete or pool binding differs')
    inputs = {relative(path): sha(path) for path in
              (protocol_path, nomination_path, plan_path, selection_dir / 'pool-manifest.json', selection_dir / 'execution.json')}
    means, transitions = {}, 0
    for key, spec in protocol['programs'].items():
        folder = selection_dir / key
        rows, manifest = read(folder / 'episodes.json'), read(folder / 'manifest.json')
        if (manifest['candidate_sha256'] != spec['sha256'] or manifest['episodes'] != 64
                or manifest['evaluator'] != IDENTITY or manifest['pool_sha256'] != pool['pool_sha256']
                or manifest['source_sha256'] != protocol['source_sha256']):
            raise ValueError('Selection source, evaluator or case identity differs')
        means[key] = selection_means(rows)
        metrics = read(folder / 'metrics.json')
        if (not math.isclose(metrics['combined_score'], means[key]['combined_score'], abs_tol=1e-12)
                or not math.isclose(metrics['public']['mean_task_score'], means[key]['task'], abs_tol=1e-12)
                or metrics['public']['escaped'] != means[key]['escaped']
                or metrics['public']['invalid'] != means[key]['invalid'] or metrics['public']['episodes'] != 64):
            raise ValueError('Saved aggregate selection metrics differ')
        transitions += sum(row['steps'] for row in rows)
        for name in ('episodes.json', 'manifest.json', 'metrics.json', 'resource.json'):
            inputs[relative(folder / name)] = sha(folder / name)
    if transitions != execution['transitions']:
        raise ValueError('Selection transition total differs from its episodes')
    winners, programs = [], {}
    for run in nominations['runs']:
        result = {'run_id': run['run_id'], 'arm': run['arm'], 'replicate': run['replicate'],
                  'actual_all_role_calls': run['actual_all_role_calls'], 'common_call_prefix': run['common_call_prefix']}
        for group in ('primary', 'common_prefix_secondary'):
            winner = choose_winner(run[group]['nominations'], means)
            winner['seed_fallback'] = run[group]['seed_fallback']
            result[group] = winner
            programs.setdefault(winner['panel_program'], {'path': winner['frozen_source'], 'sha256': winner['source_sha256']})
        winners.append(result)
    control_protocol = read(CONTROL_PROTOCOL)
    controls = {'original_seed': {'path': 'proposal/initial.py',
                                  'sha256': nominations['runs'][0]['seed']['source_sha256']},
                'memory_pathfinder': control_protocol['programs']['memory_pathfinder']}
    inputs[relative(CONTROL_PROTOCOL)] = sha(CONTROL_PROTOCOL)
    for spec in controls.values():
        if sha(ROOT / spec['path']) != spec['sha256']:
            raise ValueError('Frozen seed/comparator changed')
        key = program_key(spec['sha256'])
        programs.setdefault(key, dict(spec))
        spec['panel_program'] = key
    source_hashes = panel_sources(programs, protocol['source_sha256'])
    design = {'status': 'winners and assessment design frozen; assessment pool undrawn',
              'study': plan['study'], 'phase': 'assessment', 'evaluator': IDENTITY,
              'study_plan': {'path': relative(plan_path), 'sha256': sha(plan_path)},
              'selection_rule': plan['selection']['choose'], 'selection_case_count': 64,
              'selection_pool_sha256': pool['pool_sha256'], 'selection_means': means,
              'winners': winners, 'controls': controls, 'programs': programs,
              'case_count': 512, 'pool_path': plan['assessment']['pool'],
              'assessment_design': plan['assessment'],
              'analysis': {'bootstrap_replicates': 10000, 'bootstrap_seed': 7109512,
                           'level': .95, 'method': 'crossed paired-repetition and shared-case percentile bootstrap',
                           'draw_order': 'common case indices first, then independent paired repetition indices',
                           'repetitions': 3, 'cases': 512, 'equal_repetition_weight': True,
                           'metrics': ['combined_score', 'task', 'model_accuracy', 'final_coverage',
                                       'escaped', 'caught', 'timeout', 'invalid', 'keys', 'door_open', 'steps'],
                           'accuracy_estimand': 'Equal episode means of within-episode correct/audited-cell ratios; no global cell pooling.',
                           'primary': plan['assessment']['primary'],
                           'meaningful_F_difference': plan['assessment']['meaningful_F_difference'],
                           'failure_handling': plan['assessment']['invalids'], 'no_optional_stopping': True,
                           'limits': 'Three paired repetitions give exploratory uncertainty; common-prefix comparison is conditional on realized budgets.'},
              'source_sha256': source_hashes, 'input_sha256': inputs,
              'new_episodes': 0, 'model_calls': 0, 'pools_drawn': 0}
    save_once(output, design)
    return design


def prepare_assessment(freeze_path, plan_path, public, private, minutes, cpu_seconds):
    design, plan = read(freeze_path), read(plan_path)
    if (design['phase'] != 'assessment' or design['case_count'] != 512
            or design['study_plan']['sha256'] != sha(plan_path)
            or design['assessment_design'] != plan['assessment']):
        raise ValueError('Assessment freeze and study plan differ')
    verify_hashes(design['input_sha256'])
    verify_hashes(design['source_sha256'])
    return bind_protocol('assessment', plan_path, plan, public, private,
                         design['programs'], design['source_sha256'], minutes, cpu_seconds,
                         {'assessment_freeze': {'path': relative(freeze_path), 'sha256': sha(freeze_path)},
                          'analysis': design['analysis'], 'winners': design['winners'], 'controls': design['controls']})


def run_panel(protocol_path, private):
    """Only this explicit operation imports the unchanged episode runner."""
    protocol = read(protocol_path)
    phase = protocol['phase']
    if (phase not in PHASE_PURPOSE or protocol['evaluator'] != IDENTITY
            or protocol['case_count'] != {'selection': 64, 'assessment': 512}[phase]
            or Path(private).resolve() != (ROOT / protocol['private_output']).resolve()):
        raise ValueError('Panel phase, size or private execution directory differs')
    verify_hashes(protocol['source_sha256'])
    verify_hashes({protocol['study_plan']['path']: protocol['study_plan']['sha256']})
    evidence = protocol['nominations'] if phase == 'selection' else protocol['assessment_freeze']
    verify_hashes({evidence['path']: evidence['sha256']})
    frozen = read(ROOT / evidence['path'])
    if phase == 'selection':
        if protocol['programs'] != programs_from_nominees(frozen):
            raise ValueError('Selection programs differ from frozen nominations')
    elif any(protocol[key] != frozen[key] for key in ('programs', 'analysis', 'winners', 'controls')):
        raise ValueError('Assessment programs or analysis differ from the winner freeze')
    for spec in protocol['programs'].values():
        if protocol['source_sha256'].get(spec['path']) != spec['sha256']:
            raise ValueError('Candidate is absent from frozen panel source identities')
    budget = protocol['budget']
    if (not 5 < budget['stage_elapsed_minutes'] <= 120
            or budget['experiment_model_calls'] != 0
            or budget['total_evaluator_cpu_seconds'] < 20):
        raise ValueError('Panel execution budget differs from study boundaries')
    public = ROOT / protocol['public_output']
    if (public / 'execution.json').exists() and read(public / 'execution.json')['status'] == 'complete':
        raise ValueError('This panel is complete; do not overwrite its execution record')
    original_argv = sys.argv
    try:
        from proposal.panel import main as execute
        sys.argv = [original_argv[0], '--protocol', str(protocol_path), '--output', str(private)]
        execute()
    finally:
        sys.argv = original_argv
    # Preserve numeric artifacts; only replace the generic runner's purpose.
    for key in protocol['programs']:
        path = public / key / 'manifest.json'
        manifest = read(path)
        manifest['runner_default_purpose'] = manifest['purpose']
        manifest['purpose'] = PHASE_PURPOSE[phase]
        manifest['protocol_sha256'] = sha(protocol_path)
        manifest['metadata_adapter'] = 'proposal/study_panels.py; phase labels only'
        path.write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    for operation in ('prepare-selection', 'prepare-assessment'):
        command = commands.add_parser(operation)
        command.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
        command.add_argument('--public', type=Path, required=True)
        command.add_argument('--private', type=Path, required=True)
        command.add_argument('--minutes', type=float, required=True)
        command.add_argument('--cpu-seconds', type=float, required=True)
    commands.choices['prepare-selection'].add_argument('--nominations', type=Path, default=NOMINATIONS)
    commands.choices['prepare-assessment'].add_argument('--freeze', type=Path, required=True)
    command = commands.add_parser('freeze-winners')
    command.add_argument('--selection', type=Path, required=True)
    command.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    command.add_argument('--output', type=Path, required=True)
    command = commands.add_parser('run')
    command.add_argument('--protocol', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'run':
        run_panel(args.protocol, args.output)
        return
    if args.operation == 'prepare-selection':
        result = prepare_selection(args.nominations, args.plan, args.public, args.private, args.minutes, args.cpu_seconds)
    elif args.operation == 'prepare-assessment':
        result = prepare_assessment(args.freeze, args.plan, args.public, args.private, args.minutes, args.cpu_seconds)
    else:
        result = freeze_winners(args.selection, args.plan, args.output)
    print(json.dumps({'operation': args.operation, 'case_count': result['case_count'],
                      'distinct_programs': len(result['programs']), 'new_episodes': 0,
                      'model_calls': 0, 'pools_drawn': 0}, indent=2))


if __name__ == '__main__':
    main()
