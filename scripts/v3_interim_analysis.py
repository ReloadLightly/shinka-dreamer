"""Analyze the administratively paused v3 checkpoint; never run episodes/models.

This additive wrapper leaves the frozen full-assessment protocol/statistics intact.
Its nominal intervals are descriptive interim estimates, not confirmatory results.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from dreamer.provenance import sha256
from scripts.v3_analysis import analyze, analyze_matched, write_jsonlines_gzip
from scripts.v3_assessment import verify_plan

DEFAULT = ROOT / 'artifacts/campaign-v3/assessment-interim140'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def verify_public_sources(plan):
    """Public analysis reproduction needs no private pools or evaluator host."""
    for category in ('analysis_files', 'design_files'):
        for name, digest in plan[category].items():
            if Path(name).parts[:2] == ('results', 'private'):
                continue
            if sha256(ROOT / name) != digest:
                raise ValueError('Frozen public source changed: ' + name)
    for condition in plan['conditions']:
        if sha256(ROOT / condition['program_path']) != condition['program_sha256']:
            raise ValueError('Frozen condition source changed')


def check_public(value):
    """Reject private inputs recursively; compact audits contain hashes, not traces."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'seed', 'seeds', 'trace', 'laws', 'law', 'environment_seed',
                       'initial_law', 'replacement_law', 'current_law'}:
                raise ValueError('Private field in compact evidence: ' + key)
            check_public(item)
    elif isinstance(value, list):
        for item in value:
            check_public(item)


def load_raw(raw, protocol, plan):
    if read(raw / 'frozen-plan.json') != plan:
        raise ValueError('Raw plan differs from original preregistration')
    manifest, registration = read(raw / 'manifest.json'), read(raw / 'pool-manifest.json')
    if (manifest['plan_sha256'] != protocol['original_plan']['sha256']
            or registration['original_plan_sha256'] != protocol['original_plan']['sha256']
            or registration['frozen_plan_sha256'] != sha256(raw / 'frozen-plan.json')
            or manifest['episode_pool'] != registration['pool']
            or manifest['evaluation'] != plan['evaluation']
            or manifest['conditions'] != plan['conditions']
            or manifest['regimes'] != plan['regimes']):
        raise ValueError('Raw evaluator/pool/condition identity mismatch')
    records = {}
    for folder, field in [('cases', 'case_file_sha256'), ('matched', 'matched_file_sha256')]:
        expected = protocol[field]
        if {p.name for p in (raw / folder).glob('*.json')} != set(expected):
            raise ValueError('Checkpoint file coverage changed: ' + folder)
        records[folder] = []
        for name, digest in sorted(expected.items()):
            path = raw / folder / name
            if sha256(path) != digest:
                raise ValueError('Checkpoint bytes changed: ' + name)
            record = read(path)
            if folder == 'cases':
                if record['case'] != int(Path(name).stem):
                    raise ValueError('Case filename/content mismatch')
                if any(row['case'] != record['case'] for row in record['conditions']):
                    raise ValueError('Nested case identity mismatch')
                records[folder].extend(record['conditions'])
            else:
                records[folder].append(record)
    return records['cases'], records['matched']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--raw', type=Path, default=ROOT / 'results/campaign-v3-assessment')
    parser.add_argument('--compact', action='store_true', help='Recompute from committed compact evidence')
    args = parser.parse_args()
    start, cpu = time.monotonic(), time.process_time()
    protocol_path = args.out / 'interim-protocol.json'
    protocol = read(protocol_path)
    for key in ('original_plan', 'checkpoint', 'original_analysis'):
        reference = protocol[key]
        if sha256(ROOT / reference['path']) != reference['sha256']:
            raise ValueError('Recorded source changed: ' + key)
    plan = read(ROOT / protocol['original_plan']['path'])
    if args.compact:
        verify_public_sources(plan)
    else:
        verify_plan(plan)
    checkpoint = read(ROOT / protocol['checkpoint']['path'])
    if checkpoint['case_file_sha256'] != protocol['case_file_sha256']:
        raise ValueError('Checkpoint scope differs from recorded interim scope')
    if protocol['observed_case_indices'] != list(range(140)) or plan['sample_size'] != 1536:
        raise ValueError('This wrapper is specific to the recorded 140/1536 checkpoint')
    paths = [args.out / name for name in ('episodes.jsonl.gz', 'matched-episodes.jsonl.gz')]
    if args.compact:
        closure = read(args.out / 'interim-analysis-closure.json')
        if (closure['protocol_sha256'] != sha256(protocol_path)
                or closure['original_plan_sha256'] != protocol['original_plan']['sha256']
                or closure['analysis_source_sha256'] != protocol['original_analysis']['sha256']
                or closure['observed_cases'] != 140
                or closure['registered_cases'] != 1536
                or closure['full_assessment_complete'] is not False):
            raise ValueError('Compact closure scope/source identity mismatch')
        for path in paths:
            if sha256(path) != closure['files'][path.name]:
                raise ValueError('Compact input changed')
        def compressed(path):
            with gzip.open(path, 'rt') as handle:
                return [json.loads(line) for line in handle]
        rows, matched = map(compressed, paths)
    else:
        rows, matched = load_raw(args.raw, protocol, plan)
    sources = {c['name']: c for c in plan['conditions']}
    for row in rows:
        source = sources[row['condition']]
        if row['program_sha256'] != source['program_sha256'] or row['variant'] != source['variant']:
            raise ValueError('Outcome source/variant mismatch')
    for record in matched:
        if (record['source_sha256'] != plan['matched']['source_sha256']
                or record['selected_source_sha256'] != plan['matched']['selected']['source_sha256']):
            raise ValueError('Matched source mismatch')
    check_public(rows)
    check_public(matched)
    rows.sort(key=lambda r: (r['case'], plan['regimes'].index(r['regime']), r['condition']))
    matched.sort(key=lambda r: (r['case'], plan['regimes'].index(r['regime'])))
    projection = copy.deepcopy(plan)
    projection.update(sample_size=140, study=protocol['study'])
    result = analyze(rows, projection)
    passive = analyze_matched(matched, projection, policy_rows=rows)
    metadata = {'observed_cases': 140, 'registered_cases': 1536,
                'full_assessment_complete': False, 'simulation_resume_authorized': False,
                'scope': protocol['uncertainty'], 'protocol_sha256': sha256(protocol_path),
                'new_world_episodes': 0, 'new_experiment_model_calls': 0}
    result['interim'] = passive['interim'] = metadata
    result['selected_matched']['interim'] = metadata
    for path, data in zip(paths, (rows, matched)):
        write_jsonlines_gzip(path, data)
    write(args.out / 'analysis.json', result)
    write(args.out / 'matched-analysis.json', passive)
    files = paths + [args.out / 'analysis.json', args.out / 'matched-analysis.json']
    write(args.out / 'interim-analysis-closure.json', {
        **metadata, 'status': 'completed saved-data interim analysis; original assessment paused',
        'original_plan_sha256': protocol['original_plan']['sha256'],
        'analysis_source_sha256': protocol['original_analysis']['sha256'],
        'wrapper_sha256': sha256(__file__),
        'condition_episodes': len(rows), 'recorded_policy_episodes': len(matched),
        'shadow_predictor_passes': sum(len(r['shadows']) for r in matched),
        'files': {p.name: sha256(p) for p in files},
        'analysis_cpu_seconds': time.process_time() - cpu,
        'analysis_elapsed_seconds': time.monotonic() - start})
    print(json.dumps({'cases': 140, 'worlds': len(rows), 'recorded_episodes': len(matched),
                      'new_episodes': 0, 'new_experiment_model_calls': 0}))


if __name__ == '__main__':
    main()
