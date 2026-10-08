"""Select one frozen v3 program after exactly 50 completed native candidate slots.

Top-five distinct valid sources are chosen by development score, then assessed
on the separate 64-case selection pool. This script makes no model calls and
never exposes selection metrics to the evolutionary controller.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.evaluation_v3 import aggregate, evaluation_identity
from dreamer.provenance import pool, record, sha256
from dreamer.world_v3 import REGIMES
from scripts.recover_campaign import campaign_lock
from scripts.v3_assessment import atomic_create, digest, run_cases


def candidates_after_completion(database, total_slots=50, limit=5):
    with sqlite3.connect(f'file:{Path(database).resolve()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        programs = [dict(row) for row in db.execute('select id, generation, code, correct, combined_score from programs')]
    slots = {p['generation'] for p in programs}
    if slots != set(range(total_slots)):
        raise ValueError(f'Selection requires all {total_slots} completed slots; found {len(slots)}')
    # Island seed clones share slot 0; no other slot may contain distinct code.
    for slot in slots:
        if len({p['code'] for p in programs if p['generation'] == slot}) != 1:
            raise ValueError(f'Ambiguous candidate source in slot {slot}')
    valid = [p for p in programs if p['correct'] and isinstance(p['combined_score'], (int, float))
             and math.isfinite(p['combined_score'])]
    valid.sort(key=lambda p: (-p['combined_score'], p['generation'], p['id']))
    unique, seen = [], set()
    for program in valid:
        source_hash = hashlib.sha256(program['code'].encode()).hexdigest()
        if source_hash not in seen:
            program['program_sha256'] = source_hash
            unique.append(program)
            seen.add(source_hash)
        if len(unique) == limit:
            break
    if not unique:
        raise ValueError('No valid source is available for selection')
    return unique, {'total_slots': total_slots,
                    'valid_slots': len({p['generation'] for p in programs if p['correct']}),
                    'failed_slots': len({p['generation'] for p in programs if not p['correct']}),
                    'native_rows_including_seed_clones': len(programs),
                    'inventory_sha256': digest(sorted([{k: p[k] for k in ('id', 'generation', 'correct', 'combined_score')}
                        | {'source_sha256': hashlib.sha256(p['code'].encode()).hexdigest()} for p in programs], key=lambda p: p['id']))}


def immutable_source(path, source):
    path = Path(path)
    if path.exists():
        if path.read_text() != source:
            raise ValueError(f'Frozen source changed: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            handle.write(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default='results/campaign-v3')
    parser.add_argument('--seed-file', default='results/private/v3-selection-seeds.json')
    parser.add_argument('--out', default='results/campaign-v3-selection')
    parser.add_argument('--publish', default='artifacts/campaign-v3/selection')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    campaign, out, publish = Path(args.campaign).resolve(), Path(args.out).resolve(), Path(args.publish).resolve()
    with campaign_lock(campaign):
        campaign_manifest = json.loads((campaign / 'campaign-manifest.json').read_text())
        if campaign_manifest['evaluation'] != evaluation_identity() or campaign_manifest['target_generation_slots'] != 50:
            raise ValueError('Selection requires the frozen 50-slot v3 campaign')
        candidates, inventory = candidates_after_completion(campaign / 'programs.sqlite')
        seeds, info = pool(args.seed_file, 64)
        splits_path = ROOT / 'artifacts/campaign-v3/splits.json'
        splits = json.loads(splits_path.read_text())
        if info['sha256'] != splits['selection']['pool_sha256']:
            raise ValueError('Selection pool differs from the prespecified split')
        out.mkdir(parents=True, exist_ok=True)
        conditions = []
        for candidate in candidates:
            source = out / 'sources' / f'slot-{candidate["generation"]:02d}.py'
            immutable_source(source, candidate['code'])
            conditions.append({'name': f'slot_{candidate["generation"]:02d}', 'program_path': str(source),
                               'program_sha256': candidate['program_sha256'], 'variant': 'predictive'})
        manifest = {'split': 'selection-validation; selected results remain selection-biased',
                    'campaign_sha256': sha256(campaign / 'campaign-manifest.json'), 'inventory': inventory,
                    'evaluation': evaluation_identity(), 'pool': info, 'regimes': list(REGIMES),
                    'candidates': [{k: p[k] for k in ('id', 'generation', 'combined_score', 'program_sha256')} for p in candidates],
                    'selection_rule': 'mean absolute task, then escape count, then lower candidate slot',
                    'driver_sha256': sha256(__file__), 'checkpoint_driver_sha256': sha256(ROOT / 'scripts/v3_assessment.py'),
                    'splits_sha256': sha256(splits_path), 'model_calls': 0}
        record(out / 'manifest.json', manifest)
        rows = run_cases(out, seeds, conditions, list(REGIMES), args.workers)
        rankings = []
        for candidate, condition in zip(candidates, conditions):
            subset = [row for row in rows if row['condition'] == condition['name']]
            metrics = aggregate(subset)
            rankings.append({'slot': candidate['generation'], 'native_id': candidate['id'],
                'program_sha256': candidate['program_sha256'], 'development_task': candidate['combined_score'],
                'selection_task': metrics['combined_score'],
                'selection_escapes': sum(row['reason'] == 'escaped' for row in subset),
                'episodes': len(subset), 'invalid': sum(row['error'] is not None for row in subset),
                'public': metrics['public']})
        rankings.sort(key=lambda p: (-p['selection_task'], -p['selection_escapes'], p['slot']))
        winner = rankings[0]
        source = next(p['code'] for p in candidates if p['program_sha256'] == winner['program_sha256'])
        selection = {'selected': winner, 'ranking': rankings,
                     'selected_source_sha256': winner['program_sha256'], 'native_id': winner['native_id'],
                     'generation': winner['slot'], 'selection_manifest_sha256': sha256(out / 'manifest.json'),
                     'case_checkpoint_sha256': sha256(out / 'execution-complete.json'),
                     'case_count': 64, 'regimes': list(REGIMES), 'candidate_count': len(candidates),
                     'condition_episodes': len(rows), 'selection_biased': True, 'inventory': inventory}
        immutable_source(out / 'selected.py', source)
        record(out / 'selection.json', selection)
        immutable_source(publish / 'selected.py', source)
        record(publish / 'selection.json', selection)
        print(json.dumps(selection, indent=2), flush=True)


if __name__ == '__main__':
    main()
