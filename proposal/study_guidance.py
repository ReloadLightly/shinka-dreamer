"""Export intended native guidance outputs after terminal context-audit checks."""
import argparse
from contextlib import closing
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3

from proposal.study_search_report import raw_tool_summary

ROOT = Path(__file__).resolve().parents[1]
PROMPT_FIELDS = ('id', 'parent_id', 'generation', 'program_generation', 'name', 'description',
                 'patch_type', 'prompt_text', 'program_count', 'correct_program_count', 'fitness')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if not campaign.is_relative_to(ROOT / 'results') or not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Use repository runtime input and public artifact output paths')
    inputs = {}

    def saved(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    with (campaign / 'controller.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = saved(campaign / 'execution-resources.json')
        if not execution.get('ended_utc') or execution.get('returncode') is None:
            raise ValueError('Controller must have a terminal execution record')
        config = saved(campaign / 'campaign-manifest.json')
        if execution['manifest_sha256'] != sha(campaign / 'campaign-manifest.json'):
            raise ValueError('Terminal controller and configuration identities differ')
        calls = [saved(path) for path in sorted((campaign / 'model-calls').glob('*.json'))]
        audit_rows = []
        for call in calls:
            if (call.get('status') == 'started' or call.get('mutation_context_validated') is not True
                    or call.get('context_audit_failure') or call.get('billing_violation') or call.get('model_mismatch')):
                raise ValueError('All recorded model requests must pass their context and route audits')
            stem = call['id']
            if Path(stem).name != stem:
                raise ValueError('Invalid audit identifier')
            summary = saved(campaign / 'codex-event-audit' / (stem + '.summary.json'))
            raw = campaign / 'codex-event-audit' / (stem + '.events.jsonl')
            raw_hash = sha(raw)
            inputs[str(raw.relative_to(ROOT))] = raw_hash
            if (summary.get('unauthorized_tool_event') or summary.get('raw_events_file') != raw.name
                    or summary.get('raw_events_sha256') != raw_hash
                    or call.get('codex_tool_audit_outcome') != summary):
                raise ValueError('Recorded/raw context audit identities differ')
            audit_rows.append({'id': stem, 'role': call['role'], 'raw_events_sha256': raw_hash,
                               'context_validated': True})
        audit = raw_tool_summary(campaign, calls)
        if (audit['model_call_ids_without_completed_audit'] or audit['audit_ids_without_model_call']
                or audit['observed_tool_event_records'] or audit['wrapper_rejections']
                or any(row['partial_final_lines'] or row['event_type_counts'].get('invalid-json')
                       or row['event_type_counts'].get('non-object') for row in audit['calls'])):
            raise ValueError('Observed raw event records do not satisfy the context boundary')
        meta_path = campaign / 'run1-meta-state.json'
        meta = saved(meta_path) if meta_path.exists() else {}
        history = list(dict.fromkeys(meta.get('meta_recommendations_history') or []))
        current = meta.get('meta_recommendations')
        if current and current not in history:
            history.append(current)
        recommendations = [{'sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': text} for text in history]
        prompts, database = [], campaign / 'prompts.sqlite'
        if database.exists():
            with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                for source in connection.execute('SELECT * FROM system_prompts ORDER BY generation, id'):
                    source = dict(source)
                    row = {name: source.get(name) for name in PROMPT_FIELDS}
                    row['kind'] = 'original supplied task prompt' if row['generation'] == 0 else 'evolved model-produced prompt'
                    row['program_ids'] = json.loads(source.get('program_ids') or '[]')
                    row['prompt_sha256'] = hashlib.sha256(row['prompt_text'].encode()).hexdigest()
                    prompts.append(row)
            inputs[str(database.relative_to(ROOT))] = sha(database)
        payload = {'run_id': config['run_id'], 'arm': config['arm'],
                   'scope': 'Intended model-produced search guidance, not validated scientific findings. Native global-insight output is an explicit requested artifact, not a private reasoning channel. Descriptions may use local context program numbering.',
                   'excluded': 'No unprocessed program objects, arbitrary metadata, provider streams, hidden reasoning, authentication data or assessment cases.',
                   'meta': {'processed_program_count': meta.get('total_programs_meta_processed', 0),
                            'pending_program_count': len(meta.get('unprocessed_programs') or []),
                            'generated_program_summaries': meta.get('meta_summary'),
                            'generated_global_insights': meta.get('meta_scratch_pad'),
                            'recommendation_history': recommendations,
                            'current_recommendation_sha256': hashlib.sha256(current.encode()).hexdigest() if current else None},
                   'prompts': prompts, 'request_context_audits': audit_rows,
                   'input_sha256': inputs, 'source_sha256': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                        'proposal/study_search_report.py': sha(ROOT / 'proposal/study_search_report.py')},
                   'reproduction': f'OPENBLAS_NUM_THREADS=1 .venv/bin/python -m proposal.study_guidance --campaign {campaign.relative_to(ROOT)} --output {output.relative_to(ROOT)}',
                   'new_episodes': 0, 'model_calls': 0}
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if output.exists() and output.read_text() != text:
            raise ValueError('Refusing to change an existing guidance export')
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x') as handle:
                handle.write(text)
    return {'output': str(output.relative_to(ROOT)), 'bytes': output.stat().st_size,
            'prompts': len(prompts), 'recommendation_batches': len(recommendations),
            'context_validated_requests': len(calls), 'sha256': sha(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.campaign, args.output), indent=2))


if __name__ == '__main__':
    main()
