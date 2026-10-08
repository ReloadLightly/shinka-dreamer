"""Compact empirical native/call/resource audit without publishing private pools."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.provenance import sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def usage_from_log(path):
    for line in reversed(Path(path).read_text().splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get('usage'), dict):
            return value['usage']
    return None


def native_tool_audit(campaign, calls):
    """Read only this campaign's native session metadata/tool records, never reasoning."""
    dates = {c['started_utc'][:10].replace('-', '/') for c in calls}
    sessions = []
    for date in sorted(dates):
        directory = Path('/home/roland/.codex/sessions') / date
        for path in sorted(directory.glob('*.jsonl')):
            with path.open() as handle:
                try:
                    first = json.loads(next(handle))
                except (ValueError, StopIteration):
                    continue
                if first.get('type') != 'session_meta' or first.get('payload', {}).get('cwd') != str(campaign):
                    continue
                tools = []
                native_tokens = None
                token_timestamp = None
                for line in handle:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    payload = event.get('payload', {})
                    if event.get('type') == 'event_msg' and payload.get('type') == 'token_count':
                        reported = (payload.get('info') or {}).get('total_token_usage')
                        if reported:
                            native_tokens = {k: reported.get(k, 0) for k in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens', 'total_tokens')}
                            token_timestamp = event.get('timestamp')
                    if str(payload.get('type', '')).endswith('_call'):
                        content = payload.get('arguments') or payload.get('input') or payload.get('action') or ''
                        if not isinstance(content, str):
                            content = json.dumps(content, sort_keys=True)
                        tools.append({'name': payload.get('name', payload.get('type')), 'input_excerpt': content[:1200],
                            'input_sha256': hashlib.sha256(content.encode()).hexdigest(),
                            'input_characters': len(content),
                            'protected_pool_path_mentioned': bool(re.search(
                                r'results/private|v3-selection-seeds|v3-fit-training|v3-assessment-seeds', content))})
                sessions.append({'session_id': first['payload'].get('id'), 'session_timestamp': first['payload'].get('timestamp'),
                                 'tool_calls': tools, 'tool_call_count': len(tools),
                                 'latest_native_token_report': native_tokens, 'latest_native_token_timestamp': token_timestamp})
    return {'scope': 'Only native sessions whose recorded cwd equals this campaign; no assistant reasoning exported',
            'sessions': sessions, 'observed_tool_calls': sum(s['tool_call_count'] for s in sessions),
            'protected_pool_path_mentions': sum(t['protected_pool_path_mentioned'] for s in sessions for t in s['tool_calls']),
            'limitation': 'Observed tool requests are audited; read-only Headless policy is not a filesystem isolation guarantee.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default='results/campaign-v3')
    parser.add_argument('--out', default='artifacts/campaign-v3/native-audit.json')
    args = parser.parse_args()
    campaign, out = Path(args.campaign).resolve(), Path(args.out)
    with sqlite3.connect(f'file:{campaign / "programs.sqlite"}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('select * from programs order by generation,timestamp,id')]
    resolved = read_json(campaign / 'dreamer-resolved.json')
    sampling_records = [read_json(p) for p in sorted(campaign.glob('gen_*/sampling-*.json'))]
    by_id = {r['id']: r for r in rows}
    unique = {r['generation']: r for r in reversed(rows)}
    request_attempts = {}
    patch_attempts = []
    request_evidence = []
    for metadata in sorted(campaign.glob("gen_*/attempts/**/metadata.json")):
        patch_attempts.append(read_json(metadata))
    for attempt in sorted(campaign.glob('gen_*/attempts/**/headless_prompt.md')):
        generation = int(attempt.parts[attempt.parts.index(campaign.name) + 1][4:])
        metadata_path = attempt.parent / 'metadata.json'
        if metadata_path.exists():
            request_attempts.setdefault(sha256(attempt), []).append(read_json(metadata_path))
    for request in sorted(campaign.glob('gen_*/sampling-*.json')):
        sampling = read_json(request)
        generation = sampling['generation']
        row = by_id[sampling['parent_id']]
        parts = request.stem.split('-')
        prompts = list((campaign / f'gen_{generation}/attempts').glob(
            f'novelty_{parts[-2]}/resample_{parts[-1]}/patch_*/headless_prompt.md'))
        for prompt in prompts:
            value = prompt.read_text()
            request_evidence.append({'generation': generation, 'request_sha256': sha256(prompt),
                'parent_id': row['id'], 'parent_island_when_sampled': sampling['parent_island'],
                'parent_code_present': row['code'] in value,
                'task_present': 'Namazu unknown-enemy-dynamics v3' in value,
                'string_feedback': sampling['parent_feedback_is_string'],
                'feedback_present': sampling['parent_text_feedback'] in value,
                'archive_inspiration_ids': sampling['archive_inspiration_ids'],
                'top_k_inspiration_ids': sampling['top_k_inspiration_ids'],
                'inspiration_code_presence': {i: by_id[i]['code'] in value for i in
                    sampling['archive_inspiration_ids'] + sampling['top_k_inspiration_ids']},
                'restored_or_current_recommendation_present': bool(sampling['recommendation']) and sampling['recommendation'] in value,
                'uses_corrected_keys_description': 'keys is the raw number' in value})
    calls = []
    for path in sorted((campaign / 'model-calls').glob('*.json')):
        call = read_json(path)
        cmd = call['command']
        prompt = Path(cmd[cmd.index('--prompt-file')+1])
        text = prompt.read_text() if prompt.exists() else ''
        candidates = request_attempts.get(call['prompt_sha256'], [])
        started = datetime.fromisoformat(call['started_utc']).timestamp()
        candidates = [a for a in candidates if datetime.fromisoformat(a['timestamp']).timestamp() >= started]
        attempt = min(candidates, key=lambda a: datetime.fromisoformat(a['timestamp']).timestamp()) if candidates else None
        generation = attempt['generation'] if attempt else None
        role = 'mutation_or_repair' if generation is not None or 'Namazu unknown-enemy-dynamics v3' in text else 'unknown'
        if 'analyzing an individual program' in text[:800]: role = 'meta_program_summary'
        elif 'extract actionable optimization insights' in text[:800]: role = 'meta_global_insights'
        elif 'generating actionable recommendations' in text[:800]: role = 'meta_recommendations'
        calls.append({k: v for k,v in call.items() if k != 'command'} | {'role': role, 'generation': generation,
            'patch_attempt': attempt.get('patch_attempt') if attempt else None,
            'resample_attempt': attempt.get('resample_attempt') if attempt else None,
            'patch_application_success': attempt.get('success') if attempt else None})
    probes = [{'source': p.name, 'usage': usage_from_log(p)} for p in sorted(campaign.glob('subscription-*.log'))]
    usage = [c['usage'] for c in calls if c.get('usage')] + [p['usage'] for p in probes if p['usage']]
    routes = Counter(a.get('route', 'missing') for u in usage for a in u.get('billing', {}).get('attempts', []))
    episodes = []
    for generation, row in unique.items():
        metadata = json.loads(row['metadata'] or '{}')
        directory = Path(metadata.get('recovery_results_dir', campaign / f'gen_{generation}/results'))
        path = directory / 'episodes.json'
        if path.exists(): episodes.extend(read_json(path))
    migrations = [{'id': r['id'], 'generation': r['generation'], 'events': json.loads(r['migration_history'] or '[]')}
                  for r in rows if json.loads(r['migration_history'] or '[]')]
    meta_paths = sorted((campaign / 'meta').glob('meta_*.txt'))
    audit = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'campaign_sha256': sha256(campaign / 'campaign-manifest.json'),
        'objective_name_note': 'The mutation prompt calls the unchanged formula absolute-task-v3; the canonical evaluator identity is absolute-task-v1 within namazu-unknown-dynamics-v3. This is a naming alias, not an objective change.',
        'normal_checkpoint_requests': [read_json(p) for p in sorted(campaign.glob('checkpoint-request-*.json'))],
        'transport_compatibility_amendment': read_json(campaign / 'transport-amendment.json') if (campaign / 'transport-amendment.json').exists() else None,
        'recommendation_resume_evidence': read_json(ROOT / 'artifacts/campaign-v3/recommendation-resume-verification.json') if (ROOT / 'artifacts/campaign-v3/recommendation-resume-verification.json').exists() else None,
        'total_slots_budget': 50, 'persisted_slots': len(unique), 'native_rows_including_island_seed_copies': len(rows),
        'seed_slots': int(0 in unique), 'valid_descendants': sum(g>0 and bool(r['correct']) for g,r in unique.items()),
        'failed_slots': [g for g,r in unique.items() if not r['correct']],
        'saved_evaluation_episodes': len(episodes), 'invalid_saved_episodes': sum(bool(e['error']) for e in episodes),
        'model_calls': calls, 'probe_calls': probes, 'native_model_requests': len(calls),
        'all_model_requests_including_probes': len(calls)+len(probes),
        'model_requests_by_role': dict(Counter(c['role'] for c in calls)),
        'model_status_counts': dict(Counter(c['status'] for c in calls)),
        'model_nonzero_returncodes': sum(c.get('returncode', 0) != 0 for c in calls),
        'patch_attempts': patch_attempts,
        'patch_attempt_count': len(patch_attempts),
        'patch_retry_attempts': sum(p.get('patch_attempt', 1) > 1 for p in patch_attempts),
        'patch_application_failures': sum(not p.get('success', False) for p in patch_attempts),
        'parent_resamples': len({(p['generation'], p.get('novelty_attempt'), p.get('resample_attempt')) for p in patch_attempts if p.get('resample_attempt', 1) > 1}),
        'native_fix_sample_events': sum(read_json(p)['mode'] == 'fix' for p in campaign.glob('gen_*/sampling-*.json')),
        'native_novelty_rejections': 0,
        'billing_attempt_routes': dict(routes),
        'reported_tokens': {key: sum(u.get(key,0) for u in usage) for key in
            ('inputTokens','cacheReadTokens','cacheWriteTokens','outputTokens','reasoningOutputTokens','totalTokens')},
        'cost_note': 'All dollar figures in native logs are API-list-price estimates, not actual subscription charges.',
        'resource_scope_note': 'Model elapsed seconds sum recorded native calls and exclude readiness probes, whose durations were not separately captured. CPU and episode wall totals cover saved search evaluations. Remote model CPU and complete controller-process CPU were not measured.',
        'resources': {'model_elapsed_seconds': sum(c.get('elapsed_seconds',0) for c in calls),
                      'candidate_cpu_seconds': sum(e.get('candidate_cpu_seconds',0) for e in episodes),
                      'evaluator_cpu_seconds': sum(e.get('evaluator_cpu_seconds',0) for e in episodes),
                      'episode_wall_seconds': sum(e.get('seconds',0) for e in episodes)},
        'features': {'native_islands': {'configured':4, 'rows_observed_by_island':dict(Counter(r['island_idx'] for r in rows))},
                     'native_migration': {'configured_interval':10, 'observed':migrations},
                     'inspirations': {'configured_archive':1,'configured_top_k':1,
                                      'sampled_nonempty_requests':sum(bool(e['archive_inspiration_ids'] or e['top_k_inspiration_ids']) for e in request_evidence),
                                      'sampled_archive_contexts':sum(bool(e['archive_inspiration_ids']) for e in sampling_records),
                                      'sampled_top_k_contexts':sum(bool(e['top_k_inspiration_ids']) for e in sampling_records)},
                     'native_sampling': {'configured_parent_strategy':resolved['database']['parent_selection_strategy'],
                                         'configured_island_strategy':resolved['database']['island_selection_strategy'],
                                         'sampled_parent_islands':dict(Counter(str(e['parent_island']) for e in sampling_records)),
                                         'sampling_contexts':len(sampling_records)},
                     'mutation_operators': {'configured':resolved['evolution']['patch_types'],
                                            'configured_probabilities':resolved['evolution']['patch_type_probs'],
                                            'observed_slot_operators':dict(Counter(json.loads(r['metadata'] or '{}').get('patch_type','unknown') for g,r in unique.items() if g>0))},
                     'recommendations': {'configured_interval':10,'outputs':[{'path':p.name,'sha256':sha256(p)} for p in meta_paths],
                                         'requests_with_recommendation':sum(e['restored_or_current_recommendation_present'] for e in request_evidence)},
                     'novelty': 'inactive: embeddings and novelty judge explicitly disabled; no novelty claim',
                     'model_bandit': 'disabled: one effective model/effort configuration',
                     'prompt_evolution': 'disabled','evaluator_llm': 'none'},
        'request_evidence':request_evidence,
        'native_tool_activity':native_tool_audit(campaign, calls),
        'upstream_revision':'9912af12d423504b8d580f4179fd15f5f88b8c50',
        'headless_revision':'93cd9b06b85f848af1308c41e018991b33907c5e',
        'effective_model':'gpt-6-astra','effective_effort':'high',
        'not_forwarded_codex_controls':['temperature','max_tokens']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ('persisted_slots','valid_descendants','failed_slots','saved_evaluation_episodes',
                     'all_model_requests_including_probes','model_requests_by_role','billing_attempt_routes','resources')},indent=2))


if __name__ == '__main__':
    main()
