"""Export RUN1 native evidence from saved state; no model calls or episodes.

The controller owns every runtime file. This reporting script opens its databases
read-only and writes only compact public artifacts. A live export is a timestamped
snapshot, not a claim that side effects or an active request have completed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def clean(value):
    """Native unobserved bandit extrema use Infinity; publish strict JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + '\n')


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def provenance(path):
    data = path.read_bytes()
    return {'path': relative(path), 'sha256': digest(data), 'bytes': len(data)}


def public_text(path, text):
    # Tripwires supplement a narrow source allowlist; they are not a security proof.
    sensitive = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                 r'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}',
                 r'"(?:access_token|refresh_token|id_token)"\s*:\s*"[^"\s]+"',
                 r'Authorization\s*:\s*Bearer\s+[A-Za-z0-9_.-]{16,}']
    if any(re.search(pattern, text) for pattern in sensitive):
        raise ValueError('Credential-pattern tripwire in ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def rows(path, table):
    if not path.exists():
        return []
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        if not db.execute('select 1 from sqlite_master where type="table" and name=?', (table,)).fetchone():
            return []
        return [dict(row) for row in db.execute('select * from ' + table)]


def json_field(row, key, default):
    value = row.get(key)
    return json.loads(value) if isinstance(value, str) and value else (value or default)


def jsonl(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    # A concurrently appended final partial line is not a completed event.
    lines = data.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n')]


def prompt_context(text):
    marker = '# Previous Messages\n\n'
    if marker not in text:
        return text, False
    encoded = text.split(marker, 1)[1].split('\n\n# User Request', 1)[0]
    history = json.loads(encoded)
    return text + '\n' + '\n'.join(m['content'] for m in history if isinstance(m.get('content'), str)), True


def totals(calls):
    result = {'calls': len(calls), 'status': dict(Counter(c['status'] for c in calls)),
              'successful_process_returns': sum(c.get('status') == 'completed' and c.get('returncode') == 0 for c in calls),
              'unsuccessful_process_returns': sum(c.get('status') == 'completed' and c.get('returncode', 0) != 0 for c in calls),
              'timeout_process_returns': sum(c.get('returncode') == 124 for c in calls),
              'remote_elapsed_seconds': sum(c.get('elapsed_seconds', 0) for c in calls),
              'tokens': {key: 0 for key in ('inputTokens', 'cacheReadTokens', 'cacheWriteTokens',
                                          'outputTokens', 'reasoningOutputTokens', 'totalTokens')},
              'completed_without_reported_usage': 0, 'billing_routes': {},
              'api_list_price_estimate_usd': 0.0}
    routes = Counter()
    for c in calls:
        usage = c.get('usage') or {}
        if c['status'] != 'started' and usage.get('usageStatus') != 'reported':
            result['completed_without_reported_usage'] += 1
        for key in result['tokens']:
            result['tokens'][key] += usage.get(key, 0) or 0
        routes.update(a.get('route', 'missing') for a in usage.get('billing', {}).get('attempts', []))
        result['api_list_price_estimate_usd'] += (usage.get('cost') or {}).get('total', 0) or 0
    result['billing_routes'] = dict(routes)
    result['uncached_input_plus_output_tokens'] = result['tokens']['inputTokens'] + result['tokens']['outputTokens']
    result['reported_usage_calls'] = sum((c.get('usage') or {}).get('usageStatus') == 'reported' for c in calls)
    result['token_total_exact_for_all_admitted_calls'] = result['reported_usage_calls'] == len(calls)
    result['token_scope'] = 'Sum over reported calls only; a lower bound when any admitted call has unavailable usage. Missing usage is not zero.'
    return result


def codex_metadata(campaign, calls):
    """Allowlist session metadata only: never retain reasoning or message content."""
    sessions = []
    dates = {c['started_utc'][:10].replace('-', '/') for c in calls}
    for date in sorted(dates):
        directory = Path('/home/roland/.codex/sessions') / date
        for path in sorted(directory.glob('*.jsonl')):
            with path.open() as handle:
                try:
                    first = json.loads(next(handle))
                except (ValueError, StopIteration):
                    continue
                info = first.get('payload', {})
                if first.get('type') != 'session_meta' or info.get('cwd') != str(campaign):
                    continue
                contexts, tool_counts = [], Counter()
                tokens = None
                token_time = None
                last_time = first.get('timestamp')
                for line in handle:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue  # An active session may have a final partial line.
                    payload = event.get('payload', {})
                    last_time = event.get('timestamp', last_time)
                    if event.get('type') == 'turn_context':
                        context = {k: payload.get(k) for k in ('model', 'effort')}
                        if context not in contexts:
                            contexts.append(context)
                    if event.get('type') == 'response_item' and payload.get('type') in ('function_call', 'custom_tool_call'):
                        tool_counts[payload.get('name', payload['type'])] += 1
                    if event.get('type') == 'event_msg' and payload.get('type') == 'token_count':
                        reported = (payload.get('info') or {}).get('total_token_usage')
                        if reported:
                            tokens = {k: reported.get(k, 0) for k in ('input_tokens', 'cached_input_tokens',
                                'cache_write_input_tokens', 'output_tokens', 'reasoning_output_tokens', 'total_tokens')}
                            token_time = event.get('timestamp')
                started = datetime.fromisoformat(info['timestamp'].replace('Z', '+00:00'))
                candidates = [c for c in calls if datetime.fromisoformat(c['started_utc']) <= started and
                    (not c.get('ended_utc') or started <= datetime.fromisoformat(c['ended_utc']))]
                matched = max(candidates, key=lambda c: c['started_utc']) if candidates else None
                sessions.append({'session_id': info.get('id'), 'session_started_utc': info.get('timestamp'),
                    'last_event_utc': last_time, 'model_effort_contexts': contexts,
                    'tool_call_count': sum(tool_counts.values()), 'tool_calls_by_name': dict(tool_counts),
                    'latest_native_token_report': tokens, 'latest_token_report_utc': token_time,
                    'call_id': matched['id'] if matched else None,
                    'alignment': 'session start inside subprocess call interval; latest admission if nested' if matched else 'unmatched'})
    return {'sessions': sessions, 'observed_tool_calls': sum(s['tool_call_count'] for s in sessions),
        'scope': 'Only sessions whose session_meta.cwd exactly equals this campaign. Allowlisted metadata, context model/effort, tool counts and usage totals; no messages, tool arguments, reasoning or raw rollouts exported.',
        'token_note': 'Native input_tokens includes cached input; Headless inputTokens excludes cache. This is corroboration of the same calls, never additional token usage.'}


def run(campaign, out):
    now = datetime.now(timezone.utc).isoformat()
    resolved = read(campaign / 'dreamer-resolved.json', {})
    events = jsonl(campaign / 'native-events.jsonl')
    sampling = [e for e in events if e['kind'] == 'sampling']
    programs = sorted(rows(campaign / 'programs.sqlite', 'programs'),
                      key=lambda r: (r['generation'], r['timestamp'], r['id']))
    by_id = {r['id']: r for r in programs}
    canonical = {}
    for row in programs:
        meta = json_field(row, 'metadata', {})
        if not meta.get('_is_island_copy'):
            if row['generation'] in canonical:
                raise ValueError('Multiple nonadministrative rows for one slot')
            canonical[row['generation']] = row
    lineage = []
    for row in programs:
        meta = json_field(row, 'metadata', {})
        source = row['code'] or ''
        path = out / 'programs' / f'generation-{row["generation"]:03d}.py'
        original = canonical.get(row['generation'])
        if original and source != (original['code'] or ''):
            raise ValueError('Administrative seed clone has divergent source')
        public_text(path, source)
        history = json_field(row, 'migration_history', [])
        sampled = [e for e in sampling if e['generation'] == row['generation'] and e['parent_id'] == row['parent_id']]
        born = sampled[-1]['parent_island'] if sampled else row['island_idx']
        if history:
            born = history[0].get('from_island', history[0].get('from', born))
        lineage.append({k: row.get(k) for k in ('id', 'generation', 'timestamp', 'parent_id', 'island_idx', 'system_prompt_id')} | {
            'canonical_slot': not bool(meta.get('_is_island_copy')),
            'administrative_seed_copy': bool(meta.get('_is_island_copy')),
            'original_program_id': meta.get('_original_program_id'), 'birth_island': born,
            'correct': bool(row['correct']), 'combined_score': row['combined_score'],
            'source': relative(path), 'source_sha256': digest(source), 'source_bytes': len(source.encode()),
            'archive_inspiration_ids': json_field(row, 'archive_inspiration_ids', []),
            'top_k_inspiration_ids': json_field(row, 'top_k_inspiration_ids', []),
            'migration_history': history,
            'patch': {k: meta.get(k) for k in ('patch_type', 'patch_name', 'patch_description', 'llm_model', 'model_name', 'failure_stage')},
            'text_feedback': row.get('text_feedback'),
            'timing': {k: v for k, v in meta.items() if k.endswith('_seconds') or k.endswith('_at')},
        })
    dump(out / 'lineage.json', {'snapshot_utc': now, 'rows': lineage,
         'slot_definition': 'One nonadministrative row per generation, including seed and terminal failures; seed island copies are not additional evaluations.'})

    attempts = []
    request_to_attempt = defaultdict(list)
    for path in sorted(campaign.glob('gen_*/attempts/**/metadata.json')):
        meta = read(path)
        directory = out / 'attempts' / path.parent.relative_to(campaign)
        files = {}
        for name in ('llm_response.txt', 'patch.txt'):
            original = path.parent / name
            if original.exists():
                public_text(directory / name, original.read_text())
                files[name] = provenance(original) | {'public_path': relative(directory / name)}
        prompt = path.parent / 'headless_prompt.md'
        if prompt.exists():
            files['headless_prompt.md'] = provenance(prompt)
            request_to_attempt[files['headless_prompt.md']['sha256']].append(meta)
        compact = {k: meta.get(k) for k in ('generation', 'novelty_attempt', 'resample_attempt', 'patch_attempt',
            'success', 'num_applied', 'patch_name', 'patch_description', 'error_msg', 'timestamp', 'llm_cost', 'llm_model')}
        compact['files'] = files
        compact['classification'] = ('successful_patch_application' if meta.get('success') else
            'no_response_native_attempt' if meta.get('error_msg') == 'LLM response content was None.' else 'patch_application_rejection')
        compact['original_metadata'] = provenance(path)
        dump(directory / 'metadata.json', compact)
        attempts.append(compact)

    branches = []
    for path in sorted(campaign.glob('gen_*/proposal-sources/*.py')):
        directory = out / 'proposal-sources' / path.parent.parent.name
        public_text(directory / path.name, path.read_text())
        meta = read(path.with_suffix('.json'), {})
        branch = {'source': relative(directory / path.name), 'source_sha256': digest(path.read_bytes()),
                  'generation': int(path.parent.parent.name[4:]), 'stage': meta.get('stage'),
                  'sampling': meta.get('sampling'), 'original_metadata': provenance(path.with_suffix('.json'))}
        if meta.get('source_sha256') != branch['source_sha256']:
            raise ValueError('Saved generated source hash mismatch')
        dump(directory / path.with_suffix('.json').name, branch)
        branches.append(branch)

    call_records = []
    request_evidence = []
    for path in sorted((campaign / 'model-calls').glob('*.json')):
        call = read(path)
        public = {k: v for k, v in call.items() if k != 'command'}
        command = call.get('command', [])
        public['headless_child_timeout_seconds'] = int(command[command.index('--timeout') + 1]) if '--timeout' in command else None
        public['record_provenance'] = provenance(path)
        prompt = path.with_suffix('.prompt.md')
        text = prompt.read_text() if prompt.exists() else ''
        if text and digest(text) != call['prompt_sha256']:
            raise ValueError('Actual request hash mismatch')
        public['files'] = {suffix: provenance(path.with_suffix(suffix)) for suffix in ('.prompt.md', '.stdout', '.stderr')
                           if path.with_suffix(suffix).exists()}
        public['successful_process_return'] = call.get('status') == 'completed' and call.get('returncode') == 0
        public['outcome'] = ('timeout_without_response' if call.get('returncode') == 124 and
            path.with_suffix('.stdout').exists() and not path.with_suffix('.stdout').read_bytes() else
            'successful_return' if public['successful_process_return'] else
            'active' if call.get('status') == 'started' else 'unsuccessful_return')
        linked = request_to_attempt.get(call['prompt_sha256'], [])
        public['native_patch_attempts'] = [{k: a.get(k) for k in ('generation', 'novelty_attempt', 'resample_attempt', 'patch_attempt', 'success')} for a in linked]
        # A started call has no attempt archive yet. Associate to the latest sampling event
        # preceding admission, then prove parent and feedback presence from the request.
        candidates = [e for e in sampling if e['utc'] <= call['started_utc']]
        if call['role'] in ('mutation', 'repair') and candidates:
            sample = candidates[-1]
            if linked:
                exact = [e for e in candidates if e['generation'] == linked[0]['generation'] and
                         e['novelty_attempt'] == linked[0]['novelty_attempt'] and e['resample_attempt'] == linked[0]['resample_attempt']]
                if exact:
                    sample = exact[-1]
            context, decoded = prompt_context(text)
            parent = by_id.get(sample['parent_id'])
            inspirations = sample['archive_inspiration_ids'] + sample['top_k_inspiration_ids']
            evidence = {'call_id': call['id'], 'request_sha256': call['prompt_sha256'],
                'generation': sample['generation'], 'parent_id': sample['parent_id'],
                'parent_code_present': bool(parent and parent['code'] in context),
                'parent_feedback_is_string': sample['parent_feedback_is_string'],
                'feedback_present': sample['parent_text_feedback'] in context,
                'immutable_boundary_present': 'NON-EVOLVING EXPERIMENT BOUNDARY:' in context,
                'infrastructure_failure_note_present': 'Generation1 was killed by a scheduler timing bug before any episodes;' in context,
                'initial_task_present': resolved.get('evolution', {}).get('task_sys_msg', '') in context,
                'native_history_decoded': decoded,
                'inspiration_code_presence': {i: bool(i in by_id and by_id[i]['code'] in context) for i in inspirations},
                'recommendation_available': bool(sample['recommendation']),
                'recommendation_present': bool(sample['recommendation'] and sample['recommendation'] in context),
                'private_pool_path_present': 'results/private' in context}
            public['generation'] = sample['generation']
            request_evidence.append(evidence)
        call_records.append(public)
    dump(out / 'model-calls.json', {'snapshot_utc': now, 'calls': call_records})
    session_metadata = codex_metadata(campaign, call_records)
    dump(out / 'codex-session-metadata.json', {'snapshot_utc': now, **session_metadata})
    dump(out / 'request-audit.json', {'snapshot_utc': now, 'requests': request_evidence,
         'interpretation': 'After native prompt evolution, initial_task_present may be false; the selected prompt and immutable boundary must still be present. Native crossover may use only one selected inspiration rather than every sampled inspiration.'})
    # Preserve a complete real initial request and, once available, a later request
    # showing actual inspiration/recommendation context. Repeated histories stay hashed.
    selected = request_evidence[:1]
    rich = [e for e in request_evidence if e['inspiration_code_presence'] and e['recommendation_present']]
    if rich and rich[-1] not in selected:
        selected.append(rich[-1])
    examples = []
    for index, evidence in enumerate(selected):
        original = campaign / 'model-calls' / (evidence['call_id'] + '.prompt.md')
        path = out / ('request-example.md' if index == 0 else 'request-example-with-context.md')
        public_text(path, original.read_text())
        examples.append(evidence | {'public_path': relative(path), 'original': provenance(original)})
    dump(out / 'request-examples.json', examples)

    prompt_rows = rows(campaign / 'prompts.sqlite', 'system_prompts')
    archive = {r['prompt_id'] for r in rows(campaign / 'prompts.sqlite', 'prompt_archive')}
    prompt_records = []
    for row in sorted(prompt_rows, key=lambda r: (r['generation'], r['timestamp'])):
        record = {k: v for k, v in row.items() if k != 'metadata'}
        for key in ('program_scores', 'program_ids'):
            record[key] = json_field(row, key, [])
        record.update(in_archive=row['id'] in archive, prompt_sha256=digest(row['prompt_text']))
        public_text(out / 'prompts' / (row['id'] + '.md'), row['prompt_text'])
        prompt_records.append(record)
    dump(out / 'prompts.json', {'snapshot_utc': now, 'prompts': prompt_records,
         'credit_note': 'Pinned native prompt fitness averages percentiles over correct programs only; failures still increase program_count. This is native credit behavior, not failure-inclusive task fitness.'})
    for path in sorted((campaign / 'meta').glob('*')):
        if path.is_file() and path.suffix in ('.txt', '.json', '.md'):
            public_text(out / 'meta' / path.name, path.read_text())

    native_state = read(campaign / 'run1-native-state.json', {})
    meta_state = read(campaign / 'run1-meta-state.json', {})
    meta_snapshot = {k: v for k, v in meta_state.items() if k != 'unprocessed_programs'}
    meta_snapshot['unprocessed_program_ids'] = [r['id'] for r in meta_state.get('unprocessed_programs', [])]
    dump(out / 'meta-state.json', meta_snapshot)
    # No candidate observations, private pools, raw embedding vectors, credentials or
    # large DBs are published. Global RNG arrays remain runtime-only, with hash binding.
    compact_state = {k: v for k, v in native_state.items() if k not in ('python_rng', 'numpy_rng', 'bandit_rng')}
    compact_state['rng_state_sha256'] = {k: digest(json.dumps(native_state.get(k), sort_keys=True))
                                       for k in ('python_rng', 'numpy_rng', 'bandit_rng')}
    dump(out / 'native-state.json', compact_state)
    public_text(out / 'native-events.jsonl', ''.join(json.dumps(clean(e), sort_keys=True, allow_nan=False) + '\n' for e in events))
    held = [read(p) | {'provenance': provenance(p)} for p in sorted(campaign.glob('gen_*/held-proposal.json'))]
    for item in held:
        item['generated_source_available'] = bool(item.get('source_sha256'))
        item['retained_material'] = 'generated program and context' if item['generated_source_available'] else 'sampled parent/inspiration/recommendation context and actual request; no generated program'
    accepted = []
    for path in sorted(campaign.glob('gen_*/accepted-proposal.json')):
        value = read(path)
        job = value.get('job', {})
        accepted.append({'generation': int(path.parent.name[4:]), 'stage': value.get('stage'),
            'source_sha256': value.get('source_sha256'), 'provenance': provenance(path),
            'job': {k: job.get(k) for k in ('generation', 'parent_id', 'archive_insp_ids', 'top_k_insp_ids', 'start_time', 'proposal_started_at', 'evaluation_started_at', 'evaluation_submitted_at')},
            'embedding_evidence': value.get('embedding_evidence'), 'novelty_evidence': value.get('novelty_evidence')})
    dump(out / 'proposals.json', {'generated_branches': branches, 'accepted': accepted, 'held': held, 'patch_attempts': attempts})
    infrastructure = []
    controller_logs = list(campaign.glob('controller*.log'))
    for path in sorted(campaign.glob('gen_*/accepted-proposal.json')):
        saved = read(path)
        job = saved['job']
        generation = int(path.parent.name[4:])
        row = canonical.get(generation)
        if not row or row['correct']:
            continue
        for log in controller_logs:
            content = log.read_text()
            match = re.search(r'Process\s+' + re.escape(str(job.get('job_id'))) +
                r'\s+exceeded timeout of ([0-9:]+)\. Killing\.\s+=> Gen\.\s+' + str(generation) + r'\b', content)
            if match:
                start = job.get('start_time')
                evaluation_start = job.get('evaluation_started_at')
                infrastructure.append({'generation': generation,
                    'classification': 'native_evaluation_timeout',
                    'logged_evidence': ' '.join(match.group(0).split()),
                    'source_log': provenance(log),
                    'job_start_time': start, 'evaluation_started_at': evaluation_start,
                    'pre_evaluation_seconds_charged_by_original_timer': evaluation_start - start if start and evaluation_start else None,
                    'metrics_file_present': (path.parent / 'results/metrics.json').exists(),
                    'episode_file_present': (path.parent / 'results/episodes.json').exists(),
                    'interpretation': 'Original native local scheduler uses job.start_time, which this job records as proposal start. A timeout before evaluation can complete is infrastructure evidence, not an empirical failure of the candidate algorithm.'})
    dump(out / 'infrastructure-findings.json', infrastructure)

    novelty = [e for e in events if e['kind'] == 'novelty_decision']
    embeddings = [e for e in events if e['kind'] == 'local_embedding']
    mutation_calls = [c for c in call_records if c['role'] in ('mutation', 'repair')]
    discovery = [c for c in call_records if c['role'] != 'readiness']
    readiness = [c for c in call_records if c['role'] == 'readiness']
    migrations = [{'program_id': r['id'], 'generation': r['generation'], **event}
                  for r in lineage for event in r['migration_history']]
    observed_types = dict(Counter(r['patch']['patch_type'] for r in lineage if r['canonical_slot'] and r['generation'] > 0))
    by_role = {role: totals([c for c in call_records if c['role'] == role]) for role in sorted({c['role'] for c in call_records})}
    by_model = {model: totals([c for c in call_records if c['requested_model'] == model]) for model in sorted({c['requested_model'] for c in call_records})}
    meta_calls = [c for c in call_records if c['role'] in ('summary', 'global_insight', 'recommendation')]
    bandit = native_state.get('bandit_state') or {}
    budget_ledger = read(campaign / 'call-budget-ledger.json', {})
    blocked = budget_ledger.get('blocked_reason')
    arms_used = {c['requested_model'] for c in mutation_calls}
    executed_prompt = len(prompt_records) > 1
    report = {
        'schema': 'run1-native-report-v1', 'snapshot_utc': now,
        'campaign_manifest': provenance(campaign / 'campaign-manifest.json'),
        'runtime_compatibility_amendment': read(campaign / 'runtime-amendment.json'),
        'execution_status': {'native_state_boundary': native_state.get('boundary'),
            'budget_blocked_reason': blocked,
            'automatic_restart_allowed': False if blocked else None,
            'interpretation': 'A normal controller return/checkpoint does not mean the candidate budget was completed or every admitted model request succeeded.'},
        'source_scope': 'Saved runtime state only; no new worlds/model calls. Live files can advance between snapshots; per-file hashes and snapshot timestamp identify evidence.',
        'configured': {k: resolved.get(k) for k in ('evolution', 'database', 'max_evaluation_jobs', 'max_proposal_jobs', 'max_db_workers', 'episode_workers', 'budget', 'billing', 'deadline_utc', 'hard_checkpoint_utc', 'upstream', 'headless', 'effective_models', 'unsupported_not_forwarded', 'roles')},
        'counts': {'authorized_total_slots': native_state.get('authorized_total_slots', 32),
            'persisted_slots': len(canonical), 'native_rows': len(programs),
            'administrative_seed_copies': len(programs) - len(canonical),
            'seed_slots': int(0 in canonical),
            'valid_descendants': sum(g > 0 and bool(r['correct']) for g, r in canonical.items()),
            'failed_slots': [g for g, r in canonical.items() if not r['correct']],
            'generated_proposals': len(branches), 'accepted_proposals': len(accepted), 'held_proposals': len(held),
            'admitted_slots_including_held': len(set(canonical) | {e['generation'] for e in sampling} | {h['generation'] for h in held}),
            'held_slot_ids': [h['generation'] for h in held],
            'held_slots_with_generated_source': sum(h['generated_source_available'] for h in held),
            'held_slots_without_generated_source': sum(not h['generated_source_available'] for h in held),
            'sampling_contexts': len(sampling), 'patch_attempts': len(attempts),
            'patch_rejections': sum(not a['success'] for a in attempts),
            'patch_application_rejections_with_response': sum(a['classification'] == 'patch_application_rejection' for a in attempts),
            'no_response_native_patch_attempts': sum(a['classification'] == 'no_response_native_attempt' for a in attempts),
            'native_patch_retry_attempts': sum((a['patch_attempt'] or 1) > 1 for a in attempts),
            'patch_repairs': sum(c['role'] == 'repair' for c in call_records),
            'actual_remote_repair_calls': sum(c['role'] == 'repair' for c in call_records),
            'parent_resamples': len({(a['generation'], a['novelty_attempt'], a['resample_attempt']) for a in attempts if (a['resample_attempt'] or 1) > 1}),
            'novelty_rejections': sum(not e['accepted'] for e in novelty)},
        'calls': {'all': totals(call_records), 'readiness_and_fixture': totals(readiness), 'discovery': totals(discovery), 'by_role': by_role, 'by_model': by_model,
            'mutation_model_admissions': dict(Counter(c['requested_model'] for c in mutation_calls)),
            'semantics': 'Readiness contains two actual model-availability requests and one native novelty fixture; it consumes the shared budget but no candidate slots. Raw status completed means the subprocess returned, including unsuccessful timeout exits; successful returns are counted separately. Native patch retries can occur locally after admission is blocked and are not additional remote repair calls.'},
        'codex_session_metadata': session_metadata,
        'mechanisms': {
            'islands': {'status': 'exercised' if programs else 'pending', 'configured': resolved.get('database', {}).get('num_islands'), 'rows_by_current_island': dict(Counter(r['island_idx'] for r in programs)), 'migrations': migrations},
            'migration': {'configured': True, 'reachable': True, 'status': 'exercised' if migrations else 'unexercised_before_stop' if blocked else 'pending', 'observed_events': len(migrations), 'configured_interval': resolved.get('database', {}).get('migration_interval'), 'note': 'Four initialized islands are not evidence that a migration occurred.'},
            'inspirations': {'status': 'exercised' if any(e['archive_inspiration_ids'] or e['top_k_inspiration_ids'] for e in sampling) else 'pending', 'contexts_archive': sum(bool(e['archive_inspiration_ids']) for e in sampling), 'contexts_top_k': sum(bool(e['top_k_inspiration_ids']) for e in sampling)},
            'operators': {'persisted_by_type': observed_types, 'configured_probabilities': dict(zip(resolved.get('evolution', {}).get('patch_types', []), resolved.get('evolution', {}).get('patch_type_probs', []))), 'crossover': {'configured': True, 'reachable': True, 'status': 'exercised' if observed_types.get('cross') else 'unexercised_before_stop' if blocked else 'pending', 'observed_persisted_slots': observed_types.get('cross', 0)}},
            'model_bandit': {'status': 'two_arms_queried_with_outcome_feedback' if len(arms_used) == 2 and all(n > 0 for n in bandit.get('n_completed', [0])) else 'partially_exercised' if mutation_calls else 'pending', 'state': bandit, 'arm_models_queried': sorted(arms_used), 'snapshot_boundary': native_state.get('boundary'), 'reward_note': 'Native UCB with cost coefficient zero; submitted counts can include patch resamples. Shifted nonnegative rewards, not a controlled model comparison. Under exponential scaling s is log-sum-exp accumulated reward contributions, not mean reward. State is from the latest safe checkpoint, so an active call may not yet appear. Preserved infrastructure failures also supplied outcome feedback.'},
            'novelty': {'status': 'discovery_judge_exercised' if any(c['role'] == 'novelty' for c in call_records) else 'gate_exercised_without_discovery_judge' if novelty else 'pending', 'fixture': 'artifacts/campaign-v4/run1/novelty-fixture.json', 'decisions': novelty, 'embedding_calls': embeddings, 'limitations': 'Native embedding uses the first 10000 source characters; local general-English BGE is not validated as semantic code novelty. Native high-similarity gate is island-local. Fixture call is separate from discovery.'},
            'meta': {'status': 'recommendations_consumed' if any(e['recommendation'] for e in sampling) else 'calls_exercised' if meta_calls else 'pending', 'calls_by_role': dict(Counter(c['role'] for c in meta_calls)), 'sampling_contexts_with_recommendation': sum(bool(e['recommendation']) for e in sampling), 'state': meta_snapshot},
            'prompt_coevolution': {'configured': True, 'reachable': True, 'status': 'new_prompt_stored' if executed_prompt else 'call_exercised' if any(c['role'] == 'prompt_mutation' for c in call_records) else 'unexercised_before_stop' if blocked else 'pending', 'prompt_rows': len(prompt_records), 'archive_rows': len(archive), 'programs_with_prompt_credit': sum(r['program_count'] for r in prompt_records), 'correct_programs_with_prompt_credit': sum(r['correct_program_count'] for r in prompt_records), 'counter': native_state.get('prompt_evolution_counter'), 'note': 'Initial prompt selection and fitness credit were exercised; no prompt mutation call means prompt coevolution itself was not exercised.'},
            'resume': {'status': 'actually_restored' if any(e['kind'] == 'native_state_restored' for e in events) else 'checkpoint_available_not_live_resumed', 'events': [e for e in events if e['kind'] in ('native_state_restored', 'accepted_proposal_recovered', 'checkpoint_requested', 'proposal_held')], 'limitation': 'Accepted pending proposals can resume without repeated mutation/novelty. A generated pre-novelty held proposal is retained but requires explicit stage recovery; automatic blind continuation fails closed.'}},
        'saturation_reviews': [e for e in events if e['kind'] == 'development_saturation_review'],
        'infrastructure_findings': infrastructure,
        'supervisor_usage': {'path': 'artifacts/campaign-v4/run1/supervisor-usage.json', 'scope': 'Separate supervising Codex sessions, outside experiment-route call budget. Never combine these with experiment calls as if they shared the same ledger.'},
        'accounting_notes': ['Uncached input plus output is the primary token budget. Cached input is separately reported; reasoning output is a subset of output and is not added twice.', 'Reported dollar costs are API-list-price estimates, not actual subscription charges. All observed billing attempts must remain subscription.', 'Remote elapsed sums per-call monotonic durations, including readiness and fixtures; these may overlap other local work. They are not controller wall time or model CPU.', 'Candidate/evaluator CPU and episode resources are exported by run1_science_report.py; do not add those metrics to themselves via native pipeline timing.', 'Effort high is requested and forwarded to Codex; usage reports model while aligned native turn_context metadata independently corroborates model and effort. Native temperature/max_tokens are not forwarded.', 'Bandit nonfinite unobserved extrema are encoded as null in public strict JSON. Runtime originals remain unchanged.'],
        'scientific_limits': ['One development search with reused cases; no independent-discovery reliability claim.', 'All mechanisms enabled together in a prospective ecosystem configuration; no individual mechanism causal attribution.', 'Task improvement, prediction improvement and control benefit from adaptive prediction remain separate hypotheses.', 'No fresh assessment or selection-validation evidence is created by this exporter.'],
        'runtime_provenance': {name: provenance(campaign / name) for name in ('dreamer-resolved.json', 'run1-native-state.json', 'run1-meta-state.json', 'call-budget-ledger.json', 'native-events.jsonl') if (campaign / name).exists()},
    }
    dump(out / 'native-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'results/campaign-v4-run1')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/campaign-v4/run1/native')
    args = parser.parse_args()
    report = run(args.campaign.resolve(), args.out.resolve())
    print(json.dumps({'snapshot_utc': report['snapshot_utc'], 'counts': report['counts'], 'calls': report['calls']['all']}, indent=2))


if __name__ == '__main__':
    main()
