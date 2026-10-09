"""Export and render saved original-task native search evidence, without execution.

The runtime databases and ledgers are opened read-only. Administrative seed
copies, failed slots and admitted but unfinished proposals remain distinguishable.
This module never imports an evaluator or invokes an agent, world or model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import chromatic_fields as style
from scripts.run1_native_report import (clean, digest, dump, json_field, jsonl,
    prompt_context, provenance, public_text, read, relative, rows, totals)
from proposal.figures import draw_map, translated_belief, wilson
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

OUTCOMES = (('escaped', 'Escape', 'escape'), ('caught', 'Capture', 'death'),
            ('timeout', 'Timeout', 'timeout'), ('invalid', 'Invalid', 'invalid'))
METRICS = (('combined_score', 'Fitness F', style.COBALT),
           ('task', 'Task S', style.ORANGE),
           ('model_accuracy', 'Map accuracy A', style.MAGENTA),
           ('final_coverage', 'Last-record coverage', style.SECONDARY))
EPISODE_FIELDS = ('case', 'reason', 'steps', 'keys', 'door_open', 'task',
                  'model_accuracy', 'combined_score', 'map_correct', 'map_audited',
                  'final_coverage', 'error', 'seconds', 'candidate_cpu_seconds',
                  'evaluator_cpu_seconds')


def table(path, records):
    if not records:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for row in records for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in records)


def saved_sources(campaign, out, native, events, inputs):
    """Preserve every native source and generated branch, including nonwinners."""
    lineage, canonical = [], {}
    sampling = [e for e in events if e.get('kind') == 'sampling']
    for row in native:
        meta = json_field(row, 'metadata', {})
        administrative = bool(meta.get('_is_island_copy'))
        generation = row['generation']
        source = row.get('code') or ''
        target = out/'programs'/f'generation-{generation:03d}.py'
        if not administrative:
            if generation in canonical:
                raise ValueError(f'Multiple canonical programs for slot {generation}')
            canonical[generation] = row
            if source:
                public_text(target, source)
        history = json_field(row, 'migration_history', [])
        sampled = [e for e in sampling if e['generation'] == generation and
                   e.get('parent_id') == row.get('parent_id')]
        born = sampled[-1].get('parent_island') if sampled else row.get('island_idx')
        if history:
            born = history[0].get('from_island', history[0].get('from', born))
        lineage.append({k: row.get(k) for k in ('id', 'parent_id', 'generation',
            'timestamp', 'island_idx', 'system_prompt_id')} | {
            'administrative_seed_copy': administrative, 'canonical_slot': not administrative,
            'original_program_id': meta.get('_original_program_id'), 'birth_island': born,
            'correct': bool(row.get('correct')), 'native_fitness': row.get('combined_score'),
            'source_sha256': digest(source), 'source_bytes': len(source.encode()),
            'source': relative(target) if source else None,
            'archive_inspiration_ids': json_field(row, 'archive_inspiration_ids', []),
            'top_k_inspiration_ids': json_field(row, 'top_k_inspiration_ids', []),
            'migration_history': history,
            'embedding_dimensions': len(json_field(row, 'embedding', [])),
            'patch': {k: meta.get(k) for k in ('patch_type', 'patch_name',
                'patch_description', 'llm_model', 'model_name', 'failure_stage', 'failure_reason')},
            'text_feedback': row.get('text_feedback'),
            'timing': {k: v for k, v in meta.items() if k.endswith('_seconds') or k.endswith('_at')}})
    for record in lineage:
        row = canonical.get(record['generation'])
        if record['administrative_seed_copy'] and row and digest(row.get('code') or '') != record['source_sha256']:
            raise ValueError('Administrative copy has divergent source')
    branches = []
    for path in sorted(campaign.glob('gen_*/proposal-sources/*.py')):
        target = out/'proposal-sources'/path.parent.parent.name/path.name
        public_text(target, path.read_text())
        metadata = read(path.with_suffix('.json'), {})
        if metadata.get('source_sha256') and metadata['source_sha256'] != digest(path.read_bytes()):
            raise ValueError('Generated source hash mismatch')
        branches.append({'source': relative(target), 'source_sha256': digest(path.read_bytes()),
                         'generation': int(path.parent.parent.name[4:]), 'metadata': metadata})
        inputs.append(provenance(path))
    return canonical, lineage, branches


def evidence(campaign, out):
    now = datetime.now(timezone.utc).isoformat()
    events = jsonl(campaign/'native-events.jsonl')
    native = sorted(rows(campaign/'programs.sqlite', 'programs'),
                    key=lambda r: (r['generation'], r['timestamp'], r['id']))
    inputs = []
    canonical, lineage, branches = saved_sources(campaign, out, native, events, inputs)
    sampled_slots = {e['generation'] for e in events if e.get('kind') == 'sampling'}
    generations = sorted(set(canonical) | sampled_slots |
                         {int(p.name[4:]) for p in campaign.glob('gen_*') if p.name[4:].isdigit()})
    if not generations:
        raise ValueError('No saved native slots or admitted proposals; no report data to render')
    slots, episodes, resources, replays = [], [], [], []
    for generation in generations:
        row = canonical.get(generation, {})
        metadata = json_field(row, 'metadata', {})
        directory = Path(metadata.get('recovery_results_dir', campaign/f'gen_{generation}'/'results'))
        episode_path, metric_path = directory/'episodes.json', directory/'metrics.json'
        saved = read(episode_path, [])
        metrics = read(metric_path, {})
        correctness = read(directory/'correct.json', {})
        resource = read(directory/'resource.json', {})
        if resource:
            resources.append({'generation': generation, **resource})
        if saved and saved[0].get('trace'):
            replays.append({'generation': generation, **saved[0]})
        for path in (episode_path, metric_path, directory/'correct.json', directory/'manifest.json', directory/'resource.json'):
            if path.exists():
                inputs.append(provenance(path))
        if 'episodes' in metrics.get('public', {}) and metrics['public']['episodes'] != len(saved):
            raise ValueError(f'Episode count mismatch for slot {generation}')
        cases = [e['case'] for e in saved]
        if len(cases) != len(set(cases)):
            raise ValueError(f'Duplicate development cases for slot {generation}')
        for episode in saved:
            if episode['reason'] not in {x[0] for x in OUTCOMES}:
                raise ValueError('Unrecognized episode outcome')
            episodes.append({'generation': generation, **{k: episode.get(k) for k in EPISODE_FIELDS}})
        means = {key: sum(e[key] for e in saved)/len(saved) if saved else None
                 for key, *_ in METRICS}
        if saved and metrics and not math.isclose(metrics['combined_score'], means['combined_score'], abs_tol=1e-10):
            raise ValueError(f'Fitness mismatch for slot {generation}')
        held_record = read(campaign/f'gen_{generation}'/'held-proposal.json', {})
        status = ('valid' if row and row.get('correct') else 'failed' if row else
                  'evaluated_pending_native_record' if saved else 'held' if held_record else 'pending')
        slots.append({'generation': generation, 'status': status,
            'program_id': row.get('id'), 'episodes': len(saved), 'cases': cases,
            'held_stage': held_record.get('stage'), 'held_reason': held_record.get('reason'),
            'held_utc': held_record.get('held_utc'),
            'generated_source_available': bool(row.get('code') or held_record.get('source_sha256')),
            'counts_as_completed_slot': bool(row),
            'native_fitness': row.get('combined_score'), **means,
            **{reason: sum(e['reason'] == reason for e in saved) for reason, *_ in OUTCOMES},
            'transitions': sum(e['steps'] for e in saved),
            'episode_wall_seconds': sum(e.get('seconds', 0) for e in saved),
            'candidate_cpu_seconds': resource.get('candidate_cpu_seconds',
                sum(e['candidate_cpu_seconds'] for e in saved)
                if saved and all(e.get('candidate_cpu_seconds') is not None for e in saved) else None),
            'evaluator_cpu_seconds': resource.get('evaluator_cpu_seconds'),
            'evaluation_elapsed_seconds': resource.get('elapsed_seconds'),
            'evaluation_ended_utc': resource.get('ended_utc'),
            'native_timestamp': row.get('timestamp'),
            'mean_keys': sum(e['keys'] for e in saved)/len(saved) if saved else None,
            'door_open': sum(bool(e['door_open']) for e in saved),
            'escape_wilson95': wilson(sum(e['reason'] == 'escaped' for e in saved), len(saved)) if saved else None,
            'failure_stage': metadata.get('failure_stage'),
            'failure_reason': metadata.get('failure_reason'), 'evaluator_correctness': correctness,
            'reported_public_metrics': metrics.get('public', json_field(row, 'public_metrics', {})),
            'evaluation_errors': [e['error'] for e in saved if e.get('error')]})
    calls, request_audit = [], []
    by_id = {r['id']: r for r in native}
    sampling = [e for e in events if e.get('kind') == 'sampling']
    for path in sorted((campaign/'model-calls').glob('*.json')):
        call = read(path)
        public = {k: call.get(k) for k in ('id', 'role', 'started_utc', 'ended_utc',
            'requested_model', 'requested_effort', 'billing_required', 'status',
            'prompt_sha256', 'elapsed_seconds', 'returncode', 'usage', 'error',
            'billing_violation', 'model_mismatch', 'cli_lock_wait_seconds', 'usage_missing') if k in call}
        inputs.append(provenance(path))
        stdout = path.with_suffix('.stdout')
        public['response_stdout'] = provenance(stdout) if stdout.exists() else None
        public['response_stdout_empty'] = stdout.stat().st_size == 0 if stdout.exists() else None
        if call.get('returncode') not in (None, 0) and stdout.exists() and stdout.stat().st_size:
            public_text(out/'partial-responses'/stdout.name, stdout.read_text())
            public['retained_partial_response'] = relative(out/'partial-responses'/stdout.name)
        prior = [e for e in sampling if e['utc'] <= call['started_utc']]
        if prior and call['role'] in ('mutation', 'repair'):
            sample = prior[-1]
            public['generation'] = sample['generation']
            prompt = path.with_suffix('.prompt.md')
            if prompt.exists():
                text = prompt.read_text()
                if digest(text) != call['prompt_sha256']:
                    raise ValueError('Actual request hash mismatch')
                context, decoded = prompt_context(text)
                parent = by_id.get(sample['parent_id'])
                request_audit.append({'call_id': call['id'], 'generation': sample['generation'],
                    'request_sha256': digest(text), 'parent_id': sample['parent_id'],
                    'parent_code_present': bool(parent and parent['code'] in context),
                    'feedback_is_string': sample.get('parent_feedback_is_string'),
                    'feedback_present': bool(sample.get('parent_text_feedback') and sample['parent_text_feedback'] in context),
                    'history_decoded': decoded,
                    'recommendation_present': bool(sample.get('recommendation') and sample['recommendation'] in context),
                    'inspiration_code_presence': {i: bool(i in by_id and by_id[i]['code'] in context)
                        for i in sample.get('archive_inspiration_ids', []) + sample.get('top_k_inspiration_ids', [])}})
                if len(request_audit) == 1:
                    public_text(out/'request-example.md', text)
        calls.append(public)
    roles = {role: totals([c for c in calls if c['role'] == role]) for role in sorted({c['role'] for c in calls})}
    native_state = read(campaign/'run1-native-state.json', read(campaign/'native-state.json', {}))
    compact_state = {k: v for k, v in native_state.items() if k not in ('python_rng', 'numpy_rng', 'bandit_rng')}
    compact_state['rng_state_sha256'] = {k: digest(json.dumps(native_state[k], sort_keys=True))
                                      for k in ('python_rng', 'numpy_rng', 'bandit_rng') if k in native_state}
    meta = read(campaign/'run1-meta-state.json', read(campaign/'meta-state.json', {}))
    compact_meta = {k: v for k, v in meta.items() if k != 'unprocessed_programs'}
    compact_meta['unprocessed_program_ids'] = [r['id'] for r in meta.get('unprocessed_programs', [])]
    prompt_rows = rows(campaign/'prompts.sqlite', 'system_prompts')
    prompt_archive = {r['prompt_id'] for r in rows(campaign/'prompts.sqlite', 'prompt_archive')}
    prompts = [{k: json_field(row, k, []) if k in ('program_scores', 'program_ids') else v
                for k, v in row.items() if k != 'metadata'} | {'in_archive': row['id'] in prompt_archive}
               for row in prompt_rows]
    embedding_calls = jsonl(campaign/'embedding-calls.jsonl')
    embedding_fixture = read(campaign/'embedding-fixture.json', {})
    held = [{'generation': int(p.parent.name[4:]), **read(p)}
            for p in sorted(campaign.glob('gen_*/held-proposal.json'))]
    for record in held:
        related = [c for c in calls if c.get('generation') == record['generation']]
        record['call_ids'] = [c['id'] for c in related]
        record['failed_provider_calls'] = sum(c.get('returncode') not in (None, 0) or c.get('status') == 'error' for c in related)
        record['returned_stdout_bytes'] = sum((c.get('response_stdout') or {}).get('bytes', 0) for c in related)
        record['retained_material'] = ('Generated source retained at the recorded stage.' if record.get('source_sha256') else
            'No valid generated source; sampled context and actual requests retained. Empty timeout stdout is not a program.')
        if related:
            path = campaign/'model-calls'/(related[-1]['id']+'.prompt.md')
            if path.exists():
                target = out/'held-requests'/f'generation-{record["generation"]:03d}.md'
                public_text(target, path.read_text())
                record['last_request_source'] = relative(target)
    attempts = [{k: value.get(k) for k in ('generation', 'novelty_attempt', 'resample_attempt',
        'patch_attempt', 'success', 'num_applied', 'patch_name', 'patch_description', 'error_msg',
        'timestamp', 'llm_model')} | {'source': provenance(p)}
        for p in sorted(campaign.glob('gen_*/attempts/**/metadata.json')) for value in [read(p)]]
    migrations = [{'program_id': r['id'], 'generation': r['generation'], **e}
                  for r in lineage for e in r['migration_history']]
    resolved = read(campaign/'resolved-config.json', read(campaign/'campaign-manifest.json', {}))
    evo, database = resolved.get('evolution', {}), resolved.get('database', {})
    log_path = campaign/'evolution_run.log'
    log_text = log_path.read_text() if log_path.exists() else ''
    operator_attempts, log_generation = [], None
    log_evidence = []
    for line_number, line in enumerate(log_text.splitlines(), 1):
        match = re.search(r'Getting meta recs for gen (\d+)', line)
        if match:
            log_generation = int(match.group(1))
        match = re.search(r'Generated patch type: (\w+)', line)
        if match:
            operator_attempts.append({'generation': log_generation, 'operator': match.group(1),
                'log_line': line_number, 'provider_calls': sum(c.get('generation') == log_generation and c['role'] in ('mutation','repair') for c in calls)})
        if match or any(term in line for term in ('Performing final meta summary for', 'Step 1 failed - no individual summaries generated',
            'Percentile recomputation complete', 'Two consecutive provider transport failures')):
            log_evidence.append({'line': line_number, 'text': line})
    dump(out/'native-log-evidence.json', {'source': provenance(log_path) if log_path.exists() else None,
                                        'lines': log_evidence, 'timestamp_note': 'Native human log timestamps retain their original local timezone; structured event timestamps are UTC.'})
    role_count = Counter(c['role'] for c in calls)
    selected_ops = Counter(r['operator'] for r in operator_attempts)
    successful_ops = Counter(r['patch'].get('patch_type') for r in lineage if r['canonical_slot'] and r['generation']>0 and r['correct'])
    inspiration_eligible_samples = sum(bool(e.get('archive_inspiration_ids') or e.get('top_k_inspiration_ids')) for e in sampling)
    mechanism_status = [
        {'mechanism':'islands_and_parent_sampling', 'configured':database.get('num_islands'), 'reachable':'Native weighted sampler called.',
         'exercised':{'samples':len(sampling),'sampled_islands':sorted({e.get('parent_island') for e in sampling})}},
        {'mechanism':'mutation_bandit','configured':evo.get('llm_dynamic_selection'), 'reachable':'Both configured subscription model arms were available to native allocation.',
         'exercised':{'calls_by_model':dict(Counter(c['requested_model'] for c in calls if c['role'] in ('mutation','repair'))),
                      'native_state':compact_state.get('bandit_state')}, 'interpretation':'Two observed arms do not establish benefit over another allocation policy.'},
        *[{'mechanism':f'operator_{operator}','configured':operator in evo.get('patch_types',[]),
           'reachable':'Native operator chosen.' if selected_ops[operator] else 'Conditional on eligible inspirations; none sampled.' if operator=='cross' and inspiration_eligible_samples==0 else 'Configured but not observed.',
           'exercised':{'chosen_slots':selected_ops[operator],'successful_persisted_descendants':successful_ops[operator],
                        'provider_attempts':sum(r['provider_calls'] for r in operator_attempts if r['operator']==operator)}}
          for operator in ('diff','full','cross')],
        {'mechanism':'inspirations','configured':{'archive':database.get('num_archive_inspirations'),'top_k':database.get('num_top_k_inspirations')},
         'reachable':'Native eligibility-dependent sampling; no eligible inspiration returned.' if not inspiration_eligible_samples else 'Eligible inspirations returned.',
         'exercised':{'samples_with_inspirations':inspiration_eligible_samples}},
        {'mechanism':'local_embeddings','configured':evo.get('embedding_model'),'reachable':'Cached local endpoint executed.',
         'exercised':{'native_events':sum(e.get('kind')=='local_embedding' for e in events),'startup_fixture_is_discovery':False}},
        {'mechanism':'novelty','configured':{'cosine_threshold':evo.get('code_embed_sim_threshold'),'models':evo.get('novelty_llm_models')},
         'reachable':'Native similarity threshold condition reached.', 'exercised':{'decisions':sum(e.get('kind')=='novelty_decision' for e in events),'provider_judge_calls':role_count['novelty']}},
        {'mechanism':'migration','configured':{'interval':database.get('migration_interval'),'rate':database.get('migration_rate')},
         'reachable':'Configured schedule; movement is subject to native timing and eligibility.', 'exercised':{'recorded_moves':len(migrations)}},
        {'mechanism':'meta_memory','configured':{'interval':evo.get('meta_rec_interval'),'models':evo.get('meta_llm_models')},
         'reachable':'Final flush reached summary client; shared admission gate blocked requests.' if any(e.get('kind')=='partial_meta_batch_rejected' for e in events) else 'Native pending-program tracking.',
         'exercised':{'summary_calls':role_count['summary'],'insight_calls':role_count['global_insight'],'recommendation_calls':role_count['recommendation'],
                      'pending_programs':len(compact_meta['unprocessed_program_ids']),'programs_processed':meta.get('total_programs_meta_processed',0)}},
        {'mechanism':'prompt_evolution','configured':{'enabled':evo.get('evolve_prompts'),'interval':evo.get('prompt_evolution_interval')},
         'reachable':'Initial prompt archived and credited; mutation trigger not reached.' if role_count['prompt_mutation']==0 else 'Prompt mutation admitted.',
         'exercised':{'prompt_records':len(prompts),'credited_correct_descendants':sum(p.get('correct_program_count',0) for p in prompts),'prompt_mutation_calls':role_count['prompt_mutation']}}
    ]
    mechanism = {'event_counts': dict(Counter(e.get('kind', 'unknown') for e in events)),
        'native_database_rows': len(native), 'administrative_seed_copies': sum(r['administrative_seed_copy'] for r in lineage),
        'sampled_islands': dict(Counter(str(e.get('parent_island', 'unknown')) for e in sampling)),
        'mutation_model_calls': dict(Counter(c.get('requested_model') for c in calls if c['role'] in ('mutation', 'repair'))),
        'operators': dict(Counter(r['patch'].get('patch_type') or 'unknown' for r in lineage if r['canonical_slot'])),
        'embedding_events': sum(e.get('kind') == 'local_embedding' for e in events),
        'embedding_elapsed_seconds': sum(e.get('elapsed_seconds', 0) for e in events if e.get('kind') == 'local_embedding'),
        'novelty_decisions': [e for e in events if e.get('kind') == 'novelty_decision'],
        'migrations': migrations, 'prompt_records': len(prompts),
        'operator_attempts': operator_attempts, 'configured_reachable_exercised': mechanism_status,
        'recommendations_in_requests': sum(e['recommendation_present'] for e in request_audit),
        'observed_inspiration_code_in_requests': sum(any(e['inspiration_code_presence'].values()) for e in request_audit),
        'note': 'Counts demonstrate recorded operation, not a causal contribution to search quality. Zero conditional events do not imply an unavailable mechanism.'}
    valid = [s for s in slots if s['status'] == 'valid' and s['combined_score'] is not None]
    best = max(valid, key=lambda s: (s['combined_score'], -s['generation'])) if valid else None
    curve, cumulative_cpu, incumbent = [], 0., None
    cpu_complete = True
    for slot in slots:
        timestamp = slot.get('native_timestamp')
        if timestamp is None and not slot.get('held_utc'):
            continue
        cutoff = (datetime.fromtimestamp(timestamp, timezone.utc) if timestamp is not None else
                  datetime.fromisoformat(slot['held_utc'].replace('Z', '+00:00')))
        completed = [c for c in calls if c.get('ended_utc') and
                     datetime.fromisoformat(c['ended_utc'].replace('Z', '+00:00')) <= cutoff]
        if slot['episodes']:
            cpu_complete = cpu_complete and slot['candidate_cpu_seconds'] is not None
            cumulative_cpu += slot['candidate_cpu_seconds'] or 0
        if slot['status'] == 'valid':
            incumbent = max(incumbent if incumbent is not None else -math.inf, slot['combined_score'])
        curve.append({'generation': slot['generation'], 'native_timestamp_utc': cutoff.isoformat(),
            'timestamp_kind': 'native_program_saved' if timestamp is not None else 'proposal_held',
            'status': slot['status'], 'fitness': slot['combined_score'], 'best_valid_fitness': incumbent,
            'completed_all_role_calls': len(completed),
            'completed_provider_seconds': sum(c.get('elapsed_seconds',0) for c in completed),
            'cumulative_candidate_cpu_seconds': cumulative_cpu if cpu_complete else None})
    paired = []
    seed_by_case = {e['case']: e for e in episodes if e['generation'] == 0}
    for slot in slots:
        if slot['generation'] == 0 or not slot['episodes']:
            continue
        selected = [e for e in episodes if e['generation'] == slot['generation']]
        if set(e['case'] for e in selected) != set(seed_by_case):
            continue
        paired.append({'generation': slot['generation'], 'paired_cases': len(selected),
            'mean_differences_from_seed': {key: sum(e[key]-seed_by_case[e['case']][key] for e in selected)/len(selected)
                                         for key,*_ in METRICS},
            'escape_discordance': {'descendant_only': sum(e['reason']=='escaped' and seed_by_case[e['case']]['reason']!='escaped' for e in selected),
                                  'seed_only': sum(e['reason']!='escaped' and seed_by_case[e['case']]['reason']=='escaped' for e in selected)},
            'note': 'Descriptive matched development differences; every case retained; no held-out claim or resampling.'})
    summary = {'schema': 'proposal-full-native-report-v1', 'snapshot_utc': now,
        'evaluator': 'namazu-proposal-reconstruction-v1', 'scope': 'Five fixed development cases; selection-biased single search, no withheld assessment.',
        'slot_count': len(slots), 'persisted_slots': len(canonical),
        'valid_descendants': sum(s['generation']>0 and s['status']=='valid' for s in slots),
        'failed_slots': [s['generation'] for s in slots if s['status']=='failed'],
        'pending_slots': [s['generation'] for s in slots if s['status'] not in ('valid','failed')],
        'held_slots': held,
        'slot_status_counts': dict(Counter(s['status'] for s in slots)),
        'saved_episodes': len(episodes), 'transitions': sum(s['transitions'] for s in slots),
        'episode_wall_seconds': sum(s['episode_wall_seconds'] for s in slots),
        'candidate_cpu_seconds': sum(s['candidate_cpu_seconds'] for s in slots if s['candidate_cpu_seconds'] is not None)
            if all(s['candidate_cpu_seconds'] is not None for s in slots if s['episodes']) else None,
        'best': best, 'calls': totals(calls), 'roles': roles, 'mechanisms': mechanism,
        'paired_development_differences': paired,
        'execution_resources': read(campaign/'execution-resources.json', {}),
        'embedding_resources': {'native_endpoint_calls': len(embedding_calls),
            'status_counts': dict(Counter(c.get('status','unknown') for c in embedding_calls)),
            'native_endpoint_cpu_seconds': sum(c.get('cpu_seconds',0) for c in embedding_calls),
            'native_endpoint_wall_seconds': sum(c.get('wall_seconds',0) for c in embedding_calls),
            'startup_fixture': embedding_fixture,
            'note': 'Startup deterministic fixture is separate infrastructure verification, not an evolutionary discovery. Endpoint CPU and wrapper elapsed have distinct scopes; do not add overlapping measures.'},
        'budget_ledger': read(campaign/'call-budget-ledger.json', {}),
        'measurement_notes': ['Fitness F = 0.6 task S + 0.4 current-map accuracy A; invalid executions have F=0.',
            'A pools reported-cell audits within an episode, then averages episode ratios. Unknown cells are excluded.',
            'Coverage is reported in-bounds cells at the final recorded pre-action decision / 225.',
            'Five fixed development cases are reused for selection; nominal Wilson intervals are descriptive.',
            'Missing usage and unmeasured CPU remain null or explicitly incomplete; native API list prices are not subscription charges.']}
    for name, value in {'summary': summary, 'slots': slots, 'episodes': episodes, 'lineage': lineage,
        'model-calls': calls, 'request-audit': request_audit, 'native-state': compact_state,
        'evaluation-resources': resources, 'resource-curve': curve,
        'embedding-calls': embedding_calls,
        'meta-state': compact_meta, 'prompts': prompts, 'mechanisms': mechanism,
        'proposals': {'generated_branches': branches, 'held': held, 'patch_attempts': attempts}}.items():
        dump(out/f'{name}.json', value)
    table(out/'slots.csv', slots)
    table(out/'episodes.csv', episodes)
    table(out/'resource-curve.csv', curve)
    with (out/'representative-replays.json.gz').open('wb') as handle:
        with gzip.GzipFile(filename='', fileobj=handle, mode='wb', mtime=0) as compressed:
            compressed.write(json.dumps(clean(replays), separators=(',', ':'), allow_nan=False).encode())
    dump(out/'replay-index.json', {'selection': 'First saved case for each evaluated slot; all recorded frames retained in their original order.',
        'file': 'representative-replays.json.gz',
        'replays': [{'generation': r['generation'], 'case': r['case'], 'reason': r['reason'],
                     'records': len(r['trace']), 'first_step': r['trace'][0]['world']['step'],
                     'last_step': r['trace'][-1]['world']['step']} for r in replays]})
    public_text(out/'native-events.jsonl', ''.join(json.dumps(clean(e), sort_keys=True, allow_nan=False)+'\n' for e in events))
    configuration_references = {}
    protocol = read(out/'protocol.json')
    for name in ('campaign-manifest.json', 'proposal-config.json', 'dreamer-resolved.json', 'resolved-config.json',
                 'embedding-identity.json', 'call-budget-ledger.json', 'execution-resources.json'):
        path = campaign/name
        if path.exists():
            inputs.append(provenance(path))
            if name in ('campaign-manifest.json', 'resolved-config.json', 'proposal-config.json', 'dreamer-resolved.json') and protocol is not None and read(path) == protocol:
                configuration_references[name] = relative(out/'protocol.json')
                duplicate = out/name
                if duplicate.exists() and read(duplicate) == protocol:
                    duplicate.unlink()
            else:
                public_text(out/name, path.read_text())
                configuration_references[name] = relative(out/name)
    if embedding_fixture:
        dump(out/'embedding-fixture.json', embedding_fixture)
        inputs.append(provenance(campaign/'embedding-fixture.json'))
    if (campaign/'embedding-calls.jsonl').exists():
        inputs.append(provenance(campaign/'embedding-calls.jsonl'))
    dump(out/'source-provenance.json', {'snapshot_utc': now, 'inputs': inputs,
        'configuration_references': configuration_references,
        'database_snapshot': {'program_rows_sha256': digest(json.dumps(clean(native), sort_keys=True, allow_nan=False)),
                              'prompt_rows_sha256': digest(json.dumps(clean(prompt_rows), sort_keys=True, allow_nan=False))},
        'native_events_sha256': digest(json.dumps(clean(events), sort_keys=True, allow_nan=False))})
    return summary, slots, episodes, lineage, calls, events


def score_figure(slots, episodes, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
    fig.subplots_adjust(left=.085, right=.98, top=.82, bottom=.15, hspace=.63, wspace=.30)
    fig.suptitle('Original-task evolution: every recorded slot', x=.04, ha='left', y=.98)
    fig.text(.04, .92, 'Five reused development cases · small dots = episodes · diamonds = means', color=style.SECONDARY)
    generations = [s['generation'] for s in slots]
    for ax, (key, title, color) in zip(axes.flat, METRICS):
        ax.set_title(title, loc='left')
        for slot in slots:
            generation = slot['generation']
            values = [e[key] for e in episodes if e['generation'] == generation]
            if values:
                ax.scatter(generation + np.linspace(-.14, .14, len(values)), values,
                           color=color, s=20, alpha=.5, zorder=3)
                ax.scatter(generation, slot[key], marker='D', color=color, edgecolor=style.BACKGROUND,
                           linewidth=.8, s=55, zorder=4)
            else:
                ax.scatter(generation, -.06, marker='X' if slot['status'] == 'failed' else '|',
                           color=style.SECONDARY, s=55, clip_on=False)
            if slot['status'] == 'failed':
                ax.axvspan(generation-.3, generation+.3, color=style.MUTED, zorder=0)
        ax.set(xticks=generations, xlabel='Native candidate slot (0 = seed)',
               xlim=(min(generations)-.5, max(generations)+.5), ylim=(-.10, 1.04), yticks=[0,.5,1])
        ax.grid(axis='y'); ax.set_axisbelow(True)
    fig.text(.04, .065, 'F = 0.6S + 0.4A. Grey bands = failed slots; × = no episodes; | = unfinished slot.', color=style.SECONDARY, fontsize=10)
    fig.text(.04, .025, 'Current-map reconstruction is measured after observation; this is not evidence of predictive learning.', color=style.SECONDARY, fontsize=10)
    style.save_figure(fig, out/'search-scores')


def lineage_figure(slots, lineage, events, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.7), gridspec_kw={'width_ratios':[1.3,1]})
    fig.subplots_adjust(left=.085, right=.97, top=.79, bottom=.24, wspace=.5)
    fig.suptitle('Recorded ancestry and native mechanisms', x=.04, ha='left', y=.98)
    fig.text(.04, .91, 'Edges show saved ancestry and inspirations; they do not establish causal credit.', color=style.SECONDARY)
    ax = axes[0]
    by_id = {r['id']: r for r in lineage}
    for row in lineage:
        if row['administrative_seed_copy']:
            continue
        x, y = row['generation'], row['birth_island']
        y = y if y is not None else 0
        for key, linestyle in (('parent_id', '-'), ('archive_inspiration_ids', ':'), ('top_k_inspiration_ids', ':')):
            links = row[key] if isinstance(row[key], list) else [row[key]]
            for ident in links:
                parent = by_id.get(ident)
                if parent:
                    py = parent['island_idx'] if parent['administrative_seed_copy'] else parent['birth_island']
                    py = py if py is not None else 0
                    ax.plot([parent['generation'], x], [py,y], color=style.SECONDARY, linestyle=linestyle, lw=1, zorder=1)
        operator = row['patch'].get('patch_type') or 'seed'
        marker = {'init':'D','seed':'D','full':'o','diff':'s','cross':'^'}.get(operator,'P') if row['correct'] else 'X'
        ax.scatter(x,y,color=style.ISLAND_COLORS[int(y)%4],marker=marker,s=85,zorder=3)
        ax.annotate(str(x),(x,y),xytext=(0,11),textcoords='offset points',ha='center',fontsize=10)
    copies = [r for r in lineage if r['administrative_seed_copy']]
    for row in copies:
        ax.scatter(0,row['island_idx'],facecolors=style.BACKGROUND,edgecolors=style.ISLAND_COLORS[row['island_idx']%4],marker='D',s=65,zorder=2)
    for slot in slots:
        if slot['status']=='held':
            sampled=[e for e in events if e.get('kind')=='sampling' and e.get('generation')==slot['generation']]
            if sampled:
                island=sampled[-1]['parent_island']
                ax.scatter(slot['generation'],island,marker='|',s=130,color=style.SECONDARY)
                ax.annotate(f"{slot['generation']} held",(slot['generation'],island),xytext=(0,11),textcoords='offset points',ha='center',fontsize=10)
    ax.set(xticks=[s['generation'] for s in slots], xlabel='Native candidate slot',
           yticks=range(4), yticklabels=[f'Island {i}' for i in range(4)], ylim=(-.5,3.7))
    ax.set_title('A  Source ancestry', loc='left'); ax.grid(axis='y'); ax.set_axisbelow(True)
    ax = axes[1]
    kinds = ['sampling', 'local_embedding', 'novelty_decision', 'failed_slot_counted', 'proposal_held']
    labels = ['Parent samples','Local embeddings','Novelty decisions','Failed slots counted','Proposals held']
    counts = Counter(e.get('kind') for e in events)
    ax.barh(range(len(kinds)), [counts[k] for k in kinds], color=style.COBALT, height=.56)
    for i,k in enumerate(kinds): ax.text(counts[k]+.12,i,str(counts[k]),va='center',fontsize=10)
    ax.set(yticks=range(len(kinds)), yticklabels=labels, xlabel='Recorded events',
           xlim=(0,max([counts[k] for k in kinds]+[1])*1.3)); ax.invert_yaxis()
    ax.set_title('B  Observed events', loc='left'); ax.grid(axis='x'); ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.legend(handles=[Line2D([],[],marker=m,color=style.SECONDARY,linestyle='',label=l)
        for m,l in [('D','Seed / island copy'),('s','Diff'),('o','Full'),('^','Crossover'),('X','Failure')]],
        loc='lower center',bbox_to_anchor=(.5,.08),ncol=5)
    fig.text(.04,.027,'Solid = parent; dotted = inspiration; | = held without source. Empty diamonds = seed copies, not evaluations.',fontsize=10,color=style.SECONDARY)
    style.save_figure(fig,out/'search-native')


def cost_outcomes_figure(slots, calls, summary, out):
    fig, axes = plt.subplots(2,2,figsize=(10,9))
    fig.subplots_adjust(left=.09,right=.97,top=.83,bottom=.21,wspace=.48,hspace=.62)
    fig.suptitle('Observed play and the cost of this search block',x=.04,ha='left',y=.98)
    fig.text(.04,.92,'All returned outcomes and provider attempts retained, including failures and missing usage.',color=style.SECONDARY)
    ax=axes[0,0]
    left=np.zeros(len(slots))
    for reason,label,role in OUTCOMES:
        values=[s[reason] for s in slots]
        ax.barh(range(len(slots)),values,left=left,label=label,color=style.OUTCOMES[role],height=.65)
        for i,value in enumerate(values):
            if value: ax.text(left[i]+value/2,i,str(value),ha='center',va='center',color=style.BACKGROUND,fontsize=10)
        left+=np.asarray(values)
    for i,s in enumerate(slots):
        if not s['episodes']: ax.text(.1,i,s['status'].replace('_',' '),va='center',color=style.SECONDARY,fontsize=10)
    ax.set(yticks=range(len(slots)),yticklabels=[str(s['generation']) for s in slots],
           ylabel='Candidate slot',xlabel='Saved development episodes',xlim=(0,max([s['episodes'] for s in slots]+[5])+.3))
    ax.invert_yaxis();ax.set_title('A  Episode outcomes',loc='left')
    ax=axes[0,1]
    for i,s in enumerate(slots):
        if not s['episodes']: continue
        low,high=s['escape_wilson95'];mean=s['escaped']/s['episodes']
        ax.plot([100*low,100*high],[i,i],color=style.COBALT,lw=2)
        ax.scatter(100*mean,i,color=style.COBALT,s=40,zorder=3)
    ax.set(yticks=range(len(slots)),yticklabels=[str(s['generation']) for s in slots],ylim=(len(slots)-.5,-.5),
           xlabel='Escape (%) · descriptive 95% Wilson',xlim=(-3,103),xticks=[0,50,100])
    ax.set_title('B  Outcome uncertainty',loc='left');ax.grid(axis='x');ax.set_axisbelow(True)
    ax=axes[1,0]
    roles=list(summary['roles'])
    for i,role in enumerate(roles):
        members=[c for c in calls if c['role']==role]
        good=sum(c.get('elapsed_seconds',0) for c in members if c.get('status')=='completed' and c.get('returncode')==0)
        other=sum(c.get('elapsed_seconds',0) for c in members)-good
        ax.barh(i,good,color=style.COBALT,height=.6)
        ax.barh(i,other,left=good,color=style.MAGENTA,height=.6)
        missing=sum((c.get('usage') or {}).get('usageStatus')!='reported' for c in members)
        ax.text(good+other,i,f'  {len(members)} call(s)'+(f'; {missing} usage ?' if missing else ''),va='center',fontsize=9)
    ax.set(yticks=range(len(roles)),yticklabels=[r.replace('_',' ') for r in roles],xlabel='Summed provider seconds')
    ax.set_xlim(0,max([v['remote_elapsed_seconds'] for v in summary['roles'].values()]+[1])*1.65)
    ax.invert_yaxis();ax.set_title('C  Provider requests by role',loc='left');ax.grid(axis='x');ax.set_axisbelow(True)
    ax=axes[1,1]
    curve=read(out.parent/'resource-curve.json',[])
    observed=[r for r in curve if r['best_valid_fitness'] is not None]
    if observed:
        ax.step([r['completed_provider_seconds']/60 for r in observed],
                [r['best_valid_fitness'] for r in observed],where='post',color=style.COBALT,lw=1.5)
        for index,r in enumerate(observed):
            ax.scatter(r['completed_provider_seconds']/60,r['best_valid_fitness'],
                       color=style.COBALT if r['status']=='valid' else style.MAGENTA,
                       marker='o' if r['status']=='valid' else 'X',s=40)
            label=f"{'held ' if r['status']=='held' else ''}{r['generation']} / {r['completed_all_role_calls']} calls"
            last=index==len(observed)-1
            ax.annotate(label,(r['completed_provider_seconds']/60,r['best_valid_fitness']),
                xytext=(-4 if last else 4,7+12*(index%2)),ha='right' if last else 'left',
                textcoords='offset points',fontsize=9)
    ax.set(xlabel='Provider minutes at save / hold',ylabel='Best valid fitness F',ylim=(0,1.12),yticks=[0,.5,1])
    ax.set_title('D  Fitness against provider cost',loc='left');ax.grid();ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=style.OUTCOMES[role],label=label) for _,label,role in OUTCOMES],
               loc='lower center',bbox_to_anchor=(.5,.075),ncol=4)
    fig.text(.04,.028,'Selection cases are reused; intervals do not correct selection bias. Provider time is separate from local evaluation.',fontsize=10,color=style.SECONDARY)
    style.save_figure(fig,out/'search-outcomes-costs')


def selected_replay(summary, out):
    if not summary['best']:
        return None
    with gzip.open(out.parent/'representative-replays.json.gz','rt') as handle:
        saved=json.load(handle)
    selected=next((r for r in saved if r['generation']==summary['best']['generation']),None)
    if selected is None:
        return None
    fig,axes=plt.subplots(2,2,figsize=(10,9.2))
    fig.subplots_adjust(left=.09,right=.93,top=.84,bottom=.18,hspace=.49,wspace=.25)
    fig.suptitle(f"Search leader, slot {selected['generation']}: recorded play",x=.04,ha='left',y=.98)
    fig.text(.04,.93,f"First development case {selected['case']} · {selected['reason']} · {selected['steps']} transitions · {selected['keys']}/2 keys",color=style.SECONDARY)
    indices=[0,len(selected['trace'])-1]
    facts=[]
    for i,index in enumerate(indices):
        frame=selected['trace'][index];world=frame['world']
        belief,actual,known=translated_belief(frame)
        correct,total=int(np.sum(known & (belief==actual))),int(known.sum())
        expected=selected['timeline'][index]
        if (correct,total)!=(expected['correct'],expected['audited']):
            raise ValueError('Replay coordinates disagree with saved cell audit')
        draw_map(axes[i,0],world['grid'],world['enemies'],world['agent'])
        yy,xx=np.where(belief==5)
        draw_map(axes[i,1],belief,list(zip(xx.tolist(),yy.tolist())),world['agent'],mismatches=known & (belief!=actual))
        prefix='First' if i==0 else 'Last'
        axes[i,0].set_title(f"{prefix} record · step {world['step']}\nHidden world (retrospective)",loc='left')
        axes[i,1].set_title(f"{prefix} record · step {world['step']}\nBelief: {correct}/{total} cells correct",loc='left')
        facts.append({'trace_index':index,'step':world['step'],'correct':correct,'audited':total})
    fig.text(.04,.072,'Circle = agent; × = enemy or remembered enemy; ◇ = key; □ = door; star = exit.',fontsize=10,color=style.SECONDARY)
    fig.text(.04,.025,'Pale = unknown; orange outlines = incorrect belief. Both records precede the chosen action.',fontsize=10,color=style.SECONDARY)
    style.save_figure(fig,out/'search-replay')
    return {'generation':selected['generation'],'case':selected['case'],'records':len(selected['trace']),'shown_records':facts}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/proposal-full-native-01')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/proposal/full-native-01')
    parser.add_argument('--render-export',action='store_true',
                        help='Render from committed compact evidence in --out; no runtime database required')
    args=parser.parse_args()
    campaign,out=args.input.resolve(),args.out.resolve()
    if campaign == out or out.is_relative_to(campaign):
        parser.error('Public report output must be outside the runtime input directory')
    out.mkdir(parents=True,exist_ok=True)
    if args.render_export:
        summary,slots,episodes,lineage,calls=[read(out/(name+'.json')) for name in
            ('summary','slots','episodes','lineage','model-calls')]
        events=jsonl(out/'native-events.jsonl')
        if any(value is None for value in (summary,slots,episodes,lineage,calls)):
            parser.error('Compact evidence is incomplete; no experiment will be executed')
    else:
        summary,slots,episodes,lineage,calls,events=evidence(campaign,out)
    style.apply_theme();figures=out/'figures';figures.mkdir(exist_ok=True)
    score_figure(slots,episodes,figures)
    lineage_figure(slots,lineage,events,figures)
    cost_outcomes_figure(slots,calls,summary,figures)
    replay=selected_replay(summary,figures)
    captions={
        'search-scores': 'Every admitted candidate slot with saved evidence. Small points are all individual development episodes; diamonds are arithmetic means. F = 0.6S + 0.4A, except invalid episodes have zero F. A is the within-episode pooled ratio of correct to reported in-bounds cell audits, averaged equally across episodes; unknown cells are excluded. Coverage is last-record reported cells / 225. Failed slots are shaded and no-evaluation failures marked × below zero, not assigned fabricated component scores. Pending slots are marked |. These fixed cases informed search; gains do not establish generalization or predictive learning.',
        'search-native': 'Native recorded parent ancestry (solid) and sampled inspirations (dotted), using birth islands recovered from sampling and migration history. Symbols identify native patch operators; color and explicit island labels identify islands. Open seed diamonds are administrative island copies and do not consume additional evaluations. A vertical mark without an ancestry edge denotes a proposal held without valid generated source, shown at its sampled parent island. Event counts come directly from saved native-events.jsonl. Generated sources, actual migrations, request-context checks, prompt records and retained states are separately exported. An event or lineage edge is evidence of operation, not causal benefit.',
        'search-outcomes-costs': 'All saved episode outcomes and descriptive nominal 95% Wilson escape intervals on reused development cases; the intervals do not account for selection. Provider time includes every saved actual attempt and unsuccessful returns. Cobalt provider bars are successful process returns; magenta segments include unsuccessful/error attempts. Usage ? means missing reported usage, never zero tokens. Panel D pairs program-save or proposal-hold timestamps with the summed elapsed time and count of provider attempts completed by then, showing the best valid development fitness so far. Labels are slot / completed call count. The held marker retains the previous incumbent and does not assign fitness to an unevaluated proposal. Any later auxiliary requests remain in panel C. Local embedding and evaluation timings remain separately tabulated in summary.json and resource-curve.csv; API list-price estimates do not measure subscription charges.'}
    if replay:
        captions['search-replay']=f"Development search leader, slot {replay['generation']}, chosen by highest mean F and earlier slot tie-break. The first saved case ({replay['case']}) is shown at the first and last of its {replay['records']} recorded decisions, preserving record order. Hidden world panels are retrospective and were unavailable to the candidate. Belief coordinates use the evaluator-recorded origin; remembered enemy occupancy can be stale and does not specify underlying terrain. Orange outlines mark incorrect reported cells. Both records precede actions and the last need not be a terminal-state snapshot. This is a selected development illustration, not an independent assessment. All first-case frames for every evaluated slot are retained in representative-replays.json.gz."
    dump(figures/'captions.json',captions)
    sources=[ROOT/'proposal/search_report.py',ROOT/'proposal/figures.py',ROOT/'scripts/run1_native_report.py',ROOT/'scripts/chromatic_fields.py',ROOT/'scripts/visual_theme.py']
    dump(figures/'rendering-manifest.json',{'style':style.PRESENTATION_VERSION,
        'command':(f'.venv/bin/python -m proposal.search_report --render-export --out {relative(out)}'
                   if args.render_export else
                   f'.venv/bin/python -m proposal.search_report --input {relative(campaign)} --out {relative(out)}'),
        'sources':[provenance(p) for p in sources],
        'selected_replay':replay,
        'inputs':[provenance(out/(name+'.json')) for name in ('summary','slots','episodes','lineage','model-calls','source-provenance')],
        'outputs':[provenance(p) for p in sorted(figures.glob('*')) if p.suffix in ('.svg','.pdf','.png')],
        'execution':'Saved-data aggregation and rendering only; no agents, environments, LLM calls or resampling.'})
    print(json.dumps({'slots':summary['slot_count'],'episodes':summary['saved_episodes'],
                      'best':summary['best'],'calls':summary['calls']['calls'],'out':relative(out)},indent=2))


if __name__=='__main__':
    main()
