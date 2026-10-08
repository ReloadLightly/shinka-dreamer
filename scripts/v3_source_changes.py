"""Describe every saved candidate's structural changes against its native parent.

This is a source inventory, not evidence that a syntactic change caused a score
change. Behavioral interpretation belongs in the selected-program review.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def structure(path):
    source = path.read_bytes()
    result = {'source_sha256': digest(source), 'lines': len(source.splitlines())}
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return result | {'parse_error': str(error)}
    declarations, module = {}, []
    for node in tree.body:
        representation = ast.dump(node, include_attributes=False)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in declarations:
                # Retain order and every definition when a program rebinds a name.
                declarations[node.name] += '\n' + representation
            else:
                declarations[node.name] = representation
        else:
            module.append(representation)
    return result | {'ast_nodes': sum(1 for _ in ast.walk(tree)),
                     'declarations': declarations, 'module_statements': module}


def report(directory):
    lineage_path = directory/'lineage.json'
    lineage = json.loads(lineage_path.read_text())
    by_id = {row['id']: row for row in lineage}
    rows = {}
    for row in lineage:
        # The seed has native island copies; it remains one candidate slot.
        previous = rows.get(row['generation'])
        if previous is None or previous['metadata'].get('_is_island_copy', False):
            rows[row['generation']] = row
    parsed = {generation: structure(directory/'programs'/f'gen_{generation}.py')
              for generation in rows}
    records = []
    for generation, row in sorted(rows.items()):
        current = parsed[generation]
        if current['source_sha256'] != row['source_sha256']:
            raise ValueError(f'Saved source differs from lineage for slot {generation}')
        parent = by_id.get(row['parent_id'])
        prior = parsed[parent['generation']] if parent else None
        record = {'generation': generation, 'native_id': row['id'],
                  'parent_id': row['parent_id'],
                  'parent_generation': parent['generation'] if parent else None,
                  'archive_inspiration_generations': [by_id[i]['generation'] for i in row['archive_inspiration_ids']],
                  'top_k_inspiration_generations': [by_id[i]['generation'] for i in row['top_k_inspiration_ids']],
                  'source_sha256': current['source_sha256'],
                  'valid': bool(row['correct']), 'lines': current['lines'],
                  'ast_nodes': current.get('ast_nodes'),
                  'native_patch_name': row['metadata'].get('patch_name'),
                  'native_patch_description': row['metadata'].get('patch_description'),
                  'native_description_scope': 'Proposal author description; not independent behavioral verification.'}
        if 'parse_error' in current:
            record['parse_error'] = current['parse_error']
        elif prior and 'parse_error' not in prior:
            new, old = current['declarations'], prior['declarations']
            changed = sorted(key for key in new.keys() & old.keys() if new[key] != old[key])
            record.update(added=sorted(new.keys()-old.keys()),
                          removed=sorted(old.keys()-new.keys()), changed=changed,
                          module_statements_changed=current['module_statements'] != prior['module_statements'],
                          world_model_step_changed='world_model_step' in changed,
                          planner_changed='planner' in changed,
                          lines_delta=current['lines']-prior['lines'],
                          ast_nodes_delta=current['ast_nodes']-prior['ast_nodes'])
        records.append(record)
    result = {'lineage_sha256': digest(lineage_path.read_bytes()),
              'slots': len(records),
              'scope': 'Top-level declaration AST comparison with actual native parent; ignores formatting/comments, retains literals, docstrings, nested bodies and module statements. Changed declarations can affect other unchanged functions through calls. Syntactic co-change does not establish causal model–planner synergy.',
              'candidates': records}
    (directory/'source-changes.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Every candidate: source changes', '', result['scope'], '',
             'Proposal names are author labels. All sources, attempted patches and failures are preserved alongside this inventory.', '',
             '| Slot | Parent | Valid | Lines (change) | Added declarations | Changed declarations | Removed declarations |',
             '|--:|--:|:--:|--:|:--|:--|:--|']
    for row in records:
        fields = ['; '.join(row.get(key, [])) or '—' for key in ('added', 'changed', 'removed')]
        delta = f" ({row['lines_delta']:+d})" if 'lines_delta' in row else ''
        parent = row['parent_generation'] if row['parent_generation'] is not None else '—'
        lines.append(f"| [{row['generation']}](programs/gen_{row['generation']}.py) | {parent} | {'Yes' if row['valid'] else 'No'} | {row['lines']}{delta} | {' | '.join(fields)} |")
    (directory/'source-changes.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'source_inventory_slots':len(records)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/campaign-v3'))
    report(parser.parse_args().artifacts)
