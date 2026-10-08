"""Exploratory review of saved v3 interim decisions; never executes a policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamer.provenance import sha256
from scripts.v3_examples import digest, display_index, load_trace, public_frame, write_immutable

PAIR = ('selected', 'selected_frozen')
REGIMES = ('uniform', 'stationary', 'switch')


def final_destination(frame):
    """Recover the last attempted destination from already recorded terrain."""
    world, action = frame['world'], frame['action']
    pos = tuple(world['agent'])
    grid = world['grid']
    def blocked(point):
        x, y = point
        if not (0 <= x < 15 and 0 <= y < 15):
            return True
        cell = grid[y][x]
        if cell == 3 and not world['door_open']:
            opened = (action['interact'] and world['keys'] == 2 and
                      abs(x-pos[0])+abs(y-pos[1]) == 1)
            return not opened
        return cell == 1
    dx, dy = action['move']
    target = pos[0]+dx, pos[1]+dy
    if blocked(target) or (dx and dy and (blocked((pos[0]+dx,pos[1])) or blocked((pos[0],pos[1]+dy)))):
        return pos
    return target


def terminal_review(row, trace):
    result = {'reason':row['reason'], 'steps':row['steps'], 'keys':row['keys'], 'door':row['door']}
    if row['reason'] == 'caught' and trace:
        frame = trace[-1]
        destination = final_destination(frame)
        before = destination in {tuple(p) for p in frame['world']['enemies']}
        after = destination in {tuple(p) for p in frame['next_enemies']}
        result['collision_class'] = ('both_entry_and_post_transition' if before and after else
                                     'entry_only' if before else 'post_transition_only' if after else 'unexplained')
    if row['reason'] == 'timeout' and trace:
        tail = trace[-25:]
        result['last25'] = {'frames':len(tail),
            'waiting_actions':sum(f['action']['move'] == [0,0] for f in tail),
            'distinct_recorded_positions':len({tuple(f['world']['agent']) for f in tail}),
            'target_kinds':dict(Counter(f['model'].get('diagnostic_planning',{}).get('target_kind','unavailable') for f in tail))}
    return result


def reviewed_choice(frame, other_move):
    # Use the frozen publication allowlist; never copy raw learning state/laws.
    public = public_frame(frame)
    diagnostic = public['model']['diagnostic_planning']
    choices = {tuple(c['move']):c for c in diagnostic['choices']}
    chosen = choices.get(tuple(public['action']['move']))
    other = choices.get(tuple(other_move))
    allow = ('move','score','hazard','potential','continuation')
    compact = lambda c: {key:c[key] for key in allow} if c else None
    return {'move':public['action']['move'], 'target_kind':diagnostic['target_kind'],
        'chosen':compact(chosen), 'counterpart_move_under_this_model':compact(other),
        'counterpart_minus_chosen_score':other['score']-chosen['score'] if other and chosen else None}


def first_divergence(left, right, left_row):
    _, index = display_index(left, right)
    if index is None:
        return None
    a, b = left[index], right[index]
    da, db = a['model']['diagnostic_planning'], b['model']['diagnostic_planning']
    fields = [{tuple(p[:2]):p[2] for p in f['model']['enemy']} for f in (a,b)]
    gap = max([abs(fields[0].get(p,a['model']['default_enemy'])-fields[1].get(p,b['model']['default_enemy']))
               for p in fields[0].keys()|fields[1].keys()] or [0.])
    decisions = {PAIR[0]:reviewed_choice(a,b['action']['move']), PAIR[1]:reviewed_choice(b,a['action']['move'])}
    relevant = [d[key] for d in decisions.values() for key in ('chosen','counterpart_move_under_this_model')]
    switch = left_row.get('switch_step')
    return {'step':a['obs']['step'],
        'same_recorded_world':a['world'] == b['world'],
        'same_map_and_localization':a['model']['terrain'] == b['model']['terrain'] and a['model']['position'] == b['model']['position'],
        'same_navigation_target':da['target'] == db['target'],
        'maximum_exported_occupancy_difference':gap,
        'online_learning_updates':a['model']['learning']['updates'],
        'online_effective_learning_updates':a['model']['learning']['effective_updates'],
        'all_compared_immediate_hazards_zero':all(c is not None and abs(c['hazard']) < 1e-12 for c in relevant),
        'replacement_outcome_already_observed':switch is not None and a['obs']['step'] >= switch,
        'decisions':decisions,
        'interpretation':'First different choice under identical physical history. Subsequent terminal outcome does not identify the local counterfactual value of this choice.'}


def bottlenecks(rows):
    result = {}
    for regime in REGIMES:
        for condition in PAIR:
            group = [r for r in rows if r['regime'] == regime and r['condition'] == condition]
            stalled = [r for r in group if r['reason'] == 'timeout']
            opportunities = [r.get('audit',{}).get('observed_transition_opportunities',{}) for r in group]
            result[f'{regime}/{condition}'] = {
                'episodes':len(group), 'outcomes':dict(Counter(r['reason'] for r in group)),
                'timeout_task_states':dict(Counter(f'keys={r["keys"]};door={bool(r["door"])}' for r in stalled)),
                'encountered_switch':sum(r.get('encountered_switch',False) for r in group),
                'post_switch_steps':sum(r.get('exposure',{}).get('post_switch_steps',0) for r in group),
                'episodes_with_post_switch_observed_contrast_proxy':sum(o.get('post_switch',{}).get('observed_destination_contrast_opportunities',0)>0 for o in opportunities),
                'post_switch_scope':'Survivor-conditioned experience; proxy counts observed occupancy contrasts, not identified enemy transitions.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='results/campaign-v3-assessment')
    parser.add_argument('--interim', default='artifacts/campaign-v3/assessment-interim140')
    args = parser.parse_args()
    raw, interim = Path(args.raw), Path(args.interim)
    protocol_path = interim/'interim-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    original_plan = ROOT/protocol['original_plan']['path']
    if sha256(original_plan) != protocol['original_plan']['sha256']:
        raise ValueError('Original frozen plan changed')
    plan = json.loads(original_plan.read_text())
    if json.loads((raw/'frozen-plan.json').read_text()) != plan:
        raise ValueError('Raw and original frozen plans differ')
    sources = {c['name']:c['program_sha256'] for c in plan['conditions']}
    rows = []
    for name, expected in protocol['case_file_sha256'].items():
        path = raw/'cases'/name
        if sha256(path) != expected:
            raise ValueError('Interim outcome checkpoint changed')
        case = json.loads(path.read_text())
        for row in case['conditions']:
            if row['program_sha256'] != sources[row['condition']]:
                raise ValueError('Saved source identity changed')
            rows.append(row)
    keyed = {(r['case'],r['regime'],r['condition']):r for r in rows}
    reviews, replay_hashes = [], {}
    for regime in REGIMES:
        for case in range(32):
            pair_rows = [keyed[case,regime,name] for name in PAIR]
            traces = [load_trace(raw,row) for row in pair_rows]
            for row in pair_rows:
                replay_hashes[row['audit']['retained_replay']['path']] = row['audit']['retained_replay']['sha256']
            reviews.append({'case':case,'regime':regime,
                'first_action_divergence':first_divergence(*traces,pair_rows[0]),
                'terminal':{name:terminal_review(row,trace) for name,row,trace in zip(PAIR,pair_rows,traces)}})
    summaries = {}
    for regime in REGIMES:
        group = [r for r in reviews if r['regime'] == regime]
        different = [r for r in group if r['first_action_divergence'] is not None]
        details = [r['first_action_divergence'] for r in different]
        summaries[regime] = {'paired_cases':len(group),'first_action_divergence_cases':len(different),
            'divergences_with_same_map_and_localization':sum(d['same_map_and_localization'] for d in details),
            'divergences_with_same_target':sum(d['same_navigation_target'] for d in details),
            'divergences_with_changed_forecast_gt_1e12':sum(d['maximum_exported_occupancy_difference']>1e-12 for d in details),
            'divergences_with_positive_learning_updates':sum(d['online_learning_updates']>0 for d in details),
            'divergences_with_zero_immediate_hazards_for_both_moves':sum(d['all_compared_immediate_hazards_zero'] for d in details),
            'divergences_after_replacement_outcome_observed':sum(d['replacement_outcome_already_observed'] for d in details),
            'outcome_discordant_case_indices':[r['case'] for r in group if r['terminal'][PAIR[0]]['reason'] != r['terminal'][PAIR[1]]['reason']],
            'terminal_collision_classes':{name:dict(Counter(r['terminal'][name]['collision_class'] for r in group if 'collision_class' in r['terminal'][name])) for name in PAIR}}
    invalid = [r for r in rows if r['reason'] == 'invalid']
    development_path = ROOT/'artifacts/campaign-v3/mechanism-selected/runtime-review.json'
    development = json.loads(development_path.read_text())
    report = {'status':'exploratory saved-data interim decision review; full preregistered assessment incomplete',
        'new_environment_episodes':0,'new_model_calls':0,'candidate_executions':0,
        'scope':{'interim_cases':len(protocol['observed_case_indices']),'retained_prefix_cases_per_regime':32,
                 'retained_condition_episodes':192,'regimes':list(REGIMES),'retained_conditions':list(PAIR)},
        'inputs':{'interim_protocol_sha256':sha256(protocol_path),'original_plan_sha256':sha256(original_plan),
                  'review_script_sha256':sha256(__file__),'publication_allowlist_sha256':sha256(ROOT/'scripts/v3_examples.py'),
                  'retained_replay_inventory_sha256':digest(replay_hashes),'development_runtime_review_sha256':sha256(development_path)},
        'retained_prefix_summaries':summaries,'all_interim_task_bottlenecks':bottlenecks(rows),
        'invalid_execution_review':{'episodes':len(invalid),'by_condition':dict(Counter(r['condition'] for r in invalid)),
            'errors':dict(Counter(r['error'] for r in invalid)),
            'candidate_cpu_seconds_range':[min(r['candidate_cpu_seconds'] for r in invalid),max(r['candidate_cpu_seconds'] for r in invalid)] if invalid else None,
            'classification':'Worker SIGKILL with CPU near the fixed10s cap is consistent with resource exhaustion; exit code alone does not prove its operating-system cause. Invalids remain in every task denominator.'},
        'development_mechanism_context':{'scope':development['scope'],'decision_checks':development['decision_audit']['all_recorded_checks_pass'],
            'choice_frames':development['decision_audit']['choice_frames'],'choice_score_mismatches':development['decision_audit']['choice_score_mismatches'],
            'changed_action_pairs':{r:development['regimes'][r]['predictive_versus_frozen_behavior']['actions']['differing_pairs'] for r in REGIMES},
            'escape_discordant_pairs':{r:sum(development['regimes'][r]['predictive_minus_frozen_escape'][k] for k in ('left_only','right_only')) for r in REGIMES}},
        'case_reviews':reviews,
        'limitations':['Retained first32 cases are a small bounded prefix; review is exploratory and all these cases are exposed.',
            'A changed choice establishes decision sensitivity, not beneficial control; later terminal outcomes do not identify a local action counterfactual.',
            'Immediate horizon1 occupancy, entry-plus-post-transition collision hazard, and stage2/3 continuation costs are different quantities.',
            'Most initial divergences in switch episodes can precede observed replacement-law experience.',
            'No true movement laws, environment seeds, raw private traces or inferred-law vectors are published.']}
    write_immutable(interim/'decision-review.json',report)
    print(json.dumps({'retained_prefix_summaries':summaries,'task_bottlenecks':report['all_interim_task_bottlenecks'],
                      'invalid_execution_review':report['invalid_execution_review']},indent=2))


if __name__ == '__main__':
    main()
