"""Frozen-program v3 statistics: paired episodes, pooled forecast uncertainty."""
import argparse
import gzip
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.stats import binomtest, beta
from dreamer.provenance import sha256

BOOTSTRAP_SEED = 7219517
BOOTSTRAP_REPLICATES = 5000
FORECASTS = ('brier_near','brier_audit','brier_threat','brier_destination',
             'brier_near_post_switch','brier_audit_post_switch')


def interval(values):
    finite=np.asarray(values)[np.isfinite(values)]
    return np.quantile(finite,[.025,.975]).tolist() if len(finite) else None


def clopper(k,n,alpha=.05):
    if not n: return [0.,1.]
    return [0. if not k else float(beta.ppf(alpha/2,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))]


def paired_binary(left,right,alpha=.05):
    a,b=np.asarray(left,dtype=bool),np.asarray(right,dtype=bool)
    assert a.shape==b.shape and a.ndim==1
    if not len(a) or not 0 < alpha < 1:
        raise ValueError('Nonempty paired cases and alpha in (0, 1) required')
    wins,losses=int((a&~b).sum()),int((~a&b).sum())
    n,m=len(a),wins+losses
    # Union bound for discordance probability and conditional win probability.
    q=clopper(m,n,alpha/2); theta=clopper(wins,m,alpha/2)
    bounds=[x*(2*y-1) for x in q for y in theta]
    result={'left_only':wins,'right_only':losses,'both':int((a&b).sum()),
            'neither':int((~a&~b).sum()),'cases':n,'difference':(wins-losses)/n,
            'confidence_interval':[min(bounds),max(bounds)], 'confidence_level':1-alpha,
            'mcnemar_exact_p':float(binomtest(wins,m,.5).pvalue) if m else 1.}
    if alpha == .05:
        result['ci95'] = result['confidence_interval']
    return result


def regime_interactions(data,plan,weights):
    """Prespecified descriptive differences between paired escape effects.

    Each component effect has a conservative 97.5% interval. Subtracting
    their endpoints gives at least 95% coverage by a union bound regardless
    of shared-case dependence. Shared-case bootstrap intervals are diagnostic.
    """
    result = {}
    for contrast in plan.get('regime_interactions', []):
        left,right=contrast['left'],contrast['right']
        first,second=contrast['regime_a'],contrast['regime_b']
        if left == right or first == second:
            raise ValueError('Regime interactions require distinct conditions and regimes')
        effects,case_effects = {}, {}
        for regime in (first,second):
            if (regime,left) not in data or (regime,right) not in data:
                raise ValueError('Regime interaction names an undeclared condition or regime')
            a=np.array([outcome(r,'escape') for r in data[regime,left]],dtype=int)
            b=np.array([outcome(r,'escape') for r in data[regime,right]],dtype=int)
            effects[regime]=paired_binary(a,b,alpha=.025)
            case_effects[regime]=a-b
        difference=case_effects[first]-case_effects[second]
        a,b=effects[first]['confidence_interval'],effects[second]['confidence_interval']
        sparse = any(effects[r]['left_only']+effects[r]['right_only'] < 10 for r in (first,second))
        key=f'{first}-minus-{second}/{left}-minus-{right}'
        if key in result:
            raise ValueError('Duplicate prespecified regime interaction')
        result[key]={'left':left,'right':right,'regime_a':first,'regime_b':second,
            'cases':len(difference),'difference':float(difference.mean()),
            'ci95_conservative':[a[0]-b[1],a[1]-b[0]],
            'ci95_shared_case_bootstrap':interval(weights@difference/len(difference)),
            'component_effects':effects,
            'cases_with_nonzero_interaction':int(np.count_nonzero(difference)),
            'sparse_discordance_warning':sparse,
            'bootstrap_degenerate':bool(np.all(difference == difference[0])),
            'scope':'Prespecified descriptive escape-effect interaction; no inference from differing within-regime significance.',
            'interval_method':'Conservative 95% union-bound difference of two 97.5% paired-effect intervals; no independence assumption.',
            'bootstrap_scope':'Diagnostic pointwise shared-case bootstrap; sparse discordance or constant observed interactions can give misleadingly narrow intervals.',
            'multiplicity':'Pointwise descriptive intervals, not simultaneous coverage across all regime interactions.'}
    return result


def holm(values):
    adjusted=[None]*len(values); running=0.
    for rank,index in enumerate(sorted(range(len(values)),key=values.__getitem__)):
        running=max(running,(len(values)-rank)*values[index])
        adjusted[index]=min(1.,running)
    return adjusted


def ratio(weights,sums,counts):
    den=weights@counts
    return np.divide(weights@sums,den,out=np.full(len(weights),np.nan),where=den>0)


def pooled(rows,key,weights):
    values=np.array([r['stats'].get(key,[0,0]) for r in rows],dtype=float)
    total,count=values.sum(axis=0)
    if not count: return None
    boot = ratio(weights, values[:,0], values[:,1])
    return {'loss':float(total/count),'sum':float(total),'targets':int(count),
            'contributing_episodes':int((values[:,1]>0).sum()),
            'ci95':interval(boot), 'bootstrap_retained':int(np.isfinite(boot).sum())}


def forecast_pair(left,right,key,weights):
    la=np.array([r['stats'].get(key,[0,0]) for r in left],float)
    ra=np.array([r['stats'].get(key,[0,0]) for r in right],float)
    if not la[:,1].sum() or not ra[:,1].sum(): return None
    lv=float(la[:,0].sum()/la[:,1].sum()); rv=float(ra[:,0].sum()/ra[:,1].sum())
    lb=ratio(weights,la[:,0],la[:,1]); rb=ratio(weights,ra[:,0],ra[:,1])
    relative=np.divide(rb-lb,rb,out=np.full(len(rb),np.nan),where=rb!=0)
    return {'left_loss':lv,'right_loss':rv,'difference':lv-rv,'ci95':interval(lb-rb),
            'relative_reduction':1-lv/rv if rv else None,'relative_reduction_ci95':interval(relative),
            'left_targets':int(la[:,1].sum()),'right_targets':int(ra[:,1].sum()),
            'left_loss_sum':float(la[:,0].sum()),'right_loss_sum':float(ra[:,0].sum()),
            'left_contributing_episodes':int((la[:,1]>0).sum()),
            'right_contributing_episodes':int((ra[:,1]>0).sum()),
            'bootstrap_retained':int(np.isfinite(lb-rb).sum()),
            'relative_bootstrap_retained':int(np.isfinite(relative).sum())}


def outcome(row,event):
    return (row['reason']=={'escape':'escaped','death':'caught','timeout':'timeout','invalid':'invalid'}[event]
            if event in ('escape','death','timeout','invalid') else
            row['keys']==2 if event=='two_keys' else bool(row['door']))


def behavior_pair(left,right):
    """Descriptive behavior changes; missing/invalid audits never imply equality."""
    if len(left) != len(right):
        raise ValueError('Behavior comparison requires paired cases')
    n=len(left)
    valid=[(a,b) for a,b in zip(left,right)
           if a.get('error') is None and b.get('error') is None
           and a.get('reason') != 'invalid' and b.get('reason') != 'invalid']
    result={'cases':n,'invalid_pairs':n-len(valid),
            'scope':'Descriptive sequence differences, not evidence of beneficial control or predictive adaptation.'}
    for name,key in (('actions','actions_sha256'),('recorded_physical_states','recorded_physical_states_sha256'),
                     ('physical_observations','observations_sha256')):
        audited=[(a['audit'][key],b['audit'][key]) for a,b in valid
                 if a.get('audit',{}).get(key) and b.get('audit',{}).get(key)]
        different=sum(a != b for a,b in audited)
        result[name]={'valid_audited_pairs':len(audited),'differing_pairs':different,
                      'same_pairs':len(audited)-different,'unavailable_valid_pairs':len(valid)-len(audited),
                      'differing_fraction_all_cases':different/n if n else None}
    result['recorded_physical_states']['definition']='Recorded pre-action world snapshots and post-transition enemy occupancy; the final agent position is not separately recorded. Equality is not full terminal-state identity.'
    return result


def state_audit_summary(rows):
    audits=[r.get('audit') for r in rows if r.get('audit') is not None]
    exports=[a for a in audits if a.get('parameters_exported') is True]
    checked=[a for a in audits if (a.get('localization_checks') or 0) > 0]
    return {'episodes':len(rows),'available_episode_audits':len(audits),
        'invalid_episodes':sum(r.get('error') is not None or r.get('reason') == 'invalid' for r in rows),
        'parameters_exported_episodes':len(exports),
        'parameters_constant_episodes':sum(a.get('parameters_constant') is True for a in exports),
        'parameters_changed_episodes':sum((a.get('parameter_change_steps') or 0) > 0 for a in exports),
        'parameters_unavailable_episodes':len(rows)-len(exports),
        'parameter_change_steps':sum(a.get('parameter_change_steps') or 0 for a in exports),
        'localization_audited_episodes':len(checked),
        'localization_unavailable_episodes':len(rows)-len(checked),
        'localization_checks':sum(a['localization_checks'] for a in checked),
        'localization_errors':sum(a.get('localization_errors') or 0 for a in checked),
        'episodes_with_localization_error':sum((a.get('localization_errors') or 0) > 0 for a in checked),
        'scope':'Exported state only; absent exports are unavailable, not constant or correct. Invalid partial traces can contribute audits. Parameter change does not establish informative learning.'}


def cpu_summary(rows,key,weights):
    measured=np.array([not r.get('cpu_measurement_unavailable',False) and key in r
                       and np.isfinite(r[key]) for r in rows],dtype=bool)
    values=np.array([r[key] if present else 0. for r,present in zip(rows,measured)])
    included=values[measured]
    boot=ratio(weights,values,measured.astype(float))
    return {'mean':float(included.mean()) if len(included) else None,'ci95':interval(boot),
        'sum':float(included.sum()),'median':float(np.median(included)) if len(included) else None,
        'p95':float(np.quantile(included,.95)) if len(included) else None,
        'measured_episodes':int(measured.sum()),'unavailable_episodes':int((~measured).sum()),
        'all_case_episodes':len(rows),'recorded_sum_is_lower_bound':not bool(measured.all()),
        'bootstrap_retained':int(np.isfinite(boot).sum()),
        'scope':'CPU mean/interval condition on measured episodes; missing CPU is not zero. Recorded sum is a lower bound if any measurements are unavailable. Outcome denominators remain all cases.'}


def cpu_pair(left,right,key,weights):
    measured=np.array([all(not r.get('cpu_measurement_unavailable',False) and key in r
                          and np.isfinite(r[key]) for r in (a,b)) for a,b in zip(left,right)],dtype=bool)
    differences=np.array([a[key]-b[key] if present else 0. for a,b,present in zip(left,right,measured)])
    boot=ratio(weights,differences,measured.astype(float))
    return {'difference':float(differences.sum()/measured.sum()) if measured.any() else None,
        'ci95':interval(boot),'measured_pairs':int(measured.sum()),'unavailable_pairs':int((~measured).sum()),
        'all_case_pairs':len(left),'bootstrap_retained':int(np.isfinite(boot).sum()),
        'scope':'Conditional on jointly measured CPU; missing values are not zero. Shared-case bootstrap recomputes the measured-pair denominator.'}


def analyze(rows,plan):
    n=plan['sample_size']
    conditions=[c['name'] for c in plan['conditions']]
    regimes=plan['regimes']
    required={(case, regime, name) for case in range(n) for regime in regimes for name in conditions}
    if n <= 0 or len(set(conditions)) != len(conditions) or len(set(regimes)) != len(regimes):
        raise ValueError('Positive sample size and unique declared conditions/regimes required')
    if len(rows) != len(required) or {(r['case'], r['regime'], r['condition']) for r in rows} != required:
        raise ValueError('Missing, duplicate, or undeclared outcome rows; keep every invalid case')
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    # Shared resamples preserve the case pairing across all regimes/conditions.
    weights=rng.multinomial(n,np.full(n,1/n),size=BOOTSTRAP_REPLICATES).astype(float)
    data={(regime,name):sorted([r for r in rows if r['regime']==regime and r['condition']==name],key=lambda r:r['case'])
          for regime in regimes for name in conditions}
    for group in data.values():
        assert [r['case'] for r in group]==list(range(n)), 'Missing/duplicate case; never drop invalids'
    result={'study':plan['study'],'cases_per_regime':n,'condition_episodes':len(rows),
            'independent_searches':1,'outcomes':{},'pairs':{},
            'bootstrap':{'seed':BOOTSTRAP_SEED,'replicates':BOOTSTRAP_REPLICATES,
                         'unit':'whole case; same weights across regimes and conditions'},
            'forecast_scope':'On-policy unless explicitly identified as matched recorded experience.',
            'post_switch_scope':'Survivor-conditioned on reaching recorded window; unconditional outcomes primary.'}
    for (regime,name),group in data.items():
        summary={'episodes':n}
        for event in ('escape','death','timeout','invalid','two_keys','door'):
            count=sum(outcome(r,event) for r in group)
            summary[event]={'count':count,'rate':count/n,'ci95':clopper(count,n)}
        for key in ('task','combined_score','keys','steps','seconds'):
            values=np.array([r.get(key,0.) for r in group])
            summary[key]={'mean':float(values.mean()),'ci95':interval(weights@values/n),
                          'sum':float(values.sum()),'median':float(np.median(values)),'p95':float(np.quantile(values,.95))}
        for key in ('candidate_cpu_seconds','evaluator_cpu_seconds'):
            summary[key] = cpu_summary(group,key,weights)
        success=[r['steps'] for r in group if r['reason']=='escaped']
        summary['successful_escape_steps']={'mean':float(np.mean(success)) if success else None,'episodes':len(success),'scope':'conditional on own success'}
        summary['forecasts']={key:pooled(group,key,weights) for key in FORECASTS}
        summary['state_audit'] = state_audit_summary(group)
        summary['switch_exposure']={'encountered':sum(r.get('encountered_switch',False) for r in group),
            **{key:sum(r.get('exposure',{}).get(key,0) for r in group) for key in (
               'post_switch_steps','visible_enemy_steps','consecutive_visible_steps',
               'post_switch_visible_enemy_steps','post_switch_consecutive_visible_steps')}}
        opportunities = [r.get('audit', {}).get('observed_transition_opportunities') for r in group]
        measured = [record for record in opportunities if record is not None]
        opportunity_keys = sorted({key for record in measured for key in record.get('all', {})})
        summary['observed_transition_opportunities'] = {
            'available_episode_audits': len(measured),
            'definition': measured[0]['definition'] if measured else None,
            'limitations': measured[0]['limitations'] if measured else None,
            'post_switch_scope': measured[0]['post_switch_scope'] if measured else None,
            'all_sums': {key: sum(record['all'].get(key, 0) for record in measured) for key in opportunity_keys},
            'post_switch_sums': {key: sum(record['post_switch'].get(key, 0) for record in measured) for key in opportunity_keys},
            'distinct_cell_scope': 'Distinct source-cell counts are summed per episode, not unique enemy identities across episodes.',
            'episodes_with_observed_contrast': sum(record['all']['observed_destination_contrast_opportunities'] > 0 for record in measured),
            'post_switch_episodes_with_observed_contrast': sum(record['post_switch']['observed_destination_contrast_opportunities'] > 0 for record in measured),
            'parameter_audit_episodes': sum(record.get('parameter_changes_without_counted_contrast') is not None for record in measured),
            'parameter_changes_without_counted_contrast': sum(record.get('parameter_changes_without_counted_contrast') or 0 for record in measured),
            'exported_parameter_change_steps': sum(r.get('audit', {}).get('parameter_change_steps') or 0 for r in group),
            'parameter_change_note': 'Parameter changes can reflect forgetting or regularization; they are not counts of informative learning updates.'}
        result['outcomes'][f'{regime}/{name}']=summary
    pairs=plan.get('contrasts',[])
    for contrast in pairs:
        left,right=contrast['left'],contrast['right']
        for regime in contrast.get('regimes',regimes):
            a,b=data[regime,left],data[regime,right]
            entry={'left':left,'right':right,'regime':regime,'family':contrast.get('family','descriptive')}
            entry['outcomes']={event:paired_binary([outcome(r,event) for r in a],[outcome(r,event) for r in b])
                               for event in ('escape','death','timeout','invalid','two_keys','door')}
            for key in ('task','combined_score','seconds','steps'):
                difference=np.array([x.get(key,0)-y.get(key,0) for x,y in zip(a,b)])
                entry[key]={'difference':float(difference.mean()),'ci95':interval(weights@difference/n)}
            entry['candidate_cpu_seconds'] = cpu_pair(a,b,'candidate_cpu_seconds',weights)
            entry['on_policy_forecasts']={key:forecast_pair(a,b,key,weights) for key in FORECASTS}
            entry['behavior'] = behavior_pair(a,b)
            result['pairs'][f'{regime}/{left}-minus-{right}']=entry
    for family in sorted({c.get('family','descriptive') for c in pairs}-{'descriptive'}):
        keys=[k for k,v in result['pairs'].items() if v['family']==family]
        adjusted=holm([result['pairs'][k]['outcomes']['escape']['mcnemar_exact_p'] for k in keys])
        for key,p in zip(keys,adjusted): result['pairs'][key]['outcomes']['escape']['holm_p']=p
    result['regime_interactions'] = regime_interactions(data, plan, weights)
    result['selected_matched'] = selected_matched(data, plan, weights)
    return result


MATCHED_CONDITIONS = ('fitted_frozen', 'fitted_online', 'known_law')
MATCHED_PAIRS = (('fitted_online', 'fitted_frozen'), ('known_law', 'fitted_frozen'),
                 ('known_law', 'fitted_online'))
MATCHED_FORECASTS = FORECASTS + ('brier_near_pre_switch', 'brier_audit_pre_switch',
    'brier_destination_post_switch', 'brier_near_terminal', 'brier_audit_terminal', 'brier_destination_terminal')


def bootstrap_weights(n):
    return np.random.default_rng(BOOTSTRAP_SEED).multinomial(n, np.full(n, 1/n),
                                                          size=BOOTSTRAP_REPLICATES).astype(float)


def binned_predictions(groups, pairs, weights):
    result = []
    for index in range(8):
        # Missing later windows remain zero-count episodes in shared resamples.
        temporary = {name: [{'stats': {'brier_near': row.get('bins', {}).get(str(index), {}).get('brier', [0., 0])}}
                            for row in rows] for name, rows in groups.items()}
        result.append({'bin': index, 'start_step': index*25, 'end_step': index*25+24,
            'midpoint': index*25+12.5,
            'scope': 'Survivor-conditioned on the shared recorded policy reaching this window.',
            'conditions': {name: pooled(rows, 'brier_near', weights) for name, rows in temporary.items()},
            'pairs': {left+'-minus-'+right: forecast_pair(temporary[left], temporary[right], 'brier_near', weights)
                      for left, right in pairs}})
    return result


def selected_matched(data, plan, weights):
    """Use fixed-prediction-planning interventions only if experience matches.

    Equality is checked on all prespecified cases; no post-hoc subset of matching
    trajectories becomes a purported unbiased learning comparison.
    """
    configured = plan.get('selected_matched_conditions')
    if configured:
        left, right = configured['online'], configured['frozen']
    else:
        online = [c['name'] for c in plan['conditions'] if c['variant'] == 'no_planning']
        frozen = [c['name'] for c in plan['conditions'] if c['variant'] == 'frozen_no_planning']
        if len(online) != 1 or len(frozen) != 1:
            return {'available': False, 'reason': 'No unique prespecified selected fixed-risk intervention pair.'}
        left, right = online[0], frozen[0]
    result = {}
    for regime in plan['regimes']:
        a, b = data[regime, left], data[regime, right]
        def equal_hash(x, y, key):
            first, second = x.get('audit', {}).get(key), y.get('audit', {}).get(key)
            return bool(first and second and first == second)
        matched = sum(equal_hash(x, y, 'trajectory_sha256') for x, y in zip(a, b))
        observations = sum(equal_hash(x, y, 'observations_sha256') for x, y in zip(a, b))
        maps = sum(equal_hash(x, y, 'map_position_sha256') for x, y in zip(a, b))
        all_valid = all(row.get('error') is None for row in a+b)
        available = matched == len(a) and observations == len(a) and all_valid
        entry = {'available': available, 'left': left, 'right': right, 'episodes': len(a),
                 'trajectory_matched_episodes': matched, 'physical_observation_matched_episodes': observations,
                 'map_position_matched_episodes': maps,
                 'invalid_episodes': sum(row.get('error') is not None for row in a+b),
                 'frozen_parameter_export_episodes': sum(row.get('audit', {}).get('parameters_exported', False) for row in b),
                 'frozen_parameters_constant_episodes': sum(row.get('audit', {}).get('parameters_constant') is True for row in b),
                 'online_parameter_change_episodes': sum((row.get('audit', {}).get('parameter_change_steps') or 0) > 0 for row in a),
                 'scope': 'Identical full fixed-risk experience; forecast evidence separate from on-policy control.'}
        if available:
            entry['forecasts'] = {key: forecast_pair(a, b, key, weights) for key in FORECASTS}
            entry['bins'] = binned_predictions({left: a, right: b}, [(left, right)], weights)
        else:
            entry['reason'] = 'Intervention experience or valid execution does not match on all cases; withhold matched-learning estimate.'
        result[regime] = entry
    return result


def analyze_matched(rows, plan, policy_rows=None):
    """Pool shadow losses over whole recorded episodes with shared case resampling."""
    n = plan['sample_size']
    regimes = plan['regimes']
    specification = plan.get('matched', {})
    conditions = tuple(specification.get('shadow_conditions', MATCHED_CONDITIONS))
    pairs = tuple((pair['left'], pair['right']) for pair in specification.get('shadow_pairs', [])) or MATCHED_PAIRS
    allowed = set(MATCHED_CONDITIONS) | {'selected_online', 'selected_frozen'}
    if (len(set(conditions)) != len(conditions) or not set(conditions) <= allowed
            or not set(MATCHED_CONDITIONS) <= set(conditions)):
        raise ValueError('Unknown or duplicate frozen shadow conditions')
    if any(left not in conditions or right not in conditions or left == right for left, right in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError('Invalid or duplicate frozen shadow comparison')
    if specification.get('selected') and set(conditions) != allowed:
        raise ValueError('Selected passive replay requires all five frozen shadow conditions')
    if any(name.startswith('selected_') for name in conditions) and not specification.get('selected'):
        raise ValueError('Selected shadow conditions require a frozen selected-source specification')
    required = {(case, regime) for case in range(n) for regime in regimes}
    if n <= 0 or len(set(regimes)) != len(regimes):
        raise ValueError('Positive sample size and unique regimes required')
    if len(rows) != len(required) or {(r['case'], r['regime']) for r in rows} != required:
        raise ValueError('Missing, duplicate, or undeclared matched recorded cases')
    weights = bootstrap_weights(n)
    data = {regime: sorted([r for r in rows if r['regime'] == regime], key=lambda r: r['case']) for regime in regimes}
    for regime, group in data.items():
        if [r['case'] for r in group] != list(range(n)):
            raise ValueError('Missing/duplicate matched recorded case in ' + regime)
    policy = {(r['case'], r['regime']): r for r in policy_rows or []
              if r['condition'] == plan.get('matched', {}).get('memory_condition')}
    result = {'study': plan['study'], 'cases_per_regime': n, 'recorded_policy_episodes': len(rows),
        'shadow_conditions': list(conditions), 'shadow_pairs': [{'left':a,'right':b} for a,b in pairs],
        'scope': 'Passive predictors receive identical recorded public observations and preceding actions; no new policy rollouts.',
        'known_law_scope': 'Privileged current dynamics with the same partial observations; not an optimal-policy bound.',
        'post_switch_scope': 'Conditional on the recorded memory policy reaching the switch/window; no unconditional control inference.',
        'bootstrap': {'seed': BOOTSTRAP_SEED, 'replicates': BOOTSTRAP_REPLICATES,
                      'unit': 'whole case, same resamples for every regime and shadow; recompute pooled loss ratios'},
        'regimes': {}}
    for regime, group in data.items():
        groups = {name: [] for name in conditions}
        missing, trajectory_verified = [], 0
        for record in group:
            if not record.get('policy_trajectory_sha256'):
                raise ValueError('Matched record lacks a policy trajectory identity')
            if policy_rows is not None:
                reference = policy.get((record['case'], regime))
                if reference is None or reference.get('audit', {}).get('trajectory_sha256') != record['policy_trajectory_sha256']:
                    raise ValueError('Matched source trace disagrees with the recorded policy outcome')
                trajectory_verified += 1
            if set(record.get('shadows', {}))-set(conditions):
                raise ValueError('Undeclared shadow record')
            for name in conditions:
                shadow = record.get('shadows', {}).get(name)
                if shadow is None:
                    missing.append({'case': record['case'], 'condition': name, 'error': record.get('error')})
                    shadow = {'case': record['case'], 'regime': regime, 'stats': {}, 'bins': {}, 'audit': {},
                              'error': 'Missing complete shadow record', 'frames': 0, 'missing_forecast_frames': 0}
                if shadow['case'] != record['case'] or shadow['regime'] != regime:
                    raise ValueError('Shadow case/regime identity mismatch')
                expected_hash = (specification.get('selected', {}).get('source_sha256') if name.startswith('selected_')
                                 else specification.get('source_sha256'))
                actual_hash = shadow.get('program_sha256', record.get('program_sha256') if name in MATCHED_CONDITIONS else None)
                if expected_hash and name in record.get('shadows', {}) and actual_hash != expected_hash:
                    raise ValueError('Matched shadow source differs from frozen plan')
                groups[name].append(shadow)
            available = [record.get('shadows', {}).get(name) for name in conditions]
            if all(shadow is not None for shadow in available):
                for key in MATCHED_FORECASTS:
                    if len({shadow['stats'].get(key, [0, 0])[1] for shadow in available}) != 1:
                        raise ValueError(f'Matched shadows use different target counts: {regime}/{record["case"]}/{key}')
                for index in range(8):
                    if len({shadow.get('bins', {}).get(str(index), {}).get('brier', [0, 0])[1] for shadow in available}) != 1:
                        raise ValueError('Matched shadows use different window target counts')
        frozen, online, known = (groups[name] for name in MATCHED_CONDITIONS)
        def matching_hash(key, names=conditions):
            return sum(all(row.get('audit', {}).get(key) for row in case_rows) and
                       len({row['audit'][key] for row in case_rows}) == 1
                       for case_rows in zip(*(groups[name] for name in names)))
        def parameter_pair_audit(online_name, frozen_name):
            a, b = groups[online_name], groups[frozen_name]
            return {'online':online_name, 'frozen':frozen_name,
                'map_position_matched_episodes':matching_hash('map_position_sha256', (online_name, frozen_name)),
                'frozen_parameter_export_episodes':sum(r.get('audit', {}).get('parameters_exported', False) for r in b),
                'frozen_parameters_constant_episodes':sum(r.get('audit', {}).get('parameters_constant') is True for r in b),
                'identical_initial_parameter_episodes':sum(x.get('audit', {}).get('first_parameters') is not None and
                    x['audit']['first_parameters'] == y.get('audit', {}).get('first_parameters') for x,y in zip(a,b)),
                'online_parameter_change_episodes':sum((r.get('audit', {}).get('parameter_change_steps') or 0)>0 for r in a),
                'scope':'Within-source predictive-parameter intervention; parameter changes need not imply informative updates.'}
        observation_matches = matching_hash('observations_sha256')
        audit = {'policy_trajectory_verified_against_outcomes': trajectory_verified,
            'policy_outcomes_supplied': policy_rows is not None,
            'physical_observation_matched_episodes': observation_matches,
            'map_position_matched_episodes': matching_hash('map_position_sha256', MATCHED_CONDITIONS),
            'frozen_parameter_export_episodes': sum(r.get('audit', {}).get('parameters_exported', False) for r in frozen),
            'frozen_parameters_constant_episodes': sum(r.get('audit', {}).get('parameters_constant') is True for r in frozen),
            'identical_initial_parameter_episodes': sum(a.get('audit', {}).get('first_parameters') is not None and
                a['audit']['first_parameters'] == b.get('audit', {}).get('first_parameters') for a, b in zip(frozen, online)),
            'online_parameter_change_episodes': sum((r.get('audit', {}).get('parameter_change_steps') or 0) > 0 for r in online),
            'shadow_error_episodes': {name: sum(r.get('error') is not None for r in records) for name, records in groups.items()},
            'missing_forecast_frames': {name: sum(r.get('missing_forecast_frames', 0) for r in records) for name, records in groups.items()},
            'missing_records': missing}
        audit['parameter_pairs'] = {'fitted_online-minus-fitted_frozen':parameter_pair_audit('fitted_online','fitted_frozen')}
        if 'selected_online' in groups and 'selected_frozen' in groups:
            audit['parameter_pairs']['selected_online-minus-selected_frozen'] = parameter_pair_audit('selected_online','selected_frozen')
        audit['map_position_scope'] = 'The legacy map-match count covers the three fitted comparators. Evolved/fitted representations need not match; within-source audits are separate.'
        target_episodes = {name: sum(r['stats'].get('brier_near', [0, 0])[1] > 0 for r in records)
                           for name, records in groups.items()}
        usable = not missing and observation_matches == n and all(target_episodes.values())
        entry = {'episodes': n, 'policy_frames': sum(r.get('policy_frames', 0) for r in group),
            'encountered_switch_episodes': sum(r.get('encountered_switch', False) for r in group),
            'terminal_transition_episodes': sum(r.get('terminal_transition_included', False) for r in group),
            'audits': audit, 'complete_matched_prediction_evidence': usable,
            'episodes_with_forecast_targets': target_episodes,
            'empty_policy_episode_cases': [r['case'] for r in group if not r.get('policy_frames', 0)],
            'failure_handling': 'Worker errors retain all targets using the frozen .5 fallback; missing whole records make comparisons unavailable. Empty policy episodes stay in case resampling with zero forecast targets, and remain in unconditional outcome denominators.',
            'forecasts': {name: {key: pooled(records, key, weights) for key in MATCHED_FORECASTS}
                          for name, records in groups.items()},
            'pairs': {}, 'bins': []}
        if usable:
            entry['pairs'] = {left+'-minus-'+right: {
                'left': left, 'right': right,
                'forecasts': {key: forecast_pair(groups[left], groups[right], key, weights) for key in MATCHED_FORECASTS}}
                for left, right in pairs}
            entry['bins'] = binned_predictions(groups, pairs, weights)
        result['regimes'][regime] = entry
    return result


def write_jsonlines_gzip(path, rows):
    """Deterministic compressed public evidence, with no seeds or hidden traces."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as output:
        with gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8') as handle:
                for row in rows:
                    if 'seed' in row or 'trace' in row:
                        raise ValueError('Private seed/trace cannot enter compact evidence')
                    handle.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n')
    temporary.replace(path)


def export_compact(raw, plan_path, episode_path, matched_path):
    raw, plan_path = Path(raw), Path(plan_path)
    plan = json.loads(plan_path.read_text())
    frozen_path = raw / 'frozen-plan.json'
    if json.loads(frozen_path.read_text()) != plan:
        raise ValueError('Raw assessment and supplied frozen plan differ')
    manifest = json.loads((raw / 'manifest.json').read_text())
    registration = json.loads((raw / 'pool-manifest.json').read_text())
    if manifest['plan_sha256'] != sha256(plan_path) or registration['original_plan_sha256'] != sha256(plan_path):
        raise ValueError('Raw assessment plan hash mismatch')
    if registration['frozen_plan_sha256'] != sha256(frozen_path):
        raise ValueError('Frozen plan checkpoint changed')
    if manifest['episode_pool'] != registration['pool']:
        raise ValueError('Assessment pool registration mismatch')
    if manifest.get('evaluation') != plan.get('evaluation') or manifest.get('conditions') != plan['conditions'] or manifest.get('regimes') != plan['regimes']:
        raise ValueError('Assessment conditions or evaluator differ from frozen plan')
    completion = json.loads((raw / 'execution-complete.json').read_text())
    expected = plan['sample_size']*len(plan['conditions'])*len(plan['regimes'])
    if completion['cases'] != plan['sample_size'] or completion['condition_episodes'] != expected:
        raise ValueError('Assessment completion count differs from plan')
    paths = sorted((raw / 'cases').glob('*.json'))
    if {p.name for p in paths} != set(completion['case_file_sha256']):
        raise ValueError('Unexpected/missing complete case checkpoints')
    rows = []
    sources = {c['name']: c['program_sha256'] for c in plan['conditions']}
    for path in paths:
        if sha256(path) != completion['case_file_sha256'][path.name]:
            raise ValueError('Case checkpoint changed: ' + path.name)
        case = json.loads(path.read_text())
        for row in case['conditions']:
            if row['case'] != case['case'] or row['program_sha256'] != sources[row['condition']]:
                raise ValueError('Outcome checkpoint source/case mismatch')
            rows.append(row)
    required = {(case, regime, condition) for case in range(plan['sample_size'])
                for regime in plan['regimes'] for condition in sources}
    if len(rows) != expected or {(r['case'], r['regime'], r['condition']) for r in rows} != required:
        raise ValueError('Incomplete/duplicate outcome denominator')
    variants = {c['name']: c['variant'] for c in plan['conditions']}
    if any(r['variant'] != variants[r['condition']] for r in rows):
        raise ValueError('Outcome variant differs from the frozen condition')
    if plan.get('replays'):
        from scripts.v3_assessment import verify_retained_replay
        retained = completion.get('passive_replays', {})
        if retained.get('retention_specification') != plan['replays']:
            raise ValueError('Passive replay retention differs from frozen plan')
        for row in rows:
            verify_retained_replay(raw, row['case'], row['regime'], row['condition'], plan['replays'], row)
    matched = []
    match_paths = sorted((raw / 'matched').glob('*.json'))
    registered = completion.get('matched', {}).get('file_sha256', {})
    if {p.name for p in match_paths} != set(registered):
        raise ValueError('Unexpected/missing matched checkpoints')
    for path in match_paths:
        if sha256(path) != registered[path.name]:
            raise ValueError('Matched checkpoint changed: ' + path.name)
        record = json.loads(path.read_text())
        if record.get('source_sha256') != plan.get('matched', {}).get('source_sha256'):
            raise ValueError('Matched comparator source mismatch')
        if record.get('selected_source_sha256') != plan.get('matched', {}).get('selected', {}).get('source_sha256'):
            raise ValueError('Matched selected source mismatch')
        matched.append(record)
    if plan.get('matched'):
        expected_matched = {(case, regime) for case in range(plan['sample_size']) for regime in plan['regimes']}
        if len(matched) != len(expected_matched) or {(r['case'], r['regime']) for r in matched} != expected_matched:
            raise ValueError('Missing/duplicate recorded matched episodes')
    rows.sort(key=lambda r: (r['case'], plan['regimes'].index(r['regime']), r['condition']))
    matched.sort(key=lambda r: (r['case'], plan['regimes'].index(r['regime'])))
    write_jsonlines_gzip(episode_path, rows)
    write_jsonlines_gzip(matched_path, matched)
    return rows, matched, {'raw_completion_sha256': sha256(raw / 'execution-complete.json'),
                           'raw_manifest_sha256': sha256(raw / 'manifest.json')}


def close_analysis(plan_path, paths, closure_path, provenance, condition_episodes, matched_policy_episodes):
    """Examples can be selected only after both frozen statistical outputs exist."""
    from scripts.v3_assessment import atomic_create
    plan = json.loads(Path(plan_path).read_text())
    required = [paths[0], paths[2], paths[3]] + ([paths[1]] if plan.get('matched') else [])
    if any(not Path(path).is_file() for path in required):
        raise ValueError('Both statistical analyses and their required evidence must finish before closure')
    closure = {'status': 'analysis-complete', 'plan_sha256': sha256(plan_path),
        'analysis_source_sha256': sha256(__file__),
        'files': {Path(path).name: sha256(path) for path in paths if Path(path).exists()},
        'raw_completion_sha256': provenance.get('raw_completion_sha256'),
        'condition_episodes': condition_episodes, 'matched_policy_episodes': matched_policy_episodes,
        'cases_per_regime': plan['sample_size'], 'analysis_precedes_example_selection': True}
    atomic_create(closure_path, closure)
    return closure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', help='Completed private assessment directory; verify and export compact evidence first')
    parser.add_argument('--data',default='artifacts/campaign-v3/assessment/episodes.jsonl.gz')
    parser.add_argument('--matched-data',default='artifacts/campaign-v3/assessment/matched-episodes.jsonl.gz')
    parser.add_argument('--plan',default='artifacts/campaign-v3/assessment/preregistration.json')
    parser.add_argument('--out',default='artifacts/campaign-v3/assessment/analysis.json')
    parser.add_argument('--matched-out',default='artifacts/campaign-v3/assessment/matched-analysis.json')
    parser.add_argument('--closure',default='artifacts/campaign-v3/assessment/analysis-closure.json')
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    provenance = {}
    if args.raw:
        rows, matched, provenance = export_compact(args.raw, args.plan, args.data, args.matched_data)
    else:
        if Path(args.out).exists():
            previous = json.loads(Path(args.out).read_text()).get('inputs', {})
            if previous.get('data_sha256') == sha256(args.data) and previous.get('plan_sha256') == sha256(args.plan):
                provenance = {key: previous[key] for key in ('raw_completion_sha256', 'raw_manifest_sha256') if key in previous}
        with gzip.open(args.data,'rt') as handle: rows=[json.loads(line) for line in handle]
        if Path(args.matched_data).exists():
            with gzip.open(args.matched_data,'rt') as handle: matched=[json.loads(line) for line in handle]
        elif plan.get('matched'):
            raise ValueError('Frozen matched assessment needs its compact episode evidence')
        else:
            matched=[]
    result=analyze(rows,plan)
    inputs={'data_sha256':sha256(args.data),'plan_sha256':sha256(args.plan),'analysis_sha256':sha256(__file__), **provenance}
    result['inputs']=inputs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    matched_result = analyze_matched(matched, plan, policy_rows=rows) if plan.get('matched') else {'available': False, 'reason': 'No matched replay was frozen in this plan.'}
    matched_result['inputs'] = {**inputs, 'matched_data_sha256': sha256(args.matched_data) if Path(args.matched_data).exists() else None}
    Path(args.matched_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.matched_out).write_text(json.dumps(matched_result, indent=2, allow_nan=False)+'\n')
    close_analysis(args.plan, [args.data, args.matched_data, args.out, args.matched_out], args.closure,
                   provenance, len(rows), len(matched))
    print(json.dumps({'episodes':len(rows),'matched_records':len(matched),'out':args.out,
                      'matched_out':args.matched_out,'closure':args.closure}))


if __name__=='__main__': main()
