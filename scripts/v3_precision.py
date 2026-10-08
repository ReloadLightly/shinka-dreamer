"""Choose the frozen assessment size from development escape discordance only.

This is a prospective planning calculation, not an assessment stopping rule.
The 3 percentage point target is close to the historical selected-versus-memory
effect and large enough to matter for this high-escape task. Development cases
were used by search, so the reported power is conditional on planning assumptions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import beta, binom
from dreamer.provenance import sha256
from v3_analysis import paired_binary

EFFECT = .03
POWER = .80
PER_TEST_ALPHA = .025  # Two registered primary regime tests; conservative Holm case.
MIN_CASES = 1024
INCREMENT = 256
DISCORDANCE_FLOOR = .08


def exact_power(n, discordance, effect=EFFECT, alpha=PER_TEST_ALPHA):
    """Unconditional power of the two-sided exact paired binomial test."""
    if not effect <= discordance <= 1:
        raise ValueError('Discordance must contain the proposed absolute effect')
    k = np.arange(n+1)
    cut = binom.ppf(alpha/2, k, .5).astype(int)
    cut -= binom.cdf(cut, k, .5) > alpha/2
    theta = (1+effect/discordance)/2
    rejection = binom.cdf(cut, k, theta)+binom.sf(k-cut-1, k, theta)
    rejection[0] = 0
    return float(binom.pmf(k, n, discordance) @ rejection)


def interval_at_rounded_expected_counts(n, discordance, effect):
    wins = round(n*(discordance+effect)/2)
    losses = round(n*(discordance-effect)/2)
    left = [True]*wins+[False]*(n-wins)
    right = [False]*wins+[True]*losses+[False]*(n-wins-losses)
    result = paired_binary(left, right)
    return {k: result[k] for k in ('cases','left_only','right_only','difference','ci95')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism', required=True, help='Completed selected-program development audit directory')
    parser.add_argument('--review', required=True, help='Source-specific review declaring freezing_is_coherent')
    parser.add_argument('--out', default='artifacts/campaign-v3/assessment/precision.json')
    args = parser.parse_args()
    directory = Path(args.mechanism)
    manifest = json.loads((directory/'manifest.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    review = json.loads(Path(args.review).read_text())
    protocol = json.loads((ROOT/'artifacts/campaign-v3/protocol.json').read_text())
    if manifest['layouts'] != 24 or manifest['champion_selection']:
        raise ValueError('Use the full, separately labeled selected-program development audit')
    if manifest['pool_sha256'] != protocol['splits']['development']['pool_sha256']:
        raise ValueError('Use the registered development pool only')
    if (sha256(args.review) != manifest['review_sha256'] or
            review['program_sha256'] != manifest['program_sha256'] or
            not review.get('freezing_is_coherent')):
        raise ValueError('A source-bound coherent freezing review is required')
    if summary['manifest'] != manifest or not summary['all_frozen_parameters_constant']:
        raise ValueError('Completed runtime verification of freezing is required')
    rows = [json.loads(p.read_text()) for p in sorted((directory/'episodes').glob('*.json'))]
    expected = {(case, regime, variant) for case in range(24)
                for regime in ('uniform','stationary','switch')
                for variant in ('predictive','frozen','no_planning','frozen_no_planning')}
    if len(rows) != len(expected) or {(r['case'],r['regime'],r['variant']) for r in rows} != expected:
        raise ValueError('Development audit incomplete or duplicated')
    evidence = {}
    for regime in ('stationary','switch'):
        keyed = {(r['case'],r['variant']):r for r in rows if r['regime']==regime}
        # Invalid executions remain non-escapes. Outcome direction is not used
        # to choose the sample size; only the number of discordant pairs matters.
        discordant = sum((keyed[case,'predictive']['reason']=='escaped') !=
                         (keyed[case,'frozen']['reason']=='escaped') for case in range(24))
        upper = float(beta.ppf(.95, discordant+1, 24-discordant)) if discordant < 24 else 1.
        evidence[regime] = {'cases':24, 'discordant_escape_pairs':discordant,
                            'discordance':discordant/24, 'one_sided_95_upper':upper}
    q = max(DISCORDANCE_FLOOR, *(r['one_sided_95_upper'] for r in evidence.values()))
    n = MIN_CASES
    while exact_power(n, q) < POWER:
        n += INCREMENT
    result = {
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'development_only':True, 'assessment_pool_drawn':False,
        'program_sha256':manifest['program_sha256'],
        'development_manifest_sha256':sha256(directory/'manifest.json'),
        'development_summary_sha256':sha256(directory/'summary.json'),
        'mechanism_review_sha256':sha256(args.review),
        'script_sha256':sha256(__file__), 'sample_size_per_regime':n,
        'meaningful_escape_difference':EFFECT, 'target_power':POWER,
        'per_test_alpha':PER_TEST_ALPHA,
        'multiplicity':'Two primary regime tests use Holm; planning uses alpha=.025 for each.',
        'rule':{'minimum_cases':MIN_CASES, 'round_up_increment':INCREMENT,
                'discordance_floor':DISCORDANCE_FLOOR,
                'planning_discordance':'maximum one-sided 95% development upper bound across stationary/switch, or floor'},
        'development':evidence, 'planning_discordance':q,
        'exact_planning_power':exact_power(n,q),
        'ci_at_rounded_expected_counts_null':interval_at_rounded_expected_counts(n,q,0),
        'ci_at_rounded_expected_counts_meaningful_effect':interval_at_rounded_expected_counts(n,q,EFFECT),
        'precision_note':'Intervals are evaluated at rounded expected cell counts, not averaged over repeated sampling.',
        'sensitivity':[{'discordance':v, 'power':exact_power(n,v)}
                       for v in sorted(set([.03,.08,.15,.25,.5,1.,q]))],
        'limitations':'Conditional planning power is not guaranteed: only 24 reused development cases per regime; the selected program was searched on them. Individual 95% discordance bounds are neither a simultaneous two-regime guarantee nor selection-adjusted intervals. Conservative planning bounds and the 1024-case floor reduce, but do not eliminate, this uncertainty. Final n is frozen regardless of assessment significance.',
        'model_calls':0, 'additional_environment_episodes':0}
    target = Path(args.out)
    if target.exists():
        raise FileExistsError('Precision record already exists; preserve its original freeze')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'sample_size_per_regime':n,'planning_discordance':q,
                      'power':result['exact_planning_power']}))


if __name__ == '__main__':
    main()
