"""Winner aliasing must not lose paired run identity or reverse the contrast."""
import numpy as np
import pytest

from proposal.study_report import paired_values
from proposal.study_statistics import crossed_intervals


def winners():
    return [{'run_id': f'{arm}-r{replicate}', 'arm': arm, 'replicate': replicate,
             'actual_all_role_calls': 24, 'common_call_prefix': 24,
             'primary': {'panel_program': 'same' if arm == 'rewrite' else f'full{replicate}'}}
            for replicate in (1, 2, 3) for arm in ('full', 'rewrite')]


def test_shared_rewrite_source_keeps_three_pairs_and_full_minus_rewrite_sign():
    values = {'same': np.full((4, 2), .5), **{f'full{i}': np.full((4, 2), .5 + i / 10) for i in (1, 2, 3)}}
    differences, pairs = paired_values(list(reversed(winners())), values, 'primary')
    assert differences.shape == (3, 4, 2)
    np.testing.assert_allclose(differences[:, 0, 0], [.1, .2, .3])
    assert [pair['replicate'] for pair in pairs] == [1, 2, 3]
    assert all(pair['rewrite_program'] == 'same' for pair in pairs)


def test_identical_policies_have_exactly_zero_effect_and_intervals():
    values = {key: np.ones((4, 2)) for key in ('same', 'full1', 'full2', 'full3')}
    differences, _ = paired_values(winners(), values, 'primary')
    result = crossed_intervals(differences, draws=100)
    assert result['mean_difference'] == [0., 0.]
    assert result['crossed_interval'] == [[0., 0.], [0., 0.]]


def test_missing_rewrite_identity_is_rejected_even_when_source_exists():
    records = winners()
    records[1]['arm'] = 'full'
    with pytest.raises(ValueError, match='exactly full and rewrite'):
        paired_values(records, {}, 'primary')
