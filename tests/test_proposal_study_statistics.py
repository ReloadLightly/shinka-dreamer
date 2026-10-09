import numpy as np
import pytest

from proposal.study_statistics import crossed_intervals


def test_identical_repetitions_do_not_multiply_case_sample_size():
    cases = np.arange(20, dtype=float).reshape(1, 20, 1)
    result = crossed_intervals(np.repeat(cases, 3, axis=0), draws=1000)
    assert result['crossed_interval'][0] == result['individual_paired_case_intervals'][0][0]
    assert result['individual_paired_case_intervals'][0] == result['individual_paired_case_intervals'][2]


def test_repetition_variation_survives_constant_case_values():
    data = np.repeat(np.array([-1., 0., 1.])[:, None, None], 512, axis=1)
    result = crossed_intervals(data)
    assert result['mean_difference'] == [0.]
    assert result['crossed_interval'] == [[-1., 1.]]
    assert result['individual_paired_case_intervals'] == [[[-1., -1.]], [[0., 0.]], [[1., 1.]]]


def test_all_metrics_use_shared_resampling():
    base = np.arange(60, dtype=float).reshape(3, 20, 1)
    result = crossed_intervals(np.concatenate([base, 3 * base, -base], axis=2), draws=1000)
    lo, hi = result['crossed_interval'][0]
    assert result['crossed_interval'][1] == pytest.approx([3*lo, 3*hi])
    assert result['crossed_interval'][2] == pytest.approx([-hi, -lo])


def test_invalid_or_unpaired_input_rejected():
    with pytest.raises(ValueError):
        crossed_intervals(np.ones((3, 512)))
    with pytest.raises(ValueError):
        crossed_intervals(np.array([[[np.nan]]]))
