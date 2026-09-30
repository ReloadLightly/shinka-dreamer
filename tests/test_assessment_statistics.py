"""Sparse-pair and unequal-denominator cases that can invalidate the conclusions."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assessment_analysis import binary_pair, holm, paired_exact_interval, pooled_pair


def test_zero_discordances_do_not_imply_zero_uncertainty():
    lo, hi = paired_exact_interval(0, 0, 1024)
    assert lo < 0 < hi
    assert hi == pytest.approx(-lo)


def test_exact_sparse_mcnemar_and_direction():
    left = [{"reason": "escaped"} for _ in range(20)]
    right = [{"reason": "caught" if i < 6 else "escaped"} for i in range(20)]
    result = binary_pair(left, right)
    assert (result["left_only"], result["right_only"], result["both"]) == (6, 0, 14)
    assert result["difference"] == .3
    assert result["mcnemar_exact_p"] == .03125
    reverse = binary_pair(right, left)
    assert reverse["difference"] == -.3
    assert reverse["ci95"] == pytest.approx([-result["ci95"][1], -result["ci95"][0]])
    assert holm([.01, .03, .04]) == pytest.approx([.03, .06, .06])


def test_forecast_bootstrap_recomputes_pooled_ratios():
    left = [{"stats": {"brier_near": [1., 1]}}, {"stats": {"brier_near": [2., 4]}}]
    right = [{"stats": {"brier_near": [0., 1]}}, {"stats": {"brier_near": [1., 4]}}]
    weights = np.array([[2, 0], [0, 2], [1, 1]], dtype=float)
    result = pooled_pair(left, right, "brier_near", weights)
    assert result["difference"] == pytest.approx(.4)
    assert result["ci95"] == pytest.approx(np.quantile([1., .25, .4], [.025, .975]))
