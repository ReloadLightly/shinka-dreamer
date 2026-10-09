"""Frozen-design resampling for paired runs sharing the same assessment cases.

No evaluator, model client, seed pool or candidate is imported. Repetition blocks
and case IDs are independent resampling axes; their indices are shared across
arms and metrics. With three repetitions, intervals remain exploratory.
"""
from __future__ import annotations

import numpy as np


def crossed_intervals(differences, *, draws=10000, seed=7109512, level=.95):
    """Summarize paired differences shaped (repetitions, cases, metrics).

    Call once with all metrics, after subtracting arms within each run/case.
    Equal-weight repetition means define the primary estimand. First generate
    common case draws, then independent repetition-block draws using one stream.
    Reusing a program source does not create additional independent observations.
    """
    data = np.asarray(differences, dtype=float)
    if data.ndim != 3 or min(data.shape) < 1 or not np.isfinite(data).all():
        raise ValueError('Expected finite repetition-by-case-by-metric differences')
    if type(draws) is not int or draws < 100 or not 0 < level < 1:
        raise ValueError('Require at least 100 draws and a confidence level in (0,1)')
    repetitions, cases, metrics = data.shape
    rng = np.random.default_rng(seed)
    case_indices = rng.integers(0, cases, size=(draws, cases))
    repetition_indices = rng.integers(0, repetitions, size=(draws, repetitions))
    sampled = np.empty((draws, repetitions, metrics))
    for repetition in range(repetitions):
        for metric in range(metrics):
            sampled[:, repetition, metric] = data[repetition, :, metric][case_indices].mean(axis=1)
    crossed = sampled[np.arange(draws)[:, None], repetition_indices].mean(axis=1)
    quantiles = [(1-level)/2, (1+level)/2]
    return {
        'repetitions': repetitions, 'cases': cases, 'draws': draws,
        'seed': seed, 'level': level,
        'resampling': 'Shared case indices followed by independent paired-repetition indices; same draws for all metrics.',
        'mean_difference': data.mean(axis=(0, 1)).tolist(),
        'crossed_interval': np.quantile(crossed, quantiles, axis=0).T.tolist(),
        'individual_mean_differences': data.mean(axis=1).tolist(),
        'individual_paired_case_intervals': np.moveaxis(np.quantile(sampled, quantiles, axis=0), 0, -1).tolist(),
        'limitation': 'Small repetition counts provide limited information about search repeatability; these percentile intervals are exploratory.',
    }
