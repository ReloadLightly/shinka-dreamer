"""Trusted upstream callback for the separately versioned v3 evaluator."""
from dreamer.evaluation_v3 import run_episode


def run_evaluation(candidate_path, seed, regime):
    return run_episode(candidate_path, seed, regime=regime)
