"""Trusted Shinka callback; upstream expands run kwargs by name."""
from dreamer.evaluation import run_episode


def run_evaluation(candidate_path, seed):
    return run_episode(candidate_path, seed)
