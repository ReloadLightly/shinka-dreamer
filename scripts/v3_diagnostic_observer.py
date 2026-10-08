"""Passive rankings from the frozen directional comparator's actual planner.

The worker verifies source bytes and sets __diagnostic_source_sha256__ before
calling. This helper performs no file access, model update, or world transition.
It observes the planner's existing candidate tuple; it does not reimplement it.
"""
import math
import sys


SOURCE_SHA256 = "a8dec56f17a61a05981d619b075475068ab8469de871fd44de76ac276c2eca78"
PLANNER_FIRST_LINE = 370
CANDIDATE_OBSERVATION_LINE = 421  # Immediately after candidate assignment.


def _finite(value):
    return value if value is not None and math.isfinite(value) else None


def ranked_plan(namespace, memory, obs):
    """Return (unchanged action, diagnostics), retaining native memory writes.

    Coordinates are relative to the agent's starting point. Scores are native
    planner costs, not calibrated probabilities of death or path survival.
    No law probabilities, enemy identities, hidden map, or seed are exported.
    """
    if namespace.get("__diagnostic_source_sha256__") != SOURCE_SHA256:
        raise ValueError("Diagnostic observer requires the verified comparator source")
    planner = namespace["planner"]
    code = planner.__code__
    if code.co_name != "planner" or code.co_firstlineno != PLANNER_FIRST_LINE:
        raise ValueError("Comparator planner location differs from reviewed source")
    observed = []

    def trace(frame, event, arg):
        if frame.f_code is not code:
            return None
        if event == "line" and frame.f_lineno == CANDIDATE_OBSERVATION_LINE:
            local = frame.f_locals
            candidate = tuple(local["candidate"])
            observed.append((candidate, {
                "move": list(candidate[3]),
                "score": _finite(candidate[0]),
                "immediate": _finite(candidate[1]),
                "remaining": _finite(candidate[2]),
                "continuation": _finite(local["continuation"]),
                "target": list(local["target"]),
            }))
        return trace

    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        action = planner(memory, obs)
    finally:
        sys.settrace(previous_trace)
    choices = []
    for rank, (_, choice) in enumerate(sorted(observed, key=lambda item: item[0]), 1):
        choice.update(rank=rank, chosen=choice["move"] == action["move"])
        choices.append(choice)
    diagnostics = {
        "source_sha256": SOURCE_SHA256,
        "choices": choices,
        "chosen_move": list(action["move"]),
        "chosen_action": dict(action),
        "target": list(memory["target"]) if memory.get("target") is not None else None,
        "fallback_wait": not choices,
        "ranking_rule": "Ascending native (score, immediate, remaining, move) tuple",
        "predictive_planning_enabled": bool(obs.get("predictive_planning", True)),
        "score_scope": ("Native heuristic planner cost; immediate is next-tick occupancy risk"
                        if obs.get("predictive_planning", True) else
                        "Native heuristic planner cost; immediate is fixed proximity risk"),
        "coordinate_scope": "Relative to the agent's starting point",
    }
    return action, diagnostics
