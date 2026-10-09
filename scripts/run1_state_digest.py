"""Deterministic value digest for the reviewed selected program's built-in state.

Set/dictionary iteration order and pickle's object-memo representation do not
define these values. Container type and exact finite float values are retained.
This does not claim arbitrary executable objects are semantically equivalent.
"""
import hashlib
import json
import math


def canonical_state(value):
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite predictive state")
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [canonical_state(item) for item in value]]
    if isinstance(value, dict):
        entries = [[canonical_state(key), canonical_state(item)] for key, item in value.items()]
        return ["dict", sorted(entries, key=lambda entry: json.dumps(entry[0], separators=(",", ":"))) ]
    if isinstance(value, (set, frozenset)):
        entries = [canonical_state(item) for item in value]
        return [type(value).__name__, sorted(entries, key=lambda entry: json.dumps(entry, separators=(",", ":"))) ]
    raise TypeError(f"Unsupported reviewed-state type: {type(value).__name__}")


def state_digest(value):
    encoded = json.dumps(canonical_state(value), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()
