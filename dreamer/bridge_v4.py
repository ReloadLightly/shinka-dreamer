"""Trusted native callback; checkpoint every completed evaluator-owned episode."""
import hashlib
import json
from pathlib import Path
from dreamer.evaluation_v4 import run_episode


def run_evaluation(candidate_path, seed, regime, checkpoint_directory):
    row = run_episode(candidate_path, seed, regime=regime)
    folder = Path(checkpoint_directory)
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(f"{seed}:{regime}".encode()).hexdigest()
    path = folder / (name + ".json")
    temporary = folder / (name + ".tmp")
    temporary.write_text(json.dumps(row, allow_nan=False))
    temporary.replace(path)
    return row
