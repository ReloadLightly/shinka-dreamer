"""Build fixed original-task comparators; never execute worlds or model calls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "artifacts/proposal/full-native-01/programs/generation-002.py"
MEMORY = ROOT / "controls/v1/memory.py"
SOURCE_HASHES = {
    "selected": "6062d5e728cf9a1fd38a73fbdcea3fa9d9183a046ee0155cd2709fb1481043fe",
    "memory": "08e696c1d28a4219b471ab9e2afa949ca6f9a74456df431cf9bac4384ac9c21e",
}

MEMORY_ADAPTER = '''

# Original-task adapter appended to the immutable v1 memory source above.
# The ancestor already binds learning and predictive planning off.
_study_memory_updater = world_model_step
_study_memory_planner = planner
_study_memory_legal = _legal

def _legal(m, p, d):
    # Interaction opens only a cardinally adjacent closed door before movement.
    if (d[0] and d[1] and not m["door_open"]
            and m["map"].get(_add(p, d)) == 3):
        return False
    return _study_memory_legal(m, p, d)

def world_model_step(memory, local_obs, last_action):
    m = _study_memory_updater(memory, local_obs, last_action)
    m["believed_map"] = {p: cell for p, cell in m["map"].items()
                         if cell in (0, 1, 2, 3, 4)}
    for p in m["enemies"]:
        m["believed_map"][p] = 5
    return m

def planner(memory, local_obs):
    # A key underfoot needs interaction without walking away from it.
    if memory["map"].get(memory["pos"]) == 2:
        return {"move": [0, 0], "interact": True}
    return _study_memory_planner(memory, local_obs)
'''

LOCAL_MAP_ADAPTER = '''

# Intervention: retain only the current 5x5 spatial observation in map fields.
# Localization, inventory, visits and collected-key history remain intact.
_study_full_updater = world_model_step

def world_model_step(memory, local_obs, last_action):
    m = _study_full_updater(memory, local_obs, last_action)
    ax, ay = m["agent_pos"]
    visible = {(ax + dx, ay + dy) for dx in range(-2, 3)
               for dy in range(-2, 3)}
    for field in ("believed_map", "terrain_map", "seen_at", "enemy_sightings"):
        m[field] = {p: value for p, value in m[field].items() if p in visible}
    for field in ("outside", "unstable"):
        m[field].intersection_update(visible)
    return m
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _read_frozen(path, name):
    data = path.read_bytes()
    if digest(data) != SOURCE_HASHES[name]:
        raise ValueError(f"Frozen {name} source hash changed: {path}")
    return data.decode("utf-8")


def omit_future_enemy_risk(source):
    """Remove precisely the selected program's marginal next-tick risk block."""
    tree = ast.parse(source)
    planner = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "planner")
    loops = [node for node in planner.body if isinstance(node, ast.For)
             and ast.unparse(node.target) == "enemy"
             and ast.unparse(node.iter) == "enemies"]
    if len(loops) != 1:
        raise ValueError("Expected one source-specific enemy-risk loop")
    index = planner.body.index(loops[0])
    previous, following = planner.body[index - 1], planner.body[index + 1]
    if ast.unparse(previous) != "survival = {}" or ast.unparse(following) != "immediate = {}":
        raise ValueError("Enemy-risk source boundaries changed")
    lines = source.splitlines(keepends=True)
    comment = "    # Marginal next-cell probabilities, without inventing enemy identities.\n"
    if lines[previous.lineno - 2] != comment:
        raise ValueError("Enemy-risk source annotation changed")
    return ("".join(lines[:previous.lineno - 2])
            + "    # Intervention: remove next-tick risk; current occupied cells stay excluded.\n"
            + "    survival = {}\n\n"
            + "".join(lines[following.lineno - 1:]))


def candidate_sources():
    selected = _read_frozen(SELECTED, "selected")
    memory = _read_frozen(MEMORY, "memory")
    return {
        "memory": memory + MEMORY_ADAPTER,
        "local_map_only": selected + LOCAL_MAP_ADAPTER,
        "no_future_risk": omit_future_enemy_risk(selected),
    }


def build_controls(out):
    """Write self-contained programs and provenance; refuse changed overwrites."""
    out = Path(out)
    sources = candidate_sources()
    details = {
        "memory": {
            "ancestor": str(MEMORY.relative_to(ROOT)),
            "ancestor_sha256": SOURCE_HASHES["memory"],
            "derivation": "Exact immutable source prefix plus observation-map export and two task-interface repairs.",
            "repairs": ["Interact in place for a key underfoot.",
                        "Disallow diagonal entry into a closed door; opening requires cardinal adjacency."],
            "mechanism": "Remembered terrain, Dijkstra pathfinding and fixed local enemy avoidance; predictive update/planning remain disabled by the ancestor.",
        },
        "local_map_only": {
            "ancestor": str(SELECTED.relative_to(ROOT)),
            "ancestor_sha256": SOURCE_HASHES["selected"],
            "derivation": "Exact selected source prefix plus updater wrapper; planner is unchanged.",
            "removed": ["Terrain, believed cells, observation ages, outside coordinates, unstable coordinates and enemy sightings outside the current 5x5 window."],
            "retained": ["Localization", "inventory", "visits", "collected-key history", "current observations", "original planner"],
            "interpretation": "Removes persistent spatial maps, not all memory. Current-map accuracy has a changed reported-cell denominator; compare coverage separately.",
        },
        "no_future_risk": {
            "ancestor": str(SELECTED.relative_to(ROOT)),
            "ancestor_sha256": SOURCE_HASHES["selected"],
            "derivation": "AST-checked source splice replaces only the marginal next-tick enemy-distribution block with survival = {}.",
            "removed": ["Next-tick enemy probability calculation and its effective minimum-risk action restriction."],
            "retained": ["Identical updater and memory", "current occupied-cell exclusion", "terrain legality", "objectives", "route search", "interaction"],
            "interpretation": "Tests use of a hardcoded one-step enemy model, not learned predictive parameters.",
        },
    }
    manifest = {
        "version": "original-task-study-controls-v1",
        "selected_source": str(SELECTED.relative_to(ROOT)),
        "selected_sha256": SOURCE_HASHES["selected"],
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "programs": {name: {"file": name + ".py", "sha256": digest(source.encode()),
                            **details[name]} for name, source in sources.items()},
        "episodes_executed_by_builder": 0,
        "model_calls": 0,
    }
    outputs = {out / (name + ".py"): source for name, source in sources.items()}
    outputs[out / "manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    for path, source in outputs.items():
        if path.exists() and path.read_text() != source:
            raise ValueError(f"Refusing to overwrite a different frozen control: {path}")
    out.mkdir(parents=True, exist_ok=True)
    for path, source in outputs.items():
        path.write_text(source)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "proposal/study/programs")
    args = parser.parse_args()
    print(json.dumps(build_controls(args.out), indent=2))


if __name__ == "__main__":
    main()
