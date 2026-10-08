"""Add passive diagnostics to the source-reviewed v3 generation-4 selection.

The immutable selected.py remains untouched. Original functions and control
expressions are preserved; one observer call follows the computed action, and
an export wrapper adds raw predictive state plus the observer's record.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "aafa35fd866c353c83d0b661d79f2060d678fd2060ff50a1b10ea2f0e3ef5d3c"
HOOK = "    _diagnostic_capture(m, local_obs, risk, hazards, target, kind, risk_weight, future_weights, choices, memo, move)\n"
SUPPORT = '''

# Source-specific passive diagnostics; not a mutation or a policy revision.
def _diagnostic_capture(m, obs, risk, hazards, target, kind, weight,
                        future_weights, choices, memo, move):
    def finite(value):
        return float(value) if math.isfinite(value) else None
    def local(field):
        return [[x, y, float(p)] for (x, y), p in sorted(field.items())
                if max(abs(x-m["pos"][0]), abs(y-m["pos"][1])) <= 3]
    actions = []
    for score, hazard, potential, direction in choices:
        q = _add(m["pos"], direction)
        continuation = (0.0 if q == target and target != m["pos"]
                        else memo.get((q, 2), float("inf")))
        actions.append({"move": list(direction), "destination": list(q),
                        "score": finite(score), "hazard": float(hazard),
                        "potential": finite(potential),
                        "continuation": finite(continuation)})
    m["_diagnostic_decision"] = {
        "adapter": "gen4-planner-observer-v1", "position": list(m["pos"]),
        "target": list(target), "target_kind": kind, "move": list(move),
        "risk_weight": float(weight), "future_weights": list(future_weights),
        "law": list(m["law"] if obs.get("predictive_planning", True) else UNIFORM),
        "uniform_law_intervention": not obs.get("predictive_planning", True),
        "occupancy_horizon1_local": local(risk),
        "collision_hazards_local": [local(layer) for layer in hazards[1:]],
        "choices": actions,
    }

_diagnostic_original_export_model = export_model
def export_model(memory, local_obs):
    model = _diagnostic_original_export_model(memory, local_obs)
    model["learning"]["raw_predictive_state"] = {
        "slow": list(memory["slow"]), "fast": list(memory["fast"]),
        "log_weights": list(memory["log_weights"]),
    }
    model["diagnostic_planning"] = memory.get("_diagnostic_decision")
    return model
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(source):
    if digest(source.encode()) != ORIGINAL_SHA256:
        raise ValueError("Instrumentation is bound to the independently reviewed generation-4 source")
    original = ast.parse(source)
    planner = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "planner")
    final = planner.body[-1]
    if not isinstance(final, ast.Return):
        raise ValueError("Reviewed planner return changed")
    lines = source.splitlines(keepends=True)
    lines.insert(final.lineno-1, HOOK)
    instrumented = "".join(lines) + SUPPORT
    check = ast.parse(instrumented)
    changed_planner = next(n for n in check.body if isinstance(n, ast.FunctionDef) and n.name == "planner")
    hook = changed_planner.body.pop(-2)
    if not (isinstance(hook, ast.Expr) and isinstance(hook.value, ast.Call)
            and isinstance(hook.value.func, ast.Name) and hook.value.func.id == "_diagnostic_capture"):
        raise AssertionError("Unexpected instrumentation statement")
    if ast.dump(ast.Module(body=check.body[:len(original.body)], type_ignores=[]), include_attributes=False) != ast.dump(original, include_attributes=False):
        raise AssertionError("Original algorithm AST changed beyond the one passive observer call")
    return instrumented


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/campaign-v3/selection/selected.py")
    parser.add_argument("--out", default="artifacts/campaign-v3/selection/selected-instrumented.py")
    parser.add_argument("--manifest", default="artifacts/campaign-v3/selection/instrumentation.json")
    args = parser.parse_args()
    original = Path(args.source).read_text()
    output = build(original)
    path = Path(args.out)
    if path.exists() and path.read_text() != output:
        raise FileExistsError("Preserve previously instrumented source")
    path.write_text(output)
    manifest = {
        "original_source": args.source, "original_source_sha256": ORIGINAL_SHA256,
        "instrumented_source": args.out, "instrumented_source_sha256": digest(output.encode()),
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "parameter_key": "model.learning.raw_predictive_state",
        "original_algorithm_ast_equal_after_removing_observer_call": True,
        "original_export_fields_unchanged": True,
        "observer_writes": ["memory._diagnostic_decision", "export.learning.raw_predictive_state", "export.diagnostic_planning"],
        "observer_call_position": "After move selection and immediately before returning the action",
        "observer_reads": "Already computed planner locals and adaptive slow/fast/log_weights; no new action choice, prediction rollout, random draw or hidden environment input.",
        "nonfinite_diagnostic_costs": "Represented by null; original control calculations unchanged.",
        "passive_replay": "Without a planner call, diagnostic_planning is null; raw predictive state and original occupancy exports remain available.",
        "runtime_transparency": "Requires separately recorded paired development verification.",
        "new_environment_episodes": 0, "new_model_calls": 0,
    }
    target = Path(args.manifest)
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise FileExistsError("Preserve the previous instrumentation manifest")
    target.write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"source": args.out, "sha256": manifest["instrumented_source_sha256"], "parameter_key": manifest["parameter_key"]}))


if __name__ == "__main__":
    main()
