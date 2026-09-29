"""Small immutable records for controls, evaluation versions and episode pools."""
import hashlib
import json
from pathlib import Path
import platform
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
V1_POOL_SHA256 = "8e05f239e7067822fb200bb2fa9ad27f4c034b3dc62c52cb18cf00488a2f06d6"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, data):
    """Create once, reject drift on resume. Never replace an existing record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Configuration changed: {path}; use a new campaign/output directory")
    else:
        with path.open("x") as file:
            file.write(json.dumps(data, indent=2) + "\n")
    return data


def control_path(name):
    manifest = json.loads((ROOT / "controls/v1/manifest.json").read_text())
    entry = manifest[name]
    path = ROOT / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Original {name} control changed")
    return path


def evaluation_identity():
    from .evaluation import PROTOCOL, OBJECTIVE
    paths = ["evaluate.py", "dreamer/evaluation.py", "dreamer/world.py",
             "dreamer/worker.py", "dreamer/isolation.py", "dreamer/bridge.py",
             "dreamer/provenance.py"]
    return {"protocol": PROTOCOL, "objective": OBJECTIVE,
            "objective_formula": ".6*(.65*escape+.10*keys+.10*door+.05*escape*(1-steps/200))+.4*(1-(near_Brier+audit_Brier)/2)",
            "source_sha256": {p: sha256(ROOT / p) for p in paths},
            "controls": {n: sha256(control_path(n)) for n in ("predictive", "memory")},
            "runtime": {"evaluator_python": platform.python_version(),
                        "candidate_executable": "/usr/bin/python3",
                        "candidate_executable_sha256": sha256("/usr/bin/python3"),
                        "candidate_args": ["-s", "-S"], "hash_seed": 0,
                        "agent_random_seed_before_exec": 712934,
                        "stdlib_paths": ["/usr/lib/python3.10", "/usr/lib/python3.10/lib-dynload"],
                        "limits": {"address_space_mib": 192, "cpu_seconds": 10,
                                   "reply_seconds": 3, "source_kib": 512, "reply_mib": 2,
                                   "file_descriptors": 64},
                        "isolation": "Landlock + seccomp, unchanged from v1"}}


def pool(path, expected_count=None):
    path = Path(path)
    seeds = json.loads(path.read_text())
    if (not isinstance(seeds, list) or not seeds or
            any(type(s) is not int or not 0 <= s < 2**63 for s in seeds) or
            len(set(seeds)) != len(seeds) or
            (expected_count is not None and len(seeds) != expected_count)):
        raise ValueError(f"Invalid episode pool: {path}")
    return seeds, {"path": str(path.resolve()), "sha256": sha256(path), "episodes": len(seeds)}


def assessment_pool(campaign, path, program, count, reserve=False):
    """Lock a native descendant selection BEFORE generating fresh final cases."""
    import secrets
    campaign, path, program = Path(campaign).resolve(), Path(path).resolve(), Path(program).resolve()
    manifest_path = campaign / "campaign-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["evaluation"] != evaluation_identity():
        raise ValueError("Evaluator changed since campaign; cannot mix assessment scores")
    if str(path) != manifest["assessment_pool_path"]:
        raise ValueError("Use this campaign's explicit assessment pool path")
    selected_hash = sha256(program)
    selection_path = campaign / "selection.json"
    if not selection_path.exists():
        if not reserve:
            raise ValueError("Select/inspect the descendant, then pass --reserve-assessment once")
        with sqlite3.connect(f"file:{campaign / 'programs.sqlite'}?mode=ro", uri=True) as db:
            candidates = db.execute("select id, generation, code from programs where generation > 0 and correct = 1").fetchall()
        matches = [(i, g) for i, g, code in candidates if hashlib.sha256(code.encode()).hexdigest() == selected_hash]
        if not matches or selected_hash == manifest["evaluation"]["controls"]["predictive"]:
            raise ValueError("Final selection must be a genuinely evaluated native descendant")
        record(selection_path, {"program_sha256": selected_hash, "native_id": matches[0][0],
                                "generation": matches[0][1], "campaign_sha256": sha256(manifest_path)})
    selection = json.loads(selection_path.read_text())
    if selection["program_sha256"] != selected_hash or selection["campaign_sha256"] != sha256(manifest_path):
        raise ValueError("Selection is frozen; do not tune on or reuse final cases")
    pool_record = campaign / "assessment-manifest.json"
    if not pool_record.exists():
        if not reserve or path.exists():
            raise ValueError("Fresh final pool must be newly reserved after selection")
        excluded = set(range(10000, 100000))
        old = ROOT / "results/private/assessment-seeds.json"
        if old.exists():
            excluded.update(json.loads(old.read_text()))
        for other in (ROOT / "results/private").glob("*-assessment-seeds.json"):
            excluded.update(json.loads(other.read_text()))
        seeds = []
        while len(seeds) < count:
            seed = secrets.randbits(63)
            if seed not in excluded:
                seeds.append(seed)
                excluded.add(seed)
        record(path, seeds)
        _, info = pool(path, count)
        record(pool_record, {"pool": info, "selection_sha256": sha256(selection_path)})
    seeds, info = pool(path, count)
    if info["sha256"] == V1_POOL_SHA256:
        raise ValueError("Published v1 pool cannot be reused")
    record(pool_record, {"pool": info, "selection_sha256": sha256(selection_path)})
    return seeds, info
