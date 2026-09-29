"""Run paired development/validation/locked final assessment with checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dreamer.evaluation import run_episode, aggregate, ROOT, PROTOCOL, OBJECTIVE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "validation", "assessment"], default="development")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--program", default="initial.py")
    parser.add_argument("--variants", nargs="+", default=["reactive", "memory", "predictive", "frozen", "no_planning"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path(args.program).read_bytes()).hexdigest()
    if args.split == "assessment":
        # Generated only after agent selection. Never stored in mutation context.
        seed_file = ROOT / "results/private/assessment-seeds.json"
        seed_file.parent.mkdir(parents=True, exist_ok=True)
        if not seed_file.exists():
            seed_file.write_text(json.dumps([secrets.randbits(63) for _ in range(args.episodes)]))
        seeds = json.loads(seed_file.read_text())[:args.episodes]
        if len(seeds) != args.episodes:
            raise ValueError("Frozen assessment pool smaller than requested")
    else:
        start = 10000 if args.split == "development" else 20000
        seeds = list(range(start, start + args.episodes))
    manifest = {"protocol": PROTOCOL, "objective": OBJECTIVE, "split": args.split,
                "episodes": args.episodes, "variants": args.variants, "program_sha256": source_hash}
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to mix experiments in existing output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for variant in args.variants:
        path = out / f"{variant}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        for i in range(len(rows), len(seeds)):
            program = ROOT / "agents/reactive.py" if variant == "reactive" else args.program
            row = run_episode(program, seeds[i], variant)
            with path.open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
            rows.append(row)
            if row["error"]:
                raise RuntimeError(row["error"])
            if (i+1) % 16 == 0:
                print(variant, i+1, aggregate(rows)["public"], flush=True)
        (out / f"{variant}-summary.json").write_text(json.dumps(aggregate(rows), indent=2))


if __name__ == "__main__":
    main()
