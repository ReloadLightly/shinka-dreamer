"""Run paired development/validation/locked final assessment with checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dreamer.evaluation import run_episode, aggregate, ROOT, PROTOCOL, OBJECTIVE
from dreamer.provenance import assessment_pool, control_path, evaluation_identity, pool, record, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "validation", "assessment"], default="development")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--program", default="initial.py")
    parser.add_argument("--campaign", default="results/campaign-v2")
    parser.add_argument("--assessment-seeds")
    parser.add_argument("--reserve-assessment", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=["reactive", "memory", "original_predictive", "predictive", "frozen", "no_planning", "frozen_no_planning"],
                        default=["memory", "original_predictive", "predictive", "frozen", "no_planning", "frozen_no_planning"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path(args.program).read_bytes()).hexdigest()
    if args.split == "assessment":
        if not args.assessment_seeds:
            parser.error("assessment requires an explicit campaign-specific --assessment-seeds file")
        seeds, pool_info = assessment_pool(args.campaign, args.assessment_seeds,
                                         args.program, args.episodes, args.reserve_assessment)
    else:
        start = 10000 if args.split == "development" else 20000
        seeds = list(range(start, start + args.episodes))
        seed_file = out / f"{args.split}-seeds.json"
        record(seed_file, seeds)
        seeds, pool_info = pool(seed_file, args.episodes)
    programs = {variant: str(control_path("memory" if variant == "memory" else "predictive"))
                if variant in ("memory", "original_predictive") else
                str(ROOT / "agents/reactive.py") if variant == "reactive" else args.program
                for variant in args.variants}
    manifest = {"protocol": PROTOCOL, "objective": OBJECTIVE, "split": args.split,
                "episodes": args.episodes, "variants": args.variants, "program_sha256": source_hash,
                "programs": {v: sha256(p) for v, p in programs.items()},
                "evaluation": evaluation_identity(), "episode_pool": pool_info,
                "campaign": str(Path(args.campaign).resolve())}
    manifest_path = out / "manifest.json"
    record(manifest_path, manifest)
    for variant in args.variants:
        path = out / f"{variant}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        for i in range(len(rows), len(seeds)):
            program = programs[variant]
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
