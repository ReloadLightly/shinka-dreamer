"""Native scheduler entrypoint; never import candidate code into evaluator."""
import argparse
import json
from pathlib import Path

from dreamer.evaluation_v3 import aggregate, ROOT, evaluation_identity
from dreamer.world_v3 import REGIMES
from dreamer.provenance import pool, record, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed_file", required=True)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--campaign_manifest")
    args = parser.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    seed_file = Path(args.seed_file)
    seeds, pool_info = pool(seed_file, args.episodes)
    if len(set(args.regimes)) != len(args.regimes):
        raise ValueError("Duplicate regimes")
    cases = [(seed, regime) for seed in seeds for regime in args.regimes]
    identity = evaluation_identity()
    if args.campaign_manifest:
        campaign = json.loads(Path(args.campaign_manifest).read_text())
        if campaign["evaluation"] != identity or campaign["development_pool"] != pool_info or args.regimes != list(REGIMES):
            raise ValueError("Campaign evaluator or episode pool changed")
    record(Path(args.results_dir, "evaluation-manifest.json"),
           {"evaluation": identity, "episode_pool": pool_info, "program_sha256": sha256(args.program_path), "regimes": args.regimes, "condition_episodes": len(cases)})
    from shinka.core import run_shinka_eval

    def aggregate_metrics(results):
        # Local raw checkpoint is evaluator-owned, never accepted from candidate.
        Path(args.results_dir, "episodes.json").write_text(json.dumps(results))
        return aggregate(results)

    run_shinka_eval(
        program_path=str(ROOT / "dreamer/bridge_v3.py"), results_dir=args.results_dir,
        experiment_fn_name="run_evaluation", num_runs=len(cases), run_workers=1,
        get_experiment_kwargs=lambda index: {"candidate_path": str(Path(args.program_path).resolve()), "seed": cases[index][0], "regime": cases[index][1]},
        validate_fn=lambda result: (result["error"] is None, result["error"]),
        aggregate_metrics_fn=aggregate_metrics, verbose=False)


if __name__ == "__main__":
    main()
