"""Native scheduler entrypoint; never import candidate code into evaluator."""
import argparse
import json
from pathlib import Path

from dreamer.evaluation import aggregate, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    from shinka.core import run_shinka_eval

    def aggregate_metrics(results):
        # Local raw checkpoint is evaluator-owned, never accepted from candidate.
        Path(args.results_dir, "episodes.json").write_text(json.dumps(results))
        return aggregate(results)

    run_shinka_eval(
        program_path=str(ROOT / "dreamer/bridge.py"), results_dir=args.results_dir,
        experiment_fn_name="run_evaluation", num_runs=args.episodes, run_workers=1,
        get_experiment_kwargs=lambda index: {"candidate_path": str(Path(args.program_path).resolve()), "seed": 10000+index},
        validate_fn=lambda result: (result["error"] is None, result["error"]),
        aggregate_metrics_fn=aggregate_metrics, verbose=False)


if __name__ == "__main__":
    main()
