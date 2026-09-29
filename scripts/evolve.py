"""Native ShinkaEvolve; one worker, four islands, subscription-only clients."""
import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/campaign-v1")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--model", default="headless/codex@gpt-6-astra?effort=high")
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args()
    if not args.model.startswith("headless/codex@"):
        raise ValueError("Only subscription-backed Headless Codex is authorized")
    os.environ["HEADLESS_BILLING"] = "subscription"
    os.environ["SHINKA_HEADLESS_COMMAND"] = str(ROOT / "scripts/subscription_headless.sh")
    os.environ["SHINKA_HEADLESS_TIMEOUT"] = "600"
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(name, None)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    from dreamer.native import CheckpointRunner
    from shinka.llm.providers.headless import parse_headless_model
    route = parse_headless_model(args.model)

    results = Path(args.results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    if args.generations > 1:
        # A real, subscription-only probe prevents native resampling from retrying
        # a host startup failure indefinitely. Nothing modifies login or policy.
        probe = subprocess.run([str(ROOT / "scripts/subscription_headless.sh"), "codex",
                "--allow", "read-only", "--model", route.agent_model,
                "--reasoning-effort", route.effort or "high",
                "--timeout", "60", "--usage", "--prompt", "Reply only READY. Do not call tools."],
                capture_output=True, text=True, timeout=75)
        (results / "subscription-probe.log").write_text(probe.stdout+probe.stderr)
        if probe.returncode != 0:
            raise RuntimeError("Subscription Codex unavailable. See " + str(results / "subscription-probe.log"))
    evo = EvolutionConfig(
        task_sys_msg=(ROOT / "docs/mutation-prompt.md").read_text(),
        num_generations=args.generations, init_program_path=str(ROOT / "initial.py"),
        results_dir=str(results), llm_models=[args.model], llm_dynamic_selection=None,
        llm_kwargs={"temperatures": [1.0], "max_tokens": 24000},
        meta_rec_interval=10, meta_llm_models=[args.model], meta_llm_kwargs={},
        # Upstream wraps all proposal generation in this loop even when novelty
        # is disabled. Zero would suppress mutations, not just novelty checks.
        novelty_llm_models=None, embedding_model=None, max_novelty_attempts=1,
        use_text_feedback=True, evolve_prompts=False, prompt_llm_models=None,
        patch_types=["diff", "full"], patch_type_probs=[.6, .4],
        max_patch_resamples=2, max_patch_attempts=2,
        enable_controlled_oversubscription=False)
    db = DatabaseConfig(num_islands=4, migration_interval=10, migration_rate=.1,
                        archive_size=40, num_archive_inspirations=1, num_top_k_inspirations=1)
    job = LocalJobConfig(eval_program_path=str(ROOT / "evaluate.py"),
                         python_executable=str(ROOT / ".venv/bin/python"),
                         extra_cmd_args={"episodes": args.episodes},
                         time="00:15:00", numeric_threads_per_job=1)
    resolved = {"evolution": asdict(evo), "database": asdict(db), "job": asdict(job),
                "max_evaluation_jobs": 1, "max_proposal_jobs": 1, "max_db_workers": 1,
                "billing": "subscription", "upstream": "9912af12d423504b8d580f4179fd15f5f88b8c50",
                "headless": "93cd9b06b85f848af1308c41e018991b33907c5e"}
    (results / "dreamer-resolved.json").write_text(json.dumps(resolved, indent=2))
    async def run():
        # On this restricted host socketpair.send is denied, so asyncio's
        # cross-thread wakeup silently fails. A timer lets the native event loop
        # drain completed futures without changing any network permissions.
        loop = asyncio.get_running_loop()
        def pulse():
            loop.call_later(.05, pulse)
        pulse()
        runner = CheckpointRunner(evo_config=evo, db_config=db, job_config=job,
                                   max_evaluation_jobs=1, max_proposal_jobs=1, max_db_workers=1)
        await runner.run_async()
    asyncio.run(run())


if __name__ == "__main__":
    main()
