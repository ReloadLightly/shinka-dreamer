"""Resume a frozen campaign, preserving interrupted proposals and native lineage.

This additive entry point leaves the hash-pinned driver/evaluator untouched. The
only runner extension restores already-generated jobs before native proposals
start. A process lock spans the probe, restoration, evolution and final flush.
"""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@contextmanager
def campaign_lock(results):
    path = Path(results) / "controller.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Campaign controller already holds {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def recovery_runner(base):
    from shinka.core.async_runner import AsyncRunningJob
    from dreamer.provenance import record, sha256

    class RecoveryRunner(base):
        async def _setup_async(self):
            await super()._setup_async()
            root = Path(self.results_dir)
            if self.meta_summarizer:
                restored = restore_recommendations(self.meta_summarizer.sync_summarizer, self.db, root)
                print("Restored native recommendation state:", json.dumps(restored), flush=True)
            recovery_file = root / "recovery-jobs.json"
            pending = json.loads(recovery_file.read_text()) if recovery_file.exists() else []
            for item in pending:
                generation = item["generation"]
                source = root / f"gen_{generation}/main.py"
                if sha256(source) != item["source_sha256"]:
                    raise ValueError(f"Saved generation {generation} changed since recovery inventory")
                existing = self.db.get_programs_by_generation(generation)
                if existing:
                    if any(p.code != source.read_text() for p in existing):
                        raise ValueError(f"Generation {generation} conflicts with recovery inventory")
                    continue  # An earlier recovery already persisted this job.
                if generation != self.next_generation_to_submit:
                    raise ValueError("Recover only the next native slot, without filling/reordering history")
                if generation >= self.evo_config.num_generations:
                    raise ValueError("Interrupted proposal lies outside the requested slot limit")
                parent = self.db.get(item["parent_id"])
                if parent is None or parent.code != (source.parent / "original.py").read_text():
                    raise ValueError("Interrupted proposal's parent does not match saved original.py")
                for key in ("archive_inspiration_ids", "top_k_inspiration_ids"):
                    if any(self.db.get(i) is None for i in item[key]):
                        raise ValueError("Saved inspiration is absent from the native database")
                for relative, expected in item["evidence_sha256"].items():
                    if sha256(root / relative) != expected:
                        raise ValueError(f"Recovery evidence changed: {relative}")
                # Keep the interrupted results/logs byte-identical. A new local
                # evaluation has its own directory; its source is the saved code.
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                output = source.parent / f"recovery-evaluation-{stamp}"
                started = time.time()
                job_id, worker, submitted, evaluation_started, running = await self._submit_evaluation_job_with_slot(
                    str(source), str(output), None)
                metadata = dict(item["patch_metadata"])
                metadata.update(recovered_interrupted_evaluation=True,
                                recovery_results_dir=str(output),
                                recovery_evidence_sha256=sha256(recovery_file))
                job = AsyncRunningJob(
                    job_id=job_id, exec_fname=str(source), results_dir=str(output),
                    start_time=started, proposal_started_at=started,
                    evaluation_submitted_at=submitted, evaluation_started_at=evaluation_started,
                    evaluation_worker_id=worker, running_eval_jobs_at_submit=running,
                    generation=generation, parent_id=item["parent_id"],
                    archive_insp_ids=item["archive_inspiration_ids"],
                    top_k_insp_ids=item["top_k_inspiration_ids"],
                    code_diff=(source.parent / "edit.diff").read_text(), meta_patch_data=metadata)
                self.total_api_cost += metadata.get("api_costs", 0)
                # check_job_status_async returns True while the process runs.
                while await self.scheduler.check_job_status_async(job):
                    if time.time() - started > self._get_evaluation_runtime_limit_seconds():
                        await self.scheduler.cancel_job_async(job_id)
                        raise RuntimeError("Recovered evaluation exceeded its existing runtime limit")
                    await asyncio.sleep(1)
                job.completion_detected_at = time.time()
                if not await self._process_single_job_safely(job):
                    raise RuntimeError("Recovery evaluation was not persisted; inspect its saved outputs")
                await self._restore_resume_progress()
                record(root / f"recovered-generation-{generation}.json", {
                    "generation": generation, "source_sha256": sha256(source),
                    "results_dir": str(output), "evidence_sha256": sha256(recovery_file),
                    "native_program_ids": [p.id for p in self.db.get_programs_by_generation(generation)]})
            # Never let upstream overwrite a different unpersisted saved proposal.
            for source in root.glob("gen_*/main.py"):
                generation = int(source.parent.name.removeprefix("gen_"))
                if generation >= self.next_generation_to_submit:
                    raise ValueError(f"Uninventoried saved proposal: {source}; preserve/recover it first")

    return RecoveryRunner


def restore_recommendations(summarizer, database, root):
    """Pinned upstream saves cumulative meta text but does not load it on resume."""
    paths = sorted((Path(root) / "meta").glob("meta_*.txt"),
                   key=lambda p: int(p.stem.removeprefix("meta_")))
    processed = set()
    history = []
    for path in paths:
        text = path.read_text()
        summary, rest = text.split("# GLOBAL INSIGHTS SCRATCHPAD\n", 1)
        scratchpad, recommendations = rest.split("# META RECOMMENDATIONS\n", 1)
        summary = summary.removeprefix("# INDIVIDUAL PROGRAM SUMMARIES\n").strip()
        history.append(recommendations.strip())
        if path == paths[-1]:
            processed = set(map(int, re.findall(r"\*\*Program Identifier:\*\* Generation (\d+)", summary)))
            if len(processed) != int(path.stem.removeprefix("meta_")):
                raise ValueError("Cannot reconcile native recommendation output with processed generations")
            summarizer.meta_summary = summary
            summarizer.meta_scratch_pad = scratchpad.strip()
            summarizer.meta_recommendations = recommendations.strip()
    summarizer.meta_recommendations_history = history
    summarizer.total_programs_processed = len(processed)
    summarizer.evaluated_since_last_meta = []
    pending = []
    for generation in range(database.last_iteration + 1):
        if generation not in processed:
            programs = database.get_programs_by_generation(generation)
            if programs:
                summarizer.add_evaluated_program(programs[0])
                pending.append(generation)
    return {"source": str(paths[-1]) if paths else None,
            "processed_generations": sorted(processed), "pending_generations": pending,
            "recommendation_history_count": len(history)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/campaign-v2")
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args()
    results = Path(args.results).resolve()
    if not (results / "campaign-manifest.json").exists():
        parser.error("Recovery requires an existing frozen campaign")
    with campaign_lock(results):
        from dreamer.provenance import record, sha256
        import dreamer.native
        import evolve
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record(results / f"recovery-execution-{stamp}.json", {
            "generation_stop": args.generations, "includes_seed": True,
            "campaign_sha256": sha256(results / "campaign-manifest.json"),
            "recovery_driver_sha256": sha256(__file__),
            "billing": "subscription", "assessment": "untouched"})
        dreamer.native.CheckpointRunner = recovery_runner(dreamer.native.CheckpointRunner)
        sys.argv = [str(ROOT / "scripts/evolve.py"), "--results", str(results),
                    "--generations", str(args.generations)]
        evolve.main()


if __name__ == "__main__":
    main()
