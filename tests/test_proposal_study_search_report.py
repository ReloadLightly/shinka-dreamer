"""Saved-state fixtures: failed searches, acceptance prefixes and safe event export."""
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from proposal.study_search_report import export


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class StudySearchReportTests(unittest.TestCase):
    def fixture(self, root, arm):
        campaign = root / "runtime"
        save(campaign / "resolved-config.json", {
            "run_id": "fixture", "arm": arm, "search_cases": [987654321987654321],
            "api_key": "do-not-export-secret", "evolution": {"task_sys_msg": "private-prompt-body",
                "llm_dynamic_selection": "ucb" if arm == "full" else None,
                "embedding_model": "local/fixture" if arm == "full" else None,
                "patch_types": ["full"], "patch_type_probs": [1.], "evolve_prompts": arm == "full"},
            "database": {"num_islands": 4 if arm == "full" else 1,
                         "num_archive_inspirations": 1 if arm == "full" else 0}})
        save(campaign / "execution-resources.json", {
            "status": "interrupted_or_failed", "returncode": 1, "stop_reason": "hard_execution_deadline"})
        return campaign

    def test_empty_failed_rewrite_keeps_disabled_features_and_unknown_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = self.fixture(root, "rewrite")
            save(campaign / "execution-resources.json", {"status": "worker_returned", "returncode": 0, "stop_reason": None})
            blocked = "Codex context audit failed: missing_or_invalid_audit:FileNotFoundError"
            save(campaign / "call-budget-ledger.json", {"blocked_reason": blocked})
            save(campaign / "model-calls/call1.json", {"id": "call1", "role": "mutation",
                "status": "completed", "returncode": 124, "requested_model": "fixture-model",
                "elapsed_seconds": 12., "error": "TimeoutError: private-error-body",
                "mutation_context_validated": False,
                "context_audit_failure": "missing_or_invalid_audit:FileNotFoundError"})
            summary = export(campaign, root / "public")
            self.assertEqual(summary["valid_descendants"], 0)
            self.assertEqual(summary["saved_episodes"], 0)
            self.assertEqual(summary["all_role_calls"]["completed_without_reported_usage"], 1)
            self.assertFalse(summary["all_role_calls"]["token_total_exact_for_all_admitted_calls"])
            calls = json.loads((root / "public/model-calls.json").read_text())
            self.assertFalse(calls[0]["mutation_context_validated"])
            self.assertEqual(calls[0]["context_audit_failure"], "missing_or_invalid_audit:FileNotFoundError")
            self.assertEqual(calls[0]["failure_class"], "context-audit")
            self.assertEqual(summary["stop_reason"], blocked)
            self.assertEqual(summary["termination_cause"]["source"], "budget-ledger")
            ledger = json.loads((root / "public/budget-ledger.json").read_text())
            self.assertEqual(ledger["blocked_reason"], blocked)
            self.assertEqual(summary["embedding_audit"]["status"], "disabled_by_rewrite_treatment")
            self.assertEqual(summary["embedding_audit"]["native_endpoint_calls"], 0)
            for feature in ("mutation_bandit", "embeddings", "novelty", "inspirations", "migration", "meta_recommendations", "prompt_evolution"):
                self.assertEqual(summary["mechanisms"][feature]["status"], "disabled_by_rewrite_treatment")
            output = "".join(p.read_text() for p in (root / "public").glob("*.json"))
            for forbidden in ("do-not-export-secret", "private-prompt-body", "private-error-body", "987654321987654321"):
                self.assertNotIn(forbidden, output)

    def test_accepted_prefix_lineage_and_raw_tools_counted_without_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = self.fixture(root, "full")
            config = json.loads((campaign / "resolved-config.json").read_text())
            config["search_cases"] = list(range(5))
            save(campaign / "resolved-config.json", config)
            with sqlite3.connect(campaign / "programs.sqlite") as db:
                db.execute("CREATE TABLE programs (id TEXT, generation INTEGER, code TEXT, correct INTEGER, combined_score REAL, metadata TEXT)")
                db.executemany("INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?)", [
                    ("seed", 0, "def planner(): return 0\n", 1, .1, "{}"),
                    ("child", 1, "def planner(): return 1\n", 1, .8, '{"patch_type":"full"}')])
            candidate_hash = hashlib.sha256(b"def planner(): return 1\n").hexdigest()
            frame = {"world": {"grid": [[0] * 15 for _ in range(15)], "enemies": [],
                "agent": [1, 1], "origin": [1, 1], "step": 0, "keys": 0, "door_open": False,
                "reason": "running"}, "belief": [[0, 0, 0]], "action": {"move": [1, 0], "interact": True}}
            episode = {"case": 0, "reason": "escaped", "steps": 1, "keys": 2, "door_open": True,
                "task": .8, "model_accuracy": .8, "combined_score": .8, "final_coverage": .1,
                "map_correct": 10, "map_audited": 10, "error": None, "seconds": .01,
                "trace": [frame], "timeline": [{"step": 0, "correct": 10, "audited": 10, "coverage": .1}]}
            save(campaign / "gen_1/results/episodes.json", [episode])
            save(campaign / "gen_1/results/manifest.json", {"episodes": 1, "seed_start": 0,
                "candidate_sha256": candidate_hash, "source_sha256": {"evaluator.py": "frozen-evaluator-hash"}})
            for generation, prefix in ((0, 0), (1, 3)):
                save(campaign / f"gen_{generation}/acceptance-budget.json", {
                    "generation": generation, "all_role_calls_at_acceptance": prefix,
                    "provider_elapsed_seconds_at_acceptance": prefix * 5})
            save(campaign / "run1-native-state.json", {"bandit_state": {
                "arm_names": ["model-a", "model-b"], "n_submitted": [1, 1], "n_completed": [1, 0]}})
            save(campaign / "embedding-identity.json", {"model": "BAAI/bge-small-en-v1.5",
                "repository": "qdrant/bge-small-en-v1.5-onnx-q", "revision": "frozen-model-revision",
                "download_metadata": "private-download-metadata", "dim": 384,
                "packages": {"onnxruntime": "1.23.2", "private_package": "private-package-metadata"},
                "file_sha256": {"model.onnx": "a" * 64}})
            save(campaign / "embedding-fixture.json", {"identical_exact": True, "dimensions": 384,
                "cpu_seconds": 99., "wall_seconds": 100., "peak_rss_kib": 12})
            (campaign / "embedding-calls.jsonl").write_text(json.dumps({"status": "ok", "inputs": 1,
                "text_sha256": ["b" * 64], "tokens": 12, "chunks": 1, "cpu_seconds": 2.,
                "wall_seconds": 3., "peak_rss_kib": 34, "input": "private-embedding-text"}) + "\n" +
                json.dumps({"status": "error", "error": "private-embedding-error"}) + "\n")
            save(campaign / "model-calls/call1.json", {"id": "call1", "role": "mutation",
                "status": "completed", "returncode": 0, "requested_model": "model-a", "elapsed_seconds": 1.})
            audit = campaign / "codex-event-audit"
            audit.mkdir()
            (audit / "call1.events.jsonl").write_text(json.dumps({"type": "item.started", "item": {
                "type": "command_execution", "command": "private-command-and-context"}}) + "\n")
            save(audit / "call1.summary.json", {"unauthorized_tool_event": {
                "event_type": "item.started", "item_type": "command_execution"}, "delegate_returncode": 78})
            summary = export(campaign, root / "public")
            self.assertEqual(summary["valid_descendants"], 1)
            self.assertEqual([p["best_generation"] for p in summary["acceptance_prefixes"]], [0, 1])
            self.assertEqual(summary["raw_tool_audit"]["observed_tool_event_records"], 1)
            self.assertEqual(summary["raw_tool_audit"]["wrapper_rejections"], 1)
            self.assertEqual(summary["mechanisms"]["mutation_bandit"]["observed"]["native_arm_counts"]["model-b"]["completed"], 0)
            self.assertEqual(summary["embedding_audit"]["revision"], "frozen-model-revision")
            self.assertEqual(summary["embedding_audit"]["native_endpoint_calls"], 2)
            self.assertEqual(summary["embedding_audit"]["recorded_endpoint_cpu_seconds"], 2.)
            self.assertEqual(summary["embedding_audit"]["calls_without_resource_measurements"], 1)
            self.assertEqual(summary["embedding_audit"]["recorded_local_tokens"], 12)
            self.assertEqual(summary["all_role_calls"]["tokens"]["totalTokens"], 0)
            self.assertTrue((root / "public/programs/generation-001.py").exists())
            empirical = json.loads((root / "public/episodes.json").read_text())
            self.assertEqual(empirical[0]["candidate_sha256"], candidate_hash)
            self.assertEqual(empirical[0]["combined_score"], .8)
            self.assertNotIn("trace", empirical[0])
            replays = json.loads(gzip.decompress((root / "public/representative-replays.json.gz").read_bytes()))
            self.assertEqual(replays[0]["trace"], [frame])
            self.assertEqual(replays[0]["candidate_sha256"], candidate_hash)
            provenance = json.loads((root / "public/provenance.json").read_text())
            self.assertIn("gen_1/results/manifest.json", provenance["inputs"])
            self.assertIn("embedding-identity.json", provenance["inputs"])
            self.assertIn("embedding-calls.jsonl", provenance["inputs"])
            self.assertIn("embedding-audit.json", provenance["outputs"])
            self.assertEqual(provenance["outputs"]["episodes.json"]["sha256"],
                             hashlib.sha256((root / "public/episodes.json").read_bytes()).hexdigest())
            output = "".join(p.read_text() for p in (root / "public").glob("*.json"))
            self.assertNotIn("private-command-and-context", output)
            for forbidden in ("private-download-metadata", "private-package-metadata", "private-embedding-text", "private-embedding-error"):
                self.assertNotIn(forbidden, output)
            self.assertFalse(list((root / "public").rglob("*.sqlite")))


if __name__ == "__main__":
    unittest.main()
