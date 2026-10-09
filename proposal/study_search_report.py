"""Export compact saved independent-search evidence, without worlds or models."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re

from scripts.run1_native_report import (
    clean, digest, json_field, jsonl, public_text, read, rows, totals,
)


ROLES = ("mutation", "repair", "novelty", "summary", "global_insight",
         "recommendation", "prompt_mutation", "readiness")
TOOL_ITEMS = {"command_execution", "web_search", "mcp_tool_call", "file_change",
    "image_view", "image_generation", "collab_tool_call", "dynamic_tool_call",
    "function_call", "tool_call", "custom_tool_call", "computer_use", "browser_use"}
TOKEN_FIELDS = ("inputTokens", "cacheReadTokens", "cacheWriteTokens", "outputTokens",
                "reasoningOutputTokens", "totalTokens")
EPISODE_FIELDS = ("case", "reason", "steps", "keys", "door_open", "task", "model_accuracy",
    "combined_score", "final_coverage", "map_correct", "map_audited", "seconds",
    "candidate_cpu_seconds", "evaluator_cpu_seconds")


def pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def write(path, value):
    public_text(path, json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def failure_class(text, returncode=None):
    """Classify locally; retain no arbitrary provider error payload."""
    text = str(text or "").lower()
    if any(word in text for word in ("context audit", "context_audit", "unauthorized_tool", "raw_event_audit")):
        return "context-audit"
    if any(word in text for word in ("unauthorized", "authorization", "authentication", "billing", "forbidden", "subscription route")):
        return "authorization"
    if any(word in text for word in ("capacity", "rate limit", "usage limit", "quota", "429", "overloaded")):
        return "capacity"
    if returncode == 124 or any(word in text for word in ("timeout", "timed out", "timeoutexpired")):
        return "timeout"
    return "other"


def bounded_stop_reason(value):
    """Publish controlled adapter labels, hashing unrecognized stop payloads."""
    if not value:
        return None
    text = str(value)
    exact = {
        "Unresolved prior request; reconcile before continuation",
        "Prior billing or effective model mismatch requires review",
        "Execution deadline/reserved request duration reached", "All-role call cap reached",
        "Provider-time cap/reserved request duration reached", "Two consecutive provider transport failures",
        "Observed billing or effective model mismatch", "Only configured Codex subscription routes are authorized",
        "Headless request lacks the required read-only sandbox route",
        "Novelty judge unavailable; no silent acceptance",
        "Local embedding unavailable; native novelty cannot be skipped",
        "combined_rss_ceiling", "hard_execution_deadline", "prospective development saturation rule",
    }
    patterns = (
        r"(?:Role forbidden for this treatment|Unauthorized role or role cap): [A-Za-z_]{1,32}",
        r"Codex context audit failed: (?:unauthorized_tool_event|raw_event_audit_mismatch|missing_or_invalid_audit:[A-Za-z_]{1,64})",
        r"(?:supervisor_signal:|signal:)SIG[A-Z]{1,16}",
    )
    if text in exact or any(re.fullmatch(pattern, text) for pattern in patterns):
        return text[:240]
    return "unrecognized_stop_reason (" + failure_class(text) + ")"


def compact_config(config):
    result = pick(config, ("schema", "treatment", "arm", "run_id", "evaluator", "limits",
        "wall_minutes", "checkpoint_reserve_seconds", "native_sampling_seed", "bandit_seed",
        "candidate_rng_seed", "rewrite_model_offset", "allowed_roles", "rng_limitation",
        "shinka_revision", "headless_revision", "headless_capacity_retries", "provider_retries",
        "child_timeout_seconds", "billing", "concurrency", "episode_cap", "transition_cap",
        "candidate_cpu_seconds_cap", "combined_rss_bytes_cap", "source_sha256",
        "missing_usage_policy", "unsupported_not_forwarded", "resume_policy"))
    evo = config.get("evolution", {})
    result["evolution"] = pick(evo, ("num_generations", "language", "job_type", "llm_models",
        "llm_dynamic_selection", "llm_dynamic_selection_kwargs", "meta_rec_interval",
        "meta_llm_models", "meta_max_recommendations", "embedding_model", "code_embed_sim_threshold",
        "max_novelty_attempts", "novelty_llm_models", "use_text_feedback", "evolve_prompts",
        "prompt_llm_models", "prompt_evolution_interval", "prompt_archive_size",
        "prompt_percentile_recompute_interval", "prompt_patch_types", "prompt_patch_type_probs",
        "patch_types", "patch_type_probs", "max_patch_resamples", "max_patch_attempts",
        "enable_controlled_oversubscription", "sample_single_meta_rec", "inspiration_sort_order"))
    result["task_prompt_sha256"] = digest(evo["task_sys_msg"]) if "task_sys_msg" in evo else None
    result["database"] = pick(config.get("database", {}), ("num_islands", "migration_interval",
        "migration_rate", "archive_size", "num_archive_inspirations", "num_top_k_inspirations",
        "parent_selection_strategy", "parent_selection_lambda", "island_selection_strategy",
        "island_elitism", "enforce_island_separation", "archive_selection_strategy"))
    result["job"] = pick(config.get("job", {}), ("time", "numeric_threads_per_job"))
    result["search_case_count"] = len(config.get("search_cases", []))
    result["search_cases_sha256"] = digest(json.dumps(config.get("search_cases", [])))
    result["mutation_context"] = pick(config.get("mutation_context", {}),
        ("project_doc_max_bytes", "global_config_sha256", "tool_event_policy", "limitation"))
    return result


def compact_call(call):
    value = pick(call, ("id", "role", "started_utc", "ended_utc", "requested_model",
        "requested_effort", "billing_required", "status", "prompt_sha256", "elapsed_seconds",
        "returncode", "billing_violation", "model_mismatch", "cli_lock_wait_seconds", "usage_missing",
        "mutation_context_validated"))
    problem = call.get("context_audit_failure")
    value["context_audit_failure"] = (problem if problem is None or problem in
        ("unauthorized_tool_event", "raw_event_audit_mismatch") or
        re.fullmatch(r"missing_or_invalid_audit:[A-Za-z_][A-Za-z0-9_]*", str(problem))
        else "unrecognized_context_failure")
    value["context_audit_failure_sha256"] = digest(str(problem)) if problem else None
    value.setdefault("status", "unknown")
    value.setdefault("role", "unknown")
    usage = call.get("usage") or {}
    value["usage"] = pick(usage, (*TOKEN_FIELDS, "usageStatus", "model"))
    value["usage"]["billing"] = {"attempts": [pick(a, ("route",))
        for a in (usage.get("billing") or {}).get("attempts", [])]}
    value["usage"]["cost"] = pick(usage.get("cost") or {}, ("total",))
    value["has_error"] = bool(call.get("error"))
    value["error_sha256"] = digest(str(call["error"])) if call.get("error") else None
    failed = bool(call.get("error") or call.get("context_audit_failure") or call.get("returncode") not in (None, 0))
    value["failure_class"] = ("context-audit" if call.get("context_audit_failure") else
        failure_class(call.get("error"), call.get("returncode"))) if failed else None
    return value


def event_label(value):
    value = str(value or "unknown")
    return value if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) else "unrecognized-type"


def raw_tool_summary(campaign, calls):
    """Count local raw event types; never serialize their payloads or tool arguments."""
    directory = campaign / "codex-event-audit"
    identities = {p.name.removesuffix(".events.jsonl") for p in directory.glob("*.events.jsonl")}
    identities |= {p.name.removesuffix(".summary.json") for p in directory.glob("*.summary.json")}
    audits = []
    for call_id in sorted(identities):
        events, items, tool_events, partial = Counter(), Counter(), 0, 0
        path = directory / (call_id + ".events.jsonl")
        if path.exists():
            for line in path.read_bytes().splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    partial += 1
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    events["invalid-json"] += 1
                    continue
                if not isinstance(event, dict):
                    events["non-object"] += 1
                    continue
                kind = event_label(event.get("type"))
                item = event.get("item") or {}
                item_type = event_label(item.get("type")) if isinstance(item, dict) else "unknown"
                events[kind] += 1
                if item_type != "unknown":
                    items[item_type] += 1
                if (kind in TOOL_ITEMS or item_type in TOOL_ITEMS or any(term in kind.lower()
                        for term in ("tool_call", "exec_command", "web_search", "file_change",
                                     "execommand", "execcommand", "mcp_tool", "mcptool"))):
                    tool_events += 1
        summary = read(directory / (call_id + ".summary.json"), {})
        audits.append({"call_id": call_id, "raw_events_present": path.exists(),
            "raw_events_sha256": digest(path.read_bytes()) if path.exists() else None,
            "event_type_counts": dict(events), "item_type_counts": dict(items),
            "observed_tool_event_records": tool_events, "partial_final_lines": partial,
            "wrapper_completed_summary": bool(summary),
            "wrapper_rejected_tool_event": bool(summary.get("unauthorized_tool_event")),
            "delegate_returncode": summary.get("delegate_returncode")})
    called = {c["id"] for c in calls}
    audited = {a["call_id"] for a in audits}
    raw_present = {a["call_id"] for a in audits if a["raw_events_present"]}
    complete = {a["call_id"] for a in audits if a["wrapper_completed_summary"] and a["raw_events_present"]}
    return {"calls": audits, "model_call_ids_without_raw_audit": sorted(called - raw_present),
        "model_call_ids_without_completed_audit": sorted(called - complete),
        "audit_ids_without_model_call": sorted(audited - called),
        "observed_tool_event_records": sum(a["observed_tool_event_records"] for a in audits),
        "wrapper_rejections": sum(a["wrapper_rejected_tool_event"] for a in audits),
        "scope": "Observed event counts, not unique executed tools. Missing/incomplete raw audits cannot establish tool absence. No raw event payloads, prompts, reasoning, command arguments or outputs exported."}


def embedding_audit(campaign, config):
    """Separate cached local encoder evidence from subscription-model accounting."""
    saved_identity = read(campaign / "embedding-identity.json", {})
    identity = pick(saved_identity, ("repository", "revision", "model", "endpoint_model",
        "dim", "threads", "device", "aggregation", "limitation", "source_sha256"))
    identity["packages"] = pick(saved_identity.get("packages", {}),
        ("fastembed", "onnxruntime", "tokenizers", "numpy"))
    identity["file_sha256"] = {name: value for name, value in saved_identity.get("file_sha256", {}).items()
        if not Path(name).is_absolute() and ".." not in Path(name).parts
        and isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)}
    fixture = pick(read(campaign / "embedding-fixture.json", {}),
        ("identical_exact", "different_cosine", "dimensions", "wall_seconds", "cpu_seconds", "peak_rss_kib", "kind"))
    calls = []
    for saved in jsonl(campaign / "embedding-calls.jsonl"):
        call = pick(saved, ("utc", "status", "inputs", "tokens", "chunks", "wall_seconds", "cpu_seconds", "peak_rss_kib"))
        call["text_sha256"] = [value for value in saved.get("text_sha256", [])
                                if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)]
        call["has_error"] = bool(saved.get("error"))
        call["error_sha256"] = digest(str(saved["error"])) if saved.get("error") else None
        call["failure_class"] = failure_class(saved.get("error")) if saved.get("error") else None
        calls.append(call)
    enabled = bool(config.get("evolution", {}).get("embedding_model"))
    rewrite = config.get("arm") == "rewrite"
    summary = {
        "status": "disabled_by_rewrite_treatment" if rewrite else "enabled" if enabled else "disabled",
        "configured_model": config.get("evolution", {}).get("embedding_model"),
        "identity_recorded": bool(saved_identity), "identity_file": "embedding-identity.json",
        "model": identity.get("model"), "repository": identity.get("repository"), "revision": identity.get("revision"),
        "identity_file_hashes": len(identity["file_sha256"]), "native_endpoint_calls": len(calls),
        "call_status_counts": dict(Counter(c.get("status", "unknown") for c in calls)),
        "recorded_inputs": sum(c.get("inputs", 0) for c in calls),
        "recorded_local_tokens": sum(c.get("tokens", 0) for c in calls),
        "recorded_chunks": sum(c.get("chunks", 0) for c in calls),
        "recorded_endpoint_cpu_seconds": sum(c.get("cpu_seconds", 0.) for c in calls),
        "recorded_endpoint_wall_seconds": sum(c.get("wall_seconds", 0.) for c in calls),
        "calls_without_resource_measurements": sum("cpu_seconds" not in c or "wall_seconds" not in c for c in calls),
        "peak_rss_kib": max((c.get("peak_rss_kib", 0) for c in calls), default=0) or fixture.get("peak_rss_kib"),
        "startup_fixture_present": bool(fixture),
        "unexpected_evidence_in_disabled_treatment": bool(not enabled and (calls or saved_identity or fixture)),
        "scope": "Cached local CPU embedding endpoint only; no paid API route. Local tokens/CPU are separate from subscription-role counts. Startup fixture is infrastructure verification, not evolutionary evidence; its resource use is separate. Do not add wrapper elapsed time to overlapping endpoint wall time. Error calls missing measurements remain explicitly incomplete.",
    }
    return identity, {"summary": summary, "startup_fixture": fixture, "calls": calls}


def export(campaign, out):
    campaign, out = Path(campaign).resolve(), Path(out).resolve()
    if out == campaign or out.is_relative_to(campaign):
        raise ValueError("Public output must be outside the runtime campaign directory")
    config = read(campaign / "resolved-config.json", read(campaign / "campaign-manifest.json", {}))
    if not config:
        raise ValueError("No saved independent-search configuration")
    out.mkdir(parents=True, exist_ok=True)
    events = jsonl(campaign / "native-events.jsonl")
    samples = [e for e in events if e.get("kind") == "sampling"]
    calls = [compact_call(read(p)) for p in sorted((campaign / "model-calls").glob("*.json"))]
    native = rows(campaign / "programs.sqlite", "programs")
    canonical, lineage = {}, []
    for row in sorted(native, key=lambda row: (row["generation"], row.get("timestamp") or 0, row["id"])):
        meta = json_field(row, "metadata", {})
        administrative = bool(meta.get("_is_island_copy"))
        generation = row["generation"]
        if not administrative:
            if generation in canonical:
                raise ValueError("Multiple canonical native programs for one generation")
            canonical[generation] = row
            if row.get("code"):
                public_text(out / "programs" / f"generation-{generation:03d}.py", row["code"])
        lineage.append(pick(row, ("id", "parent_id", "generation", "timestamp", "island_idx", "system_prompt_id")) | {
            "administrative_seed_copy": administrative, "correct": bool(row.get("correct")),
            "fitness": row.get("combined_score"), "source_sha256": digest(row.get("code") or ""),
            "archive_inspiration_ids": json_field(row, "archive_inspiration_ids", []),
            "top_k_inspiration_ids": json_field(row, "top_k_inspiration_ids", []),
            "migration_count": len(json_field(row, "migration_history", [])),
            "embedding_dimensions": len(json_field(row, "embedding", [])),
            "patch_type": meta.get("patch_type"),
            "model": meta.get("llm_model", meta.get("model_name")),
            "failure_stage": meta.get("failure_stage"),
            "failure_reason_sha256": digest(str(meta["failure_reason"])) if meta.get("failure_reason") else None})
    generations = set(canonical) | {e["generation"] for e in samples}
    generations |= {int(p.name[4:]) for p in campaign.glob("gen_*") if p.name[4:].isdigit()}
    slots, evaluation_resources, evaluation_manifests = [], [], []
    episode_rows, replays = [], []
    for generation in sorted(generations):
        directory = campaign / f"gen_{generation}"
        row = canonical.get(generation, {})
        accepted = read(directory / "accepted-proposal.json", {})
        acceptance = read(directory / "acceptance-budget.json", {})
        held = read(directory / "held-proposal.json", {})
        episode_file = directory / "results/episodes.json"
        # The current study exposes only public training cases 0..4. Do not
        # follow metadata paths or extend this exporter to assessment pools.
        configured_cases = config.get("search_cases", list(range(5)))
        if episode_file.exists() and configured_cases != list(range(5)):
            raise ValueError("Episode export is restricted to public search cases 0..4")
        episodes = read(episode_file, [])
        if any(type(e.get("case")) is not int or e["case"] not in range(5) for e in episodes):
            raise ValueError("Saved search record contains a nonpublic case identifier")
        if len({e["case"] for e in episodes}) != len(episodes):
            raise ValueError("Duplicate search episode case identifier")
        manifest = read(directory / "results/manifest.json", {})
        if episodes and manifest and manifest.get("episodes") != len(episodes):
            raise ValueError("Saved episode count differs from evaluator manifest")
        candidate_hash = manifest.get("candidate_sha256") or (digest(row["code"]) if row.get("code") else None)
        if candidate_hash and row.get("code") and candidate_hash != digest(row["code"]):
            raise ValueError("Native candidate source differs from evaluated source")
        if manifest:
            evaluation_manifests.append({"generation": generation, **pick(manifest,
                ("evaluator", "episodes", "seed_start", "candidate_sha256", "source_sha256", "purpose"))})
        for episode in episodes:
            compact = {"generation": generation, "candidate_sha256": candidate_hash,
                **pick(episode, EPISODE_FIELDS), "has_error": bool(episode.get("error")),
                "error_sha256": digest(str(episode["error"])) if episode.get("error") else None}
            episode_rows.append(compact)
        if row.get("correct"):
            replay = next((e for e in episodes if e.get("trace")), None)
            if replay is not None:
                # Snapshot/belief/action triples are the established saved replay
                # interface. Copy no arbitrary additional candidate fields.
                trace = [{"world": pick(frame["world"], ("grid", "enemies", "agent", "origin", "step", "keys", "door_open", "reason")),
                          "belief": frame["belief"], "action": pick(frame["action"], ("move", "interact"))}
                         for frame in replay["trace"]]
                replays.append({"generation": generation, "candidate_sha256": candidate_hash,
                    **pick(replay, EPISODE_FIELDS), "has_error": bool(replay.get("error")),
                    "timeline": [pick(t, ("step", "correct", "audited", "coverage")) for t in replay.get("timeline", [])],
                    "trace": trace})
        resource = read(directory / "results/resource.json", {})
        correctness = read(directory / "results/correct.json", {})
        if resource:
            evaluation_resources.append({"generation": generation, **pick(resource,
                ("started_utc", "ended_utc", "elapsed_seconds", "candidate_cpu_seconds",
                 "evaluator_cpu_seconds", "episodes", "transitions", "completed"))})
        if accepted and not row and (directory / "main.py").exists():
            source = (directory / "main.py").read_text()
            if accepted.get("source_sha256") != digest(source):
                raise ValueError("Accepted pending source hash differs")
            public_text(out / "programs" / f"accepted-pending-{generation:03d}.py", source)
        status = ("valid" if row.get("correct") else "failed" if row else
                  "evaluated_pending_native_record" if episodes else "accepted_pending" if accepted else
                  "held" if held else "pending")
        slots.append({"generation": generation, "status": status, "program_id": row.get("id"),
            "fitness": row.get("combined_score"), "episodes": len(episodes),
            "transitions": sum(e.get("steps", 0) for e in episodes),
            "outcomes": dict(Counter(e.get("reason", "unknown") for e in episodes)),
            "mean_fitness": sum(e["combined_score"] for e in episodes) / len(episodes) if episodes else None,
            "mean_task": sum(e["task"] for e in episodes) / len(episodes) if episodes else None,
            "mean_map_accuracy": sum(e["model_accuracy"] for e in episodes) / len(episodes) if episodes else None,
            "accepted_for_evaluation": bool(acceptance or accepted),
            "acceptance_budget": pick(acceptance, ("all_role_calls_at_acceptance",
                "provider_elapsed_seconds_at_acceptance", "definition")),
            "held_stage": held.get("stage"),
            "held_reason_sha256": digest(str(held["reason"])) if held.get("reason") else None,
            "evaluator_correct": correctness.get("correct")})
    state = read(campaign / "run1-native-state.json", read(campaign / "native-state.json", {}))
    bandit = state.get("bandit_state") or {}
    bandit_counts = {name: {label: values[index] if index < len(values) else None
                           for label, values in (("submitted", bandit.get("n_submitted") or []),
                                                 ("completed", bandit.get("n_completed") or []))}
                     for index, name in enumerate(bandit.get("arm_names", []))}
    meta = read(campaign / "run1-meta-state.json", read(campaign / "meta-state.json", {}))
    prompts = rows(campaign / "prompts.sqlite", "system_prompts")
    evo, db = config.get("evolution", {}), config.get("database", {})
    role_counts = Counter(c["role"] for c in calls)
    event_counts = Counter(e.get("kind", "unknown") for e in events)
    mutation_calls = [c for c in calls if c["role"] in ("mutation", "repair")]
    rewrite = config.get("arm") == "rewrite"
    def feature(enabled, configured, observed):
        return {"configured": configured, "status": "enabled" if enabled else
                "disabled_by_rewrite_treatment" if rewrite else "disabled", "observed": observed}
    mechanisms = {
        "islands_and_parent_sampling": feature(True, pick(db, ("num_islands", "parent_selection_strategy")),
            {"samples": len(samples), "sampled_island_counts": dict(Counter(str(s.get("parent_island")) for s in samples))}),
        "mutation_bandit": feature(bool(evo.get("llm_dynamic_selection")), evo.get("llm_dynamic_selection"),
            {"native_arm_counts": bandit_counts, "requested_model_call_counts": dict(Counter(c.get("requested_model") for c in mutation_calls)),
             "selection_policy": "deterministic alternating model assignment, seed-only independent rewrites" if rewrite else "configured native dynamic selection"}),
        "operators": feature(True, {"types": evo.get("patch_types"), "probabilities": evo.get("patch_type_probs")},
            {"valid_descendant_patch_types": dict(Counter(r["patch_type"] or "unknown" for r in lineage
                if not r["administrative_seed_copy"] and r["generation"] > 0 and r["correct"]))}),
        "inspirations": feature(bool(db.get("num_archive_inspirations") or db.get("num_top_k_inspirations")),
            pick(db, ("num_archive_inspirations", "num_top_k_inspirations")),
            {"samples_with_inspirations": sum(bool(s.get("archive_inspiration_ids") or s.get("top_k_inspiration_ids")) for s in samples)}),
        "embeddings": feature(bool(evo.get("embedding_model")), evo.get("embedding_model"),
            {"native_embedding_events": event_counts["local_embedding"],
             "successful_nonzero_events": sum(e.get("kind") == "local_embedding" and e.get("dimensions", 0) > 0 for e in events)}),
        "novelty": feature(bool(evo.get("embedding_model")), pick(evo, ("code_embed_sim_threshold", "novelty_llm_models", "max_novelty_attempts")),
            {"native_decisions": event_counts["novelty_decision"], "judge_calls": role_counts["novelty"]}),
        "migration": feature(bool(db.get("migration_rate")) and db.get("num_islands", 1) > 1,
            pick(db, ("migration_interval", "migration_rate")), {"recorded_moves": sum(r["migration_count"] for r in lineage)}),
        "meta_recommendations": feature(bool(evo.get("meta_llm_models")), pick(evo, ("meta_llm_models", "meta_rec_interval")),
            {"calls": {r: role_counts[r] for r in ("summary", "global_insight", "recommendation")},
             "samples_with_recommendation_text": sum(bool(s.get("recommendation")) for s in samples),
             "programs_processed": meta.get("total_programs_meta_processed", 0)}),
        "prompt_evolution": feature(bool(evo.get("evolve_prompts")), pick(evo, ("evolve_prompts", "prompt_evolution_interval")),
            {"prompt_records": len(prompts), "mutation_calls": role_counts["prompt_mutation"]}),
    }
    sampling = [pick(e, ("utc", "generation", "parent_id", "parent_sha256", "parent_island",
        "archive_inspiration_ids", "top_k_inspiration_ids", "novelty_attempt", "resample_attempt")) |
        {"recommendation_present": bool(e.get("recommendation")),
         "parent_feedback_is_string": e.get("parent_feedback_is_string")} for e in samples]
    prefixes = []
    for calls_budget in sorted({s["acceptance_budget"].get("all_role_calls_at_acceptance") for s in slots
                               if s["acceptance_budget"].get("all_role_calls_at_acceptance") is not None}):
        eligible = [s for s in slots if s["status"] == "valid" and
            s["acceptance_budget"].get("all_role_calls_at_acceptance") is not None and
            s["acceptance_budget"]["all_role_calls_at_acceptance"] <= calls_budget]
        best = max(eligible, key=lambda s: (s["fitness"], -s["generation"])) if eligible else None
        prefixes.append({"all_role_call_prefix": calls_budget, "valid_generation_ids": [s["generation"] for s in eligible],
            "best_generation": best["generation"] if best else None, "best_fitness": best["fitness"] if best else None,
            "note": "Acceptance-time prefix; evaluations may complete later. Missing acceptance records are not inferred."})
    resources = read(campaign / "execution-resources.json", {})
    ledger = read(campaign / "call-budget-ledger.json", {})
    checkpoint_events = [e for e in events if e.get("kind") == "checkpoint_requested" and e.get("reason")]
    raw_stop = resources.get("stop_reason") or ledger.get("blocked_reason") or (checkpoint_events[-1]["reason"] if checkpoint_events else None)
    stop_source = ("execution-resources" if resources.get("stop_reason") else "budget-ledger" if ledger.get("blocked_reason")
                   else "native-checkpoint-event" if checkpoint_events else None)
    termination = {"source": stop_source, "reason": bounded_stop_reason(raw_stop),
        "reason_sha256": digest(str(raw_stop)) if raw_stop else None,
        "class": failure_class(raw_stop) if raw_stop else None,
        "recorded": bool(raw_stop),
        "note": None if raw_stop else "No explicit stop cause recorded; worker return status alone does not identify a blocker."}
    raw_audit = raw_tool_summary(campaign, calls)
    embedding_identity, embedding = embedding_audit(campaign, config)
    summary = {"schema": "proposal-independent-search-checkpoint-v1", "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.get("run_id"), "arm": config.get("arm"),
        "execution_status": resources.get("status", "prepared_or_active_without_final_resource_record"),
        "stop_reason": termination["reason"], "termination_cause": termination,
        "slot_counts": dict(Counter(s["status"] for s in slots)),
        "persisted_canonical_slots": len(canonical),
        "valid_descendants": sum(s["generation"] > 0 and s["status"] == "valid" for s in slots),
        "saved_episodes": sum(s["episodes"] for s in slots), "transitions": sum(s["transitions"] for s in slots),
        "all_role_calls": totals(calls),
        "roles": {r: totals([c for c in calls if c["role"] == r]) for r in sorted(set(ROLES) | set(role_counts))},
        "raw_tool_audit": {k: v for k, v in raw_audit.items() if k != "calls"},
        "embedding_audit": embedding["summary"],
        "native_event_counts": dict(event_counts), "mechanisms": mechanisms,
        "acceptance_prefixes": prefixes,
        "claim_limits": ["Development-case selection only; no assessment results or search-repeatability inference.",
            "Configured mechanisms and observed events are separate; events alone do not establish causal benefit.",
            "Empty/failed searches and missing usage remain explicit; unknown tokens are not zero.",
            "API list-price estimates are not subscription charges; assistant supervision usage is outside the ledger."],
        "new_model_calls": 0, "new_world_episodes": 0}
    outputs = {"summary": summary, "config": compact_config(config), "slots": slots,
        "embedding-identity": embedding_identity, "embedding-audit": embedding,
        "episodes": episode_rows, "evaluation-manifests": evaluation_manifests,
        "lineage": lineage, "sampling": sampling, "model-calls": calls, "raw-tool-audit": raw_audit,
        "evaluation-resources": evaluation_resources,
        "execution-grant": pick(read(campaign / "execution-grant.json", {}),
            ("started_utc", "hard_end_utc", "admission_end_utc", "manifest_sha256", "automatic_continuation")),
        "execution-resources": pick(resources, ("started_utc", "ended_utc", "hard_end_utc", "admission_end_utc",
            "supervisor_wall_seconds", "combined_rss_bytes", "returncode", "status",
            "automatic_continuation", "supervising_assistant_usage")) |
            {"stop_reason": bounded_stop_reason(resources.get("stop_reason")),
             "stop_reason_sha256": digest(str(resources["stop_reason"])) if resources.get("stop_reason") else None},
        "budget-ledger": pick(ledger,
            ("updated_utc", "limits", "totals", "deadline_utc", "primary_budgets", "token_budget_enforced",
             "cli_lock_wait_seconds", "token_definition", "elapsed_definition", "dollars")) |
            {"blocked_reason": bounded_stop_reason(ledger.get("blocked_reason")),
             "blocked_reason_sha256": digest(str(ledger["blocked_reason"])) if ledger.get("blocked_reason") else None}}
    for name, value in outputs.items():
        write(out / (name + ".json"), value)
    replay_path = out / "representative-replays.json.gz"
    with replay_path.open("wb") as handle:
        with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as compressed:
            compressed.write(json.dumps(clean(replays), separators=(",", ":"), allow_nan=False).encode())
    write(out / "replay-index.json", {"selection": "First saved public-training replay in each valid canonical generation; no episodes rerun.",
        "file": replay_path.name, "replays": [{"generation": r["generation"], "case": r["case"],
            "candidate_sha256": r["candidate_sha256"], "reason": r["reason"], "frames": len(r["trace"])} for r in replays]})
    input_paths = [campaign / "resolved-config.json", campaign / "campaign-manifest.json",
                  campaign / "execution-grant.json", campaign / "execution-resources.json",
                  campaign / "call-budget-ledger.json", campaign / "run1-native-state.json",
                  campaign / "native-events.jsonl", campaign / "programs.sqlite", campaign / "prompts.sqlite",
                  campaign / "embedding-identity.json", campaign / "embedding-fixture.json", campaign / "embedding-calls.jsonl"]
    input_paths += list((campaign / "model-calls").glob("*.json"))
    input_paths += list((campaign / "codex-event-audit").glob("*.summary.json"))
    for generation in generations:
        input_paths += [campaign / f"gen_{generation}" / name for name in
            ("acceptance-budget.json", "accepted-proposal.json", "held-proposal.json", "main.py",
             "results/episodes.json", "results/manifest.json", "results/resource.json", "results/correct.json")]
    inputs = {str(p.relative_to(campaign)): {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
              for p in sorted(set(input_paths)) if p.exists()}
    output_paths = [out / (name + ".json") for name in outputs] + [replay_path, out / "replay-index.json"]
    output_paths += list((out / "programs").glob("*.py"))
    output_hashes = {str(p.relative_to(out)): {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
                     for p in sorted(output_paths)}
    write(out / "provenance.json", {"inputs": inputs, "outputs": output_hashes,
        "reporter_sha256": digest(Path(__file__).read_bytes()),
        "snapshot_note": "Sequential saved-file snapshot; a running campaign can change between reads. Output hashes exclude this self-referential provenance file.",
        "native_database_access": "SQLite mode=ro; no database or cache exported",
        "omitted": ["Raw model/tool events", "prompts", "provider stdout/stderr", "recommendation text",
                    "private assessment seeds", "RNG states", "authentication files", "databases", "caches"]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = export(args.input, args.out)
    print(json.dumps({k: summary[k] for k in ("run_id", "arm", "execution_status", "valid_descendants", "saved_episodes")}))


if __name__ == "__main__":
    main()
