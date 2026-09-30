# Campaign execution and provenance

Supporting record for the completed 50-slot campaign. See the [research findings](campaign-findings.md) for interpretation of the experiment.


Recovery inspected the earlier tasks and host processes before starting one
controller. A SQLite backup and complete campaign-file snapshot preserve the
pre-recovery state. Saved slots 0–26 and interrupted slot 27's source were retained.
The saved proposal was evaluated into a new directory and inserted through native
Shinka processing, preserving its parent and inspirations. The recovery wrapper
also restores saved native recommendations and pending summaries, which the pinned
async runner did not reload itself. An exclusive lock prevents duplicate recovery
controllers. The controller completed slots 0–49 and flushed recommendations.

Failed slots remain visible: **21 and 42** were killed by the native 15-minute
wall-time check without complete metrics. Slot 21 was killed immediately after
its logged evaluation submission; the underlying timing cause is unresolved.
Slot 42 includes a large clock gap of unknown cause, so its elapsed time is not
candidate CPU time. **30 and 41** each contain one worker kill near the 10-second
per-episode CPU cap.
No failed slot was replaced to improve the reported results. The original native
scores, saved generation files, assessment artifacts and Namazu proposal are
preserved; the native log was only appended. Native startup refreshed the pricing
snapshot, whose original is retained in the backup.

Actual upstream ShinkaEvolve is pinned to
[`9912af1`](https://github.com/SakanaAI/ShinkaEvolve/tree/9912af12d423504b8d580f4179fd15f5f88b8c50)
and installed Python sources match that revision. Four islands, weighted parent
sampling, archive/top-k inspirations, lineage, migration and interval/final
recommendations remain native. The immutable historical manifest records 100
slots; per-invocation execution records enforce the requested **50-slot stop**.
The evaluator, development pool, objective and original driver hashes did not change.

Mutation/fix and the separate recommendation client both use only
`headless/codex@gpt-6-astra?effort=high` with `HEADLESS_BILLING=subscription` and
Headless revision [`93cd9b0`](https://github.com/RobertTLange/headless-cli/tree/93cd9b06b85f848af1308c41e018991b33907c5e).
API credentials are stripped from child environments. The evaluator uses local
Python without an LLM; embeddings, novelty LLM and prompt evolution are disabled.
Embedding-based novelty and bandit model selection are therefore not demonstrated.
Native dollar displays are API-list-price estimates, not subscription charges.
The pinned provider forwards model and reasoning effort; stored temperature and
`max_tokens` values are metadata, not enforced Codex CLI controls on this route.

The original read-only app-server failure remains documented as a historical
blocker. Recovery used tool-approved host execution through the existing login;
authentication, global Codex settings, approval policies and candidate isolation
were unchanged. [Runtime audit](../docs/upstream.md),
[preservation checks](../artifacts/campaign-v2/completed-50/preservation.json),
[completed execution](../artifacts/campaign-v2/completed-50/execution-audit.json),
[resolved settings](../artifacts/campaign-v2/completed-50/dreamer-resolved.json) and
[native recommendations](../artifacts/campaign-v2/completed-50/recommendations)
record the recovery. Earlier blocked-state artifacts are historical, not current status.
