"""One authorized native judge fixture; zero world episodes/candidate slots."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run1_readiness import configure


async def main():
    configure()
    from dreamer.native_run1 import Run1CallBudget, install_run1_call_audit, CALL_ROLE
    from dreamer.provenance import sha256
    from dreamer.native_v3 import atomic_json
    from shinka.embed import AsyncEmbeddingClient
    from shinka.edit.async_apply import get_code_embedding_async
    from shinka.database import DatabaseConfig, ProgramDatabase, Program
    from shinka.core.novelty_judge import NoveltyJudge
    from shinka.core.async_novelty_judge import AsyncNoveltyJudge
    from shinka.llm import AsyncLLMClient
    root = ROOT / 'results/campaign-v4-run1'
    output = ROOT / 'artifacts/campaign-v4/run1/novelty-fixture.json'
    marker = root / 'novelty-fixture-start.json'
    if marker.exists():
        raise SystemExit('Fixture already attempted; inspect its preserved output, do not repeat')
    atomic_json(marker, {'status': 'started', 'utc': datetime.now(timezone.utc).isoformat()})
    budget = Run1CallBudget(root, deadline_utc='2026-10-09T04:24:23Z')
    install_run1_call_audit(budget)
    before = {x['id'] for x in budget.calls}
    seed = ROOT / 'controls/v1/predictive.py'
    source = seed.read_text()
    model = 'local/bge-small-code-chunks-v1@http://127.0.0.1:8771/v1'
    embedding, cost = await get_code_embedding_async(str(seed), AsyncEmbeddingClient(model))
    if not embedding:
        raise RuntimeError('Fixture requires the actual local embedding endpoint')
    fixture_dir = root / 'fixtures'
    fixture_dir.mkdir(exist_ok=True)
    db = ProgramDatabase(DatabaseConfig(db_path=str(fixture_dir / 'novelty.sqlite'), num_islands=1), embedding_model=model)
    parent = Program(id='run1-identical-source-fixture', code=source, language='python',
        generation=0, correct=True, combined_score=0., embedding=embedding,
        metadata={'patch_name': 'fixture_only', 'research_candidate': False},
        text_feedback='Fixture only: no task episode evaluated.', timestamp=time.time())
    db.add(parent)
    client = AsyncLLMClient(model_names=['headless/codex@gpt-6.1-sol?effort=high'], headless_work_dir=str(root))
    judge = AsyncNoveltyJudge(NoveltyJudge(language='python', similarity_threshold=.95, max_novelty_attempts=3), client)
    eligible = await judge.should_check_novelty_async(embedding, 1, parent, db)
    if not eligible:
        raise RuntimeError('Native novelty conditional path was not eligible')
    token = CALL_ROLE.set('readiness')
    try:
        accepted, metadata = await judge.assess_novelty_with_rejection_sampling_async(str(seed), embedding, parent, db)
    finally:
        CALL_ROLE.reset(token)
        db.close()
    calls = [x for x in budget.calls if x['id'] not in before]
    result = {'fixture': 'identical immutable source through genuine local embedding and native novelty judge',
              'not_discovery': True, 'new_world_episodes': 0, 'candidate_slots': 0,
              'source': 'controls/v1/predictive.py', 'source_sha256': sha256(seed),
              'local_embedding_model': model, 'embedding_dimensions': len(embedding),
              'embedded_characters': min(10000, len(source)), 'threshold': .95,
              'native_should_check': eligible, 'native_accepted': accepted,
              'metadata': metadata, 'expected_duplicate_rejection': not accepted,
              'calls': [{k: c.get(k) for k in ('id','role','requested_model','requested_effort','prompt_sha256','elapsed_seconds','returncode','usage')} for c in calls],
              'call_count': len(calls), 'budget_role': 'readiness', 'ledger_totals': budget.totals()}
    atomic_json(output, result)
    atomic_json(marker, {'status': 'completed', 'output': str(output.relative_to(ROOT)), 'source_sha256': sha256(seed)})
    if len(calls) != 1 or metadata.get('novelty_checks_performed') != 1:
        raise RuntimeError('Fixture did not produce exactly one actual native judge call')
    print(json.dumps({'accepted': accepted, 'calls': len(calls), 'max_similarity': metadata.get('max_similarity'), 'output': str(output)}), flush=True)


async def with_heartbeat():
    async def pulse():
        while True:
            await asyncio.sleep(.05)
    beat = asyncio.create_task(pulse())
    try:
        await main()
    finally:
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)


if __name__ == '__main__':
    asyncio.run(with_heartbeat())
