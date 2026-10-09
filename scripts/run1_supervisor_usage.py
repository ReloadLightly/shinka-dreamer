"""Export metadata-only RUN1 supervising-agent usage, separately from search.

These four pre-existing session IDs identify the supervising conversation and
its delegated agents. No messages, reasoning, tool arguments or credentials are
retained. This cannot measure unreported service-side or approval-review usage.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = datetime.fromisoformat('2026-10-09T00:54:23+00:00')
SESSIONS = {
    'supervisor': '01a11c1a-3697-7a61-81ee-3ec1745ef2af',
    'native_audit': '01a11c38-ffc5-7f13-9b11-fd2beb0e70ef',
    'visualization': '01a11c38-e1d0-7793-a099-21166d52fed7',
    'environment': '01a11c38-bbd2-7e33-876c-b25ab55b976f',
}
KEYS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
        'output_tokens', 'reasoning_output_tokens', 'total_tokens')


def main():
    records = []
    for role, session_id in SESSIONS.items():
        paths = list(Path('/home/roland/.codex/sessions/2026/10/08').glob(f'*{session_id}.jsonl'))
        if len(paths) != 1:
            raise RuntimeError(f'Expected one session for {role}, found {len(paths)}')
        baseline = latest = None
        previous = None
        decreases = 0
        with paths[0].open() as source:
            for index, line in enumerate(source):
                # A concurrent final partial record is not a completed event.
                if not line.endswith('\n'):
                    continue
                event = json.loads(line)
                payload = event.get('payload', {})
                if index == 0:
                    if (event.get('type') != 'session_meta' or payload.get('id') != session_id
                            or payload.get('cwd') != str(ROOT)):
                        raise RuntimeError('Session identity or repository mismatch')
                if event.get('type') != 'event_msg' or payload.get('type') != 'token_count':
                    continue
                usage = (payload.get('info') or {}).get('total_token_usage')
                if not usage:
                    continue
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                point = {'timestamp': timestamp.isoformat(),
                         'usage': {key: int(usage.get(key, 0)) for key in KEYS}}
                if previous and any(point['usage'][key] < previous['usage'][key] for key in KEYS):
                    decreases += 1
                previous = point
                if timestamp < START:
                    baseline = point
                else:
                    latest = point
        if baseline is None or latest is None:
            raise RuntimeError(f'Missing before/after usage boundary for {role}')
        delta = {key: latest['usage'][key] - baseline['usage'][key] for key in KEYS}
        if decreases or any(value < 0 for value in delta.values()):
            raise RuntimeError(f'Nonmonotonic session usage for {role}; manual accounting needed')
        delta['uncached_input_plus_output_tokens'] = (
            delta['input_tokens'] - delta['cached_input_tokens'] + delta['output_tokens'])
        records.append({'role': role, 'session_id': session_id,
                        'baseline': baseline, 'latest': latest, 'delta': delta})
    totals = {key: sum(row['delta'][key] for row in records) for key in records[0]['delta']}
    output = {
        'schema': 1, 'run_id': 'v4-run1',
        'snapshot_utc': datetime.now(timezone.utc).isoformat(),
        'start_utc': START.isoformat(), 'sessions': records, 'totals': totals,
        'scope': 'Supervising conversation plus three explicitly identified delegated agents; excludes experiment model sessions.',
        'limitations': [
            'Reported cumulative metadata deltas, not a subscription allowance or bill.',
            'Input includes cached input; uncached measure subtracts cached input once.',
            'Reasoning output is already included in output and is not added twice.',
            'The start boundary is the last reported event before start, so an event spanning the boundary cannot be apportioned.',
            'The active turn and work after the latest reported event are not yet counted.',
            'Approval-review agents and unreported service-side computation are outside these four session records.',
        ],
    }
    path = ROOT / 'artifacts/campaign-v4/run1/supervisor-usage.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'snapshot_utc': output['snapshot_utc'], 'totals': totals}))


if __name__ == '__main__':
    main()
