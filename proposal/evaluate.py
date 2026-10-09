"""Native Shinka scheduler CLI and cheap standalone proposal evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from proposal.evaluation import IDENTITY, aggregate, run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--program_path', default=str(ROOT/'proposal/initial.py'))
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--replay-first', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    out = Path(args.results_dir)
    if (out/'episodes.json').exists():
        parser.error('Results already exist; use a separate output directory')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(args.episodes):
        row = run_episode(args.program_path, args.seed_start + index,
                          replay=args.replay_first and index == 0)
        rows.append(row)
    (out/'episodes.json').write_text(json.dumps(rows, separators=(',', ':')) + '\n')
    metrics = aggregate(rows)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (out/'correct.json').write_text(json.dumps({'correct': not any(r['error'] for r in rows),
        'error': [r['error'] for r in rows if r['error']] or None}) + '\n')
    files = ['proposal/initial.py', 'proposal/worker.py', 'proposal/evaluation.py',
             'proposal/evaluate.py', 'dreamer/world.py', 'dreamer/evaluation.py', 'dreamer/isolation.py']
    manifest = {'evaluator': IDENTITY, 'episodes': args.episodes, 'seed_start': args.seed_start,
        'candidate_sha256': hashlib.sha256(Path(args.program_path).read_bytes()).hexdigest(),
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        'purpose': 'Development evaluation; no held-out or evolutionary claim'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
