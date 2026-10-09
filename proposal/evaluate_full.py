"""Validate the frozen original evaluator, then record its separate resource use.

The original evaluation CLI, objective, case panel and artifacts are unchanged.
This wrapper is invoked for every native evaluation, including generation zero.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_SOURCES = (
    'proposal/initial.py', 'proposal/worker.py', 'proposal/evaluation.py',
    'proposal/evaluate.py', 'proposal/evaluate_full.py', 'dreamer/world.py',
    'dreamer/evaluation.py', 'dreamer/isolation.py',
)


def validate_sources(manifest_path):
    """Fail before importing the evaluator if any frozen dependency changed."""
    manifest = json.loads(Path(manifest_path).read_text())
    sources = manifest.get('source_sha256', manifest.get('sources'))
    if not isinstance(sources, dict) or not set(REQUIRED_SOURCES) <= set(sources):
        raise ValueError('Campaign manifest lacks required frozen evaluator sources')
    for name, expected in sources.items():
        path = (ROOT / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(ROOT):
            raise ValueError(f'Frozen source must be inside repository: {name}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Frozen source SHA256 mismatch: {name}')
    return manifest


def _cpu_usage(who):
    value = resource.getrusage(who)
    return value.ru_utime, value.ru_stime


def main():
    # Only remove our flag; proposal.evaluate owns all original CLI semantics.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--campaign-manifest', required=True)
    parser.add_argument('--record-first', choices=['yes', 'no'], default='no')
    args, evaluator_args = parser.parse_known_args()
    if args.record_first == 'yes':
        evaluator_args.append('--replay-first')
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--results_dir', required=True)
    output_args, _ = output_parser.parse_known_args(evaluator_args)
    out = Path(output_args.results_dir)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    before_self = _cpu_usage(resource.RUSAGE_SELF)
    before_children = _cpu_usage(resource.RUSAGE_CHILDREN)
    original_argv = sys.argv
    error = None
    try:
        validate_sources(args.campaign_manifest)
        from proposal.evaluate import main as evaluate_main
        sys.argv = [original_argv[0], *evaluator_args]
        evaluate_main()
        # Native Shinka forwards text_feedback but not correct.json's errors.
        # Expose actual failures without changing the original numeric metrics.
        correctness = json.loads((out / 'correct.json').read_text())
        if correctness.get('error'):
            metrics_path = out / 'metrics.json'
            metrics = json.loads(metrics_path.read_text())
            failures = correctness['error']
            if not isinstance(failures, list):
                failures = [failures]
            failures = list(dict.fromkeys(str(value) for value in failures))
            metrics['text_feedback'] += ' Actual execution failures: ' + json.dumps(failures)
            metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        sys.argv = original_argv
        after_self = _cpu_usage(resource.RUSAGE_SELF)
        after_children = _cpu_usage(resource.RUSAGE_CHILDREN)
        self_user, self_system = [end - start for start, end in zip(before_self, after_self)]
        child_user, child_system = [end - start for start, end in zip(before_children, after_children)]
        payload = {
            'started_utc': started_utc,
            'ended_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - started,
            'evaluator_user_cpu_seconds': self_user,
            'evaluator_system_cpu_seconds': self_system,
            'evaluator_cpu_seconds': self_user + self_system,
            'candidate_user_cpu_seconds': child_user,
            'candidate_system_cpu_seconds': child_system,
            'candidate_cpu_seconds': child_user + child_system,
            'total_cpu_seconds': self_user + self_system + child_user + child_system,
            'campaign_manifest': str(Path(args.campaign_manifest).resolve()),
            'completed': error is None,
            'error': error,
            'definition': 'Monotonic wrapper elapsed; getrusage deltas for evaluator and reaped candidate children. Resource metadata does not alter fitness.',
        }
        out.mkdir(parents=True, exist_ok=True)
        temporary = out / 'resource.json.tmp'
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(out / 'resource.json')


if __name__ == '__main__':
    main()
