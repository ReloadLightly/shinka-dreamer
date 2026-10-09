"""Assess frozen original-task controls through the unchanged episode runner.

This adds study validation and publication labels only. World generation,
candidate isolation, scoring and resource limits remain in the frozen runner.
"""
import argparse
from pathlib import Path

from proposal.panel import ROOT, read, sha, write

ARMS = {'generation_2', 'memory_pathfinder', 'local_map', 'no_risk'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    if protocol['case_count'] != 256 or set(protocol['programs']) != ARMS:
        raise ValueError('Expected the four prospectively specified controls on 256 pairs')
    if protocol['evaluator'] != 'namazu-proposal-reconstruction-v1':
        raise ValueError('Original-task evaluator must remain unchanged')
    for spec in protocol['programs'].values():
        if sha(ROOT / spec['path']) != spec['sha256']:
            raise ValueError('Frozen program identity changed')
    if protocol['analysis']['bootstrap_seed'] != 6109256:
        raise ValueError('Unexpected prespecified analysis stream')
    from proposal.panel import main as run_panel
    run_panel()
    public = ROOT / protocol['public_output']
    for arm in protocol['programs']:
        path = public / arm / 'manifest.json'
        manifest = read(path)
        manifest['runner_default_purpose'] = manifest['purpose']
        manifest['purpose'] = 'Fresh frozen-program comparator and mechanism assessment; no selection'
        manifest['protocol_sha256'] = sha(args.protocol)
        manifest['metadata_adapter'] = 'proposal/control_panel.py; publication labels only'
        write(path, manifest)


if __name__ == '__main__':
    main()
