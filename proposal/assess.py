"""Execute the frozen assessment with the unchanged paired-panel runner.

Only the runner's hardcoded publication-purpose label is corrected afterward;
candidate execution, scores, case generation and resource limits are unchanged.
"""
import argparse
from pathlib import Path

from proposal.panel import ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    protocol = read(args.protocol)
    frozen_path = ROOT / protocol['assessment_freeze']['path']
    if sha(frozen_path) != protocol['assessment_freeze']['sha256']:
        raise ValueError('Original assessment specification changed')
    frozen = read(frozen_path)
    if protocol['case_count'] != frozen['case_count'] or protocol['analysis'] != frozen['analysis']:
        raise ValueError('Assessment sample size or analysis differs from freeze')
    for name, expected in frozen['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen scientific source changed: ' + name)
    expected = {'seed': frozen['control'], 'generation_2': frozen['selected_program']}
    if protocol['programs'] != expected:
        raise ValueError('Assessment programs differ from freeze')
    from proposal.panel import main as run_panel
    run_panel()
    public = ROOT / protocol['public_output']
    for arm in protocol['programs']:
        path = public / arm / 'manifest.json'
        manifest = read(path)
        manifest['runner_default_purpose'] = manifest['purpose']
        manifest['purpose'] = 'Fresh assessment of two frozen programs; no selection on this panel'
        manifest['assessment_freeze_sha256'] = sha(frozen_path)
        manifest['metadata_adapter'] = 'proposal/assess.py; purpose label only, no numeric changes'
        write(path, manifest)


if __name__ == '__main__':
    main()
