"""Resume v3 with an audited whitespace-preserving Headless parser compatibility fix."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dreamer import native_v3
    from dreamer.headless_transport import install
    from dreamer.provenance import record, sha256
    from scripts import resume_v3
    from shinka.llm.providers import headless
    original_install_audit = native_v3.install_call_audit
    original_parser = inspect.getsource(headless._parse_stdout)

    def install_audited_transport(root):
        root = Path(root)
        previous = json.loads((root / 'prompt-amendment.json').read_text())
        if sha256(ROOT / 'scripts/resume_v3.py') != previous['new_resume_driver_sha256']:
            raise ValueError('Previously frozen resume driver changed')
        path = root / 'transport-amendment.json'
        if path.exists():
            amendment = json.loads(path.read_text())
            for name, expected in amendment['added_source_sha256'].items():
                if sha256(ROOT / name) != expected:
                    raise ValueError('Frozen transport compatibility source changed: ' + name)
            if amendment['upstream_parser_sha256'] != hashlib.sha256(original_parser.encode()).hexdigest():
                raise ValueError('Pinned upstream parser changed')
        else:
            attempts = []
            for metadata in sorted(root.glob('gen_*/attempts/**/metadata.json')):
                item = json.loads(metadata.read_text())
                attempts.append({'generation':item['generation'], 'novelty_attempt':item['novelty_attempt'],
                    'resample_attempt':item['resample_attempt'],'patch_attempt':item['patch_attempt'],
                    'success':item['success'],'metadata_sha256':sha256(metadata),
                    'error_msg':item.get('error_msg')})
            amendment = {'type':'transport compatibility bug fix, separate from predictive-learning extension',
                'campaign_sha256':sha256(root / 'campaign-manifest.json'),
                'preceding_prompt_amendment_sha256':sha256(root / 'prompt-amendment.json'),
                'reason':'Pinned Headless _parse_stdout removes every blank line, corrupting exact-match SEARCH blocks and potentially multiline strings.',
                'old_behavior':'Filter all blank lines before parsing usage and joining assistant content.',
                'new_behavior':'Remove only trailing blank lines after usage; retain every interior assistant line.',
                'upstream_parser_sha256':hashlib.sha256(original_parser.encode()).hexdigest(),
                'upstream_installed_files_unchanged':True,
                'runtime_hook':'Only shinka.llm.providers.headless._parse_stdout is rebound in memory.',
                'added_source_sha256':{p:sha256(ROOT/p) for p in ('scripts/resume_v3_transport.py','dreamer/headless_transport.py','tests/test_headless_transport.py')},
                'unchanged':['environment','evaluator','objective','development_pool','models','effort','operators','islands','sampling','feedback','meta interval','total slot budget'],
                'pre_fix_attempts':attempts,
                'confirmed_defect_evidence_sha256':sha256(ROOT / 'artifacts/campaign-v3/headless-whitespace-defect.json'),
                'checkpoint':json.loads((root / 'transport-fix-checkpoint.json').read_text())}
            record(path, amendment)
            record(ROOT / 'artifacts/campaign-v3/transport-amendment.json', amendment)
        original_install_audit(root)
        install()
        print('Installed recorded whitespace-preserving Headless parser compatibility hook', flush=True)

    native_v3.install_call_audit = install_audited_transport
    resume_v3.main()


if __name__ == '__main__':
    main()
