"""Counts-only local UI checks; no models, worlds or live data inspection."""
import hashlib
import io
import json
from pathlib import Path

import pytest
from scripts import v3_webui as ui


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    plan = {'sample_size': 2, 'regimes': ['uniform', 'stationary', 'switch'],
            'conditions': [{'name': 'memory'}, {'name': 'selected'}],
            'matched': {'shadow_conditions': ['a', 'b', 'c', 'd', 'e']},
            'counts': {'condition_episodes': 12, 'passive_shadow_episode_passes': 30}}
    plan_path = tmp_path / 'public/preregistration.json'
    plan_path.parent.mkdir()
    plan_path.write_text(json.dumps(plan))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    raw = tmp_path / 'private-assessment'
    raw.mkdir()
    monkeypatch.setattr(ui, 'PLAN', plan_path)
    monkeypatch.setattr(ui, 'PLAN_SHA256', digest)
    monkeypatch.setattr(ui, 'ASSESSMENT', raw)
    return raw, plan_path, digest


def touch(root, relative, content='outcome or private values MUST NOT be read'):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_progress_reads_no_outcomes_and_counts_only_registered_names(campaign, monkeypatch):
    raw, plan, digest = campaign
    for name in ('cases/000000.json', 'cases/000002.json', 'cases/000001.json.partial',
                 'episode-checkpoints/000000/memory--uniform.json',
                 'episode-checkpoints/000001/selected--switch.json',
                 'episode-checkpoints/000002/selected--switch.json',
                 'episode-checkpoints/000000/undeclared--uniform.json',
                 'episode-checkpoints/000000/memory--uniform.json.partial',
                 'matched/000000--uniform.json', 'matched/000001--switch.json',
                 'matched/000002--uniform.json', 'matched/000000--private.json',
                 'execution-complete.json'):
        touch(raw, name)
    original_text, original_bytes = Path.read_text, Path.read_bytes
    def guarded_text(path, *args, **kwargs):
        assert path == plan, 'Outcome/checkpoint content must not be read'
        return original_text(path, *args, **kwargs)
    def guarded_bytes(path, *args, **kwargs):
        assert path == plan, 'Outcome/checkpoint content must not be read'
        return original_bytes(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded_text)
    monkeypatch.setattr(Path, 'read_bytes', guarded_bytes)
    result = ui.assessment_progress()
    assert result['durable'] == {'paired_cases': 1, 'world_episode_checkpoints': 2, 'shadow_group_files': 2}
    assert result['planned'] == {'paired_cases': 2, 'world_episodes': 12, 'shadow_groups': 6, 'shadow_passes': 30}
    assert result['markers']['execution_complete'] is True
    assert 'completed_shadow_passes' not in result
    assert result['new_world_episodes'] == result['new_model_calls'] == 0


def test_launch_metadata_is_allowlisted_and_partial_write_is_tolerated(campaign):
    raw, plan, digest = campaign
    start = {'plan_sha256': digest, 'started_utc': '2026-10-08T21:00:00+00:00',
             'command': ['SECRET-SEED-ARRAY'], 'private_laws': 'SECRET-LAW',
             'treatment_effects': 'SECRET-EFFECT'}
    touch(raw, 'launch-20261008.started.json', json.dumps(start))
    result = ui.assessment_progress()
    assert result['launch']['frozen_plan_matches'] is True
    assert result['launch']['status'] == 'Started; no exit record yet'
    assert 'SECRET' not in json.dumps(result)
    touch(raw, 'launch-20261008.finished.json', '{')
    assert ui.assessment_progress()['launch']['status'] == 'Launch metadata update in progress'
    touch(raw, 'launch-20261008.finished.json', json.dumps({**start, 'exit_code': 1}))
    assert ui.assessment_progress()['launch']['status'] == 'Exited with an error'


def test_frozen_plan_identity_is_required(campaign):
    _, plan, _ = campaign
    plan.write_text(plan.read_text() + '\n')
    with pytest.raises(ValueError, match='identity'):
        ui.assessment_progress()


def test_http_routes_are_read_only_and_exact(campaign):
    # Exercise BaseHTTPRequestHandler parsing/dispatch without requiring a socket
    # capability in the sandbox. This connection captures only HTTP bytes.
    class Connection:
        def __init__(self, request):
            self.input = io.BytesIO(request)
            self.output = bytearray()

        def makefile(self, *args, **kwargs):
            return self.input

        def sendall(self, data):
            self.output.extend(data)

    def request(path, method='GET'):
        connection = Connection(f'{method} {path} HTTP/1.0\r\nHost: localhost\r\n\r\n'.encode())
        ui.CampaignUI(connection, ('127.0.0.1', 12345), object(),
                      search_root=str(campaign[0]), directory=str(campaign[0]))
        headers, body = bytes(connection.output).split(b'\r\n\r\n', 1)
        return int(headers.split(b' ', 2)[1]), headers, body

    status, headers, body = request('/assessment_progress')
    assert status == 200 and json.loads(body)['durable']['paired_cases'] == 0
    assert b'Cache-Control: no-store' in headers
    for path in ('/assessment.html', '/search-explorer.html'):
        status, _, body = request(path)
        assert status == 200 and b'<html' in body
        status, _, body = request(path, 'HEAD')
        assert status == 200 and body == b''
    for path, code in (('/assessment_progress?pool=secret', 400),
                       ('/assessment.html?path=secret', 400),
                       ('/results/private/seed.json', 404),
                       ('/assessment_progress/../preregistration.json', 404),
                       ('/get_program_count?db_path=../other.sqlite', 403)):
        assert request(path)[0] == code
    assert request('/assessment_progress', 'POST')[0] == 501
