"""Serve the installed native Shinka UI for this campaign on loopback only.

Upstream assets and handlers stay unchanged. This adapter confines HTTP reads to
the selected campaign database and native UI assets; it is not a search controller.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinka.webui import visualization as native
from scripts.web_presentation import native_html, local_html

ASSETS = {'/', '/index.html', '/viz_tree.html', '/compare.html', '/favicon.png', '/sakana.jpg'}
READ_ROUTES = {'/get_programs', '/get_programs_summary', '/get_program_count',
               '/get_program_details', '/get_meta_files', '/get_meta_content',
               '/get_database_stats', '/get_system_prompts', '/get_plots'}
ASSESSMENT = ROOT / 'results/campaign-v3-assessment'
PLAN = ROOT / 'artifacts/campaign-v3/assessment/preregistration.json'
PLAN_SHA256 = '2b292e2373cc1d5b3c558a5339ed1588323272e2dbc606ead130640bfd1aae7d'
LOCAL_ASSETS = {
    '/assessment.html': ROOT / 'scripts/assets/v3-assessment.html',
    '/search-explorer.html': ROOT / 'artifacts/campaign-v3/search-explorer.html',
}


def assessment_progress():
    """Read public plan/launcher metadata and count private filenames only.

    Never open outcome checkpoints, shadow records, private seeds or traces.
    A durable shadow-group file does not establish five successful executions.
    """
    encoded = PLAN.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != PLAN_SHA256:
        raise ValueError('Frozen assessment plan identity differs')
    plan = json.loads(encoded)
    n = plan['sample_size']
    regimes = set(plan['regimes'])
    conditions = {row['name'] for row in plan['conditions']}
    shadow_count = len(plan['matched']['shadow_conditions'])

    def case_number(text):
        return bool(re.fullmatch(r'\d{6}', text)) and 0 <= int(text) < n

    cases = sum(case_number(p.stem) for p in (ASSESSMENT / 'cases').glob('*.json'))
    worlds = 0
    for path in (ASSESSMENT / 'episode-checkpoints').glob('*/*.json'):
        pieces = path.stem.split('--')
        worlds += bool(case_number(path.parent.name) and len(pieces) == 2
                       and pieces[0] in conditions and pieces[1] in regimes)
    groups = 0
    for path in (ASSESSMENT / 'matched').glob('*.json'):
        pieces = path.stem.split('--')
        groups += bool(len(pieces) == 2 and case_number(pieces[0]) and pieces[1] in regimes)

    # A launcher finish record follows wait() on the entire timed command.
    # Its JSON is written in one small operation; tolerate a concurrent write.
    launch = {'status': 'No launch record', 'started_utc': None,
              'finished_utc': None, 'exit_code': None, 'frozen_plan_matches': None}
    starts = sorted(ASSESSMENT.glob('launch-*.started.json'))
    if starts:
        first = starts[-1]
        finished = first.with_name(first.name.replace('.started.json', '.finished.json'))
        try:
            record = json.loads((finished if finished.exists() else first).read_text())
            matches = record.get('plan_sha256') == PLAN_SHA256
            exit_code = record.get('exit_code')
            launch = {'status': ('Exited successfully' if exit_code == 0 else 'Exited with an error'
                                if isinstance(exit_code, int) else 'Started; no exit record yet'),
                      'started_utc': record.get('started_utc'),
                      'finished_utc': record.get('finished_utc'), 'exit_code': exit_code,
                      'frozen_plan_matches': matches}
        except (OSError, ValueError):
            launch['status'] = 'Launch metadata update in progress'
    paused_by_user = False
    checkpoint = PLAN.parent / 'operator-checkpoint-complete.json'
    if checkpoint.is_file():
        try:
            closure = json.loads(checkpoint.read_text())
            same_plan = closure.get('plan_sha256') == PLAN_SHA256
            closed_at = datetime.fromisoformat(closure['recorded_utc'].replace('Z', '+00:00'))
            started = launch.get('started_utc')
            later_launch = bool(started and datetime.fromisoformat(started.replace('Z', '+00:00')) > closed_at)
            paused_by_user = same_plan and closure.get('status') == 'paused-by-user' and not later_launch
            if paused_by_user: launch['status'] = 'Checkpointed and paused at user request'
        except (OSError, ValueError, KeyError):
            pass
    complete = (ASSESSMENT / 'execution-complete.json').is_file()
    closed = (PLAN.parent / 'analysis-closure.json').is_file()
    reports = (ASSESSMENT / 'report-follow-through-complete.json').is_file()
    return {
        'sampled_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': PLAN_SHA256,
        'planned': {'paired_cases': n, 'world_episodes': plan['counts']['condition_episodes'],
                    'shadow_groups': n * len(regimes),
                    'shadow_passes': plan['counts']['passive_shadow_episode_passes']},
        'durable': {'paired_cases': cases, 'world_episode_checkpoints': worlds,
                    'shadow_group_files': groups},
        'shadows_per_group': shadow_count,
        'shadow_scope': 'Recorded replay groups only. Files may record failures; group count is not a count of valid or completed individual shadow passes.',
        'launch': launch,
        'markers': {'paused_by_user': paused_by_user, 'execution_complete': complete, 'analysis_closed': closed,
                    'saved_data_reports_complete': reports},
        'status_scope': 'Marker existence and launcher metadata; not a process heartbeat or outcome inspection.',
        'count_scope': 'Durable files include failed executions. Counts may advance at different checkpoint boundaries; a paired case is saved after all its conditions and replay groups.',
        'privacy': 'No treatment effects, private seeds, movement laws or traces are read or returned.',
        'new_world_episodes': 0, 'new_model_calls': 0,
    }


class CampaignUI(native.DatabaseRequestHandler):
    campaign_label = 'Unknown dynamics · v3 wave 1'
    campaign_task = 'campaign-v3'

    def native_asset_response(self, path, head=False):
        name = 'index.html' if path == '/' else path.lstrip('/')
        asset = Path(native.__file__).resolve().parent / name
        payload = native_html(asset.read_text()).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if not head: self.wfile.write(payload)

    def local_response(self, path, head=False):
        if path == '/assessment_progress':
            try:
                payload = json.dumps(assessment_progress(), allow_nan=False).encode()
            except (OSError, ValueError, KeyError):
                return self.send_error(503, 'Frozen progress metadata unavailable')
            content_type = 'application/json; charset=utf-8'
        else:
            try:
                payload = local_html(LOCAL_ASSETS[path].read_text()).encode('utf-8')
            except OSError:
                return self.send_error(404, 'Saved campaign page unavailable')
            content_type = 'text/html; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if not head:
            self.wfile.write(payload)

    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in LOCAL_ASSETS or parsed.path == '/assessment_progress':
            if self.campaign_task != 'campaign-v3':
                return self.send_error(404, 'Historical assessment resources belong to the v3 campaign')
            if parsed.query:
                return self.send_error(400, 'This campaign resource takes no parameters')
            return self.local_response(parsed.path)
        if parsed.path == '/list_databases':
            return self.send_json_response([{'path':'programs.sqlite',
                'actual_path':'programs.sqlite', 'name':self.campaign_label,
                'task':self.campaign_task}])
        if parsed.path in ('/', '/index.html', '/viz_tree.html', '/compare.html'):
            return self.native_asset_response(parsed.path)
        if parsed.path in READ_ROUTES:
            if query.get('db_path') != ['programs.sqlite']:
                return self.send_error(403, 'Only the selected campaign is available')
            for key in ('generation', 'processed_count'):
                if key in query and (len(query[key]) != 1 or not re.fullmatch(r'\d+', query[key][0])):
                    return self.send_error(400, 'Integer generation required')
        elif parsed.path not in ASSETS:
            return self.send_error(404, 'No such campaign UI resource')
        return super().do_GET()

    def do_HEAD(self):
        parsed = urlsplit(self.path)
        if parsed.path in LOCAL_ASSETS or parsed.path == '/assessment_progress':
            if self.campaign_task != 'campaign-v3':
                return self.send_error(404)
            if parsed.query:
                return self.send_error(400)
            return self.local_response(parsed.path, head=True)
        if parsed.path in ('/', '/index.html', '/viz_tree.html', '/compare.html'):
            return self.native_asset_response(parsed.path, head=True)
        if urlsplit(self.path).path not in ASSETS:
            return self.send_error(404)
        if self.path == '/':
            self.path = '/index.html'
        return super().do_HEAD()

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT/'results/campaign-v3')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    allowed = {(ROOT/'results/campaign-v3').resolve(): ('Unknown dynamics · v3 wave 1','campaign-v3'),
               (ROOT/'results/proposal-full-native-01').resolve(): ('Original proposal · Stage 1','proposal-full-native-01'),
               (ROOT/'results/campaign-v4-run1').resolve(): ('Unknown dynamics · RUN1','campaign-v4-run1')}
    if campaign not in allowed or not (campaign/'programs.sqlite').is_file():
        raise ValueError('This launcher serves an existing allowlisted campaign database only')
    CampaignUI.campaign_label, CampaignUI.campaign_task = allowed[campaign]
    assets = Path(native.__file__).resolve().parent
    handler = partial(CampaignUI, search_root=str(campaign), directory=str(assets))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Native Shinka UI: http://localhost:{args.port}/viz_tree.html?db_path=programs.sqlite', flush=True)
        if CampaignUI.campaign_task == 'campaign-v3':
            print(f'Assessment counts: http://localhost:{args.port}/assessment.html', flush=True)
        print('Loopback only; live reads from the existing campaign. Auto-refresh is native.', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
