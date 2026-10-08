"""Serve the installed native Shinka UI for this campaign on loopback only.

Upstream assets and handlers stay unchanged. This adapter confines HTTP reads to
the selected campaign database and native UI assets; it is not a search controller.
"""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinka.webui import visualization as native

ASSETS = {'/', '/index.html', '/viz_tree.html', '/compare.html', '/favicon.png', '/sakana.jpg'}
READ_ROUTES = {'/get_programs', '/get_programs_summary', '/get_program_count',
               '/get_program_details', '/get_meta_files', '/get_meta_content',
               '/get_database_stats', '/get_system_prompts', '/get_plots'}


class CampaignUI(native.DatabaseRequestHandler):
    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == '/list_databases':
            return self.send_json_response([{'path':'programs.sqlite',
                'actual_path':'programs.sqlite', 'name':'Unknown dynamics · v3 wave 1',
                'task':'campaign-v3'}])
        if parsed.path in READ_ROUTES:
            if query.get('db_path') != ['programs.sqlite']:
                return self.send_error(403, 'Only the active v3 campaign is available')
            for key in ('generation', 'processed_count'):
                if key in query and (len(query[key]) != 1 or not re.fullmatch(r'\d+', query[key][0])):
                    return self.send_error(400, 'Integer generation required')
        elif parsed.path not in ASSETS:
            return self.send_error(404, 'No such campaign UI resource')
        return super().do_GET()

    def do_HEAD(self):
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
    if campaign != (ROOT/'results/campaign-v3').resolve() or not (campaign/'programs.sqlite').is_file():
        raise ValueError('This launcher serves the existing v3 campaign only')
    assets = Path(native.__file__).resolve().parent
    handler = partial(CampaignUI, search_root=str(campaign), directory=str(assets))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Native Shinka UI: http://localhost:{args.port}/viz_tree.html?db_path=programs.sqlite', flush=True)
        print('Loopback only; live reads from the existing campaign. Auto-refresh is native.', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
