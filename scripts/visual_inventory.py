"""Audit presentation coverage and historical evidence identities; runs no worlds."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
INVENTORY=ROOT/'docs/visual-migration-inventory.json'
VISUAL_EXTENSIONS={'.svg','.pdf','.png','.gif','.html'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


SOURCES={
    'scripts/chromatic_fields.py':('active','Shared plot roles, markers, terrain glyphs and compiled web tokens'),
    'scripts/visual_migration.py':('active','Canonical saved-data presentation wrapper for all historical/v3 publications'),
    'scripts/visual_inventory.py':('active','Read-only evidence and presentation coverage audit'),
    'scripts/web_presentation.py':('active','Fixed upstream/local HTML adapter; native data routes unchanged'),
    'scripts/v3_webui.py':('active','Read-only loopback server, exact v3/RUN1 campaign allowlist'),
    'scripts/v3_report.py':('active','Public v3 candidate/task/prediction/resource plots; current shared tokens'),
    'scripts/v3_ancestry.py':('active','Actual native parent graph; shared island/operator encodings'),
    'scripts/v3_figures.py':('active','Saved assessment/interim plots; shared condition/regime markers'),
    'scripts/v3_search_explorer.py':('active','Portable search HTML; shared CSS/JS tokens compiled inline'),
    'scripts/assets/v3-assessment.html':('active','Compiled shared CSS; public paused checkpoint/status only'),
    'scripts/run1_branch_figures.py':('active','Saved paired short-branch consequences and missingness'),
    'scripts/run1_figures.py':('pending','Saved RUN1 scientific/native execution exports; no fabricated pending evidence'),
    'scripts/visual_theme.py':('frozen-compatible','Original Chromatic Field palette/export primitives; preregistered bytes retained'),
    'scripts/v3_examples.py':('frozen-compatible','Already Chromatic Field; original final-assessment presentation source retained'),
    'scripts/v3_diagnostic20_report.py':('historical-adapted','Source-bound completed diagnostic; pure saved-summary render called by current wrapper'),
    'scripts/assessment_figures.py':('historical-adapted','Original functions retained; current wrapper remaps presentation before export'),
    'scripts/research_figure.py':('historical-adapted','Original draw function retained; wrapper uses saved analysis and remaps presentation'),
    'scripts/report.py':('historical-inactive','Scientific reproduction producer; current wrapper renders saved summary/learning curves'),
    'scripts/campaign_report.py':('historical-inactive','Scientific reproduction exporter; current wrapper renders saved campaign metrics'),
    'scripts/audit_candidate.py':('historical-inactive','Hash-bound intervention driver; current wrapper renders saved audit curves'),
    'scripts/replay.py':('historical-inactive','Original episode-producing replay entrypoint; current wrapper consumes saved traces only'),
    'scripts/evolve.py':('not-a-renderer','Sets matplotlib cache path only'),
    'scripts/evolve_v3.py':('not-a-renderer','Sets matplotlib cache path only'),
    'scripts/evolve_v4.py':('not-a-renderer','Sets matplotlib cache path only; native discovery controller'),
    'scripts/resume_v3.py':('not-a-renderer','Sets matplotlib cache path only'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-original',action='store_true')
    args=parser.parse_args();inventory=read(INVENTORY)
    manifest_path=ROOT/'artifacts/visual-migration/rendering-manifest.json';manifest=read(manifest_path)
    registered=dict(manifest['output_sha256']);additional=[]
    branch=ROOT/'artifacts/run1/branches/figures/figure-manifest.json'
    if branch.exists():
        b=read(branch)
        assert b['theme_sha256']==sha(ROOT/'scripts/chromatic_fields.py')
        files={str(p.relative_to(ROOT)):sha(p) for p in branch.parent.iterdir() if p.suffix in VISUAL_EXTENSIONS}
        registered.update(files);additional.append({'manifest':str(branch.relative_to(ROOT)),'manifest_sha256':sha(branch),'outputs':files})
    run1=ROOT/'artifacts/campaign-v4/run1/figures/figure-manifest.json'
    if run1.exists():
        r=read(run1);files={str(p.relative_to(ROOT)):sha(p) for p in run1.parent.iterdir() if p.suffix in VISUAL_EXTENSIONS}
        registered.update(files);additional.append({'manifest':str(run1.relative_to(ROOT)),'manifest_sha256':sha(run1),'outputs':files})
    public=[p for p in (ROOT/'artifacts').rglob('*') if p.is_file() and p.suffix in VISUAL_EXTENSIONS]
    public+=list((ROOT/'scripts/assets').glob('*.html'))
    uncovered=sorted(str(p.relative_to(ROOT)) for p in public if str(p.relative_to(ROOT)) not in registered)
    changed=[p for p,h in registered.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    producers={}
    for name,(status,purpose) in SOURCES.items():
        path=ROOT/name
        if path.exists():
            if status=='pending' and run1.exists():status='active'
            producers[name]={'status':status,'purpose':purpose,'sha256':sha(path)}
        elif status=='pending':producers[name]={'status':'pending','purpose':purpose}
    discovered=[]
    for p in (ROOT/'scripts').rglob('*'):
        if p.suffix not in ('.py','.html'):continue
        source=p.read_text()
        if any(t in source for t in ('matplotlib','<style>','visual_theme','chromatic_fields')):
            name=str(p.relative_to(ROOT))
            if name not in producers and name not in ('scripts/visual_inventory.py',):discovered.append(name)
    frozen={p:sha(ROOT/p) for p in inventory['frozen_presentation_inputs']}
    assert all(h==inventory['original_sources'][p] for p,h in frozen.items())
    archival=[p for p in (ROOT/'results').rglob('*') if p.is_file() and p.suffix in VISUAL_EXTENSIONS]
    archive_record={str(p.relative_to(ROOT)):sha(p) for p in archival}
    # Include actual browser proof hashes without publishing native/private UI data.
    proof_dir=ROOT/'results/visual-migration-browser'
    proofs={str(p.relative_to(ROOT)):sha(p) for p in proof_dir.glob('*.png')}
    inventory.update(status='existing-publications-migrated; RUN1 presentation pending' if not run1.exists() else
                     'all-current-published-visuals-covered' if not uncovered and not changed and not discovered else 'in-progress',
        original_publications_covered=len(set(inventory['original_publications']) & set(registered)),
        current_public_visual_files=len(public),registered_visual_files=len(registered),
        renderer_inventory=producers,unclassified_renderers=discovered,uncovered_publications=uncovered,
        output_hash_mismatches=changed,additional_rendering_manifests=additional,
        frozen_presentation_sha256=frozen,archival_runtime_visuals={'status':'inactive provenance / browser proof; not publication producers',
            'files':archive_record,'reason':'Original native screenshots and private replay/report caches retain their historical provenance; current public copies are migrated.'},
        browser_proof_files=proofs,rendering_manifest={'path':str(manifest_path.relative_to(ROOT)),'sha256':sha(manifest_path)},
        remaining=(['RUN1 search figures from new saved scientific/native exports'] if not run1.exists() else [])+uncovered+changed+discovered,
        new_environment_episodes=0,new_experiment_model_calls=0,
        interpretation='Coverage is of current publication paths. Retained historical scientific source functions are not rewritten; current themed wrappers replace their publication entrypoints.')
    if args.audit_original:
        base=inventory['original_git_commit']
        tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',base],cwd=ROOT,text=True).splitlines()
        paths=[p for p in tracked if p.startswith('artifacts/') and Path(p).suffix not in VISUAL_EXTENSIONS]
        digests={};changes=[]
        for name in paths:
            original=subprocess.check_output(['git','show',base+':'+name],cwd=ROOT)
            digest=hashlib.sha256(original).hexdigest();digests[name]=digest
            if not (ROOT/name).exists() or sha(ROOT/name)!=digest:changes.append(name)
        assert not changes,changes
        inventory['historical_nonvisual_integrity']={'original_git_commit':base,'files_checked':len(paths),'changed_files':changes,
            'path_sha256_map_sha256':hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()}
    INVENTORY.write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({k:inventory[k] for k in ('status','original_publications_covered','current_public_visual_files','unclassified_renderers','uncovered_publications','output_hash_mismatches','remaining')}))


if __name__=='__main__':main()
