"""Publish compact v3 development evidence and Chromatic Field figures."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
from visual_theme import apply_theme, save_figure, COBALT, MAGENTA, ORANGE, SECONDARY
import matplotlib.pyplot as plt
import numpy as np


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def objective_figure(out):
    data = json.loads((out / 'objective-check.json').read_text())
    names = list(data['metrics'])
    labels = ['Memory', 'Search seed', 'v2 generation 14', 'v2 frozen', 'v2 fixed risk']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={'wspace': .55})
    y = np.arange(len(names))
    axes[0].barh(y, [data['metrics'][n]['absolute_task'] for n in names], color=COBALT)
    axes[0].set(yticks=y, yticklabels=labels, xlabel='Mean absolute task score', xlim=(0, 1.04), title='A  Task determines most variation')
    axes[0].invert_yaxis()
    for i, n in enumerate(names):
        axes[0].text(data['metrics'][n]['absolute_task']-.015, i, f"{data['metrics'][n]['absolute_task']:.3f}", ha='right',va='center',color='white',fontsize=10)
    axes[1].scatter([data['metrics'][n]['absolute_task'] for n in names], [data['metrics'][n]['brier_near'] for n in names], c=[SECONDARY,ORANGE,COBALT,MAGENTA,SECONDARY], s=65)
    for i,n in enumerate(names):
        axes[1].annotate(str(i+1), (data['metrics'][n]['absolute_task'], data['metrics'][n]['brier_near']), xytext=(5,5),textcoords='offset points')
    axes[1].set(xlabel='Absolute task score',ylabel='On-policy near-cell Brier loss',title='B  Forecast quality is separate')
    fig.suptitle('Development objective check · 8 cases × 3 regimes', fontsize=16,fontweight='bold')
    fig.text(.12, -.04, 'Numbers follow bar order. Historical weighted ranges: task 0.0792; forecast 0.000676.\nThese reused development comparisons are not matched prediction-learning evidence.', fontsize=10,color=SECONDARY)
    save_figure(fig,out/'figures/objective-check')


def search_report(campaign, out, target):
    database = campaign / 'programs.sqlite'
    if not database.exists():
        return
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('select * from programs order by generation,timestamp,id')]
    if not rows:
        return
    sources = out / 'programs'
    sources.mkdir(exist_ok=True)
    lineage, metrics, episodes = [], [], []
    for row in rows:
        meta = json.loads(row['metadata'])
        entry = {k: row[k] for k in ('id','generation','parent_id','island_idx','correct','combined_score')}
        entry.update(source_sha256=hashlib.sha256(row['code'].encode()).hexdigest(),
                     **{k:json.loads(row[k] or '[]') for k in ('archive_inspiration_ids','top_k_inspiration_ids','migration_history')},
                     metadata=meta, public=json.loads(row['public_metrics']))
        history=entry['migration_history']
        entry['birth_island']=(min(history,key=lambda x:(x['generation'],x['timestamp']))['from']
                               if history else entry['island_idx'])
        lineage.append(entry)
    unique = {r['generation']:r for r in reversed(rows)}
    for g,row in sorted(unique.items()):
        (sources / f'gen_{g}.py').write_text(row['code'])
        meta = json.loads(row['metadata'])
        directory = Path(meta.get('recovery_results_dir', campaign / f'gen_{g}/results'))
        path = directory / 'episodes.json'
        saved = json.loads(path.read_text()) if path.exists() else []
        metric = {'generation':g, 'correct':bool(row['correct']), 'combined_score':row['combined_score'],
                  'episode_count':len(saved), **json.loads(row['public_metrics'])}
        metrics.append(metric)
        for index,episode in enumerate(saved):
            clean = {k:v for k,v in episode.items() if k not in ('seed','trace')}
            clean.update(generation=g, case=index//3)
            episodes.append(clean)
    # Save every generated proposal, including ones without a persisted DB row.
    proposals = out/'proposals'
    proposals.mkdir(exist_ok=True)
    for source in campaign.glob('gen_*/main.py'):
        shutil.copy2(source, proposals/(source.parent.name+'.py'))
    attempt_index=[]
    for directory in sorted(campaign.glob('gen_*/attempts/**/patch_*')):
        if not directory.is_dir(): continue
        relative=directory.relative_to(campaign)
        entry={'path':str(relative),'files':{}}
        for name in ('llm_response.txt','patch.txt','metadata.json'):
            source=directory/name
            if not source.exists(): continue
            destination=out/'attempts'/relative/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
            entry['files'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
        attempt_index.append(entry)
    write_json(out/'attempts.json',attempt_index)
    write_json(out/'lineage.json',lineage)
    write_json(out/'generation-metrics.json',metrics)
    with gzip.GzipFile(filename=str(out/'development-episodes.jsonl.gz'),mode='wb',mtime=0) as zipped:
        for row in episodes:
            zipped.write((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode())
    fields = ('generation','correct','combined_score','episode_count','escape','task','model_score','brier_near','brier_audit','invalid','seconds','candidate_cpu_seconds')
    with (out/'search.csv').open('w') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader(); writer.writerows(metrics)
    summary = {'target_slots':target,'persisted_slots':len(unique),
               'complete':sorted(unique)==list(range(target)), 'seed_slots':int(0 in unique),
               'seed_island_rows':sum(r['generation']==0 for r in rows),
               'valid_descendants':sum(m['generation']>0 and m['correct'] for m in metrics),
               'failed_slots':[m['generation'] for m in metrics if not m['correct']],
               'saved_evaluation_episodes':len(episodes),
               'candidate_cpu_seconds':sum(r.get('candidate_cpu_seconds',0) for r in episodes),
               'evaluator_cpu_seconds':sum(r.get('evaluator_cpu_seconds',0) for r in episodes),
               'episode_host_seconds':sum(r.get('seconds',0) for r in episodes),
               'scope':'One search; reused development results, no independent discovery replication.'}
    write_json(out/'search-summary.json',summary)
    for name in ('campaign-manifest.json','dreamer-resolved.json'):
        if (campaign/name).exists(): shutil.copy2(campaign/name,out/name)
    if (campaign/'meta').exists(): shutil.copytree(campaign/'meta',out/'recommendations',dirs_exist_ok=True)
    valid=[m for m in metrics if m['correct']]
    if not valid: return
    fig, axes=plt.subplots(2,2,figsize=(10,8),gridspec_kw={'hspace':.52,'wspace':.38})
    xs=[m['generation'] for m in valid]
    scores=[m['combined_score'] for m in valid]
    axes[0,0].scatter(xs,scores,color=COBALT,s=25,label='Valid candidate')
    axes[0,0].step(xs,np.maximum.accumulate(scores),where='post',color=ORANGE,label='Best so far')
    axes[0,0].set(title='A  All native slots',ylabel='Absolute task fitness',xlabel='Slot (seed = 0)')
    failed=[m for m in metrics if not m['correct']]
    if failed:
        axes[0,0].scatter([m['generation'] for m in failed],
                          [m.get('combined_score') or 0 for m in failed],
                          marker='x',c=MAGENTA,s=45,label='Failed slot (recorded score)')
    axes[0,0].legend(fontsize=9)
    axes[0,1].scatter([m.get('task',0) for m in valid],[m.get('brier_near',np.nan) for m in valid],color=COBALT,s=27)
    axes[0,1].set(title='B  Task and prediction',xlabel='Mean task score',ylabel='On-policy near Brier loss')
    by_id={r['id']:r for r in lineage}
    for r in lineage:
        p=by_id.get(r['parent_id'])
        if p:
            axes[1,0].plot([p['generation'],r['generation']],[p['birth_island'],r['birth_island']],color=SECONDARY,alpha=.3,lw=.7)
        axes[1,0].scatter(r['generation'],r['birth_island'],c=COBALT if r['correct'] else MAGENTA,marker='o' if r['correct'] else 'x',s=18)
    axes[1,0].set(title='C  Recorded parent ancestry',xlabel='Slot',ylabel='Island at source creation',yticks=range(4))
    axes[1,1].scatter(xs,[m.get('candidate_cpu_seconds',0) for m in valid],color=ORANGE,s=27)
    axes[1,1].set(title='D  Computation by candidate',xlabel='Slot',ylabel='Mean candidate CPU seconds / episode')
    from matplotlib.ticker import MaxNLocator
    for ax in (axes[0,0],axes[1,0],axes[1,1]):ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"Unknown dynamics · {len(unique)}/{target} slots · 72-episode development pool",fontsize=16,fontweight='bold')
    save_figure(fig,out/'figures/search')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',default='results/campaign-v3')
    parser.add_argument('--out',default='artifacts/campaign-v3')
    parser.add_argument('--slots',type=int,default=50)
    args=parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    apply_theme()
    objective_figure(out)
    search_report(Path(args.campaign).resolve(),out,args.slots)


if __name__=='__main__': main()
