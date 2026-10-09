"""Render the repository's published figures from saved evidence only.

No evaluator, candidate, database, analysis resampling or model call is imported.
Historical numerical JSON/CSV and frozen presentation sources remain unchanged.
Usage: .venv/bin/python scripts/visual_migration.py [--group legacy|v3|replays|all]
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
from chromatic_fields import (BACKGROUND, TEXT, SECONDARY, COBALT, MAGENTA, ORANGE,
    RULE, MUTED, WALL, ROLES, OUTCOMES, apply_theme, field_cmap, save_figure,
    restyle_legacy_figure, terrain_cmap, terrain_symbols, PRESENTATION_VERSION)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np

INPUTS = {}
OUTPUTS = set()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def read(path):
    path = ROOT/path if not Path(path).is_absolute() else Path(path)
    INPUTS[relative(path)] = sha(path)
    return json.loads(path.read_text())


def save(fig, path):
    save_figure(fig, path)
    OUTPUTS.update(relative(path.with_suffix('.'+ext)) for ext in ('svg','pdf','png'))


def role(variant):
    if variant in ('frozen', 'frozen_no_planning', 'fitted_frozen', 'selected_frozen'):
        return 'frozen'
    if variant == 'memory': return 'memory'
    if variant == 'no_planning': return 'fixed_risk'
    if variant in ('original_predictive', 'reactive'): return 'historical'
    return 'online'


NAMES = {'reactive':'Reactive', 'memory':'Original memory',
         'original_predictive':'Original predictive seed', 'predictive':'Predictive program',
         'frozen':'Frozen learning', 'no_planning':'Fixed-risk planning',
         'frozen_no_planning':'Frozen + fixed-risk'}


def summary_figures(directory):
    summary = read(directory/'summary.json')
    agents, protocol = summary['agents'], summary['protocol']
    order = [name for name in NAMES if name in agents]
    prediction = [name for name in order if name != 'reactive']
    fig, axes = plt.subplots(1,2,figsize=(11,4.9),layout='constrained')
    for ax, names, metric in ((axes[0],order,'escape'),(axes[1],prediction,'brier_threat')):
        for y, name in enumerate(names):
            style=ROLES[role(name)]; value=agents[name][metric]
            ax.plot(value,y,marker=style['marker'],color=style['color'],markersize=7,
                    linestyle='none',zorder=3)
            ax.hlines(y,0,value,color=RULE,lw=1.8,zorder=1)
            if metric=='escape':
                lo,hi=agents[name]['escape_ci95']
                ax.errorbar(value,y,xerr=[[max(0,value-lo)],[max(0,hi-value)]],
                            fmt='none',color=style['color'],capsize=3,lw=1.3)
        ax.set(yticks=range(len(names)),yticklabels=[NAMES[n] for n in names])
        ax.invert_yaxis();ax.grid(axis='x',alpha=.6);ax.set_axisbelow(True)
    axes[0].set(xlim=(0,1.035),xlabel='Escape fraction · 95% Wilson interval',title='A  Task outcome')
    axes[1].set(xlabel='Threat-cell Brier loss ↓',title='B  On-policy prediction')
    n=protocol.get('episodes',next(iter(agents.values()))['episodes'])
    prefix='Historical '+str(protocol.get('split','development')).replace('_',' ')
    fig.suptitle(f'{prefix} · {n:,} paired mazes',fontsize=16,fontweight='bold')
    fig.text(.5,-.055,'All outcomes retain invalid executions. On-policy prediction uses different encountered experience; it is not a learning contrast.',
             ha='center',fontsize=9,color=SECONDARY)
    save(fig,directory/'outcomes')
    if (directory/'learning-curve.json').exists():
        learning_figure(directory/'learning-curve.json',directory/'learning',
                        title=f'Historical seed · {protocol.get("split","development")} · {n:,} paired mazes')


def learning_figure(source, destination, *, title):
    curve=read(source)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),gridspec_kw={'width_ratios':[1.6,1]},layout='constrained')
    for variant,name,key in (('no_planning','Online updates','online'),
                             ('frozen_no_planning','Frozen parameters','frozen')):
        rows=[r for r in curve if r.get('agent',r.get('variant'))==variant]
        style=ROLES[key]
        x=[r['start_step']+12.5 for r in rows]
        count=[r.get('target_count',r.get('targets')) for r in rows]
        y=[r['loss_sum']/c for r,c in zip(rows,count)]
        axes[0].plot(x,y,color=style['color'],marker=style['marker'],
                     linestyle=style['linestyle'],label=name,markersize=5)
        episodes=[r.get('episodes_present',r.get('episodes')) for r in rows]
        axes[1].plot(x,episodes,color=style['color'],marker=style['marker'],
                     linestyle=style['linestyle'],label=name,markersize=5)
    axes[0].set(ylabel='Pooled near-cell Brier loss ↓',title='A  Identical recorded experience')
    axes[0].legend(fontsize=9)
    axes[1].set(ylabel='Episodes contributing targets',title='B  Survivor-conditioned counts',ylim=(0,None))
    for ax in axes:
        ax.set(xlabel='Episode step (25-step bins)');ax.grid(axis='y',alpha=.6)
    fig.suptitle(title,fontsize=16,fontweight='bold')
    fig.text(.5,-.055,'Saved loss sums ÷ saved target counts. These historical bin summaries provide no uncertainty intervals; later bins select survivors.',
             ha='center',fontsize=9,color=SECONDARY)
    save(fig,destination)


def campaign_figure(directory):
    metrics=read(directory/'generation-metrics.json'); summary=read(directory/'summary.json')
    valid=[r for r in metrics if r['correct']];x=[r['generation'] for r in valid]
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    scores=[r['combined_score'] for r in valid]
    axes[0].scatter(x,scores,color=SECONDARY,s=24,label='Every valid candidate')
    axes[0].step(x,np.maximum.accumulate(scores),where='post',color=COBALT,label='Best so far')
    axes[0].set(ylabel='Selection score',title='A  Historical joint objective')
    axes[1].scatter(x,[r['escape'] for r in valid],color=COBALT,s=24)
    axes[1].axhline(summary['agents']['memory']['escape'],color=SECONDARY,ls='--',label='Original memory')
    axes[1].set(ylabel='Escape fraction',ylim=(.65,1.02),title='B  Task outcome')
    for key,label,style in (('brier_near','Near cells',ROLES['online']),('brier_threat','Threat-conditioned',ROLES['frozen'])):
        axes[2].plot(x,[r[key] for r in valid],linestyle='none',marker=style['marker'],
                     color=style['color'],markersize=4,label=label)
    axes[2].set(ylabel='On-policy Brier loss ↓',title='C  Prediction diagnostics')
    for ax in axes:
        ax.set(xlabel='Candidate slot (seed = 0)',xlim=(-1,len(metrics)))
        for failed in summary['failed_slots']:
            ax.axvline(failed,color=RULE,lw=1,zorder=0)
        ax.grid(axis='y',alpha=.5);ax.legend(fontsize=8)
    fig.suptitle(f'Historical v2 · {len(metrics)} native slots · 64 reused development mazes',fontsize=16,fontweight='bold')
    fig.text(.5,-.035,'All candidate slots remain represented; pale vertical rules mark failures. Different on-policy experience does not identify prediction learning.',
             ha='center',fontsize=9,color=SECONDARY)
    save(fig,directory/'evolution')


def assessment_figures(directory):
    # Importing this immutable module defines plot functions only. Never call any
    # historical evaluator or analysis main entrypoint.
    import assessment_figures as legacy
    summary=read(directory/'analysis.json')
    read(ROOT/'artifacts/campaign-v2/completed-50/generation-metrics.json')
    original=legacy.save
    def themed_save(fig,out,name):
        restyle_legacy_figure(fig)
        for ax in fig.axes:
            for container in ax.containers:
                key = container.get_label().lower()
                if key in OUTCOMES and hasattr(container, 'patches'):
                    for patch in container.patches: patch.set_facecolor(OUTCOMES[key])
            legend=ax.get_legend()
            if legend:
                for handle,label in zip(legend.legend_handles,legend.get_texts()):
                    key=label.get_text().lower()
                    if key in OUTCOMES and hasattr(handle,'set_facecolor'): handle.set_facecolor(OUTCOMES[key])
            ax.set_xlabel(ax.get_xlabel().replace('fresh cases','originally held-out cases'))
            for line in ax.lines:
                if line.get_label()=='Frozen weights':
                    line.set_marker('s');line.set_linestyle('--')
        for text in fig.texts:
            text.set_text(text.get_text().replace('red dots','orange dots').replace('hollow red rings','hollow orange rings').replace('Fresh assessment','Completed v2 assessment').replace('on fresh mazes','on the completed v2 assessment'))
        save(fig,out/name)
    legacy.save=themed_save
    legacy.BLUE,legacy.ORANGE,legacy.TEAL,legacy.GRAY,legacy.RED=COBALT,MAGENTA,COBALT,SECONDARY,ORANGE
    try:
        apply_theme()
        out=directory/'figures';out.mkdir(exist_ok=True)
        legacy.evolution(out);legacy.outcomes(summary,out);legacy.learning(summary,out)
        if (directory/'behavior-examples.json').exists():
            legacy.behavior(read(directory/'behavior-examples.json'),out)
    finally: legacy.save=original


def research_figure(directory):
    import research_figure as legacy
    metrics=read(ROOT/'artifacts/campaign-v2/completed-50/generation-metrics.json')
    result=read(directory/'analysis.json')
    original=Figure.savefig
    def themed_savefig(fig,path,*args,**kwargs):
        restyle_legacy_figure(fig)
        kwargs.update(facecolor=BACKGROUND,edgecolor=BACKGROUND,transparent=False)
        original(fig,path,*args,**kwargs);OUTPUTS.add(relative(path))
        if Path(path).suffix=='.svg':
            pdf=Path(path).with_suffix('.pdf')
            original(fig,pdf,dpi=180,bbox_inches='tight',facecolor=BACKGROUND,
                     metadata={'CreationDate':None,'ModDate':None})
            OUTPUTS.add(relative(pdf))
    Figure.savefig=themed_savefig
    try: legacy.draw(metrics,result)
    finally: Figure.savefig=original


def replay_figures(directory):
    from PIL import Image
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    data=read(directory/'replay.json'); trace=data['frames']; episode=data['episode']
    if not trace: raise ValueError(f'Empty saved replay: {directory}')
    selected=next((i for i,f in enumerate(trace) if f['world']['step']>=25 and any(
        max(abs(x-f['world']['agent'][0]),abs(y-f['world']['agent'][1]))<=2
        for x,y in f['world']['enemies'])),min(25,len(trace)-1))
    apply_theme();cmap,norm=terrain_cmap()
    fig,axes=plt.subplots(1,3,figsize=(11.4,4.7))
    fig.subplots_adjust(left=.025,right=.91,top=.82,bottom=.25,wspace=.10)
    cax=fig.add_axes([.928,.28,.013,.50])
    bar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=field_cmap()),cax=cax)
    bar.set_label('P(enemy occupies cell at t+1)',fontsize=9);bar.ax.tick_params(labelsize=8)
    title=fig.suptitle('',fontsize=16,fontweight='bold')
    caption=fig.text(.5,.13,'',ha='center',fontsize=9,color=SECONDARY)
    handles=[Line2D([],[],marker=m,linestyle='none',color=c,markersize=6,label=l)
             for m,c,l in [('^',COBALT,'Agent / chosen move'),('x',ORANGE,'Current enemy'),
                           ('D',ORANGE,'Key'),('s',MAGENTA,'Door'),('*',COBALT,'Exit')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=5,fontsize=9)
    images=[]
    for i,frame in enumerate(trace):
        world,model=frame['world'],frame['model']; ox,oy=world['origin'];ax,ay=world['agent']
        belief=np.full((15,15),-2);risk=np.full((15,15),model.get('default_enemy',.5),dtype=float)
        for x,y,cell,seen in model.get('terrain',[]):
            if 0<=x+ox<15 and 0<=y+oy<15:belief[y+oy,x+ox]=cell
        for x,y,p in model.get('enemy',[]):
            if 0<=x+ox<15 and 0<=y+oy<15:risk[y+oy,x+ox]=p
        for j,axis in enumerate(axes):
            axis.clear()
            if j==2:
                axis.imshow(risk,cmap=field_cmap(),vmin=0,vmax=1,interpolation='nearest')
                for x,y in frame['next_enemies']:
                    axis.scatter(x,y,s=60,facecolors='none',edgecolors=TEXT,linewidth=1.2,zorder=4)
            else:
                grid=world['grid'] if j==0 else belief
                axis.imshow(grid,cmap=cmap,norm=norm,interpolation='nearest');terrain_symbols(axis,grid)
                for x,y in world['enemies']:
                    if j==0 or max(abs(x-ax),abs(y-ay))<=2:
                        axis.scatter(x,y,s=42,marker='x',c=ORANGE,linewidth=1.8,zorder=4)
            axis.scatter(ax,ay,s=55,marker='^',c=COBALT,edgecolors=BACKGROUND,linewidth=.7,zorder=5)
            axis.add_patch(Rectangle((ax-2.5,ay-2.5),5,5,fill=False,edgecolor=COBALT,lw=1.25))
            dx,dy=frame['action']['move']
            if dx or dy:axis.arrow(ax,ay,dx*.85,dy*.85,head_width=.32,width=.065,color=TEXT,zorder=6)
            else:axis.scatter(ax,ay,s=105,facecolors='none',edgecolors=TEXT,linewidth=1,zorder=6)
            axis.set(title=('A  Retrospective true world','B  Remembered terrain','C  Before-outcome forecast')[j],
                     xticks=[],yticks=[],xlim=(-.5,14.5),ylim=(14.5,-.5))
        label='v2 generation 14' if 'mechanism-gen14' in str(directory) else 'Historical predictive seed'
        title.set_text(f'{label} · saved step {world["step"]} → {world["step"]+1}')
        updates=model.get('learning',{}).get('parameter_updates',model.get('learning',{}).get('updates',0))
        caption.set_text(f'Attempted move {frame["action"]["move"]} · keys {world["keys"]}/2 · recorded updates {updates}\n'
            'Outlined square: 5×5 view · hollow black rings: next enemies, retrospective only · remembered unknown tiles: pale gray\n'
            'Absolute 15×15 world coordinates; exported relative forecasts translated by the saved origin. Field scale is probability 0–1.')
        fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        if i==selected:
            # Save without closing: the same figure supplies all original frames.
            for ext in ('svg','pdf','png'):
                path=directory/f'frame.{ext}'
                metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {}
                fig.savefig(path,dpi=180,bbox_inches='tight',facecolor=BACKGROUND,metadata=metadata)
                if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
                OUTPUTS.add(relative(path))
    images[0].save(directory/'replay.gif',save_all=True,append_images=images[1:],duration=180,loop=0,disposal=2)
    OUTPUTS.add(relative(directory/'replay.gif'));plt.close(fig)
    return {'trace':relative(directory/'replay.json'),'trace_sha256':sha(directory/'replay.json'),
            'frames':len(trace),'selected_frame_index':selected,'selection_rule':'unchanged historical first step >=25 with visible enemy; otherwise min(25,last)',
            'source_episode_outcome':episode.get('reason'),'new_environment_episodes':0}


def v3_figures():
    import v3_report
    import v3_ancestry
    import v3_figures as assessment
    import v3_diagnostic20_report
    directory=ROOT/'artifacts/campaign-v3'
    for name in ('objective-check.json','lineage.json','generation-metrics.json','selection/selection.json'):
        read(directory/name)
    apply_theme();v3_report.objective_figure(directory)
    v3_report.search_figure(read(directory/'lineage.json'),read(directory/'generation-metrics.json'),directory,50)
    v3_ancestry.render(directory)
    interim=directory/'assessment-interim140';data=read(interim/'analysis.json');out=interim/'figures'
    assessment.outcomes(data,out);assessment.forecast_cost(data,out);assessment.exposure_figure(data,out)
    fitted=read(interim/'matched-analysis.json');assessment.matched_figure(fitted,out)
    selected={**data.get('selected_matched',{}),'study':'Selected program · uniform-law planning','interim':data.get('interim')}
    assessment.matched_figure(selected,out,'selected-matched-learning')
    diagnostic=directory/'development-diagnostic20'
    v3_diagnostic20_report.render(read(diagnostic/'summary.json'),diagnostic)
    # Existing development-only smoke rendering is regenerated from its saved
    # matched-analysis, and retains its small-sample qualification.
    smoke=ROOT/'artifacts/visual-migration/sources/v3-matched-smoke-analysis.json'
    if smoke.exists():
        assessment.matched_figure(read(smoke),directory/'figures','matched-smoke-development')
    for folder in (directory/'figures',out,diagnostic):
        OUTPUTS.update(relative(p) for p in folder.iterdir() if p.suffix in ('.svg','.pdf','.png'))


def web_assets():
    import v3_search_explorer
    from web_presentation import local_html
    directory=ROOT/'artifacts/campaign-v3'
    for name in v3_search_explorer.INPUTS:read(directory/name)
    v3_search_explorer.render(directory)
    OUTPUTS.add(relative(directory/'search-explorer.html'))
    asset=ROOT/'scripts/assets/v3-assessment.html'
    asset.write_text(local_html(asset.read_text()))
    OUTPUTS.add(relative(asset))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=('legacy','replays','v3','web','all'),default='all')
    args=parser.parse_args();apply_theme();replays=[]
    inventory=read(ROOT/'docs/visual-migration-inventory.json')
    frozen_before={p:sha(ROOT/p) for p in inventory['frozen_presentation_inputs']}
    if args.group in ('legacy','all'):
        for name in ('assessment','development','validation','campaign-v2/controls','campaign-v2/mechanism-gen14'):
            summary_figures(ROOT/'artifacts'/name)
        mechanism=ROOT/'artifacts/campaign-v2/mechanism-gen14'
        learning_figure(mechanism/'matched-learning.json',mechanism/'matched-learning',title='Historical generation 14 · 64 reused development mazes')
        campaign_figure(ROOT/'artifacts/campaign-v2/completed-50')
        assessment_figures(ROOT/'artifacts/campaign-v2/assessment-1024')
        research_figure(ROOT/'artifacts/campaign-v2/research-review')
    if args.group in ('replays','all'):
        for name in ('replay-success','replay-failure','campaign-v2/mechanism-gen14/replay'):
            replays.append(replay_figures(ROOT/'artifacts'/name))
    if args.group in ('v3','all'):v3_figures()
    if args.group in ('web','all'):web_assets()
    for path,digest in INPUTS.items():
        if sha(ROOT/path)!=digest:raise RuntimeError(f'Input modified during rendering: {path}')
    for path,digest in frozen_before.items():
        if sha(ROOT/path)!=digest:raise RuntimeError(f'Frozen source changed: {path}')
    destination=ROOT/'artifacts/visual-migration';destination.mkdir(exist_ok=True)
    manifest_path=destination/'rendering-manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'presentation_version':PRESENTATION_VERSION,'original_git_commit':inventory['original_git_commit'],
        'original_sources':inventory['original_sources'],'original_publications':inventory['original_publications'],
        'input_sha256':{},'output_sha256':{},'replays':[]}
    manifest['input_sha256'].update({p:h for p,h in INPUTS.items() if p!='docs/visual-migration-inventory.json'})
    manifest['output_sha256'].update({p:sha(ROOT/p) for p in sorted(OUTPUTS)})
    if replays:manifest['replays']=replays
    manifest.update(renderer_sha256=sha(__file__),shared_theme_sha256=sha(ROOT/'scripts/chromatic_fields.py'),
                    current_renderer_sha256={p:sha(ROOT/p) for p in ('scripts/visual_migration.py','scripts/chromatic_fields.py',
                        'scripts/web_presentation.py','scripts/v3_webui.py','scripts/v3_figures.py',
                        'scripts/v3_report.py','scripts/v3_ancestry.py','scripts/v3_search_explorer.py')},
                    frozen_sources_unchanged=frozen_before,new_environment_episodes=0,new_experiment_model_calls=0,
                    statistical_analysis_rerun=False,render_command='.venv/bin/python scripts/visual_migration.py --group all')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'group':args.group,'written_visual_files':len(OUTPUTS),'read_only_inputs':len(INPUTS),
                      'manifest':relative(manifest_path),'new_environment_episodes':0,'new_experiment_model_calls':0}))


if __name__=='__main__':main()
