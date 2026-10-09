"""Render RUN1 public scientific/native snapshots in Chromatic Field.

Read-only saved data: no runtime databases, private pools, candidates, worlds,
resampling or model services. Missing/unfinished evidence stays unavailable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

from chromatic_fields import (apply_theme, save_figure, BACKGROUND, TEXT, SECONDARY,
    COBALT, MAGENTA, ORANGE, RULE, MUTED, OUTCOMES, ISLAND_COLORS, OPERATOR_MARKERS)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INPUT_HASHES={}


def read(path, default=None):
    if not path.exists():return default
    encoded=path.read_bytes();INPUT_HASHES[str(path.relative_to(ROOT))]=hashlib.sha256(encoded).hexdigest()
    return json.loads(encoded)


def number(value):return value is not None and isinstance(value,(int,float)) and np.isfinite(value)
def rows(value):return value if isinstance(value,list) else value.get('rows',[]) if value else []
def label(value):return value.replace('_',' ')
def model_name(value):return value.split('@')[-1].split('?')[0]
def fmt(value, digits=3):return f'{value:.{digits}f}' if number(value) else 'Unavailable'


def footer(fig,text):
    fig.text(.08,.018,text,fontsize=9,color=SECONDARY,va='bottom',linespacing=1.4)


def save(fig,out,name,manifest):
    save_figure(fig,out/name)
    for ext in ('svg','pdf','png'):
        p=out/f'{name}.{ext}';manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()


def unavailable(ax,title,reason):
    ax.axis('off');ax.set_title(title,loc='left')
    ax.text(.03,.65,textwrap.fill(reason,49),transform=ax.transAxes,color=SECONDARY,fontsize=11,va='top')


def progress(candidates,native,out,outputs):
    fig,axes=plt.subplots(2,2,figsize=(11.4,8.5),gridspec_kw={'hspace':.52,'wspace':.35})
    fig.subplots_adjust(top=.86,bottom=.19,left=.09,right=.97)
    if not candidates:
        for ax,title in zip(axes.flat,('A  Selection score','B  Task and prediction','C  Outcomes','D  Prediction diagnostics')):
            unavailable(ax,title,'Saved candidate metrics pending; no performance values are inferred from native metadata.')
    else:
        ax=axes[0,0];eligible=[]
        for row in candidates:
            g=row['generation'];score=row.get('native_score');correct=row.get('native_correct')
            if number(score):
                ax.scatter(g,score,c=COBALT if correct else MAGENTA,marker='o' if correct else 'X',s=44,zorder=3)
                if correct:eligible.append((g,score))
                elif row.get('infrastructure_failure'):
                    ax.annotate('infrastructure', (g,score), xytext=(0,10), textcoords='offset points', ha='center', fontsize=7, color=SECONDARY)
            else:
                ax.scatter(g,.035,transform=ax.get_xaxis_transform(),facecolor='none',edgecolor=SECONDARY,marker='o',s=45)
                ax.annotate('pending',(g,.035),xycoords=('data','axes fraction'),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8,color=SECONDARY)
        if eligible:
            gx,gy=zip(*eligible);ax.step(gx,np.maximum.accumulate(gy),where='post',c=ORANGE,lw=1.5,label='Best eligible score')
        handles=[Line2D([],[],marker='o',ls='none',c=COBALT,label='Native valid'),Line2D([],[],marker='X',ls='none',c=MAGENTA,label='Native invalid'),Line2D([],[],c=ORANGE,label='Best eligible')]
        ax.legend(handles=handles,fontsize=8,loc='lower right')
        ax.set(title='A  Every native candidate slot',xlabel='Slot (seed = 0)',ylabel='Native selection score')
        ax=axes[0,1]
        for row in candidates:
            if number(row.get('task')) and number(row.get('brier_near')):
                color=COBALT if row.get('native_correct') else MAGENTA
                ax.scatter(row['task'],row['brier_near'],c=color,marker='o' if row.get('native_correct') else 'X',s=40)
                ax.annotate(str(row['generation']),(row['task'],row['brier_near']),xytext=(5,5),textcoords='offset points',fontsize=8)
        ax.set(title='B  Task ≠ prediction quality',xlabel='Mean absolute episode task',ylabel='On-policy near-cell Brier loss ↓')
        ax=axes[1,0];generations=[r['generation'] for r in candidates];bottom=np.zeros(len(candidates))
        for event,color in OUTCOMES.items():
            value=np.array([r.get(event+'_count',0) for r in candidates])
            ax.bar(generations,value,bottom=bottom,color=color,width=.72,label=event.title());bottom+=value
        missing=np.array([r.get('missing_episodes',0) for r in candidates])
        if missing.any():ax.bar(generations,missing,bottom=bottom,facecolor=BACKGROUND,edgecolor=RULE,hatch='////',label='Not recorded',width=.72)
        ax.set(title='C  Outcome denominators stay explicit',xlabel='Slot',ylabel='Condition-episodes')
        ax.legend(ncol=3,fontsize=7.5,loc='upper center',bbox_to_anchor=(.5,.99))
        ax.set_ylim(0,max(r.get('expected_episodes',48) for r in candidates)*1.24)
        ax=axes[1,1]
        for key,name,color,marker in [('brier_near','Near cells',COBALT,'o'),('brier_audit','Audit cells',MAGENTA,'s'),('brier_destination','Destination',ORANGE,'^')]:
            complete=[r for r in candidates if r.get('missing_episodes')==0 and number(r.get(key))]
            if complete:ax.plot([r['generation'] for r in complete],[r[key] for r in complete],marker=marker,linestyle='none',c=color,ms=4,label=name)
        ax.set(title='D  Proper prediction diagnostics',xlabel='Slot',ylabel='Pooled on-policy Brier loss ↓');ax.legend(fontsize=8)
    for ax in (axes[0,0],axes[1,0],axes[1,1]):ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',alpha=.6);ax.set_axisbelow(True)
    count=native.get('counts',{});n=count.get('persisted_slots',0);cap=count.get('authorized_total_slots','?')
    fig.suptitle(f'RUN1 · {n}/{cap} native slots · reused development cases' + (f' · {len(candidates)} scientific rows' if len(candidates)!=n else ''),fontsize=16,fontweight='bold')
    footer(fig,'Eight layouts × six conditions per evaluated slot. Seed copies are administrative rows, not extra candidate evaluations.\n'
        'Invalid episodes retain zero task; native eligibility additionally requires every episode to validate.\n'
        'Hatched episodes were not recorded; they are not observed algorithm failures.\n'
        'Forecasts use different on-policy experience. No matched learning, frozen-control or transfer assessment is established by this figure.')
    save(fig,out,'search-progress',outputs)


def resources(candidates,curve,native,out,outputs):
    fig,axes=plt.subplots(2,2,figsize=(11.4,8.5),gridspec_kw={'hspace':.55,'wspace':.38})
    fig.subplots_adjust(top=.87,bottom=.18,left=.10,right=.97)
    ax=axes[0,0]
    available=[r for r in curve if number(r.get('bestsofar_task')) and number(r.get('uncached_input_plus_output_tokens'))]
    if available:
        ax.step([r['uncached_input_plus_output_tokens']/1000 for r in available],[r['bestsofar_task'] for r in available],where='post',c=COBALT,marker='o',ms=4)
        for r in available:ax.annotate(str(r['generation']),(r['uncached_input_plus_output_tokens']/1000,r['bestsofar_task']),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.set(title='A  Progress per reported token use',xlabel='Completed-call uncached input + output tokens (k)',ylabel='Best eligible task at evaluation completion')
    else:unavailable(ax,'A  Development progress per resource','Evaluation-aligned resource snapshots pending; no budget or reward is imputed.')
    ax=axes[0,1];gs=[r['generation'] for r in candidates]
    if candidates:
        candidate=[r.get('candidate_cpu_seconds_sum',0) for r in candidates];evaluation=[r.get('evaluator_cpu_seconds_sum',0) for r in candidates]
        ax.bar(gs,candidate,color=COBALT,label='Candidate CPU');ax.bar(gs,evaluation,bottom=candidate,color=ORANGE,label='Evaluator CPU')
        for r in candidates:
            missing=r.get('candidate_cpu_seconds_unavailable_recorded_episodes',0)
            if missing:ax.annotate(f'{missing} unmeasured',(r['generation'],r.get('candidate_cpu_seconds_sum',0)+r.get('evaluator_cpu_seconds_sum',0)),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        ax.set(title='B  Measured local episode computation',xlabel='Slot',ylabel='CPU seconds (sum, measured episodes only)');ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:unavailable(ax,'B  Measured local episode computation','No saved episode resource measurements.')
    ax=axes[1,0];roles=native.get('calls',{}).get('by_role',{})
    if roles:
        names=list(roles);uncached=[roles[n].get('uncached_input_plus_output_tokens',0)/1000 for n in names];cached=[roles[n].get('tokens',{}).get('cacheReadTokens',0)/1000 for n in names]
        ax.barh(names,uncached,color=COBALT,label='Uncached input + output');ax.barh(names,cached,left=uncached,color=MUTED,edgecolor=RULE,label='Cached input')
        ax.invert_yaxis();ax.set(title='C  Reported experiment-model usage by role',xlabel='Tokens (k)');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,1.0))
        ax.set_ylim(len(names)-.4,-1.6)
    else:unavailable(ax,'C  Experiment-model usage','No reported model-call usage. Pending calls have no assumed token count.')
    ax=axes[1,1];calls=native.get('calls',{}).get('all',{});budget=native.get('configured',{}).get('budget',{})
    specs=[('Call admissions','calls'),('Uncached + output\ntokens','uncached_input_plus_output_tokens'),('Reported remote\nseconds','remote_elapsed_seconds')]
    for i,(name,key) in enumerate(specs):
        value=calls.get(key,0);limit=budget.get(key)
        if number(limit) and limit>0:
            ax.barh(i,100*value/limit,color=(COBALT,MAGENTA,ORANGE)[i]);ax.text(min(99,100*value/limit+1),i,f'{value:,.0f} / {limit:,.0f}',va='center',fontsize=8)
    ax.set(yticks=range(3),yticklabels=[n for n,k in specs],xlim=(0,100),xlabel='Fraction of registered cap (%)',title='D  Shared bounded-execution budget');ax.invert_yaxis()
    fig.suptitle('RUN1 resources · completed usage and local CPU are different clocks',fontsize=16,fontweight='bold')
    footer(fig,'Readiness/novelty fixture calls consume the shared budget but no candidate slots; the readiness row remains separate from discovery.\n'
        'Pending calls have incomplete usage. Cached tokens are reported separately; reasoning tokens are already included in output.\n'
        'Remote per-call elapsed durations may overlap local work: do not add them to episode CPU or treat them as controller wall time.\n'
        'Native dollar values are API list-price estimates, not subscription charges. No model-price estimate is plotted as actual spending.')
    save(fig,out,'resources',outputs)


def machinery(native,prompts,out,outputs):
    fig=plt.figure(figsize=(12,9));grid=fig.add_gridspec(2,2,left=.08,right=.98,top=.86,bottom=.22,hspace=.73,wspace=.46)
    table_ax=fig.add_subplot(grid[:,0]);table_ax.axis('off');table_ax.set_title('A  Recorded mechanism use',loc='left')
    m=native.get('mechanisms',{});counts=native.get('counts',{});call_roles=native.get('calls',{}).get('by_role',{})
    status=[]
    def add(name,state,observed):status.append((name,state.replace('_',' '),observed))
    add('Native islands',m.get('islands',{}).get('status','pending'),f"{m.get('islands',{}).get('configured','?')} configured; {len(m.get('islands',{}).get('migrations',[]))} migration events")
    add('Parent inspirations',m.get('inspirations',{}).get('status','pending'),f"Archive contexts {m.get('inspirations',{}).get('contexts_archive',0)}; top-k {m.get('inspirations',{}).get('contexts_top_k',0)}")
    op=m.get('operators',{}).get('persisted_by_type',{});add('Mutation operators','exercised' if op else 'pending',', '.join(f'{k}: {v}' for k,v in op.items()) or 'No descendant persisted yet')
    add('Model bandit',m.get('model_bandit',{}).get('status','pending'),f"{len(m.get('model_bandit',{}).get('arm_models_queried',[]))} mutation arms queried; cost coefficient 0")
    novelty=m.get('novelty',{});add('Native novelty',novelty.get('status','pending'),f"{len(novelty.get('embedding_calls',[]))} embedding calls; {len(novelty.get('decisions',[]))} gate decisions; {counts.get('novelty_rejections',0)} rejected")
    meta=m.get('meta',{});add('Meta recommendations',meta.get('status','pending'),f"{sum(meta.get('calls_by_role',{}).values())} calls; {meta.get('sampling_contexts_with_recommendation',0)} contexts consumed advice")
    prompt=m.get('prompt_coevolution',{});add('Prompt coevolution',prompt.get('status','pending'),f"{prompt.get('prompt_rows',0)} stored prompts; {prompt.get('programs_with_prompt_credit',0)} program credits")
    add('Checkpoint / resume',m.get('resume',{}).get('status','pending'),f"{counts.get('held_proposals',0)} held proposals; {counts.get('patch_repairs',0)} patch repair attempts")
    for i,(name,state,observed) in enumerate(status):
        y=.97-i*.12;table_ax.text(0,y,name,weight='bold',fontsize=10,va='top')
        table_ax.text(0,y-.035,textwrap.fill(state,50),color=MAGENTA if state=='pending' else COBALT,fontsize=9,va='top')
        table_ax.text(0,y-.066,textwrap.fill(observed,57),color=SECONDARY,fontsize=8.8,va='top')
    ax=fig.add_subplot(grid[0,1]);bandit=m.get('model_bandit',{});state=bandit.get('state',{});arms=state.get('arm_names',[])
    if arms:
        y=np.arange(len(arms));admissions=native.get('calls',{}).get('mutation_model_admissions',{})
        for offset,values,color,name in [(-.23,[admissions.get(model_name(a),0) for a in arms],ORANGE,'Observed mutation/repair calls'),(0,state.get('n_submitted',[0]*len(arms)),COBALT,'Native submitted at checkpoint'),(.23,state.get('n_completed',[0]*len(arms)),MAGENTA,'Native reward updates')]:
            ax.barh(y+offset,values,height=.20,color=color,label=name)
        ax.set(yticks=y,yticklabels=[model_name(a) for a in arms],xlabel='Counts',title='B  Arm allocation and reward updates');ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(fontsize=7.5,loc='upper left',bbox_to_anchor=(0,1.0))
        ax.set_ylim(-.6,len(arms)+.9)
        ax.text(0,-.36,'Checkpoint boundary: '+str(bandit.get('snapshot_boundary','unavailable'))+'\nAn active call can postdate the saved UCB state.',transform=ax.transAxes,fontsize=8,color=SECONDARY)
    else:unavailable(ax,'B  Actual model allocation','No saved native bandit state; configured arms are not treated as exercised.')
    ax=fig.add_subplot(grid[1,1]);ps=prompts.get('prompts',[]) if prompts else []
    if ps:
        y=np.arange(len(ps));total=[p.get('program_count',0) for p in ps];correct=[p.get('correct_program_count',0) for p in ps]
        ax.barh(y,total,color=RULE,label='All credited programs');ax.barh(y,correct,color=COBALT,label='Correct-program credits')
        ax.set(yticks=y,yticklabels=[f"Prompt {p.get('generation',i)} · {p['id'][:6]}" for i,p in enumerate(ps)],xlabel='Program credits',title='C  Native prompt credit');ax.invert_yaxis();ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(fontsize=8,loc='upper left')
        ax.set_ylim(len(ps)-.4,-1.0)
        ax.text(0,-.36,'Native prompt fitness averages correct-program percentiles.\nCounts include failures; this is not failure-inclusive task fitness.',transform=ax.transAxes,fontsize=8,color=SECONDARY)
    else:unavailable(ax,'C  Native prompt credit','Prompt records pending.')
    fig.suptitle('RUN1 native machinery · observed events, not causal attributions',fontsize=16,fontweight='bold')
    footer(fig,'Readiness and native novelty fixture calls are separate from discovery; a fixture does not prove the discovery judge ran.\n'
        'Local BGE embeddings truncate to 10,000 source characters and are not validated semantic code novelty.\n'
        'Native UCB reward updates are shifted/scaled feedback; allocation is not a controlled comparison of model quality.\n'
        'All mechanisms are enabled together. One search cannot establish reliable discovery or isolate a mechanism’s benefit.')
    save(fig,out,'native-machinery',outputs)


def ancestry(lineage,native,out,outputs):
    rs=rows(lineage);by_id={r['id']:r for r in rs}
    fig,ax=plt.subplots(figsize=(11.4,5.9));fig.subplots_adjust(left=.10,right=.97,top=.79,bottom=.27)
    if not rs:unavailable(ax,'Actual recorded parent links','No native source rows exported yet.')
    else:
        for r in rs:
            x,y=r['generation'],r['birth_island'];parent=by_id.get(r.get('parent_id'))
            if parent:ax.plot([parent['generation'],x],[parent['birth_island'],y],c=RULE,lw=1.1,zorder=1)
            for identifier in r.get('archive_inspiration_ids',[])+r.get('top_k_inspiration_ids',[]):
                source=by_id.get(identifier)
                if source:ax.plot([source['generation'],x],[source['birth_island'],y],c=MAGENTA,ls='--',alpha=.4,lw=.8,zorder=1)
            typ=r.get('patch',{}).get('patch_type','full');marker=OPERATOR_MARKERS['seed' if x==0 else 'failure' if not r['correct'] else 'diff' if typ=='diff' else 'full']
            color=ISLAND_COLORS[int(y)%len(ISLAND_COLORS)]
            ax.scatter(x,y,s=95,marker=marker,facecolor=BACKGROUND if r.get('administrative_seed_copy') else color,edgecolor=color,lw=1.2,zorder=3)
            ax.annotate(str(x),(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8,color=TEXT)
        for e in native.get('mechanisms',{}).get('islands',{}).get('migrations',[]):
            if all(k in e for k in ('generation','from','to')):
                ax.annotate('',xy=(e['generation'],e['to']),xytext=(e['generation'],e['from']),arrowprops={'arrowstyle':'->','color':ORANGE,'lw':1,'linestyle':':'})
        ax.set(xlabel='Candidate generation (island seed copies share slot 0)',ylabel='Island at source creation',yticks=range(4),ylim=(-.4,3.7));ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',alpha=.6)
        handles=[Line2D([],[],c=RULE,label='Recorded parent'),Line2D([],[],c=MAGENTA,ls='--',label='Supplied inspiration'),Line2D([],[],c=ORANGE,ls=':',label='Recorded migration')]
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.08),ncol=3,fontsize=9)
    count=native.get('counts',{})
    fig.suptitle(f"RUN1 source ancestry · {count.get('persisted_slots',0)} slots / {count.get('native_rows',0)} native rows",fontsize=16,fontweight='bold')
    footer(fig,'Open diamonds are administrative seed copies; they add neither candidate slots nor episode evaluations. Colors encode birth island.\n'
        'Circle: full/crossover proposal; square: diff; X: native-invalid slot. Node labels identify exact source slots.\n'
        'Parent, inspiration and migration links record supplied context and native actions; they do not establish causal necessity.')
    save(fig,out,'ancestry',outputs)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT/'artifacts/campaign-v4/run1');args=parser.parse_args()
    base=args.root.resolve();out=base/'figures';out.mkdir(parents=True,exist_ok=True)
    native=read(base/'native/native-report.json',{});lineage=read(base/'native/lineage.json',{});prompts=read(base/'native/prompts.json',{})
    candidates=rows(read(base/'science/candidate-metrics.json',[]));curve=rows(read(base/'science/resource-curve.json',[]));summary=read(base/'science/summary.json',{})
    if not native:raise ValueError('Public native report required; do not infer current execution from a configured plan')
    apply_theme();outputs={};progress(candidates,native,out,outputs);resources(candidates,curve,native,out,outputs);machinery(native,prompts,out,outputs);ancestry(lineage,native,out,outputs)
    for p,h in INPUT_HASHES.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Input advanced while plotting; rerun from a stable saved snapshot: '+p)
    record={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'theme_sha256':hashlib.sha256((ROOT/'scripts/chromatic_fields.py').read_bytes()).hexdigest(),
        'input_sha256':INPUT_HASHES,'output_sha256':outputs,'science_snapshot_utc':summary.get('exported_utc'),'native_snapshot_utc':native.get('snapshot_utc'),
        'native_persisted_slots':native.get('counts',{}).get('persisted_slots'),'science_candidate_rows':len(candidates),
        'evidence_status':{'task':'Reused development only','prediction':'On-policy diagnostics only','matched_prediction_learning':'Not established in RUN1','frozen_control_benefit':'Not established in RUN1','transfer_assessment':'Not performed in RUN1','reliable_discovery':'Not tested by one search'},
        'new_environment_episodes':0,'new_candidate_executions':0,'new_experiment_model_calls':0,'statistical_resampling':False}
    (out/'figure-manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    (out/'README.md').write_text('# RUN1 saved-snapshot figures\n\n'
        'These plots use the public scientific and native exports identified in [figure-manifest.json](figure-manifest.json). Source exports can advance independently; their timestamps and hashes are retained.\n\n'
        'All task results reuse eight development layouts across six conditions. Candidate selection is biased by those same cases. Prediction scores are on-policy; no matched-learning, frozen-control, transfer or reliable-discovery result is implied.\n\n'
        '- [Search progress](search-progress.svg): all candidate slots, missing/invalid execution, absolute task and proper prediction diagnostics.\n'
        '- [Resources](resources.svg): measured local CPU, reported experiment-model tokens and registered caps; no estimated price is treated as a subscription charge.\n'
        '- [Native machinery](native-machinery.svg): configured versus observed execution, bandit checkpoint counts, prompt credit. Native exponential reward sums are not mislabelled as mean rewards.\n'
        '- [Ancestry](ancestry.svg): recorded parent/inspiration/migration relationships, retaining administrative seed copies.\n\n'
        '```bash\n.venv/bin/python scripts/run1_figures.py\n```\n\nThis rendering command performs no experiment, statistical resampling or model call. SVG, PDF and PNG copies are supplied.\n')
    print(json.dumps({'visual_files':len(outputs),'candidate_rows':len(candidates),'native_slots':record['native_persisted_slots'],'new_environment_episodes':0,'new_experiment_model_calls':0}))


if __name__=='__main__':main()
