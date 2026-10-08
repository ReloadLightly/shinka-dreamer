"""Render v3 frozen assessment evidence; never runs environments or models."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from visual_theme import apply_theme,save_figure,COBALT,MAGENTA,ORANGE,SECONDARY,RULE
import matplotlib.pyplot as plt
import numpy as np

LABELS={'memory':'Original memory','v2_gen14':'v2 generation 14',
        'seed':'Original predictive seed','selected':'Selected evolved',
        'selected_online':'Selected, online',
        'selected_frozen':'Selected, frozen','selected_fixed_risk':'Selected, uniform-law planning',
        'selected_frozen_fixed_risk':'Selected, frozen + uniform law',
        'fitted_online':'Fitted predictor, online','fitted_frozen':'Fitted predictor, frozen',
        'known_law':'Known-law reference'}


def display(name): return LABELS.get(name,name.replace('_',' '))


def outcomes(data,out):
    regimes=('uniform','stationary','switch')
    names=[k.split('/',1)[1] for k in data['outcomes'] if k.startswith('uniform/')]
    fig,axes=plt.subplots(3,2,figsize=(10,max(11,len(names)*1.25)),
                         gridspec_kw={'width_ratios':[1.2,1],'wspace':.65,'hspace':.55})
    for row,regime in enumerate(regimes):
        ax=axes[row,0]; y=np.arange(len(names)); left=np.zeros(len(names))
        for event,color,label in [('escape',COBALT,'Escape'),('death',MAGENTA,'Death'),
                                  ('timeout',ORANGE,'Timeout'),('invalid',SECONDARY,'Invalid')]:
            values=np.array([data['outcomes'][f'{regime}/{n}'][event]['rate']*100 for n in names])
            ax.barh(y,values,left=left,color=color,label=label,height=.65)
            left+=values
        ax.set(yticks=y,yticklabels=[display(n) for n in names],xlim=(0,100),
               xlabel='All episodes (%)',title=f'{regime.title()} · n = {data["cases_per_regime"]:,}')
        ax.invert_yaxis()
        if row==0: ax.legend(ncol=2,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.12))
        pairs=[v for v in data['pairs'].values() if v['regime']==regime]
        ax=axes[row,1]
        for i,p in enumerate(pairs):
            effect=p['outcomes']['escape']; value=100*effect['difference']; lo,hi=np.array(effect['ci95'])*100
            ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color=COBALT if p['family']=='primary' else MAGENTA,capsize=3)
        ax.axvline(0,c=SECONDARY,lw=.7)
        ax.set(yticks=range(len(pairs)),yticklabels=[f"{display(p['left'])}\n− {display(p['right'])}" for p in pairs],
               xlabel='Escape difference (percentage points)',title='Paired effects · pointwise 95% intervals')
        ax.invert_yaxis()
    fig.suptitle('Unknown enemy dynamics · frozen-program assessment',fontsize=16,fontweight='bold',y=.995)
    save_figure(fig,out/'outcomes-effects')


def forecast_cost(data,out):
    names=[k.split('/',1)[1] for k in data['outcomes'] if k.startswith('uniform/')]
    fig,axes=plt.subplots(1,2,figsize=(10,6.5),gridspec_kw={'wspace':.65})
    for index,regime in enumerate(('uniform','stationary','switch')):
        y=np.arange(len(names))+(index-1)*.23
        color=(SECONDARY,COBALT,MAGENTA)[index]
        for ax,key in zip(axes,('forecast','cost')):
            values=[]; low=[]; high=[]
            for name in names:
                row=data['outcomes'][f'{regime}/{name}']
                stat=row['forecasts']['brier_near'] if key=='forecast' else row['candidate_cpu_seconds']
                if not stat or stat.get('ci95') is None or (key == 'cost' and stat.get('mean') is None):
                    values.append(np.nan);low.append(np.nan);high.append(np.nan)
                    continue
                value=stat['loss'] if key=='forecast' else stat['mean']
                values.append(value); low.append(value-stat['ci95'][0]); high.append(stat['ci95'][1]-value)
            ax.errorbar(values,y,xerr=[low,high],fmt='o',color=color,markersize=4,capsize=2,label=regime.title())
    for ax in axes:
        ax.set(yticks=range(len(names)),yticklabels=[display(n) for n in names]);ax.invert_yaxis()
        ax.grid(axis='x',alpha=.5)
    axes[0].set(title='On-policy predictions',xlabel='Pooled near-cell Brier loss')
    axes[1].set(title='Candidate computation',xlabel='CPU seconds per episode')
    axes[0].legend(fontsize=9)
    fig.suptitle('Prediction quality and computation are separate endpoints',fontsize=16,fontweight='bold')
    save_figure(fig,out/'prediction-cost')


def write_tables(data,out):
    lines=['# Frozen v3 outcomes','','All outcome denominators include invalid executions. Forecast scores here are on-policy.','',
           'Paired 95% intervals are pointwise, not Holm-adjusted simultaneous intervals. Holm p-values apply only to the registered test families. CPU means use measured executions only; unavailable measurements are counted separately and never treated as zero.','',
           '| Regime | Condition | Escape | Death | Timeout | Invalid | Mean task | Near Brier | CPU s/episode |',
           '|:--|:--|--:|--:|--:|--:|--:|--:|--:|']
    for key,row in data['outcomes'].items():
        regime,name=key.split('/',1)
        near=row['forecasts']['brier_near']
        near_text=f"{near['loss']:.6f}" if near else 'Unavailable'
        cpu=row['candidate_cpu_seconds']
        cpu_text=f"{cpu['mean']:.3f}" if cpu['mean'] is not None else 'Unavailable'
        if cpu.get('unavailable_episodes'):
            cpu_text+=f" ({cpu['unavailable_episodes']} unmeasured)"
        lines.append(f"| {regime} | {display(name)} | {row['escape']['count']}/{row['episodes']} | {row['death']['count']} | {row['timeout']['count']} | {row['invalid']['count']} | {row['combined_score']['mean']:.4f} | {near_text} | {cpu_text} |")
    lines+=['','| Regime | Paired contrast | Escape Δ, pp [95% CI] | Wins / losses | Exact p | Holm p |','|:--|:--|--:|--:|--:|--:|']
    for row in data['pairs'].values():
        effect=row['outcomes']['escape'];lo,hi=np.array(effect['ci95'])*100
        hp=f"{effect['holm_p']:.4g}" if 'holm_p' in effect else '—'
        lines.append(f"| {row['regime']} | {display(row['left'])} − {display(row['right'])} | {100*effect['difference']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {effect['left_only']} / {effect['right_only']} | {effect['mcnemar_exact_p']:.4g} | {hp} |")
    if data.get('regime_interactions'):
        lines+=['','Descriptive differences between regime-specific escape effects; conservative pointwise intervals preserve shared-case dependence.','',
                '| Regime difference | Within-regime contrast | Difference of effects, pp [95% CI] |',
                '|:--|:--|--:|']
        for row in data['regime_interactions'].values():
            lo,hi=np.array(row['ci95_conservative'])*100
            lines.append(f"| {row['regime_a']} − {row['regime_b']} | {display(row['left'])} − {display(row['right'])} | {100*row['difference']:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    (out.parent/'tables.md').write_text('\n'.join(lines)+'\n')


def matched_figure(data,out,name='matched-learning'):
    regimes=data.get('regimes',data)
    available=[r for r in ('uniform','stationary','switch') if r in regimes and regimes[r].get('available',True)]
    if not available: return
    fig,axes=plt.subplots(len(available),2,figsize=(10,max(4.8,3.7*len(available))),squeeze=False,
                          gridspec_kw={'wspace':.48,'hspace':.62})
    fig.subplots_adjust(top=.84 if len(available)==1 else .90,bottom=.19 if len(available)==1 else .08)
    colors={'fitted_online':COBALT,'fitted_frozen':MAGENTA,'known_law':ORANGE,
            'selected_online':COBALT,'selected_frozen':MAGENTA,
            'selected_fixed_risk':COBALT,'selected_frozen_fixed_risk':MAGENTA,
            'no_planning':COBALT,'frozen_no_planning':MAGENTA}
    for row,regime in enumerate(available):
        group=regimes[regime];ax=axes[row,0]
        bins=group.get('bins',[])
        conditions=list(bins[0].get('conditions',{})) if bins else list(group.get('forecasts',{}))
        for i,condition in enumerate(conditions):
            points=[b for b in bins if b.get('conditions',{}).get(condition)]
            if not points: continue
            x=[b['midpoint'] for b in points];stats=[b['conditions'][condition] for b in points]
            y=[s['loss'] for s in stats];color=colors.get(condition,(COBALT,MAGENTA,ORANGE)[i%3])
            style = 's--' if condition.startswith('fitted_') else 'o-'
            ax.plot(x,y,style,color=color,label=display(condition),markersize=3)
            ax.fill_between(x,[s['ci95'][0] for s in stats],[s['ci95'][1] for s in stats],color=color,alpha=.1)
        if conditions:
            counts=[str((b.get('conditions',{}).get(conditions[0]) or {}).get('contributing_episodes',0)) for b in bins]
            ax.text(0,-.27,'Contributing episodes: '+', '.join(counts),transform=ax.transAxes,fontsize=9,color=SECONDARY)
        ax.set(title=regime.title()+' · identical recorded experience',xlabel='Episode step (later bins select survivors)',ylabel='Near-cell Brier loss')
        if row==0:ax.legend(fontsize=8)
        ax=axes[row,1]
        pairs=group.get('pairs',{})
        if not pairs and group.get('left'):
            pairs={group['left']+'-minus-'+group['right']:{'forecasts':group['forecasts']}}
        for i,(key,pair) in enumerate(pairs.items()):
            stat=pair.get('forecasts',{}).get('brier_near')
            if not stat or stat.get('relative_reduction') is None:continue
            value=100*stat['relative_reduction'];lo,hi=np.array(stat['relative_reduction_ci95'])*100
            ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color=COBALT if i==0 else ORANGE,capsize=3)
        ax.axvline(0,color=SECONDARY,lw=.8)
        ax.set(yticks=range(len(pairs)),yticklabels=['\n− '.join(display(p) for p in k.split('-minus-')) for k in pairs],
               xlabel='Pooled Brier reduction (%)',title='Paired whole-episode uncertainty')
        ax.invert_yaxis()
    fig.suptitle('Predictive learning on matched experience\n'+data.get('study','Selected frozen program'),
                 fontweight='bold',fontsize=14,y=.997)
    save_figure(fig,out/name)


def write_matched_tables(fitted,selected,out):
    lines=['# Prediction on identical experience','',
           'All five passive predictors share the original memory-policy recordings. The additional selected uniform-planning comparison has its own policy; losses across these policies are not a head-to-head predictor comparison. Positive reductions favor the left condition; intervals resample whole paired episodes. Post-switch losses condition on the recorded policy reaching the switch.','',
           '| Experience | Regime | Left − right | Target | Left / right Brier | Reduction, % [95% CI] | Left / right targets |',
           '|:--|:--|:--|:--|--:|--:|--:|']
    for experience,data in [('Original memory policy',fitted),('Selected uniform-law planning policy',selected)]:
        for regime,group in data.get('regimes',data).items():
            if regime not in ('uniform','stationary','switch'):continue
            if not group.get('available',True):
                lines.append(f"| {experience} | {regime} | Unavailable | — | — | — | — |")
                continue
            pairs=group.get('pairs',{})
            if not pairs and group.get('left'):
                pairs={group['left']+'-minus-'+group['right']:{'forecasts':group['forecasts']}}
            for key,pair in pairs.items():
                for metric in ('brier_near','brier_destination','brier_near_post_switch'):
                    if metric.endswith('post_switch') and regime!='switch':continue
                    stat=pair.get('forecasts',{}).get(metric)
                    if not stat:continue
                    value,ci=stat['relative_reduction'],stat['relative_reduction_ci95']
                    reduced=(f"{100*value:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]"
                             if value is not None and ci else 'Unavailable')
                    lines.append(f"| {experience} | {regime} | {key.replace('-minus-',' − ').replace('_',' ')} | {metric.replace('brier_','').replace('_',' ')} | {stat['left_loss']:.6f} / {stat['right_loss']:.6f} | {reduced} | {stat['left_targets']} / {stat['right_targets']} |")
    (out.parent/'matched-tables.md').write_text('\n'.join(lines)+'\n')


def exposure_figure(data,out):
    """Display unconditional exposure counts; no new statistical decisions."""
    groups=[(key.split('/',1)[1],row) for key,row in data['outcomes'].items()
            if key.startswith('switch/')]
    if not groups:return
    fig,axes=plt.subplots(1,2,figsize=(11,max(5.4,len(groups)*.49)),
                          gridspec_kw={'wspace':.25})
    y=np.arange(len(groups))
    for offset,key,color,label in [(-.16,'encountered',COBALT,'Encountered replacement law'),
            (.16,'contrast',MAGENTA,'Observed a post-switch contrast')]:
        values=[]
        for _,row in groups:
            value=(row['switch_exposure']['encountered'] if key=='encountered' else
                   row['observed_transition_opportunities']['post_switch_episodes_with_observed_contrast'])
            values.append(100*value/row['episodes'])
        axes[0].barh(y+offset,values,height=.27,color=color,label=label)
    means=[row['observed_transition_opportunities']['post_switch_sums'].get(
        'observed_destination_contrast_opportunities',0)/row['episodes'] for _,row in groups]
    axes[1].barh(y,means,height=.55,color=ORANGE)
    for index,value in enumerate(means):
        axes[1].annotate(f'{value:.2f}',(value,index),xytext=(5,0),textcoords='offset points',
                         va='center',fontsize=9,color=SECONDARY)
    axes[0].set(yticks=y,yticklabels=[display(name) for name,_ in groups],xlim=(0,100),
                xlabel='Fraction of all episodes (%)',title='Exposure is policy dependent')
    axes[1].set(yticks=y,yticklabels=[],xlabel='Source-cell contrasts per episode',
                title='Observed learning opportunities')
    axes[1].set_xlim(0,max(means,default=0)*1.25 or 1)
    for ax in axes:ax.invert_yaxis()
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.13),fontsize=9)
    fig.suptitle('An unannounced change does not guarantee informative experience',
                 fontsize=15,fontweight='bold',y=1.08)
    fig.text(.125,-.04,'All episode denominators include early endings and invalid execution.\n'
             'Anonymous occupancy contrasts are an observation proxy, not measured information gain.',
             fontsize=9,color=SECONDARY)
    save_figure(fig,out/'switch-exposure')


def write_mechanism_tables(data,out):
    lines=['# Behavior, adaptive state and switch exposure','',
           'Sequence changes describe behavior, not beneficial control. Invalid pairs remain in the outcome analysis; missing audits are unavailable. Recorded physical-state hashes cover pre-action world snapshots and post-transition enemies, but not the final agent position.','',
           '| Regime | Paired contrast | Actions differ / all cases | Observations differ | Invalid pairs | Action audits unavailable |',
           '|:--|:--|--:|--:|--:|--:|']
    for row in data['pairs'].values():
        b=row.get('behavior')
        if not b:continue
        lines.append(f"| {row['regime']} | {display(row['left'])} − {display(row['right'])} | {b['actions']['differing_pairs']}/{b['cases']} | {b['physical_observations']['differing_pairs']} | {b['invalid_pairs']} | {b['actions']['unavailable_valid_pairs']} |")
    lines+=['','Parameter changes can include forgetting or regularization. Exports are source-specific and do not by themselves prove predictive learning.','',
            '| Regime | Condition | Exported parameter audits | Constant | Changed | Unavailable | Localization errors / checks |',
            '|:--|:--|--:|--:|--:|--:|--:|']
    for key,row in data['outcomes'].items():
        regime,name=key.split('/',1);a=row['state_audit']
        lines.append(f"| {regime} | {display(name)} | {a['parameters_exported_episodes']} | {a['parameters_constant_episodes']} | {a['parameters_changed_episodes']} | {a['parameters_unavailable_episodes']} | {a['localization_errors']}/{a['localization_checks']} |")
    lines+=['','Switch exposure uses all episodes as its denominator. Post-switch forecast windows and observed contrasts require reaching those transitions/observations and are survivor-conditioned. Contrasts are anonymous occupied-source-cell opportunities, not enemy identities or independent samples.','',
            '| Condition | Encountered switch / all episodes | Episodes with post-switch contrast | Post-switch contrasts | Fully observed contrasts | Opportunity audits available |',
            '|:--|--:|--:|--:|--:|--:|']
    for key,row in data['outcomes'].items():
        if not key.startswith('switch/'):continue
        a=row['observed_transition_opportunities'];s=a['post_switch_sums']
        lines.append(f"| {display(key.split('/',1)[1])} | {row['switch_exposure']['encountered']}/{row['episodes']} | {a['post_switch_episodes_with_observed_contrast']} | {s.get('observed_destination_contrast_opportunities',0)} | {s.get('fully_observed_transition_opportunities',0)} | {a['available_episode_audits']} |")
    (out.parent/'mechanism-tables.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',default='artifacts/campaign-v3/assessment/analysis.json')
    parser.add_argument('--out',default='artifacts/campaign-v3/assessment/figures')
    parser.add_argument('--matched',default='artifacts/campaign-v3/assessment/matched-analysis.json')
    args=parser.parse_args()
    data=json.loads(Path(args.analysis).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    apply_theme();outcomes(data,out);forecast_cost(data,out);write_tables(data,out)
    exposure_figure(data,out);write_mechanism_tables(data,out)
    fitted=json.loads(Path(args.matched).read_text()) if Path(args.matched).exists() else {}
    if fitted: matched_figure(fitted,out)
    if data.get('selected_matched'):matched_figure(data['selected_matched'],out,'selected-matched-learning')
    write_matched_tables(fitted,data.get('selected_matched',{}),out)


if __name__=='__main__':main()
