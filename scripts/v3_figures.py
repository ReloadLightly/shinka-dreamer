"""Render v3 frozen assessment evidence; never runs environments or models."""
import argparse
import json
import os
from pathlib import Path
import sys
import textwrap

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


def interim_label(data):
    interim=data.get('interim')
    if not interim:return None
    return (f"INTERIM · {interim['observed_cases']:,} / {interim['registered_cases']:,} "
            'registered cases per regime')


def save_evidence_figure(fig,path,data):
    label=interim_label(data)
    if label:
        top=max(1.,fig._suptitle.get_position()[1] if fig._suptitle else 1.)
        fig.text(.5,top+.20/fig.get_figheight(),label+'\n'
                 'Nominal descriptive 95% intervals · not stopping-adjusted or confirmatory',
                 ha='center',va='bottom',fontsize=10,fontweight='bold',color=MAGENTA,
                 linespacing=1.45)
    save_figure(fig,path)


def write_evidence_tables(path,lines,data):
    label=interim_label(data)
    if label:
        annotated=[]
        for index,line in enumerate(lines):
            if line.startswith('|') and index+1<len(lines) and lines[index+1].startswith('|:--'):
                annotated.extend(['**'+label+'.** Nominal descriptive 95% intervals; '
                    'not stopping-adjusted or confirmatory. The original sample-size power claim does not apply.',''])
            annotated.append(line)
        lines=annotated
    path.write_text('\n'.join(lines)+'\n')


def outcomes(data,out):
    regimes=('uniform','stationary','switch')
    names=[k.split('/',1)[1] for k in data['outcomes'] if k.startswith('uniform/')]
    fig,axes=plt.subplots(3,2,figsize=(11.6,max(15,len(names)*1.5)),
                         gridspec_kw={'width_ratios':[1.3,1],'wspace':.22,'hspace':.60})
    fig.subplots_adjust(left=.24,right=.78,top=.89,bottom=.06)
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
               xlabel='Escape difference\n(percentage points)',title='Paired escape effects\nPointwise 95% intervals')
        ax.yaxis.tick_right()
        ax.tick_params(axis='y',labelsize=9,pad=7)
        ax.invert_yaxis()
    fig.suptitle('Unknown enemy dynamics · frozen-program assessment',fontsize=16,fontweight='bold',y=.995)
    save_evidence_figure(fig,out/'outcomes-effects',data)


def forecast_cost(data,out):
    names=[k.split('/',1)[1] for k in data['outcomes'] if k.startswith('uniform/')]
    fig,axes=plt.subplots(1,2,figsize=(10,6.5),gridspec_kw={'wspace':.65})
    fig.subplots_adjust(top=.80)
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
    axes[1].set(title='Candidate computation',xlabel='CPU seconds per episode\n(measured executions only)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,fontsize=9)
    fig.suptitle('Prediction quality and computation are separate endpoints',fontsize=16,fontweight='bold')
    save_evidence_figure(fig,out/'prediction-cost',data)


def write_tables(data,out):
    lines=['# '+('Interim' if interim_label(data) else 'Frozen')+' v3 outcomes','','All outcome denominators include invalid executions. Forecast scores here are on-policy.','',
           'Paired 95% intervals are pointwise, not Holm-adjusted simultaneous intervals. Holm p-values apply only to the registered test families. CPU means use measured executions only; unavailable measurements are counted separately and never treated as zero.','',
           '| Regime | Condition | Escape | Death | Timeout | Invalid | Mean task | Near Brier | CPU s/measured episode |',
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
    write_evidence_tables(out.parent/'tables.md',lines,data)


def matched_available(group):
    return bool(group.get('available',True) and group.get('complete_matched_prediction_evidence',True))


def matched_unavailable_reason(group):
    if group.get('reason'):
        return group['reason']
    audit=group.get('audits',{})
    reasons=[]
    if audit.get('missing_records'):
        reasons.append(f"{len(audit['missing_records'])} required shadow records missing")
    count=audit.get('physical_observation_matched_episodes')
    if count is not None and count!=group.get('episodes'):
        reasons.append(f"identical observations verified on {count}/{group.get('episodes','?')} episodes")
    no_targets=[display(n) for n,k in group.get('episodes_with_forecast_targets',{}).items() if not k]
    if no_targets:
        reasons.append('no scored targets for '+', '.join(no_targets))
    return '; '.join(reasons)+'.' if reasons else 'The required complete matched-experience audit did not pass.'


def matched_figure(data,out,name='matched-learning'):
    regimes=data.get('regimes',data)
    present=[r for r in ('uniform','stationary','switch') if r in regimes]
    if not present:
        fig,ax=plt.subplots(figsize=(10,3.4));ax.axis('off')
        ax.text(.03,.65,'Matched prediction evidence unavailable',weight='bold',fontsize=14)
        ax.text(.03,.4,textwrap.fill(matched_unavailable_reason(data),90),fontsize=11)
        save_evidence_figure(fig,out/name,data)
        return
    if not any(matched_available(regimes[r]) for r in present):
        fig,ax=plt.subplots(figsize=(10,max(3.8,1.4+1.1*len(present))));ax.axis('off')
        fig.subplots_adjust(left=.04,right=.97,top=.75,bottom=.14)
        fig.suptitle('Matched prediction estimates unavailable\n'+textwrap.fill(
            data.get('study','Selected program · uniform-law planning'),72),fontsize=14,fontweight='bold')
        for i,regime in enumerate(present):
            y=.94-i/len(present)
            ax.text(0,y,regime.title(),fontsize=11,fontweight='bold',va='top')
            ax.text(.16,y,textwrap.fill(matched_unavailable_reason(regimes[regime]),86),
                    fontsize=10,va='top',color=SECONDARY)
        fig.text(.04,.055,'Estimates are withheld unless every registered matching and coverage check passes.',
                 fontsize=9,color=SECONDARY)
        save_evidence_figure(fig,out/name,data)
        return
    fig,axes=plt.subplots(len(present),2,figsize=(11.6,max(5.8,4.2*len(present))),squeeze=False,
                          gridspec_kw={'width_ratios':[1.2,1],'wspace':.32,'hspace':.75})
    fig.subplots_adjust(left=.09,right=.76,top=.73 if len(present)==1 else .85,
                        bottom=.28 if len(present)==1 else .13)
    colors={'fitted_online':COBALT,'fitted_frozen':MAGENTA,'known_law':ORANGE,
            'selected_online':COBALT,'selected_frozen':MAGENTA,
            'selected_fixed_risk':COBALT,'selected_frozen_fixed_risk':MAGENTA,
            'no_planning':COBALT,'frozen_no_planning':MAGENTA}
    legend_handles=legend_labels=None
    for row,regime in enumerate(present):
        group=regimes[regime];ax=axes[row,0]
        if not matched_available(group):
            for panel in axes[row]:panel.axis('off')
            ax.set_title(regime.title()+' · estimate unavailable',loc='left')
            ax.text(0,.70,textwrap.fill(matched_unavailable_reason(group),53),
                    transform=ax.transAxes,fontsize=10,va='top',color=SECONDARY)
            axes[row,1].text(.1,.65,'Matched estimate withheld\nAll registered audit checks are required.',
                            transform=axes[row,1].transAxes,fontsize=10,va='top',color=SECONDARY)
            continue
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
            ax.text(0,-.36,'Episodes/bin: '+', '.join(counts),transform=ax.transAxes,fontsize=8,color=SECONDARY)
        ax.set(title=regime.title()+' · recorded experience',xlabel='Episode step',ylabel='Near-cell Brier loss')
        if legend_handles is None:legend_handles,legend_labels=ax.get_legend_handles_labels()
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
               xlabel='Pooled Brier reduction (%)',title='Paired reductions\nWhole-episode 95% intervals')
        ax.yaxis.tick_right()
        ax.tick_params(axis='y',labelsize=9,pad=7)
        ax.invert_yaxis()
    if legend_handles:
        fig.legend(legend_handles,legend_labels,loc='upper center',bbox_to_anchor=(.45,.89 if len(present)==1 else .945),
                   ncol=3,fontsize=9)
    fig.suptitle('Prediction on matched experience\n'+textwrap.fill(data.get('study','Selected frozen program'),72),
                 fontweight='bold',fontsize=14,y=.997)
    fig.text(.09,.035,'Later bins select survivors. Online–frozen pairs test updating within a source.\n'
             'Cross-source and known-law pairs compare absolute losses; positive reductions favor the left predictor.',
             fontsize=9,color=SECONDARY)
    save_evidence_figure(fig,out/name,data)


def write_matched_tables(fitted,selected,out):
    lines=['# Prediction on identical experience','',
           'All five passive predictors share the original memory-policy recordings. Online–frozen comparisons test updating within a source; cross-source and known-law comparisons measure absolute predictor losses and do not isolate learning. The additional selected uniform-planning comparison has its own policy; losses across these policies are not a head-to-head predictor comparison. Positive reductions favor the left condition; intervals resample whole paired episodes. Post-switch losses condition on the recorded policy reaching the switch.','',
           '| Experience | Regime | Left − right | Target | Left / right Brier | Reduction, % [95% CI] | Left / right targets |',
           '|:--|:--|:--|:--|--:|--:|--:|']
    for experience,data in [('Original memory policy',fitted),('Selected uniform-law planning policy',selected)]:
        if data.get('available') is False and not any(r in data.get('regimes',data) for r in ('uniform','stationary','switch')):
            reason=matched_unavailable_reason(data).replace('|','\\|').replace('\n',' ')
            lines.append(f"| {experience} | All | Unavailable: {reason} | — | — | — | — |")
            continue
        for regime,group in data.get('regimes',data).items():
            if regime not in ('uniform','stationary','switch'):continue
            if not matched_available(group):
                reason=matched_unavailable_reason(group).replace('|','\\|').replace('\n',' ')
                lines.append(f"| {experience} | {regime} | Unavailable: {reason} | — | — | — | — |")
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
    write_evidence_tables(out.parent/'matched-tables.md',lines,
                          fitted if fitted.get('interim') else selected)


def exposure_figure(data,out):
    """Display unconditional exposure counts; no new statistical decisions."""
    groups=[(key.split('/',1)[1],row) for key,row in data['outcomes'].items()
            if key.startswith('switch/')]
    if not groups:return
    fig,axes=plt.subplots(1,2,figsize=(11,max(5.4,len(groups)*.49)),
                          gridspec_kw={'wspace':.25})
    fig.subplots_adjust(top=.77,bottom=.19)
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
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.96),ncol=2,fontsize=9)
    fig.suptitle('An unannounced change does not guarantee informative experience',
                 fontsize=15,fontweight='bold',y=1.04)
    fig.text(.125,.015,'All episode denominators include early endings and invalid execution.\n'
             'Anonymous occupancy contrasts are an observation proxy, not measured information gain.',
             fontsize=9,color=SECONDARY)
    save_evidence_figure(fig,out/'switch-exposure',data)


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
    write_evidence_tables(out.parent/'mechanism-tables.md',lines,data)


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
    selected={**data.get('selected_matched',{}),
              'study':'Selected program · uniform-law planning',
              'interim':data.get('interim')}
    if data.get('selected_matched'):matched_figure(selected,out,'selected-matched-learning')
    write_matched_tables(fitted,selected,out)


if __name__=='__main__':main()
