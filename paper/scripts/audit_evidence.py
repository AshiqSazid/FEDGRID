"""Independently trace manuscript data to raw logs; optionally write audit tables.

This parser does not import the manuscript's summarizer or plotting helpers.
It never changes training code, source data, historical choices or experiment logs.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,re
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT/'paper';OUT=PAPER/'tables'
ATT=['flip','scale','noise','normalized','meter','reward']
METHODS=['fedavg','median','trimmed','krum','fltrust','flame','tafdrl','tafdrl final','fedgrid','fedgrid final']
LABEL={'fedavg':'FedAvg','median':'Median','trimmed':'Trimmed mean','krum':'Krum','fltrust':'FLTrust','flame':'FLAME',
       'tafdrl':'TAFDRL-style','tafdrl final':'TAFDRL-style tuned','fedgrid':'FedGrid original','fedgrid final':'FedGrid final',
       'central':'Central SAC','local':'Local SAC','fedprox':'FedProx'}
NAME=re.compile(r'([a-z]+)_sac_(\d+)_seed(\d+)(?:_([a-z]+)_share(\d+))?(?:_(.*))?$')
METRIC=re.compile(r'^([^\n]+?)\s{2,}(-?(?:\d+(?:\.\d*)?|\.\d+)|nan)\s*$',re.M)
def read(path):
    match=NAME.fullmatch(path.stem)
    if not match:raise ValueError(path)
    method,days,seed,attack,share,variant=match.groups();variant=variant or ''
    if variant=='final' or variant.endswith('_final'):
        method+=' final';variant=variant[:-5].rstrip('_')
    text=path.read_text()
    metrics={k.strip():float(v) for k,v in METRIC.findall(text)}
    rejection=re.search(r'rejected: honest homes ([\d.]+)%.*most rejected home ([\d.]+)%.*bad homes ([\d.na]+)%',text)
    meta=dict(method=method,days=int(days),seed=int(seed),attack=attack or '',share=share or '',variant=variant,
              source=str(path.relative_to(ROOT)),metrics=metrics,text=text,
              valid=[float(v) for v in re.findall(r'^day \d+: validation ([\d.]+)',text,re.M)])
    meta['honest_rejected']=float(rejection[1]) if rejection else np.nan
    meta['maximum_rejected']=float(rejection[2]) if rejection else np.nan
    meta['bad_rejected']=float(rejection[3]) if rejection else np.nan
    bad=re.search(r'from homes \[([\d, ]*)\]',text)
    meta['bad']=[int(v) for v in bad[1].split(',') if v.strip()] if bad else []
    homes=re.search(r'rejected share by home \(solar kWp\): (.*)',text)
    meta['home_rates']=np.array(re.findall(r'([\d.]+) \(([\d.]+)\)',homes[1]),float) if homes else None
    return meta
paths=sorted((ROOT/'results').glob('*_sac_*.log'));records=[read(p) for p in paths]
candidate_paths=sorted((ROOT/'results/candidates').glob('*_sac_*.log'))
candidates=[read(p) for p in candidate_paths]
assert len(records)==706 and len(candidates)==266
index={(r['method'],r['days'],r['seed'],r['attack'],r['share'],r['variant']):r for r in records}
assert len(index)==len(records)
def lookup(method,seed,attack,variant='',share='30',days=500):
    return index[(method,days,seed,attack,share,variant)]
def damage(r,prefix=''):
    ref=lookup(r['method'],r['seed'],'none',r['variant'],r['share'])
    column=prefix+('bill honest homes $/home/yr' if r['attack'] in ['meter','reward'] else 'bill $/home/yr')
    honest_key=(r['method'],r['days'],r['seed'],'none',r['share'],'honest')
    if r['attack'] in ['meter','reward'] and not prefix and not r['variant'] and honest_key in index:
        ref=index[honest_key]
    elif column not in ref['metrics'] and not prefix:
        ref=lookup(r['method'],r['seed'],'none','honest',r['share'])
    return r['metrics'][column]-ref['metrics'][column]
def close(a,b):
    assert np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=1e-10,atol=1e-9,equal_nan=True),(a,b)
verified={}
reference_trace=[]
for setting,variant,prefix,n in [('ausgrid','','',300),('act','act','',180),('transfer','','ACT transfer ',156)]:
    data=pd.read_csv(OUT/f'{setting}_damage_per_seed.csv')
    assert len(data)==n
    for row in data.itertuples():
        r=lookup(row.method,row.seed,row.attack,variant)
        close(row.damage,damage(r,prefix));assert row.source==r['source']
        reference=lookup(r['method'],r['seed'],'none',r['variant'],r['share'])
        column=prefix+('bill honest homes $/home/yr' if r['attack'] in ['meter','reward'] else 'bill $/home/yr')
        honest_key=(r['method'],r['days'],r['seed'],'none',r['share'],'honest')
        if r['attack'] in ['meter','reward'] and not prefix and not r['variant'] and honest_key in index:reference=index[honest_key]
        reference_trace.append(dict(setting=setting,method=r['method'],seed=r['seed'],attack=r['attack'],attacked_source=r['source'],reference_source=reference['source'],metric=column,attacked_bill=r['metrics'][column],reference_bill=reference['metrics'][column],damage=row.damage))
    verified[setting+'_damage_observations']=len(data)
# Every plotted electrical and rejection observation comes directly from the raw log.
electrical=pd.read_csv(OUT/'electrical_source.csv')
for _,row in electrical.iterrows():
    r=lookup(row.method,int(row.seed),'none','act' if row.setting=='ACT' else '')
    for col in ['half hours over 253 V %','max V','transformer peak loading %']:
        close(row[col],r['metrics'].get(col,np.nan))
verified['electrical_observations']=len(electrical)
pd.DataFrame(reference_trace).to_csv(OUT/'damage_reference_trace.csv',index=False)
rejection=pd.read_csv(OUT/'rejection_heatmap_source.csv')
for _,row in rejection.iterrows():
    r=lookup(row.method,int(row.seed),row.attack)
    close(row['honest rejected %'],r['honest_rejected'])
    close(row['bad rejected %'],r['bad_rejected'])
verified['rejection_observations']=len(rejection)
# Validate paired revisions independently from the raw bill differences.
paired=pd.read_csv(OUT/'paired_revision_source.csv')
for row in paired.itertuples():
    variant='act' if row.setting=='ACT' else '';prefix='ACT transfer ' if row.setting=='Transfer' else ''
    expected=damage(lookup('fedgrid final',row.seed,row.attack,variant),prefix)-damage(lookup('fedgrid',row.seed,row.attack,variant),prefix)
    close(row.delta,expected)
verified['paired_revision_observations']=len(paired)
# Independently enumerate exact signed-rank null outcomes and recompute Holm families.
tests=pd.read_csv(OUT/'paired_exact_tests.csv')
for row in tests.itertuples():
    delta=np.array([damage(lookup('fedgrid final',s,row.attack))-damage(lookup(row.comparator,s,row.attack)) for s in range(5)]).round(8)
    ranks=stats.rankdata(abs(delta));assert len(set(abs(delta)))==5 and (delta!=0).all()
    score=ranks[delta>0].sum();null=np.array([sum(ranks[j] for j in range(5) if mask&(1<<j)) for mask in range(32)])
    p=min(1.,2*min((null<=score).mean(),(null>=score).mean()))
    close(row.p_exact,p)
for _,part in tests.groupby('family'):
    order=part.sort_values('p_exact').index
    expected=np.maximum.accumulate(tests.loc[order,'p_exact'].to_numpy()*np.arange(len(order),0,-1)).clip(max=1)
    close(tests.loc[order,'p_holm'],expected)
verified['exact_comparisons']=len(tests)
# Validate source provenance before adding coverage beyond the existing manuscript.
manifest_counts={}
for name in ['source_logs.sha256','expanded_sources.sha256','act_imputation_raw_sources.sha256']:
    count=0
    for line in (OUT/name).read_text().splitlines():
        digest,path=line.split('  ',1);assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path;count+=1
    manifest_counts[name]=count
# Paper analysis scripts legitimately change during this audit. Independently
# verify the frozen research implementation, runner and dependency lock instead.
frozen_count=0
for line in (OUT/'source_code.sha256').read_text().splitlines():
    digest,path=line.split('  ',1)
    if path.startswith('src/') or path in ['run_experiments.sh','requirements-lock.txt']:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        frozen_count+=1
assert frozen_count==17
verified['frozen_training_files']=frozen_count
# Reconcile every top-level learning log to an experimental group.
groups=Counter()
for r in records:
    v,a,sh=r['variant'],r['attack'],r['share']
    if not a:group='clean_learning'
    elif v=='act':group='ACT'
    elif v=='honest':group='honest_references'
    elif v.startswith('without'):group='ablations'
    elif v=='lost10_delayed10':group='message_faults'
    elif v=='feederO':group='feeder_O'
    elif v=='pu102':group='voltage_stress'
    elif sh!='30':group='share_references' if a=='none' else 'extra_share_attacks'
    else:group='Ausgrid_main'
    groups[group]+=1
assert sum(groups.values())==706

def interval(values,digits=1):
    values=np.asarray(values,float);values=values[np.isfinite(values)];n=len(values)
    if not n:return '--'
    mean=values.mean()
    if n==1:return f'{mean:.{digits}f} [1]'
    half=stats.t.ppf(.975,n-1)*values.std(ddof=1)/np.sqrt(n)
    return f'{mean:.{digits}f} $\\pm$ {half:.{digits}f} [{n}]'
def table(name,caption,heads,rows):
    lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',r'\label{tab:'+name+'}',
           r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{'+'l'*len(heads)+'}',r'\toprule',' & '.join(heads)+r' \\',r'\midrule']
    lines+=[' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    (OUT/(name+'.tex')).write_text('\n'.join(lines)+'\n')

if '--write-tables' in __import__('sys').argv:
    # Correct only formatting: preserve the source observations, including small nonzero violation rates.
    rows=[]
    for setting in ['Ausgrid','ACT']:
        for method in ['fedavg','fltrust','flame','tafdrl final','fedgrid','fedgrid final']:
            d=electrical[electrical.setting.eq(setting)&electrical.method.eq(method)]
            rows.append([setting,LABEL[method],interval(d['half hours over 253 V %'],3),interval(d['max V']),interval(d['transformer peak loading %'])])
    table('electrical',r'Electrical outcomes in no-attack last-policy runs: means and 95\% interval half-widths [observed seeds]. Voltage-exceedance percentages use three decimals to retain small nonzero means calculated from two-decimal logged input rates. Run maxima and peaks are computed before seed averaging; they are not safety bounds.',
          ['Setting','Method',r'Over 253 V (\%)','Run maximum (V)',r'Transformer peak (\%)'],rows)
    long=[];rows=[]
    metrics=['bill $/home/yr','curtailed kWh/home/yr','battery cycles/yr','battery charging in MSL half hours kWh']
    for method in ['local','fedavg','fedprox','central']:
        for days in [500,1000]:
            selected=[r for r in records if r['method']==method and r['days']==days and not r['attack'] and not r['variant']]
            assert len(selected)==3
            rows.append([LABEL[method],days,'0.01' if method=='fedprox' else '--',*[interval([r['metrics'][m] for r in selected]) for m in metrics]])
            for r in selected:long.append(dict(method=method,days=days,mu=.01 if method=='fedprox' else np.nan,seed=r['seed'],source=r['source'],**{m:r['metrics'][m] for m in metrics}))
    for mu in ['0.001','0.1']:
        selected=[r for r in records if r['method']=='fedprox' and r['variant']=='mu'+mu and not r['attack']]
        assert len(selected)==3
        rows.append(['FedProx',500,mu,*[interval([r['metrics'][m] for r in selected]) for m in metrics]])
        for r in selected:long.append(dict(method='fedprox',days=500,mu=float(mu),seed=r['seed'],source=r['source'],**{m:r['metrics'][m] for m in metrics}))
    pd.DataFrame(long).to_csv(OUT/'extended_clean_source.csv',index=False)
    table('extended_clean',r'All 30 clean-learning runs: duration and FedProx-coefficient comparisons with validation-selected actors. Entries are mean $\pm$ 95\% interval half-width [seeds]. Bills and curtailment are per household annually; cycles are fleet-equivalent, and MSL charging is a street total.',
          ['Method','Days',r'$\mu$',r'Bill (\$/yr)','Curtailment (kWh/yr)','Cycles/yr','MSL charge (kWh)'],rows)
    # Complete deterministic baseline context across the three archived scenarios.
    baseline_rows=[];baseline_data=[]
    for scenario,name in [('Ausgrid 1.00 pu','baselines.log'),('ACT 1.00 pu','baselines_act.log'),('Ausgrid 1.02 pu','baselines_pu102.log')]:
        text=(ROOT/'results'/name).read_text()
        found=re.findall(r'^([^\n]+?)\s{2,}(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$',text,re.M)
        cols={key.strip():[float(a),float(b),float(c)] for key,a,b,c in found}
        for i,method in enumerate(['Idle','Rule','Perfect-foresight MPC']):
            row=dict(scenario=scenario,method=method,source='results/'+name,**{key:values[i] for key,values in cols.items()});baseline_data.append(row)
            keys=['bill $/home/yr','curtailed kWh/home/yr','half hours over 253 V %','max V','transformer peak loading %']
            baseline_rows.append([scenario,method,*[('--' if key not in row else f'{row[key]:.{3 if "half hours" in key else 2}f}') for key in keys]])
    pd.DataFrame(baseline_data).to_csv(OUT/'all_baselines_source.csv',index=False)
    table('all_baselines',r'All archived deterministic baselines, in Australian dollars per household per year and curtailed kWh per household per year. Electrical metrics describe observed evaluation intervals. Missing legacy transformer measurements are shown as dashes.',
          ['Scenario','Controller',r'Bill (\$/yr)','Curtailment (kWh/yr)',r'Over 253 V (\%)','Max V',r'Peak loading (\%)'],baseline_rows)
    # Show attack-time grid outcomes, not only benign references.
    physical=[]
    for r in records:
        if r['method'] not in METHODS or r['attack'] not in ATT or r['share']!='30' or r['variant'] not in ['', 'act']:continue
        physical.append(dict(setting='ACT' if r['variant']=='act' else 'Ausgrid',method=r['method'],seed=r['seed'],attack=r['attack'],source=r['source'],**{c:r['metrics'].get(c,np.nan) for c in ['half hours over 253 V %','max V','transformer peak loading %']}))
    physical=pd.DataFrame(physical);physical.to_csv(OUT/'attack_electrical_source.csv',index=False)
    rows=[]
    for method in METHODS:
        values=[]
        for setting in ['Ausgrid','ACT']:
            d=physical[physical.method.eq(method)&physical.setting.eq(setting)]
            values += [f'{d[c].max():.{3 if "half hours" in c else 2}f}' if d[c].notna().any() else '--' for c in ['half hours over 253 V %','max V','transformer peak loading %']]
        rows.append([LABEL[method],*values])
    assert len(physical)==480
    for method in METHODS:
        for setting,expected in [('Ausgrid',30),('ACT',18)]:
            d=physical[physical.method.eq(method)&physical.setting.eq(setting)]
            assert len(d)==expected
            assert d['max V'].notna().sum()==expected
            assert d['half hours over 253 V %'].notna().sum()==expected
            peaks=26 if setting=='Ausgrid' and method in ['fedavg','median','trimmed','krum'] else expected
            assert d['transformer peak loading %'].notna().sum()==peaks
    table('attack_electrical',r'Maximum observed run-level electrical metrics over all six attacks at 30\% nominal compromise: five Ausgrid seeds and three ACT seeds. Each cell is a maximum across available logs; different maxima may arise from different attacks or seeds. Ausgrid transformer measurements cover 26/30 runs for FedAvg, median, trimmed mean and Krum, and 30/30 otherwise; ACT covers 18/18 throughout. All voltage metrics have full coverage. Missing measurements are excluded, never set to zero; per-run inputs are retained in the CSV.',
          ['Method',r'Ausgrid over 253 V (\%)','Ausgrid max V',r'Ausgrid peak (\%)',r'ACT over 253 V (\%)','ACT max V',r'ACT peak (\%)'],rows)
    # Describe heterogeneous rejection from rounded per-home logs.
    strata=[];rows=[]
    for method in ['krum','fltrust','flame','tafdrl','tafdrl final','fedgrid','fedgrid final']:
        for attack in ['none','normalized']:
            selected=[r for r in records if r['method']==method and r['attack']==attack and r['share']=='30' and not r['variant'] and r['home_rates'] is not None]
            for r in selected:
                rates,pv=r['home_rates'].T;honest=np.setdiff1d(np.arange(len(rates)),r['bad']);lo,hi=np.quantile(pv,[1/6,5/6]);unusual=(pv[honest]<=lo)|(pv[honest]>=hi)
                strata.append(dict(method=method,attack=attack,seed=r['seed'],source=r['source'],overall=r['honest_rejected'],unusual=100*rates[honest][unusual].mean(),usual=100*rates[honest][~unusual].mean(),maximum=r['maximum_rejected']))
            d=pd.DataFrame(strata);d=d[d.method.eq(method)&d.attack.eq(attack)]
            rows.append([LABEL[method],attack.title(),*[interval(d[c]) for c in ['overall','unusual','usual','maximum']]])
    pd.DataFrame(strata).to_csv(OUT/'heterogeneity_source.csv',index=False)
    table('heterogeneity',r'Logged honest rejection (\%) by solar-capacity stratum, no attack and Normalized, seeds 0--4 where available. Unusual homes fall at or beyond the lowest/highest sextile thresholds of rounded street capacities; ties may enlarge the stratum. Per-home rates are rounded in the logs. Entries are mean $\pm$ 95\% interval half-width [observed seeds]. The last column averages the maximum household rejection within each run; these are filter diagnostics, not effective-influence measurements.',
          ['Method','Condition',r'Overall (\%)',r'Unusual PV (\%)',r'Other PV (\%)',r'Mean run maximum (\%)'],rows)
    # Reconstruct the historical candidate statistic and its unequal rejection pooling.
    pool=[]
    for r in [*records,*candidates]:
        if r['seed']>2 or r['days']!=500 or r['method'] not in ['fedgrid','tafdrl'] or r['variant'].startswith(('act','honest','lost','feeder','pu')):continue
        rr=dict(r);rr['version']=r['variant'] or 'original';pool.append(rr)
    settings=[];rows=[]
    cases=[('Final FedGrid','fedgrid','filter_first_clip2','3 bands, z 2.5'),('Final FedGrid','fedgrid','filter_first_clip2_bands2','2 bands'),('Final FedGrid','fedgrid','filter_first_clip2_bands4','4 bands'),('Final FedGrid','fedgrid','filter_first_clip2_z2','z 2.0'),('Final FedGrid','fedgrid','filter_first_clip2_z3','z 3.0'),
           ('Earlier home trust','fedgrid','filter_first_home_trust','3 bands, z 2.5'),('Earlier home trust','fedgrid','filter_first_home_trust_bands2','2 bands'),('Earlier home trust','fedgrid','filter_first_home_trust_bands4','4 bands'),('Earlier home trust','fedgrid','filter_first_home_trust_z2','z 2.0'),('Earlier home trust','fedgrid','filter_first_home_trust_z3','z 3.0')]
    cases += [('TAFDRL-style','tafdrl',f'lam{lam}_level{level}',f'lambda {lam}, level .{level}') for lam in [1,2] for level in ['99','999']]
    for family,method,version,label in cases:
        selected=[r for r in pool if r['method']==method and r['version']==version and r['share']=='30']
        none={r['seed']:np.mean(r['valid'][-4:]) for r in selected if r['attack']=='none'}
        deltas=defaultdict(list)
        for r in selected:
            if r['attack'] in ATT[:4]:deltas[r['attack']].append(np.mean(r['valid'][-4:])-none[r['seed']])
        worst=max(np.mean(v) for v in deltas.values());honest=np.mean([r['honest_rejected'] for r in selected])
        forty=[np.mean(r['valid'][-4:])-none[r['seed']] for r in pool if r['method']==method and r['version']==version and r['share']=='40' and r['attack']=='normalized']
        forty_mean=float(np.mean(forty)) if forty else np.nan
        settings.append(dict(family=family,method=method,version=version,worst=worst,honest_rejected=honest,rejection_runs=len(selected),attacks=len(deltas),paired_runs=sum(map(len,deltas.values())),forty_percent_damage=forty_mean,forty_percent_runs=len(forty)))
        rows.append([family,label,f'{worst:.1f}',f'{honest:.1f}',len(selected),len(deltas),'--' if not forty else f'{forty_mean:.1f}'])
    pd.DataFrame(settings).to_csv(OUT/'selection_settings_source.csv',index=False)
    table('selection_settings',r'Completed later settings and TAFDRL-style tuning on validation data. Worst damage uses the available update attacks (four for FedGrid; two for tuned TAFDRL-style). Recorded honest rejection pools all available 30\% runs, including no attack; the two families therefore have different pools. TAFDRL-style tuning fixes minimum trust at 0.35. The 40\% column is three-seed Normalized validation damage for FedGrid, excluded from the ranking criterion. Values follow the historical one-decimal reporting and are not annual evaluation damage.',
          ['Family','Setting',r'Worst validation damage (\$)',r'Honest rejection (\%)','Rejection runs','Attacks',r'40\% damage (\$)'],rows)
    verified.update(extended_clean_observations=len(long),deterministic_baseline_rows=len(baseline_data),attack_electrical_observations=len(physical),household_rejection_observations=len(strata),selection_settings=len(settings))
# A checkable source inventory includes inputs omitted from the earlier manifest.
audit_inputs=set([*paths,*candidate_paths,*(ROOT/'src').rglob('*.py'),ROOT/'run_experiments.sh',Path(__file__)])
audit_inputs.update(ROOT/'results'/n for n in ['baselines.log','baselines_act.log','baselines_pu102.log','device_single_core.log','diagnostics/README.md'])
audit_inputs.update((ROOT/'results/candidates').glob('choice_*.txt'));audit_inputs.update((ROOT/'results/candidates').glob('*.flags'))
with (OUT/'evidence_audit_sources.sha256').open('w') as out:
    for path in sorted(audit_inputs):out.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(ROOT))+'\n')
report=dict(verified=verified,manifests=manifest_counts,learning_log_groups=dict(groups),learning_logs=706,candidate_logs=266,
            findings=['Voltage percentage formatting concealed nonzero observations.','Selection rejection statistics use unequal pools and rounded thresholds.','Earlier residual-controller choices used evaluation-year outcomes.','Extended clean, baseline and attack-grid outcomes need explicit coverage.','Median and cosine numerical conventions need specification.'],
            training_code_changed=False,result_logs_changed=False,scope='Checks recorded computations and reporting, not external validity or a rerun of every controller.')
(OUT/'evidence_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
