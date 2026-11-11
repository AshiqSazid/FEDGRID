"""Generate manuscript and appendix artifacts from archived logs only.

Run after build_artifacts.py. Every numerical panel has a CSV source. No training.
"""
from pathlib import Path
import sys, re, json, hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'paper'
sys.path.insert(0,str(ROOT/'src'))
from summarize import read, DEFENCES, DATA_ATTACKS, TRANSFER
from select_candidate import tables as candidate_tables
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
ATT=['flip','scale','noise','normalized','meter','reward']
LABEL={'fedavg':'FedAvg','median':'Median','trimmed':'Trimmed mean','krum':'Krum','fltrust':'FLTrust',
       'flame':'FLAME','tafdrl':'TAFDRL-style','tafdrl final':'TAFDRL-style tuned',
       'fedgrid':'FedGrid original','fedgrid final':'FedGrid final','central':'Central SAC',
       'local':'Local SAC','fedprox':'FedProx'}
paths=sorted((ROOT/'results').glob('*_sac_*.log'))
runs=pd.DataFrame([dict(read(p), source=str(p.relative_to(ROOT))) for p in paths])
runs['seed']=runs.seed.astype(int)
runs[['attack','share','variant']]=runs[['attack','share','variant']].fillna('')
figs=[]; tabs=[]
def esc(x):
    return str(x).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
def interval(x):
    a=np.asarray(x,float); a=a[np.isfinite(a)]; n=len(a)
    if not n:return np.nan,np.nan,0
    return a.mean(), (stats.t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n) if n>1 else np.nan),n
def fmt(x,digits=1):
    m,h,n=interval(x)
    return '--' if n==0 else (f'{m:.{digits}f} [{n}]' if n==1 else f'{m:.{digits}f} $\\pm$ {h:.{digits}f} [{n}]')
def table(name,caption,headers,rows):
    lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',r'\label{tab:'+name+'}',
           r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',
           ' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    (OUT/'tables'/f'{name}.tex').write_text('\n'.join(lines)+'\n');tabs.append(name)
def save(name,fig,data=None):
    fig.savefig(OUT/'figures'/f'{name}.pdf');fig.savefig(OUT/'figures'/f'{name}.png',dpi=200)
    plt.close(fig);figs.append(name)
    if data is not None:data.to_csv(OUT/'tables'/f'{name}_source.csv',index=False)
def damage(part,prefix='',reference=None):
    part=part.copy(); keys=['method','seed','share','variant']
    none=part[part.attack.eq('none')]
    if none.duplicated(keys).any(): raise ValueError('Duplicate reference rows')
    allref=none.set_index(keys)[prefix+'bill $/home/yr'].to_dict()
    col=prefix+'bill honest homes $/home/yr'
    href=none.dropna(subset=[col]).set_index(keys)[col].to_dict()
    if reference is not None and not prefix:
        # Explicit named-column iteration avoids pandas tuple-name mangling.
        for _,row in reference.iterrows():
            href[(row.method,row.seed,row.share,'')]=row['honest bill']
    hit=part[part.attack.isin(ATT)].copy()
    vals=[]
    for _,r in hit.iterrows():
        key=tuple(r[k] for k in keys)
        ref=(href if r.attack in DATA_ATTACKS else allref).get(key,np.nan)
        bill=r[col] if r.attack in DATA_ATTACKS else r[prefix+'bill $/home/yr']
        # Original update logs' all-home bills are used directly.
        vals.append(bill-ref)
    hit['damage']=vals
    return hit
main=runs[runs.variant.eq('') & runs.share.eq('30') & runs.seed.le(4)]
ref=runs[runs.variant.eq('honest') & runs.share.eq('30')].copy()
aus=damage(main,reference=ref)
act=damage(runs[runs.variant.eq('act') & runs.share.eq('30') & runs.seed.le(2)])
transpart=main[main[TRANSFER+'bill $/home/yr'].notna()]
trans=damage(transpart,TRANSFER)
for name,part in [('ausgrid',aus),('act',act),('transfer',trans)]:
    assert part.damage.notna().all(),f'Missing damage: {name}'
    part[['method','seed','attack','damage','source']].to_csv(OUT/'tables'/f'{name}_damage_per_seed.csv',index=False)
    if name!='ausgrid':
        table(name+'_damage',f'Annual bill damage on {name.upper() if name=="act" else "ACT transfer"}. Mean $\\pm$ 95\\% interval half-width [seeds], dollars per household. Transfer coverage differs by method.',
              ['Method',*map(str.title,ATT)],[[LABEL[m],*[fmt(part.loc[part.method.eq(m)&part.attack.eq(a),'damage']) for a in ATT]] for m in DEFENCES])
# Coverage table is explicit rather than treating every archived run as an independent replicate.
table('coverage','Experimental coverage. The principal Ausgrid, ACT retraining and transfer comparisons contain six attacks and a no-attack reference per method and seed; sensitivity and clean-learning groups have separate designs.',
      ['Setting','Methods','Seeds per method','Households','Policy'],[
      ['Ausgrid main','10','0--4 (5)','36','Last'],['ACT retraining','10','0--2 (3)','36','Last'],
      ['ACT transfer, final variants','2','0--4 (5)','36','Last'],['ACT transfer, other methods','8','3--4 (2)','36','Last'],
      ['Share, fault, scaling, stress','Varies','Mostly 0 (1)','36 or 83','Last'],
      ['Clean learning','4 learned baselines','0--2 (3)','36','Validation-selected']])
# Three heatmaps share the same scale and print every value.
fig,axs=plt.subplots(1,3,figsize=(13,6),layout='constrained')
for ax,part,title in zip(axs,[aus,act,trans],['Ausgrid (5 seeds)','ACT retraining (3 seeds)','ACT transfer (2 or 5 seeds)']):
    mat=part.pivot_table(index='method',columns='attack',values='damage').reindex(index=DEFENCES,columns=ATT)
    im=ax.imshow(mat,cmap='RdYlBu_r',vmin=-100,vmax=800,aspect='auto')
    ax.set_xticks(range(6),[a.title() for a in ATT],rotation=55,ha='right')
    ax.set_yticks(range(10),[LABEL[m] for m in DEFENCES] if ax is axs[0] else [])
    for (i,j),v in np.ndenumerate(mat.to_numpy()): ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=7,color='white' if v>570 else 'black')
    ax.set_title(title)
fig.colorbar(im,ax=axs,label='Mean annual bill damage ($/home)',shrink=.75)
save('attack_damage',fig,pd.concat([p[['method','seed','attack','damage']].assign(setting=n) for n,p in [('Ausgrid',aus),('ACT',act),('Transfer',trans)]]))
# Rejection heatmaps: same seeds as main five-seed damage results.
reject=main[main.attack.isin(['none',*ATT])]
fig,axs=plt.subplots(1,2,figsize=(12,5.7),layout='constrained')
kept_methods=['krum','fltrust','flame','tafdrl','tafdrl final','fedgrid','fedgrid final']
for ax,col,title in zip(axs,['honest rejected %','bad rejected %'],['Honest updates','Compromised updates']):
    mat=reject.pivot_table(index='method',columns='attack',values=col).reindex(index=kept_methods,columns=['none',*ATT])
    im=ax.imshow(mat,cmap='magma_r',vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(range(7),['None',*map(str.title,ATT)],rotation=45,ha='right')
    ax.set_yticks(range(7),[LABEL[m] for m in kept_methods] if ax is axs[0] else [])
    for (i,j),v in np.ndenumerate(mat.to_numpy()):
        ax.text(j,i,'--' if not np.isfinite(v) else f'{v:.0f}',ha='center',va='center',fontsize=8,color='white' if v>55 else 'black')
    ax.set_title(title)
fig.colorbar(im,ax=axs,label='Logged rejection (%)',shrink=.75)
save('rejection_heatmap',fig,reject[['method','seed','attack','honest rejected %','bad rejected %']])
table('rejection','Logged honest update rejection on Ausgrid, nominal 30\\% compromise, seeds 0--4 where available. Entries are mean $\\pm$ 95\\% interval half-width [observed seeds]. Admission semantics differ across defences.',
      ['Method','None',*map(str.title,ATT)],[[LABEL[m],*[fmt(reject.loc[reject.method.eq(m)&reject.attack.eq(a),'honest rejected %']) for a in ['none',*ATT]]] for m in kept_methods])
# Paired final-minus-original changes: sign and sample count are unambiguous.
paired=[]
for setting,part in [('Ausgrid',aus),('ACT',act),('Transfer',trans)]:
    for a in ATT:
        left=part[part.method.eq('fedgrid final')&part.attack.eq(a)].set_index('seed').damage
        right=part[part.method.eq('fedgrid')&part.attack.eq(a)].set_index('seed').damage
        delta=(left-right).dropna()
        for seed,value in delta.items():paired.append(dict(setting=setting,attack=a,seed=seed,delta=value))
paired=pd.DataFrame(paired)
table('paired_revision','Paired change in attack damage, final minus original FedGrid, in dollars per household annually. Negative values favour the revision. Intervals are descriptive, unadjusted Student-t intervals; transfer uses only matched seeds 3--4.',
      ['Attack','Ausgrid [5]','ACT [3]','Transfer [2]'],[[a.title(),*[fmt(paired.loc[paired.setting.eq(s)&paired.attack.eq(a),'delta']) for s in ['Ausgrid','ACT','Transfer']]] for a in ATT])
fig,axs=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
for ax,s in zip(axs,['Ausgrid','ACT','Transfer']):
    for j,a in enumerate(ATT):
        v=paired.loc[paired.setting.eq(s)&paired.attack.eq(a),'delta'];m,h,n=interval(v)
        ax.errorbar(m,j,xerr=h,fmt='o',color='#b23b26',capsize=3)
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(6),[a.title() for a in ATT]);ax.invert_yaxis()
    ax.set_title(s);ax.set_xlabel('Final minus original damage ($/home/year)');ax.grid(axis='x',alpha=.2)
save('paired_revision',fig,paired)
# Clean energy outcomes. Baseline parsing permits absent old transformer entries.
metrics=['bill $/home/yr','curtailed kWh/home/yr','battery cycles/yr','battery charging in MSL half hours kWh','transformer peak loading %']
def baseline(log):
    rows=re.findall(r'^(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)$',(ROOT/'results'/log).read_text(),re.M)
    return pd.DataFrame({k:list(map(float,v)) for k,*v in rows},index=['Idle','Rule','MPC oracle'])
base=baseline('baselines.log')
clean=runs[runs.attack.eq('') & runs.days.eq('500') & runs.variant.eq('') & runs.seed.le(2)]
energy=[]
for method,row in base.iterrows():
    energy.append(dict(method=method,seed=-1,**{k:row.get(k,np.nan) for k in metrics}))
for _,r in clean.iterrows():energy.append(dict(method=LABEL[r.method],seed=r.seed,**{k:r[k] for k in metrics}))
energy=pd.DataFrame(energy);order=['Idle','Rule','Local SAC','FedAvg','FedProx','Central SAC','MPC oracle']
table('clean_energy','Clean 500-day learning with validation-selected policies, Ausgrid. Mean $\\pm$ 95\\% interval half-width [seeds]; deterministic baselines have one observation. MSL charging is a street-wide total; bills and curtailment are household averages. Cycles are total throughput divided by twice total fleet capacity.',
      ['Method','Bill (\\$/yr)','Curtailment (kWh/yr)','Cycles/yr','MSL charge (kWh)'],
      [[m,*[fmt(energy.loc[energy.method.eq(m),k]) for k in metrics[:4]]] for m in order])
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,k,label in zip(axs.flat,metrics[:4],['Annual bill ($/home)','Curtailed solar (kWh/home/year)','Fleet-equivalent cycles/year','Street charging in MSL periods (kWh/year)']):
    for j,m in enumerate(order):
        mean,half,n=interval(energy.loc[energy.method.eq(m),k]);ax.bar(j,mean,color='#235789' if m not in ['Rule','MPC oracle'] else '#7f8c8d')
        if n>1:ax.errorbar(j,mean,yerr=half,color='black',capsize=3)
    ax.set_xticks(range(len(order)),order,rotation=45,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
save('clean_energy',fig,energy)
# Shares: compare same share reference for share-aware methods, 30% otherwise.
sharepart=runs[runs.variant.eq('') & runs.seed.eq(0) & runs.share.isin(['10','20','30','40'])]
sharedata=[]
for m in DEFENCES:
    for sh in ['10','20','30','40']:
        hit=sharepart[sharepart.method.eq(m)&sharepart.share.eq(sh)&sharepart.attack.eq('normalized')]
        refsh=sh if m in ['trimmed','krum'] else '30'
        no=sharepart[sharepart.method.eq(m)&sharepart.share.eq(refsh)&sharepart.attack.eq('none')]
        if len(hit)==1 and len(no)==1:sharedata.append(dict(method=m,share=int(sh),damage=hit['bill $/home/yr'].iloc[0]-no['bill $/home/yr'].iloc[0]))
sharedata=pd.DataFrame(sharedata)
table('attack_share','Normalized-attack annual bill damage by nominal compromised share, seed 0 only. Trimmed mean and Krum use share-matched references; other methods use their 30\\% no-attack reference.',
      ['Method','10\\%','20\\%','30\\%','40\\%'],[[LABEL[m],*[fmt(sharedata.loc[sharedata.method.eq(m)&sharedata.share.eq(sh),'damage']) for sh in [10,20,30,40]]] for m in DEFENCES])
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,methods,title in zip(axs,[['fedavg','median','trimmed','krum'],['fltrust','flame','tafdrl final','fedgrid','fedgrid final']],['Coordinate and distance rules','Reference and trust rules']):
    for m in methods:
        d=sharedata[sharedata.method.eq(m)];ax.plot(d.share,d.damage,'o-',label=LABEL[m])
    ax.set_xticks([10,20,30,40]);ax.set_xlabel('Nominal compromised households (%)');ax.set_ylabel('Annual bill damage ($/home)');ax.set_title(title);ax.legend(fontsize=7);ax.grid(alpha=.2)
save('attack_share',fig,sharedata)
# Ablation uses all-home update damage with same variant and seed reference.
abpart=runs[runs.variant.str.startswith('without') & runs.method.eq('fedgrid') & runs.share.eq('30') & runs.seed.le(2)]
ab=damage(abpart)
ab=pd.concat([ab,aus[aus.method.eq('fedgrid') & aus.seed.le(2)].assign(variant='original'),aus[aus.method.eq('fedgrid final') & aus.seed.le(2)].assign(variant='final')])
ab=ab[ab.attack.eq('normalized')]
aborder=['original','without_clip','without_groups','without_trust','final']
table('ablation','Normalized-attack damage and logged honest rejection for the original FedGrid ablations and final variant, seeds 0--2. The final variant changes both filter order and clipping multiplier; it is not a one-factor ablation.',
      ['Variant','Damage (\\$/home/yr)','Honest rejection (\\%)'],[[esc(v),fmt(ab.loc[ab.variant.eq(v),'damage']),fmt(ab.loc[ab.variant.eq(v),'honest rejected %'])] for v in aborder])
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,col,title in zip(axs,['damage','honest rejected %'],['Annual bill damage ($/home)','Logged honest rejection (%)']):
    for j,v in enumerate(aborder):
        mean,half,n=interval(ab.loc[ab.variant.eq(v),col]);ax.errorbar(j,mean,yerr=half,fmt='o',capsize=4,color='#235789')
    ax.set_xticks(range(5),['Original','No clipping','No grouping','No trust','Final'],rotation=30,ha='right');ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
save('ablation',fig,ab[['method','seed','variant','attack','damage','honest rejected %']])
# Fault and stress conditions: their own references, all-home update damage.
senspart=runs[runs.variant.isin(['lost10_delayed10','feederO','pu102']) & runs.share.eq('30')]
sens=damage(senspart)
sens=sens[sens.attack.eq('normalized')]
table('sensitivity','Normalized-attack damage under message faults, feeder O and 1.02 pu supply, seed 0. A dash means unavailable, not zero. Damage is in dollars per household per year, using the same variant and method no-attack reference.',
      ['Method','Lost/delayed (\\$/home/yr)','Feeder O (\\$/home/yr)','1.02 pu (\\$/home/yr)'],[[LABEL[m],*[fmt(sens.loc[sens.method.eq(m)&sens.variant.eq(v),'damage']) for v in ['lost10_delayed10','feederO','pu102']]] for m in DEFENCES if m in set(sens.method)])
fig,axs=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
for ax,v,title in zip(axs,['lost10_delayed10','feederO','pu102'],['10% lost, 10% delayed','Feeder O: 83 households','Source voltage: 1.02 pu']):
    d=sens[sens.variant.eq(v)].sort_values('damage');ax.barh([LABEL[m] for m in d.method],d.damage,color=['#b23b26' if m=='fedgrid final' else '#235789' for m in d.method]);ax.set_title(title);ax.set_xlabel('Annual bill damage ($/home)');ax.grid(axis='x',alpha=.2)
save('sensitivity',fig,sens[['method','seed','variant','damage']])
# Candidate scatter is selection data, never independent test evidence.
candidates=candidate_tables('fedgrid',ROOT/'results')
versions=['original','filter_first','filter_first_clip2','filter_first_guard','filter_first_home_trust','without_clip']
ct=candidates.reindex(versions)
table('selection_candidates','Development-set selection of FedGrid variants. Damage is in dollars over the validation weeks and uses a bill-plus-curtailment score. The post-hoc 4.0\\% bound uses the rounded original rejection rate. Rejection pools differ: 21 original runs versus 15 per complete candidate. The last column is the mean of per-run maximum honest-household rejection, not the maximum across runs.',
      ['Variant','Worst validation damage','Honest rejection (\\%)','Mean run maximum (\\%)'],[[esc(v),*[f'{ct.loc[v,k]:.1f}' for k in ['worst','honest rejected %','worst honest home %']]] for v in versions])
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
for v,r in ct.iterrows():
    ax.scatter(r['honest rejected %'],r.worst,s=55,color='#b23b26' if v=='filter_first_clip2' else '#235789')
    offsets={'original':(5,5),'filter_first':(5,5),'filter_first_clip2':(5,-18),'without_clip':(5,5),'filter_first_guard':(5,5),'filter_first_home_trust':(-130,8)}
    ax.annotate(v.replace('_',' '),(r['honest rejected %'],r.worst),xytext=offsets[v],textcoords='offset points',fontsize=8)
ax.axvline(4,color='#777777',ls='--',label='Post-hoc eligibility bound (4.0%)');ax.set_xlim(0,37);ax.set_ylim(0,30)
ax.set_xlabel('Logged honest rejection (%)');ax.set_ylabel('Worst validation score damage ($/home)');ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2)
save('selection_candidates',fig,ct.reset_index())
# Validation trajectories: last-policy evaluation; shaded areas are seed ranges, not CIs.
curves=[]
for m in ['fedavg','fedgrid','fedgrid final','fltrust','flame','tafdrl final']:
    rows=main[main.method.eq(m)&main.seed.le(2)&main.attack.isin(['none','normalized','reward'])]
    for _,r in rows.iterrows():
        for day,score in re.findall(r'^day (\d+): validation ([\d.]+)',(ROOT/r.source).read_text(),re.M):
            curves.append(dict(method=m,seed=r.seed,attack=r.attack,day=int(day),score=float(score)))
curves=pd.DataFrame(curves)
fig,axs=plt.subplots(1,3,figsize=(13,4.2),sharey=True,layout='constrained')
for ax,a in zip(axs,['none','normalized','reward']):
    for m in ['fedavg','fedgrid','fedgrid final','fltrust','flame','tafdrl final']:
        d=curves[curves.method.eq(m)&curves.attack.eq(a)].groupby('day').score.agg(['mean','min','max'])
        line=ax.plot(d.index,d['mean'],label=LABEL[m],lw=1.4)[0];ax.fill_between(d.index,d['min'],d['max'],alpha=.08,color=line.get_color())
    ax.set_title(a.title());ax.set_xlabel('Training days');ax.grid(alpha=.2)
axs[0].set_ylabel('Validation score ($/home over validation weeks)');axs[-1].legend(fontsize=7)
save('training_curves',fig,curves)
# Diagram: architecture includes the trusted reference and simulator enforcement boundary.
fig,ax=plt.subplots(figsize=(12,5.3));ax.set_xlim(0,12);ax.set_ylim(0,5.3);ax.axis('off')
def box(x,y,w,h,text,color='#e8eef5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#42637f'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def arrow(p,q,label=''):
    ax.annotate('',q,p,arrowprops=dict(arrowstyle='->',color='#42637f',lw=1.5))
    if label:ax.text((p[0]+q[0])/2,(p[1]+q[1])/2+.12,label,ha='center',fontsize=8)
box(.2,.4,3,1.3,'Household data and replay\nLocal SAC + private output layers\n36 households; 48 steps/day')
box(.2,3.2,3,1.3,'Disjoint reference households\nCentral reference learner\nServer reference update u₀')
box(4.3,2.1,3.3,1.8,'Daily server aggregation\nSolar bands → robust filter\n→ norm clip → group trust\nShared actor + critic changes','#f6e6df')
box(8.6,2.1,3,1.8,'Battery actions + rule residual\nEnergy and export enforcement\nOpenDSS power flow\nRewards and grid metrics')
arrow((3.2,1.4),(4.3,2.5),'updates uᵢ');arrow((3.2,3.8),(4.3,3.4),'reference u₀')
arrow((7.6,3),(8.6,3),'shared model');arrow((10,.0+2.1),(3.2,.8),'observations and rewards')
ax.text(6,.05,'Simulation shares electrical state through the feeder; raw replay transitions are not aggregated.',ha='center',fontsize=8)
save('architecture',fig)
# Time partitions from the actual split function semantics, without importing simulator dependencies.
fig,axs=plt.subplots(2,1,figsize=(12,3.8),layout='constrained')
ausweeks=pd.date_range('2010-07-01',pd.Timestamp('2012-07-01')-pd.Timedelta(days=7),freq='7D')
codes=np.zeros(len(ausweeks)+53);codes[np.arange(len(ausweeks))[4::8]]=1;codes[len(ausweeks):]=2
from matplotlib.colors import ListedColormap
cmap=ListedColormap(['#b4cce2','#e4b363','#bd5144'])
axs[0].imshow(codes[None,:],aspect='auto',cmap=cmap,vmin=0,vmax=2);axs[0].set_title('Ausgrid: training and validation blocks, then full evaluation year')
axs[0].set_xticks([0,len(ausweeks)-1,len(codes)-1],['July 2010','June 2012','June 2013']);axs[0].set_yticks([])
codes2=np.zeros(52);codes2[1::8]=1;codes2[3::4]=2
axs[1].imshow(codes2[None,:],aspect='auto',cmap=cmap,vmin=0,vmax=2);axs[1].set_title('ACT 2018: 32 training, 7 validation and 13 evaluation weeks');axs[1].set_yticks([]);axs[1].set_xlabel('Week index (zero-based)');axs[1].set_xticks([0,13,26,39,51])
fig.text(.5,-.03,'Blue: training   |   Gold: validation   |   Red: evaluation. Ausgrid final year is schematic; ACT cells are exact weeks.',ha='center',fontsize=9)
save('data_splits',fig,pd.DataFrame({'week':np.arange(52),'act_partition':np.array(['train','validation','test'])[codes2.astype(int)]}))
# Numerical configuration and threats make assumptions inspectable.
table('configuration','Implemented configuration for the principal case. Tariffs are fixed experimental inputs; MSL denotes minimum system load.',
 ['Parameter','Value','Parameter','Value'],[
 ['Main/scaling households','36 / 83','Time step','0.5 h'],['Solar scale / cap','6.6 / 10 kWp','Battery capacity','3 kWh per kWp'],
 ['Battery power','5 kW','Round-trip efficiency','0.90'],['Source voltage','1.00 pu; stress 1.02','Voltage bounds','216.2--253 V'],
 ['Export cap / ordinary floor','10 / 1.5 kW','MSL export cap','0 kW'],['Demand trigger','3660 MW (scenario)','Voltage margin','1 V'],
 ['Import: 11:00--16:00','0.22680 \\$/kWh','Import: 16:00--21:00','0.45712 \\$/kWh'],
 ['Import: other hours','0.28035 \\$/kWh','Export credit','0.08660 \\$/kWh'],['Daily supply charge','1.48834 \\$/day','Curtailment coefficient','0.08660 \\$/kWh'],
 ['Hidden width / batch','128 / 128','Learning rate','0.0003'],['Discount / target rate','0.99 / 0.005','Warmup','5 days'],
 ['Training / aggregation','500 days / daily','Shared payload','54,400 float32 values'],['Solar bands / robust threshold','3 / 2.5','Final clipping bound','2 median norms']])
table('threats','Implemented perturbations. Compromised households are fixed within a run; message faults affect honest households.',
 ['Threat','Perturbation','Knowledge or scope'],[
 ['Flip','Negate model changes','Own update'],['Scale','Multiply changes by household count','Own update; system size'],
 ['Noise','Gaussian update variance 1000','Parameter dimension'],['Normalized','Direction and magnitude grid search','All updates; aggregation queries'],
 ['Meter','Swap observed load and solar channels','Local training observations'],['Reward','Reverse local reward sign','Local training rewards'],
 ['Message faults','10\\% drop and 10\\% one-round delay','Honest changes; separate scenario']])
table('comparators','Comparator scope and admission semantics. Source algorithms are adapted to shared SAC parameter changes; original guarantees are not transferred to this setting.',
 ['Method','Aggregation principle','Whole-update rejection statistic'],[
 ['FedAvg / FedProx','Mean / mean with local proximal term','Not defined'],['Median / trimmed mean','Coordinate statistics','Not defined'],
 ['Krum','Select one neighbour-consistent update','Unselected updates'],['FLTrust','Positive cosine trust; norm rescaling','Zero-trust updates'],
 ['FLAME','Majority clustering, clipping and noise','Outside admitted cluster'],['TAFDRL-style','PCA distance and Beta trust','Distance or minimum-trust exclusion'],
 ['FedGrid','Within-band filter and group trust','Filter exclusion; zero-weight groups omitted']])
# Device measurements are archived repeated trials on one laptop, not embedded deployment.
table('device','Archived single-core laptop measurements (Intel i7-10610U). Three repeated timing trials on one device; these are not three independent hardware platforms.',
 ['Trial','Decision (ms)','Gradient step (ms)','Peak process memory (MiB)'],[
 ['1','0.06','4.4','353'],['2','0.04','4.1','356'],['3','0.05','4.2','356']])
# Electrical outcomes use no-attack last policies on the respective evaluation sets.
gridrows=[]
gridcols=['half hours over 253 V %','max V','transformer peak loading %']
for setting,part in [('Ausgrid',main),('ACT',runs[runs.variant.eq('act') & runs.share.eq('30') & runs.seed.le(2)])]:
    none=part[part.attack.eq('none')]
    for _,row in none.iterrows():
        gridrows.append(dict(setting=setting,method=row.method,seed=row.seed,**{c:row.get(c,np.nan) for c in gridcols}))
grid=pd.DataFrame(gridrows)
table('electrical','Electrical outcomes in no-attack last-policy runs: means and 95\\% interval half-widths [observed seeds]. Voltage-exceedance percentages use three decimals to retain small nonzero means calculated from two-decimal logged input rates. Run maxima and peaks are computed before seed averaging; they are not safety bounds.',
      ['Setting','Method','Over 253 V (\%)','Run maximum (V)','Transformer peak (\%)'],
      [[setting,LABEL[m],*[fmt(grid.loc[grid.setting.eq(setting)&grid.method.eq(m),c],3 if c=='half hours over 253 V %' else 1) for c in gridcols]]
       for setting in ['Ausgrid','ACT'] for m in ['fedavg','fltrust','flame','tafdrl final','fedgrid','fedgrid final']])
fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
chosen=['fedavg','fltrust','flame','tafdrl final','fedgrid','fedgrid final']
for ax,setting in zip(axs,['Ausgrid','ACT']):
    for j,m in enumerate(chosen):
        mean,half,n=interval(grid.loc[grid.setting.eq(setting)&grid.method.eq(m),'transformer peak loading %'])
        ax.errorbar(j,mean,yerr=half,fmt='o',capsize=3,color='#b23b26' if m=='fedgrid final' else '#235789')
    ax.axhline(100,color='#777777',ls='--',label='Nameplate rating');ax.set_title(setting+'; no-attack last policy')
    ax.set_xticks(range(6),[LABEL[m] for m in chosen],rotation=40,ha='right');ax.set_ylabel('Mean run-level transformer peak (%)');ax.grid(axis='y',alpha=.2)
axs[1].legend(fontsize=8)
save('electrical',fig,grid)
# Source manifest includes candidate files and baseline files used by these artifacts.
extra=[*paths,*sorted((ROOT/'results/candidates').glob('*.log')),
       ROOT/'results/baselines.log',ROOT/'results/baselines_act.log',ROOT/'results/device_single_core.log']
extra += [ROOT/'results/baselines_pu102.log',ROOT/'results/diagnostics/README.md',*sorted((ROOT/'results/candidates').glob('choice_*.txt')),*sorted((ROOT/'results/candidates').glob('*.flags'))]
with (OUT/'tables/expanded_sources.sha256').open('w') as f:
    for p in sorted(set(extra)):f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n')
(OUT/'artifact_inventory.json').write_text(json.dumps({'figures':['damage_rejection',*figs], 'tables':['main_damage','algorithm (static in methods)',*tabs,'imputation_audit','exact_statistics','extended_clean','all_baselines','attack_electrical','heterogeneity','selection_settings'],
    'notes':'13 figures and 24 tables after the statistical, imputation and evidence audits. Scientific blockers remain.'},indent=2)+'\n')
print(f'Rebuilt {len(figs)} figures and {len(tabs)} tables; run the audit scripts to complete the artifact inventory.')
