"""Exploratory exact paired Wilcoxon tests on archived Ausgrid damage.

Families follow the 8 Nov plan's grouping of damage comparisons, but applying
these tests now does not retrospectively preregister this five-seed analysis.
No rejection tests are made across incompatible admission definitions.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
OUT=Path(__file__).resolve().parents[1]/'tables'
data=pd.read_csv(OUT/'ausgrid_damage_per_seed.csv')
attacks=['flip','scale','noise','normalized','meter','reward']
final=data[data.method.eq('fedgrid final')].set_index(['attack','seed']).damage
rows=[]; observations=[]
for comparator in ['fltrust','flame','tafdrl final','fedgrid']:
    other=data[data.method.eq(comparator)].set_index(['attack','seed']).damage
    for attack in attacks:
        delta=(final-other).loc[attack].dropna().round(8)
        assert len(delta)==5 and delta.index.nunique()==5
        # Exact signed-rank enumeration is used with no zeros or tied absolute ranks.
        assert delta.ne(0).all() and delta.abs().nunique()==len(delta)
        result=wilcoxon(delta,alternative='two-sided',method='exact')
        family='revision' if comparator=='fedgrid' else 'robust_comparators'
        rows.append(dict(family=family,comparator=comparator,attack=attack,n=len(delta),
            mean_difference=float(delta.mean()),median_difference=float(delta.median()),
            wilcoxon_statistic=float(result.statistic),p_exact=float(result.pvalue)))
        observations += [dict(comparator=comparator,attack=attack,seed=int(s),difference=float(d)) for s,d in delta.items()]
results=pd.DataFrame(rows)
for family,part in results.groupby('family'):
    order=part.sort_values('p_exact').index.to_numpy()
    sorted_p=results.loc[order,'p_exact'].to_numpy()
    adjusted=np.maximum.accumulate(sorted_p*np.arange(len(order),0,-1)).clip(max=1)
    results.loc[order,'p_holm']=adjusted
results['reject_holm_005']=results.p_holm.lt(.05)
results.to_csv(OUT/'paired_exact_tests.csv',index=False)
pd.DataFrame(observations).to_csv(OUT/'paired_exact_observations.csv',index=False)
# Enumerate the null distribution to state achievable resolution, not power.
signs=np.array(list(itertools.product([-1,1],repeat=5)))
assert len(signs)==32
print('Exact two-sided p-value minimum for five nonzero pairs:',2/len(signs))
print(results[['family','comparator','attack','p_exact','p_holm']].to_string(index=False))
labels={'fltrust':'FLTrust','flame':'FLAME','tafdrl final':'TAFDRL-style tuned','fedgrid':'FedGrid original'}
lines=[r'\begin{table}[htbp]',r'\centering\small',
 r'\caption{Exploratory two-sided exact paired Wilcoxon tests on five Ausgrid seeds. Differences are final FedGrid minus the comparator. Holm adjustment uses 18 tests for robust comparators and six for the original-method revision; no adjusted test rejects at 0.05.}',
 r'\label{tab:exact_statistics}',r'\begin{tabular}{llrrr}',r'\toprule',
 r'Comparator & Attack & Mean difference (\$/yr) & Exact $p$ & Holm $p$ \\',r'\midrule']
for row in results.itertuples():
 lines.append(f'{labels[row.comparator]} & {row.attack.title()} & {row.mean_difference:.1f} & {row.p_exact:.4f} & {row.p_holm:.3f}'+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(OUT/'exact_statistics.tex').write_text('\n'.join(lines)+'\n')
