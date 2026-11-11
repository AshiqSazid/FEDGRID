"""Read-only audit of ACT imputation donors in the frozen preprocessing.

Reconstructs missing half-hours from raw profiles, identifies the first donor
selected by the implemented nearest-day rule, and checks partition crossings.
Does not modify data, splits, or experiment logs; writes audit outputs in paper/.
"""
from pathlib import Path
import sys, hashlib, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from prep.nextgen import GRID, FILL_DAYS, read_home, half_hours
OUT=ROOT/'paper/tables'
customers=pd.read_csv(ROOT/'data/processed/nextgen_customers.csv',index_col=0)
clean=customers.index[customers.clean]
street=np.random.default_rng(0).choice(clean,36,replace=False)
refclean=clean[~clean.isin(street)]
reference=np.random.default_rng(0).choice(refclean,36,replace=False)
week=np.arange(len(GRID))//(7*48)
partition=np.full(len(GRID),'train',dtype=object)
partition[np.isin(week,np.arange(52)[1::8])]='validation'
partition[np.isin(week,np.arange(52)[3::4])]='test'
records=[]; hashes=[]
for path in sorted((ROOT/'data/raw/nextgen').glob('*.csv')):
    home=int(path.stem.split('id_')[1])
    raw=read_home(path)
    cols={'load':half_hours(raw['load power (kW)']),
          'solar':half_hours((-raw['solar power (kW)']).clip(lower=0))}
    scope='street' if home in street else ('reference' if home in reference else 'other')
    for field,series in cols.items():
        values=series.to_numpy(); missing=np.flatnonzero(~np.isfinite(values))
        unresolved=missing.copy()
        donors={}
        for k in range(1,FILL_DAYS+1):
            for offset in [-k*48,k*48]:
                target=unresolved+offset
                valid=(target>=0)&(target<len(values))
                use=np.zeros(len(target),dtype=bool)
                use[valid]=np.isfinite(values[target[valid]])
                donors.update(zip(unresolved[use],target[use]))
                unresolved=unresolved[~use]
        if len(unresolved): raise ValueError(f'Unfilled cells: {home}/{field}')
        # Confirm donor reconstruction against the processed table for this field.
        proc=ROOT/'data/processed'/('nextgen_load_kwh.csv.gz' if field=='load' else 'nextgen_pv_kwh_per_kwp.csv.gz')
        processed=pd.read_csv(proc,usecols=[str(home)])[str(home)].to_numpy()
        expected=np.array([values[donors[t]] for t in missing])
        if field=='solar':expected=expected/customers.loc[home,'capacity_kwp']
        assert np.allclose(processed[missing],expected,atol=5.1e-4 if field=='load' else 5.1e-5), (home,field)
        for t,donor in donors.items():
            records.append(dict(home=home,eligible=bool(home in clean),scope=scope,field=field,
             recipient=str(GRID[t]),donor=str(GRID[donor]),recipient_partition=partition[t],
             donor_partition=partition[donor],cross_partition=partition[t]!=partition[donor],future_donor=donor>t))
    hashes.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(ROOT)))
records=pd.DataFrame(records)
records.to_csv(OUT/'act_imputation_donors.csv',index=False)
summary=[]
for label,selected in [('all_raw_homes',records),('eligible_homes',records[records.eligible]),
 ('evaluation_street',records[records.scope.eq('street')]),('reference_street',records[records.scope.eq('reference')])]:
    homes={'all_raw_homes':len(hashes),'eligible_homes':len(clean),'evaluation_street':len(street),'reference_street':len(reference)}[label]
    for target in ['train','validation','test']:
        d=selected[selected.recipient_partition.eq(target)]
        target_n=int((partition==target).sum())*homes*2
        summary.append(dict(scope=label,partition=target,homes=homes,total_profile_cells=target_n,
            filled_cells=len(d),cross_partition_cells=int(d.cross_partition.sum()),future_donor_cells=int(d.future_donor.sum()),
            donor_from_test=int(d.donor_partition.eq('test').sum()),
            percent_profile_cells_crossing=100*d.cross_partition.sum()/target_n))
summary=pd.DataFrame(summary)
summary.to_csv(OUT/'act_imputation_summary.csv',index=False)
(OUT/'act_imputation_raw_sources.sha256').write_text('\n'.join(hashes)+'\n')
print(summary[summary.scope.isin(['evaluation_street','reference_street'])].to_string(index=False))
lines=[r'\begin{table}[htbp]',r'\centering\small',
 r'\caption{Read-only ACT imputation donor audit for the 36-home evaluation and reference streets. Counts pool load and solar half-hour cells. Cross-partition percentages use all profile cells in the recipient partition, not only filled cells. Copied evaluation data are a qualification of the existing evidence; no policy rerun is implied.}',
 r'\label{tab:imputation_audit}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{llrrrr}',r'\toprule',
 r'Street & Recipient partition & Filled cells & Cross-partition cells & Cross-partition (\%) & Donors from test \\',r'\midrule']
for row in summary[summary.scope.isin(['evaluation_street','reference_street'])].itertuples():
 label='Evaluation' if row.scope=='evaluation_street' else 'Reference'
 lines.append(f'{label} & {row.partition} & {row.filled_cells:,} & {row.cross_partition_cells:,} & {row.percent_profile_cells_crossing:.3f} & {row.donor_from_test:,}'+r' \\')
lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
(OUT/'imputation_audit.tex').write_text('\n'.join(lines)+'\n')
