"""Check manuscript integrity and audit invariants. Does not certify research quality."""
from pathlib import Path
import json,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
seen=[]
def expand(path,stack=()):
    if path in stack:raise ValueError(f'Circular input: {path}')
    seen.append(path)
    text=path.read_text()
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(ROOT/(m[1]+'.tex'),stack+(path,)),text)
text=expand(ROOT/'manuscript_template.tex')
from assemble_manuscript import assemble
assert (ROOT/'main.tex').read_text()==assemble(),'main.tex is stale; run assemble_manuscript.py'
assert len(seen)==len(set(seen)),'A source file is included more than once'
labels=re.findall(r'\\label\{([^}]+)\}',text)
assert len(labels)==len(set(labels)),'Duplicate labels'
for name in re.findall(r'\\ref\{([^}]+)\}',text):assert name in labels,name
bib=re.findall(r'\\bibitem\{([^}]+)\}',text)
for names in re.findall(r'\\cite\{([^}]+)\}',text):
    for name in names.split(','):assert name in bib,name
figpaths=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text)
assert len(figpaths)==len(set(figpaths)),'Repeated figure'
for name in figpaths:assert (ROOT/name).exists(),name
for p in (ROOT/'tables').glob('*.tex'):assert p in seen,f'Unused table {p.name}'
for name,n in [('ausgrid',300),('act',180),('transfer',156)]:
    d=pd.read_csv(ROOT/'tables'/f'{name}_damage_per_seed.csv')
    assert len(d)==n and d.damage.notna().all()
    assert not d.duplicated(['method','seed','attack']).any()
stats=pd.read_csv(ROOT/'tables/paired_exact_tests.csv')
assert len(stats)==24 and not stats.reject_holm_005.any()
assert stats.groupby('family').size().to_dict()=={'revision':6,'robust_comparators':18}
assert stats.p_exact.ge(.0625).all()
audit=pd.read_csv(ROOT/'tables/act_imputation_summary.csv')
street=audit[audit.scope.eq('evaluation_street')&audit.partition.eq('train')].iloc[0]
assert street.homes==36 and street.donor_from_test==1540 and street.cross_partition_cells==2004
highlights=(ROOT/'highlights.txt').read_text().splitlines()
assert 3<=len(highlights)<=5 and all(len(h)<=85 for h in highlights)
maintext=text.split(r'\appendix')[0]
report=dict(figures=len(figpaths),tables=text.count(r'\begin{table}'),bibliography_entries=len(bib),
            main_figures=maintext.count(r'\begin{figure}'),main_tables=maintext.count(r'\begin{table}'),
            exact_comparisons=len(stats),cross_partition_training_cells=int(street.cross_partition_cells),
            test_donors_to_training=int(street.donor_from_test),source_checks='passed',
            quality_verdict='Scientific blockers remain; checks do not certify journal acceptance.')
(ROOT/'tables/manuscript_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
