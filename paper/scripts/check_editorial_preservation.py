"""Validate the academic edit against its original source backup."""
from pathlib import Path
from collections import Counter
import json,re,tarfile
ROOT=Path(__file__).resolve().parents[1]
backup=Path(json.loads((ROOT/'tables/editorial_backup.json').read_text())['backup_directory'])
before=(backup/'main_before_edit.tex').read_text();after=(ROOT/'main.tex').read_text()
patterns={'headings':r'\\(?:section|subsection|paragraph)\*?\{[^}]+\}',
 'labels':r'\\label\{[^}]+\}','citations':r'\\cite\{[^}]+\}',
 'references':r'\\ref\{[^}]+\}',
 'figures':r'\\includegraphics(?:\[[^]]*\])?\{[^}]+\}'}
report={}
for name,pattern in patterns.items():
    old=re.findall(pattern,before);new=re.findall(pattern,after)
    ok=old==new if name in ['headings','labels','figures'] else Counter(old)==Counter(new)
    assert ok,name
    report[name+'_preserved']=True
pattern=r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}'
assert re.findall(pattern,before,re.S)==re.findall(pattern,after,re.S)
report['display_equations_preserved']=True
edited=[]
with tarfile.open(backup/'academic_edit_sources_before.tar.gz') as archive:
    for member in archive.getmembers():
        if member.name.startswith('paper/tables/') and member.name.endswith('.tex'):
            old=archive.extractfile(member).read().decode()
            new=(ROOT.parent/member.name).read_text()
            def values(text):
                body=text.split('\\toprule',1)[1].split('\\bottomrule',1)[0]
                return re.findall(r'(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?',body)
            assert values(old)==values(new),member.name
            if old!=new:edited.append(member.name)
    assert archive.extractfile('paper/references.tex').read()==(ROOT/'references.tex').read_bytes()
report['bibliography_unchanged']=True
report['all_table_numerical_rows_preserved']=True
report['edited_table_captions_or_units']=edited
report['explicit_corrections']=[
 'Filter MAD-floor placement aligned with frozen code; no experiment changed.',
 'Fleet-equivalent cycle definition clarified; all values unchanged.',
 'Linux memory unit corrected from MB to MiB; values unchanged.',
 'Five-day MPC replay scope and learner-only meter poisoning clarified.']
(ROOT/'tables/editorial_preservation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Editorial preservation checks passed: structure, citations, equations, bibliography and numerical table rows.')
