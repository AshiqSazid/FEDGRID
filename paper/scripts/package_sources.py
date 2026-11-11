"""Package an Overleaf working project and a flat Elsevier source archive."""
from pathlib import Path
import re,zipfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def expand(path,stack=()):
    if path in stack:raise ValueError('Circular input')
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(ROOT/(m[1]+'.tex'),stack+(path,)),path.read_text())
text=expand(ROOT/'main.tex')
figurepaths=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text)
for name in figurepaths:text=text.replace('{'+name+'}','{fig_'+Path(name).name+'}')
with zipfile.ZipFile(ROOT/'energy_ai_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('manuscript.tex',text)
    for name in figurepaths:z.write(ROOT/name,'fig_'+Path(name).name)
    z.write(ROOT/'highlights.txt','highlights.txt')
    z.writestr('READ_FIRST.txt','Working Energy and AI draft. Scientific blockers and author declarations remain. Compile manuscript.tex with the standard elsarticle class. This archive is prepared for review, not approval to submit.\n')
with (ROOT/'tables/source_code.sha256').open('w') as f:
    files=[*(ROOT.parent/'src').rglob('*.py'),ROOT.parent/'run_experiments.sh',ROOT.parent/'requirements-lock.txt',*ROOT.glob('scripts/*.py')]
    for p in sorted(files):f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT.parent))+'\n')
with zipfile.ZipFile(ROOT/'overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and not any(part in ['__pycache__','build','.venv'] for part in p.relative_to(ROOT).parts) and p.suffix in ['.tex','.pdf','.png','.md','.csv','.sha256','.py','.txt','.json']:
            z.write(p,p.relative_to(ROOT))
print('Updated Overleaf project and flat Energy and AI source archives.')
