"""Assemble the complete readable manuscript into main.tex.

Edit manuscript_template.tex for document structure and sections/*.tex or
individual table sources for content, then run this script before compiling.
Figure assets remain external. Inputs are expanded with source comments.
"""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
HEADER='% Complete manuscript assembled by scripts/assemble_manuscript.py.\n% Editable sources: manuscript_template.tex, sections/, and tables/.\n'
def expand(path,stack=()):
    if path in stack:raise ValueError(f'Circular LaTeX input: {path}')
    def include(match):
        child=ROOT/(match[1]+'.tex')
        return ('\n% BEGIN SOURCE: '+str(child.relative_to(ROOT))+'\n'+
                expand(child,stack+(path,))+'\n% END SOURCE: '+str(child.relative_to(ROOT))+'\n')
    return re.sub(r'\\input\{([^}]+)\}',include,path.read_text())
def assemble():return HEADER+expand(ROOT/'manuscript_template.tex')
if __name__=='__main__':
    text=assemble();(ROOT/'main.tex').write_text(text)
    print(f'Wrote complete main.tex: {len(text.splitlines())} lines, {len(text):,} characters.')
