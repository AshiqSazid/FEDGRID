# FedGrid-RL — Energy and AI working manuscript

`main.tex` uses the standard Elsevier `elsarticle` preprint class and targets **Energy and AI**. The paper is **not yet ready to submit**; see `SUBMISSION_READINESS.md` for evidence and closure criteria.

There are **13 figures, 19 tables and 19 bibliography entries**. The main argument contains six figures and six tables; the appendix preserves the detailed numerical tables, sensitivity analyses and two new audits. The explicit statistical and preprocessing caveats are part of the results, not text to remove for a stronger-looking submission.

## Files

- `main.tex`: complete, readable manuscript text in the standard Elsevier class. Figures remain external.
- `main.pdf`: current compiled manuscript.
- `manuscript_template.tex`: editable document structure; section and table inputs are assembled into `main.tex`.
- `build/main.pdf`: locally compiled 30-page review PDF.
- `overleaf.zip`: working source project, analysis scripts, tables, figures and review records.
- `energy_ai_sources.zip`: flat journal-source layout; compile `manuscript.tex`. Preparing this file does not authorise submission.
- `highlights.txt`: five short research highlights, each <=85 characters.
- `sections/`: manuscript, appendix and draft declarations requiring author confirmation.
- `figures/`: all vector PDF and preview PNG figures.
- `tables/`: per-seed results, imputation donors, exact statistical comparisons, provenance and validation reports.
- `SOURCE_REVIEW.md`: primary-source checking and unresolved provenance.
- `EDITORIAL_REVIEW.md`: detailed section-by-section academic assessment, changes, verified corrections and remaining concerns.
- `tables/editorial_backup.json`: location of the pre-edit manuscript backup.
- `tables/editorial_preservation.json`: checks against that backup.
- `SUBMISSION_READINESS.md`: current scientific verdict and specific next evidence required.

## Rebuild reproducible artifacts

Use an independent environment with `requirements-figures.txt`, from the repository root:

```bash
python paper/scripts/build_artifacts.py
python paper/scripts/expand_artifacts.py
python paper/scripts/audit_statistics.py
python paper/scripts/audit_preprocessing.py
python paper/scripts/assemble_manuscript.py
python paper/scripts/check_manuscript.py
python paper/scripts/check_editorial_preservation.py
python paper/scripts/package_sources.py
```

The preprocessing audit reads the raw ACT files and checks copies against the processed tables; it changes neither. The statistical audit uses the existing five-seed Ausgrid damage results. No policy training, algorithm tuning or result-log rewrite is performed by these scripts.

The session's plotting environment is `/tmp/fedgrid-paper-figures`; it may not persist. It links the existing research environment for pandas/scipy and installs matplotlib separately. Temporary LaTeX tooling is also kept outside the research environment. The final manuscript compiled successfully with Tectonic 0.15.0 to `build/main.pdf` (30 A4 pages). No undefined references, missing citations or overfull boxes remain in the final log; two underfull bibliography paragraphs remain, without overflowing text. Title, main results and audit-table pages were visually inspected. This production check does not resolve the scientific blockers.

## Reproducibility and interpretation

CSV data and hashes travel with figures. Missing statistics remain missing; negative damage is retained; single-seed sensitivities are labelled. The trade-off uses method-specific logged rejection, not a uniform participation score. ACT imputation overlap and the lack of exact statistical rejection are disclosed.

Journal-specific guide requirements still require direct author verification because the journal guide was unavailable to the browser during this audit. Author names, affiliations, contributions, funding/conflicts, licences and final responsibility statements remain unconfirmed.

## Editing the paper

The complete paper is now present directly in `main.tex`. To preserve the editable source organisation, make ongoing changes in `sections/`, `tables/` and `manuscript_template.tex`, then run `scripts/assemble_manuscript.py`. Direct edits to the generated `main.tex` must also be copied to the relevant source before reassembly.
