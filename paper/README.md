# FedGrid-RL — Energy and AI working manuscript

`main.tex` uses the standard Elsevier `elsarticle` preprint class and targets **Energy and AI**. The paper is **not yet ready to submit**; see `SUBMISSION_READINESS.md` for evidence and closure criteria.

There are **13 figures, 24 tables and 19 bibliography entries**. The main argument contains six figures and six tables; the appendix preserves detailed numerical tables, sensitivity analyses and statistical, preprocessing and code-to-results audits. The explicit caveats are part of the results, not text to remove for a stronger-looking submission.

## Files

- `main.tex`: complete, readable manuscript text in the standard Elsevier class. Figures remain external.
- `main.pdf`: current compiled manuscript.
- `manuscript_template.tex`: editable document structure; section and table inputs are assembled into `main.tex`.
- `build/main.pdf`: locally compiled review PDF.
- `overleaf.zip`: working source project, analysis scripts, tables, figures and review records.
- `energy_ai_sources.zip`: flat journal-source layout; compile `manuscript.tex`. Preparing this file does not authorise submission.
- `highlights.txt`: five short research highlights, each <=85 characters.
- `sections/`: manuscript, appendix and draft declarations requiring author confirmation.
- `figures/`: all vector PDF and preview PNG figures.
- `tables/`: per-seed results, imputation donors, exact statistical comparisons, provenance and validation reports.
- `SOURCE_REVIEW.md`: primary-source checking and unresolved provenance.
- `EDITORIAL_REVIEW.md`: detailed section-by-section academic assessment, changes, verified corrections and remaining concerns.
- `CODE_RESULTS_AUDIT.md`: later independent check against implementation and archived results, with reporting corrections and coverage reconciliation.
- `tables/editorial_backup.json`: location of the pre-edit manuscript backup.
- `tables/editorial_preservation.json`: historical checks at the earlier academic-edit checkpoint; the later evidence audit intentionally adds content and corrects voltage precision.
- `SUBMISSION_READINESS.md`: current scientific verdict and specific next evidence required.

## Rebuild reproducible artifacts

Use an independent environment with `requirements-figures.txt`, from the repository root:

```bash
python paper/scripts/build_artifacts.py
python paper/scripts/expand_artifacts.py
python paper/scripts/audit_statistics.py
python paper/scripts/audit_preprocessing.py
python paper/scripts/audit_evidence.py --write-tables
python paper/scripts/assemble_manuscript.py
python paper/scripts/check_manuscript.py
python paper/scripts/package_sources.py
```

The preprocessing audit reads the raw ACT files and checks copies against the processed tables; it changes neither. The statistical audit uses the existing five-seed Ausgrid damage results. The independent evidence audit traces all 636 principal attack-damage observations, 80 no-attack electrical observations, 350 rejection observations, 60 paired changes and 24 exact tests to archived logs. Its added tables cover extended clean learning, deterministic baselines, attack-time electrical outcomes, household rejection and candidate settings. No policy training, algorithm tuning or result-log rewrite is performed by these scripts.

`check_editorial_preservation.py` is a historical, local-only check requiring the original ignored backup. It is not part of a fresh-clone rebuild: the later evidence audit intentionally changes table precision and adds five appendix subsections. `check_manuscript.py` and `audit_evidence.py` validate the current manuscript and evidence instead.

The session's plotting environment is `/tmp/fedgrid-paper-figures`; it may not persist. It links the existing research environment for pandas/scipy and installs matplotlib separately. Temporary LaTeX tooling is also kept outside the research environment. Build and visual-check results for the current evidence-audit revision are recorded in `CODE_RESULTS_AUDIT.md`. A production check does not resolve the scientific blockers.

## Reproducibility and interpretation

CSV data and hashes travel with figures. Missing statistics remain missing; negative damage is retained; single-seed sensitivities are labelled. The trade-off uses method-specific logged rejection, not a uniform participation score. ACT imputation overlap and the lack of exact statistical rejection are disclosed.

Journal-specific guide requirements still require direct author verification because the journal guide was unavailable to the browser during this audit. Author names, affiliations, contributions, funding/conflicts, licences and final responsibility statements remain unconfirmed.

## Editing the paper

The complete paper is now present directly in `main.tex`. To preserve the editable source organisation, make ongoing changes in `sections/`, `tables/` and `manuscript_template.tex`, then run `scripts/assemble_manuscript.py`. Direct edits to the generated `main.tex` must also be copied to the relevant source before reassembly.
