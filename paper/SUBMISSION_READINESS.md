# Energy and AI submission audit — 11 November 2026

**Verdict: not ready to submit yet.** The manuscript has a clearer Energy and AI presentation and reproducible evidence, but its central scientific gaps cannot be repaired by wording, references or extra displays. This is an internal research assessment, not a prediction of an editorial decision or confirmation of a journal quartile.

## Evidence that is currently defensible

- Final FedGrid has a lower **observed mean** Normalized-attack bill damage than original FedGrid on the archived Ausgrid and ACT runs. Paired intervals are descriptive.
- Clean learning substantially reduces curtailed solar compared with the implemented self-consumption rule. It does not show a decisive general bill advantage over that rule.
- Logged outlier rejection is low for FedGrid under the implemented definition. Positive effective contribution has not been measured consistently across methods.
- The aggregate norm is bounded by twice the median received-update norm. This follows from clipping and nonnegative weights and is not an attack-resilience or grid-safety theorem.

## Material blockers and concrete closure criteria

| Priority | Finding | Existing evidence | What closes it |
| --- | --- | --- | --- |
| 1 | The participation claim lacks a common effective-exclusion measure | FedGrid can admit a group but give it zero weight; FLTrust logs that exclusion | Log filter decisions and realised positive coefficients separately for all comparable methods; retain the existing methods and report per-home outcomes on a fixed evaluation protocol |
| 1 | ACT preprocessing crosses partitions | Reconstructed donors: 2,004 cross-partition training cells in the 36-home street, of which 1,540 use evaluation-week donors; same test-donor count in reference street | Audit and rerun under a documented partition-isolated preprocessing rule, preserving old data and logs. Do not tune controller choices against the new outcome |
| 1 | Existing exact statistical comparisons are inconclusive | All 24 two-sided five-seed exact Wilcoxon comparisons have unadjusted p >= 0.0625; no Holm-adjusted rejection | Use a fixed, justified replication and analysis plan, independent evaluation where feasible, and report all resulting seeds. Ten seeds are not an automatic guarantee of significance or validity |
| 1 | Strongest novelty and mechanism claims remain unestablished | Final revision changes order and clipping multiplier; grouping benefit is not independently established | Complete a structured current-literature comparison and identify what is new; isolate mechanism interactions with a fixed design or narrow the claim to an empirical evaluation |
| 2 | Energy comparison uses an optimistic MPC and incomplete costs | Perfect daily foresight, ideal-information envelopes, no monetary degradation | Add a fixed forecast MPC and degradation sensitivity under matched information; report all controller and grid outcomes |
| 2 | Comparator fidelity and resource differences | TAFDRL-style reconstruction; reference data for selected methods; unequal tuning attack sets | Verify component mapping against primary papers and explicitly match or disclose information/tuning budgets |
| 2 | Development evaluation has been seen | Post-hoc numerical rejection bound and prior Ausgrid evaluation exposure | Preserve chronology. New seeds alone do not create an untouched temporal test; obtain independent validation or frame conclusions as retrospective |
| Production | Authorship, sources and final layout need sign-off | Elsevier class, 19 references, source checks and draft declarations | Confirm author list, contributions, funding/conflicts, licences, remaining provenance, AI disclosure and final PDF |

Small cross-partition percentages do not bound the effect on policy training or evaluation. Failure to reject is not evidence of equivalence. These points are retained in the manuscript.

## What this audit changed

- Targeted the standard Elsevier article class to Energy and AI.
- Reduced the main argument to **six figures and six tables**, retaining all remaining displays in the appendix: **13 figures and 19 tables in total**.
- Added a deterministic aggregate-norm argument and explicit realised contribution coefficients.
- Ran 24 exact paired Wilcoxon tests with two explicitly defined Holm families; exported every pair, p-value and adjustment.
- Reconstructed ACT imputation donors from raw data and checked donor values against stored processed profiles, allowing for file rounding.
- Verified the 2025–26 retailer import-price schedule against the official retailer page.
- Added concise highlights and draft author declarations without asserting unconfirmed authorship, conflicts or human review.
- Prepared a flat source archive for Elsevier's submission system; no submission or public release was performed.
- Compiled the 30-page A4 PDF with temporary Tectonic 0.15.0 tooling. Final log has no undefined references, missing citations or overfull boxes; two underfull bibliography paragraphs remain without overflowing text. Selected title, results and statistical-audit pages were visually inspected.

## Journal requirements checked and limits

- [Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions): elsarticle front matter and flat source files for Editorial Manager.
- [Elsevier highlights](https://www.elsevier.com/researcher/author/tools-and-resources/highlights): 3–5 highlights, at most 85 characters each; the included five lines pass that check.
- [Elsevier AI policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals): transparent disclosure and author verification, including reproducible data-visualisation workflows.
- The [Energy and AI author guide](https://www.sciencedirect.com/journal/energy-and-ai/publish/guide-for-authors) returned HTTP 403 during this audit. Its current journal-specific length, display limits, required files and reference style have **not** been verified. No claim of complete guide compliance is made.

The next work that materially raises submission readiness is closing the scientific blockers above, rather than adding more presentation volume.

## Academic editing pass

The later section-by-section edit is documented in `EDITORIAL_REVIEW.md`, with the original source backed up before editing. It preserved the numerical evidence and document structure while correcting the described MAD floor, cycle definition, Linux memory unit, MPC check scope and meter-attack scope. The MPC relaxation now has an explicit full-horizon validation qualification. These editorial and technical-description improvements do not close the scientific blockers above.
