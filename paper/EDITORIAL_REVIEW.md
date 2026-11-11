# Academic writing review and revision

This record follows the manuscript order. It separates improvements in writing from unresolved scientific issues; the edits do not certify journal acceptance or repair the experiment design.

The original main manuscript, section sources, tables, references and scripts were backed up before editing. See `tables/editorial_backup.json` for the backup directory and source count. The original manuscript is also retained there as `main_before_edit.tex`.

## 1. Abstract

**Assessment.** The abstract contains useful quantitative findings but moves too quickly from method description to improvement claims. Its attribution of the revision to filtering order is incomplete because the clipping multiplier also changes. The 69% curtailment result is attributed too broadly to federated learning. Several different evidence types—sample means, a parametric interval, a signed-rank test and a deterministic coefficient identity—need clearer separation.

**Revision.** Present the problem, method, experimental scope, principal numerical effect, statistical qualification and operational implications in that order. Identify the final configuration as a combined revision. Attribute curtailment to the clean FedAvg comparison, and distinguish recorded rejection from positive aggregation weight.

**Preserved evidence.** 36 households; six attacks; five Ausgrid seeds; approximately $88 to $11 annual Normalized damage; paired change −$76.8 ± $67.5; descriptive 95% Student-t interval; exact p = 0.0625; approximately $112 ACT reward damage; approximately 69% FedAvg curtailment reduction. No new outcome is introduced.

**Remaining concern.** The central effect is retrospective and does not pass the specified exact five-pair test at 0.05. ACT preprocessing overlap and the incomplete participation statistic remain material limitations. The abstract can be made precise; those concerns cannot be edited away.

## 2. Introduction

**Assessment.** The problem and operational setting are clear, but the transition from network constraints to malicious learning contributions is abrupt. The three research questions overlap with the contribution paragraph, and the phrase 'honest-household bills' does not describe the all-home outcome used for update attacks. The reported reduction is stronger than the available evidence for a general robustness claim.

**Revision.** Connect battery scheduling, federated information exchange, heterogeneous honest updates and compromised contributions in a single argument. State the evaluation questions once and distinguish the implemented testbed, comparative evidence and contribution-weight diagnosis. Use 'bill damage' without incorrectly implying that every attack metric excludes compromised households. Retain the exact cited literature and operational export figures.

**Remaining concern.** A defensible empirical contribution is present, but an exhaustive novelty claim and a general fairness claim are not supported. The revised introduction specifies the study's scope instead of asserting either.

## 3. Related work

**Assessment.** The cited methods are relevant, but the original prose reads partly as a sequence of disclaimers rather than a synthesis. The distinction between prior federated battery scheduling and this paper's attack/participation evaluation needs a clearer statement. The reconstruction of TAFDRL and the adaptation of FLAME must remain explicit to avoid implying a full reproduction.

**Subsection review.** Federated optimisation and residential control: connect heterogeneity to the household setting, and identify the specific evaluation difference from Sievers et al. Robust aggregation: group prior methods by the information and consistency criterion they use. Poisoning and trust-aware FRL: explain why the adaptive directional attack matters and delimit the reconstructed comparator.

**Revision.** Organise the discussion around the assumptions relevant to this study, replace repeated defensive phrasing with direct descriptions of comparator scope, and make the household-control gap specific. Existing citation keys, bibliography entries and source claims are retained; no new paper or unverified novelty statement is added.

**Remaining concern.** The reference set supports the stated positioning, but the manuscript does not document a systematic search. Comparator fidelity and unequal auxiliary information remain methodological qualifications.

## 4. System model and learning methods

**Assessment.** The model is technically detailed, but dense paragraphs mix implementation, assumptions and limitations. Some terms are insufficiently defined, and several descriptions are broader than the code supports. The mathematical development is otherwise compatible with the reported aggregation mechanism.

**Subsection review and revisions.**

- **Data and household assignment:** clarify timestamp conversion, the code-defined 21:00–04:00 solar-quality window, monthly irradiance scaling and the fixed-assignment scope of seed intervals. Put the ACT imputation qualification beside its preprocessing description rather than inside the export-scheduling paragraph.
- **Battery operation, exchange and cost:** preserve the state, power and bill equations; distinguish realised battery power from the requested action and identify the fixed historical tariff.
- **Export schedules and feeder simulation:** separate snapshot assumptions, ideal-information schedule construction, minimum-system-load scenario inputs and action enforcement. Define minimum system load (MSL) and distinguish export compliance from voltage/transformer guarantees.
- **Residual SAC control:** define Soft Actor-Critic (SAC), observations, action mapping, rewards and training blocks in a clear order. Replace 'deterministic mean action' with tanh of the Gaussian mean parameter, matching `src/agents/sac.py`.
- **Daily federation and FedGrid:** define the shared change, then grouping, filtering, clipping and weighting. Clarify that the final method changes both order and multiplier.
- **Bounded update size and coefficients:** preserve the argument and coefficient equations, while making their scope explicit: a norm bound is not a direction, convergence, bill or physical-safety guarantee.
- **Comparators and attack model:** distinguish the reference-data assumption, transmitted-update and training-data threats, and the finite adaptive search. Explain that meter poisoning changes the learner observation channels; the residual rule still uses the simulator's true load and solar. Define model predictive control (MPC) and state the exact five-day scope of its replay check.

**Explicit verified correction.** The original filter text floors the *scaled* MAD at 10^-12, whereas `src/agents/federated.py:198` floors the *unscaled* MAD before multiplying by 1.4826. Keeping the existing sigma definition, the admission threshold is corrected from `2.5 max(sigma, 10^-12)` to `2.5 max(sigma, 1.4826 × 10^-12)`. This changes only the description of the numerical stabiliser to match the frozen code; no code, parameter, experiment or result is changed. All other equations are retained.

**Verified scope qualifications.** The MPC LP uses separate nonnegative charge/discharge variables and exports their net power to the simulator. The five-day cost check is not a full-horizon consistency proof. This is disclosed as an unresolved validation concern, not presented as evidence that archived results are wrong.

**Remaining concern.** Trusted reference data, ideal-information export schedules, fixed household assignment, cross-partition imputation, comparator reconstruction and lack of full MPC model-consistency validation limit the conclusions independently of writing quality.

## 5. Evaluation design and reporting (Experimental Setup)

**Assessment.** The outcome definitions and pairing are sound, but replication, household alignment and statistical interpretation are presented in long paragraphs with repeated cautions. The coverage caption incorrectly extends the six-attack design to clean learning and limited sensitivity cases. Battery cycling lacks an explicit definition in the prose. The exact tests and descriptive intervals need to be distinguished without suggesting that a non-rejection proves equivalence.

**Subsection review.** Partitions/policy retention: state what changes with a seed and separate validation-selected clean policies from last-policy attack runs. Damage: retain the matched-home equation, all-home versus honest-home scopes, ACT annualisation and maximum-of-means definition. Rejection: describe reception, filtering and weighting distinctly, retaining the twelve-update diagnostic and missing-rate treatment. Uncertainty: retain the interval formula, exact tests, family sizes, sample sizes and p-value interpretation. Artifact construction: make reproducibility information factual and concise.

**Revision.** Reorder each subsection around its measured quantity and its interpretation. Correct the coverage caption to restrict six attacks plus a reference to the principal comparison rows. Define the reported cycling statistic as throughput divided by twice fleet capacity, matching the evaluator. Clarify that seed intervals and zero exceedance counts are conditional observations, not population-wide safety claims.

**Remaining concern.** Small fixed-assignment samples, unequal transfer coverage, incomplete rejection histories and preprocessing overlap remain. No experiment, replication count, outcome value or analysis family is changed.

## 6. Configuration selection and development history

**Assessment.** The section gives necessary disclosure but repeats the evaluation description and sometimes blurs the distinction between a validation score and annual damage. The chronology should be factual, not framed as a prospective protocol. The final sentence could imply that effective contribution was empirically measured, although historical logs do not contain the necessary weights.

**Subsection review.** Time partitions: preserve every date, split and evaluation-exposure qualification. Validation criterion: identify last-four averaging, attack-wise means, paired no-attack references and unequal search sets. Post-hoc bound: preserve the 6.2/30.8/2.0/4.0 selection values, the completed earlier sweep and the final settings. Scope: state retrospective analysis and the frozen configurations without implying preregistration or measuring unavailable effective weights.

**Revision.** Tighten chronology and distinguish selection score, recorded rejection and annual bill damage. Preserve all reported settings and development outcomes, including the post-hoc nature of the bound.

**Remaining concern.** Numerical selection uses validation lines but took place in a development process exposed to evaluation results. That cannot be remedied by rephrasing or additional seeds alone.

## 7. Results

**Assessment.** The numerical coverage is strong, but interpretation and caveats repeatedly interrupt the presentation. Broad phrases such as 'improves solar utilisation' and 'retaining much more protection' should be tied to measured quantities and specified attacks. The paired-analysis subsection is the best-supported comparative result and must keep its uncertainty, exact p-value and transfer pairing visible. Statements about cycles need the verified fleet-normalised interpretation.

**Subsection review and revisions.**

- **Clean energy:** identify the 500-day, validation-selected population and the specific FedAvg 69% reduction. Separate bills, curtailment, street charging and fleet-equivalent cycling; preserve all values and the oracle-information qualification.
- **Attack damage:** report means and worst-attack means explicitly, retaining every method and unfavourable outcome. Avoid interpreting a maximum of means as a statistically established ranking.
- **ACT and transfer:** distinguish retraining from policy transfer, preserve three/five/two seed coverage, and state that unequal aggregate means are not paired revision effects.
- **Matched revision:** keep the direction of the difference, all interval half-widths, p = 0.0625 and Holm p = 0.375, and the uncertain adverse changes under other attacks. Remove any suggestion that a favourable descriptive interval substitutes for the exact test.
- **Rejection trade-off:** describe the scatter's two quantities, missing methods and method-specific masks; do not treat it as a comparable fairness frontier.
- **Electrical outcomes:** identify averaging of run-level peaks and distinguish observed operating metrics from guaranteed limits.

**Revision.** Present observations before their interpretation, improve transitions, reduce repeated disclaimers and retain the qualifications necessary to interpret each display. Figure labels, placement, table links and all numerical results remain intact.

**Remaining concern.** The central comparison is retrospective; exact tests do not reject, and transfer precision is weak. The observed participation trade-off is not a validated effective-exclusion comparison.

## 8. Discussion

**Assessment.** Interpretation is relevant but repeats the results and relies on general phrases such as 'stronger findings' and 'better established'. The mechanism attribution, participation implications and deployment scope need a clearer progression. The laptop timing paragraph incorrectly labels Linux binary memory units as MB.

**Subsection review.** Revision: distinguish an observed joint effect from isolated causal effects of grouping, trust, order or multiplier. Participation: relate reception, admission and realised weight to the missing household-cost analysis. Energy/physical protection: distinguish the explicit reward objective, cycling, envelope enforcement and network limits. Implementation: state what the archived benchmark measures and what it omits.

**Revision.** Interpret the findings once, tie each implication to a measured or derived quantity, and replace vague performance claims with specific scope. Preserve the 110% transformer observation and 0.04–0.06 ms timing values. Correct 353–356 MB to **353–356 MiB**, matching Linux `ru_maxrss / 1024` in `src/device_benchmark.py`; numerical values remain unchanged.

**Remaining concern.** The data do not identify the independent mechanism responsible for the combined revision, household financial consequences of exclusion, degradation-adjusted value or embedded-device feasibility.

## 9. Limitations

**Assessment.** All principal concerns are disclosed, but the prose duplicates discussion passages and mixes uncertainty, missing experiments and implementation assumptions. The MPC's five-day check and separate charge/discharge variables also require explicit qualification after verification against the code.

**Paragraph review.** Development/selection: retain evaluation exposure, prior sweep and post-hoc bound. Effective participation: retain missing weights, rounding and unavailable household costs. Data/generalisation: explain irradiance scaling, ACT donor overlap and separated-week initialisation. Information/physical assumptions: identify ideal information, omitted costs, reference trust and MPC replay scope. Comparator/threat scope: retain reconstruction, finite attack search and resource differences. Statistical scope: preserve sample sizes, unadjusted intervals and non-rejecting exact tests.

**Revision.** Make each paragraph state the limitation and its implication, avoiding repeated generic warnings. Add the verified MPC relaxation concern as an unresolved validation issue without changing the baseline, its numbers or the interpretation of its archived realised bills.

**Remaining concern.** These are substantive scientific limitations, not linguistic defects. The manuscript's conclusions remain bounded by them.

## 10. Conclusion

**Assessment.** The conclusion captures the main finding but again attributes the result to filter order alone, and it repeats multiple limitations rather than stating what the study establishes. 'Useful collaboration' needs a measured interpretation.

**Revision.** State the combined revision, retain the $88 to $11 effect and 0.05 qualification, distinguish energy outcomes from participation, and end with the evaluation requirements implied by the evidence. Do not add significance, safety, fairness or deployment claims.

**Remaining concern.** The paper can conclude an observed attack-specific reduction and a measurement distinction, not general superiority or validated fair participation.

## 11. Declarations

**Assessment and revision.** The declarations accurately distinguish supplied computational artifacts from public release, and disclose assistance without inventing author approval. Wording is tightened only. Unconfirmed funding, contributions, conflicts, authorship and release terms remain explicitly unconfirmed; they cannot be completed from editorial judgement.

## 12. Extended evaluation and reproducibility (Appendix)

**Assessment.** The appendix preserves valuable evidence but repeats interpretations from the main text and sometimes mixes measured quantities with broader explanatory language. All ten subsections, thirteen tables and seven figures require consistency with the revised terminology and sample scopes.

**Subsection review and revisions.**

- **Detailed tabulations/development displays:** preserve all source links and show which figures concern development rather than independent evaluation.
- **Transfer coverage:** retain five-seed final means versus the two matched seeds for original comparisons.
- **Compromised share:** preserve 0, 10/20/30/40%, the share-specific references and the single-seed qualification.
- **Ablation/order:** distinguish one-factor removals from the final combined revision; avoid assigning a unique causal mechanism.
- **Faults/scaling/stress:** retain all scenario values, matched references, units and one-seed scope. Clarify table units as dollars per household per year.
- **Validation trajectories:** distinguish objective trajectories and observed seed ranges from annual outcomes and confidence intervals.
- **Computational measurements:** retain 54,916 parameters and one-device/three-trial scope, correct the memory unit to MiB, and specify omitted deployment costs.
- **Reproducibility:** make the role of paired observations, donor records and hashes clear without implying public release or new experiments.
- **Imputation overlap:** retain the cell denominator and all counts/fractions; avoid equating a small copied fraction with a small outcome effect.
- **Exact tests:** retain family sizes, no-zero/no-tie assumptions, the exact 0.0625 resolution, Holm 0.375 and every exported comparison. Explain why the t interval and rank test are not interchangeable.

**Remaining concern.** Single-seed sensitivity results do not establish thresholds; altered preprocessing has not been rerun; comparator model fidelity and full-horizon MPC consistency remain unresolved. The appendix retains those limits rather than smoothing them out.

## Citation and numerical checks

Independent read-only reviews rechecked the relevance of the primary sources for FedAvg, FLAME, Sievers et al., NextGen and the historical Ergon tariff. Existing keys and bibliography entries are preserved. The TAFDRL publisher page returned HTTP 403; no fresh equation-level validation is claimed. NASA/AEMO provenance limits remain explicit.

Independently reconstructed Normalized paired changes agree with the existing CSVs: Ausgrid −76.83 (half-width 67.496; five pairs), ACT −160.5867 (97.4839; three pairs), transfer −74.13 (175.8539; two pairs). Existing rounded manuscript values and all p-values are retained.

## Manuscript-wide assessment and consistency review

The revised narrative follows one argument: heterogeneous household learners exchange potentially corrupted changes; the implemented defence produces attack-specific economic outcomes; its admission mask must be interpreted separately from realised weight and household benefit. The Abstract and Conclusion now describe the same combined revision as the Methods and Results. The three research questions align with the experiments and coefficient analysis rather than implying an unmeasured fairness outcome.

Terminology is standardised around **final/original FedGrid**, **bill damage**, **recorded rejection**, **positive aggregation weight**, **fleet-equivalent cycles**, **annualised ACT blocks**, and **perfect-foresight MPC**. ACT, SAC, PCA, MSL, MPC and QCA are expanded where relevant. Confidence intervals remain explicitly descriptive; paired changes use common seeds; exact tests and Holm adjustment retain their original family definitions.

Repeated caveats have been consolidated without deleting the methodological qualifications. The main Results section reports observations and their immediate uncertainty; Discussion interprets mechanisms and practical implications; Limitations states unresolved design constraints. Detailed numerical and audit evidence remains in the appendix.

### Verified corrections and scope clarifications

1. **Filter stabiliser:** correct the inline threshold to match the placement of the MAD floor in the frozen code. All displayed equations, model settings and experimental outcomes remain unchanged.
2. **Cycle statistic:** clarify fleet-normalised equivalent cycling; preserve every reported cycle value.
3. **MPC replay:** identify the five checked days and flag the charge/discharge relaxation as unvalidated over the full horizon; do not infer that the recorded baseline bills are wrong.
4. **Meter attack:** clarify corruption of learner channels while the residual rule retains simulator readings.
5. **Memory units:** change MB to MiB in the benchmark narrative and table without altering 353–356.
6. **Coverage and damage units:** restrict the six-attack caption to the principal comparisons and specify dollars per household per year in the sensitivity table.

### Preservation and production checks

The machine-readable preservation record is `tables/editorial_preservation.json`. It checks the source against the pre-edit backup, including ordered headings/labels/figure references, citation and cross-reference multisets, exact displayed equations, unchanged bibliography bytes and numerical table rows. The inline threshold correction is separately recorded. No training code, processed dataset, result log, numerical CSV or figure asset is changed by this edit.

Table-caption and unit wording is also synchronised with the artifact generator so that rebuilding artifacts does not restore the earlier ambiguity. `main.tex` is reassembled from the edited sources and remains the complete manuscript text. The source-consistency checker, PDF compilation, selected-page visual review and source-package verification are performed after editing.

### Remaining scientific and author concerns

Writing is substantially improved, but submission readiness still depends on the existing scientific gaps: a comparable positive-contribution statistic; partition-isolated ACT evaluation; independent validation and adequate replication; comparator fidelity and information matching; forecast/degradation economics; and full-horizon MPC consistency. Author identity, affiliations, funding/conflicts, contributions and release terms also remain unconfirmed. A writing edit cannot establish those facts.

### Primary-source relevance checks retained

- Federated model averaging: https://proceedings.mlr.press/v54/mcmahan17a.html
- FLAME's clustering/clipping/noise scope: https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen
- Sievers et al., author postprint: https://publikationen.bibliothek.kit.edu/1000181773/159986782
- ACT NextGen data: https://zenodo.org/records/14885589
- Ergon historical tariff inputs: https://www.ergon.com.au/retail/residential/tariffs-and-prices/residential-tariffs

No citation key or bibliography entry was added, removed or replaced. The publisher access limitation for detailed TAFDRL checking remains recorded.

## Final production result

All sections and subsections were edited in manuscript order. The full manuscript compiles with Tectonic 0.15.0 to 30 A4 pages. Headings, labels, citation keys, cross-reference targets, figure links, all displayed equations, bibliography entries and numerical table rows pass preservation checks. The final build has no undefined references/citations, overfull boxes or oversized-float warnings; two underfull bibliography paragraphs remain. A full-document thumbnail review and selected full-size page checks found no clipped displays or missing content. Dense numerical panels remain better read from their vector PDFs and should receive a final journal-scale legibility check.

The original 32-page source is retained in the backup. The current 30-page version preserves the 13 figures and 19 tables; its shorter length results from prose revision and reflow, not removal of evidence. The five table edits clarify captions or units without changing numeric rows.

A final independent technical check confirmed the revised filter floor, evaluation action, cycle definition, attack scope, MPC scope, memory unit and statistical pairing. No additional factual discrepancy was identified in that focused check.
