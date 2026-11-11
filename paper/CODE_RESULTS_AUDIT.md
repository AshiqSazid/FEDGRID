# Independent code and results audit — 11 November 2026

The principal reported attack-damage values agree with the archived implementation outputs. The previous manuscript nevertheless had one misleading precision choice and several omissions in methods, selection history and experimental coverage. Those reporting issues have been corrected. This audit establishes fidelity to the available code and logs; it does not validate the physical model against a field system or resolve the study's scientific limitations.

## Evidence checked

The independent parser in `scripts/audit_evidence.py` does not import the manuscript's summariser or plotting helpers. It checks bill arithmetic against raw logs, distinguishes all-home update attacks from matched honest-home data attacks, and uses the explicit honest-reference reruns where available. `tables/damage_reference_trace.csv` records both source files and bill operands for every principal observation.

| Check | Verified observations |
| --- | ---: |
| Ausgrid attack damage | 300 |
| ACT retraining attack damage | 180 |
| ACT transfer attack damage | 156 |
| No-attack electrical observations | 80 |
| Honest and compromised rejection rows | 350 |
| Paired final-minus-original changes | 60 |
| Exact paired tests and Holm adjustments | 24 |
| Frozen training-source, runner and dependency files | 17 |
| Extended clean-learning runs | 30 |
| Deterministic baseline rows | 9 |
| Attack-time electrical runs | 480 |
| Household rejection summaries | 68 |
| Later candidate and comparator settings | 14 |

All 636 principal damage observations match. Headline means, attack-wise maxima, seed scopes and Student-t intervals were also independently recomputed. All 24 exact tests were reproduced by enumerating the 32 five-pair rank-sign outcomes, with Holm adjustment separately within the 18-test comparator family and six-test revision family. No adjusted comparison rejects at 0.05. The Normalized revision comparison has unadjusted p=0.0625 and adjusted p=0.375.

The original 706 primary-log, 975 expanded-source and 100 raw ACT input hashes matched before reporting changes. The expanded manifest now has 984 entries, adding the stress baselines, diagnostic history and selection reports/flags. A separate 1001-entry evidence manifest fingerprints the independently audited inputs. The 17 frozen research files match their earlier recorded hashes. Paper source and analysis scripts change intentionally; experiment logs, training code and datasets were not edited or retrained.

## Corrections made

### Small voltage violations were hidden by rounding

`tables/electrical.tex` previously displayed percentages to one decimal, making small nonzero rates appear as `0.0 ± 0.0`. The table now uses three decimals and identifies the two-decimal precision of the underlying logged rates. Final FedGrid's no-attack Ausgrid mean is **0.010 ± 0.028%**, over five seeds. Its seed-4 log records **0.05%** exceedance and **253.60 V** maximum voltage (`results/fedgrid_sac_500_seed4_none_share30_final.log`). FedAvg and tuned TAFDRL-style have means of 0.004%; original FedGrid has 0.002%.

These are mean run-level rates and descriptive seed intervals, not precision beyond the source logs or safety bounds. The source observations were already correct and remain unchanged. The generator now preserves this formatting on rebuild.

### Selection used unequal condition pools and rounded summaries

`src/select_candidate.py` averages rejection over every available nominal-30% run per version, and rounds the returned summaries to one decimal. Original FedGrid uses 21 runs: no attack plus six attacks over three seeds. Complete FedGrid candidates use 15: no attack plus four update attacks. Tuned TAFDRL-style uses nine: no attack, flip and Normalized.

The historical original rejection mean is 2.02857%, printed as 2.0%; the final candidate mean is 3.40667%, printed as 3.4%. `src/choose_final.py` doubles the rounded original statistic to produce the historical **4.0%** eligibility bound. Restricting original FedGrid to the candidate's conditions would give 2.27333%, so these statistics must not be presented as a condition-matched rejection comparison. The paper now explains the pools and rounding. Historical flags, reports and configuration decisions were preserved. The candidate table's maximum-household column is labelled as a **mean of within-run maxima**, rather than a worst case across runs.

### Evaluation exposure preceded aggregation revision

`results/diagnostics/README.md` states that residual-controller alternatives were compared using evaluation-year bills before the current zero-initialised actor output and day-1 validation checkpoint. Selection, limitations and development history now disclose this earlier exposure as well as the later FedGrid revision and post-hoc rejection bound. The old implementation diagnostics are development-history evidence, not current-method replicates. New seeds on the same profiles do not restore an untouched evaluation year.

### Numerical conventions and local learner state needed specification

The methods now specify the lower middle order statistic used by `torch.median` for even sample counts, including within-band filtering and clipping. They also specify the `torch.cosine_similarity` convention: each norm is floored at 1e-8, and zero-vector agreement is zero. This matters for zero reference updates during warmup. Local target critics, optimiser states and entropy temperatures persist across broadcasts; validation occurs on day 1 and every 25 days. Currency is explicitly Australian dollars.

The inspected reward, battery equations, residual action, shared payload, reference learner, tariffs, time partitions, annualisation, fleet-equivalent cycling, attack operations and MPC qualifications otherwise agree with the implementation. A suspected meter-observation aliasing issue was checked with the actual environment and learner: float64 observations are copied into float32 tensors, so swaps do not modify the source arrays or alternate under repeated calls. No controller change was warranted.

## Coverage added

Five appendix tables retain evidence that had been omitted or only partly described:

| Table | Added coverage |
| --- | --- |
| `extended_clean` | All 30 clean-learning runs, including 1000-day learning and two additional FedProx coefficients |
| `all_baselines` | Idle, rule and perfect-foresight MPC for Ausgrid, ACT and 1.02 pu voltage stress |
| `attack_electrical` | Maximum observed electrical metrics across all 480 principal attacked runs, with measurement availability |
| `heterogeneity` | Rounded per-home honest rejection by solar-capacity stratum under no attack and Normalized |
| `selection_settings` | Both later FedGrid settings sweeps and all four TAFDRL-style tuning settings, including the available 40% FedGrid validation sensitivity |

Associated CSVs retain per-run sources. All new tables were separately checked against raw logs; all 14 selection summaries reproduce the historical selector. The new physical display retains unfavourable outcomes: FedAvg's Normalized runs reach transformer peaks of 108.89% and 113.84%. The text also reports final FedGrid's single-seed no-attack stress result of 0.27% voltage exceedance and 254.09 V maximum voltage. These observations do not support an unconditional grid-safety claim.

The household analysis uses rounded capacities and rates and is explicitly retrospective. For final FedGrid under Normalized, unusual-PV honest homes average approximately 9.8% rejection versus 4.1% for other honest homes. These are filter diagnostics, not measured effective contribution, individual economic harm or a causal fairness finding.

The entire current top-level learning-log archive reconciles as follows:

| Experimental group | Logs | Representation |
| --- | ---: | --- |
| Principal Ausgrid | 350 | Attack damage, rejection, electrical metrics and transfer where present |
| ACT retraining | 210 | Damage and electrical metrics |
| Clean learning | 30 | Clean and extended-clean tables |
| Original-method ablations | 18 | Ablation table and figure |
| Additional compromised-share attacks | 30 | Share sensitivity |
| Share-specific clean references | 6 | Matched damage denominators |
| Message faults | 20 | Matched scenario sensitivity |
| Feeder O | 10 | Matched scaling sensitivity |
| Voltage stress | 8 | Matched scenario sensitivity and explicit violations |
| Honest-home reference reruns | 24 | Matched meter/reward damage denominators |
| **Total** | **706** | |

The 266 candidate logs support the base-candidate and settings displays; they are not pooled with principal attack replications. Fourteen older diagnostic logs and 23 pre-residual logs concern earlier implementations and remain excluded from direct current-controller comparisons. Coverage means that experimental groups and relevant outcomes are represented; it does not treat every historical implementation as the same algorithm or print every scalar in every log.

## Rebuild and checks

The sources immediately before this audit were backed up to `paper_backups/evidence_audit_2026-11-11_193548/paper_sources_before.tar.gz`. The earlier academic-edit preservation record is a historical checkpoint. Its equality checks intentionally do not describe this revision, which adds content and corrects display precision.

The current manuscript contains **13 figures, 24 tables and 19 references**: six figures/six tables in the main argument and seven figures/18 tables in the appendix. The assembled `main.tex` agrees with its editable section/table sources. Tectonic 0.15.0 compiled the revised manuscript to **34 A4 pages**. The log has no undefined references, missing citations or overfull boxes; two underfull bibliography paragraphs remain. Electrical and new appendix table pages were visually inspected. The current PDF and both source archives accompany this report.

## What remains unresolved

The study is still retrospective and conditional on the supplied simulator, household assignment, attacks and information assumptions. ACT imputation crosses partitions; recorded rejection has different meanings across methods and omits zero-weight FedGrid groups; small seed counts limit inference; comparator fidelity and tuning budgets differ; MPC uses ideal information and a relaxation with limited replay checking; battery ageing is not monetised. These require additional research or narrower claims, as detailed in `SUBMISSION_READINESS.md`. Accurate reporting alone does not establish readiness for a top-tier journal.
