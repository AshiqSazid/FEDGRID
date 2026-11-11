# Display guide — Energy and AI draft

**Main argument: 6 figures and 6 tables. Appendix: 7 figures and 18 tables. Total: 13 figures and 24 tables.** All experimental groups in the current 706-log learning archive are represented. Legacy implementation diagnostics remain development-history evidence rather than direct comparisons with the current controller.

## Main figures

- `architecture.pdf`: local learners, trusted reference, server and physical enforcement.
- `clean_energy.pdf`: bill, curtailed solar, throughput and minimum-load charging.
- `attack_damage.pdf`: six attacks and three evaluation settings.
- `paired_revision.pdf`: matched final-minus-original damage and uncertainty.
- `damage_rejection.pdf`: descriptive damage versus logged rejection, with caveats.
- `rejection_heatmap.pdf`: honest and compromised rejection by attack.

## Appendix figures

Selection candidates, data splits, transformer outcomes, compromised-share sensitivity, original-method ablations, fault/scaling/stress scenarios and validation trajectories.

## Main tables

Configuration, algorithm procedure, comparator scope, threats, coverage and Ausgrid attack damage.

## Appendix tables

Selection candidates; clean energy; ACT damage; paired revisions; rejection; electrical outcomes; transfer damage; attack share; ablation; scenario sensitivity; device timing; imputation donor audit; exact statistical audit; extended clean learning; deterministic baseline context; attack-time electrical maxima; household rejection strata; full later candidate settings.

## Numerical sources

Every numerical figure has a source CSV in `tables/`. In addition:

- `paired_exact_observations.csv` and `paired_exact_tests.csv`: all 24 paired comparisons and two Holm families.
- `act_imputation_donors.csv` and `act_imputation_summary.csv`: reconstructed donor events, partition assignments and denominators.
- `*_damage_per_seed.csv`: the seed-matched economic outcomes.
- `damage_reference_trace.csv`: the attacked log, matched no-attack log, bill column and arithmetic for all 636 principal damage observations.
- `extended_clean_source.csv`, `all_baselines_source.csv`, `attack_electrical_source.csv`, `heterogeneity_source.csv`, `selection_settings_source.csv`: per-run inputs to the five additional tables.
- `evidence_audit.json`: independent computation checks and reconciliation of the 706 current learning logs.
- `*.sha256`: source hashes.

Captions distinguish seed intervals, observed ranges, missing entries and one-seed observations. The schematic and split timeline disclose AI assistance with the code. All charts derive from reproducible computational workflows; no control trajectories are invented from aggregate logs.
