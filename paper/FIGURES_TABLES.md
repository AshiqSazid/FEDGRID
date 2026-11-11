# Display guide — Energy and AI draft

**Main argument: 6 figures and 6 tables. Appendix: 7 figures and 13 tables. Total: 13 figures and 19 tables.** No evidence has been dropped to favour the proposed method. Detail has moved to the appendix to keep the central argument readable.

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

Selection candidates; clean energy; ACT damage; paired revisions; rejection; electrical outcomes; transfer damage; attack share; ablation; scenario sensitivity; device timing; imputation donor audit; exact statistical audit.

## Numerical sources

Every numerical figure has a source CSV in `tables/`. In addition:

- `paired_exact_observations.csv` and `paired_exact_tests.csv`: all 24 paired comparisons and two Holm families.
- `act_imputation_donors.csv` and `act_imputation_summary.csv`: reconstructed donor events, partition assignments and denominators.
- `*_damage_per_seed.csv`: the seed-matched economic outcomes.
- `*.sha256`: source hashes.

Captions distinguish seed intervals, observed ranges, missing entries and one-seed observations. The schematic and split timeline disclose AI assistance with the code. All charts derive from reproducible computational workflows; no control trajectories are invented from aggregate logs.
