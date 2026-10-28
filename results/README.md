# Results

All results are on the 2012-13 test year. Feeder L has 36 homes and every run trains for 500 days.
- **Bill:** $ per home per year, including the daily supply charge.
- **Lost solar:** curtailed kWh per home per year.
- **Bill + lost solar:** lost solar valued at the 8.66c feed-in tariff, which is what training minimises.

Each log ends with the full set of test metrics. To rerun:
- the training runs: `./run_experiments.sh seeds` and `./run_experiments.sh attacks` (runs whose log already exists are skipped);
- idle, the rule and MPC: `.venv/bin/python src/baselines/mpc.py`, which writes `baselines.log`.

`before_residual/` holds the old runs from before residual learning and held-out week selection were added. They are kept only for the record. Don't use them.

## 1. Clean runs, 3 seeds (`*_sac_500_seed{0,1,2}.log`)

Each run keeps the policy that scored best on the 13 held-out training weeks. Figures are mean ± sd over seeds 0, 1 and 2.

| Method | Bill | Lost solar | Bill + lost solar | Battery cycles/yr |
|---|---|---|---|---|
| Idle | 1,457.78 | 537.74 | 1,504.35 | 0 |
| Self consumption rule | 952.96 | 473.18 | 993.94 | 77 |
| Local SAC | 963.78 ± 24.64 | 134.83 ± 19.89 | 975.46 ± 25.00 | 247 |
| FedAvg | 954.31 ± 31.82 | 144.57 ± 36.17 | 966.83 ± 34.46 | 256 |
| FedProx (mu 0.01) | 947.32 ± 26.26 | 153.54 ± 34.14 | 960.62 ± 27.93 | 253 |
| Central SAC | 879.05 ± 47.40 | 137.94 ± 64.53 | 891.00 ± 47.30 | 278 |
| Perfect foresight MPC | 590.95 | 62.87 | 596.39 | 213 |

Bills for seeds 0 / 1 / 2:
- Central: 875.19 / 928.26 / 833.70
- FedAvg: 930.27 / 990.39 / 942.27
- FedProx: 924.20 / 975.87 / 941.89
- Local: 935.89 / 982.61 / 972.85

**What the clean runs show**
- **Every run beats the rule** on the held-out weeks and on bill + lost solar.
- **Lost solar falls about 70%.**
- **More solar absorbed during minimum-load hours:** battery charging during the 3,660 MW minimum-load half hours is 12,400–13,900 kWh, against 1,971 kWh for the rule.
- **The bill alone barely moves for the federated methods.**
  - FedProx beats local in all 3 seeds.
  - FedAvg beats local in only 2 of 3.
- **Every federated run kept its policy from day 475**, so training had not finished at 500 days.
- **Central seed 0 went over 253 V** in 0.02% of half hours (max 253.19 V). The rule never does.

## 2. Attack study, seed 0 (`*_seed0_<attack>_share<percent>.log`)

**Setup**
- **Bad homes:** chosen at random with seed 0. They train honestly but send a corrupted update every round.
- **Defences:** trimmed mean and Krum are told the true share of bad homes.
- **Which policy is kept:** the last policy, not the best one, so damage late in training shows.
- **The attacks:**
  - flip: minus the home's own update;
  - scale: the home's update × 36;
  - noise: random values with variance 1,000;
  - Normalized: the attack of Fang, Wang and Gong (WWW 2025, arXiv 2502.08123), with full knowledge.

### 30% bad homes (11 of 36)

Bill, with the change from the same defence with no attack in brackets.

| Defence | No attack | Flip | Scale ×36 | Noise | Normalized |
|---|---|---|---|---|---|
| FedAvg | 943.70 | 1,024.75 (+81) | 999.56 (+56) | 1,458.82 (+515) | 1,459.78 (+516) |
| Median | 922.76 | 1,007.87 (+85) | 935.26 (+12) | 929.23 (+6) | 1,190.68 (+268) |
| Trimmed mean | 918.54 | 992.63 (+74) | 906.86 (−12) | 895.02 (−24) | 1,111.05 (+193) |
| Krum | 927.46 | 1,012.05 (+85) | 966.89 (+39) | 891.38 (−36) | 889.92 (−38) |

### Normalized attack by share of bad homes

Bill, with the change from no attack in brackets. FedAvg and median don't use the share, so each has one no-attack run.

| Defence | 20% (7 homes) | 30% (11 homes) | 40% (14 homes) |
|---|---|---|---|
| FedAvg | 1,390.25 (+447) | 1,459.78 (+516) | 1,459.93 (+516) |
| Median | 1,016.71 (+94) | 1,190.68 (+268) | 1,224.10 (+301) |
| Trimmed mean | 1,100.22 (+147) | 1,111.05 (+193) | 1,347.41 (+395) |
| Krum | 1,089.92 (+77) | 889.92 (−38) | 1,081.25 (+111) |

No-attack bills used for the brackets:

| Defence | 20% | 30% | 40% |
|---|---|---|---|
| Trimmed mean | 952.99 | 918.54 | 952.48 |
| Krum | 1,013.03 | 927.46 | 970.31 |

**What the attack study shows**
- **FedAvg has no protection.** Noise and Normalized both turn it into an idle battery, even with only 20% bad homes.
- **Median and trimmed mean resist noise and scale, but not Normalized.** The damage grows with the share of bad homes, up to +$395 for trimmed mean at 40%.
- **Flip costs every defence $74–85**, which puts all of them above the rule.
- **Krum's single-seed results can't be trusted.** With no attack it scores $927–1,013 depending only on how many bad homes it expects, so its result at 30% under Normalized is inside that spread.
- **Overall:** no standard defence protects the batteries in every case.

**Caveats**
- **One seed per run.** Gaps under about $25 are training randomness.
- **The Normalized attack is simplified.** It searches fixed grids for lambda and zeta instead of the paper's step rule, whose starting values the paper doesn't give.
