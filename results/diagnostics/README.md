# Why the agent learns a correction to the rule

These are the 500-day central SAC runs from 28 October 2026 that led to residual learning. They test what fixes the failed seeds, and they're the evidence for that design choice.

Before residual learning, the agent's zero action meant idle and seed 1 failed for every method. Its policy stayed close to idle: a held-out score of about 367, against about 383 for idle and 247 for the rule. The likely reason is that discharging an empty battery changes nothing, so it gives no learning signal, while charging costs money straight away.

## Results

Test bill in $/home/yr. Figures in brackets are lost solar in kWh/home/yr.

| What was tried | Seed 0 | Seed 1 | Seed 2 | Logs |
|---|---|---|---|---|
| Held-out week selection only | 891 | 1,355 (455) | 894 | `central_valid_seed*` |
| Plus reward shaping (Ng et al. 1999), charge valued at peak price × efficiency | 1,190 | 1,322 | 1,315 | `fix_<seed>_1_5` |
| 25 warmup days instead of 5, no shaping | – | 1,062 | – | `fix_1_0_25` |
| Starting entropy weight 0.1 instead of 1.0 | 1,066 | 1,454 (538) | 1,452 (517) | `alpha_<seed>_0.1` |
| **Residual learning: action = rule + correction** | **844** | **919** | **858** | `residual_central_seed*` |

For reference:
- self consumption rule: 952.96 (473);
- idle: 1,457.78 (538).

`alpha_1_1.0` is the same setting as the first row, run again, and gives the same result (1,354.79).

**What this shows**
- **Shaping made every seed fail.**
- **A lower entropy weight left seeds 1 and 2 idle.**
- **Residual learning was the only fix that worked for all three seeds.**

## Notes
- **The residual runs came before two later changes:** the zero-initialised last actor layer and the day-1 checkpoint. The committed runs in `results/` (central seeds 0/1/2: 875 / 928 / 834) include both.
- **`try_fix.py` and `try_alpha.py` won't run on the current code.** They ran against working versions of `src/agents/sac.py` that had `shaping` and `alpha` options, which were removed once residual learning worked. They're kept as a record of what was run.
