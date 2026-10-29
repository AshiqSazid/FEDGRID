"""The paper's tables, from the logs in results/. Values are mean ± 95% confidence interval half width [seeds].

  1. clean runs against idle, the rule and MPC (500 and 1000 days, FedProx's mu)
  2. bill of honest homes under each attack, 30% bad homes, and the damage against no attack, paired by seed
  3. damage by share of bad homes (Normalized attack)
  4. FedGrid ablation, message faults, feeder O
  5. rejection: honest and bad homes rejected, and honest homes with unusual solar sizes against the rest

Run: .venv/bin/python src/summarize.py [results folder] > results/tables.md
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
NAME = re.compile(r"(?P<method>[a-z]+)_sac_(?P<days>\d+)_seed(?P<seed>\d+)"
                  r"(?:_(?P<attack>[a-z]+)_share(?P<share>\d+))?(?:_(?P<variant>.+))?$")
METRIC = re.compile(r"^(\S.*?)\s{2,}(-?[\d.]+|nan)$", re.M)
DEFENCES = ["fedavg", "median", "trimmed", "krum", "fltrust", "flame", "tafdrl", "fedgrid"]
ATTACKS = ["none", "flip", "scale", "noise", "normalized", "meter", "reward"]


def read(path):
    run = NAME.match(path.stem).groupdict()
    text = path.read_text()
    tail = text[text.index(" SAC,"):]  # the validation lines come before the header
    run.update({k: float(v) for k, v in METRIC.findall(tail)})
    run["bad"] = [int(h) for h in re.search(r"from homes \[([\d, ]*)\]", tail)[1].split(",") if h.strip()] \
        if "from homes" in tail else []
    if m := re.search(r"rejected: honest homes ([\d.]+)%.*bad homes ([\d.na]+)%.*delayed honest changes ([\d.]+)%", tail):
        run["honest rejected %"], run["bad rejected %"], run["delayed rejected %"] = map(float, m.groups())
    if m := re.search(r"rejected share by home \(solar kWp\): (.*)", tail):
        rate, pv = np.array(re.findall(r"([\d.]+) \(([\d.]+)\)", m[1]), float).T
        honest = np.setdiff1d(np.arange(len(rate)), run["bad"])
        # Unusual: solar size in the street's smallest or largest sixth
        low, high = np.quantile(pv, [1 / 6, 5 / 6])
        unusual = (pv[honest] <= low) | (pv[honest] >= high)
        run["unusual honest rejected %"] = 100 * rate[honest][unusual].mean()
        run["usual honest rejected %"] = 100 * rate[honest][~unusual].mean()
    # Older logs have no honest homes' bill; their attacks changed updates only, so their own bill stands in
    run["honest bill"] = run.get("bill honest homes $/home/yr", run["bill $/home/yr"])
    run["bill + lost solar"] = run["bill $/home/yr"] + 0.0866 * run["curtailed kWh/home/yr"]
    return run


def ci(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return ""
    if len(v) == 1:
        return f"{v[0]:,.0f} [1]"
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():,.0f} ± {half:,.0f} [{len(v)}]"


def damage(runs, by):
    """Bill under attack minus the same rule's bill with no attack, paired by seed (same seed, same bad homes).
    The no-attack run at the same share if there is one (trimmed mean and Krum use the share), else at 30%."""
    none = runs[runs.attack == "none"].set_index(["method", "seed", "share"])["honest bill"].to_dict()
    hit = runs[runs.attack != "none"].copy()
    reference = [none.get((m, s, sh), none.get((m, s, "30"), np.nan)) for m, s, sh in zip(hit.method, hit.seed, hit.share)]
    hit["damage"] = hit["honest bill"] - np.array(reference)
    return table(hit, by, "damage", signed=True)


def table(runs, columns, value, signed=False, rows="method", order=DEFENCES):
    cell = runs.groupby([rows, columns])[value].apply(lambda v: ("+" if signed and v.mean() > 0 else "") + ci(v))
    out = cell.unstack(columns).fillna("")
    known = [c for c in [*ATTACKS, "10", "20", "30", "40"] if c in out.columns]
    out = out[known + [c for c in out.columns if c not in known]]
    return out.reindex([r for r in order if r in out.index] + [r for r in out.index if r not in order])


def markdown(frame):
    rows = [["", *map(str, frame.columns)], *[[str(i), *map(str, r)] for i, r in zip(frame.index, frame.values)]]
    return "\n".join("| " + " | ".join(r) + " |" for r in [rows[0], ["---"] * len(rows[0]), *rows[1:]])


def show(title, frame, note=""):
    if len(frame):
        print(f"\n## {title}\n\n{note + chr(10) * 2 if note else ''}{markdown(frame)}")


def baselines():
    rows = re.findall(r"^(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)$", (RESULTS / "baselines.log").read_text(), re.M)
    b = pd.DataFrame({k: list(map(float, v)) for k, *v in rows}, index=["idle", "rule", "MPC"])
    b["bill + lost solar"] = b["bill $/home/yr"] + 0.0866 * b["curtailed kWh/home/yr"]
    return b


if __name__ == "__main__":
    runs = pd.DataFrame([read(p) for p in sorted(RESULTS.glob("*_sac_*.log"))])
    runs["seed"] = runs.seed.astype(int)
    runs[["attack", "share", "variant"]] = runs[["attack", "share", "variant"]].fillna("")
    print(f"# Tables from {len(runs)} runs\n\nmean ± 95% CI half width [seeds]. Bills in $/home/yr on the 2012-13 test year.")

    clean = runs[runs.attack == ""].copy()
    clean["run"] = clean.method + " " + clean.days + " days" + clean.variant.map(lambda v: f", {v}" if v else "")
    metrics = ["bill $/home/yr", "curtailed kWh/home/yr", "bill + lost solar", "battery cycles/yr",
               "battery charging in MSL half hours kWh", "half hours over 253 V %", "transformer peak loading %"]
    metrics = [m for m in metrics if m in clean and clean[m].notna().any()]
    show("1. Clean runs (best policy on the held out weeks)",
         pd.concat([baselines().reindex(columns=metrics).round(1).astype(str).replace("nan", ""),
                    clean.groupby("run")[metrics].agg(ci)]))

    main = runs[(runs.share == "30") & (runs.variant == "")]
    show("2a. Bill of honest homes, 30% bad homes (last policy)", table(main, "attack", "honest bill"))
    show("2b. Damage against no attack, 30% bad homes", damage(main, "attack"),
         "Honest homes' bill under the attack minus with no attack, paired by seed.")

    norm = runs[runs.variant == ""]
    shares = damage(norm[norm.attack.isin(["none", "normalized"])], "share")
    show("3. Damage by share of bad homes, Normalized attack", shares,
         "FedAvg, median, FLTrust, FLAME, TAFDRL and FedGrid don't use the share, so their 30% no-attack run is the "
         "reference at every share. Trimmed mean and Krum have their own.")

    variants = runs[runs.variant.str.startswith(("without", "lost", "feeder"))].copy()
    if len(variants):
        variants["run"] = variants.method + ", " + variants.variant
        show("4. FedGrid ablation, message faults and feeder O (honest homes' bill)",
             table(variants, "attack", "honest bill", rows="run", order=[]))

    rejecting = runs[runs["honest rejected %"].notna()] if "honest rejected %" in runs else runs.iloc[:0]
    for letter, value in zip("abcde", ["honest rejected %", "bad rejected %", "unusual honest rejected %", "usual honest rejected %",
                  "delayed rejected %"]):
        if value in rejecting and rejecting[value].notna().any():
            part = rejecting[rejecting[value].notna() & (rejecting.share == "30")].copy()
            part["run"] = part.method + part.variant.map(lambda v: f", {v}" if v else "")
            show(f"5{letter}. {value.capitalize()} of changes", table(part, "attack", value, rows="run"))
