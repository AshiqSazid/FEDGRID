"""Choose among FedGrid versions and TAFDRL settings from the validation weeks only. This reads each log's validation
lines and rejection rates and never a test-year bill, as Dr Hossain asked on 3 Nov.

Damage: mean of a run's last 4 validation lines under attack, minus the same with no attack, paired by seed, $/home over
the 13 validation weeks. Measure, fixed before tuning: lowest worst-case mean damage over the update attacks at 30% bad,
with fairness reported beside it.

Only seeds 0-2 on the Ausgrid street count: seeds 3 and 4, ACT NextGen and the final reruns influence no choice.

Run: .venv/bin/python src/select_candidate.py fedgrid|tafdrl [results folder]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAME = re.compile(r"(?P<method>[a-z]+)_sac_500_seed(?P<seed>\d+)_(?P<attack>[a-z]+)_share(?P<share>\d+)(?:_(?P<version>.+))?$")
SKIP = ("honest", "lost", "feeder", "pu", "act")  # not versions of the rule under test
SEEDS = ("0", "1", "2")
UPDATE_ATTACKS = ["normalized", "noise", "flip", "scale"]  # false rewards inflate the validation bill of the bad homes


def read(path):
    match = NAME.match(path.stem)
    version = (match["version"] or "") if match else ""
    if not match or version.startswith(SKIP) or version.endswith("final") or match["seed"] not in SEEDS:
        return None
    text = path.read_text()
    run = match.groupdict()
    run["version"] = run["version"] or "original"
    run["validation"] = np.mean([float(v) for v in re.findall(r"^day \d+: validation ([\d.]+)", text, re.M)[-4:]])
    if m := re.search(r"rejected: honest homes ([\d.]+)% .*most rejected home ([\d.]+)%\), bad homes ([\d.na]+)%", text):
        run["honest rejected %"], run["worst honest home %"], run["bad rejected %"] = map(float, m.groups())
    return run


def tables(method, folder):
    paths = [*folder.glob(f"{method}_sac_500_*.log"), *folder.glob(f"candidates/{method}_sac_500_*.log")]
    runs = pd.DataFrame([r for r in map(read, paths) if r])
    none = runs[runs.attack == "none"].set_index(["version", "seed"])["validation"]
    hit = runs[runs.attack.isin(UPDATE_ATTACKS)].copy()
    hit["damage"] = hit.validation - [none.get((v, s), np.nan) for v, s in zip(hit.version, hit.seed)]
    main = hit[hit.share == "30"]
    damage = main.pivot_table(index="version", columns="attack", values="damage")
    damage["worst"] = damage.max(axis=1)
    damage["attacks"] = main.groupby("version").attack.nunique()  # worst means little until this reaches 4
    damage["seeds"] = main.groupby("version").seed.nunique()
    damage["runs"] = main.groupby("version").damage.count()  # paired with a no-attack run: attacks x seeds when complete
    fair = runs[runs.share == "30"].groupby("version")[["honest rejected %", "worst honest home %"]].mean()
    bad = main.groupby("version")["bad rejected %"].mean()
    wide = hit[hit.share == "40"].groupby("version")[["damage", "honest rejected %", "worst honest home %"]].mean()
    wide.columns = ["40% damage", "40% honest rejected %", "40% worst honest home %"]
    return damage.sort_values("worst").join([fair, bad, wide]).round(1)


if __name__ == "__main__":
    folder = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "results"
    print(tables(sys.argv[1], folder).to_string())
