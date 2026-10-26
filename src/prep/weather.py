"""Rescale the Ausgrid (Sydney) solar profiles to Toowoomba, per kWp of panels.

Toowoomba yield = Ausgrid yield per kWp x (Toowoomba / Sydney long term mean irradiance for that month),
using NASA POWER climatology. System size is applied later, in the street simulation.

Writes data/processed/pv_toowoomba_kwh_per_kwp.csv.gz: kWh per kWp per half hour, same layout as ausgrid_pv_kwh.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/weather"
OUT = ROOT / "data/processed"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def monthly_irradiance(path):
    """Long term mean daily irradiance (kWh/m2/day) for each month 1 to 12."""
    head, values = open(path).read().split("-END HEADER-")[1].strip().splitlines()[:2]
    row = dict(zip(head.split(","), values.split(",")))
    return pd.Series([float(row[m]) for m in MONTHS], index=range(1, 13))


def main():
    # ponytail: a ratio of horizontal irradiance by month. It ignores panel tilt, temperature and day length shape.
    # Upgrade path: model both sites with pvlib if a reviewer asks for more than a sensitivity test.
    # Sydney stands in for all Ausgrid homes, which also include the Central Coast and Hunter.
    ratio = monthly_irradiance(RAW / "nasa_power_climatology_toowoomba.csv") / monthly_irradiance(
        RAW / "nasa_power_climatology_sydney.csv")
    print("Toowoomba / Sydney irradiance by month:", ratio.round(3).to_dict())

    pv = pd.read_csv(OUT / "ausgrid_pv_kwh.csv.gz", index_col=0, parse_dates=True)
    pv.columns = pv.columns.astype(int)
    kwp = pd.read_csv(OUT / "ausgrid_customers.csv", index_col=0)["capacity_kwp"]
    sydney = pv / kwp.reindex(pv.columns).to_numpy()
    toowoomba = sydney.mul(ratio.reindex(sydney.index.month).to_numpy(), axis=0)
    toowoomba.to_csv(OUT / "pv_toowoomba_kwh_per_kwp.csv.gz", float_format="%.4f", index_label="time_aest")

    # Check: annual yield per kWp should rise by about the annual irradiance ratio (5.50 / 4.65 = 1.18).
    s, t = sydney.sum().median() / 3, toowoomba.sum().median() / 3
    print(f"median annual yield: Sydney {s:.0f} kWh/kWp, Toowoomba {t:.0f} kWh/kWp, ratio {t / s:.3f}")
    assert 1.12 < t / s < 1.25, "yield ratio is far from the irradiance ratio"


if __name__ == "__main__":
    main()
