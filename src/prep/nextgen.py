"""Clean the ACT NextGen energy storage trial data into half-hourly tables on Queensland time.

One file per home, 5 minute average power, timestamps in Unix seconds, solar power negative. Output rows are indexed by
the half hour START in AEST (UTC+10), the same clock as the Ausgrid tables. The files end at 22:55 AEST on 31 December,
so only whole AEST days are kept: 1 January to 30 December 2018, exactly 52 weeks.

Gaps are scattered (about 8.5 days a home, no date missing for every home), so each missing half hour is filled from
the same home's same half hour on the nearest day with a reading, up to FILL_DAYS away, using real readings only.
Homes using more than MAX_LOAD_KWH a year are left out as unlikely to be ordinary households.

Writes to data/processed:
  nextgen_load_kwh.csv.gz        household load, kWh per half hour, one column per home, gaps filled
  nextgen_pv_kwh_per_kwp.csv.gz  solar generation per kWp of solar size, kWh per half hour (Canberra's own, not rescaled)
  nextgen_customers.csv          solar and battery sizes and data quality checks per home
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/nextgen"
OUT = ROOT / "data/processed"
GRID = pd.date_range("2018-01-01", "2018-12-30 23:30", freq="30min", tz="Australia/Brisbane")
FILL_DAYS = 14
# Set on 4 Nov 2026, before any ACT result: 6 homes are above it (18,736 to 57,368 kWh a year, median 7,620).
MAX_LOAD_KWH = 18000


def fill_gaps(frame, days=FILL_DAYS):
    """Each missing half hour takes the same half hour of the nearest day with a real reading: 1 day before, 1 after,
    2 before, and so on. Filled values are never used to fill others."""
    filled = frame.copy()
    for k in range(1, days + 1):
        for shift in (k * 48, -k * 48):  # previous day first
            filled = filled.fillna(frame.shift(shift))
    return filled


def read_home(path):
    d = pd.read_csv(path, index_col=0)
    d.index = pd.to_datetime(d.index, unit="s", utc=True).tz_convert("Australia/Brisbane")
    return d[(d.index >= GRID[0]) & (d.index < GRID[-1] + pd.Timedelta("30min"))]


def half_hours(kw):
    """Mean power of each half hour as kWh. Empty only if all of its 5 minute values are missing."""
    return 0.5 * kw.resample("30min").mean().reindex(GRID)


def main():
    load, pv, info = {}, {}, {}
    for path in sorted(RAW.glob("*.csv"), key=lambda p: int(p.name.split("-")[0])):
        home = int(re.search(r"id_(\d+)", path.name)[1])
        d = read_home(path)
        # ponytail: solar is generation only, so any small positive reading is inverter standby and counts as zero.
        solar = (-d["solar power (kW)"]).clip(lower=0)
        load[home], pv[home] = half_hours(d["load power (kW)"]), half_hours(solar)
        info[home] = {"capacity_kwp": d["solar capacity (kW)"].iloc[0], "battery_kwh": d["battery capacity (kWh)"].iloc[0],
                      "battery_kw": d["battery peak power (kW)"].iloc[0],
                      "missing_5min": int(d[["load power (kW)", "solar power (kW)"]].isna().any(axis=1).sum()),
                      "missing_halfhours": int(load[home].isna().sum() + pv[home].isna().sum())}
    load, pv, info = pd.DataFrame(load), pd.DataFrame(pv), pd.DataFrame(info).T
    check(load, pv)  # on the measured values, before any gap is filled
    load, pv = fill_gaps(load), fill_gaps(pv)
    kwp = info["capacity_kwp"].astype(float)
    night = (GRID.hour >= 21) | (GRID.hour < 4)  # dark all year in Canberra, on AEST
    info["missing_after_fill"] = load.isna().sum() + pv.isna().sum()
    info["night_pv_share"] = (pv[night].sum() / pv.sum().clip(lower=1e-9)).round(4)
    info["pv_over_capacity"] = pv.gt(1.1 * 0.5 * kwp, axis=1).sum()  # >110% of rated power for 30 min
    info["load_kwh_per_year"], info["pv_kwh_per_year"] = load.sum().round(1), pv.sum().round(1)
    # The Ausgrid rules, so the clean homes of both datasets mean the same thing, plus the household size limit.
    info["clean"] = (info["missing_after_fill"].eq(0) & info["night_pv_share"].le(0.01) & info["pv_over_capacity"].eq(0)
                     & info["load_kwh_per_year"].le(MAX_LOAD_KWH))

    OUT.mkdir(parents=True, exist_ok=True)
    load.to_csv(OUT / "nextgen_load_kwh.csv.gz", float_format="%.3f", index_label="time_aest")
    pv.div(kwp).to_csv(OUT / "nextgen_pv_kwh_per_kwp.csv.gz", float_format="%.4f", index_label="time_aest")
    info.to_csv(OUT / "nextgen_customers.csv", index_label="home")
    print(f"{len(GRID)} half hours, {load.shape[1]} homes, {info['clean'].sum()} pass all quality checks "
          f"({(info['load_kwh_per_year'] > MAX_LOAD_KWH).sum()} above {MAX_LOAD_KWH:,} kWh a year)")
    print(info.drop(columns="clean").astype(float).describe().round(3).T.to_string())
    # Filling: no gap left, and every filled value is a real reading of the same home and half hour within FILL_DAYS
    assert info["missing_after_fill"].eq(0).all(), "gaps left after filling"
    home = info["missing_halfhours"].astype(int).idxmax()
    raw = half_hours(read_home(next(RAW.glob(f"*-id_{home}.csv")))["load power (kW)"])
    gap = raw.index[raw.isna()][0]
    nearby = [raw.get(gap + pd.Timedelta(days=s * k)) for k in range(1, FILL_DAYS + 1) for s in (-1, 1)]
    assert load.at[gap, home] == next(v for v in nearby if v is not None and not np.isnan(v)), "fill took a wrong value"
    print("gap filling ok")


def check(load, pv):
    assert len(GRID) == 52 * 7 * 48, "not exactly 52 weeks of half hours"
    # Half hour energy must equal the 5 minute energy it came from.
    home = load.columns[0]
    d = read_home(next(RAW.glob(f"*-id_{home}.csv")))
    assert np.isclose(load[home].sum(), d["load power (kW)"].sum() / 12, rtol=1e-3), "load energy changed"
    # On the wrong clock, summer solar would peak an hour or more away from noon AEST.
    summer = pv[pv.index.month.isin([12, 1, 2])]
    peak = summer.mean(axis=1).groupby(summer.index.time).mean().idxmax()
    print("summer solar peak, AEST half hour start:", peak)
    assert 11 <= peak.hour <= 12, "solar peak isn't near noon AEST"


if __name__ == "__main__":
    main()
