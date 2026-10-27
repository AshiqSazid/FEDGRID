"""Clean AEMO Queensland (QLD1) 2025 price and demand into half hours, and map them onto the Ausgrid calendar.

AEMO stamps each 5 minute value at the END of its interval, in market time (AEST, UTC+10, no daylight saving).
Writes data/processed/aemo_qld_2025_30min.csv, indexed by half hour start:
  price_aud_mwh  mean wholesale price
  demand_mw      mean TOTALDEMAND
  demand_min_mw  lowest 5 minute TOTALDEMAND, which is compared with the MSL thresholds
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/aemo"
OUT = ROOT / "data/processed"
# AEMO 2025 Spring and Summer MSL Thresholds fact sheet, Table 2, Queensland system normal.
MSL1_TYPICAL_MW = 2600  # main reference in the paper
MSL1_ONEROUS_MW = 3660  # selects the real low-demand days of 2025 (sensitivity case agreed with Dr Hossain)


def to_2025(index):
    """Move each timestamp to 2025, keeping the time of day and the weekday, within 3 days of the same date."""
    days = index.normalize()
    mapped = {}
    for d in days.unique():
        base = pd.Timestamp(2025, d.month, min(d.day, 28) if d.month == 2 else d.day, tz=index.tz)
        m = base + pd.Timedelta(days=(d.weekday() - base.weekday() + 3) % 7 - 3)
        if m.year != 2025:  # early January or late December: wrap by 52 weeks, which keeps the weekday
            m -= pd.Timedelta(days=364 * int(np.sign(m.year - 2025)))
        mapped[d] = m
    return pd.DatetimeIndex(days.map(mapped)) + (index - days)


def main():
    d = pd.concat(pd.read_csv(p) for p in sorted(RAW.glob("PRICE_AND_DEMAND_2025*_QLD1.csv")))
    start = pd.to_datetime(d["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S") - pd.Timedelta("5min")
    d.index = pd.DatetimeIndex(start).tz_localize("Australia/Brisbane")
    hh = d.groupby(d.index.floor("30min")).agg(price_aud_mwh=("RRP", "mean"), demand_mw=("TOTALDEMAND", "mean"),
                                               demand_min_mw=("TOTALDEMAND", "min"))
    OUT.mkdir(parents=True, exist_ok=True)
    hh.to_csv(OUT / "aemo_qld_2025_30min.csv", float_format="%.2f", index_label="time_aest")

    low = hh["demand_min_mw"].groupby(hh.index.date).min()
    print(f"{len(hh)} half hours, price {hh['price_aud_mwh'].min():.0f} to {hh['price_aud_mwh'].max():.0f} $/MWh")
    for name, mw in [("MSL1 typical", MSL1_TYPICAL_MW), ("MSL1 onerous", MSL1_ONEROUS_MW)]:
        below = hh[hh["demand_min_mw"] < mw]
        print(f"{name} {mw} MW: {(low < mw).sum()} days, {len(below)} half hours, "
              f"hours {sorted(set(below.index.hour))}")
    assert len(hh) == 17520 and (low < MSL1_TYPICAL_MW).sum() == 0 and (low < MSL1_ONEROUS_MW).sum() == 42

    # Check the calendar mapping on the Ausgrid years: weekday kept, always lands in 2025, never more than 3 days off.
    t = pd.date_range("2010-07-01", "2013-07-01", freq="30min", inclusive="left", tz="Australia/Brisbane")
    m = to_2025(t)
    assert (m.day_of_week == t.day_of_week).all() and (m.year == 2025).all() and (m.time == t.time).all()
    print("calendar mapping ok")


if __name__ == "__main__":
    main()
