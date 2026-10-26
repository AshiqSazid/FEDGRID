"""Clean the Ausgrid Solar Home data into half-hourly tables on Queensland time.

Ausgrid timestamps are Sydney local time, with daylight saving in summer, and each value is the
energy in the half hour ENDING at its column time. Output rows are indexed by the half hour
START in AEST (UTC+10, no daylight saving), the same clock as AEMO and the Queensland tariffs.

Writes to data/processed:
  ausgrid_load_kwh.csv.gz   household load (general + controlled load), kWh per half hour, one column per customer
  ausgrid_pv_kwh.csv.gz     gross solar generation, kWh per half hour, one column per customer
  ausgrid_customers.csv     postcode, solar size and data quality checks per customer
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/ausgrid"
OUT = ROOT / "data/processed"
HH = [f"{h}:{m}" for h in range(24) for m in ("00", "30")][1:] + ["0:00"]  # 0:30, 1:00 ... 23:30, 0:00
OFFSETS = pd.to_timedelta(np.arange(48) * 30, unit="min").to_numpy()  # column k starts k*30 min after midnight


def read_raw():
    # Only blanks are missing values. By default pandas would also read the "NA" (estimated) quality flag as missing.
    df = pd.concat([pd.read_csv(p, skiprows=1, keep_default_na=False, na_values=[""], dtype={"Row Quality": str})
                    for p in sorted(RAW.glob("Solar home 20*.csv"))], ignore_index=True)
    dates = df["date"].unique()  # 2010-11 uses 1-Jul-10, the later years use 1/07/2012
    df["date"] = df["date"].map(dict(zip(dates, pd.to_datetime(dates, format="mixed", dayfirst=True))))
    dup = df.duplicated(["Customer", "Consumption Category", "date"])
    if dup.any():
        print(f"dropping {dup.sum()} duplicate rows")
    return df[~dup]


def matrix(df, cat, grid):
    """kWh per half hour for one consumption category, rows = AEST half hour start, columns = customer."""
    d = df[df["Consumption Category"] == cat]
    t = (d["date"].to_numpy()[:, None] + OFFSETS[None, :]).ravel()
    m = pd.DataFrame({"t": t, "c": np.repeat(d["Customer"].to_numpy(), 48), "v": d[HH].to_numpy(float).ravel()})
    m = m.pivot(index="t", columns="c", values="v")
    # The hour skipped or repeated at each daylight saving change can't be placed, so drop it and refill below.
    local = m.index.tz_localize("Australia/Sydney", ambiguous="NaT", nonexistent="NaT")
    m = m[~local.isna()]
    m.index = local[~local.isna()].tz_convert("Australia/Brisbane")
    # ponytail: linear fill of up to 2 hours covers the daylight saving gaps (night time only). Longer gaps stay empty.
    return m.reindex(grid).interpolate(limit=4, limit_area="inside")


def main():
    df = read_raw()
    first, last = df["date"].min(), df["date"].max()
    grid = pd.date_range(first, last + pd.Timedelta("23h30min"), freq="30min", tz="Australia/Brisbane")
    gc, cl, gg = (matrix(df, c, grid) for c in ("GC", "CL", "GG"))
    load = gc + cl.reindex(columns=gc.columns).fillna(0)  # homes without controlled load have no CL rows
    pv = gg.reindex(columns=gc.columns)

    info = df.groupby("Customer").agg(postcode=("Postcode", "first"), capacity_kwp=("Generator Capacity", "first"))
    hour = pv.index.hour
    night = (hour >= 21) | (hour < 4)  # dark all year in Sydney, on AEST
    # ponytail: heuristic quality rules. Tighten them if the clean set still looks noisy.
    info["has_controlled_load"] = info.index.isin(cl.columns)
    info["estimated_rows"] = df[df["Row Quality"].eq("NA")].groupby("Customer").size().reindex(info.index, fill_value=0)
    info["missing_halfhours"] = load.isna().sum() + pv.isna().sum()
    info["night_pv_share"] = (pv[night].sum() / pv.sum()).round(4)
    info["pv_over_capacity"] = (pv.gt(1.1 * 0.5 * info["capacity_kwp"], axis=1)).sum()  # >110% of rated power for 30 min
    info["clean"] = info["missing_halfhours"].eq(0) & info["night_pv_share"].le(0.01) & info["pv_over_capacity"].eq(0)

    OUT.mkdir(parents=True, exist_ok=True)
    load.to_csv(OUT / "ausgrid_load_kwh.csv.gz", float_format="%.3f", index_label="time_aest")
    pv.to_csv(OUT / "ausgrid_pv_kwh.csv.gz", float_format="%.3f", index_label="time_aest")
    info.to_csv(OUT / "ausgrid_customers.csv")
    print(f"{len(grid)} half hours, {load.shape[1]} customers, {info['clean'].sum()} pass all quality checks")
    print(info[["estimated_rows", "missing_halfhours", "night_pv_share", "pv_over_capacity"]].describe().round(3))
    check(load, pv)


def check(load, pv):
    # Mean annual kWh per customer from Ausgrid's data notes (Aug 2014), July to June years.
    notes = {2010: (6980, 2119), 2011: (6596, 2083), 2012: (6387, 2181)}
    fy = load.index.year - (load.index.month < 7)
    for y, (cons, gen) in notes.items():
        c, g = load[fy == y].sum().mean(), pv[fy == y].sum().mean()
        print(f"{y}-{y + 1 - 2000}: consumption {c:.0f} (notes {cons}), generation {g:.0f} (notes {gen})")
        assert abs(c / cons - 1) < 0.01 and abs(g / gen - 1) < 0.01, "annual totals don't match the data notes"
    # Without the daylight saving fix, summer solar would peak an hour late (12:30 to 13:00 start).
    summer = pv[pv.index.month.isin([12, 1, 2])]
    peak = summer.mean(axis=1).groupby(summer.index.time).mean().idxmax()
    print("summer solar peak, AEST half hour start:", peak)
    assert 11 <= peak.hour <= 12, "solar peak isn't near noon AEST"


if __name__ == "__main__":
    main()
