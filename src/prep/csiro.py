"""Catalogue the 23 CSIRO representative LV feeders, to pick the testbed street.

Loads every feeder in OpenDSS, runs one power flow, and writes to data/processed:
  csiro_feeders.csv   one row per feeder: size, how loads are modelled, line length, voltages
  csiro_loads.csv     one row per load: feeder, bus, phases, kV, kW
"""
from pathlib import Path

import opendssdirect as dss
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/csiro/RepresentativeLVNetworks-master/data"
OUT = ROOT / "data/processed"
KM = {0: None, 1: 1.609344, 2: 0.3048, 3: 1.0, 4: 1e-3, 5: 3.048e-4, 6: 2.54e-5, 7: 1e-5, 8: 1e-6}  # OpenDSS length units


def each(api):
    i = api.First()
    while i:
        yield
        i = api.Next()


def survey(master):
    dss.Text.Command(f'Redirect "{master}"')  # Master.dss clears, builds the circuit and solves it
    loads = []
    for _ in each(dss.Loads):
        loads.append(dict(load=dss.Loads.Name(), bus=dss.CktElement.BusNames()[0].split(".")[0],
                          phases=dss.CktElement.NumPhases(), kv=dss.Loads.kV(), kw=dss.Loads.kW()))
    line_km = sum(dss.Lines.Length() * (KM[dss.Lines.Units()] or 1.0) for _ in each(dss.Lines))
    lv = []
    for b in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(b)
        if 0 < dss.Bus.kVBase() < 1:
            lv += [v for v in dss.Bus.puVmagAngle()[::2] if v > 0]
    dss.Vsources.First()
    phases = pd.Series([l["phases"] for l in loads]).value_counts().sort_index()
    feeder = dict(
        feeder=master.parent.name,
        source_kv=dss.Vsources.BasekV(),
        transformers=dss.Transformers.Count(),
        loads=len(loads),
        load_phases=" ".join(f"{p}ph:{n}" for p, n in phases.items()),
        load_buses=len({l["bus"] for l in loads}),
        lines=dss.Lines.Count(),
        line_km=round(line_km, 3),
        converged=dss.Solution.Converged(),
        lv_vmin_pu=round(min(lv), 4) if lv else None,
        lv_vmax_pu=round(max(lv), 4) if lv else None,
    )
    return feeder, [dict(feeder=feeder["feeder"], **l) for l in loads]


def main():
    feeders, loads = [], []
    for master in sorted(RAW.glob("*/Master.dss")):
        f, l = survey(master)
        feeders.append(f)
        loads += l
    feeders, loads = pd.DataFrame(feeders), pd.DataFrame(loads)
    OUT.mkdir(parents=True, exist_ok=True)
    feeders.to_csv(OUT / "csiro_feeders.csv", index=False)
    loads.to_csv(OUT / "csiro_loads.csv", index=False)
    print(feeders.to_string(index=False))
    assert len(feeders) == 23, "expected the 23 representative feeders"


if __name__ == "__main__":
    main()
