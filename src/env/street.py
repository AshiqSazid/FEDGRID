"""A CSIRO feeder street (L by default): one Ausgrid home on each load, one OpenDSS power flow per half hour.

No batteries yet. Each home's net power (load minus solar) is set on its load, so a negative value exports.
Running this file simulates the 2012-13 test year and reports how often the street goes over voltage.
"""
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEEDERS = ROOT / "data/raw/csiro/RepresentativeLVNetworks-master/data"
PROC = ROOT / "data/processed"
V_MIN, V_MAX = 216.2, 253.0  # AS 61000.3.100 limits for 230 V (-6% / +10%)
KVAR_PER_KW = 0.203  # household load at power factor 0.98, as in the feeder's own load data. Solar runs at unity.


class Street:
    def __init__(self, seed=0, pv_median_kwp=6.6, pv_max_kwp=10.0, feeder="L", exclude=()):
        """exclude: Ausgrid homes not to use, so a reference street can be kept apart from the federation's street."""
        dss.Text.Command(f'Redirect "{FEEDERS / feeder / "Master.dss"}"')
        self.loads = dss.Loads.AllNames()
        customers = pd.read_csv(PROC / "ausgrid_customers.csv", index_col=0)
        clean = customers.index[customers["clean"]]
        if len(exclude):
            clean = clean[~clean.isin(exclude)]
        self.homes = np.random.default_rng(seed).choice(clean, len(self.loads), replace=False)

        # ponytail: modern size = 2010 era size scaled so the median is pv_median_kwp, capped at pv_max_kwp.
        # It keeps each home's relative size. Swap in a real Queensland size distribution if one is found.
        cap = customers["capacity_kwp"]
        self.pv_kwp = np.minimum(cap[self.homes] * pv_median_kwp / cap[clean].median(), pv_max_kwp).to_numpy()

        cols = [str(h) for h in self.homes]
        load = pd.read_csv(PROC / "ausgrid_load_kwh.csv.gz", index_col=0, usecols=["time_aest", *cols])
        pv = pd.read_csv(PROC / "pv_toowoomba_kwh_per_kwp.csv.gz", index_col=0, usecols=["time_aest", *cols])
        self.time = pd.DatetimeIndex(pd.to_datetime(load.index, utc=True).tz_convert("Australia/Brisbane"))
        self.load_kw = 2 * load[cols].to_numpy()  # kWh per half hour -> average kW
        self.pv_kw = 2 * pv[cols].to_numpy() * self.pv_kwp

    def solve(self, load_kw, pv_kw):
        """Set every home's power, run one power flow, return each home's voltage in volts."""
        # ponytail: solar as a negative load is exact for a constant power snapshot. Use PVSystem elements
        # if inverter volt-var control is ever added.
        for name, p, q in zip(self.loads, load_kw - pv_kw, KVAR_PER_KW * load_kw):
            dss.Loads.Name(name)
            dss.Loads.kW(p)
            dss.Loads.kvar(q)
        dss.Solution.Solve()
        assert dss.Solution.Converged(), "power flow did not converge"
        volts = []
        for name in self.loads:
            dss.Circuit.SetActiveElement(f"Load.{name}")
            volts.append(dss.CktElement.VoltagesMagAng()[0])
        return np.array(volts)

    def transformer_kva(self):
        """Apparent power through the street's transformer at the last power flow, and its rating, in kVA."""
        if not dss.Transformers.First():
            return np.nan, np.nan
        pq = np.array(dss.CktElement.Powers())[:2 * dss.CktElement.NumConductors()]  # first terminal, P and Q
        return np.hypot(pq[0::2].sum(), pq[1::2].sum()), dss.Transformers.kVA()

    def run(self, start, end):
        steps = np.flatnonzero((self.time >= start) & (self.time < end))
        volts = np.array([self.solve(self.load_kw[i], self.pv_kw[i]) for i in steps])
        return pd.DataFrame(volts, index=self.time[steps], columns=self.homes)


if __name__ == "__main__":
    street = Street()
    v = street.run(pd.Timestamp("2012-07-01", tz="Australia/Brisbane"), pd.Timestamp("2013-07-01", tz="Australia/Brisbane"))
    v.to_csv(PROC / "street_L_no_battery_volts.csv.gz", float_format="%.2f", index_label="time_aest")
    over, under = v.gt(V_MAX), v.lt(V_MIN)
    print(f"{len(street.homes)} homes, solar {street.pv_kwp.min():.1f} to {street.pv_kwp.max():.1f} kWp, {len(v)} half hours")
    print(f"voltage {v.min().min():.1f} to {v.max().max():.1f} V (limits {V_MIN} to {V_MAX} V)")
    print(f"half hours with any home over {V_MAX} V: {over.any(axis=1).mean():.1%}, under {V_MIN} V: {under.any(axis=1).mean():.1%}")
    print(f"homes ever over voltage: {over.any().sum()} of {v.shape[1]}")
    print("street max voltage by hour of day (mean over the year):")
    print(v.max(axis=1).groupby(v.index.hour).mean().round(1).to_string())
