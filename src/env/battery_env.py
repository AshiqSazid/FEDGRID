"""Feeder L with a battery in every home, export limits and solar curtailment.

One step is one half hour and one episode is one day. Each home's action is a number from -1 (full discharge)
to +1 (full charge). Inside a step, in the order of the research plan:
  1. battery power from the action, limited by how much the battery holds
  2. export limit check: if a home would export above its limit, first reduce battery discharge, then curtail solar
  3. one power flow for the street, then each home's reward

Running this file compares an idle battery with a simple rule over the 2012-13 test year.
"""
import sys
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so src/prep and src/env import as packages
from env.street import PROC, V_MAX, V_MIN, Street
from prep.aemo import MSL1_ONEROUS_MW, to_2025
from prep.tariffs import TARIFF_12D, import_price

DT = 0.5  # hours per step
TRAIN = ("2010-07-01", "2012-07-01")
TEST = ("2012-07-01", "2013-07-01")


class BatteryStreet:
    def __init__(self, seed=0, battery_kwh_per_kwp=3.0, battery_kw=5.0, efficiency=0.9, max_export_kw=10.0,
                 min_export_kw=1.5, doe_margin_v=1.0, msl_export_kw=0.0, msl_mw=MSL1_ONEROUS_MW, price="tariff",
                 supply_pu=1.0, w_curtail=0.0866, w_breach=1.0, w_voltage=0.1):
        s = self.street = Street(seed)
        dss.Vsources.First()
        dss.Vsources.PU(supply_pu)
        self.n = len(s.homes)
        # ponytail: battery size follows solar size. 3 kWh per kWp gives 20 kWh at the 6.6 kWp median, close to the
        # Clean Energy Regulator Q3 2025 average (19.7 kWh). Later quarters averaged 27 to 31 kWh.
        self.capacity = battery_kwh_per_kwp * s.pv_kwp
        self.p_max = np.full(self.n, battery_kw)
        self.eta = np.sqrt(efficiency)  # one way, so charge then discharge loses 1 - efficiency
        self.max_export_kw, self.min_export_kw, self.doe_margin_v = max_export_kw, min_export_kw, doe_margin_v
        self.phase = np.empty(self.n, dtype=int)
        for i, name in enumerate(s.loads):
            dss.Loads.Name(name)
            self.phase[i] = int(dss.CktElement.BusNames()[0].split(".")[1])
        # Voltage rise at each home (V per kW) when every home on its phase exports 1 kW more
        zero, base = np.zeros(self.n), s.solve(np.zeros(self.n), np.zeros(self.n))
        self.sensitivity = np.empty(self.n)
        for ph in np.unique(self.phase):
            on = self.phase == ph
            self.sensitivity[on] = (s.solve(zero, on.astype(float)) - base)[on]
        self.msl_export_kw = msl_export_kw
        # ponytail: penalty weights are starting values. w_curtail values lost solar at the 2025-26 feed-in tariff.
        self.w_curtail, self.w_breach, self.w_voltage = w_curtail, w_breach, w_voltage

        aemo = pd.read_csv(PROC / "aemo_qld_2025_30min.csv", index_col=0)
        aemo.index = pd.to_datetime(aemo.index, utc=True).tz_convert("Australia/Brisbane")
        a = aemo.reindex(to_2025(s.time))
        assert not a.isna().any().any(), "AEMO data missing for some mapped half hours"
        self.msl = (a["demand_min_mw"] < msl_mw).to_numpy()  # exports pushed to msl_export_kw in these half hours
        if price == "tariff":
            self.buy = import_price(s.time).to_numpy()
            self.sell = np.full(len(s.time), TARIFF_12D["2025-26"]["feed_in"])
        else:  # wholesale, both directions
            self.buy = self.sell = a["price_aud_mwh"].to_numpy() / 1000
        self.hours = s.time.hour + s.time.minute / 60

    def _day_limits(self):
        """Dynamic export limits for the day, as a network operator would set them from a forecast without batteries.

        Per phase schedule: every home on a phase may change its export by the same amount, the largest change that
        keeps the most sensitive home on that phase at V_MAX minus a margin. Capped at max_export_kw, never below
        min_export_kw (the Energex and Ergon minimum when limits are reduced), and pushed to msl_export_kw in minimum
        system load half hours.
        """
        # ponytail: linear voltage sensitivity per phase, not an optimal power flow DOE (the plan leaves that to
        # later work). Coupling between phases is left to the margin.
        s, steps = self.street, range(self.t, self.end)
        v0 = np.array([s.solve(s.load_kw[k], s.pv_kw[k]) for k in steps])
        export0 = s.pv_kw[self.t:self.end] - s.load_kw[self.t:self.end]
        headroom = V_MAX - self.doe_margin_v - v0
        limits = np.empty_like(export0)
        for ph in np.unique(self.phase):
            on = self.phase == ph
            sens = self.sensitivity[on]
            change = np.where(sens > 0.01, headroom[:, on] / np.where(sens > 0.01, sens, 1), np.inf).min(axis=1)
            limits[:, on] = export0[:, on] + change[:, None]
        limits = np.clip(limits, self.min_export_kw, self.max_export_kw)
        return np.where(self.msl[self.t:self.end, None], self.msl_export_kw, limits)

    def reset(self, day):
        """Start an episode at midnight on day. Returns the observation, one row per home."""
        s = self.street
        self.t = self.start = s.time.get_loc(pd.Timestamp(day, tz="Australia/Brisbane"))
        self.end = self.t + 48
        self.limits = self._day_limits()
        self.soc = 0.5 * self.capacity  # ponytail: each day starts half full. Carry charge over if episodes get longer.
        self.volts = s.solve(s.load_kw[self.t], s.pv_kw[self.t])
        return self.observe()

    def limit(self):
        return self.limits[self.t - self.start]

    def observe(self):
        """Per home: charge level, load kW, solar kW, price, price period (day, peak, night), time of day,
        export limit kW, last voltage V."""
        s, t, h = self.street, self.t, self.hours[self.t]
        period = [11 <= h < 16, 16 <= h < 21, h >= 21 or h < 11]
        shared = [self.buy[t], *period, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]
        return np.column_stack([self.soc / self.capacity, s.load_kw[t], s.pv_kw[t],
                                np.tile(shared, (self.n, 1)), self.limit(), self.volts])

    def step(self, action):
        s, t = self.street, self.t
        load, pv, limit = s.load_kw[t], s.pv_kw[t], self.limit()

        # 1. battery power (kW, + charge, - discharge) within its power rating and stored energy
        p = np.clip(action, -1, 1) * self.p_max
        p = np.clip(p, -np.minimum(self.p_max, self.soc * self.eta / DT),
                    np.minimum(self.p_max, (self.capacity - self.soc) / (self.eta * DT)))
        # 2. export limit: reduce discharge first, then curtail solar
        excess = np.maximum(pv - load - p - limit, 0)
        less_discharge = np.minimum(excess, np.maximum(-p, 0))
        p = p + less_discharge
        curtail = excess - less_discharge
        pv_used = pv - curtail
        net = load + p - pv_used  # kW from the grid, negative when exporting
        breach = np.maximum(-net - limit, 0)  # stays zero while curtailment is possible
        self.soc = np.clip(self.soc + DT * np.where(p > 0, p * self.eta, p / self.eta), 0, self.capacity)
        assert (curtail <= pv + 1e-9).all() and (breach < 1e-9).all()

        # 3. power flow (battery and solar at unity power factor) and reward
        self.volts = s.solve(load, pv_used - p)
        cost = DT * (np.maximum(net, 0) * self.buy[t] - np.maximum(-net, 0) * self.sell[t])
        v_out = np.maximum(self.volts - V_MAX, 0) + np.maximum(V_MIN - self.volts, 0)
        reward = -cost - self.w_curtail * DT * curtail - self.w_breach * DT * breach - self.w_voltage * v_out
        info = dict(cost=cost, curtail_kwh=DT * curtail, battery_kw=p, net_kw=net, volts=self.volts, msl=self.msl[t])
        self.t += 1
        return self.observe() if self.t < self.end else None, reward, self.t == self.end, info


def idle(env):
    return np.zeros(env.n)


def self_consumption(env):
    """Charge from surplus solar. Discharge to cover the home's own load in the 4pm to 9pm peak."""
    s, t = env.street, env.t
    if 16 <= env.hours[t] < 21:
        return -np.clip((s.load_kw[t] - s.pv_kw[t]) / env.p_max, 0, 1)
    return np.clip((s.pv_kw[t] - s.load_kw[t]) / env.p_max, 0, 1)


def evaluate(env, policy, start, end):
    days = pd.date_range(start, end, freq="D", inclusive="left")
    cost = curtail = throughput = msl_absorbed = 0.0
    over, vmax, msl_export, limits = [], 0.0, 0.0, []
    for day in days:
        env.reset(day)
        limits.append(env.limits)
        done = False
        while not done:
            _, _, done, i = env.step(policy(env))
            cost += i["cost"].sum()
            curtail += i["curtail_kwh"].sum()
            throughput += np.abs(i["battery_kw"]).sum() * DT
            over.append((i["volts"] > V_MAX).any())
            vmax = max(vmax, i["volts"].max())
            if i["msl"]:
                msl_export += np.maximum(-i["net_kw"], 0).sum() * DT
                msl_absorbed += np.maximum(i["battery_kw"], 0).sum() * DT
    supply = TARIFF_12D["2025-26"]["supply_per_day"] * len(days)
    midday = np.stack(limits)[:, 20:28]  # 10am to 2pm
    return {"bill $/home/yr": cost / env.n + supply, "curtailed kWh/home/yr": curtail / env.n,
            "battery cycles/yr": throughput / 2 / env.capacity.sum(), "half hours over 253 V %": 100 * np.mean(over),
            "max V": vmax, "export in MSL half hours kWh": msl_export,
            "battery charging in MSL half hours kWh": msl_absorbed, "midday export limit kW (median)": np.median(midday)}


if __name__ == "__main__":
    env = BatteryStreet()
    print(f"{env.n} homes, batteries {env.capacity.min():.0f} to {env.capacity.max():.0f} kWh, "
          f"homes per phase {np.bincount(env.phase)[1:].tolist()}, "
          f"voltage sensitivity {env.sensitivity.min():.2f} to {env.sensitivity.max():.2f} V per kW (own phase), "
          f"MSL half hours in data {env.msl.sum()}")
    results = {name: evaluate(env, pol, *TEST) for name, pol in [("idle", idle), ("self consumption", self_consumption)]}
    print(pd.DataFrame(results).round(2).to_string())
