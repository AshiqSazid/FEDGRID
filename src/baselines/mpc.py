"""Perfect foresight MPC: the best case reference.

At midnight it solves one linear program for the whole day and every home at once, knowing the day's load, solar,
prices and export limits exactly, then plays that schedule in the environment. Each home minimises its own bill
plus curtailed solar, within its battery and export limits. Voltage is left to the export limits, as for every
other method.

Running this file compares idle, the self consumption rule and this MPC over the test period: the 2012-13 year, or the
13 test weeks of 2018 for ACT NextGen.
Run: .venv/bin/python src/baselines/mpc.py [supply pu, 1.0] [ausgrid|nextgen]
"""
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.battery_env import DT, TEST, BatteryStreet, idle, self_consumption, test


class PerfectForesightMPC:
    def __call__(self, env):
        if env.t == env.start:
            self.plan, self.planned_cost = self.solve(env)
        return self.plan[env.t - env.start]

    @staticmethod
    def solve(env):
        k = slice(env.start, env.end)
        load, pv = env.street.load_kw[k], env.street.pv_kw[k]
        buy, sell = env.buy[k][:, None], env.sell[k][:, None]
        T, n = load.shape
        charge, discharge, curtail, grid_in, grid_out = (cp.Variable((T, n), nonneg=True) for _ in range(5))
        soc = cp.Variable((T + 1, n))
        constraints = [
            charge <= env.p_max, discharge <= env.p_max, curtail <= pv, grid_out <= env.limits,
            grid_in - grid_out == load + charge - discharge - (pv - curtail),
            soc[0] == env.soc,  # called at midnight, before the day's first step
            soc[1:] == soc[:-1] + DT * (env.eta * charge - discharge / env.eta),
            soc >= 0, soc <= np.tile(env.capacity, (T + 1, 1)),
        ]
        bill = DT * cp.sum(cp.multiply(buy, grid_in) - cp.multiply(sell, grid_out))
        # Energy left at midnight is worth the price it will offset next (first half hour of the next day),
        # so the one day horizon doesn't dump it to the grid.
        stored_value = env.buy[env.end % len(env.buy)] * env.eta * cp.sum(soc[T])
        cp.Problem(cp.Minimize(bill + env.w_curtail * DT * cp.sum(curtail) - stored_value),
                   constraints).solve(solver=cp.HIGHS)
        return (charge.value - discharge.value) / env.p_max, bill.value


if __name__ == "__main__":
    env = BatteryStreet(supply_pu=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0,  # 1.02 for the stress test
                        dataset=sys.argv[2] if len(sys.argv) > 2 else "ausgrid")
    mpc = PerfectForesightMPC()

    # Check: playing the plan in the environment gives the bill the optimiser expected, so both use the same model.
    first = TEST[0] if env.test_weeks is None else env.test_weeks[0]
    for day in pd.date_range(first, periods=5, freq="D"):
        env.reset(day)
        done, cost = False, 0.0
        while not done:
            _, _, done, info = env.step(mpc(env))
            cost += info["cost"].sum()
        assert abs(cost - mpc.planned_cost) < 1e-3 * max(1, abs(cost)), (day, cost, mpc.planned_cost)
    print("environment reproduces the MPC plan")

    results = {name: test(env, pol)
               for name, pol in [("idle", idle), ("self consumption", self_consumption), ("MPC (best case)", mpc)]}
    print(pd.DataFrame(results).round(2).to_string())
