"""Ergon Energy Retail tariff 12D (residential time of use) for regional Queensland, including Toowoomba.

Retail prices are what a household pays, in $/kWh including GST. They're the same every day of the week, with
the same three windows as Ergon's RTOUE network tariff (Ergon Energy Network Tariff Guide 2025-26).
Sources:
  ergon.com.au/retail/residential/tariffs-and-prices/residential-tariffs (rates from 1 July 2025 and 1 July 2026)
  QCA regional Queensland feed-in tariff, 2025-26 (8.66 c/kWh) and 2026-27 (6.006 c/kWh)
"""
import numpy as np
import pandas as pd

TARIFF_12D = {
    "2025-26": dict(day=0.22680, peak=0.45712, night=0.28035, supply_per_day=1.48834, feed_in=0.0866),
    "2026-27": dict(day=0.18512, peak=0.40651, night=0.25044, supply_per_day=1.56730, feed_in=0.06006),
}


def import_price(index, year="2025-26"):
    """$/kWh for each half hour starting at index (AEST). Day 11am-4pm, peak 4pm-9pm, night 9pm-11am."""
    t, h = TARIFF_12D[year], index.hour
    return pd.Series(np.select([(h >= 11) & (h < 16), (h >= 16) & (h < 21)], [t["day"], t["peak"]], t["night"]),
                     index=index)


if __name__ == "__main__":
    idx = pd.DatetimeIndex(["2025-01-01 10:30", "2025-01-01 11:00", "2025-01-01 15:30", "2025-01-01 16:00",
                            "2025-01-01 20:30", "2025-01-01 21:00"])
    assert import_price(idx).tolist() == [0.28035, 0.22680, 0.22680, 0.45712, 0.45712, 0.28035]
    print("ok")
