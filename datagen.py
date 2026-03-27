from __future__ import annotations

import numpy as np
import pandas as pd

def generate_dynamic_pricing_data(
    start: str = "2026-01-01",
    periods: int = 24 * 6,   # 10-min intervals → 1 day = 144 steps
    freq: str = "10min",
    seed: int | None = 42,
    base_demand: float = 60.0,
    base_supply: float = 100.0,
    base_price: float = 100.0,
    noise_std: float = 0.1,
    event_prob: float = 0.02,
):
    rng = np.random.default_rng(seed)

    # 🔹 Time index
    ts = pd.date_range(start=start, periods=periods, freq=freq)

    # 🔹 Time features
    minutes = (ts.hour * 60 + ts.minute).to_numpy()
    phase = 2 * np.pi * (minutes / (24 * 60))

    # Demand peaks morning & evening
    demand_tod = 1 + 0.5 * np.sin(phase - np.pi / 2)

    # Supply slightly shifted
    supply_tod = 1 + 0.3 * np.sin(phase)

    # 🔹 Noise
    demand_noise = np.exp(rng.normal(0, noise_std, periods))
    supply_noise = np.exp(rng.normal(0, noise_std, periods))

    # 🔹 Events
    is_event = rng.random(periods) < event_prob
    event_multiplier = np.ones(periods)
    event_multiplier[is_event] = 1 + rng.uniform(0.5, 1.5, size=int(is_event.sum()))

    # 🔹 Initialize arrays
    demand = np.zeros(periods)
    supply = np.zeros(periods)
    price = np.zeros(periods)

    prev_price = base_price

    for t in range(periods):
        # 🟢 Price (smooth variation)
        price[t] = prev_price * rng.uniform(0.95, 1.05)

        # 🔴 Price elasticity (IMPORTANT)
        price_effect = np.exp(-0.01 * price[t])

        # 🟢 Demand
        demand[t] = (
            base_demand
            * demand_tod[t]
            * demand_noise[t]
            * event_multiplier[t]
            * price_effect
        )

        demand[t] = max(demand[t], 5)

        # 🟢 Supply
        supply[t] = (
            base_supply
            * supply_tod[t]
            * supply_noise[t]
            * (1 + 0.2 * (event_multiplier[t] - 1))
        )

        supply[t] = max(supply[t], 10)

        prev_price = price[t]

    # 🔹 Derived features
    ratio = demand / (supply + 1e-6)

    moving_avg_demand = (
        pd.Series(demand)
        .rolling(window=6, min_periods=1)
        .mean()
        .values
    )

    # 🔹 DataFrame
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "hour": ts.hour,
            "demand": demand.astype(int),
            "supply": supply.astype(int),
            "price": price,
            "demand_supply_ratio": ratio,
            "is_event": is_event,
            "moving_avg_demand": moving_avg_demand,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_dynamic_pricing_data(periods=144 * 5)  # 5 days (10-min intervals)
    print(df.head())
    print("\nStats:\n", df.describe())


print(df)