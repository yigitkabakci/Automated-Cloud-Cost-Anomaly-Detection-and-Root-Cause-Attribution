import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    ANOMALY_MULTIPLIER_RANGE,
    BASE_COST,
    N_ANOMALIES,
    N_DAYS,
    NOISE_STD,
    SEASONALITY_AMPLITUDE,
    SEED,
    TREND_SLOPE,
)


def generate_cost_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    dates = pd.date_range(start="2024-01-01", periods=N_DAYS, freq="D")
    t = np.arange(N_DAYS)

    trend = BASE_COST + TREND_SLOPE * t
    seasonality = SEASONALITY_AMPLITUDE * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, NOISE_STD, N_DAYS)

    cost = trend + seasonality + noise

    is_anomaly = np.zeros(N_DAYS, dtype=bool)
    anomaly_indices = rng.choice(N_DAYS, size=N_ANOMALIES, replace=False)
    multipliers = rng.uniform(*ANOMALY_MULTIPLIER_RANGE, size=N_ANOMALIES)
    cost[anomaly_indices] *= multipliers
    is_anomaly[anomaly_indices] = True

    cost = np.maximum(cost, 0)

    return pd.DataFrame({"date": dates, "cost": cost, "is_anomaly": is_anomaly})


if __name__ == "__main__":
    df = generate_cost_data()
    print(df.head(10).to_string())
    print(f"\nToplam gün sayısı : {len(df)}")
    print(f"Anomali sayısı    : {df['is_anomaly'].sum()}")
    print(f"Min maliyet       : {df['cost'].min():.2f}")
    print(f"Max maliyet       : {df['cost'].max():.2f}")
    print(f"Ortalama maliyet  : {df['cost'].mean():.2f}")
