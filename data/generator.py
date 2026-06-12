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


_SERVICE_PROFILES = {
    "ec2":    dict(base=120, trend=0.10, seasonality=30, noise=12),
    "s3":     dict(base=40,  trend=0.05, seasonality=8,  noise=4),
    "lambda": dict(base=25,  trend=0.08, seasonality=15, noise=8),
    "rds":    dict(base=60,  trend=0.06, seasonality=10, noise=6),
}

_SERVICE_N_ANOMALIES = 8


def _generate_service_series(rng, profile: dict) -> tuple:
    t = np.arange(N_DAYS)
    cost = (
        profile["base"]
        + profile["trend"] * t
        + profile["seasonality"] * np.sin(2 * np.pi * t / 7)
        + rng.normal(0, profile["noise"], N_DAYS)
    )
    cost = np.maximum(cost, 0)
    is_anomaly = np.zeros(N_DAYS, dtype=bool)
    idx = rng.choice(N_DAYS, size=_SERVICE_N_ANOMALIES, replace=False)
    mults = rng.uniform(*ANOMALY_MULTIPLIER_RANGE, size=_SERVICE_N_ANOMALIES)
    cost[idx] *= mults
    is_anomaly[idx] = True
    return cost, is_anomaly


def generate_multiservice_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(start="2024-01-01", periods=N_DAYS, freq="D")

    data: dict = {"date": dates}
    for svc, profile in _SERVICE_PROFILES.items():
        cost, is_anom = _generate_service_series(rng, profile)
        data[f"cost_{svc}"]       = cost
        data[f"is_anomaly_{svc}"] = is_anom

    df = pd.DataFrame(data)
    df["cost_total"]    = df[["cost_ec2", "cost_s3", "cost_lambda", "cost_rds"]].sum(axis=1)
    df["is_anomaly_any"] = (
        df["is_anomaly_ec2"]
        | df["is_anomaly_s3"]
        | df["is_anomaly_lambda"]
        | df["is_anomaly_rds"]
    )
    return df


if __name__ == "__main__":
    df = generate_cost_data()
    print(df.head(10).to_string())
    print(f"\nToplam gün sayısı : {len(df)}")
    print(f"Anomali sayısı    : {df['is_anomaly'].sum()}")
    print(f"Min maliyet       : {df['cost'].min():.2f}")
    print(f"Max maliyet       : {df['cost'].max():.2f}")
    print(f"Ortalama maliyet  : {df['cost'].mean():.2f}")
