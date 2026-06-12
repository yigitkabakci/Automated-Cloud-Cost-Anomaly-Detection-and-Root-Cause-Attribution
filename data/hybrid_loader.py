import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    SEED,
    N_DAYS,
    BASE_COST,
    TREND_SLOPE,
    SEASONALITY_AMPLITUDE,
    NOISE_STD,
)

CSV_PATH = DATA_DIR / "cloud_data.csv"


def load_hybrid_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # ── 1. Synthetic baseline (trend + seasonality + noise, no anomalies) ────
    dates = pd.date_range(start="2024-01-01", periods=N_DAYS, freq="D")
    t = np.arange(N_DAYS)
    cost = (
        BASE_COST
        + TREND_SLOPE * t
        + SEASONALITY_AMPLITUDE * np.sin(2 * np.pi * t / 7)
        + rng.normal(0, NOISE_STD, N_DAYS)
    )
    cost = np.maximum(cost, 0)
    is_anomaly = np.zeros(N_DAYS, dtype=bool)

    # ── 2. Kaggle anomaly statistics ─────────────────────────────────────────
    raw = pd.read_csv(CSV_PATH)
    anomaly_rows = raw[raw["Anomaly_Label"] == 1].copy()
    total_kaggle  = len(raw)
    kaggle_rate   = len(anomaly_rows) / total_kaggle

    anomaly_rows["kaggle_cost"] = (
        anomaly_rows["CPU_Usage"]    * 0.0464
        + anomaly_rows["Memory_Usage"] * 0.0116
        + anomaly_rows["Disk_IO"]      * 0.0001
        + anomaly_rows["Network_IO"]   * 0.0009
    ).clip(lower=0)
    anomaly_magnitude = anomaly_rows["kaggle_cost"].mean()

    # ── 3. Inject Kaggle-pattern anomalies into synthetic series ─────────────
    n_inject = max(1, round(N_DAYS * kaggle_rate))
    inject_idx = rng.choice(N_DAYS, size=n_inject, replace=False)
    multipliers = rng.uniform(2.5, 6.0, size=n_inject)
    cost[inject_idx] = anomaly_magnitude * multipliers
    is_anomaly[inject_idx] = True

    df = pd.DataFrame({"date": dates, "cost": cost, "is_anomaly": is_anomaly})

    # store metadata as module-level for tests
    load_hybrid_data._kaggle_rate   = kaggle_rate
    load_hybrid_data._n_inject      = n_inject

    return df


def load_hybrid_multiservice_data() -> pd.DataFrame:
    from data.generator import generate_multiservice_data, _SERVICE_PROFILES, ANOMALY_MULTIPLIER_RANGE

    rng = np.random.default_rng(SEED)

    df = generate_multiservice_data()

    # Kaggle anomaly magnitude
    raw = pd.read_csv(CSV_PATH)
    anomaly_rows = raw[raw["Anomaly_Label"] == 1].copy()
    kaggle_rate = len(anomaly_rows) / len(raw)
    anomaly_rows["kaggle_cost"] = (
        anomaly_rows["CPU_Usage"]    * 0.0464
        + anomaly_rows["Memory_Usage"] * 0.0116
        + anomaly_rows["Disk_IO"]      * 0.0001
        + anomaly_rows["Network_IO"]   * 0.0009
    ).clip(lower=0)
    anomaly_magnitude = anomaly_rows["kaggle_cost"].mean()

    n_inject = max(1, round(N_DAYS * kaggle_rate))

    for svc in _SERVICE_PROFILES:
        cost_col  = f"cost_{svc}"
        anom_col  = f"is_anomaly_{svc}"
        inject_idx = rng.choice(N_DAYS, size=n_inject, replace=False)
        multipliers = rng.uniform(2.5, 6.0, size=n_inject)
        df.loc[inject_idx, cost_col] = anomaly_magnitude * multipliers
        df.loc[inject_idx, anom_col] = True

    df["cost_total"]    = df[["cost_ec2", "cost_s3", "cost_lambda", "cost_rds"]].sum(axis=1)
    df["is_anomaly_any"] = (
        df["is_anomaly_ec2"]
        | df["is_anomaly_s3"]
        | df["is_anomaly_lambda"]
        | df["is_anomaly_rds"]
    )
    return df


if __name__ == "__main__":
    df = load_hybrid_data()
    print(df.head(10).to_string())
    print(f"\nToplam gün           : {len(df)}")
    print(f"Kaggle anomali oranı : {load_hybrid_data._kaggle_rate * 100:.2f}%")
    print(f"Inject edilen anomali: {load_hybrid_data._n_inject}")
    print(f"Min maliyet          : {df['cost'].min():.2f}")
    print(f"Max maliyet          : {df['cost'].max():.2f}")
    print(f"Ortalama maliyet     : {df['cost'].mean():.2f}")
