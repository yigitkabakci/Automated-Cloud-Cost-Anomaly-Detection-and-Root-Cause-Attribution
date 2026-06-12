import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest

from config import STL_PERIOD, ZSCORE_THRESHOLD, IF_CONTAMINATION, IF_N_ESTIMATORS, SEED

_SERVICES = ["ec2", "s3", "lambda", "rds"]


def _zscore_series(cost: pd.Series) -> pd.Series:
    stl = STL(cost, period=STL_PERIOD, robust=True)
    residual = stl.fit().resid
    rm = residual.rolling(window=30, center=True, min_periods=1).mean()
    rs = residual.rolling(window=30, center=True, min_periods=1).std().fillna(1)
    return (residual - rm).abs() / rs > ZSCORE_THRESHOLD


def _if_series(df: pd.DataFrame, cost_col: str) -> pd.Series:
    tmp = df[["date", cost_col]].rename(columns={cost_col: "cost"}).copy()
    tmp["dow"]  = pd.to_datetime(tmp["date"]).dt.dayofweek
    tmp["rm7"]  = tmp["cost"].rolling(7, min_periods=1).mean()
    tmp["rs7"]  = tmp["cost"].rolling(7, min_periods=1).std()
    feats = ["cost", "dow", "rm7", "rs7"]
    mask  = tmp[feats].notna().all(axis=1)
    X     = tmp.loc[mask, feats]
    preds = IsolationForest(
        contamination=IF_CONTAMINATION,
        n_estimators=IF_N_ESTIMATORS,
        random_state=SEED,
    ).fit_predict(X)
    result = pd.Series(False, index=tmp.index)
    result[mask] = preds == -1
    return result


def detect_multiservice(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for svc in _SERVICES:
        cost_col = f"cost_{svc}"
        df[f"zscore_anomaly_{svc}"] = _zscore_series(df[cost_col])
        df[f"if_anomaly_{svc}"]     = _if_series(df, cost_col)
    return df


def root_cause_attribution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cost_cols = [f"cost_{svc}" for svc in _SERVICES]

    # rolling baseline per service (previous 7 days mean)
    baselines = {
        col: df[col].shift(1).rolling(7, min_periods=1).mean()
        for col in cost_cols
    }
    baseline_df = pd.DataFrame(baselines, index=df.index).fillna(
        df[cost_cols].mean()
    )

    pct_increase = (df[cost_cols].values - baseline_df.values) / (baseline_df.values + 1e-9)

    root_cause_idx = np.argmax(pct_increase, axis=1)
    svc_labels = [svc.upper() for svc in _SERVICES]
    df["root_cause"] = [
        svc_labels[i] if df["is_anomaly_any"].iloc[row] else ""
        for row, i in enumerate(root_cause_idx)
    ]
    return df
