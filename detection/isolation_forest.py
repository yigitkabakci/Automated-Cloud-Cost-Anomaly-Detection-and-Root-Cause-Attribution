import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import IF_CONTAMINATION, IF_N_ESTIMATORS, SEED


def detect_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["rolling_mean_7"] = df["cost"].rolling(window=7, min_periods=1).mean()
    df["rolling_std_7"] = df["cost"].rolling(window=7, min_periods=1).std()

    features = ["cost", "day_of_week", "rolling_mean_7", "rolling_std_7"]
    valid_mask = df[features].notna().all(axis=1)
    X = df.loc[valid_mask, features]

    model = IsolationForest(
        contamination=IF_CONTAMINATION,
        n_estimators=IF_N_ESTIMATORS,
        random_state=SEED,
    )
    preds = model.fit_predict(X)

    df["if_anomaly"] = False
    df.loc[valid_mask, "if_anomaly"] = preds == -1

    df.drop(columns=["day_of_week", "rolling_mean_7", "rolling_std_7"], inplace=True)

    return df
