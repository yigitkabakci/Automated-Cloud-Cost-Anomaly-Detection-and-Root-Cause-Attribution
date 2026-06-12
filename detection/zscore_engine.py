import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from config import STL_PERIOD, ZSCORE_THRESHOLD


def detect_zscore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    stl = STL(df["cost"], period=STL_PERIOD, robust=True)
    result = stl.fit()
    residual = result.resid

    rolling_mean = residual.rolling(window=30, center=True, min_periods=1).mean()
    rolling_std = residual.rolling(window=30, center=True, min_periods=1).std().fillna(1)

    zscore = (residual - rolling_mean) / rolling_std
    df["zscore_anomaly"] = zscore.abs() > ZSCORE_THRESHOLD

    return df
