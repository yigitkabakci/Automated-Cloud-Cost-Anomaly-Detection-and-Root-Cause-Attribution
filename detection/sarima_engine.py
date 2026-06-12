import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import STL_PERIOD


def detect_sarima(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            df["cost"],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, STL_PERIOD),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

    fitted = result.fittedvalues
    residual = df["cost"] - fitted
    threshold = 2 * residual.std()
    df["sarima_anomaly"] = residual.abs() > threshold

    return df
