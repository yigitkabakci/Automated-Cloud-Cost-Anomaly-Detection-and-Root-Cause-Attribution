import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import SEED


def detect_prophet(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Prophet is not installed. Run: pip install prophet")

    df = df.copy()

    prophet_df = df[["date", "cost"]].rename(columns={"date": "ds", "cost": "y"})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )

    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    model.fit(prophet_df)

    forecast = model.predict(prophet_df[["ds"]])

    residual = prophet_df["y"].values - forecast["yhat"].values
    threshold = 2 * residual.std()
    df["prophet_anomaly"] = np.abs(residual) > threshold

    return df
