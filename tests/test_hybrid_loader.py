import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data.hybrid_loader import load_hybrid_data
from config import DATA_DIR, N_DAYS


def _df():
    return load_hybrid_data()


def test_shape():
    assert len(_df()) == 365


def test_columns():
    df = _df()
    for col in ["date", "cost", "is_anomaly"]:
        assert col in df.columns, f"Missing column: {col}"


def test_no_negative_cost():
    assert (_df()["cost"] >= 0).all()


def test_anomaly_exists():
    assert _df()["is_anomaly"].sum() >= 1


def test_kaggle_pattern():
    import pandas as _pd

    raw = _pd.read_csv(DATA_DIR / "cloud_data.csv")
    kaggle_rate = raw["Anomaly_Label"].mean()
    expected = N_DAYS * kaggle_rate

    df = load_hybrid_data()
    actual = int(df["is_anomaly"].sum())

    tolerance = 0.02 * N_DAYS          # ±2 % of 365 ≈ ±7.3 days
    assert abs(actual - expected) <= tolerance, (
        f"Anomaly count {actual} not within ±2% of expected {expected:.1f}"
    )
