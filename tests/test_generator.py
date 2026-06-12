import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import generate_cost_data
from config import N_ANOMALIES


def test_shape():
    df = generate_cost_data()
    assert len(df) == 365


def test_anomaly_count():
    df = generate_cost_data()
    assert df["is_anomaly"].sum() == N_ANOMALIES


def test_no_negative_cost():
    df = generate_cost_data()
    assert (df["cost"] >= 0).all()


def test_columns():
    df = generate_cost_data()
    assert list(df.columns) == ["date", "cost", "is_anomaly"]
