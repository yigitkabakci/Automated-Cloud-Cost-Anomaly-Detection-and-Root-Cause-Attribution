import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import generate_cost_data
from detection.prophet_engine import detect_prophet
from detection.sarima_engine import detect_sarima


def _base_df():
    return generate_cost_data()


def test_prophet_output_columns():
    df = detect_prophet(_base_df())
    assert "prophet_anomaly" in df.columns


def test_prophet_anomaly_type():
    df = detect_prophet(_base_df())
    assert df["prophet_anomaly"].dtype == bool


def test_sarima_output_columns():
    df = detect_sarima(_base_df())
    assert "sarima_anomaly" in df.columns


def test_sarima_anomaly_type():
    df = detect_sarima(_base_df())
    assert df["sarima_anomaly"].dtype == bool
