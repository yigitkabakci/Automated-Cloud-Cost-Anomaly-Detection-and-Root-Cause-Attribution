import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import generate_multiservice_data
from detection.multiservice_detector import detect_multiservice, root_cause_attribution

_SERVICES = ["ec2", "s3", "lambda", "rds"]


def _full_df():
    df = generate_multiservice_data()
    df = detect_multiservice(df)
    df = root_cause_attribution(df)
    return df


def test_multiservice_columns():
    df = generate_multiservice_data()
    for svc in _SERVICES:
        assert f"cost_{svc}"       in df.columns
        assert f"is_anomaly_{svc}" in df.columns
    assert "cost_total"     in df.columns
    assert "is_anomaly_any" in df.columns


def test_multiservice_shape():
    assert len(generate_multiservice_data()) == 365


def test_root_cause_column():
    df = _full_df()
    assert "root_cause" in df.columns


def test_service_anomalies():
    df = generate_multiservice_data()
    for svc in _SERVICES:
        assert df[f"is_anomaly_{svc}"].sum() >= 1, f"No anomalies for {svc}"
