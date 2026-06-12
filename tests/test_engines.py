import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data.generator import generate_cost_data
from detection.zscore_engine import detect_zscore
from detection.isolation_forest import detect_isolation_forest
from evaluation.metrics import calculate_metrics


def _base_df():
    return generate_cost_data()


def test_zscore_output_columns():
    df = detect_zscore(_base_df())
    assert "zscore_anomaly" in df.columns


def test_zscore_anomaly_type():
    df = detect_zscore(_base_df())
    assert df["zscore_anomaly"].dtype == bool


def test_if_output_columns():
    df = detect_isolation_forest(_base_df())
    assert "if_anomaly" in df.columns


def test_if_anomaly_type():
    df = detect_isolation_forest(_base_df())
    assert df["if_anomaly"].dtype == bool


def test_metrics_keys():
    y_true = [True, False, True, False, True]
    y_pred = [True, False, False, True, True]
    result = calculate_metrics(y_true, y_pred, "test_model")
    expected_keys = {"model", "precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"}
    assert set(result.keys()) == expected_keys
