import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(y_true, y_pred, model_name: str) -> dict:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    return {
        "model": model_name,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def compare_models(df: pd.DataFrame) -> None:
    y_true = df["is_anomaly"]

    rows = [
        calculate_metrics(y_true, df["zscore_anomaly"],    "Z-Score (STL)"),
        calculate_metrics(y_true, df["if_anomaly"],        "Isolation Forest"),
        calculate_metrics(y_true, df["prophet_anomaly"],   "Prophet"),
        calculate_metrics(y_true, df["sarima_anomaly"],    "SARIMA"),
    ]

    result_df = pd.DataFrame(rows).set_index("model")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(result_df.to_string())
    print("=" * 60)

    best = max(rows, key=lambda r: r["f1"])
    print(f"Best Model: {best['model']}  (F1 = {best['f1']:.4f})\n")
