import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import DATA_DIR

CSV_PATH = DATA_DIR / "cloud_data.csv"


def load_kaggle_data() -> pd.DataFrame:
    raw = pd.read_csv(CSV_PATH)
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])

    raw["cost"] = (
        raw["CPU_Usage"]    * 0.0464 +
        raw["Memory_Usage"] * 0.0116 +
        raw["Disk_IO"]      * 0.0001 +
        raw["Network_IO"]   * 0.0009
    ).clip(lower=0)

    return pd.DataFrame({
        "date":       raw["Timestamp"],
        "cost":       raw["cost"],
        "is_anomaly": raw["Anomaly_Label"].astype(bool),
        "cpu":        raw["CPU_Usage"],
        "memory":     raw["Memory_Usage"],
        "workload":   raw["Workload_Type"],
    }).reset_index(drop=True)


if __name__ == "__main__":
    df = load_kaggle_data()
    print(df.head(10).to_string())
    print(f"\nToplam satır sayısı : {len(df)}")
    print(f"Anomali sayısı      : {df['is_anomaly'].sum()}")
    print(f"Anomali oranı       : {df['is_anomaly'].mean() * 100:.2f}%")
    print(f"Min maliyet         : {df['cost'].min():.4f}")
    print(f"Max maliyet         : {df['cost'].max():.4f}")
    print(f"Ortalama maliyet    : {df['cost'].mean():.4f}")
