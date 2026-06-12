import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR


def test_file_exists():
    assert (DATA_DIR / "cloud_data.csv").exists()


def test_output_columns():
    from data.kaggle_loader import load_kaggle_data
    df = load_kaggle_data()
    for col in ["date", "cost", "is_anomaly"]:
        assert col in df.columns, f"Missing column: {col}"


def test_no_negative_cost():
    from data.kaggle_loader import load_kaggle_data
    df = load_kaggle_data()
    assert (df["cost"] >= 0).all()


def test_anomaly_exists():
    from data.kaggle_loader import load_kaggle_data
    df = load_kaggle_data()
    assert df["is_anomaly"].sum() >= 1
