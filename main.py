import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.generator import generate_cost_data
from detection.zscore_engine import detect_zscore
from detection.isolation_forest import detect_isolation_forest
from detection.prophet_engine import detect_prophet
from detection.sarima_engine import detect_sarima
from evaluation.metrics import compare_models
from config import DATA_DIR


def main():
    print("Generating cost data...")
    df = generate_cost_data()

    print("Running Z-Score (STL) detection...")
    df = detect_zscore(df)

    print("Running Isolation Forest detection...")
    df = detect_isolation_forest(df)

    print("Running Prophet detection...")
    df = detect_prophet(df)

    print("Running SARIMA detection...")
    df = detect_sarima(df)

    compare_models(df)

    output_path = DATA_DIR / "results.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
