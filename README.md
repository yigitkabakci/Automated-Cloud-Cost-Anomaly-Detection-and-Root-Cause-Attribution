# Automated Cloud Cost Anomaly Detection and Root-Cause Attribution

**Konya Food and Agriculture University**
Mert Kırçiçek (232010020044) · Yiğit Kabakcı (212010020092)

---

## Overview

This project treats cloud billing data as a real-time observability telemetry stream.
It detects anomalous expenditures using statistical (Z-Score + STL) and machine learning
(Isolation Forest) methods, and visualizes results on an interactive Streamlit dashboard.

---

## Project Structure

```
cloud-cost-anomaly/

├── data/

│   └── generator.py        # Synthetic time-series data generator

├── detection/

│   ├── zscore_engine.py     # STL decomposition + rolling Z-score

│   └── isolation_forest.py  # Isolation Forest anomaly detection

├── dashboard/

│   └── app.py              # Streamlit interactive dashboard

├── evaluation/

│   └── metrics.py          # Precision, Recall, F1 comparison

├── tests/                  # Unit tests for all modules

├── main.py                 # Pipeline entry point

├── config.py               # Central configuration

└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

Run full pipeline:
```bash
python main.py
```

Launch dashboard:
```bash
streamlit run dashboard/app.py
```

Run tests:
```bash
pytest tests/ -v
```

---

## Detection Methods

| Method | Approach | Strength |
|---|---|---|
| Z-Score + STL | Statistical | Fast, interpretable |
| Isolation Forest | Machine Learning | Handles complex patterns |

---

## Tech Stack

Python 3.10 · Pandas · NumPy · Scikit-learn · Statsmodels · SciPy · Streamlit · Plotly
