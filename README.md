# Automated Cloud Cost Anomaly Detection and Root-Cause Attribution

**Konya Food and Agriculture University**  
Mert Kırçiçek (232010020044) · Yiğit Kabakcı (212010020092)

---

## Overview

This project treats cloud billing data as a real-time observability telemetry stream.
It detects anomalous expenditures using four different detection engines and visualizes
results on an interactive Streamlit dashboard with automated Slack alerting.

---

## Features

- 4 anomaly detection algorithms: Z-Score + STL, Isolation Forest, Prophet, SARIMA
- 3 data sources: Synthetic, Hybrid (Synthetic + Kaggle), Multi-Service
- Multi-service analysis: EC2, S3, Lambda, RDS with individual anomaly tracking
- Root-cause attribution: identifies which service caused each anomaly
- Automated Slack notifications with per-anomaly details
- Interactive parameter tuning via dashboard sliders
- Date range filtering and peak anomaly highlighting

---

## Project Structure

```
cloud-cost-anomaly/

├── data/

│   ├── generator.py           # Synthetic + multi-service data generator

│   ├── kaggle_loader.py       # Kaggle dataset loader

│   └── hybrid_loader.py       # Hybrid dataset (synthetic + Kaggle patterns)

├── detection/

│   ├── zscore_engine.py       # STL decomposition + rolling Z-score

│   ├── isolation_forest.py    # Isolation Forest

│   ├── prophet_engine.py      # Facebook Prophet

│   ├── sarima_engine.py       # SARIMA

│   └── multiservice_detector.py # Per-service anomaly detection + root-cause

├── dashboard/

│   └── app.py                 # Streamlit interactive dashboard

├── evaluation/

│   └── metrics.py             # Precision, Recall, F1 comparison

├── notifications/

│   └── slack_notifier.py      # Slack webhook alerting

├── tests/                     # Unit tests for all modules

├── main.py                    # Pipeline entry point

├── config.py                  # Central configuration

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

Run all tests:
```bash
pytest tests/ -v
```

---

## Detection Methods

| Method | Type | Strength |
|---|---|---|
| Z-Score + STL | Statistical | Fast, interpretable, handles seasonality |
| Isolation Forest | Machine Learning | No assumption on data distribution |
| Prophet | Time-Series ML | Trend + seasonality decomposition |
| SARIMA | Statistical | Strong baseline for seasonal data |

---

## Data Sources

| Source | Description |
|---|---|
| Synthetic | 365-day generated time-series with injected anomalies |
| Hybrid | Synthetic trend + real Kaggle anomaly patterns |
| Multi-Service | EC2, S3, Lambda, RDS individual cost streams |

---

## Slack Notifications

Add your webhook URL to `.env`:
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

Enable alerts from the dashboard sidebar.

---

## Tech Stack

Python 3.10 · Pandas · NumPy · Scikit-learn · Statsmodels · Prophet · SciPy · Streamlit · Plotly · Slack API

---

## Results

On synthetic data, Prophet and SARIMA achieve perfect detection (F1=1.0) due to the
structured nature of generated data. On hybrid data, Isolation Forest demonstrates
stronger generalization with fewer false positives compared to Z-Score.
