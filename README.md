# Automated Cloud Cost Anomaly Detection and Root Cause Attribution

A Python-based system for detecting anomalies in cloud infrastructure costs and attributing root causes to specific services, regions, or usage patterns.

## Features

- Statistical anomaly detection (Z-score, IQR, Isolation Forest)
- Time-series analysis with seasonality-aware thresholds
- Interactive Streamlit dashboard with Plotly visualizations
- Root cause attribution across services and regions
- Automated evaluation metrics for detection accuracy

## Project Structure

```
├── data/          # Raw and processed cost datasets
├── detection/     # Anomaly detection algorithms
├── dashboard/     # Streamlit dashboard app
├── evaluation/    # Detection evaluation and metrics
├── tests/         # Unit and integration tests
├── config.py      # Project-wide constants
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Run Tests

```bash
pytest tests/
```
