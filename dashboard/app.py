import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from data.generator import generate_cost_data
from detection.zscore_engine import detect_zscore
from detection.isolation_forest import detect_isolation_forest
from evaluation.metrics import calculate_metrics
import config


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cloud Cost Anomaly Detection", layout="wide")

st.title("☁️ Cloud Cost Anomaly Detection")
st.markdown("Automated anomaly detection using Z-Score and Isolation Forest")


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    zscore_threshold = st.slider(
        "Z-Score Threshold", min_value=1.0, max_value=5.0, value=2.5, step=0.1
    )
    if_contamination = st.slider(
        "IF Contamination Rate", min_value=0.01, max_value=0.10, value=0.03, step=0.01
    )
    rolling_window = st.slider(
        "Rolling Window", min_value=7, max_value=60, value=30, step=1
    )
    regenerate = st.button("🔄 Regenerate Data")


# ── Data generation (session state cache) ─────────────────────────────────────
if "raw_df" not in st.session_state or regenerate:
    st.session_state["raw_df"] = generate_cost_data()

raw_df = st.session_state["raw_df"]


# ── Detection (respects slider values by monkey-patching config) ──────────────
config.ZSCORE_THRESHOLD = zscore_threshold
config.IF_CONTAMINATION = if_contamination

import detection.zscore_engine as _zmod
import detection.isolation_forest as _ifmod
_zmod.ZSCORE_THRESHOLD = zscore_threshold
_ifmod.IF_CONTAMINATION = if_contamination

# Also override rolling window inside zscore engine at call time via a wrapper
import numpy as np
from statsmodels.tsa.seasonal import STL

def _detect_zscore_custom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stl = STL(df["cost"], period=config.STL_PERIOD, robust=True)
    result = stl.fit()
    residual = result.resid
    rm = residual.rolling(window=rolling_window, center=True, min_periods=1).mean()
    rs = residual.rolling(window=rolling_window, center=True, min_periods=1).std().fillna(1)
    zscore = (residual - rm) / rs
    df["zscore_anomaly"] = zscore.abs() > zscore_threshold
    return df

df = _detect_zscore_custom(raw_df)
df = detect_isolation_forest(df)


# ── Main chart ────────────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["date"], y=df["cost"],
    mode="lines",
    name="Daily Cost",
    line=dict(color="royalblue", width=1.5),
))

ground_truth = df[df["is_anomaly"]]
fig.add_trace(go.Scatter(
    x=ground_truth["date"], y=ground_truth["cost"],
    mode="markers",
    name="Ground Truth Anomaly",
    marker=dict(color="red", size=10, symbol="circle"),
))

zscore_hits = df[df["zscore_anomaly"]]
fig.add_trace(go.Scatter(
    x=zscore_hits["date"], y=zscore_hits["cost"],
    mode="markers",
    name="Z-Score Detection",
    marker=dict(color="orange", size=10, symbol="x"),
))

if_hits = df[df["if_anomaly"]]
fig.add_trace(go.Scatter(
    x=if_hits["date"], y=if_hits["cost"],
    mode="markers",
    name="Isolation Forest Detection",
    marker=dict(color="mediumpurple", size=10, symbol="square"),
))

fig.update_layout(
    title="Daily Cloud Cost with Anomaly Detection",
    xaxis_title="Date",
    yaxis_title="Cost ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=480,
)

st.plotly_chart(fig, use_container_width=True)


# ── Metric cards ──────────────────────────────────────────────────────────────
y_true = df["is_anomaly"]
zscore_m = calculate_metrics(y_true, df["zscore_anomaly"], "Z-Score (STL)")
if_m = calculate_metrics(y_true, df["if_anomaly"], "Isolation Forest")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Ground Truth")
    total_anomalies = int(df["is_anomaly"].sum())
    anomaly_rate = total_anomalies / len(df) * 100
    st.metric("Total Anomalies", total_anomalies)
    st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

with col2:
    st.subheader("Z-Score Results")
    st.metric("Precision", f"{zscore_m['precision']:.4f}")
    st.metric("Recall",    f"{zscore_m['recall']:.4f}")
    st.metric("F1 Score",  f"{zscore_m['f1']:.4f}")

with col3:
    st.subheader("Isolation Forest Results")
    st.metric("Precision", f"{if_m['precision']:.4f}")
    st.metric("Recall",    f"{if_m['recall']:.4f}")
    st.metric("F1 Score",  f"{if_m['f1']:.4f}")


# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("Model Comparison")

comparison_df = pd.DataFrame([zscore_m, if_m]).set_index("model")[
    ["precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"]
]
st.dataframe(comparison_df, use_container_width=True)


# ── Raw data expander ─────────────────────────────────────────────────────────
with st.expander("📊 Raw Data"):
    st.dataframe(df, use_container_width=True)
