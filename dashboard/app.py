import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

from data.generator import generate_cost_data
from detection.zscore_engine import detect_zscore
from detection.isolation_forest import detect_isolation_forest
from detection.prophet_engine import detect_prophet
from detection.sarima_engine import detect_sarima
from evaluation.metrics import calculate_metrics
import config


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cloud Cost Anomaly Detection", layout="wide")

st.title("☁️ Cloud Cost Anomaly Detection")
st.markdown("Automated anomaly detection using Z-Score, Isolation Forest, Prophet and SARIMA")


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

    st.divider()
    show_prophet = st.checkbox("Show Prophet", value=True)
    show_sarima  = st.checkbox("Show SARIMA",  value=True)

    st.divider()
    regenerate = st.button("🔄 Regenerate Data")


# ── Data generation (session state cache) ─────────────────────────────────────
if "raw_df" not in st.session_state or regenerate:
    st.session_state["raw_df"] = generate_cost_data()
    # clear slow-model cache when data changes
    st.session_state.pop("prophet_df", None)
    st.session_state.pop("sarima_df",  None)

raw_df = st.session_state["raw_df"]


# ── Fast detections (re-run on every slider interaction) ──────────────────────
def _detect_zscore_custom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stl = STL(df["cost"], period=config.STL_PERIOD, robust=True)
    res = stl.fit()
    residual = res.resid
    rm = residual.rolling(window=rolling_window, center=True, min_periods=1).mean()
    rs = residual.rolling(window=rolling_window, center=True, min_periods=1).std().fillna(1)
    df["zscore_anomaly"] = (np.abs((residual - rm) / rs)) > zscore_threshold
    return df

config.IF_CONTAMINATION = if_contamination
import detection.isolation_forest as _ifmod
_ifmod.IF_CONTAMINATION = if_contamination

df = _detect_zscore_custom(raw_df)
df = detect_isolation_forest(df)


# ── Slow detections (cached per raw_df identity) ──────────────────────────────
if "prophet_df" not in st.session_state:
    with st.spinner("Fitting Prophet model…"):
        st.session_state["prophet_df"] = detect_prophet(raw_df)

if "sarima_df" not in st.session_state:
    with st.spinner("Fitting SARIMA model…"):
        st.session_state["sarima_df"] = detect_sarima(raw_df)

df["prophet_anomaly"] = st.session_state["prophet_df"]["prophet_anomaly"].values
df["sarima_anomaly"]  = st.session_state["sarima_df"]["sarima_anomaly"].values


# ── Main chart ────────────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["date"], y=df["cost"],
    mode="lines", name="Daily Cost",
    line=dict(color="royalblue", width=1.5),
))

ground_truth = df[df["is_anomaly"]]
fig.add_trace(go.Scatter(
    x=ground_truth["date"], y=ground_truth["cost"],
    mode="markers", name="Ground Truth Anomaly",
    marker=dict(color="red", size=10, symbol="circle"),
))

zscore_hits = df[df["zscore_anomaly"]]
fig.add_trace(go.Scatter(
    x=zscore_hits["date"], y=zscore_hits["cost"],
    mode="markers", name="Z-Score Detection",
    marker=dict(color="orange", size=10, symbol="x"),
))

if_hits = df[df["if_anomaly"]]
fig.add_trace(go.Scatter(
    x=if_hits["date"], y=if_hits["cost"],
    mode="markers", name="Isolation Forest Detection",
    marker=dict(color="mediumpurple", size=10, symbol="square"),
))

if show_prophet:
    prophet_hits = df[df["prophet_anomaly"]]
    fig.add_trace(go.Scatter(
        x=prophet_hits["date"], y=prophet_hits["cost"],
        mode="markers", name="Prophet Detection",
        marker=dict(color="limegreen", size=11, symbol="triangle-up"),
    ))

if show_sarima:
    sarima_hits = df[df["sarima_anomaly"]]
    fig.add_trace(go.Scatter(
        x=sarima_hits["date"], y=sarima_hits["cost"],
        mode="markers", name="SARIMA Detection",
        marker=dict(color="gold", size=11, symbol="diamond"),
    ))

fig.update_layout(
    title="Daily Cloud Cost with Anomaly Detection",
    xaxis_title="Date",
    yaxis_title="Cost ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500,
)

st.plotly_chart(fig, use_container_width=True)


# ── Metric cards (4 columns) ──────────────────────────────────────────────────
y_true    = df["is_anomaly"]
zscore_m  = calculate_metrics(y_true, df["zscore_anomaly"],  "Z-Score (STL)")
if_m      = calculate_metrics(y_true, df["if_anomaly"],      "Isolation Forest")
prophet_m = calculate_metrics(y_true, df["prophet_anomaly"], "Prophet")
sarima_m  = calculate_metrics(y_true, df["sarima_anomaly"],  "SARIMA")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Ground Truth")
    total_anomalies = int(df["is_anomaly"].sum())
    st.metric("Total Anomalies", total_anomalies)
    st.metric("Anomaly Rate", f"{total_anomalies / len(df) * 100:.1f}%")

with col2:
    st.subheader("Z-Score")
    st.metric("Precision", f"{zscore_m['precision']:.4f}")
    st.metric("Recall",    f"{zscore_m['recall']:.4f}")
    st.metric("F1 Score",  f"{zscore_m['f1']:.4f}")

with col3:
    st.subheader("Isolation Forest")
    st.metric("Precision", f"{if_m['precision']:.4f}")
    st.metric("Recall",    f"{if_m['recall']:.4f}")
    st.metric("F1 Score",  f"{if_m['f1']:.4f}")

with col4:
    st.subheader("Prophet")
    st.metric("Precision", f"{prophet_m['precision']:.4f}")
    st.metric("Recall",    f"{prophet_m['recall']:.4f}")
    st.metric("F1 Score",  f"{prophet_m['f1']:.4f}")

    st.subheader("SARIMA")
    st.metric("Precision", f"{sarima_m['precision']:.4f}")
    st.metric("Recall",    f"{sarima_m['recall']:.4f}")
    st.metric("F1 Score",  f"{sarima_m['f1']:.4f}")


# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("Model Comparison")

comparison_df = pd.DataFrame([zscore_m, if_m, prophet_m, sarima_m]).set_index("model")[
    ["precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"]
]
st.dataframe(comparison_df, use_container_width=True)


# ── Raw data expander ─────────────────────────────────────────────────────────
with st.expander("📊 Raw Data"):
    st.dataframe(df, use_container_width=True)
