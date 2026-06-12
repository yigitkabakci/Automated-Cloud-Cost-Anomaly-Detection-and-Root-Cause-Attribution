import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

from data.generator import generate_cost_data, generate_multiservice_data
from data.kaggle_loader import load_kaggle_data
from data.hybrid_loader import load_hybrid_data
from detection.zscore_engine import detect_zscore
from detection.isolation_forest import detect_isolation_forest
from detection.prophet_engine import detect_prophet
from detection.sarima_engine import detect_sarima
from detection.multiservice_detector import detect_multiservice, root_cause_attribution
from evaluation.metrics import calculate_metrics
from notifications.slack_notifier import send_slack_alert, test_slack_connection
import config


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cloud Cost Anomaly Detection", layout="wide")

st.title("☁️ Cloud Cost Anomaly Detection")
st.markdown("Automated anomaly detection using Z-Score, Isolation Forest, Prophet and SARIMA")


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    data_source = st.radio(
        "Data Source",
        ["Synthetic Data", "Hybrid Dataset", "Multi-Service View", "Kaggle Real Data"],
    )

    if data_source == "Kaggle Real Data":
        st.info("Dataset: Kaggle Cloud Resource Usage\n14,400 dakika bazında kayıt")
    elif data_source == "Hybrid Dataset":
        st.info("Sentetik trend + Kaggle anomali pattern'ları\n365 günlük hibrit veri")
    elif data_source == "Multi-Service View":
        st.info("EC2 · S3 · Lambda · RDS — servis bazlı anomali tespiti")

    st.divider()

    if data_source == "Multi-Service View":
        selected_services = st.multiselect(
            "Services",
            ["EC2", "S3", "Lambda", "RDS"],
            default=["EC2", "S3", "Lambda", "RDS"],
        )
        regenerate = st.button("🔄 Regenerate Data")
    else:
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
        regenerate = st.button("🔄 Regenerate / Reload Data")

    # ── Slack notifications (all sources) ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔔 Notifications")

    webhook_url = st.text_input(
        "Slack Webhook URL",
        value=os.getenv("SLACK_WEBHOOK_URL", ""),
        type="password",
    )
    enable_slack = st.checkbox("Enable Slack Alerts", value=False)

    if st.button("Test Connection"):
        if webhook_url:
            if test_slack_connection(webhook_url):
                st.success("✅ Connected!")
            else:
                st.error("❌ Connection failed!")
        else:
            st.warning("Webhook URL gerekli")


# ── Session state: track already-sent anomalies ───────────────────────────────
if "slack_sent" not in st.session_state:
    st.session_state.slack_sent = set()


def _send_new_anomalies(anomaly_df: pd.DataFrame) -> None:
    """Send only anomalies not yet dispatched this session."""
    new = anomaly_df[
        ~anomaly_df.apply(
            lambda r: f"{r['date']}_{r['cost']}" in st.session_state.slack_sent,
            axis=1,
        )
    ]
    if len(new) == 0:
        st.info("ℹ️ Yeni anomali yok — tümü zaten bildirildi.")
        return
    if send_slack_alert(new[["date", "cost", "root_cause"]], webhook_url):
        for _, row in new.iterrows():
            st.session_state.slack_sent.add(f"{row['date']}_{row['cost']}")
        st.success(f"✅ {len(new)} yeni anomali Slack'e bildirildi!")
    else:
        st.error("❌ Slack bildirimi gönderilemedi!")


# ══════════════════════════════════════════════════════════════════════════════
# BRANCH A — Multi-Service View
# ══════════════════════════════════════════════════════════════════════════════
if data_source == "Multi-Service View":

    if "ms_df" not in st.session_state or regenerate:
        with st.spinner("Generating multi-service data…"):
            ms_raw = generate_multiservice_data()
            ms_raw = detect_multiservice(ms_raw)
            ms_raw = root_cause_attribution(ms_raw)
        st.session_state["ms_df"] = ms_raw

    ms = st.session_state["ms_df"]

    _SVC_COLOR = {"EC2": "royalblue", "S3": "limegreen", "Lambda": "orange", "RDS": "mediumpurple"}
    _SVC_LOWER = {s: s.lower() for s in ["EC2", "S3", "Lambda", "RDS"]}

    # ── Multi-service chart ───────────────────────────────────────────────────
    fig = go.Figure()
    for svc in selected_services:
        svc_l = svc.lower()
        cost_col = f"cost_{svc_l}"
        fig.add_trace(go.Scatter(
            x=ms["date"], y=ms[cost_col],
            mode="lines", name=svc,
            line=dict(color=_SVC_COLOR[svc], width=1.4),
        ))
        anom_mask = ms[f"is_anomaly_{svc_l}"]
        fig.add_trace(go.Scatter(
            x=ms.loc[anom_mask, "date"], y=ms.loc[anom_mask, cost_col],
            mode="markers", name=f"{svc} Anomaly",
            marker=dict(color=_SVC_COLOR[svc], size=10, symbol="circle",
                        line=dict(color="white", width=1)),
            showlegend=True,
        ))

    fig.update_layout(
        title="Multi-Service Cloud Cost with Anomaly Detection",
        xaxis_title="Date", yaxis_title="Cost ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Service metric cards ──────────────────────────────────────────────────
    st.subheader("Per-Service Summary")
    cols = st.columns(4)
    for i, svc in enumerate(["EC2", "S3", "Lambda", "RDS"]):
        svc_l = svc.lower()
        with cols[i]:
            st.markdown(f"**{svc}**")
            n_anom  = int(ms[f"is_anomaly_{svc_l}"].sum())
            total_c = ms[f"cost_{svc_l}"].sum()
            st.metric("Anomalies",     n_anom)
            st.metric("Total Cost",    f"${total_c:,.0f}")
            st.metric("Avg Daily",     f"${ms[f'cost_{svc_l}'].mean():.1f}")

    # ── Root-cause table ──────────────────────────────────────────────────────
    st.subheader("Root Cause Attribution")
    rc_df = ms[ms["is_anomaly_any"]][
        ["date", "cost_total", "root_cause",
         "is_anomaly_ec2", "is_anomaly_s3", "is_anomaly_lambda", "is_anomaly_rds"]
    ].reset_index(drop=True)
    st.dataframe(rc_df, use_container_width=True)

    with st.expander("📊 Raw Multi-Service Data"):
        st.dataframe(ms, use_container_width=True)

    # ── Slack alerts (multi-service) ──────────────────────────────────────────
    if enable_slack and webhook_url:
        anom_rows = ms[ms["is_anomaly_any"]].copy()
        anom_rows = anom_rows.rename(columns={"cost_total": "cost"})
        if "root_cause" not in anom_rows.columns:
            anom_rows["root_cause"] = "Unknown"
        _send_new_anomalies(anom_rows)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# BRANCH B — Single-series views (Synthetic / Kaggle / Hybrid)
# ══════════════════════════════════════════════════════════════════════════════

# ── Cache keys per source ─────────────────────────────────────────────────────
_src_key     = {"Synthetic Data": "synth", "Kaggle Real Data": "kaggle", "Hybrid Dataset": "hybrid"}[data_source]
_raw_key     = f"raw_df_{_src_key}"
_prophet_key = f"prophet_df_{_src_key}"
_sarima_key  = f"sarima_df_{_src_key}"


# ── Data loading ──────────────────────────────────────────────────────────────
if _raw_key not in st.session_state or regenerate:
    if data_source == "Synthetic Data":
        st.session_state[_raw_key] = generate_cost_data()
    elif data_source == "Kaggle Real Data":
        with st.spinner("Loading Kaggle dataset…"):
            st.session_state[_raw_key] = load_kaggle_data()
    else:
        with st.spinner("Building hybrid dataset…"):
            st.session_state[_raw_key] = load_hybrid_data()
    st.session_state.pop(_prophet_key, None)
    st.session_state.pop(_sarima_key,  None)

raw_df = st.session_state[_raw_key]
stl_period = 24 if data_source == "Kaggle Real Data" else 7


# ── Fast detections ───────────────────────────────────────────────────────────
def _detect_zscore_custom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    effective_period = max(2, min(stl_period, len(df) // 2 - 1))
    stl = STL(df["cost"], period=effective_period, robust=True)
    res = stl.fit()
    residual = res.resid
    win = min(rolling_window, len(df))
    rm = residual.rolling(window=win, center=True, min_periods=1).mean()
    rs = residual.rolling(window=win, center=True, min_periods=1).std().fillna(1)
    df["zscore_anomaly"] = (np.abs((residual - rm) / rs)) > zscore_threshold
    return df

config.IF_CONTAMINATION = if_contamination
import detection.isolation_forest as _ifmod
_ifmod.IF_CONTAMINATION = if_contamination

df = _detect_zscore_custom(raw_df)
df = detect_isolation_forest(df)


# ── Slow detections (cached, subsampled for large datasets) ───────────────────
_SLOW_MODEL_ROW_CAP = 1000

def _prepare_slow_df(source_df):
    if len(source_df) > _SLOW_MODEL_ROW_CAP:
        step    = len(source_df) // _SLOW_MODEL_ROW_CAP
        sampled = source_df.iloc[::step].reset_index(drop=True)
        return sampled, source_df.index[::step]
    return source_df.reset_index(drop=True), source_df.index

def _map_back(result_col, orig_idx, full_len):
    if len(result_col) == full_len:
        return result_col
    out = pd.Series(False, index=range(full_len))
    out.iloc[list(orig_idx)] = result_col.values
    return out

if _prophet_key not in st.session_state:
    with st.spinner("Fitting Prophet model… (first load only)"):
        fit_df, orig_idx = _prepare_slow_df(raw_df)
        res = detect_prophet(fit_df)
        st.session_state[_prophet_key] = _map_back(res["prophet_anomaly"], orig_idx, len(raw_df))

if _sarima_key not in st.session_state:
    with st.spinner("Fitting SARIMA model… (first load only)"):
        fit_df, orig_idx = _prepare_slow_df(raw_df)
        res = detect_sarima(fit_df)
        st.session_state[_sarima_key] = _map_back(res["sarima_anomaly"], orig_idx, len(raw_df))

df["prophet_anomaly"] = st.session_state[_prophet_key].values
df["sarima_anomaly"]  = st.session_state[_sarima_key].values


# ── Main chart ────────────────────────────────────────────────────────────────
chart_cfg = {
    "Synthetic Data":   ("Daily Cloud Cost with Anomaly Detection (Synthetic)",       "Daily Cost",   "Date"),
    "Kaggle Real Data": ("Minute-level Cloud Cost with Anomaly Detection (Real Data)", "Minute Cost",  "Timestamp (minute)"),
    "Hybrid Dataset":   ("Daily Cloud Cost with Anomaly Detection (Hybrid)",           "Daily Cost",   "Date"),
}
chart_title, cost_label, xaxis_label = chart_cfg[data_source]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["cost"], mode="lines", name=cost_label,
                         line=dict(color="royalblue", width=1.5)))

gt = df[df["is_anomaly"]]
fig.add_trace(go.Scatter(x=gt["date"], y=gt["cost"], mode="markers",
                         name="Ground Truth Anomaly",
                         marker=dict(color="red", size=10, symbol="circle")))

zs = df[df["zscore_anomaly"]]
fig.add_trace(go.Scatter(x=zs["date"], y=zs["cost"], mode="markers",
                         name="Z-Score Detection",
                         marker=dict(color="orange", size=10, symbol="x")))

ifh = df[df["if_anomaly"]]
fig.add_trace(go.Scatter(x=ifh["date"], y=ifh["cost"], mode="markers",
                         name="Isolation Forest Detection",
                         marker=dict(color="mediumpurple", size=10, symbol="square")))

if show_prophet:
    ph = df[df["prophet_anomaly"]]
    fig.add_trace(go.Scatter(x=ph["date"], y=ph["cost"], mode="markers",
                             name="Prophet Detection",
                             marker=dict(color="limegreen", size=11, symbol="triangle-up")))

if show_sarima:
    sh = df[df["sarima_anomaly"]]
    fig.add_trace(go.Scatter(x=sh["date"], y=sh["cost"], mode="markers",
                             name="SARIMA Detection",
                             marker=dict(color="gold", size=11, symbol="diamond")))

fig.update_layout(title=chart_title, xaxis_title=xaxis_label, yaxis_title="Cost ($)",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  height=500)
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


# ── Slack alerts (single-series) ──────────────────────────────────────────────
if enable_slack and webhook_url:
    anomaly_rows = df[df["is_anomaly"]].copy()
    if "root_cause" not in anomaly_rows.columns:
        anomaly_rows["root_cause"] = data_source
    _send_new_anomalies(anomaly_rows)
