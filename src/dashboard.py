"""Streamlit real-time monitoring dashboard.

Run with::

    streamlit run src/dashboard.py

The dashboard reads directly from the SQLite database the streaming
pipeline writes to. It does **not** start the pipeline itself — start
``python -m src.run_pipeline`` in another terminal first.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure the project root is on sys.path so `src.*` / `models.*` resolve
# when Streamlit runs this file directly (not via `python -m`).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Streamlit may run from any CWD — resolve relative paths (e.g. DB path)
# against the project root so the dashboard always finds the database.
import os
os.chdir(_PROJECT_ROOT)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config
from src.database import Database

st.set_page_config(
    page_title="Real-Time Anomaly Detection",
    page_icon=None,
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
@st.cache_resource
def get_db() -> Database:
    cfg = load_config()
    return Database(
        path=cfg["database"]["path"],
        batch_size=int(cfg["database"]["batch_size"]),
        flush_interval_seconds=float(cfg["database"]["flush_interval_seconds"]),
    )


@st.cache_data(ttl=2)
def fetch_state(history_window: int) -> Dict[str, Any]:
    db = get_db()
    return {
        "events": db.fetch_recent_events(history_window),
        "anomalies": db.fetch_recent_anomalies(history_window * 4),
        "alerts": db.fetch_recent_alerts(20),
        "metrics": db.fetch_latest_metrics(),
        "event_count": db.event_count(),
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
SEVERITY_COLORS = {
    "LOW":    "#FFC107",
    "MEDIUM": "#FF6D00",
    "HIGH":   "#D50000",
}


def severity_badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#888")
    return (
        f"<span style='background-color:{color};color:white;"
        f"padding:2px 8px;border-radius:6px;font-weight:600;'>"
        f"{severity}</span>"
    )


def build_event_chart(events_df: pd.DataFrame, anomaly_df: pd.DataFrame,
                      feature: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=events_df["dt"], y=events_df[feature],
        mode="lines", name=feature, line=dict(width=1, color="#1f77b4"),
    ))

    if not anomaly_df.empty:
        anom_points = events_df[events_df["timestamp"].isin(anomaly_df["timestamp"])]
        if not anom_points.empty:
            fig.add_trace(go.Scatter(
                x=anom_points["dt"], y=anom_points[feature],
                mode="markers", name="Detected anomaly",
                marker=dict(size=9, color="#D50000", symbol="x"),
            ))

    truth_points = events_df[events_df["is_true_anomaly"] == 1]
    if not truth_points.empty:
        fig.add_trace(go.Scatter(
            x=truth_points["dt"], y=truth_points[feature],
            mode="markers", name="True anomaly (label)",
            marker=dict(size=11, color="rgba(213,0,0,0.25)", line=dict(width=2, color="#D50000"), symbol="circle-open"),
        ))

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title=None,
        yaxis_title=feature,
    )
    return fig


def confusion_matrix_fig(metric_row: Dict[str, Any]) -> go.Figure:
    z = [
        [int(metric_row.get("tn", 0)), int(metric_row.get("fp", 0))],
        [int(metric_row.get("fn", 0)), int(metric_row.get("tp", 0))],
    ]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted Normal", "Predicted Anomaly"],
        y=["Actual Normal", "Actual Anomaly"],
        text=z, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(
        title=metric_row["detector"],
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    dash_cfg = cfg["dashboard"]
    refresh = float(dash_cfg.get("refresh_seconds", 2))
    history_window = int(dash_cfg.get("history_window", 500))

    st.title(dash_cfg.get("page_title", "Real-Time Anomaly Detection"))
    st.caption(
        "Live view of streaming sensor data, detector outputs and alerts. "
        f"Auto-refreshes every {refresh:.0f}s."
    )

    state = fetch_state(history_window)
    events_df = pd.DataFrame(state["events"])
    anomaly_df = pd.DataFrame(state["anomalies"])
    alerts = state["alerts"]
    metrics = state["metrics"]

    if events_df.empty:
        st.info("Waiting for the streaming pipeline... start it with `python -m src.run_pipeline`.")
        time.sleep(refresh)
        st.rerun()
        return

    events_df["dt"] = pd.to_datetime(events_df["timestamp"], unit="s")
    if not anomaly_df.empty:
        anomaly_df = anomaly_df[anomaly_df["is_anomaly"] == 1]

    # ------------------------------------------------------------------
    # Top KPIs
    # ------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total events", f"{state['event_count']:,}")
    col2.metric("Anomalies (window)", f"{int(events_df['is_true_anomaly'].sum())}")
    col3.metric("Active sensors", events_df["sensor_id"].nunique())
    col4.metric("Alerts (recent)", len(alerts))
    ensemble_metric = next((m for m in metrics if m["detector"] == "ensemble"), None)
    if ensemble_metric:
        col5.metric("Ensemble F1", f"{ensemble_metric['f1']:.2f}")
    else:
        col5.metric("Ensemble F1", "—")

    st.divider()

    # ------------------------------------------------------------------
    # Sensor charts
    # ------------------------------------------------------------------
    st.subheader("Sensor readings")
    sensor_options = sorted(events_df["sensor_id"].unique())
    selected_sensor = st.selectbox("Sensor", sensor_options, index=0)
    sensor_events = events_df[events_df["sensor_id"] == selected_sensor].copy()
    sensor_anomalies = anomaly_df[anomaly_df["sensor_id"] == selected_sensor] if not anomaly_df.empty else anomaly_df

    chart_cols = st.columns(2)
    for i, feature in enumerate(["temperature", "pressure", "vibration", "power_consumption"]):
        with chart_cols[i % 2]:
            st.plotly_chart(
                build_event_chart(sensor_events, sensor_anomalies, feature),
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # Anomaly score timeline + alert feed
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Anomaly score timeline (ensemble)")
    if not anomaly_df.empty:
        ens = anomaly_df[anomaly_df["detector"] == "ensemble"].copy()
        if not ens.empty:
            ens["dt"] = pd.to_datetime(ens["timestamp"], unit="s")
            fig = px.area(ens, x="dt", y="score", color="sensor_id",
                          height=260, labels={"score": "Ensemble vote"})
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No ensemble anomaly scores yet.")
    else:
        st.caption("No anomaly history yet.")

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Recent alerts")
        if not alerts:
            st.caption("No alerts fired yet.")
        else:
            for a in alerts:
                ts = pd.to_datetime(a["timestamp"], unit="s").strftime("%H:%M:%S")
                st.markdown(
                    f"{severity_badge(a['severity'])} &nbsp; "
                    f"`{ts}` **{a['sensor_id']}** — {a['message']} "
                    f"(score `{a.get('score', 0):.2f}`)",
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("Detector performance")
        if metrics:
            metric_df = pd.DataFrame(metrics)[
                ["detector", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
            ]
            st.dataframe(metric_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Metrics will appear after the first snapshot interval.")

    # ------------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Confusion matrices")
    if metrics:
        cm_cols = st.columns(len(metrics))
        for col, m in zip(cm_cols, metrics):
            with col:
                st.plotly_chart(confusion_matrix_fig(m), use_container_width=True)

    st.divider()
    st.caption(f"Last refreshed at {time.strftime('%H:%M:%S')}")

    time.sleep(refresh)
    st.rerun()


if __name__ == "__main__":
    main()
