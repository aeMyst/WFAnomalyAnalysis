import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from charts_util import (
    CHART_HEIGHT,
    make_sensor_bar,
    make_severity_boxplot,
    make_severity_histogram,
    make_wind_power_scatter,
)
from data_loader import load_status_distribution, load_summary
from helper import compute_sensor_separation_for_farm, load_wind_power_points
from styles import apply_global_styles

load_dotenv()

st.set_page_config(
    page_title="Wind Turbine Anomaly Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_global_styles()

summary = load_summary()

farm = st.sidebar.selectbox("Wind Farm", sorted(summary["farm"].unique().tolist()))

farm_summary = summary[summary["farm"] == farm].copy()

st.markdown('<div class="dashboard-title">Wind Turbine Anomaly Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Compare normal and anomalous behavior patterns while accounting for operational state.</div>',
    unsafe_allow_html=True
)

total_events = len(farm_summary)
anomaly_events = int((farm_summary["event_label_display"] == "anomaly").sum())
normal_events = int((farm_summary["event_label_display"] == "normal").sum())

severity_col = "severity_score_normalized" if "severity_score_normalized" in farm_summary.columns else "severity_score"
avg_severity = farm_summary[severity_col].mean() if severity_col in farm_summary.columns else float("nan")
avg_severity_display = f"{avg_severity:.2f}" if pd.notna(avg_severity) else "N/A"

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Events</div>
            <div class="kpi-value">{total_events}</div>
            <div class="kpi-sub">Events in {farm}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Anomaly Events</div>
            <div class="kpi-value">{anomaly_events}</div>
            <div class="kpi-sub">Labeled anomalous cases</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Normal Events</div>
            <div class="kpi-value">{normal_events}</div>
            <div class="kpi-sub">Labeled normal cases</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Severity</div>
            <div class="kpi-value">{avg_severity_display}</div>
            <div class="kpi-sub">Normalized across events</div>
        </div>
        """,
        unsafe_allow_html=True
    )

col1, col2 = st.columns(2)

with col1:
    fig_hist = make_severity_histogram(farm_summary)
    st.plotly_chart(fig_hist, use_container_width=True, height=CHART_HEIGHT, config={"displayModeBar": False})

with col2:
    fig_box = make_severity_boxplot(farm_summary)
    st.plotly_chart(fig_box, use_container_width=True, height=CHART_HEIGHT, config={"displayModeBar": False})

col3, col4 = st.columns(2)

with col3:
    wind_power_df = load_wind_power_points(farm)

    if not wind_power_df.empty:
        fig_wp = make_wind_power_scatter(wind_power_df)
        st.plotly_chart(fig_wp, use_container_width=True, height=CHART_HEIGHT, config={"displayModeBar": False})
    else:
        st.info("No wind-power data was detected for this wind farm.")

with col4:
    separation = compute_sensor_separation_for_farm(farm)

    if not separation.empty:
        metric_col = "importance_difference" if "importance_difference" in separation.columns else "z_difference"
        top_sensors = (
            separation
            .dropna(subset=[metric_col])
            .sort_values(metric_col, ascending=False)
            .head(12)
        )
        fig_bar = make_sensor_bar(top_sensors)
        st.plotly_chart(fig_bar, use_container_width=True, height=CHART_HEIGHT, config={"displayModeBar": False})
    else:
        st.info("Sensor comparison data is not available for this wind farm.")

status_df = load_status_distribution(farm)