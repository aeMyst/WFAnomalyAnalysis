import plotly.express as px

from dashboard_elements.theme import ANOMALY, BAR, CARD_BG, GRID, HIST, NAVY, NORMAL, TEXT_MID, TEXT_SOFT


CHART_HEIGHT = 400


def apply_plot_theme(fig, height: int = CHART_HEIGHT):
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        font=dict(color=TEXT_MID),
        title_font=dict(color=NAVY, size=18),
        legend_title_font=dict(color=TEXT_MID),
        legend_font=dict(color=TEXT_MID),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            automargin=True,
            title_font=dict(color=TEXT_MID),
            tickfont=dict(color=TEXT_SOFT)
        ),
        yaxis=dict(
            gridcolor=GRID,
            zeroline=False,
            automargin=True,
            title_font=dict(color=TEXT_MID),
            tickfont=dict(color=TEXT_SOFT)
        )
    )
    return fig


def _label_color_map():
    return {
        "anomaly": ANOMALY,
        "anom": ANOMALY,
        "normal": NORMAL,
        "norm": NORMAL,
    }


def make_severity_histogram(farm_summary):
    hist_col = "severity_score_log" if "severity_score_log" in farm_summary.columns else "severity_score"

    fig = px.histogram(
        farm_summary,
        x=hist_col,
        nbins=20,
        title="Severity Score Distribution"
    )
    fig.update_traces(marker_color=HIST, marker_line_width=0)
    fig.update_layout(
        xaxis_title="Log Severity Score" if hist_col == "severity_score_log" else "Severity Score",
        yaxis_title="Frequency",
        showlegend=False
    )
    return apply_plot_theme(fig)


def make_severity_boxplot(farm_summary):
    score_col = "severity_score_normalized" if "severity_score_normalized" in farm_summary.columns else "severity_score"

    fig = px.box(
        farm_summary,
        x="event_label_display",
        y=score_col,
        color="event_label_display",
        points="all",
        title="Severity Score Comparison",
        color_discrete_map=_label_color_map()
    )
    fig.update_layout(
        xaxis_title="Event Type",
        yaxis_title="Normalized Severity Score" if score_col == "severity_score_normalized" else "Severity Score"
    )
    return apply_plot_theme(fig)


def make_wind_power_scatter(wind_power_df):
    fig = px.scatter(
        wind_power_df,
        x="wind_speed",
        y="power_output",
        color="event_label_display",
        opacity=0.65,
        title="Wind Speed vs Power Output",
        color_discrete_map=_label_color_map()
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        xaxis_title="Wind Speed",
        yaxis_title="Power Output"
    )
    return apply_plot_theme(fig)


def make_sensor_bar(top_sensors):
    metric_col = "importance_difference" if "importance_difference" in top_sensors.columns else "z_difference"

    fig = px.bar(
        top_sensors.sort_values(metric_col, ascending=True),
        x=metric_col,
        y="sensor",
        orientation="h",
        title="Sensors Most Associated with Anomalous Behavior"
    )
    fig.update_traces(marker_color=BAR)
    fig.update_layout(
        xaxis_title=(
            "Average Anomaly Importance - Average Normal Importance"
            if metric_col == "importance_difference"
            else "Average Anomaly Z-Shift - Average Normal Z-Shift"
        ),
        yaxis_title=""
    )
    return apply_plot_theme(fig)

def make_asset_severity_bar(farm_summary):
    fig = px.bar(
        farm_summary.groupby("asset_id")["severity_score"]
        .mean()
        .reset_index()
        .sort_values("severity_score", ascending=True),
        x = "severity_score",
        y = "asset_id",
        orientation="h",
        title="Average Severity Score by Turbine",
        color="severity_score",
        color_continuous_scale=[[0, NORMAL],
            [0.5, "#d29922"],
            [1, ANOMALY]]
    )
    fig.update_layout(
        xaxis_title="Average Severity Score",
        yaxis_title="Turbine ID",
        coloraxis_showscale=False
    )
    return apply_plot_theme(fig)