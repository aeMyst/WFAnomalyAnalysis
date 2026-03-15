import plotly.express as px

from theme import ANOMALY, BAR, CARD_BG, GRID, HIST, NAVY, NORMAL, TEXT_MID, TEXT_SOFT


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


def make_severity_histogram(farm_summary):
    fig = px.histogram(
        farm_summary,
        x="severity_score",
        nbins=20,
        title="Severity Score Distribution"
    )
    fig.update_traces(marker_color=HIST, marker_line_width=0)
    fig.update_layout(
        xaxis_title="Severity Score",
        yaxis_title="Frequency",
        showlegend=False
    )
    return apply_plot_theme(fig)


def make_severity_boxplot(farm_summary):
    fig = px.box(
        farm_summary,
        x="event_label",
        y="severity_score",
        color="event_label",
        points="all",
        title="Severity Score Comparison",
        color_discrete_map={
            "anomaly": ANOMALY,
            "normal": NORMAL
        }
    )
    fig.update_layout(
        xaxis_title="Event Type",
        yaxis_title="Severity Score"
    )
    return apply_plot_theme(fig)


def make_wind_power_scatter(wind_power_df):
    fig = px.scatter(
        wind_power_df,
        x="wind_speed",
        y="power_output",
        color="event_label",
        opacity=0.65,
        title="Wind Speed vs Power Output",
        color_discrete_map={
            "anomaly": ANOMALY,
            "normal": NORMAL
        }
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        xaxis_title="Wind Speed",
        yaxis_title="Power Output"
    )
    return apply_plot_theme(fig)


def make_sensor_bar(top_sensors):
    fig = px.bar(
        top_sensors.sort_values("z_difference", ascending=True),
        x="z_difference",
        y="sensor",
        orientation="h",
        title="Sensors Most Associated with Anomalous Behavior"
    )
    fig.update_traces(marker_color=BAR)
    fig.update_layout(
        xaxis_title="Average Anomaly Deviation - Average Normal Deviation",
        yaxis_title=""
    )
    return apply_plot_theme(fig)