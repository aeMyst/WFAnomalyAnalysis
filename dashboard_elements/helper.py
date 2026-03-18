import numpy as np
import pandas as pd
import streamlit as st

from dashboard_elements.config import META_COLS
from dashboard_elements.data_loader import load_event_data, load_summary


def normalize_event_label(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"anom", "anomaly", "anomalous"}:
        return "anomaly"
    if value in {"norm", "normal"}:
        return "normal"
    return value


def get_numeric_sensor_columns(df: pd.DataFrame):
    return [
        c for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def compute_sensor_metrics(df: pd.DataFrame):
    sensor_cols = get_numeric_sensor_columns(df)

    train = df[df["train_test"] == "train"].copy() if "train_test" in df.columns else pd.DataFrame()
    window = df[df["in_event_window"] == 1].copy() if "in_event_window" in df.columns else pd.DataFrame()

    if train.empty or window.empty or not sensor_cols:
        return pd.DataFrame(columns=[
            "sensor",
            "train_mean",
            "window_mean",
            "train_std",
            "window_std",
            "mean_delta",
            "z_shift",
            "volatility_ratio",
            "importance_score",
        ])

    train_mean = train[sensor_cols].mean()
    train_std = train[sensor_cols].std().replace(0, np.nan)
    window_mean = window[sensor_cols].mean()
    window_std = window[sensor_cols].std()

    metrics = pd.DataFrame({
        "sensor": sensor_cols,
        "train_mean": train_mean.values,
        "window_mean": window_mean.values,
        "train_std": train_std.values,
        "window_std": window_std.values,
    })

    metrics["mean_delta"] = metrics["window_mean"] - metrics["train_mean"]
    metrics["z_shift"] = (metrics["mean_delta"] / metrics["train_std"]).abs()
    metrics["volatility_ratio"] = metrics["window_std"] / metrics["train_std"]
    metrics.replace([np.inf, -np.inf], np.nan, inplace=True)

    metrics["importance_score"] = (
        metrics["z_shift"] + np.log1p(metrics["volatility_ratio"].clip(lower=0))
    )

    return metrics


def compute_sensor_separation_for_farm(farm: str):
    summary = load_summary()
    farm_events = summary[summary["farm"] == farm][["event_id", "event_label"]].copy()

    rows = []

    for _, row in farm_events.iterrows():
        event_id = int(row["event_id"])
        event_label = normalize_event_label(row["event_label"])

        try:
            df = load_event_data(farm, event_id)
            metrics = compute_sensor_metrics(df)
            if metrics.empty:
                continue

            metrics["event_id"] = event_id
            metrics["event_label"] = event_label
            rows.append(metrics[["sensor", "z_shift", "importance_score", "event_label"]])
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    all_metrics = pd.concat(rows, ignore_index=True)

    grouped = (
        all_metrics
        .groupby(["event_label", "sensor"], as_index=False)[["z_shift", "importance_score"]]
        .mean()
    )

    z_pivot = grouped.pivot(index="sensor", columns="event_label", values="z_shift")
    imp_pivot = grouped.pivot(index="sensor", columns="event_label", values="importance_score")

    result = pd.DataFrame(index=sorted(set(grouped["sensor"])))
    result["avg_anomaly_z"] = z_pivot["anomaly"] if "anomaly" in z_pivot.columns else np.nan
    result["avg_normal_z"] = z_pivot["normal"] if "normal" in z_pivot.columns else np.nan
    result["z_difference"] = result["avg_anomaly_z"] - result["avg_normal_z"]

    result["avg_anomaly_importance"] = imp_pivot["anomaly"] if "anomaly" in imp_pivot.columns else np.nan
    result["avg_normal_importance"] = imp_pivot["normal"] if "normal" in imp_pivot.columns else np.nan
    result["importance_difference"] = result["avg_anomaly_importance"] - result["avg_normal_importance"]

    return result.reset_index().rename(columns={"index": "sensor"})


def detect_wind_power_columns_from_event(df: pd.DataFrame):
    cols = list(df.columns)

    wind_candidates = [
        c for c in cols
        if ("wind" in c.lower()) and ("speed" in c.lower())
    ]
    wind_candidates = sorted(
        wind_candidates,
        key=lambda x: (
            0 if "avg" in x.lower() else 1,
            0 if "mean" in x.lower() else 1,
            len(x)
        )
    )

    power_candidates = [
        c for c in cols
        if ("power" in c.lower())
        and ("reactive" not in c.lower())
        and ("apparent" not in c.lower())
    ]
    power_candidates = sorted(
        power_candidates,
        key=lambda x: (
            0 if "avg" in x.lower() else 1,
            0 if "mean" in x.lower() else 1,
            0 if "active" in x.lower() else 1,
            len(x)
        )
    )

    wind_col = wind_candidates[0] if wind_candidates else None
    power_col = power_candidates[0] if power_candidates else None

    return wind_col, power_col


@st.cache_data
def load_wind_power_points(farm: str, max_points_per_event: int = 1200):
    summary = load_summary()
    farm_events = summary[summary["farm"] == farm].copy()

    all_points = []

    for _, row in farm_events.iterrows():
        event_id = int(row["event_id"])
        event_label = normalize_event_label(row["event_label"])

        try:
            df = load_event_data(farm, event_id)

            wind_col = row["wind_speed_column"] if "wind_speed_column" in row.index else None
            power_col = row["power_output_column"] if "power_output_column" in row.index else None

            if pd.isna(wind_col):
                wind_col = None
            if pd.isna(power_col):
                power_col = None

            if wind_col is None or power_col is None or wind_col not in df.columns or power_col not in df.columns:
                wind_col, power_col = detect_wind_power_columns_from_event(df)

            if wind_col is None or power_col is None:
                continue
            if wind_col not in df.columns or power_col not in df.columns:
                continue

            temp = df.copy()

            if "status_type_id" in temp.columns:
                temp = temp[temp["status_type_id"].isin([0, 2])]

            temp = temp[[wind_col, power_col]].dropna().copy()

            if temp.empty:
                continue

            temp["event_label"] = event_label
            temp["event_label_display"] = temp["event_label"]
            temp["event_id"] = event_id

            if len(temp) > max_points_per_event:
                temp = temp.sample(max_points_per_event, random_state=42)

            temp = temp.rename(columns={
                wind_col: "wind_speed",
                power_col: "power_output"
            })

            all_points.append(temp)

        except Exception:
            continue

    if not all_points:
        return pd.DataFrame(columns=["wind_speed", "power_output", "event_label", "event_label_display", "event_id"])

    return pd.concat(all_points, ignore_index=True)