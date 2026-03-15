from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

PROCESSED_DIR = Path(os.getenv("OUTPUT_PATH"))


FARM_SLUGS = {
    "Wind Farm A": "wind_farm_a",
    "Wind Farm B": "wind_farm_b",
    "Wind Farm C": "wind_farm_c",
}


META_COLS = {
    "time_stamp",
    "asset_id",
    "id",
    "train_test",
    "status_type_id",
    "event_id",
    "event_label",
    "farm",
    "in_event_window",
}


@st.cache_data
def load_summary() -> pd.DataFrame:
    path = PROCESSED_DIR / "summary_all_farms.parquet"
    df = pd.read_parquet(path)

    # Safe dtypes
    if "farm" in df.columns:
        df["farm"] = df["farm"].astype(str)
    if "event_label" in df.columns:
        df["event_label"] = df["event_label"].astype(str)

    return df


@st.cache_data
def load_event_data(farm: str, event_id: int) -> pd.DataFrame:
    farm_slug = FARM_SLUGS[farm]
    path = PROCESSED_DIR / "events" / farm_slug / f"{int(event_id)}.parquet"
    df = pd.read_parquet(path)

    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")

    if "train_test" in df.columns:
        df["train_test"] = df["train_test"].astype(str).str.lower()

    return df


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def get_numeric_sensor_columns(df: pd.DataFrame) -> list[str]:
    sensor_cols = get_sensor_columns(df)
    out = []
    for c in sensor_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def choose_default_sensor(df: pd.DataFrame) -> str | None:
    candidates = [
        "power_29_avg",
        "power_30_avg",
        "wind_speed_3_avg",
        "wind_speed_4_avg",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    numeric_cols = get_numeric_sensor_columns(df)
    return numeric_cols[0] if numeric_cols else None


def choose_wind_power_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    wind_candidates = ["wind_speed_3_avg", "wind_speed_4_avg"]
    power_candidates = ["power_29_avg", "power_30_avg"]

    wind_col = next((c for c in wind_candidates if c in df.columns), None)
    power_col = next((c for c in power_candidates if c in df.columns), None)

    return wind_col, power_col


def compute_event_metrics(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = get_numeric_sensor_columns(df)

    train = df[df["train_test"] == "train"].copy() if "train_test" in df.columns else pd.DataFrame()
    window = df[df["in_event_window"] == 1].copy() if "in_event_window" in df.columns else pd.DataFrame()

    if train.empty or window.empty or not sensor_cols:
        return pd.DataFrame(columns=[
            "sensor",
            "train_mean",
            "window_mean",
            "z_shift",
            "train_std",
            "window_std",
            "volatility_ratio",
            "mean_delta",
        ])

    train_mean = train[sensor_cols].mean()
    train_std = train[sensor_cols].std().replace(0, np.nan)
    window_mean = window[sensor_cols].mean()
    window_std = window[sensor_cols].std()

    out = pd.DataFrame({
        "sensor": sensor_cols,
        "train_mean": train_mean.values,
        "window_mean": window_mean.values,
        "train_std": train_std.values,
        "window_std": window_std.values,
    })

    out["mean_delta"] = out["window_mean"] - out["train_mean"]
    out["z_shift"] = np.abs(out["mean_delta"] / out["train_std"])
    out["volatility_ratio"] = out["window_std"] / out["train_std"]

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out.sort_values("z_shift", ascending=False)


def detect_behavior_type(df: pd.DataFrame, sensor: str) -> str:
    """
    Simple heuristic for non-technical messaging:
    - sudden: last pre-window mean differs sharply from early pre-window mean
    - gradual: slope into the event window is more progressive
    """
    if sensor not in df.columns or "in_event_window" not in df.columns:
        return "Unknown"

    work = df[["id", "in_event_window", sensor]].dropna().copy()
    if work.empty:
        return "Unknown"

    pre = work[work["in_event_window"] == 0]
    win = work[work["in_event_window"] == 1]

    if len(pre) < 20 or len(win) < 10:
        return "Unknown"

    recent_pre = pre.tail(min(100, len(pre)))
    early_pre = pre.tail(min(300, len(pre))).head(min(100, len(pre)))

    if early_pre.empty or recent_pre.empty:
        return "Unknown"

    recent_mean = recent_pre[sensor].mean()
    early_mean = early_pre[sensor].mean()
    pre_std = pre[sensor].std()

    if pd.isna(pre_std) or pre_std == 0:
        return "Unknown"

    jump_score = abs(recent_mean - early_mean) / pre_std

    # simple slope estimate on recent pre-window
    x = np.arange(len(recent_pre))
    y = recent_pre[sensor].to_numpy()
    if len(x) > 1 and np.nanstd(y) > 0:
        slope = np.polyfit(x, y, 1)[0]
        slope_score = abs(slope) / pre_std
    else:
        slope_score = 0

    if jump_score > 1.5 and slope_score < 0.02:
        return "Sudden change"
    if jump_score > 0.75 or slope_score >= 0.02:
        return "Gradual drift"
    return "Mostly stable"


def risk_band(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return "Unknown"
    if score >= 1.5:
        return "High"
    if score >= 1.0:
        return "Medium"
    return "Low"


def color_for_label(label: str) -> str:
    return {"anomaly": "red", "normal": "green"}.get(str(label).lower(), "gray")
