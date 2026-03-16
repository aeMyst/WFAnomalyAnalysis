import pandas as pd
import streamlit as st

from config import FARM_SLUGS, PROCESSED_DIR, STATUS_LABELS


def normalize_event_label(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"anom", "anomaly", "anomalous"}:
        return "anomaly"
    if value in {"norm", "normal"}:
        return "normal"
    return value


@st.cache_data
def load_summary():
    df = pd.read_parquet(PROCESSED_DIR / "summary_all_farms.parquet").copy()

    if "farm" in df.columns:
        df["farm"] = df["farm"].astype(str)

    if "event_label" in df.columns:
        df["event_label"] = df["event_label"].astype(str).str.lower()
        df["event_label_display"] = df["event_label"].apply(normalize_event_label)
    else:
        df["event_label_display"] = "unknown"

    return df


@st.cache_data
def load_event_data(farm: str, event_id: int):
    farm_slug = FARM_SLUGS[farm]
    path = PROCESSED_DIR / "events" / farm_slug / f"{int(event_id)}.parquet"
    df = pd.read_parquet(path).copy()

    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")

    if "event_label" in df.columns:
        df["event_label"] = df["event_label"].astype(str).str.lower()
        df["event_label_display"] = df["event_label"].apply(normalize_event_label)

    if "train_test" in df.columns:
        df["train_test"] = df["train_test"].astype(str).str.lower()

    if "status_type_id" not in df.columns and "status_type" in df.columns:
        df["status_type_id"] = pd.to_numeric(df["status_type"], errors="coerce")
    elif "status_type_id" in df.columns:
        df["status_type_id"] = pd.to_numeric(df["status_type_id"], errors="coerce")

    return df


@st.cache_data
def load_status_distribution(farm: str):
    summary = load_summary()
    farm_events = summary[summary["farm"] == farm]["event_id"].tolist()

    rows = []

    for event_id in farm_events:
        try:
            df = load_event_data(farm, int(event_id))
            if "status_type_id" not in df.columns:
                continue
            rows.append(df[["status_type_id"]].copy())
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["status_type_id", "count", "status_label"])

    combined = pd.concat(rows, ignore_index=True)
    out = combined["status_type_id"].value_counts(dropna=False).reset_index()
    out.columns = ["status_type_id", "count"]
    out["status_label"] = out["status_type_id"].map(STATUS_LABELS)
    return out.sort_values("status_type_id")