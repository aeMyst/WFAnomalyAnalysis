from pathlib import Path
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(os.getenv("SCADA_ROOT"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH"))
FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]

EVENT_OUTPUT_DIR = OUTPUT_PATH / "events"

SAVE_FARM_MASTER = os.getenv("SAVE_FARM_MASTER", "true").lower() == "true"
SAVE_ALL_MASTER = os.getenv("SAVE_ALL_MASTER", "false").lower() == "true"

STATUS_LABELS = {
    0: "Normal Operation",
    1: "Derated Operation",
    2: "Idling",
    3: "Service",
    4: "Downtime",
    5: "Other",
}


def load_event_info(event_info_path: Path) -> pd.DataFrame:
    events = pd.read_csv(event_info_path, sep=";").copy()

    required_cols = ["event_id", "event_label", "event_start_id", "event_end_id"]
    missing_cols = [c for c in required_cols if c not in events.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {event_info_path}: {missing_cols}")

    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce")
    events["event_start_id"] = pd.to_numeric(events["event_start_id"], errors="coerce")
    events["event_end_id"] = pd.to_numeric(events["event_end_id"], errors="coerce")

    events = events.dropna(subset=["event_id", "event_start_id", "event_end_id"]).copy()
    events["event_id"] = events["event_id"].astype(int)

    events["start_id"] = events[["event_start_id", "event_end_id"]].min(axis=1).astype(int)
    events["end_id"] = events[["event_start_id", "event_end_id"]].max(axis=1).astype(int)

    return events


def load_event_dataset(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";").copy()

    if "id" not in df.columns:
        raise ValueError(f"'id' column missing in {file_path}")

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype("int32")

    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")

    return df


def normalize_train_test_column(df: pd.DataFrame) -> pd.DataFrame:
    if "train_test" in df.columns:
        df["train_test"] = df["train_test"].astype(str).str.strip().str.lower()
    return df


def normalize_status_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize operational status to status_type_id.
    """
    if "status_type_id" in df.columns:
        df["status_type_id"] = pd.to_numeric(df["status_type_id"], errors="coerce")
    elif "status_type" in df.columns:
        df["status_type_id"] = pd.to_numeric(df["status_type"], errors="coerce")

    return df


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    meta_cols = {
        "time_stamp",
        "asset_id",
        "id",
        "train_test",
        "status_type_id",
        "status_type",
        "event_id",
        "event_label",
        "farm",
        "in_event_window",
    }
    return [col for col in df.columns if col not in meta_cols]


def optimize_event_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("int32")

    if "asset_id" in df.columns:
        df["asset_id"] = pd.to_numeric(df["asset_id"], errors="coerce").astype("int16")

    if "status_type_id" in df.columns:
        df["status_type_id"] = pd.to_numeric(df["status_type_id"], errors="coerce")

    if "in_event_window" in df.columns:
        df["in_event_window"] = pd.to_numeric(df["in_event_window"], errors="coerce").astype("int8")

    if "event_id" in df.columns:
        df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("int32")

    sensor_cols = get_sensor_columns(df)
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df


def detect_wind_power_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    preferred_wind = [
        "wind_speed_3_avg",
        "wind_speed_4_avg",
        "wind_speed_avg",
        "windspeed_avg",
    ]
    preferred_power = [
        "power_29_avg",
        "power_30_avg",
        "power_avg",
    ]

    for c in preferred_wind:
        if c in lower_map:
            wind_col = lower_map[c]
            break
    else:
        wind_col = None

    for c in preferred_power:
        if c in lower_map:
            power_col = lower_map[c]
            break
    else:
        power_col = None

    if wind_col is None:
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
        if wind_candidates:
            wind_col = wind_candidates[0]

    if power_col is None:
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
        if power_candidates:
            power_col = power_candidates[0]

    return wind_col, power_col


def compute_metric_block(g: pd.DataFrame, sensor_cols: list[str]) -> dict:
    """
    Compute metric block for a given subset of rows.
    """
    train = g[g["train_test"] == "train"] if "train_test" in g.columns else pd.DataFrame()
    prediction = g[g["train_test"] == "prediction"] if "train_test" in g.columns else pd.DataFrame()
    window = g[g["in_event_window"] == 1] if "in_event_window" in g.columns else pd.DataFrame()

    out = {
        "rows": int(len(g)),
        "train_rows": int(len(train)),
        "prediction_rows": int(len(prediction)),
        "window_rows": int(len(window)),
        "max_z_shift": np.nan,
        "max_volatility_ratio": np.nan,
        "energy_score": np.nan,
        "num_sensors_z_gt_1": 0,
        "num_sensors_z_gt_1_5": 0,
        "severity_score": np.nan,
    }

    if len(train) > 0 and len(window) > 0 and len(sensor_cols) > 0:
        train_mean = train[sensor_cols].mean()
        train_std = train[sensor_cols].std().replace(0, np.nan)
        window_mean = window[sensor_cols].mean()
        window_std = window[sensor_cols].std()

        z_scores = ((window_mean - train_mean) / train_std).abs()
        vol_ratio = window_std / train_std

        max_z_shift = float(z_scores.max()) if pd.notna(z_scores.max()) else np.nan
        max_volatility_ratio = float(vol_ratio.max()) if pd.notna(vol_ratio.max()) else np.nan
        energy_score = float(np.sqrt(np.nansum(z_scores.to_numpy() ** 2)))

        out["max_z_shift"] = max_z_shift
        out["max_volatility_ratio"] = max_volatility_ratio
        out["energy_score"] = energy_score
        out["num_sensors_z_gt_1"] = int((z_scores > 1).sum())
        out["num_sensors_z_gt_1_5"] = int((z_scores > 1.5).sum())

        if pd.notna(max_volatility_ratio) and max_volatility_ratio > 0:
            out["severity_score"] = float(max_z_shift + np.log(max_volatility_ratio))

    return out


def compute_event_metrics(
    g: pd.DataFrame,
    sensor_cols: list[str],
    wind_col: str | None = None,
    power_col: str | None = None,
) -> dict:
    asset_id = g["asset_id"].iloc[0] if "asset_id" in g.columns else np.nan
    event_label = g["event_label"].iloc[0] if "event_label" in g.columns else None
    farm_name = g["farm"].iloc[0] if "farm" in g.columns else None
    event_id = g["event_id"].iloc[0] if "event_id" in g.columns else None

    # Full metrics across all rows
    full_metrics = compute_metric_block(g, sensor_cols)

    # Operational metrics
    operational_0 = pd.DataFrame()
    operational_0_2 = pd.DataFrame()
    pct_status = {f"pct_status_{k}": np.nan for k in range(6)}

    if "status_type_id" in g.columns:
        status_counts = g["status_type_id"].value_counts(normalize=True, dropna=False)
        for k in range(6):
            pct_status[f"pct_status_{k}"] = float(status_counts.get(k, 0.0))

        operational_0 = g[g["status_type_id"] == 0].copy()
        operational_0_2 = g[g["status_type_id"].isin([0, 2])].copy()

    operational_0_metrics = compute_metric_block(operational_0, sensor_cols) if not operational_0.empty else None
    operational_0_2_metrics = compute_metric_block(operational_0_2, sensor_cols) if not operational_0_2.empty else None

    return {
        "farm": farm_name,
        "event_id": int(event_id),
        "asset_id": asset_id,
        "event_label": event_label,

        "total_rows": full_metrics["rows"],
        "train_rows": full_metrics["train_rows"],
        "prediction_rows": full_metrics["prediction_rows"],
        "window_rows": full_metrics["window_rows"],

        "max_z_shift": full_metrics["max_z_shift"],
        "max_volatility_ratio": full_metrics["max_volatility_ratio"],
        "energy_score": full_metrics["energy_score"],
        "num_sensors_z_gt_1": full_metrics["num_sensors_z_gt_1"],
        "num_sensors_z_gt_1_5": full_metrics["num_sensors_z_gt_1_5"],
        "severity_score": full_metrics["severity_score"],

        "operational_rows_status_0": int(len(operational_0)),
        "operational_rows_status_0_2": int(len(operational_0_2)),

        "max_z_shift_status_0": operational_0_metrics["max_z_shift"] if operational_0_metrics else np.nan,
        "max_volatility_ratio_status_0": operational_0_metrics["max_volatility_ratio"] if operational_0_metrics else np.nan,
        "severity_score_status_0": operational_0_metrics["severity_score"] if operational_0_metrics else np.nan,

        "max_z_shift_status_0_2": operational_0_2_metrics["max_z_shift"] if operational_0_2_metrics else np.nan,
        "max_volatility_ratio_status_0_2": operational_0_2_metrics["max_volatility_ratio"] if operational_0_2_metrics else np.nan,
        "severity_score_status_0_2": operational_0_2_metrics["severity_score"] if operational_0_2_metrics else np.nan,

        **pct_status,

        "wind_speed_column": wind_col,
        "power_output_column": power_col,
    }


def process_farm(root: Path, farm_name: str) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    farm_dir = root / farm_name
    datasets_dir = farm_dir / "datasets"
    event_info_path = farm_dir / "event_info.csv"

    if not farm_dir.exists():
        raise FileNotFoundError(f"Farm directory not found: {farm_dir}")
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")
    if not event_info_path.exists():
        raise FileNotFoundError(f"event_info.csv not found: {event_info_path}")

    print(f"\nProcessing {farm_name}...")
    events = load_event_info(event_info_path)

    farm_slug = farm_name.lower().replace(" ", "_")
    farm_event_output_dir = EVENT_OUTPUT_DIR / farm_slug
    farm_event_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    master_parts = []

    for event_id in events["event_id"]:
        try:
            file_path = datasets_dir / f"{int(event_id)}.csv"
            if not file_path.exists():
                print(f"  Skipping missing file: {file_path.name}")
                continue

            df = load_event_dataset(file_path)
            df = normalize_train_test_column(df)
            df = normalize_status_column(df)

            row = events.loc[events["event_id"] == event_id].iloc[0]
            start_id = int(row["start_id"])
            end_id = int(row["end_id"])

            df["in_event_window"] = ((df["id"] >= start_id) & (df["id"] <= end_id)).astype("int8")
            df["event_id"] = np.int32(event_id)
            df["event_label"] = str(row["event_label"]).lower()
            df["farm"] = str(farm_name)

            wind_col, power_col = detect_wind_power_columns(df)

            df = optimize_event_dataframe(df)

            sensor_cols = get_sensor_columns(df)
            metrics = compute_event_metrics(df, sensor_cols, wind_col=wind_col, power_col=power_col)
            summary_rows.append(metrics)

            event_out_path = farm_event_output_dir / f"{int(event_id)}.parquet"
            df.to_parquet(event_out_path, index=False)

            if SAVE_FARM_MASTER:
                master_parts.append(df)

            print(
                f"  Processed event {event_id}"
                f" | wind={wind_col if wind_col else 'None'}"
                f" | power={power_col if power_col else 'None'}"
            )

        except Exception as e:
            print(f"  Failed event {event_id}: {e}")

    event_summary = pd.DataFrame(summary_rows)

    farm_master = None
    if SAVE_FARM_MASTER and master_parts:
        farm_master = pd.concat(master_parts, ignore_index=True, sort=False)

    print(f"  Event summary shape: {event_summary.shape}")
    print(f"  Events processed: {event_summary['event_id'].nunique() if not event_summary.empty else 0}")
    if farm_master is not None:
        print(f"  Farm master shape: {farm_master.shape}")

    return farm_master, event_summary


def main() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    EVENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_master = []

    for farm in FARMS:
        try:
            farm_master, farm_summary = process_farm(ROOT, farm)

            farm_slug = farm.lower().replace(" ", "_")

            farm_summary.to_parquet(OUTPUT_PATH / f"{farm_slug}_event_summary.parquet", index=False)
            all_summary.append(farm_summary)

            if SAVE_FARM_MASTER and farm_master is not None:
                farm_master.to_parquet(OUTPUT_PATH / f"{farm_slug}_master.parquet", index=False)

                if SAVE_ALL_MASTER:
                    all_master.append(farm_master)

        except Exception as e:
            print(f"Error processing {farm}: {e}")

    if not all_summary:
        raise RuntimeError("No farms were successfully processed.")

    summary_all = pd.concat(all_summary, ignore_index=True, sort=False)
    summary_all.to_parquet(OUTPUT_PATH / "summary_all_farms.parquet", index=False)

    if SAVE_ALL_MASTER and all_master:
        master_all = pd.concat(all_master, ignore_index=True, sort=False)
        master_all.to_parquet(OUTPUT_PATH / "master_all_farms.parquet", index=False)
        print(f"Combined master shape: {master_all.shape}")

    print("\nFinished processing all farms.")
    print(f"Combined summary shape: {summary_all.shape}")
    print(f"Output directory: {OUTPUT_PATH}")
    print(f"Per-event files directory: {EVENT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
