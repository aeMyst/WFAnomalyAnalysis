import os
from pathlib import Path
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
    "status_type",
    "event_id",
    "event_label",
    "farm",
    "in_event_window",
}

STATUS_LABELS = {
    0: "Normal Operation",
    1: "Derated Operation",
    2: "Idling",
    3: "Service",
    4: "Downtime",
    5: "Other",
}