"""Central configuration for the BijliCast project.

All paths, column names, feature groups and split ratios live here so that
notebooks, training scripts and deployments use exactly the same definitions.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = ROOT / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = ROOT / "models"

MERGED_DATASET = PROCESSED_DIR / "merged_dataset.parquet"
FEATURES_DATASET = PROCESSED_DIR / "features_dataset.parquet"
SPLIT_SUMMARY = PROCESSED_DIR / "split_summary.json"

# --------------------------------------------------------------------------- keys
TIME_COL = "datetime"
HOUSE_COL = "house_id"
CITY_COL = "city"
TARGET = "next_hour_energy_kwh"
SOURCE_USAGE = "usage_kw"

# --------------------------------------------------------------------------- features
# Weather: current-hour (origin) values only. Next-hour weather is NOT used,
# so the model never depends on a perfect forecast.
WEATHER_FEATURES = [
    "temperature",
    "humidity",
    "dew",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "pressure",
    "solar_radiation",
    "wind_dir_sin",
    "wind_dir_cos",
]

# Raw weather columns supplied to the pipeline before wind-direction encoding.
INPUT_WEATHER_FEATURES = [
    "temperature",
    "humidity",
    "dew",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "pressure",
    "solar_radiation",
]

# Lags are expressed in hours BEFORE THE TARGET hour (target = usage at t+1).
LAG_FEATURES = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
ROLLING_FEATURES = ["rolling_mean_3h", "rolling_mean_24h", "rolling_std_24h"]
CALENDAR_FEATURES = ["hour", "day_of_week", "month", "is_weekend"]

# Static household features from metadata_ultimate.xlsx.
STATIC_NUMERIC_FEATURES = [
    "No. of people (Temp+Perm)",
    "No. of Adults (14-60)",
    "No. of Children (0-13)",
    "Property Area (Marla)",
    "Covered Area",
    "No of Floors",
    "No. of rooms",
    "Air Conditioners",
    "Air Coolers",
    "Refrigerators",
    "Washing Machines",
    "Celling Fans",
    "Wall Fans",
    "Water Pumps",
    "Geysers",
    "Electric heaters",
    "UPS",
    "Build year of house",
]
STATIC_CATEGORICAL_FEATURES = ["city", "Owner/Rented"]

NUMERIC_FEATURES = (
    WEATHER_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
    + CALENDAR_FEATURES
    + STATIC_NUMERIC_FEATURES
)
CATEGORICAL_FEATURES = list(STATIC_CATEGORICAL_FEATURES)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Raw frame accepted by the pipeline. Lag/rolling columns are deterministic,
# causal history features; everything else comes straight from the user/app.
INPUT_COLUMNS = (
    [TIME_COL]
    + INPUT_WEATHER_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
    + STATIC_NUMERIC_FEATURES
    + STATIC_CATEGORICAL_FEATURES
)

# Rows missing any of these cannot be built without fabricating history.
REQUIRED_HISTORY = LAG_FEATURES + ROLLING_FEATURES + [TARGET]

# --------------------------------------------------------------------------- split
SPLIT_RATIOS = (0.70, 0.15, 0.15)
RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create output directories used by the pipeline."""
    for path in (PROCESSED_DIR, SPLITS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
