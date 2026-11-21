"""Inference helpers for BijliCast (train / serve parity).

A deployed app cannot build 24h/168h lags from a single form row. The contract
is therefore:

    household profile (static attributes)
  + the last 168 hours of hourly usage for that house
  + current-hour weather
  + the origin timestamp
      -> one row in ``INPUT_COLUMNS`` -> the saved sklearn pipeline -> prediction

``build_inference_row`` reproduces exactly the lag/rolling definitions used by
``src.features.build_features``. ``self_check`` proves that parity on real data.

CLI:
    python -m src.features.inference
exports ``models/house_profiles.csv`` and runs the parity check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    CITY_COL,
    FEATURES_DATASET,
    HOUSE_COL,
    INPUT_COLUMNS,
    INPUT_WEATHER_FEATURES,
    LAG_FEATURES,
    MERGED_DATASET,
    MODELS_DIR,
    ROLLING_FEATURES,
    SOURCE_USAGE,
    SPLITS_DIR,
    STATIC_CATEGORICAL_FEATURES,
    STATIC_NUMERIC_FEATURES,
    TARGET,
    TIME_COL,
)

HISTORY_HOURS = 168
PROFILE_COLUMNS = [HOUSE_COL] + STATIC_CATEGORICAL_FEATURES + STATIC_NUMERIC_FEATURES


def export_house_profiles(
    merged: pd.DataFrame,
    out_path: Path | str | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Write one static-profile row per household (used by all three apps)."""
    out_path = Path(out_path or (MODELS_DIR / "house_profiles.csv"))
    profiles = (
        merged[PROFILE_COLUMNS]
        .drop_duplicates(subset=[HOUSE_COL])
        .sort_values(HOUSE_COL)
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(out_path, index=False)
    return profiles, out_path


def build_inference_row(
    origin_time,
    city: str,
    weather: dict,
    usage_history: pd.Series,
    static: dict,
    house_id: str | None = None,
) -> pd.DataFrame:
    """Build one model-ready row from raw deployment inputs.

    Parameters
    ----------
    origin_time
        Timestamp of the last known hour ``t``; the model predicts ``t+1``.
    city
        One of the six city names.
    weather
        Current-hour values for ``INPUT_WEATHER_FEATURES``.
    usage_history
        Hourly usage Series spanning at least ``origin_time - 167h .. origin_time``.
    static
        Household profile: the 18 static numeric fields plus ``Owner/Rented``.
    """
    origin = pd.Timestamp(origin_time)

    missing_weather = [k for k in INPUT_WEATHER_FEATURES if k not in weather]
    if missing_weather:
        raise ValueError(f"weather is missing: {missing_weather}")
    missing_static = [k for k in STATIC_NUMERIC_FEATURES if k not in static]
    if missing_static:
        raise ValueError(f"static profile is missing: {missing_static}")

    hist = pd.Series(usage_history, dtype="float64").copy()
    hist.index = pd.to_datetime(hist.index)
    hist = hist.sort_index()

    stamps = [origin - pd.Timedelta(hours=k) for k in range(HISTORY_HOURS)]
    missing_hours = [t for t in stamps if t not in hist.index]
    if missing_hours:
        raise ValueError(
            f"usage_history is missing {len(missing_hours)} of the last "
            f"{HISTORY_HOURS} hours (e.g. {missing_hours[:3]})"
        )

    window = hist.reindex(stamps)  # most recent first
    if window.isna().any():
        raise ValueError("usage_history contains NaN inside the last 168 hours")

    row = {
        TIME_COL: origin,
        "lag_1h": float(window.iloc[0]),
        "lag_2h": float(window.iloc[1]),
        "lag_3h": float(window.iloc[2]),
        "lag_24h": float(window.iloc[23]),
        "lag_168h": float(window.iloc[167]),
        "rolling_mean_3h": float(window.iloc[:3].mean()),
        "rolling_mean_24h": float(window.iloc[:24].mean()),
        "rolling_std_24h": float(window.iloc[:24].std()),
    }
    row.update({k: float(weather[k]) for k in INPUT_WEATHER_FEATURES})
    row.update({k: static[k] for k in STATIC_NUMERIC_FEATURES})
    row[CITY_COL] = city
    row["Owner/Rented"] = static.get("Owner/Rented", np.nan)

    frame = pd.DataFrame([row])[INPUT_COLUMNS]
    if house_id is not None:
        frame.insert(0, HOUSE_COL, house_id)
    return frame


def self_check() -> None:
    """Verify the inference row matches the training feature builder."""
    merged = pd.read_parquet(MERGED_DATASET)
    features = pd.read_parquet(FEATURES_DATASET)
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    features[TIME_COL] = pd.to_datetime(features[TIME_COL])

    chosen = None
    for house, g in merged.groupby(HOUSE_COL):
        g = g.sort_values(TIME_COL)
        grid = pd.date_range(g[TIME_COL].min(), g[TIME_COL].max(), freq="h")
        complete = len(g) == len(grid) and g[SOURCE_USAGE].notna().all()
        if complete and g["Air Conditioners"].notna().all():
            chosen = house
            break
    if chosen is None:
        raise AssertionError("no clean household found for the parity check")

    g = merged[merged[HOUSE_COL] == chosen].sort_values(TIME_COL)
    history = g.set_index(TIME_COL)[SOURCE_USAGE]
    origin = history.index[300]
    first = g.iloc[0]
    weather = (
        g.loc[g[TIME_COL] == origin, INPUT_WEATHER_FEATURES].iloc[0].to_dict()
    )
    static = {k: first[k] for k in STATIC_NUMERIC_FEATURES}
    static["Owner/Rented"] = first["Owner/Rented"]

    row = build_inference_row(
        origin, first[CITY_COL], weather, history, static, house_id=chosen
    )

    expected = features[
        (features[HOUSE_COL] == chosen) & (features[TIME_COL] == origin)
    ]
    if len(expected) != 1:
        raise AssertionError("could not locate the matching training row")
    expected = expected.iloc[0]

    numeric = LAG_FEATURES + ROLLING_FEATURES + INPUT_WEATHER_FEATURES
    numeric += STATIC_NUMERIC_FEATURES
    for col in numeric:
        if not np.allclose(row[col].iloc[0], expected[col], equal_nan=True):
            raise AssertionError(f"inference/training mismatch in {col!r}")
    for col in STATIC_CATEGORICAL_FEATURES:
        if row[col].iloc[0] != expected[col]:
            raise AssertionError(f"inference/training mismatch in {col!r}")
    print(f"parity check passed for {chosen} at {origin}")

    train_path = SPLITS_DIR / "train.parquet"
    if train_path.exists():
        from src.preprocessing.pipeline import build_feature_pipeline

        train = pd.read_parquet(train_path)
        prep = build_feature_pipeline()
        prep.fit(train[INPUT_COLUMNS])
        X = prep.transform(row[INPUT_COLUMNS])
        if X.shape != (1, 49) or np.isnan(X).any():
            raise AssertionError(f"pipeline transform failed: shape={X.shape}")
        print(f"pipeline accepts the inference row: {X.shape}, no NaNs")


def main() -> None:
    merged = pd.read_parquet(MERGED_DATASET)
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])
    profiles, path = export_house_profiles(merged)
    print(f"exported {len(profiles)} household profiles -> {path}")
    print(f"columns: {list(profiles.columns)}")
    self_check()


if __name__ == "__main__":
    main()
