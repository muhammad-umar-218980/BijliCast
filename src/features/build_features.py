"""Build the modelling table for BijliCast (Phase 3, step 2 - part 1).

Design
------
* Each row is indexed by the **origin hour t**. All features are known at the
  end of hour t; the label is the household's usage in hour t+1.
* Lags/rolling statistics are computed on a continuous hourly grid per house,
  using past values only (no centered windows, no future values).
* Rows without a complete lag/rolling history (warm-up period or around meter
  outages) are dropped instead of being imputed, so no usage value is
  fabricated. Only the static metadata may contain missing values; those are
  imputed later inside the scikit-learn Pipeline, fitted on the training split.

Run from the repository root:
    python -m src.features.build_features
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python src/features/build_features.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    CALENDAR_FEATURES,
    CITY_COL,
    FEATURES_DATASET,
    FEATURE_COLUMNS,
    HOUSE_COL,
    INPUT_WEATHER_FEATURES,
    LAG_FEATURES,
    MERGED_DATASET,
    REQUIRED_HISTORY,
    ROLLING_FEATURES,
    SOURCE_USAGE,
    STATIC_CATEGORICAL_FEATURES,
    STATIC_NUMERIC_FEATURES,
    TABLES_DIR,
    TARGET,
    TIME_COL,
    WEATHER_FEATURES,
    ensure_dirs,
)


def _house_features(g: pd.DataFrame) -> pd.DataFrame:
    """Create lag/rolling/calendar/target columns for one household."""
    g = g.sort_values(TIME_COL).set_index(TIME_COL)

    # Continuous hourly grid so that "24h" really means 24 hours.
    grid = pd.date_range(g.index.min(), g.index.max(), freq="h", name=TIME_COL)
    g = g.reindex(grid)

    # Static household attributes are constant: carry them across the grid.
    for col in STATIC_NUMERIC_FEATURES + STATIC_CATEGORICAL_FEATURES + [HOUSE_COL]:
        if col in g.columns:
            g[col] = g[col].ffill().bfill()

    usage = g[SOURCE_USAGE]

    out = pd.DataFrame(index=g.index)
    # Lags relative to the TARGET hour (t+1): lag_1h is usage(t).
    out["lag_1h"] = usage
    out["lag_2h"] = usage.shift(1)
    out["lag_3h"] = usage.shift(2)
    out["lag_24h"] = usage.shift(23)
    out["lag_168h"] = usage.shift(167)

    roll3 = usage.rolling(3, min_periods=3)
    roll24 = usage.rolling(24, min_periods=24)
    out["rolling_mean_3h"] = roll3.mean()
    out["rolling_mean_24h"] = roll24.mean()
    out["rolling_std_24h"] = roll24.std()

    out[TARGET] = usage.shift(-1)

    # Current-hour weather (origin), never t+1.
    for col in INPUT_WEATHER_FEATURES:
        out[col] = g[col]

    out["wind_dir_sin"] = np.sin(np.deg2rad(g["wind_direction"]))
    out["wind_dir_cos"] = np.cos(np.deg2rad(g["wind_direction"]))

    for col in STATIC_NUMERIC_FEATURES + STATIC_CATEGORICAL_FEATURES:
        out[col] = g[col]

    out[HOUSE_COL] = g[HOUSE_COL]
    out[CITY_COL] = g[CITY_COL]

    idx = out.index
    out["hour"] = idx.hour.astype("int8")
    out["day_of_week"] = idx.dayofweek.astype("int8")
    out["month"] = idx.month.astype("int8")
    out["is_weekend"] = (idx.dayofweek >= 5).astype("int8")

    out = out.dropna(subset=REQUIRED_HISTORY)
    return out.reset_index()


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Turn the integrated hourly table into the modelling table."""
    merged = merged.copy()
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL])

    # A house id must map to exactly one city (join key consistency).
    cities_per_house = merged.groupby(HOUSE_COL)[CITY_COL].nunique()
    if (cities_per_house > 1).any():
        bad = cities_per_house[cities_per_house > 1].index.tolist()
        raise ValueError(f"house_id maps to multiple cities: {bad}")

    frames = [
        _house_features(g)
        for _, g in merged.groupby(HOUSE_COL, sort=True)
    ]
    features = pd.concat(frames, ignore_index=True)

    front = [TIME_COL, HOUSE_COL, CITY_COL, TARGET]
    features = features[front + [c for c in features.columns if c not in front]]
    features = features.sort_values([TIME_COL, HOUSE_COL]).reset_index(drop=True)
    return features


def _check_lags_against_raw(features: pd.DataFrame, merged: pd.DataFrame) -> str:
    """Recompute one clean household from raw usage and compare every lag."""
    raw = merged.set_index([HOUSE_COL, TIME_COL])[SOURCE_USAGE]
    candidates = []
    for house, g in merged.groupby(HOUSE_COL):
        s = g.set_index(TIME_COL)[SOURCE_USAGE].sort_index()
        expected_len = len(pd.date_range(s.index.min(), s.index.max(), freq="h"))
        if s.isna().sum() == 0 and len(s) == expected_len:
            candidates.append(house)
    if not candidates:
        raise AssertionError("no gap-free household found for lag verification")

    house = sorted(candidates)[0]
    raw_s = raw.loc[house].sort_index()
    got = features[features[HOUSE_COL] == house].set_index(TIME_COL).sort_index()

    expected = {
        "lag_1h": raw_s,
        "lag_2h": raw_s.shift(1),
        "lag_3h": raw_s.shift(2),
        "lag_24h": raw_s.shift(23),
        "lag_168h": raw_s.shift(167),
        "rolling_mean_3h": raw_s.rolling(3, min_periods=3).mean(),
        "rolling_mean_24h": raw_s.rolling(24, min_periods=24).mean(),
        "rolling_std_24h": raw_s.rolling(24, min_periods=24).std(),
        TARGET: raw_s.shift(-1),
    }
    for col, series in expected.items():
        exp = series.reindex(got.index)
        if not np.allclose(got[col].to_numpy(), exp.to_numpy(), equal_nan=True):
            raise AssertionError(f"lag verification failed for column {col!r}")
    return house


def validate_features(features: pd.DataFrame, merged: pd.DataFrame) -> dict:
    """Leakage and integrity checks. Raises AssertionError on failure."""
    checks: dict[str, object] = {}

    checks["rows"] = int(len(features))
    checks["houses"] = int(features[HOUSE_COL].nunique())
    checks["features"] = int(len(FEATURE_COLUMNS))

    missing_feats = [c for c in FEATURE_COLUMNS if c not in features.columns]
    assert not missing_feats, f"missing feature columns: {missing_feats}"
    assert SOURCE_USAGE not in features.columns, "raw usage must not be a model input"

    dup = int(features.duplicated([HOUSE_COL, TIME_COL]).sum())
    assert dup == 0, f"{dup} duplicated (house, hour) keys"

    assert features[TARGET].notna().all(), "target contains missing values"
    for col in REQUIRED_HISTORY:
        assert features[col].notna().all(), f"{col} contains missing values"

    weather_missing = int(features[list(INPUT_WEATHER_FEATURES)].isna().sum().sum())
    checks["weather_missing_cells"] = weather_missing
    assert weather_missing == 0, "weather features contain missing values"

    # No target leakage: the expected label equals lag_1h one hour later.
    d = features.sort_values([HOUSE_COL, TIME_COL])
    nxt_time = d.groupby(HOUSE_COL)[TIME_COL].shift(-1)
    consecutive = (nxt_time - d[TIME_COL]) == pd.Timedelta(hours=1)
    nxt_lag = d.groupby(HOUSE_COL)["lag_1h"].shift(-1)
    assert np.allclose(
        d.loc[consecutive, TARGET].to_numpy(),
        nxt_lag[consecutive].to_numpy(),
    ), "target(t) must equal lag_1h(t+1h)"

    checks["verified_house"] = _check_lags_against_raw(features, merged)
    checks["rows_dropped_pct"] = round(
        100 * (1 - len(features) / len(merged)), 2
    )
    checks["mean_target"] = round(float(features[TARGET].mean()), 4)
    checks["max_target"] = round(float(features[TARGET].max()), 4)
    return checks


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Per-column data profile used for the report."""
    rows = []
    for col in features.columns:
        s = features[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "role": (
                    "target" if col == TARGET
                    else "id" if col in (TIME_COL, HOUSE_COL)
                    else "feature" if col in FEATURE_COLUMNS
                    else "context"
                ),
                "missing_pct": round(float(s.isna().mean() * 100), 4),
                "n_unique": int(s.nunique(dropna=True)),
                "min": round(float(s.min()), 4) if is_num else np.nan,
                "median": round(float(s.median()), 4) if is_num else np.nan,
                "mean": round(float(s.mean()), 4) if is_num else np.nan,
                "max": round(float(s.max()), 4) if is_num else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    merged = pd.read_parquet(MERGED_DATASET)
    print(f"loaded merged dataset: {merged.shape[0]:,} rows x {merged.shape[1]} cols")

    features = build_features(merged)
    checks = validate_features(features, merged)

    print(f"features dataset      : {features.shape[0]:,} rows x {features.shape[1]} cols")
    print(f"model features        : {len(FEATURE_COLUMNS)}")
    print(f"houses                : {checks['houses']}")
    print(f"rows dropped          : {checks['rows_dropped_pct']}%")
    print(f"rows verified vs raw  : {checks['verified_house']}")
    print(f"mean target (kW/kWh)  : {checks['mean_target']}")

    features.to_parquet(FEATURES_DATASET, index=False)
    print(f"saved -> {FEATURES_DATASET}")

    summary = feature_summary(features)
    out = TABLES_DIR / "feature_summary.csv"
    summary.to_csv(out, index=False)
    print(f"saved -> {out}")

    groups = {
        "weather": WEATHER_FEATURES,
        "lags": LAG_FEATURES,
        "rolling": ROLLING_FEATURES,
        "calendar": CALENDAR_FEATURES,
        "static_numeric": STATIC_NUMERIC_FEATURES,
        "categorical": STATIC_CATEGORICAL_FEATURES,
    }
    print("\nfeature groups:")
    for name, cols in groups.items():
        print(f"  {name:<15} {len(cols):>2}  {cols if len(cols) <= 5 else ''}")


if __name__ == "__main__":
    main()
