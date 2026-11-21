"""Deployment-data export (Phase 9).

The apps cannot read ``data/processed/merged_dataset.parquet`` (git-ignored and
large), so this script exports the minimum needed for inference:

* ``models/usage_history.parquet`` - the last 168 contiguous hourly usage values
  per household (the lag/rolling window required by the pipeline).
* ``models/latest_weather.csv``    - current-hour weather at that origin, used
  as the default values of the app sliders.
* ``models/app_manifest.json``     - shapes, generation time, covered houses.

Houses without a complete 168-hour NaN-free window are skipped and reported.

CLI:
    python -m src.deploy.export_app_data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.inference import build_inference_row  # noqa: E402
from src.utils.config import (  # noqa: E402
    CITY_COL,
    HOUSE_COL,
    INPUT_WEATHER_FEATURES,
    MERGED_DATASET,
    MODELS_DIR,
    SOURCE_USAGE,
    STATIC_NUMERIC_FEATURES,
    TIME_COL,
)

HISTORY_HOURS = 168


def _find_last_valid_window(run: pd.DataFrame) -> pd.DataFrame | None:
    """Last 168-row NaN-free contiguous window ending on a row with weather."""
    if len(run) < HISTORY_HOURS:
        return None
    for end in range(len(run) - 1, HISTORY_HOURS - 2, -1):
        window = run.iloc[end - HISTORY_HOURS + 1 : end + 1]
        if window[SOURCE_USAGE].isna().any():
            continue
        if window[INPUT_WEATHER_FEATURES].iloc[-1].isna().any():
            continue
        return window
    return None


def export_app_data() -> dict:
    columns = [HOUSE_COL, TIME_COL, CITY_COL, SOURCE_USAGE] + INPUT_WEATHER_FEATURES
    merged = pd.read_parquet(MERGED_DATASET, columns=columns).sort_values(
        [HOUSE_COL, TIME_COL]
    )

    history_parts: list[pd.DataFrame] = []
    latest_rows: list[dict] = []
    skipped: list[str] = []

    for house, g in merged.groupby(HOUSE_COL, sort=True):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        # contiguous hourly runs: a change in the 1h difference starts a new run
        run_id = (g[TIME_COL].diff() != pd.Timedelta(hours=1)).cumsum()
        best_window = None
        for _, run in g.groupby(run_id):
            window = _find_last_valid_window(run)
            if window is None:
                continue
            if best_window is None or window[TIME_COL].iloc[-1] > best_window[TIME_COL].iloc[-1]:
                best_window = window
        if best_window is None:
            skipped.append(house)
            continue

        history_parts.append(
            best_window[[HOUSE_COL, TIME_COL, SOURCE_USAGE]].rename(
                columns={SOURCE_USAGE: "usage_kw"}
            )
        )
        origin = best_window.iloc[-1]
        row = {
            HOUSE_COL: house,
            CITY_COL: origin[CITY_COL],
            TIME_COL: origin[TIME_COL],
        }
        row.update({c: float(origin[c]) for c in INPUT_WEATHER_FEATURES})
        latest_rows.append(row)

    history = pd.concat(history_parts, ignore_index=True).sort_values(
        [HOUSE_COL, TIME_COL]
    )
    latest = pd.DataFrame(latest_rows).sort_values(HOUSE_COL)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = MODELS_DIR / "usage_history.parquet"
    weather_path = MODELS_DIR / "latest_weather.csv"
    history.to_parquet(history_path, index=False)
    latest.to_csv(weather_path, index=False)

    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "history_hours": HISTORY_HOURS,
        "houses": int(history[HOUSE_COL].nunique()),
        "rows": int(len(history)),
        "skipped_houses": skipped,
        "origin_range": {
            "min": str(latest[TIME_COL].min()),
            "max": str(latest[TIME_COL].max()),
        },
    }
    (MODELS_DIR / "app_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"usage history : {len(history):,} rows, {history[HOUSE_COL].nunique()} houses -> {history_path}")
    print(f"latest weather: {len(latest)} rows -> {weather_path}")
    if skipped:
        print(f"skipped (no complete 168h window): {skipped}")

    _self_check(history, latest)
    return manifest


def _self_check(history: pd.DataFrame, latest: pd.DataFrame) -> None:
    """Predict one next hour per house using the saved deployment pipeline."""
    from src.utils.config import STATIC_CATEGORICAL_FEATURES

    model_path = MODELS_DIR / "deployment_model.joblib"
    profiles_path = MODELS_DIR / "house_profiles.csv"
    if not model_path.exists() or not profiles_path.exists():
        print("self-check skipped: deployment_model.joblib / house_profiles.csv missing")
        return
    model = joblib.load(model_path)
    profiles = pd.read_csv(profiles_path).set_index(HOUSE_COL)

    preds = []
    for house in latest[HOUSE_COL]:
        row = latest[latest[HOUSE_COL] == house].iloc[0]
        hist = history[history[HOUSE_COL] == house].set_index(TIME_COL)["usage_kw"]
        static = profiles.loc[house]
        static_dict = {c: float(static[c]) for c in STATIC_NUMERIC_FEATURES}
        static_dict.update({c: static[c] for c in STATIC_CATEGORICAL_FEATURES})
        frame = build_inference_row(
            origin_time=row[TIME_COL],
            city=row[CITY_COL],
            weather={c: float(row[c]) for c in INPUT_WEATHER_FEATURES},
            usage_history=hist,
            static=static_dict,
            house_id=house,
        )
        preds.append(float(model.predict(frame)[0]))

    preds = np.asarray(preds)
    if not np.isfinite(preds).all():
        raise AssertionError("non-finite prediction from the deployment pipeline")
    print(
        f"self-check: {len(preds)} houses predicted | "
        f"min {preds.min():.3f} | median {np.median(preds):.3f} | max {preds.max():.3f} kWh"
    )


if __name__ == "__main__":
    export_app_data()
