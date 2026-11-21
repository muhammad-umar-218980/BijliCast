"""Model training - baseline candidates trained on train, scored on validation.

Four course algorithms are compared on exactly the same raw inputs and the
same chronological splits:

===============  ==============================================  ==============
Algorithm        Estimator                                       Notes
===============  ==============================================  ==============
Linear           ``LinearRegression``                            linear ceiling
KNN              ``KNeighborsRegressor``                         time-ordered 40k
Random Forest    ``RandomForestRegressor``                       main candidate
Boosting         ``HistGradientBoostingRegressor``               fast GBDT
===============  ==============================================  ==============

Every model is wrapped by :func:`src.preprocessing.pipeline.build_pipeline`, so
feature engineering, imputation, scaling and one-hot encoding are re-fitted
inside the training data on every call. The test split is **never** touched in
this module - selection happens on validation, the final test evaluation lives
in :mod:`src.models.evaluate`.

CLI:
    python -m src.models.train
writes ``reports/tables/model_comparison_untuned.csv`` and
``models/untuned_val_metrics.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.baseline import evaluate_baselines, load_splits  # noqa: E402
from src.models.metrics import metrics_row, metrics_table  # noqa: E402
from src.preprocessing.pipeline import build_pipeline, get_xy  # noqa: E402
from src.utils.config import (  # noqa: E402
    INPUT_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET,
)

MODEL_NAMES = ["Linear Regression", "KNN", "Random Forest", "HistGradientBoosting"]

# Compute guard: KNN prediction cost grows with the reference set, so it is
# fitted on a time-ordered subsample. The limit is documented in every results
# table and in the final report.
KNN_TRAIN_ROWS = 40_000

BASE_PARAMS: dict[str, dict] = {
    "Linear Regression": {},
    "KNN": {"n_neighbors": 10, "weights": "distance", "n_jobs": -1},
    "Random Forest": {
        "n_estimators": 150,
        "min_samples_leaf": 2,
        "max_features": 0.6,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "HistGradientBoosting": {
        "max_iter": 300,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "random_state": RANDOM_STATE,
    },
}


def make_estimator(name: str, params: dict | None = None, n_jobs: int | None = None):
    """Instantiate one of the four course algorithms with merged defaults."""
    if name not in BASE_PARAMS:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    merged = {**BASE_PARAMS[name], **(params or {})}
    if n_jobs is not None:
        merged["n_jobs"] = n_jobs

    if name == "Linear Regression":
        return LinearRegression(**merged)
    if name == "KNN":
        merged.setdefault("n_jobs", -1)
        return KNeighborsRegressor(**merged)
    if name == "Random Forest":
        merged.setdefault("n_jobs", -1)
        return RandomForestRegressor(**merged)
    return HistGradientBoostingRegressor(**merged)


def time_ordered_subsample(df: pd.DataFrame, n_rows: int | None) -> pd.DataFrame:
    """Systematic sample that keeps the full time span (and all houses).

    ``df.iloc[::step]`` is used rather than ``sample()`` so the subsample is
    deterministic, spans every month, and never shuffles time.
    """
    if n_rows is None or len(df) <= n_rows:
        return df
    step = len(df) / float(n_rows)
    idx = np.unique((np.arange(n_rows) * step).astype(int))
    return df.iloc[idx]


def train_subsample_rows(name: str) -> int | None:
    """Rows used to fit a model during the untuned comparison phase."""
    return KNN_TRAIN_ROWS if name == "KNN" else None


def fit_and_predict(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    params: dict | None = None,
    train_subsample: int | None = None,
):
    """Fit one pipeline on train and predict train (and optionally validation).

    Returns a dict with the fitted pipeline, predictions, metrics and timings.
    No test data is accepted on purpose.
    """
    fit_df = time_ordered_subsample(train_df, train_subsample)
    pipe = build_pipeline(make_estimator(name, params))
    X_fit, y_fit = get_xy(fit_df)

    started = time.perf_counter()
    pipe.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started

    result = {
        "model": name,
        "pipeline": pipe,
        "params": params or {},
        "n_train": int(len(fit_df)),
        "fit_seconds": float(fit_seconds),
        "metrics": {"train": metrics_row(name, "train", y_fit, pipe.predict(X_fit))},
        "predictions": {"train": pipe.predict(X_fit)},
        "index": {"train": fit_df},
    }

    if val_df is not None:
        X_val, y_val = get_xy(val_df)
        started = time.perf_counter()
        pred_val = pipe.predict(X_val)
        result["predict_seconds_val"] = float(time.perf_counter() - started)
        result["metrics"]["val"] = metrics_row(name, "val", y_val, pred_val)
        result["predictions"]["val"] = pred_val
        result["index"]["val"] = val_df

    return result


def train_all(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    models: list[str] | None = None,
    tuned_params: dict[str, dict] | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """Fit every requested model on train and score it on validation."""
    models = models or MODEL_NAMES
    tuned_params = tuned_params or {}
    results: dict[str, dict] = {}
    for name in models:
        subsample = train_subsample_rows(name)
        if verbose:
            print(f"[{name}] fitting on {subsample or len(train_df):,} rows ...", end=" ", flush=True)
        res = fit_and_predict(
            name,
            train_df,
            val_df,
            params=tuned_params.get(name),
            train_subsample=subsample,
        )
        results[name] = res
        if verbose:
            print(
                f"done in {res['fit_seconds']:.1f}s | "
                f"train MAE {res['metrics']['train']['mae']:.4f} | "
                f"val MAE {res['metrics']['val']['mae']:.4f}"
            )
    return results


def comparison_table(
    results: dict[str, dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Long metrics table for the trained models plus naive baselines."""
    rows: list[dict] = []
    for res in results.values():
        rows.append(res["metrics"]["train"])
        rows.append(res["metrics"]["val"])
    mean = float(train_df[TARGET].mean())
    rows += evaluate_baselines(train_df, "train", mean)
    rows += evaluate_baselines(val_df, "val", mean)
    return metrics_table(rows)


def validation_summary(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model with train/val metrics and the over-fitting gap."""
    rows = []
    for name, res in results.items():
        tr, va = res["metrics"]["train"], res["metrics"]["val"]
        rows.append(
            {
                "model": name,
                "n_train": res["n_train"],
                "fit_seconds": round(res["fit_seconds"], 2),
                "train_mae": round(tr["mae"], 4),
                "val_mae": round(va["mae"], 4),
                "mae_gap_val_minus_train": round(va["mae"] - tr["mae"], 4),
                "train_rmse": round(tr["rmse"], 4),
                "val_rmse": round(va["rmse"], 4),
                "train_r2": round(tr["r2"], 4),
                "val_r2": round(va["r2"], 4),
                "params": res["params"],
            }
        )
    return pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)


def save_untuned_results(
    results: dict[str, dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """Persist the untuned comparison table, baseline table and JSON payload."""
    summary = validation_summary(results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = TABLES_DIR / "model_comparison_untuned.csv"
    summary.drop(columns="params").to_csv(out_csv, index=False)

    train_mean = float(train_df[TARGET].mean())
    baseline_rows = evaluate_baselines(train_df, "train", train_mean)
    baseline_rows += evaluate_baselines(val_df, "val", train_mean)
    baseline_table = metrics_table(baseline_rows)
    baseline_csv = TABLES_DIR / "baseline_metrics.csv"
    baseline_table.to_csv(baseline_csv, index=False)

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "selection_split": "val",
        "knn_train_rows": KNN_TRAIN_ROWS,
        "models": {
            name: {
                "params": res["params"],
                "n_train": res["n_train"],
                "fit_seconds": res["fit_seconds"],
                "train": res["metrics"]["train"],
                "val": res["metrics"]["val"],
            }
            for name, res in results.items()
        },
    }
    meta_out = MODELS_DIR / "untuned_val_metrics.json"
    meta_out.write_text(json.dumps(payload, indent=2))

    print("\n=== untuned validation summary ===")
    print(summary.drop(columns="params").round(4).to_string(index=False))
    print("\n=== naive baselines ===")
    print(baseline_table.round(4).to_string(index=False))
    print(f"\nsaved -> {out_csv}\nsaved -> {baseline_csv}\nsaved -> {meta_out}")
    return payload


def main() -> None:
    train, val, test = load_splits()
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} (test untouched)")
    results = train_all(train, val)
    save_untuned_results(results, train, val)


if __name__ == "__main__":
    main()
