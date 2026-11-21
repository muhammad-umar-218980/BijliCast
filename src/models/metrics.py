"""Regression metrics used throughout the modelling phase.

The task is regression, so the guideline's classification metrics
(F1 / ROC-AUC / confusion matrix) are replaced by:

* **MAE**  - primary metric (same unit as the target, kWh; robust to outliers)
* **RMSE** - penalises large errors (outage / AC-start spikes)
* **R2**   - share of variance explained, comparable across splits

Every function accepts array-likes and ignores rows where either the truth or
the prediction is non-finite, so it can be used directly on model output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PRIMARY_METRIC = "mae"
METRIC_COLUMNS = ["n", "mae", "rmse", "r2", "median_ae"]


def regression_metrics(y_true, y_pred) -> dict:
    """Return n / MAE / RMSE / R2 / median absolute error."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        raise ValueError("no finite rows to score")

    mse = mean_squared_error(y_true, y_pred)
    return {
        "n": int(y_true.size),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
    }


def metrics_row(name: str, split: str, y_true, y_pred, **extra) -> dict:
    """One tidy row: model, split, metrics and any extra fields."""
    row = {"model": name, "split": split}
    row.update(regression_metrics(y_true, y_pred))
    row.update(extra)
    return row


def metrics_table(rows) -> pd.DataFrame:
    """Build a DataFrame ordered by model, then split (train, val, test)."""
    frame = pd.DataFrame(rows)
    split_order = {"train": 0, "val": 1, "test": 2}
    frame["_split_order"] = frame["split"].map(split_order).fillna(3)
    return frame.sort_values(["model", "_split_order"]).drop(columns="_split_order").reset_index(drop=True)


def pretty_metrics(frame: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Format a metrics table for display in notebooks (4 decimal places)."""
    out = frame.copy()
    for col in ("mae", "rmse", "r2", "median_ae"):
        if col in out.columns:
            out[col] = out[col].round(decimals)
    return out
