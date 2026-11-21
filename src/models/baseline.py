"""Naive baselines - the reference any learned model must beat.

Persistence baselines need **no fitting**, so they are the honest floor for a
next-hour forecasting task:

* ``persistence_lag_1h``    - tomorrow's value equals the last known value
  (``lag_1h`` = usage at origin hour ``t``; target = usage at ``t+1``).
* ``seasonal_naive_lag_24h`` - same hour on the previous day (``lag_24h``).
* ``train_mean``            - predict the training-split mean (a constant).

All three use only values that exist in the raw feature frame, so nothing is
learned from validation or test.

CLI:
    python -m src.models.baseline
writes ``reports/tables/baseline_metrics.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.metrics import metrics_table  # noqa: E402
from src.utils.config import (  # noqa: E402
    RANDOM_STATE,
    SPLITS_DIR,
    TABLES_DIR,
    TARGET,
)

PERSISTENCE_BASELINES = {
    "persistence_lag_1h": "lag_1h",
    "seasonal_naive_lag_24h": "lag_24h",
}


def persistence_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {baseline name: predictions} using existing lag columns."""
    return {name: df[col].to_numpy(dtype=float) for name, col in PERSISTENCE_BASELINES.items()}


def evaluate_baselines(
    df: pd.DataFrame,
    split: str,
    train_mean: float | None = None,
) -> list[dict]:
    """Metric rows for every baseline on one split."""
    y_true = df[TARGET].to_numpy(dtype=float)
    rows = []
    for name, pred in persistence_predictions(df).items():
        rows.append(_row(name, split, y_true, pred))
    if train_mean is not None:
        rows.append(_row("train_mean", split, y_true, np.full_like(y_true, train_mean)))
    return rows


def _row(name: str, split: str, y_true, y_pred) -> dict:
    from src.models.metrics import metrics_row

    return metrics_row(name, split, y_true, y_pred)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for name in ("train", "val", "test"):
        path = SPLITS_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python -m src.preprocessing.split` first")
        frames.append(pd.read_parquet(path))
    return tuple(frames)  # type: ignore[return-value]


def main() -> None:
    train, val, test = load_splits()
    train_mean = float(train[TARGET].mean())

    rows = []
    rows += evaluate_baselines(train, "train", train_mean)
    rows += evaluate_baselines(val, "val", train_mean)

    table = metrics_table(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "baseline_metrics.csv"
    table.to_csv(out, index=False)

    print(f"train rows: {len(train):,} | val rows: {len(val):,} | test rows: {len(test):,}")
    print(f"train mean target: {train_mean:.4f} kWh")
    print(table.round(4).to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
