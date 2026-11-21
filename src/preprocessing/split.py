"""Chronological train / validation / test splitting (Phase 3, step 3).

The split is global and time-ordered: every household uses the same calendar
boundaries, so the test period is genuinely "the future" for all houses.

Because a row at origin `t` predicts `t+1`, the last row of the train split has
a label inside the validation period. That row (and the equivalent one between
validation and test) is **purged** so no label crosses a split boundary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    FEATURES_DATASET,
    HOUSE_COL,
    SPLIT_RATIOS,
    SPLIT_SUMMARY,
    SPLITS_DIR,
    TARGET,
    TIME_COL,
    ensure_dirs,
)

SPLIT_NAMES = ("train", "val", "test")


def _describe(df: pd.DataFrame, total: int) -> dict:
    d = df[TARGET]
    return {
        "rows": int(len(df)),
        "pct_of_total": round(100 * len(df) / total, 2),
        "houses": int(df[HOUSE_COL].nunique()),
        "start": pd.Timestamp(df[TIME_COL].min()).isoformat(),
        "end": pd.Timestamp(df[TIME_COL].max()).isoformat(),
        "target_mean": round(float(d.mean()), 4),
        "target_std": round(float(d.std()), 4),
        "target_min": round(float(d.min()), 4),
        "target_p25": round(float(d.quantile(0.25)), 4),
        "target_median": round(float(d.median()), 4),
        "target_p75": round(float(d.quantile(0.75)), 4),
        "target_max": round(float(d.max()), 4),
    }


def chronological_split(
    df: pd.DataFrame,
    ratios: Iterable[float] = SPLIT_RATIOS,
    time_col: str = TIME_COL,
    target: str = TARGET,
    boundary: str = "rows",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split a time-ordered frame into train/val/test and purge boundaries.

    boundary="rows" (default) puts the cut at the timestamp where the cumulative
    row count reaches each ratio, so the splits hold ~70/15/15 of the rows while
    staying strictly chronological. boundary="time" splits the calendar evenly.
    """
    ratios = tuple(ratios)
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be three numbers summing to 1, got {ratios}")

    df = df.sort_values([time_col, HOUSE_COL]).reset_index(drop=True)
    counts = df.groupby(time_col, sort=True).size()
    n_times = len(counts)
    if n_times < 10:
        raise ValueError("not enough distinct timestamps to split")

    # Boundaries are actual timestamps in the data (no row is split).
    if boundary == "rows":
        cumulative = counts.cumsum().to_numpy() / len(df)
        i_val = int(np.searchsorted(cumulative, ratios[0], side="left"))
        i_test = int(
            np.searchsorted(cumulative, ratios[0] + ratios[1], side="left")
        )
    elif boundary == "time":
        i_val = int(np.floor(ratios[0] * n_times))
        i_test = int(np.floor((ratios[0] + ratios[1]) * n_times))
    else:
        raise ValueError(f"boundary must be 'rows' or 'time', got {boundary!r}")

    i_val = min(max(i_val, 1), n_times - 2)
    i_test = min(max(i_test, i_val + 1), n_times - 1)
    val_start = pd.Timestamp(counts.index[i_val])
    test_start = pd.Timestamp(counts.index[i_test])

    train = df[df[time_col] < val_start]
    val = df[(df[time_col] >= val_start) & (df[time_col] < test_start)]
    test = df[df[time_col] >= test_start]

    # Purge the last origin of train/val: its label lies in the next split.
    purge_train = train[time_col] >= val_start - pd.Timedelta(hours=1)
    purge_val = val[time_col] >= test_start - pd.Timedelta(hours=1)
    n_purged = int(purge_train.sum() + purge_val.sum())
    train = train[~purge_train]
    val = val[~purge_val]

    total = len(df)
    summary = {
        "ratios": list(ratios),
        "boundary_rule": boundary,
        "boundaries": {
            "val_start": val_start.isoformat(),
            "test_start": test_start.isoformat(),
        },
        "rows_total_input": int(total),
        "rows_purged_at_boundaries": n_purged,
        "splits": {
            "train": _describe(train, total),
            "val": _describe(val, total),
            "test": _describe(test, total),
        },
    }

    _verify(train, val, test, val_start, test_start, time_col, total, n_purged)
    return train, val, test, summary


def _verify(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    time_col: str,
    total: int,
    n_purged: int,
) -> None:
    for name, part in zip(SPLIT_NAMES, (train, val, test)):
        if part.empty:
            raise AssertionError(f"{name} split is empty")

    if not (train[time_col].max() < val[time_col].min()):
        raise AssertionError("train/val overlap")
    if not (val[time_col].max() < test[time_col].min()):
        raise AssertionError("val/test overlap")

    # Train labels must stay strictly before the validation period.
    if not (train[time_col].max() + pd.Timedelta(hours=1) < val_start):
        raise AssertionError("train label leaks into validation period")
    if not (val[time_col].max() + pd.Timedelta(hours=1) < test_start):
        raise AssertionError("val label leaks into test period")

    if len(train) + len(val) + len(test) != total - n_purged:
        raise AssertionError("row counts inconsistent after purging")


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    summary: dict,
) -> list[Path]:
    ensure_dirs()
    paths = []
    for name, part in zip(SPLIT_NAMES, (train, val, test)):
        path = SPLITS_DIR / f"{name}.parquet"
        part.to_parquet(path, index=False)
        paths.append(path)
    SPLIT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.append(SPLIT_SUMMARY)
    return paths


def main() -> None:
    features = pd.read_parquet(FEATURES_DATASET)
    train, val, test, summary = chronological_split(features)
    for name, part in zip(SPLIT_NAMES, (train, val, test)):
        s = summary["splits"][name]
        print(
            f"{name:<5} rows={s['rows']:>7,} ({s['pct_of_total']:>5.2f}%) "
            f"{s['start'][:10]} -> {s['end'][:10]}"
        )
    print(f"purged boundary rows: {summary['rows_purged_at_boundaries']:,}")
    for path in save_splits(train, val, test, summary):
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
