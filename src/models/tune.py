"""Hyper-parameter tuning with 5-fold cross-validation on the training split.

* CV object: ``TimeSeriesSplit(n_splits=5, gap=120)`` - folds respect time
  order and a small gap purges the one-hour target overlap at each boundary.
* Search: ``GridSearchCV`` for Linear Regression, ``RandomizedSearchCV`` for
  KNN / Random Forest / HistGradientBoosting.
* Scoring: negated MAE (the primary metric).
* The **whole pipeline** is passed to the search, so imputation, scaling,
  one-hot encoding and feature engineering are re-fitted inside every fold.
* Validation and test are not used here at all. The winning configuration is
  selected by cross-validated MAE on train only.

Compute guards (documented in the report):
    KNN  - 25k time-ordered subsample
    RF   - 60k time-ordered subsample, then refit best config on full train
    HistGB - 150k time-ordered subsample
    LR   - full train (cheap)

CLI:
    python -m src.models.tune
writes ``reports/tables/cv_results_<model>.csv`` and
``models/tuning_results.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.baseline import load_splits  # noqa: E402
from src.models.train import (  # noqa: E402
    MODEL_NAMES,
    RANDOM_STATE,
    make_estimator,
    time_ordered_subsample,
)
from src.preprocessing.pipeline import build_pipeline, get_xy  # noqa: E402
from src.utils.config import MODELS_DIR, TABLES_DIR, TARGET  # noqa: E402

N_SPLITS = 5
CV_GAP = 120  # rows skipped at each fold boundary (~2 hours of panel data)

SEARCH_SPACES: dict[str, dict] = {
    "Linear Regression": {
        "model__fit_intercept": [True, False],
        "model__positive": [False, True],
    },
    "KNN": {
        "model__n_neighbors": [5, 10, 15, 25, 40],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "Random Forest": {
        "model__n_estimators": [80, 120],
        "model__max_depth": [12, 20, None],
        "model__min_samples_leaf": [2, 5, 10],
        "model__max_features": [0.4, 0.7],
    },
    "HistGradientBoosting": {
        "model__learning_rate": [0.03, 0.06, 0.10, 0.15],
        "model__max_iter": [200, 300, 400],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 1.0, 5.0],
        "model__min_samples_leaf": [20, 50],
    },
}

TUNING_PLAN: dict[str, dict] = {
    "Linear Regression": {"kind": "grid", "subsample": None},
    "KNN": {"kind": "random", "n_iter": 5, "subsample": 20_000},
    "Random Forest": {"kind": "random", "n_iter": 5, "subsample": 40_000},
    "HistGradientBoosting": {"kind": "random", "n_iter": 5, "subsample": 150_000},
}


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def make_cv() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=N_SPLITS, gap=CV_GAP)


def tune_model(
    name: str,
    train_df: pd.DataFrame,
    plan: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Run one 5-fold search on the training split and return a result dict."""
    plan = plan or TUNING_PLAN[name]
    fit_df = time_ordered_subsample(train_df, plan.get("subsample"))
    X, y = get_xy(fit_df)
    pipe = build_pipeline(make_estimator(name, n_jobs=1 if name in ("KNN", "Random Forest") else None))

    common = dict(
        estimator=pipe,
        scoring="neg_mean_absolute_error",
        cv=make_cv(),
        return_train_score=True,
        n_jobs=-1,
        verbose=0,
    )
    started = time.perf_counter()
    if plan["kind"] == "grid":
        search = GridSearchCV(param_grid=SEARCH_SPACES[name], **common)
    else:
        search = RandomizedSearchCV(
            param_distributions=SEARCH_SPACES[name],
            n_iter=plan["n_iter"],
            random_state=RANDOM_STATE,
            **common,
        )
    if verbose:
        print(f"[{name}] {plan['kind']} search on {len(fit_df):,} rows x {N_SPLITS} folds ...", flush=True)
    search.fit(X, y)
    elapsed = time.perf_counter() - started

    cv_frame = pd.DataFrame(search.cv_results_).copy()
    cv_frame["mean_train_mae"] = -cv_frame["mean_train_score"]
    cv_frame["mean_test_mae"] = -cv_frame["mean_test_score"]
    cv_frame["std_test_mae"] = cv_frame["std_test_score"]
    cv_frame = cv_frame[
        ["rank_test_score", "mean_train_mae", "mean_test_mae", "std_test_mae",
         "mean_fit_time", "params"]
    ].sort_values("rank_test_score").reset_index(drop=True)
    cv_frame["mean_train_mae"] = cv_frame["mean_train_mae"].round(4)
    cv_frame["mean_test_mae"] = cv_frame["mean_test_mae"].round(4)
    cv_frame["std_test_mae"] = cv_frame["std_test_mae"].round(4)
    cv_frame["mean_fit_time"] = cv_frame["mean_fit_time"].round(2)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cv_frame.to_csv(TABLES_DIR / f"cv_results_{slug(name)}.csv", index=False)

    best = cv_frame.iloc[0]
    result = {
        "model": name,
        "search": plan["kind"],
        "subsample_rows": int(plan.get("subsample") or len(train_df)),
        "n_candidates": int(len(cv_frame)),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "best_cv_mae": float(-search.best_score_),
        "best_cv_std": float(best["std_test_mae"]),
        "best_cv_train_mae": float(best["mean_train_mae"]),
        "cv_overfit_gap": float(best["mean_test_mae"] - best["mean_train_mae"]),
        "elapsed_seconds": round(elapsed, 1),
        "candidates": cv_frame.to_dict(orient="records"),
    }
    if verbose:
        print(
            f"[{name}] best CV MAE {result['best_cv_mae']:.4f} "
            f"(+/- {result['best_cv_std']:.4f}) in {elapsed:.0f}s | {result['best_params']}"
        )

    # JSON-safe candidate list (numpy scalars -> python floats via round/float)
    clean = []
    for row in result["candidates"]:
        clean.append(
            {
                "rank": int(row["rank_test_score"]),
                "train_mae": float(row["mean_train_mae"]),
                "test_mae": float(row["mean_test_mae"]),
                "std_test_mae": float(row["std_test_mae"]),
                "fit_seconds": float(row["mean_fit_time"]),
                "params": {k.replace("model__", ""): v for k, v in row["params"].items()},
            }
        )
    result["candidates"] = clean
    return result


def tune_all(train_df: pd.DataFrame, models: list[str] | None = None) -> dict:
    """Run the 5-fold search for every model and persist the results."""
    models = models or MODEL_NAMES
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "cv": f"TimeSeriesSplit(n_splits={N_SPLITS}, gap={CV_GAP})",
        "scoring": "neg_mean_absolute_error",
        "selection_split": "train only (cross-validation)",
        "target": TARGET,
        "models": {},
    }
    for name in models:
        payload["models"][name] = tune_model(name, train_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "tuning_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {out}")
    return payload


def load_tuning_results(path: Path | str | None = None) -> dict:
    path = Path(path or (MODELS_DIR / "tuning_results.json"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.models.tune` first")
    return json.loads(path.read_text())


def best_params_from(results: dict) -> dict[str, dict]:
    return {name: info["best_params"] for name, info in results["models"].items()}


def main() -> None:
    train, val, test = load_splits()
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} (only train is searched)")
    payload = tune_all(train)
    print("\n=== best configuration per model (5-fold CV on train) ===")
    for name, info in payload["models"].items():
        print(f"{name:22s} CV MAE {info['best_cv_mae']:.4f} +/- {info['best_cv_std']:.4f} | {info['best_params']}")


if __name__ == "__main__":
    main()
