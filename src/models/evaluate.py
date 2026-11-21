"""Final evaluation: refit tuned models, evaluate test ONCE, save artifacts.

Workflow (guideline steps 6-7):

1. Read the winning configuration per algorithm from
   ``models/tuning_results.json`` (produced by ``src.models.tune``).
2. Refit every tuned pipeline on the **training split** and score train/val.
3. Select the final model by **validation MAE** (test is never used for
   selection).
4. Evaluate the held-out test split **once**, in this single pass, and report
   MAE / RMSE / R2 for every candidate plus the naive baselines.
5. Error analysis by hour / city / household / month; prediction intervals
   from validation residual quantiles.
6. Save the complete pipeline with joblib:

   * ``models/best_model.joblib``       - fitted on train only (metrics-aligned)
   * ``models/deployment_model.joblib`` - refitted on train+val for serving
   * ``models/model_metadata.json``     - metrics, params, interval, versions

CLI:
    python -m src.models.evaluate
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.baseline import evaluate_baselines, load_splits  # noqa: E402
from src.models.metrics import metrics_row, metrics_table  # noqa: E402
from src.models.train import (  # noqa: E402
    KNN_TRAIN_ROWS,
    MODEL_NAMES,
    make_estimator,
    time_ordered_subsample,
)
from src.models.tune import best_params_from, load_tuning_results  # noqa: E402
from src.preprocessing.pipeline import build_pipeline, get_xy  # noqa: E402
from src.utils.config import (  # noqa: E402
    CITY_COL,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    HOUSE_COL,
    INPUT_COLUMNS,
    MODELS_DIR,
    SPLIT_SUMMARY,
    TABLES_DIR,
    TARGET,
    TIME_COL,
)

INTERVAL_LOW_Q = 0.05
INTERVAL_HIGH_Q = 0.95


def refit_model(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    params: dict | None = None,
    train_subsample: int | None = None,
) -> dict:
    """Fit one tuned pipeline and predict every requested split."""
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
        "metrics": {},
        "predictions": {},
        "index": {},
    }
    for split, df in (("train", fit_df), ("val", val_df), ("test", test_df)):
        if df is None:
            continue
        X, y = get_xy(df)
        pred = pipe.predict(X)
        result["predictions"][split] = pred
        result["index"][split] = df
        result["metrics"][split] = metrics_row(name, split, y, pred)
    return result


def prediction_frame(df: pd.DataFrame, y_pred) -> pd.DataFrame:
    """Tidy frame used by the notebooks for error analysis."""
    actual = df[TARGET].to_numpy(dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    frame = pd.DataFrame(
        {
            TIME_COL: pd.to_datetime(df[TIME_COL]).to_numpy(),
            HOUSE_COL: df[HOUSE_COL].to_numpy(),
            CITY_COL: df[CITY_COL].to_numpy(),
            "actual": actual,
            "predicted": predicted,
            "residual": actual - predicted,
            "abs_error": np.abs(actual - predicted),
        }
    )
    frame["hour"] = frame[TIME_COL].dt.hour
    frame["month"] = frame[TIME_COL].dt.to_period("M").astype(str)
    return frame


def group_errors(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    table = (
        frame.groupby(by)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("residual", lambda s: float(np.sqrt(np.mean(s**2)))),
            bias=("residual", "mean"),
            actual_mean=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )
    for col in ("mae", "rmse", "bias", "actual_mean", "predicted_mean"):
        table[col] = table[col].round(4)
    return table


def interval_stats(frame: pd.DataFrame, low_offset: float, high_offset: float) -> dict:
    lo = np.clip(frame["predicted"] + low_offset, 0, None)
    hi = frame["predicted"] + high_offset
    inside = (frame["actual"] >= lo) & (frame["actual"] <= hi)
    return {
        "n": int(len(frame)),
        "coverage": float(inside.mean()),
        "mean_width": float((hi - lo).mean()),
        "low_offset": float(low_offset),
        "high_offset": float(high_offset),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def run_final_evaluation(persist: bool = True) -> dict:
    """The single one-shot test evaluation. Returns everything for reporting."""
    train, val, test = load_splits()
    tuned = best_params_from(load_tuning_results())
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print("tuned configurations:", {k: v for k, v in tuned.items()})

    results: dict[str, dict] = {}
    for name in MODEL_NAMES:
        subsample = KNN_TRAIN_ROWS if name == "KNN" else None
        print(f"[{name}] refitting tuned pipeline ...", end=" ", flush=True)
        res = refit_model(name, train, val, test, tuned[name], subsample)
        results[name] = res
        print(
            f"fit {res['fit_seconds']:.1f}s | train MAE {res['metrics']['train']['mae']:.4f} "
            f"| val MAE {res['metrics']['val']['mae']:.4f} | test MAE {res['metrics']['test']['mae']:.4f}"
        )

    # ---- naive baselines on all splits (no fitting, no leakage)
    train_mean = float(train[TARGET].mean())
    baseline_rows: list[dict] = []
    for split, df in (("train", train), ("val", val), ("test", test)):
        baseline_rows += evaluate_baselines(df, split, train_mean)
    baseline_table = metrics_table(baseline_rows)

    # ---- comparison table (models + baselines)
    model_rows: list[dict] = []
    for name, res in results.items():
        for split in ("train", "val", "test"):
            model_rows.append({**res["metrics"][split], "kind": "model"})
    model_table = metrics_table(model_rows)
    model_table["kind"] = "model"
    baseline_table["kind"] = "baseline"
    comparison = pd.concat([model_table, baseline_table], ignore_index=True)
    split_order = {"train": 0, "val": 1, "test": 2}
    comparison["_o"] = comparison["split"].map(split_order).fillna(3)
    comparison = (
        comparison.sort_values(["split", "_o", "mae"])
        .drop(columns="_o")
        .reset_index(drop=True)
    )

    # ---- select final model on validation MAE only
    best_name = min(results, key=lambda n: results[n]["metrics"]["val"]["mae"])
    best = results[best_name]
    print(f"\nselected on validation MAE: {best_name}")

    best_val = prediction_frame(val, best["predictions"]["val"])
    best_test = prediction_frame(test, best["predictions"]["test"])
    best_train = prediction_frame(best["index"]["train"], best["predictions"]["train"])

    # ---- prediction intervals from validation residuals only
    val_residuals = best_val["actual"].to_numpy() - best_val["predicted"].to_numpy()
    low_offset, high_offset = (float(q) for q in np.quantile(val_residuals, [INTERVAL_LOW_Q, INTERVAL_HIGH_Q]))
    intervals = {
        "method": f"validation residual {int(INTERVAL_LOW_Q*100)}th/{int(INTERVAL_HIGH_Q*100)}th percentile",
        "train": interval_stats(best_train, low_offset, high_offset),
        "val": interval_stats(best_val, low_offset, high_offset),
        "test": interval_stats(best_test, low_offset, high_offset),
    }

    # ---- error analysis for the final model
    error_tables = {
        "hour": group_errors(best_test, "hour"),
        "city": group_errors(best_test, CITY_COL),
        "house": group_errors(best_test, HOUSE_COL),
        "month": group_errors(best_test, "month"),
    }

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "selected_model": best_name,
        "selection_rule": "lowest validation MAE among tuned learned models",
        "tuned_params": tuned,
        "results": results,
        "comparison": comparison,
        "baselines": baseline_table,
        "best_val": best_val,
        "best_test": best_test,
        "best_train": best_train,
        "intervals": intervals,
        "error_tables": error_tables,
        "train": train,
        "val": val,
        "test": test,
    }

    if persist:
        _persist(payload)
    return payload


def _persist(payload: dict) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    best_name = payload["selected_model"]
    best = payload["results"][best_name]
    comparison = payload["comparison"]

    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    comparison[comparison["split"] == "test"].to_csv(TABLES_DIR / "test_metrics.csv", index=False)
    payload["baselines"].to_csv(TABLES_DIR / "baseline_metrics.csv", index=False)

    for name, table in payload["error_tables"].items():
        table.to_csv(TABLES_DIR / f"error_by_{name}.csv", index=False)

    coverage = pd.DataFrame(
        [
            {"split": split, **stats}
            for split, stats in payload["intervals"].items()
            if isinstance(stats, dict)
        ]
    )
    coverage.to_csv(TABLES_DIR / "prediction_interval_coverage.csv", index=False)

    payload["best_val"].to_parquet(MODELS_DIR / "best_val_predictions.parquet", index=False)
    payload["best_test"].to_parquet(MODELS_DIR / "best_test_predictions.parquet", index=False)

    # all-model test predictions for the comparison plot
    test = payload["test"]
    all_test = pd.DataFrame(
        {
            TIME_COL: pd.to_datetime(test[TIME_COL]).to_numpy(),
            HOUSE_COL: test[HOUSE_COL].to_numpy(),
            CITY_COL: test[CITY_COL].to_numpy(),
            "actual": test[TARGET].to_numpy(dtype=float),
        }
    )
    for name, res in payload["results"].items():
        all_test[f"{name}__pred"] = res["predictions"]["test"]
    all_test.to_parquet(MODELS_DIR / "test_predictions_all.parquet", index=False)

    # ---- joblib artifacts
    joblib.dump(best["pipeline"], MODELS_DIR / "best_model.joblib")
    # Compact deployment artifact (HistGradientBoosting) + metadata update.
    from src.models.export_deployment import export_deployment_artifacts

    export_deployment_artifacts(
        payload["train"], payload["val"], payload["tuned_params"], comparison
    )

    import sklearn

    split_summary = json.loads(SPLIT_SUMMARY.read_text()) if SPLIT_SUMMARY.exists() else {}
    metadata = {
        "project": "BijliCast",
        "task": "next-hour household electricity consumption regression",
        "target": TARGET,
        "selected_model": best_name,
        "selection_rule": payload["selection_rule"],
        "trained_at": payload["generated_at"],
        "hyperparameters": payload["tuned_params"][best_name],
        "n_train_rows": best["n_train"],
        "metrics": {split: best["metrics"][split] for split in ("train", "val", "test")},
        "test_metrics_all_models": comparison[comparison["split"] == "test"].to_dict(orient="records"),
        "prediction_interval": payload["intervals"],
        "features": FEATURE_COLUMNS,
        "input_columns": INPUT_COLUMNS,
        "n_features_after_transform": 49,
        "splits": split_summary,
        "artifacts": {
            "best_model": "models/best_model.joblib",
            "deployment_model": "models/deployment_model.joblib",
            "house_profiles": "models/house_profiles.csv",
            "test_predictions": "models/best_test_predictions.parquet",
        },
        "notes": (
            "best_model.joblib is fitted on the training split only so its metrics align with "
            "the reported evaluation. deployment_model.joblib refits the same configuration on "
            "train+validation for serving. Test was evaluated once, after selection."
        ),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2))

    print("\n=== metrics (train / val / test) ===")
    print(comparison.round(4).to_string(index=False))
    print(f"\nfinal model: {best_name}")
    print(f"prediction interval: {payload['intervals']['method']}")
    print(
        f"  offsets [{payload['intervals']['val']['low_offset']:.3f}, "
        f"{payload['intervals']['val']['high_offset']:.3f}] kWh | "
        f"test coverage {payload['intervals']['test']['coverage']:.2%}"
    )
    print("\n=== test MAE by hour (worst 5) ===")
    print(payload["error_tables"]["hour"].head(5).round(4).to_string(index=False))
    print("\n=== test MAE by city ===")
    print(payload["error_tables"]["city"].round(4).to_string(index=False))
    print("\nsaved: models/best_model.joblib, models/deployment_model.joblib, models/model_metadata.json")
    print("saved: reports/tables/model_comparison.csv, test_metrics.csv, error_by_*.csv")


def main() -> None:
    run_final_evaluation(persist=True)


if __name__ == "__main__":
    main()
