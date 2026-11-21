"""Export the compact deployment model (HistGradientBoosting).

Why HistGradientBoosting instead of the validation-winning Random Forest?

* RF and HGB are statistically tied on validation (0.3216 vs 0.3273 MAE,
  CV fold std ~0.067) and HGB is slightly better on the one-shot test.
* The RF artifact is ~270 MB, which GitHub rejects (>100 MB per file) and
  Streamlit Community Cloud cannot clone. HGB is ~2 MB.
* HGB inference is faster and needs no special hosting/LFS.

The RF pipeline remains `models/best_model.joblib` for the report/metrics
alignment; this module writes the artifact the apps actually serve.

CLI:
    python -m src.models.export_deployment
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

from src.models.baseline import load_splits  # noqa: E402
from src.models.train import make_estimator  # noqa: E402
from src.models.tune import best_params_from, load_tuning_results  # noqa: E402
from src.preprocessing.pipeline import build_pipeline, get_xy  # noqa: E402
from src.utils.config import (  # noqa: E402
    INPUT_COLUMNS,
    MODELS_DIR,
    TABLES_DIR,
    TARGET,
    TIME_COL,
)

DEPLOY_NAME = "HistGradientBoosting"
INTERVAL_LOW_Q = 0.05
INTERVAL_HIGH_Q = 0.95


def _interval_coverage(y_true, pred, low, high) -> float:
    lo = np.clip(pred + low, 0, None)
    hi = pred + high
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def export_deployment_artifacts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tuned_params: dict[str, dict],
    comparison: pd.DataFrame | None = None,
    persist: bool = True,
) -> dict:
    """Fit HGB on train+val, save it, and update ``model_metadata.json``."""
    params = tuned_params[DEPLOY_NAME]

    # --- interval calibration: train-fit HGB -> out-of-sample validation
    calibration = build_pipeline(make_estimator(DEPLOY_NAME, params))
    calibration.fit(*get_xy(train_df))
    val_pred = calibration.predict(val_df[INPUT_COLUMNS])
    val_true = val_df[TARGET].to_numpy(dtype=float)
    residuals = val_true - val_pred
    low, high = (float(q) for q in np.quantile(residuals, [INTERVAL_LOW_Q, INTERVAL_HIGH_Q]))

    # --- deployment fit: train + validation (test is never used)
    deploy_df = pd.concat([train_df, val_df]).sort_values(TIME_COL)
    deploy_pipe = build_pipeline(make_estimator(DEPLOY_NAME, params))
    deploy_pipe.fit(*get_xy(deploy_df))

    model_path = MODELS_DIR / "deployment_model.joblib"
    joblib.dump(deploy_pipe, model_path)

    # --- test coverage from the already-computed one-shot predictions
    test_coverage = None
    all_test_path = MODELS_DIR / "test_predictions_all.parquet"
    if all_test_path.exists():
        all_test = pd.read_parquet(all_test_path)
        col = f"{DEPLOY_NAME}__pred"
        if col in all_test.columns:
            test_coverage = _interval_coverage(
                all_test["actual"].to_numpy(dtype=float),
                all_test[col].to_numpy(dtype=float),
                low,
                high,
            )

    if comparison is None:
        comparison = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    rows = comparison[comparison["model"] == DEPLOY_NAME]
    metrics = {
        row["split"]: {
            "n": int(row["n"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "r2": float(row["r2"]),
            "median_ae": float(row["median_ae"]),
        }
        for _, row in rows.iterrows()
    }

    deployment = {
        "name": DEPLOY_NAME,
        "artifact": "models/deployment_model.joblib",
        "hyperparameters": params,
        "n_train_rows": int(len(deploy_df)),
        "fit_data": "train + validation (test never used for fitting)",
        "metrics": metrics,
        "prediction_interval": {
            "method": "train-fit HGB validation residual 5th/95th percentile",
            "low_offset": low,
            "high_offset": high,
            "val_coverage": _interval_coverage(val_true, val_pred, low, high),
            "test_coverage": test_coverage,
        },
        "selection_rationale": (
            "Random Forest had the best validation MAE by a statistically "
            "insignificant margin (0.3216 vs 0.3273, CV fold std ~0.067). "
            "HistGradientBoosting was chosen for deployment because it is "
            "tied on accuracy, has the best one-shot test MAE, and its "
            "artifact is ~2 MB instead of ~270 MB (GitHub rejects files "
            ">100 MB, and Streamlit Community Cloud deploys from GitHub)."
        ),
    }

    if persist:
        meta_path = MODELS_DIR / "model_metadata.json"
        metadata = json.loads(meta_path.read_text())
        metadata["deployment_model"] = deployment
        metadata["artifacts"]["deployment_model"] = "models/deployment_model.joblib"
        metadata["artifacts"]["best_model_note"] = (
            "models/best_model.joblib is the validation-selected Random Forest, "
            "kept locally for report/metric alignment and git-ignored because "
            "it exceeds GitHub's 100 MB file limit."
        )
        metadata["notes"] = (
            "Random Forest won the validation MAE and is saved as best_model.joblib "
            "(train-fit). The deployed artifact is HistGradientBoosting: tied on "
            "validation within CV noise, best one-shot test MAE, ~2 MB, so it can be "
            "committed to GitHub and served by Streamlit Community Cloud. Test was "
            "evaluated once, after selection."
        )
        meta_path.write_text(json.dumps(metadata, indent=2))
        deployment["size_mb"] = round(model_path.stat().st_size / (1024**2), 3)
        meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"deployment model: {DEPLOY_NAME}")
    print(f"  trained on {len(deploy_df):,} rows (train + val)")
    print(f"  artifact: {model_path} ({model_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"  val metrics: MAE {metrics['val']['mae']:.4f} | R2 {metrics['val']['r2']:.4f}")
    print(f"  test metrics: MAE {metrics['test']['mae']:.4f} | R2 {metrics['test']['r2']:.4f}")
    test_cov_text = f"{test_coverage:.2%}" if test_coverage is not None else "n/a"
    print(f"  interval: [{low:+.3f}, {high:+.3f}] kWh | val coverage "
          f"{deployment['prediction_interval']['val_coverage']:.2%} | test coverage "
          f"{test_cov_text}")
    return deployment


def main() -> None:
    train, val, test = load_splits()
    tuned = best_params_from(load_tuning_results())
    export_deployment_artifacts(train, val, tuned)


if __name__ == "__main__":
    main()
