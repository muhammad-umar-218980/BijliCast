"""The tracked deployment artifacts must be present and loadable."""
from __future__ import annotations

import json

import joblib

REQUIRED = [
    "models/deployment_model.joblib",
    "models/house_profiles.csv",
    "models/usage_history.parquet",
    "models/latest_weather.csv",
    "models/model_metadata.json",
    "models/app_manifest.json",
]


def test_required_artifacts_exist(root):
    missing = [path for path in REQUIRED if not (root / path).exists()]
    assert not missing, f"missing artifacts: {missing}"


def test_deployment_model_is_hist_gradient_boosting(root):
    model = joblib.load(root / "models/deployment_model.joblib")
    assert type(model.named_steps["model"]).__name__ == "HistGradientBoostingRegressor"


def test_metadata_has_deployment_block(root):
    meta = json.loads((root / "models/model_metadata.json").read_text())
    deployment = meta.get("deployment_model")
    assert deployment, "missing deployment_model block"
    assert int(deployment["n_train_rows"]) > 0
    assert deployment["prediction_interval"]["val_coverage"] is not None
    assert deployment.get("metrics_source"), "deployment metrics source is not labelled"


def test_manifest_covers_all_houses(root):
    manifest = json.loads((root / "models/app_manifest.json").read_text())
    assert manifest["houses"] == 59
    assert manifest["skipped_houses"] == []
