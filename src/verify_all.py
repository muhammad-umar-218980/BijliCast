"""End-to-end verification for a BijliCast checkout.

Run from the repository root::

    python src/verify_all.py

The script uses the tracked deployment artifacts. Data-dependent checks
(split summary, train/serve parity, pipeline smoke test) are skipped with a
clear message when the git-ignored ``data/processed/`` files are absent, so the
same script works on a fresh clone and in CI.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    suffix = f": {detail}" if detail else ""
    print(f"[{status:4s}] {name}{suffix}")


def check(name: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        record(name, "FAIL", f"{type(exc).__name__}: {exc}")
    else:
        record(name, "PASS", "" if detail is None else str(detail))


def skip(name: str, reason: str) -> None:
    record(name, "SKIP", reason)


# --------------------------------------------------------------------------- checks
def check_artifacts() -> str:
    required = [
        "models/deployment_model.joblib",
        "models/house_profiles.csv",
        "models/usage_history.parquet",
        "models/latest_weather.csv",
        "models/model_metadata.json",
        "models/app_manifest.json",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f"missing artifacts: {missing}")
    return f"{len(required)} artifacts present"


def check_model() -> str:
    import joblib

    model = joblib.load(ROOT / "models/deployment_model.joblib")
    estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    name = type(estimator).__name__
    if name != "HistGradientBoostingRegressor":
        raise AssertionError(f"unexpected estimator: {name}")
    return name


def check_metadata() -> str:
    meta = json.loads((ROOT / "models/model_metadata.json").read_text())
    deployment = meta.get("deployment_model")
    if not deployment:
        raise AssertionError("missing deployment_model block")
    n_train = int(deployment.get("n_train_rows", 0))
    if n_train <= 0:
        raise AssertionError("deployment_model.n_train_rows is not positive")
    return f"deployed rows={n_train:,}"


def check_service() -> str:
    from app.service import BijliCastService

    service = BijliCastService(ROOT)
    houses = service.house_ids()
    if len(houses) != 59:
        raise AssertionError(f"expected 59 houses, got {len(houses)}")
    result = service.predict("House#1")
    if not np.isfinite(result.prediction_kwh):
        raise AssertionError("prediction is not finite")
    if not (result.interval_low_kwh <= result.prediction_kwh <= result.interval_high_kwh):
        raise AssertionError("prediction is outside its interval")
    if len(result.history) != 72:
        raise AssertionError(f"expected 72 history points, got {len(result.history)}")
    return (
        f"House#1={result.prediction_kwh:.4f} kWh, "
        f"interval={result.interval_low_kwh:.3f}-{result.interval_high_kwh:.3f}"
    )


def check_all_houses() -> str:
    from app.service import BijliCastService

    service = BijliCastService(ROOT)
    bad: list[tuple[str, str]] = []
    for house_id in service.house_ids():
        result = service.predict(house_id)
        if not np.isfinite(result.prediction_kwh):
            bad.append((house_id, "non-finite prediction"))
        elif not (
            result.interval_low_kwh
            <= result.prediction_kwh
            <= result.interval_high_kwh
        ):
            bad.append((house_id, "prediction outside interval"))
    if bad:
        raise AssertionError(f"{len(bad)} bad predictions: {bad[:5]}")
    return f"{len(service.house_ids())} houses OK"


def check_feature_count() -> str:
    import joblib

    from src.preprocessing.pipeline import get_feature_names

    model = joblib.load(ROOT / "models/deployment_model.joblib")
    names = get_feature_names(model)
    if not names:
        raise AssertionError("pipeline exposes no feature names")
    meta = json.loads((ROOT / "models/model_metadata.json").read_text())
    recorded = meta.get("n_features_after_transform")
    if recorded is not None and int(recorded) != len(names):
        raise AssertionError(f"metadata says {recorded}, pipeline has {len(names)}")
    return f"{len(names)} transformed features"


def check_notebooks() -> str:
    import nbformat

    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        raise FileNotFoundError("no notebooks found")
    unexecuted: list[str] = []
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code" and cell.get("execution_count") is None:
                unexecuted.append(path.name)
                break
    if unexecuted:
        raise AssertionError(f"unexecuted notebooks: {unexecuted}")
    return f"{len(paths)} notebooks valid and executed"


def check_api() -> str:
    from fastapi.testclient import TestClient

    from app.api import app

    client = TestClient(app)
    health = client.get("/health")
    houses = client.get("/houses")
    predict = client.post(
        "/predict", json={"house_id": "House#1", "weather": {"temperature": 30}}
    )
    unknown = client.post("/predict", json={"house_id": "House#999"})
    bad_weather = client.post(
        "/predict", json={"house_id": "House#1", "weather": {"temperature": 999}}
    )
    if health.status_code != 200:
        raise AssertionError(f"/health -> {health.status_code}")
    if houses.status_code != 200 or len(houses.json()["houses"]) != 59:
        raise AssertionError(f"/houses -> {houses.status_code}")
    if predict.status_code != 200:
        raise AssertionError(f"/predict -> {predict.status_code}")
    if unknown.status_code != 404:
        raise AssertionError(f"unknown house -> {unknown.status_code}")
    if bad_weather.status_code != 422:
        raise AssertionError(f"bad weather -> {bad_weather.status_code}")
    return "health/houses/predict/404/422 OK"


def check_bundle_sync() -> str:
    pairs = [
        ("app/service.py", "deploy/huggingface/service.py"),
        ("app/streamlit_ui.py", "deploy/huggingface/streamlit_ui.py"),
        ("src/utils/config.py", "deploy/huggingface/src/utils/config.py"),
        ("src/features/inference.py", "deploy/huggingface/src/features/inference.py"),
        ("src/preprocessing/pipeline.py", "deploy/huggingface/src/preprocessing/pipeline.py"),
    ]
    missing = [b for _, b in pairs if not (ROOT / b).exists()]
    if missing:
        raise FileNotFoundError(f"HF bundle not generated: {missing}")
    mismatched: list[str] = []
    for left, right in pairs:
        left_hash = hashlib.sha256((ROOT / left).read_bytes()).hexdigest()
        right_hash = hashlib.sha256((ROOT / right).read_bytes()).hexdigest()
        if left_hash != right_hash:
            mismatched.append(f"{left} != {right}")
    if mismatched:
        raise AssertionError(
            "HF bundle is stale; run `python -m src.deploy.export_hf_space`. "
            + "; ".join(mismatched)
        )
    return f"{len(pairs)} shared modules in sync"


def check_split_summary() -> str:
    from src.utils.config import SPLIT_SUMMARY

    if not SPLIT_SUMMARY.exists():
        raise FileNotFoundError(f"{SPLIT_SUMMARY} not found; run the pipeline first")
    summary = json.loads(SPLIT_SUMMARY.read_text())
    splits = summary["splits"]
    pcts = {
        name: float(splits[name]["pct_of_total"]) for name in ("train", "val", "test")
    }
    total = sum(pcts.values())
    if abs(total - 100.0) > 0.1:
        raise AssertionError(f"split percentages sum to {total:.2f}")
    return ", ".join(f"{k}={v:.2f}%" for k, v in pcts.items())


def check_parity() -> str:
    from src.features.inference import self_check

    self_check()
    return "train/serve parity passed"


def check_pipeline_smoke() -> str:
    from src.preprocessing.pipeline import smoke_test

    smoke_test()
    return "pipeline smoke test passed"


# --------------------------------------------------------------------------- main
def main() -> int:
    from src.utils.config import FEATURES_DATASET, MERGED_DATASET, SPLITS_DIR

    check("artifacts", check_artifacts)
    check("model loads", check_model)
    check("metadata deployment block", check_metadata)
    check("service prediction", check_service)
    check("all 59 houses", check_all_houses)
    check("feature count", check_feature_count)
    check("notebooks", check_notebooks)
    check("FastAPI", check_api)
    check("HF bundle sync", check_bundle_sync)

    split_ready = (SPLITS_DIR / "train.parquet").exists()
    parity_ready = split_ready and MERGED_DATASET.exists() and FEATURES_DATASET.exists()

    if split_ready:
        check("split summary", check_split_summary)
    else:
        skip("split summary", "data/processed/splits/train.parquet not present")

    if parity_ready:
        check("train/serve parity", check_parity)
    else:
        skip("train/serve parity", "processed merged/features data not present")

    if split_ready:
        check("pipeline smoke test", check_pipeline_smoke)
    else:
        skip("pipeline smoke test", "data/processed/splits/train.parquet not present")

    failures = [name for name, status, _ in RESULTS if status == "FAIL"]
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print(f"ALL CHECKS PASSED ({len(RESULTS)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
