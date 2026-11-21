"""Sync the self-contained Gradio Space bundle (ZeroGPU).

Mirrors ``export_hf_space.py``: the Space is a separate git repository, so this
copies the runtime pieces it needs and runs a real prediction inside the
generated folder.

CLI:
    python -m src.deploy.export_hf_gradio
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import MODELS_DIR, ROOT  # noqa: E402

SPACE_DIR = ROOT / "deploy" / "huggingface-gradio"

ARTIFACTS = [
    "deployment_model.joblib",
    "house_profiles.csv",
    "usage_history.parquet",
    "latest_weather.csv",
    "model_metadata.json",
    "app_manifest.json",
]

SOURCE_FILES = [
    ("app/service.py", "service.py"),
    ("src/__init__.py", "src/__init__.py"),
    ("src/utils/__init__.py", "src/utils/__init__.py"),
    ("src/utils/config.py", "src/utils/config.py"),
    ("src/preprocessing/__init__.py", "src/preprocessing/__init__.py"),
    ("src/preprocessing/pipeline.py", "src/preprocessing/pipeline.py"),
    ("src/features/__init__.py", "src/features/__init__.py"),
    ("src/features/inference.py", "src/features/inference.py"),
]

REQUIRED_STATIC = ["app.py", "README.md", "requirements.txt"]


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def export_space(check: bool = True) -> Path:
    missing_static = [n for n in REQUIRED_STATIC if not (SPACE_DIR / n).exists()]
    if missing_static:
        raise FileNotFoundError(
            f"missing static Space files in {SPACE_DIR}: {missing_static}"
        )
    missing_artifacts = [n for n in ARTIFACTS if not (MODELS_DIR / n).exists()]
    if missing_artifacts:
        raise FileNotFoundError(
            f"missing deployment artifacts: {missing_artifacts}. "
            "Run the export scripts first."
        )

    for relative, destination in SOURCE_FILES:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"missing source file: {source}")
        _copy(source, SPACE_DIR / destination)

    for name in ARTIFACTS:
        _copy(MODELS_DIR / name, SPACE_DIR / "models" / name)

    _clean_caches()
    if check:
        _self_check()
        _clean_caches()

    files = [p for p in SPACE_DIR.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"space directory: {SPACE_DIR}")
    print(f"files: {len(files)} | total size: {total / 1024**2:.2f} MB")
    print(
        "model: "
        f"{(SPACE_DIR / 'models' / 'deployment_model.joblib').stat().st_size / 1024**2:.2f} MB"
    )
    return SPACE_DIR


def _clean_caches() -> None:
    for cache in SPACE_DIR.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    for stale in SPACE_DIR.rglob("*.pyc"):
        stale.unlink(missing_ok=True)


def _self_check() -> None:
    code = (
        "from service import BijliCastService\n"
        "s = BijliCastService('.')\n"
        "r = s.predict(s.house_ids()[0])\n"
        "assert r.prediction_kwh >= 0, r.prediction_kwh\n"
        "assert r.interval_low_kwh <= r.prediction_kwh <= r.interval_high_kwh\n"
        "assert len(r.history) == 72, len(r.history)\n"
        "print(f\"self-check: {len(s.house_ids())} houses | {r.house_id} \"\n"
        "      f\"{r.prediction_kwh:.3f} kWh \"\n"
        "      f\"[{r.interval_low_kwh:.2f}, {r.interval_high_kwh:.2f}]\")\n"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SPACE_DIR,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Gradio Space self-check failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    print(result.stdout.strip())


if __name__ == "__main__":
    export_space()
