"""Preprocessing pipeline shape and train-only fitting."""
from __future__ import annotations

import joblib
import pytest

from src.utils.config import SPLITS_DIR

pytestmark = pytest.mark.skipif(
    not (SPLITS_DIR / "train.parquet").exists(),
    reason="data/processed/splits/ is not present in this checkout",
)


def test_pipeline_feature_count_matches_metadata(root):
    from src.preprocessing.pipeline import get_feature_names

    model = joblib.load(root / "models/deployment_model.joblib")
    names = get_feature_names(model)
    assert len(names) == 49
    assert len(set(names)) == len(names)


def test_pipeline_smoke_test():
    from src.preprocessing.pipeline import smoke_test

    smoke_test()
