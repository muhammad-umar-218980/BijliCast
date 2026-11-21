"""Train/serve feature parity, when the processed data is available."""
from __future__ import annotations

import pytest

from src.utils.config import FEATURES_DATASET, MERGED_DATASET, SPLITS_DIR

DATA_READY = (
    MERGED_DATASET.exists()
    and FEATURES_DATASET.exists()
    and (SPLITS_DIR / "train.parquet").exists()
)

pytestmark = pytest.mark.skipif(
    not DATA_READY, reason="data/processed/ is not present in this checkout"
)


def test_inference_row_matches_training_features():
    from src.features.inference import self_check

    self_check()
