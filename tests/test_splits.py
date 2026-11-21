"""Chronological split summary structure and percentages."""
from __future__ import annotations

import json

import pytest

from src.utils.config import SPLIT_SUMMARY

pytestmark = pytest.mark.skipif(
    not SPLIT_SUMMARY.exists(), reason="split_summary.json is not present"
)


def test_split_summary_structure():
    summary = json.loads(SPLIT_SUMMARY.read_text())
    assert set(summary["splits"]) == {"train", "val", "test"}
    for name in ("train", "val", "test"):
        assert summary["splits"][name]["rows"] > 0
        assert summary["splits"][name]["houses"] > 0


def test_split_percentages_sum_to_100():
    summary = json.loads(SPLIT_SUMMARY.read_text())
    total = sum(
        summary["splits"][name]["pct_of_total"] for name in ("train", "val", "test")
    )
    assert abs(total - 100.0) < 0.1
