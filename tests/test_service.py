"""Shared inference service contract."""
from __future__ import annotations

import numpy as np
import pytest

from app.service import UnknownHouseError


def test_house_count(service):
    assert len(service.house_ids()) == 59


def test_house1_prediction(service):
    result = service.predict("House#1")
    assert np.isfinite(result.prediction_kwh)
    assert result.interval_low_kwh <= result.prediction_kwh <= result.interval_high_kwh
    assert len(result.history) == 72


def test_every_house_has_a_valid_interval(service):
    for house_id in service.house_ids():
        result = service.predict(house_id)
        assert np.isfinite(result.prediction_kwh), house_id
        assert (
            result.interval_low_kwh
            <= result.prediction_kwh
            <= result.interval_high_kwh
        ), house_id


def test_unknown_house_raises(service):
    with pytest.raises(UnknownHouseError):
        service.predict("House#999")
