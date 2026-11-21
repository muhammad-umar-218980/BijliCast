"""Shared inference service for the BijliCast deployments.

Both the FastAPI backend and the Streamlit UI use one contract:

    household profile
  + the last 168 hourly usage values for that house
  + current-hour weather
  + the origin timestamp
      -> build_inference_row -> deployment_model.joblib -> next-hour kWh

Keeping this out of the web frameworks means the API, the Streamlit app and
the Hugging Face Space cannot drift apart.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.inference import build_inference_row
from src.utils.config import (
    CITY_COL,
    HOUSE_COL,
    INPUT_WEATHER_FEATURES,
    STATIC_CATEGORICAL_FEATURES,
    STATIC_NUMERIC_FEATURES,
    TIME_COL,
)

REQUIRED_ARTIFACTS = (
    "deployment_model.joblib",
    "house_profiles.csv",
    "usage_history.parquet",
    "latest_weather.csv",
    "model_metadata.json",
    "app_manifest.json",
)


class BijliCastError(RuntimeError):
    """Base error for the deployment service."""


class ArtifactsMissingError(BijliCastError):
    """Raised when the committed deployment bundle is incomplete."""


class UnknownHouseError(BijliCastError):
    """Raised when a request names a household that is not in the bundle."""


def _finite(value) -> float | None:
    """JSON-safe float: NaN/inf become None."""
    number = float(value)
    return number if np.isfinite(number) else None


def _profile_payload(profile: pd.Series) -> dict:
    """Small, JSON-safe profile block shown by the apps."""
    out: dict = {}
    for col in STATIC_NUMERIC_FEATURES:
        out[col] = _finite(profile[col])
    for col in STATIC_CATEGORICAL_FEATURES:
        value = profile[col]
        out[col] = None if pd.isna(value) else str(value)
    return out


@dataclass
class PredictionResult:
    house_id: str
    city: str
    origin_time: str
    target_time: str
    prediction_kwh: float
    interval_low_kwh: float
    interval_high_kwh: float
    last_hour_kwh: float
    delta_kwh: float
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class BijliCastService:
    """Loads the deployment bundle once and serves predictions."""

    def __init__(self, root: str | Path, history_tail: int = 72) -> None:
        self.root = Path(root).resolve()
        self.models_dir = self.root / "models"
        self.history_tail = history_tail

        missing = [n for n in REQUIRED_ARTIFACTS if not (self.models_dir / n).exists()]
        if missing:
            raise ArtifactsMissingError(
                "Missing deployment artifacts in "
                f"{self.models_dir}: {', '.join(missing)}. Run "
                "`python -m src.models.export_deployment` and "
                "`python -m src.deploy.export_app_data` first."
            )

        self.model = joblib.load(self.models_dir / "deployment_model.joblib")
        self.profiles = pd.read_csv(self.models_dir / "house_profiles.csv")
        self.profiles[HOUSE_COL] = self.profiles[HOUSE_COL].astype(str)
        self.profiles = self.profiles.sort_values(HOUSE_COL).reset_index(drop=True)

        history = pd.read_parquet(self.models_dir / "usage_history.parquet")
        history[HOUSE_COL] = history[HOUSE_COL].astype(str)
        history[TIME_COL] = pd.to_datetime(history[TIME_COL])
        self.history = history.sort_values([HOUSE_COL, TIME_COL])

        latest = pd.read_csv(self.models_dir / "latest_weather.csv")
        latest[HOUSE_COL] = latest[HOUSE_COL].astype(str)
        latest[TIME_COL] = pd.to_datetime(latest[TIME_COL])
        latest[CITY_COL] = latest[CITY_COL].astype(str)
        self.latest_weather = latest.sort_values(HOUSE_COL).reset_index(drop=True)

        self.metadata = json.loads(
            (self.models_dir / "model_metadata.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (self.models_dir / "app_manifest.json").read_text(encoding="utf-8")
        )

        self._profile_by_house = {
            str(row[HOUSE_COL]): row
            for _, row in self.profiles.iterrows()
        }
        self._latest_by_house = {
            str(row[HOUSE_COL]): row
            for _, row in self.latest_weather.iterrows()
        }
        self._history_by_house = {
            str(house): g.set_index(TIME_COL)["usage_kw"].sort_index()
            for house, g in self.history.groupby(HOUSE_COL, sort=False)
        }

        deployment = self.metadata.get("deployment_model", {})
        self.model_name = str(deployment.get("name", type(self.model).__name__))
        self.interval_offsets = deployment.get("prediction_interval", {})
        self.deployment_metrics = deployment.get("metrics", {})

    # ------------------------------------------------------------------ meta
    def model_info(self) -> dict:
        test = self.deployment_metrics.get("test", {})
        interval = self.interval_offsets
        return {
            "name": self.model_name,
            "n_train_rows": int(
                self.metadata.get("deployment_model", {}).get(
                    "n_train_rows", self.metadata.get("n_train_rows", 0)
                )
            ),
            "test_mae": _finite(test.get("mae", float("nan"))),
            "test_rmse": _finite(test.get("rmse", float("nan"))),
            "test_r2": _finite(test.get("r2", float("nan"))),
            "interval_val_coverage": _finite(
                interval.get("val_coverage", float("nan"))
            ),
            "interval_test_coverage": _finite(
                interval.get("test_coverage", float("nan"))
            ),
            "fit_data": self.metadata.get("deployment_model", {}).get("fit_data"),
        }

    def _house_payload(self, house_id: str) -> dict:
        latest = self._latest_by_house[house_id]
        profile = self._profile_by_house[house_id]
        origin = pd.Timestamp(latest[TIME_COL])
        return {
            "house_id": house_id,
            "city": str(latest[CITY_COL]),
            "origin_time": origin.isoformat(),
            "target_time": (origin + pd.Timedelta(hours=1)).isoformat(),
            "defaults": {
                key: float(latest[key]) for key in INPUT_WEATHER_FEATURES
            },
            "profile": _profile_payload(profile),
        }

    def house_ids(self) -> list[str]:
        return list(self._latest_by_house)

    def houses(self) -> list[dict]:
        return [self._house_payload(h) for h in self.house_ids()]

    def houses_response(self) -> dict:
        return {"model": self.model_info(), "houses": self.houses()}

    # ------------------------------------------------------------- prediction
    def _weather(self, house_id: str, overrides: dict | None) -> dict:
        row = self._latest_by_house[house_id]
        values = {key: float(row[key]) for key in INPUT_WEATHER_FEATURES}
        if overrides:
            for key in INPUT_WEATHER_FEATURES:
                value = overrides.get(key)
                if value is None:
                    continue
                number = float(value)
                if not np.isfinite(number):
                    raise BijliCastError(f"weather value {key!r} is not finite")
                values[key] = number
        return values

    def predict(
        self,
        house_id: str,
        weather: dict | None = None,
        history_tail: int | None = None,
    ) -> PredictionResult:
        house_id = str(house_id)
        if house_id not in self._latest_by_house:
            raise UnknownHouseError(f"unknown household: {house_id!r}")

        latest = self._latest_by_house[house_id]
        profile = self._profile_by_house[house_id]
        origin = pd.Timestamp(latest[TIME_COL])
        city = str(latest[CITY_COL])
        hist = self._history_by_house[house_id]

        static = {col: profile[col] for col in STATIC_NUMERIC_FEATURES}
        for col in STATIC_CATEGORICAL_FEATURES:
            static[col] = profile[col]

        frame = build_inference_row(
            origin_time=origin,
            city=city,
            weather=self._weather(house_id, weather),
            usage_history=hist,
            static=static,
            house_id=house_id,
        )
        raw = float(self.model.predict(frame)[0])
        if not np.isfinite(raw):
            raise BijliCastError("the model returned a non-finite prediction")

        low_offset = float(self.interval_offsets.get("low_offset", -0.5))
        high_offset = float(self.interval_offsets.get("high_offset", 0.5))
        prediction = max(0.0, raw)
        low = max(0.0, raw + low_offset)
        high = max(low, raw + high_offset)

        tail = (history_tail or self.history_tail)
        recent = hist.tail(tail)
        history = [
            {
                "time": pd.Timestamp(stamp).isoformat(),
                "usage_kwh": float(value),
            }
            for stamp, value in recent.items()
        ]

        return PredictionResult(
            house_id=house_id,
            city=city,
            origin_time=origin.isoformat(),
            target_time=(origin + pd.Timedelta(hours=1)).isoformat(),
            prediction_kwh=prediction,
            interval_low_kwh=low,
            interval_high_kwh=high,
            last_hour_kwh=float(hist.iloc[-1]),
            delta_kwh=prediction - float(hist.iloc[-1]),
            history=history,
        )


@lru_cache(maxsize=4)
def get_service(root: str | Path) -> BijliCastService:
    """Process-wide cached loader (Streamlit and FastAPI each call once)."""
    return BijliCastService(root)
