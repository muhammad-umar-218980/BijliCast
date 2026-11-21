"""FastAPI endpoints and Pydantic validation."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app


def client() -> TestClient:
    return TestClient(app)


def test_health():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_houses():
    response = client().get("/houses")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["houses"]) == 59
    assert payload["model"]["name"] == "HistGradientBoosting"


def test_predict_with_weather_override():
    response = client().post(
        "/predict",
        json={"house_id": "House#1", "weather": {"temperature": 30, "humidity": 55}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["house_id"] == "House#1"
    assert payload["interval_low_kwh"] <= payload["prediction_kwh"] <= payload["interval_high_kwh"]
    assert len(payload["history"]) == 72


def test_predict_without_weather_uses_defaults():
    response = client().post("/predict", json={"house_id": "House#1"})
    assert response.status_code == 200


def test_unknown_house_returns_404():
    response = client().post("/predict", json={"house_id": "House#999"})
    assert response.status_code == 404


def test_invalid_weather_returns_422():
    response = client().post(
        "/predict",
        json={"house_id": "House#1", "weather": {"temperature": 999}},
    )
    assert response.status_code == 422
