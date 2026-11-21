"""BijliCast FastAPI backend + single-page HTML front end.

Run locally:
    uvicorn app.api:app --reload --port 8000

Then open:
    http://127.0.0.1:8000/          UI
    http://127.0.0.1:8000/docs      OpenAPI docs

Endpoints
---------
GET  /health    liveness + loaded model
GET  /houses    households, weather defaults and model metrics
POST /predict   next-hour kWh with a 90% prediction interval
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for _path in (str(ROOT), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from service import (  # noqa: E402
    ArtifactsMissingError,
    BijliCastError,
    BijliCastService,
    UnknownHouseError,
)

INDEX_HTML = APP_DIR / "static" / "index.html"


class WeatherInput(BaseModel):
    """Current-hour weather (the model never uses next-hour weather).

    Every field is optional: omitted values fall back to the household's last
    known hour, so callers can send only the variable they want to change.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    temperature: Annotated[
        float | None, Field(default=None, ge=-60, le=60, description="°C")
    ]
    humidity: Annotated[
        float | None, Field(default=None, ge=0, le=100, description="%")
    ]
    dew: Annotated[
        float | None, Field(default=None, ge=-60, le=60, description="°C")
    ]
    precipitation: Annotated[
        float | None, Field(default=None, ge=0, le=300, description="mm")
    ]
    wind_speed: Annotated[
        float | None, Field(default=None, ge=0, le=200, description="km/h")
    ]
    wind_direction: Annotated[
        float | None, Field(default=None, ge=0, le=360, description="degrees")
    ]
    pressure: Annotated[
        float | None, Field(default=None, ge=500, le=1100, description="hPa")
    ]
    solar_radiation: Annotated[
        float | None, Field(default=None, ge=0, le=1500, description="W/m²")
    ]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    house_id: Annotated[str, Field(min_length=1, max_length=64)]
    weather: WeatherInput | None = None


class ModelInfo(BaseModel):
    name: str
    n_train_rows: int
    test_mae: float | None
    test_rmse: float | None
    test_r2: float | None
    interval_val_coverage: float | None
    interval_test_coverage: float | None
    fit_data: str | None


class HouseInfo(BaseModel):
    house_id: str
    city: str
    origin_time: str
    target_time: str
    defaults: dict[str, float]
    profile: dict[str, Any]


class HousesResponse(BaseModel):
    model: ModelInfo
    houses: list[HouseInfo]


class HistoryPoint(BaseModel):
    time: str
    usage_kwh: float


class PredictionResponse(BaseModel):
    house_id: str
    city: str
    origin_time: str
    target_time: str
    prediction_kwh: float
    interval_low_kwh: float
    interval_high_kwh: float
    last_hour_kwh: float
    delta_kwh: float
    history: list[HistoryPoint]


@lru_cache(maxsize=1)
def get_service() -> BijliCastService:
    return BijliCastService(ROOT)


app = FastAPI(
    title="BijliCast API",
    version="1.0.0",
    description=(
        "Weather-aware next-hour household electricity forecasting for Pakistan. "
        "Inputs: household profile, last 168 h of usage, current-hour weather. "
        "Model: HistGradientBoosting pipeline trained on train+validation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(ArtifactsMissingError)
async def artifacts_missing_handler(_request, exc: ArtifactsMissingError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail="index.html is missing")
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/health", tags=["system"])
def health() -> dict:
    service = get_service()
    return {
        "status": "ok",
        "model": service.model_name,
        "houses": len(service.house_ids()),
    }


@app.get("/houses", response_model=HousesResponse, tags=["inference"])
def houses() -> dict:
    return get_service().houses_response()


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PredictRequest) -> dict:
    service = get_service()
    weather = payload.weather.model_dump() if payload.weather else None
    try:
        result = service.predict(payload.house_id, weather)
    except UnknownHouseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BijliCastError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)
