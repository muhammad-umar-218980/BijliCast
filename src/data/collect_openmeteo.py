"""
BijliCast — Open-Meteo ERA5 weather collector.

Data source 2 for the project: hourly historical weather for the six LUMS cities.
This replaces the plans' NASA POWER step (same purpose: fill/cover weather gaps),
but uses Open-Meteo's ERA5 archive which needs no API key and matches the LUMS
weather CSV schema.

Coverage: 2023-07-01 -> 2024-12-01 (covers ALL household date ranges, incl. Skardu).

Run from the repo root:
    python src/data/collect_openmeteo.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "datasets" / "weather_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2023-07-01"
END_DATE = "2024-12-01"

# city -> (latitude, longitude)
CITIES = {
    "Lahore": (31.5497, 74.3436),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Multan": (30.1575, 71.5249),
    "Peshawar": (34.0151, 71.5805),
    "Skardu": (35.2971, 75.6333),
}

# Open-Meteo hourly variable -> LUMS weather CSV column name
COLUMN_MAP = {
    "time": "datetime",
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "dew_point_2m": "Dew",
    "precipitation": "Precipitation",
    "wind_speed_10m": "Wind Speed",
    "wind_direction_10m": "Wind Direction",
    "surface_pressure": "Pressure",
    "shortwave_radiation": "Solar Radiation",
    "uv_index": "UV Index",
}
HOURLY_VARS = ",".join(k for k in COLUMN_MAP if k != "time")

# Exact output column order (matches the existing LUMS-style weather CSVs)
OUTPUT_COLS = [
    "datetime", "Temperature", "Humidity", "Dew", "Precipitation",
    "Wind Speed", "Wind Direction", "Pressure", "Solar Radiation",
    "Solar Energy", "UV Index",
]


def fetch_city(city: str, lat: float, lon: float, retries: int = 5) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": HOURLY_VARS,
        "timezone": "Asia/Karachi",
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=90)
            r.raise_for_status()
            payload = r.json()
            hourly = payload["hourly"]
            df = pd.DataFrame(hourly).rename(columns=COLUMN_MAP)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = 2 * attempt
            print(f"  [{city}] attempt {attempt}/{retries} failed: {exc} (retry in {wait}s)")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Failed to fetch {city}: {last_err}")

    # Solar Energy in the LUMS files is identical to Solar Radiation.
    df["Solar Energy"] = df["Solar Radiation"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[OUTPUT_COLS].sort_values("datetime").reset_index(drop=True)
    return df


def main() -> None:
    print(f"Fetching Open-Meteo ERA5: {START_DATE} -> {END_DATE}")
    for city, (lat, lon) in CITIES.items():
        print(f"- {city} ({lat}, {lon})")
        df = fetch_city(city, lat, lon)
        out = OUT_DIR / f"{city}.csv"
        df.to_csv(out, index=False, date_format="%Y-%m-%dT%H:%M")
        print(f"  saved {out.name}: {len(df):,} rows  "
              f"({df['datetime'].min()} -> {df['datetime'].max()})")
    print("Done.")


if __name__ == "__main__":
    main()
