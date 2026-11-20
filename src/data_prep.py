"""
BijliCast — data preparation for EDA.

Builds one hourly combined dataset from:
  - 59 LUMS REWD-P household electricity CSVs (minute-level -> hourly mean)
  - 6 Open-Meteo ERA5 city weather CSVs (hourly)

Output: data/processed/hourly_combined.parquet

Run from the repo root:
    python src/data_prep.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "rewdp_dataset"
WEATHER_DIR = ROOT / "datasets" / "weather_dataset"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["Lahore", "Karachi", "Islamabad", "Multan", "Peshawar", "Skardu"]

# Weather columns we keep (rename to snake_case).
# Dropped: "Solar Energy" (exact duplicate of Solar Radiation), "UV Index" (100% null).
WEATHER_RENAME = {
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Dew": "dew",
    "Precipitation": "precipitation",
    "Wind Speed": "wind_speed",
    "Wind Direction": "wind_direction",
    "Pressure": "pressure",
    "Solar Radiation": "solar_radiation",
}


def load_weather(city: str) -> pd.DataFrame:
    path = WEATHER_DIR / f"{city}.csv"
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
    keep = ["datetime"] + [c for c in WEATHER_RENAME if c in df.columns]
    df = df[keep].rename(columns=WEATHER_RENAME)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def find_usage_column(df: pd.DataFrame) -> str:
    """Return the main total-usage column (contains 'usage', not an appliance name)."""
    cands = [c for c in df.columns if "usage" in str(c).lower()]
    if not cands:
        raise ValueError("No usage column found")
    # Prefer the shortest/most generic name (e.g. 'Usage (kW)' over 'Usage..kW.')
    cands.sort(key=lambda c: (len(str(c)), str(c)))
    return cands[0]


def load_house(path: Path, city: str, house_id: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    header.columns = [str(c).strip() for c in header.columns]
    ucol = find_usage_column(header)
    dtcol = "datetime" if "datetime" in header.columns else header.columns[0]

    df = pd.read_csv(path, usecols=[dtcol, ucol])
    df.columns = ["datetime", "usage_kw"]
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", errors="coerce")
    df["usage_kw"] = pd.to_numeric(df["usage_kw"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Minute-level -> hourly mean
    df = df.set_index("datetime").sort_index()
    hourly = df["usage_kw"].resample("h").mean().reset_index()

    hourly["house_id"] = house_id
    hourly["city"] = city
    return hourly


def house_id_from_filename(fname: str) -> str:
    """lahore_House1.csv -> House#1 ; House21.csv -> House#21 ; peshawar_House21.csv -> House#21."""
    m = re.search(r"[Hh]ouse_?(\d+)", fname)
    if not m:
        raise ValueError(f"Cannot parse house number from {fname}")
    return f"House#{int(m.group(1))}"


def main() -> None:
    frames: list[pd.DataFrame] = []
    for city in CITIES:
        cdir = RAW / city
        if not cdir.is_dir():
            print(f"[skip] missing city dir: {cdir}")
            continue
        weather = load_weather(city)
        files = sorted([f for f in os.listdir(cdir) if f.lower().endswith(".csv")])
        print(f"{city}: {len(files)} house files, weather rows={len(weather)}")
        for fname in files:
            hid = house_id_from_filename(fname)
            house = load_house(cdir / fname, city, hid)
            merged = house.merge(weather, on="datetime", how="left")
            frames.append(merged)
            print(f"  {fname:28s} {hid:9s} hourly rows={len(merged):5d} "
                  f"usage_nan={merged['usage_kw'].isna().mean()*100:5.1f}%")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["city", "house_id", "datetime"]).reset_index(drop=True)

    out = OUT_DIR / "hourly_combined.parquet"
    combined.to_parquet(out, index=False)

    print("\n=== COMBINED HOURLY DATASET ===")
    print(f"rows={len(combined):,}  cols={combined.shape[1]}")
    print(f"houses={combined['house_id'].nunique()}  cities={combined['city'].nunique()}")
    print(f"date range: {combined['datetime'].min()} -> {combined['datetime'].max()}")
    print(f"usage_kw NaN: {combined['usage_kw'].isna().mean()*100:.2f}%")
    print("\nMissing % per column:")
    print((combined.isna().mean() * 100).round(2).to_string())
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
