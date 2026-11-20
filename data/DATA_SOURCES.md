# DATA SOURCES — BijliCast Project

This document describes the provenance of all datasets used in the BijliCast project,
as required by CT-378 Machine Learning Guidelines.

---

## Source 1: LUMS REWD-P (Residential Energy and Weather Data — Pakistan)

| Attribute | Description |
|---|---|
| **Provider** | LUMS Energy Institute (LEI), Lahore, Pakistan |
| **Data type** | Household electricity consumption + city-level weather |
| **Collection method** | Downloaded CSV archive from LEI website; electricity data from smart-meter readings (1-minute granularity), weather data from on-site sensors |
| **Licence / Terms** | LUMS REWD-P is described as open-source; however, precise licence terms must be verified at <https://lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html>. Redistribution of original measurements may be restricted; this project uses the data for academic ML research only. |
| **Geographic coverage** | Six Pakistani cities: Karachi, Lahore, Multan, Peshawar, Islamabad, Skardu |
| **Number of households** | 59 unique households (60 CSVs; Peshawar has 10 duplicate file pairs) |
| **Electricity data date range** | 2023-11-01 → 2024-10-31 (most houses); some houses extend to 2024-11-19 (Karachi House10) or 2024-11-28 (Skardu House51) |
| **Electricity measurement** | Whole-home power in kW, recorded at 1-minute interval; aggregated to hourly mean kWh for analysis |
| **Weather variables (6 CSVs)** | Temperature (°C), Humidity (%), Dew Point (°C), Precipitation (mm), Wind Speed (km/h), Wind Direction (°), Pressure (hPa), Solar Radiation (W/m²), Solar Energy (kWh/m²), UV Index |
| **Weather data date range** | 2023-07-01 → 2024-11-30 (12,456 hourly rows per city) |
| **Collection date** | 2025 (dataset published; actual measurement period 2023-2024) |
| **File locations** | `datasets/rewdp_dataset/<city>/<house>.csv` (electricity)<br>`datasets/weather_dataset/<city>.csv` (weather) |
| **Record count** | • Electricity: ~502,000 unique house-hour observations (59 houses)<br>• Weather: 8,784 hourly rows × 6 cities = 52,704 total observations |

**Source citation:**  
LUMS Energy Institute. (2025). *Residential Energy and Weather Data Pakistan (REWD-P)*.
Retrieved from <https://web.lums.edu.pk/~eig/gallery/datasets/rewdp/>.

---

## Source 2: Open-Meteo ERA5 Historical Weather Reanalysis

| Attribute | Description |
|---|---|
| **Provider** | Open-Meteo (using ERA5 reanalysis data from Copernicus Climate Data Store) |
| **Data type** | Hourly meteorological parameters (open-access historical weather) |
| **Collection method** | Automated retrieval via Open-Meteo archive API:<br>`https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={YYYY-MM-DD}&end_date={YYYY-MM-DD}&hourly={variables}&timezone=Asia/Karachi&models=era5`<br>Variables fetched: temperature_2m, relative_humidity_2m, dew_point_2m, precipitation, wind_speed_10m, wind_direction_10m, pressure_msl, shortwave_radiation |
| **Licence / Terms** | Open-Meteo data is free for academic and research use; see <https://open-meteo.com/en/docs/> for full terms. Attribution required: "Contains Copernicus Climate Data Store ERA5 data." |
| **Geographic coverage** | Global; data for the six Pakistani cities at their respective coordinates |
| **Date range** | 2023-11-01 → 2024-10-31 (aligned with LUMS electricity period) |
| **Spatial resolution** | Hourly (1-hour aggregated ERA5 reanalysis) |
| **Fetched variables** | • Temperature (2m, °C)<br>• Humidity (2m, %)<br>• Dew point (2m, °C)<br>• Precipitation (mm)<br>• Wind speed (10m, km/h)<br>• Wind direction (10m, degrees)<br>• Pressure (MSL, hPa)<br>• Solar radiation (shortwave, W/m²) |
| **Collection date** | 2025 (API accessed [date of retrieval]) |
| **File locations** | `datasets/weather_dataset/<city>.csv` (6 CSV files, one per city)<br>Each CSV: 8,784 rows × 11 columns (datetime + 10 meteorological variables) |
| **Record count** | 8,784 hours × 6 cities = 52,704 total house‑independent observations |

**Source citation:**  
Open-Meteo. (2025). *Archive API – Historical Weather Data (ERA5)*.
Retrieved from <https://open-meteo.com/en/docs/historical-weather-api>.

---

## How the Two Sources Were Combined

1. **Electricity data** (`datasets/rewdp_dataset/`) was parsed, timestamps normalized, and aggregated to hourly mean kW.
2. **LUMS weather data** (`datasets/weather_dataset/`) was the initial weather source for EDA and baseline modelling.
3. **Open-Meteo ERA5 data** (`datasets/weather_dataset/` — overwritten/replaced) was fetched independently using the Open-Meteo API (no API key required; free and open-access).
4. **Merge**: Both datasets were aligned on `(datetime, city)` using `datetime.floor('h')` as the key. Invalid timestamps were handled with explicit format parsing (`%Y-%m-%d %H:%M:%S` and `%Y-%m-%dT%H:%M`). Missing hours were marked and excluded from model training to prevent data leakage.
5. **Resulting merged dataset**: ~500,000+ house-hour rows with 15+ features (temporal lags, rolling means, weather variables, calendar features), well above the CT-378 minimum of 5,000 rows and 15 features.

---

## Compliance with CT-378 Guidelines

| Requirement | Status | Note |
|---|---|---|
| Combine at least two public sources | ✅ | LUMS REWD-P (provider: LUMS) + Open-Meteo ERA5 (provider: Open-Meteo/NASA Copernicus) are independent sources with different collection methods. |
| Minimum 5,000 rows | ✅ | ~502,000 house-hour rows after merging. |
| Minimum 15 features | ✅ | Engineered features include: hour, day_of_week, month, lag_1/24, rolling_mean_24, temperature, humidity, dew, precipitation, wind_speed, wind_direction_sin/cos, pressure, solar_radiation, UV_index, plus metadata (AC count, occupants, etc.). |
| Document source, method, licence, date | ✅ | This `DATA_SOURCES.md` file provides full provenance for both datasets. |
| Remove personal identifiers | ✅ | House IDs only (e.g., `House#1`, `Karachi_House31`); no names, addresses, or sensitive household data. |

---

## Important Notes

- **LUMS data redistribution**: The original LUMS electricity CSVs are ~2.3 GB total. This repository does **not** commit raw LUMS data to GitHub (see `.gitignore`: `data/raw/`). Scripts and provenance are provided so the data can be re-downloaded from the LEI website.
- **Open-Meteo terms**: The ERA5 reanalysis data is provided by Copernicus Climate Data Store via Open-Meteo. See their [terms of use](https://open-meteo.com/en/docs/) for attribution and redistribution rules.
- **Timestamp handling**: Both datasets use Pakistan Standard Time (Asia/Karachi, UTC+5). Extra care was taken to normalize datetime formats (date-only vs. T-separator vs. space-separator) before merging.
- **Second source necessity**: The CT-378 guidelines require "combine at least two public sources." Using only the two LUMS CSVs (electricity + weather) would **not** satisfy this requirement because both are from the same provider. Open-Meteo provides the mandatory independent second source.

---

**Generated:** 2025  
**Project:** BijliCast — Weather-Aware Household Electricity Consumption Forecasting in Pakistan  
**Course:** CT-378 Machine Learning, Sessional Project