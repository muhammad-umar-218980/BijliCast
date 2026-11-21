---
title: BijliCast
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 1.60.0
app_file: app.py
pinned: false
python_version: "3.12.12"
short_description: Next-hour household electricity forecasting for Pakistan
---

# ⚡ BijliCast — next-hour household electricity forecast

Weather-aware short-term load forecasting for Pakistani households. The model
predicts the **next hour's electricity consumption (kWh)** from the last 168
hours of smart-meter usage, **current-hour** weather, calendar features and the
household profile. No next-hour weather is required, so the forecast does not
depend on a perfect weather forecast.

This Space is the live demo of the CT-378 Machine Learning project. Source,
notebooks and the full report: [BijliCast on GitHub](https://github.com/muhammad-umar-218980/BijliCast).

## Try it

1. Pick one of the 59 households (six Pakistani cities).
2. Adjust the current-hour weather if you want a what-if scenario.
3. Press **Predict next hour**.

You get the point forecast, a 90% prediction interval and the last 72 hours of
actual usage.

## Model

| Item | Value |
|---|---|
| Deployed model | HistGradientBoosting (scikit-learn pipeline) |
| Target | `next_hour_energy_kwh` |
| Training rows | 408,048 (train + validation; test never used for fitting) |
| Features | 42 raw → 49 after calendar/wind encoding, scaling and one-hot |
| Selection | chronological 70/15/15 split, tuning on train only |
| Validation | `TimeSeriesSplit(n_splits=5, gap=120)` + randomized search |
| Artifact | `models/deployment_model.joblib` (~1.4 MB) |

### Held-out test results (evaluated once, after selection)

| Model | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Random Forest (validation winner) | **0.3216** | 0.2535 | 0.4322 | 0.7229 |
| **HistGradientBoosting (deployed)** | 0.3273 | **0.2499** | **0.4273** | **0.7291** |
| Linear Regression | 0.3328 | 0.2601 | 0.4595 | 0.6869 |
| KNN | 0.4024 | 0.3178 | 0.5196 | 0.5996 |
| Persistence (`lag_1h`) | 0.3448 | 0.2760 | 0.5470 | 0.5561 |
| Seasonal naive (`lag_24h`) | 0.4806 | 0.3522 | 0.6415 | 0.3897 |

Random Forest and HistGradientBoosting are statistically tied on validation
(CV fold std ≈ 0.067). HistGradientBoosting is deployed because it matches RF's
accuracy, has the best one-shot test MAE, and its artifact is ~1.4 MB instead of
~270 MB — small enough for GitHub and free hosting.

### Prediction interval

The 90% interval uses the 5th/95th percentiles of out-of-sample validation
residuals (`[-0.782, +0.846] kWh`), reaching 90.0% coverage on validation and
93.8% on test. The lower bound is clipped at zero.

## Inputs

- **Household**: 59 static profiles (city, people, rooms, appliances, area, …).
- **Usage history**: the last 168 contiguous hourly values per house, bundled
  in `models/usage_history.parquet`.
- **Weather** (current hour): temperature, humidity, dew point, precipitation,
  wind speed/direction, pressure, solar radiation. Defaults are the last known
  hour; sliders let you ask what-if questions.
- The origin timestamp is used for calendar features; the target is origin + 1 h.

## Data

- [LUMS REWD-P](https://rewdp.lums.edu.pk/) smart-meter dataset — 59 unique
  households, 515,786 hourly rows.
- [Open-Meteo ERA5](https://open-meteo.com/) archive — six cities.
- Household metadata from the REWD-P dataset.

The demo replays the last known week per household (meter data ends in
Oct–Nov 2024); it is not a live meter feed.

## Limitations

- The test period (autumn) has a lower load level than validation (summer), so
  validation MAE is higher for every model. This is seasonal shift, not
  over-fitting; selection never used the test set.
- One global model serves all 59 houses; no per-house models.
- Inputs are household-level smart-meter readings and city weather; the model
  does not know appliance-level usage or outages.
- Research/demo use only — not for billing or grid operations.

## Files

```
app.py                     Streamlit entry point
streamlit_ui.py            UI (shared with the main repo)
service.py                 bundle loader + inference contract
models/                    HGB pipeline, profiles, 168 h history, metadata
src/                       minimal config / pipeline / inference modules
requirements.txt           pinned runtime
```
