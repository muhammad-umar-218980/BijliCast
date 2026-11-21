---
title: BijliCast
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
python_version: "3.12"
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
usage in one chart.

## Model

| | |
|---|---|
| Model | HistGradientBoostingRegressor |
| Fit data | train + validation (test never used for fitting) |
| Test MAE | 0.250 kWh |
| Test R² | 0.729 |
| Interval | 90%, calibrated on validation residuals |

The compact artifact is ~1.4 MB. The validation-winning Random Forest is kept
locally for report/metric alignment; the served model is the statistically
tied HistGradientBoosting, which is small enough for GitHub and Spaces.

## Other deployments

- Streamlit Community Cloud: <https://bijlicast.streamlit.app/>
- FastAPI + HTML: `uvicorn app.api:app` (localhost demo)

## Limitations

- One global model for all 59 households, with city/static features.
- `House#32` has 15 of 18 static fields missing; its rows are median-imputed.
- The test split covers 48 of 59 households because households end on
  different dates.
- Research/demo purposes only, not billing or grid operations.
