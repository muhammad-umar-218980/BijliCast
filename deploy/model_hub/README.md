---
library_name: scikit-learn
tags:
  - scikit-learn
  - regression
  - time-series-forecasting
  - electricity-consumption
  - pakistan
---

# ⚡ BijliCast

## Overview

BijliCast is a weather-aware household electricity
consumption forecasting system developed for Pakistan.

It uses historical household electricity consumption,
weather conditions, calendar information and household
characteristics to predict next-hour electricity usage.

The prediction target is `next_hour_energy_kwh`.

## Model Information

| Property | Value |
|---|---|
| Model | HistGradientBoostingRegressor |
| Framework | Scikit-learn |
| Task | Time-series regression |
| Artifact | deployment_model.joblib |
| Model size | Approximately 1.4 MB |
| Input features | 42 raw features |

The saved artifact includes the fitted preprocessing
pipeline and the trained estimator.

## Dataset

BijliCast combines two independent public data sources:

1. LUMS REWD-P household electricity dataset.
2. Open-Meteo ERA5 historical weather data.

Dataset characteristics:

- 59 households
- Six Pakistani cities
- 515,786 hourly electricity records

The weather information is joined with household
electricity measurements using temporal and geographic
alignment.

## Methodology

The ML lifecycle includes:

- Exploratory Data Analysis
- Missing-value handling
- Feature engineering
- Lag and rolling statistics
- Chronological train/validation/test splitting
- Scikit-learn preprocessing Pipeline
- Baseline comparisons
- Multiple regression algorithms
- Five-fold TimeSeriesSplit cross-validation
- Randomized hyperparameter optimization

The final deployment pipeline was fitted using
training and validation data. The held-out test set
was excluded from model fitting and selection.

## Model Selection

Random Forest achieved a validation MAE of 0.3216 kWh,
while HistGradientBoosting achieved 0.3273 kWh.

Their cross-validation variation indicated practically
comparable performance.

HistGradientBoosting was selected for deployment
because it offered comparable predictive quality
with a substantially smaller serialized artifact.

Random Forest: approximately 270 MB
HistGradientBoosting: approximately 1.4 MB

This made HGB considerably more practical for
GitHub-based and lightweight deployment.

## Evaluation Results

Held-out test results:

| Metric | Result |
|---|---:|
| MAE | 0.2499 kWh |
| RMSE | 0.4273 kWh |
| R² | 0.7291 |

These are final evaluation results, not the
criteria used to select the model.

## Prediction Interval

The application provides a 90% prediction interval
using residual-based offsets estimated from
out-of-sample validation predictions.

## Intended Use

- Academic research
- Household electricity consumption analysis
- Short-term residential load forecasting
- Educational machine learning demonstrations

## Limitations

- The dataset covers 59 households.
- Coverage is limited to six Pakistani cities.
- The demo uses historical replay data rather
  than live smart-meter readings.
- The model is not intended for electricity billing
  or grid operational decisions.

## Repository

Full source code, notebooks, reports and deployment:

https://github.com/muhammad-umar-218980/BijliCast

## Data Sources

- https://rewdp.lums.edu.pk/
- https://open-meteo.com/