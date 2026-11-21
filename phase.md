# BijliCast — Phase Tracker

> Weather-Aware Household Electricity Consumption Forecasting in Pakistan
> Course: **CT-378 Machine Learning** · Task: **time-series regression**
> Target: `next_hour_energy_kwh`

Phases below follow the **CT-378 guideline sections** (problem/data → EDA →
lifecycle steps 2–7 → deployment → report). `plan1.md` and `plan2.md` are the
detailed v2 blueprints; **this file is the short execution tracker**.

Legend: ✅ done · 🔄 in progress · ⏳ pending

---

## Status at a glance

| Phase | Scope (guideline) | Status |
|---|---|---|
| 0 | Project setup, repo, proposal | ✅ done |
| 1 | Problem statement & data collection | ✅ done |
| 2 | EDA — distributions, missing, outliers, correlations, balance, ≥5 plots | ✅ done |
| 3 | Preprocessing & feature engineering — inside a Pipeline, train-only fit | ✅ done |
| 4 | Split — train/validation/test 70/15/15, time-ordered | ✅ done |
| 5 | Modelling — simple baseline + ≥3 course algorithms | ⏳ next |
| 6 | Tuning — 5-fold CV with GridSearchCV / RandomizedSearchCV | ⏳ pending |
| 7 | Evaluation — MAE/RMSE/R² once on test + over/under-fitting discussion | ⏳ pending |
| 8 | Save — joblib pipeline + pinned requirements | 🔄 in progress |
| 9 | Deployment — Streamlit Cloud, Hugging Face Spaces, FastAPI + HTML | ⏳ pending |
| 10 | Report, README, demo & viva | ⏳ pending |

**Overall:** Phases 0–4 complete. Phase 5 is the immediate next task.

---

## Phase 0 — Setup ✅

- [x] GitHub repository `muhammad-umar-218980/BijliCast`
- [x] Folder structure (`data/`, `datasets/`, `docs/`, `notebooks/`, `reports/`, `src/`)
- [x] `requirements.txt` with pinned versions
- [x] CT-378 guidelines stored in repo
- [x] Topic is a regression/time-series problem (not a banned dataset)
- [ ] Proposal approval / no duplicate topic — **external, confirm with instructor**

## Phase 1 — Problem & data ✅

- [x] Clear user and problem: next-hour household electricity consumption
- [x] Source 1: LUMS REWD-P — 59 unique households, minute → hourly
- [x] Source 2: Open-Meteo ERA5 — 6 cities, 12,480 hourly rows each
- [x] Static household profile: `metadata_ultimate.xlsx` (18 numeric + 1 categorical)
- [x] Provenance, method, licence, date: `data/DATA_SOURCES.md`
- [x] No PII (house codes only)
- [x] 515,786 hourly house-rows before feature engineering

## Phase 2 — EDA ✅

- [x] Data understanding (`notebooks/01`)
- [x] Electricity EDA — distributions, outliers, load profiles (`02`)
- [x] Weather EDA — distributions, correlations, autocorrelation (`03`)
- [x] Data integration (`04`)
- [x] Integrated EDA — weather/usage, ACs vs usage, weekday/weekend (`05`)
- [x] **25 figures** in `reports/figures/` (requirement: ≥5)
- [x] Target-bin balance plot (regression analogue of class balance)

## Phase 3 — Preprocessing & feature engineering ✅

- [x] `src/features/build_features.py` → `data/processed/features_dataset.parquet`
- [x] 480,209 rows × 42 model features (requirement: ≥15)
- [x] Lags 1/2/3/24/168 h, trailing rolling mean 3/24 h, rolling std 24 h
- [x] `FeatureEngineer` (calendar + wind sin/cos) **inside** the saved pipeline
- [x] `ColumnTransformer`: median impute → StandardScaler; mode impute → OneHotEncoder
- [x] Pipeline fitted on **training data only**; no NaN after transform
- [x] Inference contract `src/features/inference.py` + `models/house_profiles.csv`
- [x] Train/serve parity self-check passes
- [x] `notebooks/06_preprocessing_feature_engineering.ipynb` executed

## Phase 4 — Split ✅

- [x] Global chronological split, no shuffle
- [x] Row-balanced boundaries: **69.98% / 14.99% / 15.01%**
- [x] Train 2023-07-22 → 2024-06-29 · Val → 2024-08-28 · Test → 2024-11-27
- [x] Boundary purge (102 rows) so no label crosses a split
- [x] `data/processed/splits/*.parquet` + `split_summary.json`
- [x] `notebooks/07_data_splitting.ipynb` executed
- [x] Test set untouched for modelling

## Phase 5 — Modelling ⏳ (next)

- [ ] Naive persistence baseline (`lag_1h`, `lag_24h`)
- [ ] Linear Regression
- [ ] K-Nearest Neighbours
- [ ] Random Forest
- [ ] HistGradientBoosting
- [ ] Train on train, select on validation

## Phase 6 — Tuning ⏳

- [ ] `TimeSeriesSplit(n_splits=5)` on training data only
- [ ] `RandomizedSearchCV` / `GridSearchCV` with the full pipeline
- [ ] Best configuration per algorithm

## Phase 7 — Evaluation ⏳

- [ ] MAE, RMSE, R² reported **once** on test
- [ ] Model comparison table → `reports/tables/model_comparison.csv`
- [ ] Over/under-fitting discussion (train vs validation gaps)
- [ ] Error analysis and prediction intervals
- [ ] Seasonality caveat (validation is summer: mean 1.12 vs train 0.63)

## Phase 8 — Save 🔄

- [x] Pinned `requirements.txt`
- [ ] `models/best_model.joblib` (full pipeline)
- [ ] `models/model_metadata.json` (metrics, feature list, interval width)

## Phase 9 — Deployment ⏳

- [ ] Streamlit Community Cloud (public URL)
- [ ] Hugging Face Space (model card + live app)
- [ ] FastAPI `/predict` + HTML front end
- [ ] Input validation, error messages, prediction interval

## Phase 10 — Report & demo ⏳

- [ ] README: setup, run order, all three deployment links
- [ ] 3–4 page PDF report (abstract, data, EDA, methodology, results, deployment, limitations, references, contributions)
- [ ] 10-minute demo rehearsal + individual viva preparation

---

## ML lifecycle checklist

| # | Guideline step | Status | Evidence |
|---|---|---|---|
| 1 | EDA incl. ≥5 plots | ✅ | notebooks 01–05, 25 figures |
| 2 | Preprocessing in Pipeline, train-only | ✅ | `src/preprocessing/pipeline.py`, notebook 07 |
| 3 | Train/val/test 70/15/15, time-ordered | ✅ | `src/preprocessing/split.py`, `split_summary.json` |
| 4 | Baseline + ≥3 algorithms | ⏳ | Phase 5 |
| 5 | 5-fold CV + GridSearch/RandomizedSearch | ⏳ | Phase 6 |
| 6 | MAE/RMSE/R² once on test + fit discussion | ⏳ | Phase 7 |
| 7 | joblib + pinned requirements | 🔄 | requirements ✅, joblib pending |

## Deliverables checklist (guidelines)

- [x] `notebooks/`
- [x] `requirements.txt`
- [x] Data provenance + download scripts (`src/data/`, `src/data_prep.py`)
- [x] `models/house_profiles.csv`
- [ ] `models/*.joblib`
- [ ] `app/` (three deployments)
- [ ] README with setup + three links
- [ ] Report PDF
- [ ] Demo + viva

## Phase-number mapping

| Old plan1/plan2 wording | This tracker |
|---|---|
| Phase 3 — preprocessing + features + split | Phases 3 + 4 |
| Phase 4 — machine learning | Phases 5 + 6 + 7 + 8 |
| Phase 5 — deployment | Phase 9 |
| Phase 6 — report/viva | Phase 10 |
