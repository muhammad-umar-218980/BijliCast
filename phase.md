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
| 5 | Modelling — simple baseline + ≥3 course algorithms | ✅ done |
| 6 | Tuning — 5-fold CV with GridSearchCV / RandomizedSearchCV | ✅ done |
| 7 | Evaluation — MAE/RMSE/R² once on test + over/under-fitting discussion | ✅ done |
| 8 | Save — joblib pipeline + pinned requirements | ✅ done |
| 9 | Deployment — Streamlit Cloud, Hugging Face Spaces, FastAPI + HTML | 🔄 in progress |
| 10 | Report, README, demo & viva | ⏳ pending |

**Overall:** Phases 0–8 complete. Phase 9: Streamlit, Hugging Face Space bundle and FastAPI + HTML are built and locally browser-verified; only the public URLs need the user's GitHub/HF accounts. Phase 10 (report/demo) remains.

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

## Phase 5 — Modelling ✅

- [x] Naive persistence baseline (`lag_1h`, `lag_24h`) + train mean
- [x] Linear Regression
- [x] K-Nearest Neighbours
- [x] Random Forest
- [x] HistGradientBoosting
- [x] Train on train, select on validation
- [x] `notebooks/08_baseline_model.ipynb` and `09_model_training.ipynb` executed
- [x] Validation MAE: persistence **0.3448**; HistGB 0.3261, RF 0.3307, LR 0.3336, KNN 0.4094
- [x] `reports/tables/baseline_metrics.csv`, `model_comparison_untuned.csv`

## Phase 6 — Tuning ✅

- [x] `TimeSeriesSplit(n_splits=5, gap=120)` on training data only
- [x] `GridSearchCV` (Linear Regression) and `RandomizedSearchCV` (KNN/RF/HistGB) with the full pipeline
- [x] Best configuration per algorithm → `models/tuning_results.json`
- [x] Best CV MAE: RF **0.2168**, HistGB 0.2180, LR 0.2192, KNN 0.2961
- [x] `reports/tables/cv_results_*.csv` + `notebooks/10_hyperparameter_tuning.ipynb` executed

## Phase 7 — Evaluation ✅

- [x] MAE, RMSE, R² reported **once** on test: **MAE 0.2535, RMSE 0.4322, R² 0.7229**
- [x] Model comparison table → `reports/tables/model_comparison.csv`
- [x] Over/under-fitting discussion (train vs validation vs test gaps in notebook 11)
- [x] Error analysis by hour / city / household / month + 90% prediction intervals
- [x] Seasonality caveat (validation is summer: mean 1.12 vs train 0.63; test 0.84)
- [x] Random Forest beats persistence on test (0.2535 vs 0.2760 MAE, 8.2% lower)
- [x] Interval coverage: validation 90.0%, test 93.5%
- [x] `notebooks/11_final_evaluation.ipynb` executed

## Phase 8 — Save ✅

- [x] Pinned `requirements.txt`
- [x] `models/best_model.joblib` (full pipeline, fitted on train)
- [x] `models/deployment_model.joblib` (same config refit on train+val)
- [x] `models/model_metadata.json` (metrics, feature list, interval width, versions)
- [x] Joblib reload reproduces the saved test predictions

## Phase 9 — Deployment 🔄

- [x] Compact deployment artifact `models/deployment_model.joblib` (HistGradientBoosting, 1.4 MB)
- [x] App data bundle: `usage_history.parquet`, `latest_weather.csv`, `app_manifest.json`
- [x] Shared inference service `app/service.py` (one contract for all three deployments)
- [x] Streamlit UI `app/streamlit_ui.py` + entry point `app/streamlit_app.py`
- [x] Streamlit theme `.streamlit/config.toml` (blue, viewer toolbar)
- [x] Streamlit verified in a real browser: 0.526 kWh, interval 0.00–1.37, no console/network errors
- [x] FastAPI `app/api.py`: `GET /`, `/health`, `/houses`, `POST /predict`, `/docs`
- [x] Pydantic weather validation (422), unknown-house handling (404)
- [x] HTML front end `app/static/index.html` (house selector, weather inputs, interval, SVG chart)
- [x] FastAPI verified: 59 houses, prediction 0.585 kWh, 404/422 paths, browser screenshot
- [x] Hugging Face Space bundle `deploy/huggingface/` (~1.6 MB) + model card
- [x] HF Space sync script `python -m src.deploy.export_hf_space` with in-folder self-check
- [x] HF Space bundle verified locally on port 8502 in a browser
- [x] `streamlit`, `fastapi`, `uvicorn`, `pydantic` pinned in `requirements.txt`
- [x] README: local run + all three deployment instructions
- [ ] Streamlit Community Cloud public URL — **needs the user's GitHub push + Streamlit account**
- [ ] Hugging Face Space public URL — **needs the user's HF account/upload**
- [ ] FastAPI public hosting (optional; local/uvicorn is verified)
- [x] Large RF artifact `models/best_model.joblib` git-ignored (270 MB > GitHub limit)

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
| 4 | Baseline + ≥3 algorithms | ✅ | Phase 5, notebooks 08–09 |
| 5 | 5-fold CV + GridSearch/RandomizedSearch | ✅ | Phase 6, `tuning_results.json`, notebook 10 |
| 6 | MAE/RMSE/R² once on test + fit discussion | ✅ | Phase 7, notebook 11, `test_metrics.csv` |
| 7 | joblib + pinned requirements | ✅ | `best_model.joblib`, `deployment_model.joblib`, `model_metadata.json` |

## Deliverables checklist (guidelines)

- [x] `notebooks/`
- [x] `requirements.txt`
- [x] Data provenance + download scripts (`src/data/`, `src/data_prep.py`)
- [x] `models/house_profiles.csv`
- [x] `models/*.joblib`
- [x] `app/` (three deployments)
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
