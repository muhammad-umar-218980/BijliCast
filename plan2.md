# ⚡ BijliCast — Updated Project Plan (v2)

> **Weather-Aware Household Electricity Consumption Forecasting in Pakistan**
> Course: **CT-378** · Task: **Time-series regression** · Target: `next_hour_energy_kwh`
>
> This plan reflects what we have actually built and the CT-378 ML lifecycle.
> `plan1.md` and `plan2.md` are kept in sync.

---

## 0. Status Snapshot

| Phase | Status |
|---|---|
| Phase 1 — Dataset understanding | ✅ Complete |
| Phase 2 — EDA + integration | ✅ Complete |
| Phase 3 — Preprocessing + feature engineering + split | ⏳ Next |
| Phase 4 — Machine learning | ⏳ Pending |
| Phase 5 — Deployment (3 platforms) | ⏳ Pending |
| Phase 6 — Report / viva | ⏳ Pending |

**Already built**
- `src/data_prep.py` — combined hourly dataset from all households + weather.
- `src/data/collect_openmeteo.py` — Open-Meteo ERA5 weather for 6 cities.
- `data/processed/hourly_combined.parquet` — 515,786 rows, 59 houses, 6 cities.
- `data/processed/merged_dataset.parquet` — + 19 static household features.
- `notebooks/01_data_understanding.ipynb` … `05_integrated_eda.ipynb` — executed.
- `reports/figures/` — **15 EDA plots**.
- `docs/FEATURE_SPEC.md`, `data/EDA_PLAN.md`, `data/DATA_SOURCES.md`.

---

## 1. Project at a Glance

| Item | Details |
|---|---|
| **Goal** | Predict a household's electricity for the **next hour** using past usage + weather + household profile |
| **ML task** | Supervised **time-series regression** |
| **Target** | `next_hour_energy_kwh` |
| **Source 1** | **LUMS REWD-P** — 59 unique household electricity CSVs + `metadata_ultimate.xlsx` |
| **Source 2** | **Open-Meteo ERA5** — 6 city weather CSVs (12,480 hourly rows each) |
| **Cities** | Lahore · Karachi · Islamabad · Multan · Peshawar · Skardu |
| **Dataset minimums** | 5,000+ rows · 15+ features · 5+ EDA plots |
| **Metrics** | MAE · RMSE · R² |

---

## 2. Data Sources (updated)

| Role | Source | Notes |
|---|---|---|
| Source 1 | **LUMS REWD-P** electricity | 59 unique houses (10 Peshawar duplicate files removed) |
| Source 1 (static) | `metadata_ultimate.xlsx` | 59 rows × 46 cols of household survey features |
| Source 2 | **Open-Meteo ERA5** | 6 cities, 12,480 hourly rows each, 2023-07-01 → 2024-12-01 |

### Decision: Open-Meteo replaces NASA POWER
- **Change:** old plans named NASA POWER; we use **Open-Meteo ERA5**.
- **Why:** free, no API key, matches the LUMS weather CSV schema, reproducible fetch.
- **Compliance:** still satisfies “combine at least two public sources” (LUMS ≠ Open-Meteo/Copernicus).
- **Attribution:** “Contains Copernicus Climate Data Store ERA5 data.”

### Data scale
- 515,786 hourly house observations (≫ 5,000 minimum).
- Weather: 12,480 × 6 = 74,880 rows; **0% weather join gaps** after extending the window.
- `usage_kw` meter gaps ~2.5% (genuine; handled in Phase 3).

---

## 3. Repository Structure (actual + target)

```text
BijliCast/
├── data/
│   ├── DATA_SOURCES.md              ✅
│   ├── EDA_PLAN.md                  ✅
│   └── processed/
│       ├── hourly_combined.parquet  ✅
│       ├── merged_dataset.parquet   ✅
│       ├── features_dataset.parquet ⏳ Phase 3
│       └── (optional) train/val/test parquet ⏳ Phase 3
├── datasets/                         ✅ raw (keep local; LFS/ignore for public release)
│   ├── rewdp_dataset/<City>/*.csv   ✅ 59 houses
│   ├── weather_dataset/*.csv        ✅ 6 cities (Open-Meteo)
│   └── metadata_ultimate.xlsx       ✅
├── notebooks/
│   ├── 01_data_understanding.ipynb  ✅
│   ├── 02_electricity_eda.ipynb     ✅
│   ├── 03_weather_eda.ipynb         ✅
│   ├── 04_data_integration.ipynb    ✅
│   ├── 05_integrated_eda.ipynb      ✅
│   ├── 06_preprocessing_feature_engineering.ipynb ⏳
│   ├── 07_data_splitting.ipynb      ⏳
│   ├── 08_baseline_model.ipynb      ⏳
│   ├── 09_model_training.ipynb      ⏳
│   ├── 10_hyperparameter_tuning.ipynb ⏳
│   └── 11_final_evaluation.ipynb    ⏳
├── src/
│   ├── data_prep.py                 ✅
│   ├── data/collect_openmeteo.py    ✅
│   ├── features/build_features.py   ⏳
│   ├── preprocessing/pipeline.py    ⏳
│   ├── models/{baseline,train,tune,evaluate}.py ⏳
│   └── utils/config.py              ⏳
├── models/{baseline_model,best_model}.joblib, model_metadata.json ⏳
├── reports/
│   ├── figures/                     ✅ 15 plots
│   ├── tables/                      ⏳
│   └── BijliCast_Report.pdf         ⏳
├── app/{streamlit,huggingface,api}/ ⏳
├── tests/                           ⏳
├── docs/FEATURE_SPEC.md             ✅
├── requirements.txt                 ⏳
└── README.md
```

> Layout note: raw data lives in `datasets/` (not the old `data/raw/lums/...`).
> This is a deliberate, documented change; the pipeline is reproducible via `src/`.

---

## 4. Target & Feature Specification

### Target
`next_hour_energy_kwh` = the household's electricity for the **next hour** = hourly mean
power (kW) shifted by −1 within each house. Averaged over a 1-hour window, the numeric
value equals the energy for that hour in kWh. It is **never** used as an input.

### Feature groups (≥15 required; we use 30+)

| Group | Features | Count |
|---|---|---|
| A. Weather | temperature, humidity, dew, precipitation, wind_speed, wind_direction, pressure, solar_radiation | 8 |
| B. Lags (per house) | lag_1h, lag_2h, lag_3h, lag_24h, lag_168h | 5 |
| C. Rolling (per house) | rolling_mean_3h, rolling_mean_24h, rolling_std_24h | 3 |
| D. Calendar | hour, day_of_week, month, is_weekend | 4 |
| E. Context | city (categorical) | 1 |
| F. Static household (metadata) | people, adults, children, area, covered area, floors, rooms, ACs, coolers, fridges, washing machines, ceiling/wall fans, water pumps, geysers, electric heaters, UPS, build year, owner/rented | ~19 |

Dropped / excluded:
- `UV Index` — 100% null in ERA5.
- `Solar Energy` — duplicate of `Solar Radiation`.
- Appliance sub-meter columns — inconsistent across houses; EDA-only.

---

## 5. ML Lifecycle (CT-378 — all steps required)

Authoritative checklist, mapped to artifacts.

### Step 1 — EDA (distributions, missing values, outliers, correlations, class balance; ≥5 plots)
- ✅ Done in notebooks 01–05; **15 plots** in `reports/figures/`.
- “Class balance” for regression = **target-bin balance** (`02_target_bins.png`).

### Step 2 — Pre-processing (imputation, encoding, scaling, feature engineering) inside a scikit-learn `Pipeline`, fitted on training data only
- `src/features/build_features.py`: lags, rolling, calendar, target.
- `src/preprocessing/pipeline.py`:
  - numeric: `SimpleImputer(median)` → `StandardScaler`
  - categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`
  - `ColumnTransformer` + `Pipeline`
- **No leakage:** fit only on train; refit inside each CV fold. Lags/rolling use past values only.

### Step 3 — Split: train/validation/test (70/15/15), time-ordered for time series
- Global chronological split by `datetime` (earliest 70% / middle 15% / latest 15%).
- `notebooks/07_data_splitting.ipynb`; no random shuffle.

### Step 4 — Modelling: simple baseline + at least three algorithms from the course
- **Baseline:** naive persistence (`ŷ = lag_1h`, variant `lag_24h`).
- **Algorithms (≥3):** Linear Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting.
- Optional: Support Vector Regressor (SVR).
- `notebooks/08_baseline_model.ipynb`, `09_model_training.ipynb`.

### Step 5 — Tuning: 5-fold cross-validation with `GridSearchCV` / `RandomizedSearchCV`
- Use `TimeSeriesSplit(n_splits=5)` (time-aware 5-fold).
- Select on validation/CV, never on test.
- `notebooks/10_hyperparameter_tuning.ipynb`, `src/models/tune.py`.

### Step 6 — Evaluation: metrics reported once on the held-out test set; discuss over/under-fitting and errors
- Metrics: **MAE, RMSE, R²**.
- Report once on test; discuss train-vs-validation gaps and residual errors.
- `notebooks/11_final_evaluation.ipynb`, `src/models/evaluate.py`.
- Comparison table → `reports/tables/model_comparison.csv`.

### Step 7 — Save: final pipeline with `joblib`; pin versions in `requirements.txt`
- `models/best_model.joblib` (fitted preprocessing + model).
- `models/baseline_model.joblib`, `models/model_metadata.json`.
- `requirements.txt` with pinned versions.

---

## 6. Phase 3 — Preprocessing, Feature Engineering & Split (order matters)

**Order:** after EDA → preprocessing/feature engineering → train/val/test split → train model.

Notebooks: `06_preprocessing_feature_engineering.ipynb`, `07_data_splitting.ipynb`.

- [ ] Build `features_dataset.parquet` from `merged_dataset.parquet`:
  - per-house hourly mean `usage_kw`
  - lags 1/2/3/24/168 h
  - rolling mean 3h/24h, rolling std 24h
  - calendar: hour, day_of_week, month, is_weekend
  - weather (8) + static metadata (19) + city
  - target `next_hour_energy_kwh = usage_kw.shift(-1)` per house
- [ ] Handle missing:
  - drop rows with missing target
  - impute features inside pipeline (not before split)
  - decide zero-usage/outage handling (flag option)
- [ ] Verify **5,000+ rows** and **15+ features**.
- [ ] Chronological 70/15/15 split; optional saved split artifacts.
- [ ] Sanity test: no target leakage; lags are past-only.

---

## 7. Phase 4 — Machine Learning

- [ ] `src/utils/config.py`: paths, cities, seeds, feature lists.
- [ ] `src/preprocessing/pipeline.py`: ColumnTransformer + Pipeline.
- [ ] `src/models/baseline.py`: persistence baseline.
- [ ] `src/models/train.py`: LR, KNN, RF, GB.
- [ ] `src/models/tune.py`: `TimeSeriesSplit(5)` + `RandomizedSearchCV`/`GridSearchCV`.
- [ ] `src/models/evaluate.py`: MAE/RMSE/R² + plots.
- [ ] Train/validation selection; single test evaluation.
- [ ] Save `best_model.joblib` + `model_metadata.json`.
- [ ] Write `requirements.txt` (pinned).

---

## 8. Phase 5 — Deployment (all three platforms)

| Platform | Path | Must include |
|---|---|---|
| Streamlit Community Cloud | `app/streamlit/app.py` | city + household selectors, weather inputs, predict button, next-hour kWh, one explanatory chart |
| Hugging Face Spaces | `app/huggingface/` | model card + pipeline + live Space (Gradio/Streamlit) |
| FastAPI + HTML | `app/api/` | `/predict` endpoint, Pydantic validation, HTML/CSS/JS frontend, clear errors |

All apps load `models/best_model.joblib` and derive features inside the saved pipeline.

---

## 9. Phase 6 — Report & Viva

- [ ] 3–4 page PDF: Abstract, Problem/motivation, Data collection, EDA, Methodology, Results (table + plots), Deployment screenshots, Limitations/future work, References, Contribution table.
- [ ] README with setup + all three deployment links.
- [ ] Demo rehearsal + individual viva prep.

---

## 10. CT-378 Compliance Checklist

| Requirement | Status |
|---|---|
| 5,000+ rows | ✅ 515,786 |
| 15+ features | ✅ 30+ after Phase 3 |
| Two public sources | ✅ LUMS + Open-Meteo |
| Document source/method/licence/date | ✅ `DATA_SOURCES.md` |
| Remove PII | ✅ house IDs only |
| Not a banned dataset | ✅ |
| EDA ≥5 plots | ✅ 15 |
| Preprocessing in Pipeline (train-only fit) | ⏳ Phase 3–4 |
| Time-ordered 70/15/15 split | ⏳ Phase 3 |
| Baseline + ≥3 algorithms | ⏳ Phase 4 |
| 5-fold CV tuning | ⏳ Phase 4 |
| MAE/RMSE/R² once on test | ⏳ Phase 4 |
| Save joblib + pinned requirements | ⏳ Phase 4 |
| Three deployments | ⏳ Phase 5 |

---

## 11. Decisions Log

| # | Decision | Rationale |
|---|---|---|
| D1 | Open-Meteo ERA5 replaces NASA POWER | Free, no key, schema match; still two independent sources |
| D2 | 59 unique houses (drop 10 Peshawar duplicates) | Byte-identical duplicates |
| D3 | Minute → hourly mean power; target = next-hour kWh | Defines forecasting unit |
| D4 | Static household metadata included as features | Adds ~19 legitimate predictors |
| D5 | Drop `UV Index` (100% null) and `Solar Energy` (duplicate) | No information |
| D6 | Global chronological 70/15/15 split | Realistic forecasting, no leakage |
| D7 | Models: baseline + LR, KNN, RF, GB | Matches course algorithm list |
| D8 | Build all three deployments | Marking scheme assesses all three |
| D9 | Keep raw data in `datasets/`, reproducible via scripts | Documented layout change |

---

## 12. Tracker

### Phase 1 — Dataset understanding ✅
- [x] Repository + data organized (59 houses, 6 weather, metadata)
- [x] Inspect schemas, timestamps, row counts
- [x] `DATA_SOURCES.md` with provenance

### Phase 2 — EDA ✅
- [x] Electricity EDA (02)
- [x] Weather EDA (03)
- [x] Missing values + outliers
- [x] Data integration (04)
- [x] Integrated EDA (05)
- [x] ≥5 plots (15 saved)

### Phase 3 — Preprocessing + features + split ⏳
- [ ] Feature engineering (lags, rolling, calendar, weather, static)
- [ ] Target `next_hour_energy_kwh`
- [ ] Missing-value strategy
- [ ] Chronological split
- [ ] Verify 5,000+ rows / 15+ features

### Phase 4 — ML ⏳
- [ ] Pipeline (leak-free)
- [ ] Baseline + ≥3 algorithms
- [ ] 5-fold time-aware CV tuning
- [ ] Final test evaluation (MAE/RMSE/R²)
- [ ] Save joblib + `requirements.txt`

### Phase 5 — Deployment ⏳
- [ ] Streamlit
- [ ] Hugging Face Space
- [ ] FastAPI + HTML
- [ ] Tests

### Phase 6 — Submission ⏳
- [ ] PDF report
- [ ] README + links
- [ ] Demo + viva prep

---

## 13. Risks / Open Questions

| Risk | Mitigation |
|---|---|
| NASA POWER vs Open-Meteo in old plan | Documented D1; Open-Meteo satisfies two-source rule |
| Large raw data in git (~2.7 GB) | Keep local / Git LFS / ignore before public push |
| Skardu cold-climate behaviour differs | City feature + per-city analysis |
| Usage zeros (outages/absence) | Flag/handle in Phase 3 |
| Overfitting with static features | Time-ordered split; CV on train only |
| Deployment resource limits | Keep model small (linear/tree), no deep nets required |

---

*End of plan2.md (v2). Kept in sync with plan1.md.*
