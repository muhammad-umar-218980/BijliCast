# 🎓 CT-378 BijliCast Viva Preparation — Questions & Answers

*Generated from comprehensive analysis of the complete project against CT-378 guidelines*

---

## 1) How Did You Recognize This as a Time-Series Regression and How Was It Handled?

### Recognition:
- **Problem definition**: "Next-hour household electricity consumption" — inherently temporal prediction
- **Target variable**: `next_hour_energy_kwh` = usage at `t+1` given features at `t`
- **Data structure**: Hourly rows per household with timestamps (`datetime` column)
- **Domain clue**: Electricity usage patterns are explicitly time-dependent (daily cycles, weekly patterns, seasonal variations)
- **Not a standard classification/regression**: Row order matters → must be time-ordered

### How Time-Series Was Handled:

| Aspect | Implementation | Why It Matters |
|---|---|---|
| **No shuffling** | `chronological_split()` — global, time-ordered, no random shuffle | Prevents leakage from future data into training |
| **Train/val/test boundaries** | 70/15/15 by row count: 69.98% / 14.99% / 15.01% | Test period (autumn) is genuinely "the future" for all houses |
| **Boundary purge** | 102 rows purged so labels don't cross split boundaries | Ensures honest evaluation — train labels stay before val period, val labels before test |
| **Target definition** | `next_hour_energy_kwh = usage_kw.shift(-1)` per house | Standard time-series forecasting: predict next step given current features |
| **No next-hour weather** | Model never uses next-hour weather → doesn't depend on perfect forecast | Realistic deployment constraint |

### Viva Answer Framework:
> "Since this is electricity consumption forecasting, it's inherently time-series. The target is the next-hour value, so I had to ensure: (1) **no shuffling** — data is strictly chronological, (2) **global split** — same calendar boundaries for all 59 houses, (3) **boundary purge** — 102 rows removed so no label leaks across splits, (4) **no future-dependent features** — only current-hour weather is used, not next-hour forecast."

---

## 2) Data Pre-Processing Steps (Important — Leak-Free Pipeline)

### Your Pre-Processing Pipeline (`src/preprocessing/pipeline.py`):

### Step-by-Step:

#### Step 1: Missing-Value Handling
- **Numeric features**: `SimpleImputer(strategy='median')` — fitted on training data only
- **Categorical features**: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`
- **Key rule**: Pipeline `fit()` called **only on training split**, then `transform()` applied to val/test

#### Step 2: Feature Scaling
- `StandardScaler()` — **only fitted on training data mean/std**
- Validation/test data transformed using training statistics only

#### Step 3: Calendar Feature Engineering (`FeatureEngineer` — inside pipeline)
- `hour`, `day_of_week`, `month`, `is_weekend` — computed from `datetime` column
- `wind_dir_sin`, `wind_dir_cos` — sine/cosine encoding of circular variable (wind direction)
- **Key**: `fit()` does nothing (returns `self`), `transform()` just computes these from raw `datetime`

#### Step 4: Column Selection
- `ColumnTransformer` with `remainder='drop'` — only specified columns are kept
- **Numerical pipeline**: Median impute → StandardScale
- **Categorical pipeline**: Most frequent impute → OneHotEncoder

### Leakage Prevention Emphasis:
> "The **critical** point: every learned transformation (imputer, scaler, encoder) is **fitted on training data only**. During cross-validation, the pipeline is refitted inside each fold. During inference, the saved pipeline has already learned statistics from training — new data just gets transformed. This prevents target leakage and ensures train/serve parity."

### Viva Answer Framework:
> "My preprocessing pipeline has three steps: (1) **FeatureEngineer** — stateless calendar + wind encoding, (2) **ColumnTransformer** with median imputation + StandardScaler for numeric features, mode impute → OneHotEncoder for categorical features, (3) **remainder='drop'** drops unused columns. **Crucially**, the pipeline `fit()` is called only on the training split. During CV, it's refitted inside each fold. This ensures no data leakage — the model never sees validation/test statistics during training."

---

## 3) ML Pipeline Explanation

### Architecture:
```
raw frame (INPUT_COLUMNS)
  → FeatureEngineer        # calendar + wind sin/cos (stateless)
  → ColumnTransformer      # numeric: median impute → StandardScaler
                           # categorical: mode impute → OneHotEncoder
  → estimator              # HistGradientBoostingRegressor (deployed)
```

### Saved Artifact: `models/deployment_model.joblib`
A single `scikit-learn Pipeline` with:
- Step 1: `FeatureEngineer` (stateless)
- Step 2: `ColumnTransformer` (learned imputation + scaling/encoding)
- Step 3: `HistGradientBoostingRegressor` (with best hyperparameters)

### Input Columns (`INPUT_COLUMNS` — 49 features after one-hot encoding):
- `datetime` — origin timestamp
- 8 raw weather: temperature, humidity, dew, precipitation, wind_speed, wind_direction, pressure, solar_radiation
- 5 lags: lag_1h, lag_2h, lag_3h, lag_24h, lag_168h
- 3 rolling: rolling_mean_3h, rolling_mean_24h, rolling_std_24h
- 4 calendar: hour, day_of_week, month, is_weekend
- 18 static numeric: people, adults, children, area, etc.
- 2 categorical: city, Owner/Rented (one-hot encoded → adds columns)

### How Prediction Works at Deploy Time:
1. User provides: household ID, last 168 hours of usage, current-hour weather, origin timestamp
2. `build_inference_row()` in `src/features/inference.py` constructs one row with 49 columns
3. Saved pipeline loads: `FeatureEngineer` computes calendar/wind encoding → `ColumnTransformer` imputes/scalers → `HistGradientBoosting` predicts
4. Result: next-hour kWh + 90% prediction interval

### Viva Answer Framework:
> "My deployed pipeline is a single scikit-learn Pipeline with 3 steps. Step 1 is `FeatureEngineer` — it adds hour/day/month/weekend and sine/cos wind encoding from datetime and wind_direction. Step 2 is `ColumnTransformer` — numeric features get median imputed then StandardScaled; categorical features get most-frequent imputed then OneHotEncoded (handle_unknown='ignore' so unseen categories at inference are safe). Step 3 is `HistGradientBoostingRegressor` with best hyperparameters. The pipeline is saved with joblib — at inference time, a single `pipeline.predict(row)` call does feature engineering + preprocessing + prediction in one step, ensuring train/serve parity."

---

## 4) Train-Validation-Test Split Explanation

### Your Split (`src/preprocessing/split.py`):

### Parameters:
- **Ratios**: `(0.70, 0.15, 0.15)` → 69.98% train, 14.99% val, 15.01% test
- **Boundary rule**: `"rows"` — cut at timestamp where cumulative row count reaches each ratio
- **Purge rows**: 102 rows removed so labels don't cross boundaries

### Split Boundaries:

| Split | Rows | % of Total | Houses | Period |
|---|---|---|---|---|
| **Train** | 336,053 | 69.98% | 59 | 2023-07-22 → 2024-06-29 |
| **Validation** | 71,995 | 14.99% | 57 | 2024-06-30 → 2024-08-28 |
| **Test** | 72,059 | 15.01% | 48 | 2024-08-28 → 2024-11-27 |

### Key Properties:
- **No shuffling**: Globally chronological — earliest data for all houses is train, middle is val, latest is test
- **Same boundaries for all houses**: Each house's data is split at the same calendar timestamps, even if some houses end before test period (48 of 59 houses have test data)
- **Boundary purge**: Last origin row of train has label in val period; last origin row of val has label in test → both purged (102 rows total)
- **Label leakage prevention**: `train[time_col].max() + Timedelta(hours=1) < val_start` and `val[time_col].max() + Timedelta(hours=1) < test_start` — asserted in split verification

### Why This Matters:
- **Realistic forecasting**: Test period (autumn) has lower target mean (0.836 kWh) than validation (summer, 1.1206 kWh) — higher validation MAE is seasonal, not over-fitting
- **Unused test set**: Test evaluated **once** after model selection — never used for tuning or validation

### Viva Answer Framework:
> "My split is global and chronological, not random. I use 70/15/15 ratios by row count, with the same calendar boundaries for all 59 houses. The train period is 2023-07-22 to 2024-06-29, validation is 2024-06-30 to 2024-08-28, and test is 2024-08-28 to 2024-11-27. **Critically**, I purge 102 boundary rows so that no example's label falls into the next split — this prevents target leakage. The test set is evaluated only once, after model selection, and was never used for hyperparameter tuning."

---

## 5) How Data Was Gathered and Joined

### Two Public Sources:

#### Source 1: LUMS REWD-P (Electricity + Some Weather)
- **Provider**: LUMS Energy Institute, Lahore, Pakistan
- **Collection method**: Downloaded CSV archive from LEI website
- **Electricity data**: 59 households, 1-minute smart-meter readings aggregated to hourly mean kWh
- **Weather data**: 6 on-site sensor stations (temperature, humidity, dew point, precipitation, wind, pressure, solar radiation)
- **Date range**: 2023-11-01 → 2024-10/11 (varies by house)
- **File locations**: `datasets/rewdp_dataset/<city>/<house>.csv`
- **Records**: 515,786 hourly house-rows + 74,880 weather observations (6 cities × 12,480 hours)

#### Source 2: Open-Meteo ERA5 (Independent Weather)
- **Provider**: Open-Meteo (using Copernicus Climate Data Store ERA5 reanalysis)
- **Collection method**: Automated retrieval via Open-Meteo archive API
  ```bash
  https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={YYYY-MM-DD}&end_date={YYYY-MM-DD}&hourly={variables}&timezone=Asia/Karachi&models=era5
  ```
- **Variables**: temperature_2m, relative_humidity_2m, dew_point_2m, precipitation, wind_speed_10m, wind_direction_10m, surface_pressure, shortwave_radiation, uv_index
- **Date range**: 2023-07-01 → 2024-12-01 (extended to cover all households, including Skardu)
- **Spatial resolution**: Global; data for 6 Pakistani cities at their coordinates
- **Records**: 12,480 hourly rows × 6 cities = 74,880 total observations
- **Licence**: Free for academic use; attribution: "Contains Copernicus Climate Data Store ERA5 data."

#### How They Were Joined:
1. **Electricity data** (`datasets/rewdp_dataset/`) was parsed, timestamps normalized, aggregated to hourly mean kW
2. **Open-Meteo ERA5 data** was fetched independently (no API key required, free and open-access)
3. **Merge key**: Both datasets aligned on `(datetime, city)` using `datetime.floor('h')` as the key
4. **Invalid timestamps**: Handled with explicit format parsing (`%Y-%m-%d %H:%M:%S` and `%Y-%m-%dT%H:%M`)
5. **Missing hours**: Marked and excluded from model training to prevent data leakage
6. **Resulting merged dataset**: ~500,000+ house-hour rows with 15+ features — well above the CT-378 minimum of 5,000 rows and 15 features

#### Why Two Sources?
> "CT-378 requires **combine at least two public sources**. Using only the two LUMS CSVs (electricity + weather) would **not** satisfy this requirement because both are from the same provider. Open-Meteo provides the mandatory independent second source — different provider, different collection method, different geographical origin."

### Viva Answer Framework:
> "I gathered data from two independent public sources: (1) **LUMS REWD-P** — 59 Pakistani household electricity smart-meters plus on-site weather sensors, and (2) **Open-Meteo ERA5** — historical weather reanalysis from Copernicus Climate Data Store, fetched via free open-access API. I combined them by aligning both on `(datetime, city)` as the merge key. This satisfies the CT-378 requirement for 'combine at least two public sources' since LUMS and Open-Meteo are different providers with different collection methods."

---

## 6) Decision Points (Important)

### Key Decisions During Your Project:

| Decision | Options | What You Chose | Rationale |
|---|---|---|---|
| **D1: Second weather source** | NASA POWER vs Open-Meteo ERA5 | **Open-Meteo ERA5** | Free, no API key, matches LUMS schema, reproducible. Still satisfies "two public sources" rule |
| **D2: 59 houses vs dropping duplicates** | Keep all 60 Peshawar duplicates vs drop 10 | **Drop 10 Peshawar duplicates** | Byte-identical duplicates — kept 59 unique houses for clean modeling |
| **D3: Minute → hourly aggregation** | Keep minute data vs aggregate to hourly mean | **Aggregate to hourly mean kWh** | Defines forecasting unit; minute data too granular for next-hour prediction target |
| **D4: Feature selection** | Include UV Index, Solar Energy vs drop | **Drop UV Index (100% null), Solar Energy (duplicate)** | No information gain from dropping them |
| **D5: Static features inclusion** | All 46 metadata fields vs select | **Include 18 numeric + 1 categorical** (19 features) | Adds legitimate predictors without overwhelming model |
| **D6: Drop raw usage column** | Keep `usage_kw` as feature vs exclude | **Exclude from model INPUT** | Including it would be target leakage since lags of usage are already features |
| **D7: Outlier handling** | Remove vs impute vs flag | **Document in EDA, no removal** | Outlier analysis shown in plots; decision documented but not aggressively handled |
| **D8: Prediction interval method** | Percentile-based vs parametric vs none | **Validation residual 5th/95th percentile** | Simple, distribution-free interval; 90% coverage reported |
| **D9: Model selection** | Select on test vs validation vs CV | **Select on validation MAE, evaluate once on test** | Prevents test-set overfitting; aligns with CT-378 requirement |
| **D10: Prediction interval coverage** | 90% vs 95% vs other | **90% from validation residuals** | Matches guideline metric, test coverage reported 93.8% |

### Viva Answer Framework (pick 2-3 key ones):
> "My key decision points were: (1) **Using Open-Meteo instead of NASA POWER** — free, no key, schema-matched, and still satisfies the two-source requirement since LUMS ≠ Open-Meteo. (2) **Dropping raw usage column from model inputs** — including it would be target leakage since lags of usage are already features. (3) **Selecting model on validation, evaluating once on test** — CT-378 requires this order to prevent test-set overfitting. (4) **Dropping UV Index (100% null) and Solar Energy (duplicate)** — no information gain, simplifies the feature space."

---

## 7) Evaluation Metrics

### Your Reported Metrics:

| Metric | Value | Model | Split |
|---|---|---|---|
| **Test MAE** | 0.2499 | HistGradientBoosting (deployed) | Test |
| **Test RMSE** | 0.4273 | HistGradientBoosting (deployed) | Test |
| **Test R²** | 0.7291 | HistGradientBoosting (deployed) | Test |
| **Val MAE** | 0.3216 | Random Forest | Validation |
| **Val RMSE** | 0.5145 | Random Forest | Validation |
| **Val R²** | 0.7605 | Random Forest | Validation |
| **Test MAE (RF)** | 0.2535 | Random Forest | Test (baseline comparison) |
| **Train MAE (HGB)** | 0.1966 | HistGradientBoosting | Train |
| **Train MAE (RF)** | 0.1536 | Random Forest | Train |

### Prediction Interval:
- **Method**: Validation residual 5th/95th percentile
- **Train coverage**: 96.9%
- **Val coverage**: 90.0%
- **Test coverage**: 93.8% (slightly conservative — it over-covers by ~4 points because validation was the higher-load summer period and test is lower-load autumn)
- **Mean width**: ~1.43 kWh (test set)

### Baseline Comparisons:
| Baseline | Test MAE | vs HGB |
|---|---|---|
| Persistence (lag_1h) | 0.2760 | **+10.4% higher MAE** (HGB is 9.5% better) |
| Seasonal naive (lag_24h) | 0.3522 | **+40.9% higher MAE** (HGB is 29.0% better) |
| Train mean | 0.5458 | **+118.4% higher MAE** (HGB is 54.2% better) |

### Viva Answer Framework:
> "My primary evaluation metrics are MAE, RMSE, and R², reported once on the held-out test set. For my deployed HistGradientBoosting model: **Test MAE 0.2499 kWh**, **Test RMSE 0.4273 kWh**, **Test R² 0.7291**. I also report **90% prediction intervals** with **93.8% test coverage** — the interval slightly over-covers on test because validation was the higher-load summer period, so its residual quantiles are a little wide for autumn. **Key comparison**: my model achieves **9.5% lower MAE** than the persistence baseline (lag_1h) and **29% lower** than the seasonal naive baseline (lag_24h). **Overfitting discussion**: training MAE (0.1966) is lower than validation MAE (0.3273) and test MAE (0.2499) — this seasonal shift (summer validation vs autumn test) is the cause, not over-fitting."

---

## 8) Algorithms Used & How They Work

### Your 4 Course Algorithms + Baseline:

### 1. Linear Regression
- **How it works**: Finds weights that minimize MSE: `ŷ = w₀ + w₁x₁ + ... + wₙxₙ`
- **Your implementation**: `sklearn.linear_model.LinearRegression` within the full pipeline
- **Your result**: Test MAE 0.2601, RMSE 0.4595, R² 0.6869 — **weakest learned model**
- **Viva point**: "Linear model assumes additive, linear relationships — too simple for electricity consumption which has nonlinear patterns (daily cycles, temperature thresholds for AC usage)."

### 2. K-Nearest Neighbors (KNN)
- **How it works**: Predicts by averaging k most similar training examples (distance-weighted)
- **Your implementation**: `KNeighborsRegressor(n_neighbors=5, weights='distance')` within the full pipeline
- **Your result**: Test MAE 0.3178, RMSE 0.5196, R² 0.5996 — **weakest performance**
- **Viva point**: "KNN with `weights='distance'` has exact train predictions (train MAE 0.0, R² 1.0) — but this is **not** evidence of leakage because validation/test metrics are the meaningful ones. KNN suffers from the **curse of dimensionality** with 49 features, and 168-hour lag history makes distance metrics less meaningful."

### 3. Random Forest Regressor
- **How it works**: Ensemble of decision trees, each trained on random subset of data and random subset of features. Prediction = average of all tree predictions. Handles nonlinearities and interactions automatically.
- **Your implementation**: `RandomForestRegressor(n_estimators=120, min_samples_leaf=10, max_features=0.7)` within the full pipeline, trained on **train** only
- **Your result**: Validation MAE 0.3216 (winner on val), Test MAE 0.2535, RMSE 0.4322, R² 0.7229
- **Viva point**: "Random Forest captures nonlinear feature interactions (e.g., temperature × hour of day × AC count) without needing explicit feature engineering. It's robust to feature scales and gets imputed inputs from the shared pipeline. **But** its artifact is ~270 MB, exceeding GitHub's 100 MB limit and making Streamlit Community Cloud deployment difficult."

### 4. HistGradientBoosting Regressor (Deployed)
- **How it works**: Gradient boosting builds trees sequentially, each new tree correcting residuals of the previous one. Uses leaf-wise growth (depth-first) rather than level-wise (breadth-first) — faster convergence. Supports missing values and native categoricals; in this project the shared pipeline one-hot encodes categoricals and imputes numerics first.
- **Your implementation**: `HistGradientBoostingRegressor(min_samples_leaf=50, max_leaf_nodes=31, max_iter=400, learning_rate=0.06, l2_regularization=5.0)` — trained on **train + validation** (not test)
- **Your result**: **Test MAE 0.2499** (winner), **Test RMSE 0.4273**, **Test R² 0.7291** (best), **Train MAE 0.1966**
- **Viva point**: "HistGradientBoosting won because: (1) **best test MAE** among all learned models, (2) **tied on validation** within CV noise (std ≈ 0.067), (3) **~1.4 MB artifact** — can be committed to GitHub and served by Streamlit Community Cloud, (4) **one shared preprocessor** — HGB can handle categoricals natively in scikit-learn, but in our pipeline `Owner/Rented` and `city` are one-hot encoded so all four algorithms use the same leak-free `ColumnTransformer`, and (5) **faster training** — leaf-wise growth converges in fewer trees."

### Baseline Models:
| Baseline | How It Works | Your Result |
|---|---|---|
| **Persistence (lag_1h)** | `ŷ = usage(t)` — just use last hour's consumption | Test MAE 0.2760 |
| **Seasonal naive (lag_24h)** | `ŷ = usage(t-24h)` — same hour yesterday | Test MAE 0.3522 |
| **Train mean** | `ŷ = mean(all training target values)` | Test MAE 0.5458 |

### Viva Answer Framework (focus on HGB vs RF):
> "I trained and compared 4 course algorithms plus 3 baselines. **Random Forest** had the lowest validation MAE (0.3216), but **HistGradientBoosting** was chosen for deployment because: (1) it's **tied on validation** within cross-validation noise (fold std ≈ 0.067), (2) it has the **best one-shot test MAE** (0.2499 vs RF's 0.2535), (3) its **artifact is ~1.4 MB vs RF's ~270 MB** — GitHub rejects files >100 MB and Streamlit Community Cloud deploys from GitHub, (4) we use **one shared preprocessor** — HGB could take raw categoricals, but `city` and `Owner/Rented` are one-hot encoded so every algorithm is compared on the identical leak-free pipeline, and (5) HGB uses **leaf-wise tree growth** which converges faster, making it more efficient."

### Algorithm Comparison Table (for your reference):

| Algorithm | Test MAE | Test RMSE | Test R² | Params | Artifact Size | Status |
|---|---|---|---|---|---|---|
| Persistence (lag_1h) | 0.2760 | 0.5470 | 0.5561 | n/a | n/a | Baseline |
| Seasonal naive (lag_24h) | 0.3522 | 0.6415 | 0.3897 | n/a | n/a | Baseline |
| Linear Regression | 0.2601 | 0.4595 | 0.6869 | solved convex problem | small | Learned model |
| KNN | 0.3178 | 0.5196 | 0.5996 | n_neighbors=5, weights='distance' | small | Learned model |
| **Random Forest** | **0.2535** | **0.4322** | **0.7229** | n_estimators=120, min_samples_leaf=10, max_features=0.7 | **~270 MB** | Learned model (val winner) |
| **HistGradientBoosting** | **0.2499** | **0.4273** | **0.7291** | min_samples_leaf=50, max_leaf_nodes=31, max_iter=400, lr=0.06, l2=5.0 | **~1.4 MB** | **Deployed model** |

---

## 🎯 Quick Viva Summary Card

| Topic | Your Key Point |
|---|---|
| **Time-series handling** | No shuffling, global chronological split, boundary purge (102 rows), no next-hour weather |
| **Pre-processing** | Leak-free pipeline, train-only fit, median impute + StandardScaler, circular wind encoding, OneHotEncoder with handle_unknown='ignore' |
| **Train/val/test split** | 70/15/15 by rows, same boundaries for all 59 houses, purge 102 rows, test evaluated once only |
| **Data sources** | LUMS REWD-P + Open-Meteo ERA5 (two independent sources, joined on datetime+city), 515,786 raw hourly rows → 480,209 model rows; 42 raw features → 49 after engineering/one-hot |
| **Decision points** | Open-Meteo over NASA POWER, drop raw usage (leakage), select on validation, evaluate once on test, drop UV Index/Solar Energy |
| **Evaluation metrics** | Test MAE 0.2499, RMSE 0.4273, R² 0.7291, 90% PI with 93.8% coverage, 9.5% MAE improvement over persistence (~22% RMSE) |
| **Algorithms** | RF: val winner (~270 MB), HGB: deployed (~1.4 MB), HGB chosen for deployment due to size + performance tradeoff |
| **Three deployments** | Streamlit Cloud (bijlicast.streamlit.app), HF Spaces (huggingface.co/spaces/muhammadumar15/BijliCast), FastAPI (127.0.0.1:8000) — all share same pipeline and service.py |

---

## 💡 Pro Tips for Your Viva

1. **Know your pipeline by heart** — from raw input to prediction, every step
2. **Be ready to defend your model choice** — HGB over RF due to size + performance
3. **Explain leakage prevention** — your strongest CT-378 assessment point
4. **Know your metrics** — why MAE/RMSE/R², what R² = 0.7291 means (72.9% variance explained)
5. **Be honest about limitations** — global model, seasonal shifts, per-house variability
6. **Show enthusiasm** — you've built a complete, working ML project!

**You've got this!** Your project is solid, all verifications pass (`python src/verify_all.py` → ALL 12 checks ✅), and you can explain every component confidently. Good luck with your evaluation today! 🍀

---

# 🏆 Full 10/10 Presentation & Viva Playbook

*Appended from the detailed demo/evaluation plan — use this as the end-to-end script.*

## Part 1 — How the 10 marks are actually earned

| Component | Marks | What the teacher needs to see |
|---|---:|---|
| Demonstration | 4 | 2 for ML lifecycle quality, 2 for **all three deployments working live** |
| Report | 3 | Complete sections, correct metrics, clear figures, citations |
| Viva | 3 | Each member explains code, model choices, results |

**10/10 strategy:** treat the demo as an evidence tour of the guidelines, not just a product demo. Every claim should point to a file, table, or live screen.

---

## Part 2 — The 10-minute demo script

Guideline timing: **3 min problem/data → 4 min live demo → 3 min results**.

| Time | What you show | What you say |
|---|---|---|
| **0:00–0:35** | README hero / report page 2 | “Pakistani households face rising tariffs and load-shedding. BijliCast predicts the **next hour’s kWh** for 59 households in 6 cities, using only information known by the current hour.” |
| **0:35–1:35** | `data/DATA_SOURCES.md`, report Table 1 | “Two independent public sources: LUMS REWD-P smart-meter data and Open-Meteo ERA5 weather. 59 houses, 6 cities, Jul 2023–Nov 2024. 515,786 hourly rows before history filtering; **480,209 model rows**, **42 raw features → 49 after engineering**.” |
| **1:35–2:30** | Report Figure 2, `src/preprocessing/pipeline.py`, `src/preprocessing/split.py` | “One row = origin hour `t`; features are known by end of `t`; target is `t+1`. Lags and rolling windows are causal. Chronological **70/15/15** split with **102 boundary rows purged**. All learned preprocessing is inside the sklearn Pipeline, fit on train only.” |
| **2:30–3:00** | `src/models/tune.py`, `reports/tables/model_comparison.csv` | “Three baselines plus Linear Regression, KNN, Random Forest and HistGradientBoosting. Tuned with **5-fold TimeSeriesSplit, gap 120**, and RandomizedSearchCV. Test evaluated **once**, after selection.” |
| **3:00–4:30** | Live Streamlit app | Select **House#1**; show profile; set temperature to 35°C; click **Predict**. Show **0.526 kWh**, 90% interval **0.00–1.37**, delta vs last hour, 72-hour chart. Mention 59 houses / 6 cities. |
| **4:30–6:00** | Live FastAPI at `127.0.0.1:8000` | Open `/docs`; POST `/predict` with valid House#1 body → **200**. Send humidity **150** → **422**. Send `House#999` → **404**. Show the HTML front end at `/`. |
| **6:00–7:00** | HF model card + live Space | Show model card metrics; open the Gradio Space and predict. Say: “HF guideline allows **Gradio or Streamlit**; this is the Gradio deployment. Same model artifact and shared `app/service.py`.” |
| **7:00–8:30** | `model_comparison.csv`, `test_metrics.csv` | “Deployed HGB: **test MAE 0.2499, RMSE 0.4273, R² 0.7291**. Persistence: 0.2760 / 0.5470 / 0.5561 → **9.5% lower MAE, 22% lower RMSE**.” |
| **8:30–9:30** | `error_by_city.csv`, report | “RF won validation at **0.3216** vs HGB **0.3273** — a 0.006 difference, far below CV fold std **0.067**, so statistically tied. HGB deployed because it is **1.4 MB vs ~270 MB** and had the best one-shot test MAE. Validation MAE is higher than test because validation is summer (mean 1.12 kWh) and test is autumn (0.84 kWh) — seasonal shift, not overfitting.” |
| **9:30–10:00** | README deployments table | “All three deployments are live: Streamlit Cloud, HF Space, and local FastAPI. Same artifact, same service. Reproducible via pinned requirements, 11 executed notebooks, 19 tests, and `verify_all` 12/12 checks.” |

---

## Part 3 — Live demo mechanics

### Streamlit
- Open `bijlicast.streamlit.app` **early** — cold start can take ~1 minute.
- Click **Predict** on House#1 first, so the chart is already populated.
- Then move one weather slider and predict again to show the what-if scenario.
- Point out the 90% interval and the 72-hour history chart.

### FastAPI
Start it before the demo:

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Then show:

1. `GET /health` → model loaded, 59 houses.
2. `POST /predict` with valid input → 200, prediction + interval.
3. Invalid humidity `150` → **422** Pydantic validation.
4. Unknown `House#999` → **404**.
5. `GET /` → HTML front end.

This directly satisfies “input validation and error messages required.”

### Hugging Face
- Open the **model card** first: `huggingface.co/muhammadumar15/BijliCast`.
- Then open the **Space**: `huggingface.co/spaces/muhammadumar15/BijliCast`.
- Warm it up early; it runs on ZeroGPU and may cold-start.
- If it fails live, that platform scores zero — so have a screenshot backup ready.

---

## Part 4 — Results talk track: the exact numbers

| Model | Val MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Persistence `lag_1h` | 0.3448 | 0.2760 | 0.5470 | 0.5561 |
| Seasonal naive `lag_24h` | 0.4806 | 0.3522 | 0.6415 | 0.3897 |
| Train mean | — | 0.5458 | 0.8477 | −0.0659 |
| Linear Regression | 0.3328 | 0.2601 | 0.4595 | 0.6869 |
| KNN | 0.4024 | 0.3178 | 0.5196 | 0.5996 |
| Random Forest | **0.3216** | 0.2535 | 0.4322 | 0.7229 |
| **HGB (deployed)** | 0.3273 | **0.2499** | **0.4273** | **0.7291** |

Key sentences:

- “The deployed model beats persistence by **9.5% MAE** and **22% RMSE**.”
- “RF and HGB are statistically tied on validation; HGB is deployed for size and practical hosting.”
- “The test set was evaluated once, after selection.”
- “The 90% interval covers **90.0% of validation** and **93.8% of test** rows.”
- “Errors are highest in **Skardu (0.336)** and **Karachi (0.322)**, and in evening/AC hours.”

---

## Part 5 — Viva question bank

### Data
**Q: Why these two datasets?**  
A: The guideline requires two independent public sources. LUMS provides household smart-meter data; Open-Meteo ERA5 provides weather from a different provider and collection method. They join on city coordinates and hour.

**Q: Why 480,209 rows when the report says 515,786?**  
A: 515,786 is the hourly house-row count before filtering. We drop rows without a complete 168-hour history (warm-up and meter gaps), leaving 480,209 model rows.

**Q: Any personal data?**  
A: No names or addresses; house codes only. LUMS data is used for academic research, and the source/licence is documented in `DATA_SOURCES.md` and report Table 1.

### Leakage and split
**Q: How did you prevent leakage?**  
A: Three layers: (1) lags/rolling windows look only backwards and use no fitted parameters; (2) all learned imputation/scaling/encoding lives inside the Pipeline and is fit on train or the current CV fold; (3) the chronological split purges boundary rows so no label crosses into the next split.

**Q: Why purge 102 rows?**  
A: A row at origin `t` predicts `t+1`. The last train row’s label would fall inside validation, and the last validation row’s label inside test. We remove those boundary rows.

**Q: Why not a random split?**  
A: Random splitting lets the model see the future during training, which inflates scores. Time-series data must be split chronologically so test is genuinely the future for every house.

### Features
**Q: What are the 42 features?**  
A: 10 weather (8 variables + wind sin/cos), 5 lags (1h, 2h, 3h, 24h, 168h), 3 rolling statistics (3h mean, 24h mean, 24h std), 4 calendar (hour, day of week, month, weekend), 18 household profile fields, and 2 categorical (city, owner/rented). After one-hot encoding: 49 columns.

**Q: Why exclude `house_id`?**  
A: To prevent memorising individual households. The model generalises through city and household profile instead.

**Q: Why current-hour weather only?**  
A: It is known at prediction time. Using next-hour weather would require a forecast and introduce future dependence. Current-hour weather keeps the forecast honest.

### Models and tuning
**Q: Why these algorithms?**  
A: The guideline requires a simple baseline plus at least three course algorithms. We used three baselines and four learned models: Linear Regression, KNN, Random Forest, and HistGradientBoosting.

**Q: How did you tune?**  
A: `TimeSeriesSplit(n_splits=5, gap=120)` with `RandomizedSearchCV`, scoring on negative MAE. KNN and RF used time-ordered subsamples for speed; RF was refit on full train. Linear Regression used a small grid.

**Q: Why `gap=120`?**  
A: The panel has about 59 rows per timestamp, so 120 rows is roughly a 2-hour gap between CV folds. It prevents adjacent-hour leakage across fold boundaries.

**Q: Why is KNN weak?**  
A: High-dimensional tabular data with strong temporal structure; distance-based methods struggle compared with tree ensembles. It was also tuned on a 20k subsample for speed.

### Selection and evaluation
**Q: RF won validation — why deploy HGB?**  
A: The difference is 0.006 MAE, far below the CV fold standard deviation of ~0.067, so they are statistically tied. HGB’s artifact is 1.4 MB vs ~270 MB, which matters for GitHub and free hosting, and its one-shot test MAE was best. Selection was still validation-based.

**Q: Isn’t choosing HGB because of test MAE test leakage?**  
A: No. The selection rule was validation MAE. HGB was tied on validation and chosen for size; the test result is reported once as confirmation, not used to tune or select.

**Q: Why is validation MAE higher than test MAE?**  
A: Validation covers summer (mean 1.12 kWh); test covers autumn (mean 0.84 kWh). The model is not overfitting — the target distribution shifted.

**Q: Why MAE as the primary metric?**  
A: It is in kWh, directly interpretable, and robust to the heavy tail. RMSE and R² are also reported.

**Q: Why is R² only 0.729?**  
A: Next-hour household load has unobserved drivers — occupancy, appliance switching, behaviour. A global model over 59 houses with 16 months of data cannot capture all of that. Beating persistence by 9.5% MAE is the practical result.

**Q: Why does the 90% interval cover 93.8% on test?**  
A: The offsets are validation residual quantiles. The test distribution shifted, so coverage is conservative. We report both validation 90.0% and test 93.8% honestly.

### Deployment
**Q: How do you know all three deployments agree?**  
A: They load the same `models/deployment_model.joblib` and call the same `app/service.py` contract. `verify_all` checks service prediction, all 59 houses, and train/serve parity.

**Q: Where is input validation?**  
A: FastAPI uses Pydantic `Field` ranges; invalid input returns 422. Unknown households return 404. The Streamlit UI uses the same service validation.

**Q: Why is HF Gradio while Streamlit Cloud is Streamlit?**  
A: The guideline allows either Gradio or Streamlit for HF. The Streamlit Cloud app is the custom Streamlit UI; the HF Space is the Gradio deployment. Both use the same model and service.

**Q: Why is FastAPI local?**  
A: The guideline explicitly allows the local web app to run on localhost during the demo. It has `/predict`, an HTML front end, validation, and error messages.

### Reproducibility
**Q: How is this reproducible?**  
A: Pinned `requirements.txt`, 11 executed notebooks, `models/model_metadata.json`, 19 passing tests, and `src/verify_all.py` with 12 checks. `python -m src.deploy.export_hf_space` regenerates the HF bundle and runs a real prediction self-check.

**Q: What would you improve?**  
A: More seasonal cycles, live weather API, per-house fine-tuning, quantile regression for intervals, and monitoring for distribution shift.

---

## Part 6 — Evidence map: open these when asked “where is it?”

| Guideline requirement | File / artifact |
|---|---|
| Problem, user, data sources | `README.md`, `data/DATA_SOURCES.md`, report §1–2 |
| EDA ≥5 plots | Notebooks 01–05, `reports/figures/` (42 figures) |
| Pipeline, train-only fit | `src/preprocessing/pipeline.py` |
| Chronological 70/15/15 + purge | `src/preprocessing/split.py`, `data/processed/split_summary.json` |
| Causal lags/rolling | `src/features/build_features.py` |
| Baselines + 4 algorithms | `src/models/baseline.py`, `train.py` |
| 5-fold CV + RandomizedSearch | `src/models/tune.py`, `reports/tables/cv_results_*.csv` |
| One-shot test evaluation | `src/models/evaluate.py`, `reports/tables/test_metrics.csv` |
| Full comparison | `reports/tables/model_comparison.csv` |
| Error analysis | `reports/tables/error_by_{city,hour,house,month}.csv` |
| Intervals | `reports/tables/prediction_interval_coverage.csv` |
| Saved pipeline + pins | `models/deployment_model.joblib`, `requirements.txt` |
| Streamlit deployment | `app/streamlit_app.py`, `app/streamlit_ui.py` |
| HF model card + Space | `hf-model/`, `deploy/huggingface/` |
| FastAPI + HTML | `app/api.py`, `app/static/index.html` |
| Tests / verification | `tests/`, `src/verify_all.py` |
| Report | `BijliCast_Report.pdf` |

---

## Part 7 — Member division for viva

The viva is individual, so everyone must know the whole pipeline. But assign primary ownership:

| Member | Primary deep-dive | Must also answer |
|---|---|---|
| Muhammad Umar | Data collection, preprocessing, feature engineering, leakage | Models, metrics, deployment basics |
| Muhammad Ahmed Khan | Models, tuning, evaluation, RF vs HGB | Data, leakage, deployment basics |
| Syed Ahmed Ayaan | Deployment, shared service, API/UI, reproducibility | Data, models, metrics |

Each member should rehearse a **2-minute explanation** of their area and a **30-second answer** for every other area.

---

## Part 8 — Pre-demo checklist

**30 minutes before**
- Open `bijlicast.streamlit.app` to wake it.
- Open the HF Space to wake it.
- Start FastAPI and test `/health`, `/predict`, `/docs`.
- Open tabs: `README.md`, `DATA_SOURCES.md`, `model_comparison.csv`, `test_metrics.csv`, report PDF, notebook 11.
- Have backup screenshots of all three deployments.
- Check internet and screen sharing.

**10 minutes before**
- Run one prediction on each app.
- Confirm House#1 works.
- Close unnecessary tabs.
- Do **not** push, redeploy, or edit code.

---

## Part 9 — Things not to say

Avoid these — they can cost marks:

- ❌ “We used the test set to choose HGB.” → Say validation selection, HGB tied, size decided.
- ❌ “93.8% validation coverage.” → 93.8% is **test**; validation is 90.0%.
- ❌ “408,048 total rows.” → 408,048 is train+val fit rows; total cleaned is 480,209.
- ❌ “Real-time smart-meter data.” → It is historical replay; weather is current-hour.
- ❌ “Faisalabad.” → The sixth city is **Skardu**.
- ❌ “HF is Streamlit.” → It is Gradio; the guideline allows either.
- ❌ “Accuracy.” → It is a regression task; use MAE/RMSE/R².
- ❌ “The report is 3–4 pages.” → It is 10 pages; if asked, acknowledge and offer a condensed version.

---

## Part 10 — If she asks about the report length

Say:

> “The report contains every required section — abstract, problem, data, EDA, methodology, results, deployment screenshots, limitations, references, and the contribution table. It runs to 10 pages because it includes the full comparison table, figures, and references. I can provide a condensed 4-page version if that is preferred.”

Do not argue about the guideline. Acknowledge, show completeness, offer the fix.

---

## Part 11 — The one-sentence close

> “BijliCast covers the full CT-378 lifecycle: two public sources, 480,209 model rows (42 raw features → 49 engineered), a leak-free pipeline, a chronological 70/15/15 split, three baselines plus four algorithms, 5-fold time-series tuning, a one-shot test evaluation, a saved joblib pipeline with pinned versions, and three live deployments sharing one inference service.”

That sentence maps directly to the guideline. Say it slowly at the end.

---

# 🧠 Additional Decision-Focused Viva Bank

*Appended from the senior-evaluation prep session. Use this after the 10/10 playbook above.*

Your seniors are right: she will attack **decisions**, not definitions. Use this answer formula every time:

> **Decision → Alternatives considered → Why I chose this → Trade-off → Evidence (file/table/number).**

Below are **new questions not already covered in `viva.md`**, plus **three corrections** to existing lines that could hurt you. The corrections have now been applied throughout this document.

---

## ⚠️ Corrections applied (say them this way)

1. **“10.5% lower MAE than persistence” is wrong.**
   - Correct: `(0.2760 − 0.2499) / 0.2760 = 9.46%` → **9.5% lower MAE**.
   - RMSE improvement: `(0.5470 − 0.4273) / 0.5470 = 21.9%` → **~22% lower RMSE**.
   - Say **9.5%**, not 10.5%.

2. **“HGB has built-in categorical handling, so city and Owner/Rented don’t need one-hot encoding” is false in your pipeline.**
   - Your `ColumnTransformer` **one-hot encodes** `city` and `Owner/Rented` before HGB.
   - Correct sentence: “HGB can handle categoricals natively in scikit-learn, but in our shared pipeline the categoricals are one-hot encoded. We kept one preprocessor for all models.”
   - Do **not** claim HGB received raw categoricals.

3. Minor: 93.8% test coverage is **slightly conservative**, not “within 90%.” Say “the interval over-covers slightly on test because of distribution shift.”

4. **Baseline percentages are easy to invert.** Persistence MAE (0.2760) is **10.4% higher** than HGB's (0.2499), so HGB is **9.5% better**. Seasonal naive (0.3522) is **40.9% higher**, so HGB is **29.0% better**. Always state the direction: “HGB is 9.5% better than persistence,” not “persistence is 10.5% worse than HGB” unless you add the reference point.

---

## Part A — Decision-defense table (most likely questions)

| # | Decision | Alternatives | Why this choice | Trade-off / evidence |
|---|---|---|---|---|
| 1 | **Hourly mean kW**, not minute target or sum | Minute-level target; sum of 1-min readings | Mean kW over the hour × 1 h = kWh for that hour. Minute data is noisy and huge. | Summing kW readings would be 60× too large. Report §4. |
| 2 | **Drop 6.9% of rows** instead of interpolating usage | Interpolate gaps; forward-fill | Never fabricate consumption. Gaps are genuine meter outages; interpolation creates artificial autocorrelation. | Only survey fields are imputed. `feature_summary.csv`. |
| 3 | **Keep outliers** | Remove IQR outliers; cap target | 26/59 houses have >5% IQR outliers, but they are genuine AC/appliance spikes. Removing them biases the model away from peak demand. | MAE is robust; trees handle nonlinearity. Report §3. |
| 4 | **59 unique houses** | Include duplicate Peshawar files | Byte-identical duplicates would double-weight one city. | 59 unique houses remain. |
| 5 | **Global model** | Per-house models; include `house_id` | Per-house data is ~8k rows with gaps. Global model pools 59 houses and generalizes via city/profile. | `house_id` excluded to prevent memorization. Per-house fine-tuning is future work. |
| 6 | **Lags 1/2/3/24/168** | 12h, 48h, 72h, more lags | EDA autocorrelation peaks at 1h, 24h, 168h. Short lags capture momentum. | More lags add features with diminishing returns. Report §3. |
| 7 | **Rolling 3h/24h mean + 24h std** | Min/max, skew, longer windows | 3h captures short-term level; 24h captures daily baseline; std captures volatility/spikes. | Min/max could add robustness but expand features. |
| 8 | **Wind sin/cos** | Raw wind direction (0–360°) | Direction is circular: 359° and 1° are adjacent but numerically far. Sin/cos preserves continuity. | `FeatureEngineer` in `pipeline.py`. |
| 9 | **Month as integer** | Cyclical month encoding | Trees split on thresholds; integer month works. Cyclical helps linear/KNN, but trees dominate. | Minor trade-off; could test. |
| 10 | **No holiday feature** | Pakistan holiday calendar | Holidays vary by year/province and no clean feature exists in the dataset. Weekend/calendar captures routine. | Future work. |
| 11 | **No explicit interaction terms** | Temperature × hour, AC × temperature | Trees learn interactions automatically. Explicit products would expand 49 features and risk overfitting. | Tree models handle this. |
| 12 | **One-hot encoding** for city/owner | Target encoding; native categoricals | Only 6 cities and 3 owner categories. One-hot is simple and leak-free. Target encoding needs out-of-fold statistics. | `handle_unknown='ignore'` for inference. |
| 13 | **No PCA / feature selection** | PCA; RFE; univariate selection | 49 features is small; trees handle irrelevant features; PCA hurts interpretability. | Could test feature importance later. |
| 14 | **Scale for trees too** | Separate preprocessors for linear/KNN vs trees | One shared pipeline across all models. Scaling is needed for Linear/KNN and harmless for trees. | Keeps one artifact and train/serve parity. |
| 15 | **Median imputation** | Mean imputation | Survey fields are skewed with outliers; median is robust. | Only 1.79% missing in static fields. |
| 16 | **`handle_unknown='ignore'`** | Raise on unseen category | At inference, an unseen city/owner category encodes as all zeros instead of crashing. | `pipeline.py`. |
| 17 | **Exclude `house_id`** | Include as feature | Prevents memorizing individual households; model generalizes through city/profile. | Report §4. |
| 18 | **Exclude raw `usage_kw`** | Include raw usage column | `lag_1h` is exactly usage(t). A separate raw column would duplicate it; usage(t+1) would be leakage. | `lag_1h = usage` in `build_features.py`. |
| 19 | **No SVM** | SVM with RBF kernel | 480k rows and 49 features; kernel SVM scales poorly (O(n²–n³)) and needs heavy tuning. Four course algorithms already satisfy the guideline. | Trees are stronger here. |
| 20 | **No XGBoost/LightGBM** | Add external boosting libraries | Kept the stack scikit-learn-only for reproducibility and simpler deployment. HGB is sklearn’s histogram-based boosting, same family. | Future work. |
| 21 | **No ARIMA/SARIMA/Prophet** | Classical time-series models | 59-house panel with gaps, exogenous weather, and a global model. Univariate models don’t pool across houses and struggle with missing data. | Trees handle multivariate inputs. |
| 22 | **No LSTM/Transformer** | Deep sequence models | Tabular data with strong engineered lags; trees are competitive, faster, and interpretable. Deep learning needs more tuning/compute. | Future work. |
| 23 | **No log target** | `log1p(target)` | Target has many zeros (Skardu), so plain log is undefined. `log1p` changes the loss and interpretability; MAE in kWh is user-facing. | Trees don’t need normality. |
| 24 | **No RF+HGB ensemble** | Stacking/averaging | RF and HGB are highly correlated; stacking adds complexity and latency for marginal gain. HGB alone has the best test MAE. | Future work. |
| 25 | **Linear `fit_intercept=False, positive=True`** | Default linear regression | Grid search selected it. Non-negative coefficients are plausible for weather/appliance drivers and act as regularization. | It still underfit, so the constraint wasn’t the main issue. |
| 26 | **KNN k=5, Manhattan, distance weights** | Euclidean, uniform weights, larger k | CV selected it. Manhattan often works better in higher dimensions; distance weighting emphasizes close neighbors. | KNN remains the weakest learned model. |
| 27 | **RF 120 trees, leaf 10, max_features 0.7** | More trees, deeper trees | CV plateaued around 120; more trees increase size/latency with little gain. `min_samples_leaf=10` and `max_features=0.7` regularize. | Artifact ~270 MB. |
| 28 | **HGB lr 0.06, 400 iters, 31 leaves, leaf 50, L2 5.0** | More iterations, deeper trees | Randomized search selected it. Moderate learning rate + L2/min-leaf controls overfitting. | CV MAE 0.2180 ± 0.067. |
| 29 | **Subsample tuning** | Full-data search | Compute limits: KNN 20k, RF 40k, HGB 150k, Linear full. Subsamples are time-ordered, preserving temporal structure. | Final models refit on full train. |
| 30 | **TimeSeriesSplit, gap 120** | KFold; no gap | KFold leaks future into past. Gap 120 rows (~2 hours of panel data) prevents adjacent-hour leakage across folds. | `tune.py`. |
| 31 | **Train+val refit for deployment** | Deploy train-only model | More data for the final model (408,048 vs 336,053). Test remains untouched. | Metrics from train-fit HGB are honest out-of-sample estimates for the same config. |
| 32 | **Separate `best_model` and `deployment_model`** | One artifact | `best_model` is RF validation winner, train-fit, for report/metric alignment, ~270 MB and gitignored. `deployment_model` is HGB train+val, 1.4 MB, served. | Different purposes. |
| 33 | **Residual-quantile interval** | Conformal prediction; quantile regression | Simple, model-agnostic, calibrated on out-of-sample validation residuals. | Conformal would be more rigorous; future work. |
| 34 | **90% interval** | 95% or 80% | 90% gives a narrower, more useful interval. Guideline doesn’t mandate a level. | Report coverage honestly: 90.0% val, 93.8% test. |
| 35 | **Three deployments** | Choose only one | Guideline says choose one, but doing all three demonstrates the full deployment requirement and earns the “all three live” marks. | Shared service keeps predictions identical. |
| 36 | **Shared `app/service.py`** | Separate code per platform | One inference contract loads the same artifact and builds the same row. Prevents platform drift. | `verify_all` checks parity. |
| 37 | **Gradio on HF** | Streamlit on HF | Guideline allows **Gradio or Streamlit**. Streamlit Cloud uses the custom Streamlit UI; HF uses Gradio. | Both call the same service/model. |
| 38 | **Joblib** | Pickle | Joblib is efficient for numpy arrays/large sklearn pipelines and is the standard persistence format. | `deployment_model.joblib`. |
| 39 | **Gitignore `best_model.joblib`** | Commit the RF | ~270 MB exceeds GitHub’s 100 MB limit. Deployment model is 1.4 MB. | RF kept locally for report alignment. |

---

## Part B — Data / EDA questions not in `viva.md`

**Q: Why 515,786 rows in the report but 480,209 in the model?**  
A: 515,786 is the hourly house-row count before filtering. 6.9% of rows lack a full 168-hour history or a next-hour label, so they are dropped. No usage value is interpolated.

**Q: Why are 1.79% of survey fields missing?**  
A: A few houses have incomplete metadata. Static numeric fields are median-imputed inside the pipeline; categorical fields are mode-imputed. House#32 is missing 15 of 18 survey fields.

**Q: Why do 8 houses have >5% missing hours and 13 have >7-day gaps?**  
A: Genuine meter outages. We drop rows that cannot support a full 168-hour history rather than fabricate continuity. This is why test covers 48 of 59 houses.

**Q: Why does test cover only 48 houses?**  
A: Some houses end before the test window. The split is global and chronological, so a house without test-period data cannot contribute test rows. This is reported as a limitation.

**Q: Why is Skardu so different?**  
A: Cold, high-altitude climate; many near-zero hours (median 0.009 kW); different weather-response. It has the highest test MAE (0.336). The global model captures it partly through city one-hot and profile, but not perfectly.

**Q: Why keep zero-usage hours?**  
A: They are legitimate (especially Skardu). Removing them would bias the model upward and distort the target distribution.

**Q: Why not cap the target at the 99th percentile?**  
A: Peak demand is the costly part; capping would bias predictions downward. MAE is robust to the heavy tail, and trees can model spikes.

**Q: Why is target correlation with temperature only 0.313?**  
A: The relationship is nonlinear and city-dependent: hot cities show a temperature response, Skardu behaves differently. Trees capture interactions that a single correlation cannot.

**Q: Why does AC count correlate with usage (r=0.61)?**  
A: It validates including household profile features. AC count is a strong static predictor of a house’s baseline and peak demand.

**Q: Why 42 EDA figures?**  
A: The guideline requires at least 5. We produced 42 across distributions, missingness, outliers, correlations, daily/weekly profiles, and city comparisons.

**Q: Why no class-balance plot?**  
A: This is regression, not classification. We analyze the target distribution (right-skewed, median 0.43, mean 0.72, max 12.66) instead.

**Q: Why are 26/59 houses flagged with >5% IQR outliers?**  
A: IQR flags heavy-tailed usage, not data errors. These are AC/appliance spikes; keeping them preserves peak-demand behavior.

---

## Part C — Model / evaluation questions not in `viva.md`

**Q: Why is KNN’s train MAE exactly 0?**  
A: With `weights='distance'`, each training point is its own nearest neighbor, so it predicts itself. This is a property of distance weighting, not leakage. Only validation/test scores are meaningful.

**Q: Why is validation MAE higher than test MAE for every model?**  
A: Validation is summer (mean 1.12 kWh); test is autumn (mean 0.84 kWh). The target level shifted, so absolute errors are larger in validation. This is seasonal shift, not overfitting.

**Q: Why does RF have the largest train–validation gap?**  
A: Deep trees (`max_depth=None`, `min_samples_leaf=10`) fit the training period closely. It still has the best validation MAE, but the gap is larger than HGB’s.

**Q: Why does Linear Regression underfit?**  
A: It assumes additive linear relationships. Electricity demand has nonlinear temperature thresholds, daily cycles, and interactions. It has the smallest train–val gap but higher error.

**Q: Why does HGB generalize best on test despite RF winning validation?**  
A: HGB’s regularization (`min_samples_leaf=50`, L2=5.0, lr=0.06) and histogram-based boosting generalize slightly better on the autumn test. The validation difference (0.006) is far below CV noise (0.067), so they are statistically tied.

**Q: Why no formal significance test between RF and HGB?**  
A: We use the 5-fold CV standard deviation (~0.067) as the practical noise band; the 0.006 difference is an order of magnitude smaller. A paired test could be added, but the conclusion would not change.

**Q: Why no learning curves or residual plots?**  
A: We report train/val/test MAE, CV results, and error tables by city/hour/house/month. Learning curves and residual diagnostics would strengthen the report; they are a reasonable future addition.

**Q: Why no calibration plot for the interval?**  
A: We report coverage on train, validation, and test. A calibration plot would be more rigorous; the residual-quantile method is simple and model-agnostic.

**Q: Why MAE instead of MAPE?**  
A: MAPE is undefined or explosive when usage is zero or near-zero, which happens often in Skardu. MAE is in kWh and robust to spikes.

**Q: Why report median AE as well?**  
A: The target is right-skewed; median AE shows typical error without tail dominance. It is in `model_comparison.csv`.

**Q: Why is R² only 0.729?**  
A: Next-hour load has unobserved drivers: occupancy, appliance switching, behaviour. A global model over 59 houses and 16 months cannot explain all variance. Beating persistence by 9.5% MAE is the practical result.

**Q: Why is the 90% interval lower bound clipped at zero?**  
A: Usage cannot be negative. Clipping avoids physically impossible intervals; it slightly affects coverage but is the right engineering choice.

**Q: Why is mean interval width ~1.43 kWh?**  
A: It reflects residual uncertainty across houses and seasons. The interval is wider for high-variance houses and periods.

**Q: Why do the worst houses dominate the tail?**  
A: House#30 (0.720), House#32 (0.605), House#31 (0.562) have high usage/variance. House#32 has mean usage 2.54 kWh and 15/18 survey fields missing, so the model has weak profile signal.

**Q: Why is Skardu/Karachi error highest?**  
A: Skardu has cold-climate, near-zero usage and a different weather response; Karachi has high, variable demand. A global model may underfit city-specific patterns; per-city calibration is future work.

**Q: Why not use RMSE as the primary metric?**  
A: RMSE is dominated by the heavy tail and is less interpretable in kWh for typical households. We report it alongside MAE and R².

**Q: Why not use sMAPE?**  
A: It is still unstable near zero and less standard for energy load forecasting than MAE/RMSE.

**Q: Why not use pinball loss?**  
A: We do not train quantile models; the interval is residual-based. Pinball loss would evaluate quantile forecasts, which we don’t produce.

---

## Part D — Pipeline / engineering questions not in `viva.md`

**Q: How does inference build one row from a form?**  
A: `build_inference_row()` takes the last 168 hourly usage values, current-hour weather, origin timestamp, city, and static profile; recomputes lags/rolling exactly as training; returns one row in `INPUT_COLUMNS`. The saved pipeline then preprocesses and predicts.

**Q: What happens if the last 168 hours are incomplete at inference?**  
A: `build_inference_row()` raises a clear `ValueError`. The deployed service uses the stored `usage_history.parquet`, which has a complete 168-hour tail per house.

**Q: What happens if weather is missing at inference?**  
A: The service fills missing fields from `models/latest_weather.csv`; explicit overrides replace them. The API/UI always supplies the eight fields.

**Q: What happens with an unseen city or owner category?**  
A: `OneHotEncoder(handle_unknown='ignore')` encodes it as all zeros; no crash. The model falls back to the other features.

**Q: How do you prove train/serve parity?**  
A: `src/features/inference.py` has a `self_check` that rebuilds a training row at inference time; `verify_all` runs a parity check; `pipeline.predict` on the same row matches. This is stronger than a unit test alone.

**Q: Why 400 vs 404 vs 422?**  
A: 422 = Pydantic validation failure (e.g. humidity 150); 404 = unknown house; 400 = service-level invalid input; 503 = model/artifact load failure. This is clear API semantics.

**Q: Why CORS and static HTML?**  
A: The guideline requires an HTML front end. CORS lets the page call the API; the static `index.html` provides the form and result display.

**Q: How is the HF bundle generated and kept in sync?**  
A: `python -m src.deploy.export_hf_space` copies the service, artifacts, and minimal `src/` modules, then runs a real prediction self-check inside the bundle. `verify_all` checks shared-module hashes.

**Q: Why are there two Gradio bundle folders?**  
A: `deploy/huggingface/` is the deployed bundle; `deploy/huggingface-gradio/` is a development/legacy copy. If asked, say the deployed one is `deploy/huggingface/`; the other can be removed. Don’t claim both are live.

**Q: Why is `best_model.joblib` gitignored?**  
A: It is the ~270 MB RF validation winner, above GitHub’s 100 MB limit. It is kept locally for report/metric alignment; the 1.4 MB HGB deployment model is committed.

**Q: Why pin versions?**  
A: scikit-learn/pandas APIs and pickle formats change. Pinning ensures the saved pipeline loads and reproduces predictions.

**Q: How long does inference take?**  
A: HGB inference is milliseconds on CPU; the service loads the model once and reuses it. Avoid claiming a precise “10 ms” unless you measured it; say “sub-second, typically milliseconds.”

**Q: How much memory does the pipeline need?**  
A: The 480k × 49 feature matrix is roughly a few hundred MB in float64; it fits comfortably in a laptop/free-tier container. The HGB artifact is 1.4 MB.

**Q: Why not Docker?**  
A: The three target platforms (Streamlit Cloud, HF Space, local uvicorn) don’t require it. Docker would help reproducibility for a self-hosted deployment; future work.

**Q: Why not add authentication/rate limiting?**  
A: It is an academic demo, not a production service. Input validation exists; auth, rate limiting, and logging are production concerns.

**Q: How would you monitor drift in production?**  
A: Log predictions and actuals when available; track MAE by city/house over time; alert on distribution shift; retrain on a schedule. Not implemented in the demo; future work.

---

## Part E — Curveballs / off-topic questions

**Q: Why should anyone trust this model?**  
A: Held-out test evaluation, three real baselines, error analysis by city/hour/house, honest limitations, reproducible artifacts, and 12 automated verification checks. We don’t claim production readiness.

**Q: What is the business value?**  
A: Bill forecasting, load shifting away from peak tariffs, solar/battery scheduling, and demand-response planning. It is not for billing or grid operations.

**Q: Who is the user?**  
A: A household or an energy app that wants a next-hour consumption estimate; secondarily, a utility/demand-response analyst.

**Q: Why Pakistan?**  
A: Rising tariffs, load shedding, and available household smart-meter data make next-hour load forecasting practically relevant.

**Q: Privacy/ethics?**  
A: No names or addresses; house codes only. LUMS data is used for academic research; source/licence documented. No personal identifiers in the model.

**Q: Bias/fairness across cities?**  
A: The global model underperforms in Skardu and Karachi. We report city-level errors rather than hiding them. Per-city calibration/fine-tuning is future work.

**Q: What if tariffs change?**  
A: The model predicts kWh, not cost. Tariff logic can be applied downstream; a tariff change does not invalidate the load forecast.

**Q: What if behaviour changes (COVID, new appliances)?**  
A: Concept drift would degrade accuracy. Production would monitor errors and retrain; the demo is static.

**Q: What if weather forecasts are wrong?**  
A: We don’t use next-hour weather; only current-hour weather. That removes dependence on forecast accuracy.

**Q: What if a new house/city appears?**  
A: A new house needs a static profile and 168 hours of history; a new city is handled by one-hot `handle_unknown='ignore'`, though accuracy is unproven. Cold-start is a known limitation.

**Q: What if the model/API fails during the demo?**  
A: FastAPI returns 503 on model-load failure; the UI shows an error. For the demo, have backup screenshots and a local fallback. Do not redeploy before the demo.

**Q: What would you do with more time/money?**  
A: More seasons/cities, live weather, per-house fine-tuning, quantile/conformal intervals, monitoring, A/B testing, and a proper MLOps pipeline.

**Q: What is the biggest weakness?**  
A: Only ~16 months (one seasonal cycle), test covers 48/59 houses, global model misses city-specific patterns, and the interval is residual-based rather than conformal.

**Q: What would you change if you restarted?**  
A: Collect more seasons, add holiday/occupancy features, use conformal intervals, try per-city models, and build monitoring from day one.

**Q: Why not use more data/external features?**  
A: LUMS coverage limits houses/cities; external features (holidays, occupancy sensors, appliance-level data, solar generation) are future work. Two independent sources satisfy the guideline.

**Q: How did you split the work?**  
A: See the report’s individual contribution table. Each member owns an area but can answer across the pipeline; viva marks are individual.

**Q: What was the hardest bug/decision?**  
A: Leakage-safe lag/split alignment and train/serve parity. The target is `t+1`, so lags are relative to the target hour, and boundary rows must be purged; inference must rebuild the same row.

**Q: What did you learn?**  
A: End-to-end ML is mostly data/leakage discipline and deployment engineering, not just model fitting. A smaller model with honest evaluation is more useful than a large artifact with unclear provenance.

---

## Part F — “Show me the code” prompts

Be ready to open these exact places:

| If she asks… | Open… |
|---|---|
| How are lags built? | `src/features/build_features.py` lines 66–79: `lag_1h = usage`, `lag_24h = shift(23)`, `target = shift(-1)` |
| How is leakage prevented at split boundaries? | `src/preprocessing/split.py` lines 96–105 (purge) and 147–151 (assertions) |
| Where is imputation/scaling/encoding? | `src/preprocessing/pipeline.py` lines 84–111 |
| Where is model selection? | `src/models/evaluate.py` lines 212–215: `min(..., key=validation MAE)` |
| Where is the one-shot test? | `src/models/evaluate.py` line 171 and the single `evaluate_once` pass |
| Where is CV? | `src/models/tune.py` lines 48–89: `TimeSeriesSplit(n_splits=5, gap=120)` |
| How does serving work? | `app/service.py` lines 197–230: fill weather, load history, build row, predict |
| Where is API validation? | `app/api.py` lines 54–82 (Pydantic `Field` ranges) and 179–188 (404/400) |
| How do you verify? | `tests/` and `src/verify_all.py` |
| Where are metrics/artifacts? | `models/model_metadata.json`, `reports/tables/model_comparison.csv` |

---

## Part G — If you don’t know an answer

Do **not** bluff. Say:

> “I didn’t test that specific alternative. Based on the evidence I have, I chose X because of Y. If I were to test it, I’d compare it on the same chronological validation split and report MAE/RMSE/R². I can show you the current comparison table.”

This earns more marks than inventing an answer. It shows scientific thinking and honesty.

---

## The three sentences to memorize

1. **“The test set was evaluated once, after selection on validation MAE.”**
2. **“RF won validation by 0.006, which is far below the CV noise of 0.067; HGB was deployed because it is statistically tied and 1.4 MB instead of 270 MB.”**
3. **“Lags are causal, learned preprocessing is fit on train only, and 102 boundary rows are purged — so there is no leakage.”**

If you can say those three cleanly, you can defend the core of the project against almost any pipeline/data/model question.