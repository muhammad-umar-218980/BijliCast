# BijliCast — Feature Specification (as built)

> **Status:** Phase 3 complete. This document replaces the pre-EDA draft and
> matches the implemented code (`src/features/build_features.py`,
> `src/preprocessing/pipeline.py`, `src/features/inference.py`).
>
> **Task:** supervised **time-series regression** — predict a household's
> electricity consumption in the next hour.
> **Target:** `next_hour_energy_kwh`.

---

## 1. Data sources

| Role | Source | Scale |
|---|---|---|
| Electricity | LUMS REWD-P, 59 unique households | 515,786 hourly house-rows |
| Static household profile | `datasets/metadata_ultimate.xlsx` | 59 rows × 46 cols (18 curated + 1 categorical) |
| Weather | Open-Meteo ERA5, 6 cities | 12,480 hourly rows per city (2023-07-01 → 2024-12-01) |

Full provenance, licences and dates: `data/DATA_SOURCES.md`.

---

## 2. Row and target definition

Each row is the **origin hour `t`**. All features are known by the end of `t`;
the label is the household's usage in `t+1`.

| Symbol | Meaning |
|---|---|
| `t` | origin hour (last known reading) |
| `next_hour_energy_kwh` | `usage(t+1)` — hourly mean power ≈ energy for that hour in kWh |
| Lag "N h" | value N hours **before the target**, i.e. `usage(t+1−N)` |

Because the power is averaged over a one-hour window, its numeric value equals
the energy consumed in that hour in kWh.

---

## 3. Feature groups (42 model features)

### A. Weather — current hour (10)
`temperature`, `humidity`, `dew`, `precipitation`, `wind_speed`,
`wind_direction`, `pressure`, `solar_radiation`, `wind_dir_sin`, `wind_dir_cos`

Only current-hour weather is used. Next-hour weather is **not** used, so the
model never depends on a perfect forecast.

### B. Autoregressive lags — per house (5)
| Column | Definition |
|---|---|
| `lag_1h` | usage(t) |
| `lag_2h` | usage(t−1) |
| `lag_3h` | usage(t−2) |
| `lag_24h` | usage(t−23) |
| `lag_168h` | usage(t−167) |

### C. Rolling windows — per house (3)
| Column | Definition |
|---|---|
| `rolling_mean_3h` | mean usage over t−2 … t |
| `rolling_mean_24h` | mean usage over t−23 … t |
| `rolling_std_24h` | sample std over t−23 … t |

All windows are trailing (`min_periods` = window length); none is centered.

### D. Calendar (4) — built **inside the pipeline**
`hour`, `day_of_week`, `month`, `is_weekend`

(These columns also exist in the engineered table for EDA/plots, but the model
input uses `INPUT_COLUMNS`, which excludes them: `FeatureEngineer` recomputes
them from `datetime` at fit/transform time.)

### E. Static household profile (19)
18 numeric: people, adults, children, property area, covered area, floors,
rooms, air conditioners, air coolers, refrigerators, washing machines, ceiling
fans, wall fans, water pumps, geysers, electric heaters, UPS, build year.
1 categorical: `Owner/Rented` (Owner / Rented / Residing as tenant).

### F. Context (1)
`city` — 6 levels, one-hot encoded inside the pipeline.

**Total = 10 + 5 + 3 + 4 + 18 + 2 = 42 features**
→ 49 columns after one-hot encoding.

### Dropped columns
| Column | Reason |
|---|---|
| `UV Index` | 100% null in the ERA5 archive |
| `Solar Energy` | duplicate of `Solar Radiation` |
| `usage_kw` | raw label source; kept only as lags |
| `house_id` | excluded from features to avoid memorising houses |
| appliance sub-meters | inconsistent across houses; EDA only |

---

## 4. Preprocessing pipeline (leak-free)

Saved artifact: a single scikit-learn `Pipeline` with three steps.

```
raw frame (INPUT_COLUMNS)
  → FeatureEngineer        # stateless: calendar + wind sin/cos
  → ColumnTransformer      # numeric: median impute → StandardScaler
                           # categorical: mode impute → OneHotEncoder
  → estimator
```

Rules:

1. **Learned transforms** (median, mean/std, category list) are fitted on the
   **training split only** and re-fitted inside each CV fold.
2. **Lag/rolling features** are deterministic and causal: no fitted parameter,
   past values only, built identically at training and inference time.
3. Rows without a complete 168-hour history (warm-up, meter outages) are
   **dropped**, never interpolated. Only static metadata may contain missing
   values (median-imputed from train).
4. For cross-validation, pass the **raw frame** plus `build_pipeline(estimator)`;
   do not pass pre-transformed arrays.

`INPUT_COLUMNS` = `datetime` + 8 raw weather + 5 lags + 3 rolling + 18 static
numeric + 2 categorical. Calendar and wind encoding are **not** in the raw
frame — they are created by `FeatureEngineer`.

---

## 5. Split (Phase 3)

Global, strictly chronological, row-balanced boundaries:

| Split | Rows | Share | Period | Houses |
|---|---|---|---|---|
| Train | 336,053 | 69.98% | 2023-07-22 → 2024-06-29 | 59 |
| Validation | 71,995 | 14.99% | 2024-06-30 → 2024-08-28 | 57 |
| Test | 72,059 | 15.01% | 2024-08-28 → 2024-11-27 | 48 |

- No shuffling; boundaries are timestamps; no row is split across sets.
- **Boundary purge:** the last train/validation origin is removed because its
  label falls into the next period (102 rows purged).
- Test is evaluated **once**, in Phase 4.

---

## 6. Inference contract (train / serve parity)

A deployed app cannot compute 24h/168h lags from one form row. Inputs are:

| Input | Source |
|---|---|
| Household profile | `models/house_profiles.csv` (59 rows, static attributes) |
| Last 168 hourly usage values | bundled history for the selected house |
| Current-hour weather (8 values) | user input / weather API |
| Origin timestamp `t` | user input / "now" |

`build_inference_row(...)` in `src/features/inference.py` turns these into one
`INPUT_COLUMNS` row; the saved pipeline then returns the prediction.

Parity is verified by `python -m src.features.inference` (reproduces a training
row exactly and transforms it to 49 columns with no NaNs).

---

## 7. Artifacts

| Path | Contents |
|---|---|
| `data/processed/hourly_combined.parquet` | electricity + weather, hourly |
| `data/processed/merged_dataset.parquet` | + static household features |
| `data/processed/features_dataset.parquet` | 480,209 × 45 modelling table |
| `data/processed/splits/{train,val,test}.parquet` | chronological splits |
| `data/processed/split_summary.json` | boundaries, counts, target stats |
| `reports/tables/feature_summary.csv` | per-column profile |
| `models/house_profiles.csv` | deployment profile lookup |

---

## 8. Deployment contracts (Phase 5)

| Surface | Inputs | Outputs |
|---|---|---|
| Streamlit Community Cloud | house selector (auto-fills profile), city, weather sliders, origin hour | next-hour kWh + prediction interval + one explanatory chart |
| Hugging Face Spaces | model card + pipeline + live Gradio/Streamlit Space | same prediction |
| FastAPI + HTML | `POST /predict` JSON validated with Pydantic | `{prediction_kwh, interval, model, warnings}` + clear errors |

All three load the same joblib pipeline and the same inference helper.

---

## 9. Reproducibility

```bash
python src/data/collect_openmeteo.py      # weather (already committed)
python src/data_prep.py                   # hourly_combined.parquet
# notebooks/04_data_integration.ipynb     # merged_dataset.parquet
python -m src.features.build_features     # features_dataset.parquet
python -m src.preprocessing.split         # train/val/test + summary
python -m src.preprocessing.pipeline      # pipeline smoke test
python -m src.features.inference          # profiles + parity check
```

---

## 10. Status checklist

- [x] ≥5,000 rows → 480,209
- [x] ≥15 features → 42
- [x] Two independent public sources
- [x] Provenance documented
- [x] No PII (house codes only)
- [x] Leak-free feature engineering and pipeline
- [x] Chronological 70/15/15 split with boundary purge
- [x] Deployment artifacts defined
- [ ] Models, tuning, test metrics (Phase 4)
- [ ] Three live deployments (Phase 5)
