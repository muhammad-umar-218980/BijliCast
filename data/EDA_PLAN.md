# BijliCast — EDA Planning & Plot Targets (Phase 2)

**Goal:** Produce ≥5 polished plots saved to `reports/figures/` that validate the feature groups, target distribution, and key relationships before model training.

---

## 📊 Required Plot List (5+ total)

| # | Plot Type | Description | Feature Group(s) Validated | Save As |
|---|---|---|---|---|
| **1** | **Electricity consumption distribution** | Histogram + KDE of `Usage (kW)` across all 59 houses (or a sample of 3–5). Shows typical usage range, skew, and zero/low‑usage prevalence. | A (target) | `reports/figures/electricity_distribution.png` |
| **2** | **24‑hour load profile** | Average consumption by hour‑of‑day across all houses in one city (or across all cities on separate subplots). Reveals peak‑day patterns (e.g., evening peak when people return home). | A + D (hour) | `reports/figures/load_profile.png` |
| **3** | **Temperature vs. Usage scatter with regression** | Scatter plot of `temperature` (weather) vs. `Usage (kW)` (hourly), with OLS regression line. Quantifies the weather‑usage relationship per city (we saw r=0.906 in Karachi, 0.227 in Skardu). | A + F (weather) | `reports/figures/temp_usage_scatter.png` |
| **4** | **Missing‑value heatmap per house** | Count/ratio of NaNs in `Usage (kW)` and each weather column, shown as a small matrix or bar chart per city. Highlights Islamabad’s ~48% zeros / NaNs and which columns are complete. | All (data quality) | `reports/figures/missing_heatmap.png` |
| **5** | **Lag‑correlation bar chart** | Pearson correlation between `usage_lag_1h` and target `Usage (kW)` (or the target itself). Shows how strongly past usage predicts current usage — the basis for autoregressive features Group B. | B (lags) | `reports/figures/lag_correlation.png` |
| **6 (optional)** | **City‑comparison boxplots** | Boxplot of `Usage (kW)` per city (Lahore, Karachi, …). Shows between‑city variability and whether some climates have systematically higher usage. | A + E (city) | `reports/figures/city_comparison.png` |
| **7 (optional)** | **Feature-importance bar chart** | From a quick Random Forest run on a single house or pooled data. Highlights which weather/static features matter most — guides Group F curation. | F (static) | `reports/figures/rf_importance.png` |

**Minimum:** 5 plots (1–5). **Recommended:** 7 (add 6–7 for richer report).

---

## 📁 Save All Plots To

```
reports/figures/
  electricity_distribution.png
  load_profile.png
  temp_usage_scatter.png
  missing_heatmap.png
  lag_correlation.png
  [city_comparison.png]
  [rf_importance.png]
```

Each plot must have a **caption** in the final PDF report explaining:
- What the plot shows
- Which feature group it validates
- Any notable finding (e.g., "Karachi shows the strongest temp‑usage correlation (r=0.906)")

---

## 📓 EDA Notebook Order (recommended)

| Notebook | Primary Tasks | Key Plots Produced |
|---|---|---|
| **01_data_understanding.ipynb** | Load 1 house + 1 weather CSV, parse timestamps, merge, verify 5,000+ rows, first plot | Plot 1 (electricity distribution) |
| **02_electricity_eda.ipynb** | Distributions, 24‑h load profiles, outlier analysis, missing patterns | Plots 1, 2, 4 |
| **03_weather_eda.ipynb** | Weather distributions, city‑to‑city comparisons, temperature‑usage scatter | Plots 3, 4, 6 |
| **04_integrated_eda.ipynb** (optional) | Lag correlations, feature‑importance, final insights | Plots 5, 7 |

**Total estimated time:** 2–3 hours across the 4 notebooks.

---

## ⚠️ Data‑Quality Flags to Document in EDA

| Issue | Expected Finding | Action |
|---|---|---|
| **UV Index** | 100% NaN across all 6 weather CSVs | Drop from model; not a usable feature |
| **Solar Energy** | Identical distribution to Solar Radiation | Drop one (keep solar_radiation) |
| **Islamabad usage zeros** | ~48% of rows are `0` | Flag as possible outages/absence; decide on imputation or rain‑flag in Phase 3 |
| **Duplicate timestamps** | Some houses have repeated `datetime` entries | Average or drop duplicates before engineering lags |
| **Extreme precipitation** | Lahore/Islamabad: up to ~118 mm/h | Consider log1p() or rain‑flag; winsorize if extreme skews model |
| **Variable appliance columns** | 50+ different col names across 59 houses | Use only `Usage (kW)` as unified target; appliance columns for EDA insight only |

---

## ✅ EDA Readiness Sign‑off (before Phase 3)

All five required plots saved to `reports/figures/` + this checklist completed:

- [ ] Timestamp parsing works for all 3 formats (date‑only, T‑sep, space‑sep)
- [ ] Merge of electricity + weather per house yields ≥5,000 rows with valid target
- [ ] Missing‑value patterns identified and documented (UV dropped, Solar Energy dropped)
- [ ] Outlier strategy decided (log1p, winsorize, or flag)
- [ ] Lag/rolling feature engineering code validated on one house
- [ ] Temporal split script (70/15/15) runs without data leakage

**End of EDA Planning.** Use this (`data/EDA_PLAN.md`) to guide the four notebooks. After saving 5+ plots, you're ready to proceed to Phase 3 (dataset prep → feature engineering → model training).