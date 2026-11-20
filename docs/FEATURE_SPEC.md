# BijliCast — Feature Specification (before EDA)

**Problem:** Weather-aware household electricity consumption forecasting in Pakistan
**Target:** Predict `Usage (kW)` of the next hour (regression, time-series)
**Data sources:** LUMS REWD-P (59 house CSVs + metadata) + Open-Meteo ERA5 (6 city weather CSVs)

---

## 📦 Groups of Features Used for Training

| Group | Features | Count | Source |
|---|---|---|---|
| **A. Weather (hourly, per city)** | `temperature`, `humidity`, `dew`, `precipitation`, `wind_speed`, `wind_direction`, `pressure`, `solar_radiation` | 8 | Open-Meteo ERA5 CSVs (Islamabad–Skardu) |
| **B. Autoregressive lags (per house)** | `usage_lag_1h`, `usage_lag_2h`, `usage_lag_3h`, `usage_lag_24h`, `usage_lag_168h` | 5 | Previous hour(s) of `Usage (kW)` within each house |
| **C. Rolling‑window (per house)** | `usage_roll_mean_3h`, `usage_roll_mean_24h`, `usage_roll_std_24h` | 3 | Sliding‑window stats on `Usage (kW)` within each house |
| **D. Calendar / temporal** | `hour`, `day_of_week`, `month`, `is_weekend` | 4 | Derived from `datetime` (extracted hour/DOW/month/weekend flag) |
| **E. City (categorical)** | `city` (6 levels: Lahore, Karachi, Islamabad, Multan, Peshawar, Skardu) | 1 | Weather city label; one‑hot or target‑encoded |
| **F. Static household (metadata)** | See **Curated Static Feature List** below | ~15–20 | `metadata_ultimate.xlsx` (59 rows × 46 cols) |

**Total: ~36–41 features** — well above the CT‑378 minimum of 15.

### ✅ Curated Static Feature List (from metadata)
Keep these 18 that most physically drive consumption:

| Feature | Type | Reason |
|---|---|---|
| `No. of people (Temp+Perm)` | Numeric | Occupant count → more people → more usage |
| `Air Conditioners` | Numeric (#) | Major kW draw, especially in Lahore/Karachi |
| `Refrigerators` | Numeric (#) | Always‑on load |
| `Geysers` | Numeric (#) | Electric water heating |
| `Electric heaters` | Numeric (#) | Space heating |
| `Washing Machines` | Numeric (#) | Periodic high‑kW usage |
| `Water Pumps` | Numeric (#) | Motor load |
| `UPS` | Numeric (#) | Backup, occasionally used |
| `Ceiling Fans` | Numeric (#) | Moderate kW, common in all cities |
| `Tube Lights` | Numeric (#) | Low kW but many |
| `Wall Fans` | Numeric (#) | Moderate kW |
| `Celling Fans` | Numeric (#) | Same as ceiling fans (possible duplicate) |
| `c K` (you pick 2) | — | — |
| `Residents` | Numeric | Alias for `No. of people` |
| `Build year` | Numeric | Older houses may have more insulation gaps |
| `Property Area (Marla)` | Numeric | Living space size |
| `Owner/Rented` | Categorical | Owner‑occupied vs rented may change usage patterns |
| `ACs` | Numeric | Short for Air Conditioners (keep one) |

**Drop / handle separately:** `Room Dimensions`, `Other Electronic Devices`, free‑text fields; impute missing with median/mode or add a "missing" flag. One house (Skardu) has many blanks — decide whether to keep or drop.

---

## 🔗 How Features Are Combined (Join Recipe)

For each of the 59 houses:

1. **Load** the house's electricity CSV (`rewdp_dataset/<city>/<house>.csv`)
2. **Normalize** column names:
   - Rename `Usage_kW` → `Usage (kW)` if present
   - Ensure `datetime` column exists (mixed formats: date‑only, `T`‑separator, space‑separator)
3. **Parse** `datetime` with explicit format: `pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')` and fallback to `'%Y-%m-%d'`
4. **Aggregate** minute‑level usage to hourly mean: `df['Usage (kW)'].resample('h', on='datetime').mean()` — this yields 8,784 rows per house
5. **Load** the city's weather CSV (`weather_dataset/<City>.csv`) — 8,784 rows, 8 usable columns
6. **Merge** on `datetime`: `merged = el_df.merge(w_df, on='datetime', how='left')` → 8,784 rows, all 8 weather columns + usage
7. **Engineer** Groups B–D (lags, rolling means, calendar) on the merged DataFrame **within each house**
8. **Join** the single metadata row for that house (match `House#N` + `City` → metadata_ultimate.xlsx)
9. **Drop** `House` join key; keep `city` as categorical

**Result:** One DataFrame per house with ~8,784 rows and ~36–41 columns (Groups A–F), ready for Phase 3 (dataset prep) and Phase 4 (ML).

---

## 🗓️ Temporal Train/Val/Test Split (Critical Rule)

**Never random‑shuffle rows.** Use **chronological ordering** per house:

| Split | Data range | Rationale |
|---|---|---|
| **Train** | First 70% of each house's hourly rows (earliest dates) | Model learns patterns from the start of the period |
| **Validation** | Next 15% | Hyperparameter tuning, early stopping |
| **Test** | Last 15% (latest dates) | Final, unseen‑chronologically evaluation |

**Why:** Static house features (metadata) are constant per house → a random split would leak the same house across train & test, inflating results. Time‑ordered split mimics real forecasting: "given known house + past usage + weather → predict next hour."

**Implementation sketch:**

```python
merged = merged.sort_values('datetime')
n = len(merged)
train_size = int(0.70 * n)
val_size = int(0.85 * n)

train = merged.iloc[:train_size]
val   = merged.iloc[train_size:val_size]
test  = merged.iloc[val_size:]
```

---

## 📦 Deployment Contracts (what the three apps will accept)

| Surface | Input contract | Model output |
|---|---|---|
| **Streamlit Community Cloud** | - City dropdown (6 options)<br>- Sliders/number inputs for the **8 weather features**<br>- Recent `Usage (kW)` auto‑filled from data or manual<br>- House selector (optional, auto‑fills static metadata from `metadata_ultimate.xlsx`)<br>- Date/hour picker for `datetime` | Predicted **next‑hour kW**, optional prediction interval, **one explanatory chart** (e.g., 24‑hour profile or feature‑importance bar) |
| **Hugging Face Spaces** | Same form as Streamlit (mirrors the app)<br>- Model card + pipeline on HF Hub | Same prediction; Gradio or Streamlit iframe |
| **FastAPI + `/predict` endpoint** | `POST` JSON:<br>`{ "city": "Lahore", "temperature": 35.2, "humidity": 50, "precipitation": 0, "wind_speed": 5, "hour": 14, "day_of_week": 2, "month": 8, "usage_prev": 2.1, ... }`<br>*(All 36–41 features, or raw inputs that the server‑side pipeline engineers into lags/rolling/calendar)* | JSON: `{ "predicted_kw": 3.87, "unit": "kW", "confidence_interval": [3.12, 4.62] }`<br>Input validation + clear error messages required |

> **Key point:** The saved `sklearn.pipeline` (or `joblib`) must **engineer lags/rolling/calendar inside the pipeline**, fitted on training data only (no data leakage). The deployed API/ui just passes raw inputs; the pipeline derives the same feature Groups B–D as during training.

---

## 📌 Quick-Start Checklist (before EDA notebook 01)

- [ ] `docs/FEATURE_SPEC.md` created ✅ (just now)
- [ ] `data/DATA_SOURCES.md` already committed (LUMS + Open-Meteo provenance)
- [ ] `datasets/metadata_ultimate.xlsx` added to git (untracked → `git add datasets/metadata_ultimate.xlsx`)
- [ ] Column names in metadata normalized (strip trailing spaces, `\n` in `"No. of Seniors\n(above 60)"`)
- [ ] Join recipe verified: `House#N` + `City` → correct CSV filename
- [ ] Target defined: `usage_next_hour = Usage (kW)` shifted −1
- [ ] Temporal split rule documented (70/15/15 chronological, per house)
- [ ] Three deployment input/output contracts noted above

---

**End of Feature Specification.** Use this as the reference document before opening the EDA notebooks. All plot ideas, missing‑value handling, and model‑training choices should align with the groups, split rule, and deployment contracts above.