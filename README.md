# BijliCast — Weather-Aware Household Electricity Consumption Forecasting in Pakistan

**CT-378 Machine Learning project** · time-series regression · target:
next-hour household electricity consumption (`next_hour_energy_kwh`, kWh).

BijliCast predicts how much electricity a Pakistani household will use in the
next hour from:

- its **last 168 hours of usage** (lags + rolling statistics),
- **current-hour weather** (temperature, humidity, dew point, precipitation,
  wind, pressure, solar radiation),
- calendar features (hour, weekday, month, weekend),
- its static profile (city, people, appliances, area, …).

Two independent public sources are combined: **LUMS REWD-P** smart-meter data
(59 households, 6 cities) and **Open-Meteo ERA5** weather. No next-hour weather
is used, so the model never depends on a perfect weather forecast.

---

## Results (held-out test set, evaluated once)

| Model | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| **Random Forest** (validation winner) | **0.3216** | 0.2535 | 0.4322 | 0.7229 |
| **HistGradientBoosting** (deployed) | 0.3273 | **0.2499** | 0.4273 | 0.7291 |
| Linear Regression | 0.3328 | 0.2601 | 0.4595 | 0.6869 |
| KNN | 0.4024 | 0.3178 | 0.5196 | 0.5996 |
| Persistence baseline (`lag_1h`) | 0.3448 | 0.2760 | 0.5470 | 0.5561 |
| Seasonal naive (`lag_24h`) | 0.4806 | 0.3522 | 0.6415 | 0.3897 |

- Tuning used `TimeSeriesSplit(n_splits=5)` on the training split only.
- Random Forest and HistGradientBoosting are statistically tied on validation
  (CV fold std ≈ 0.067). **HistGradientBoosting is the deployed model** because
  its artifact is ~1.4 MB instead of ~270 MB (GitHub rejects files >100 MB and
  Streamlit Community Cloud deploys from GitHub).
- 90% prediction interval from validation residual quantiles; test coverage 93.8%.

## Data

| Source | Role | Scale |
|---|---|---|
| [LUMS REWD-P](https://rewdp.lums.edu.pk/) smart-meter dataset | electricity usage | 59 unique households, 515,786 hourly rows |
| [Open-Meteo ERA5](https://open-meteo.com/) archive | weather | 6 cities, 12,480 hourly rows each |
| `datasets/metadata_ultimate.xlsx` | household profiles | 18 numeric + 1 categorical per house |

Provenance, licences and collection dates: [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md).
Feature definitions and inference contract: [`docs/FEATURE_SPEC.md`](docs/FEATURE_SPEC.md).
Phase status: [`phase.md`](phase.md).

## Repository layout

```
app/                Streamlit UI, shared inference service, FastAPI backend
.streamlit/         Streamlit theme (blue, viewer toolbar)
data/               raw / processed data (processed/ is git-ignored)
datasets/           LUMS REWD-P and weather CSVs
deploy/huggingface/ self-contained Hugging Face Space bundle (generated)
docs/               feature specification
models/             deployment model, profiles, metadata, prediction artifacts
notebooks/          01–11: EDA → preprocessing → split → modelling → evaluation
reports/            figures (42) and tables (metrics, CV, errors)
src/
  data/             Open-Meteo downloader
  data_prep.py      hourly cleaning / aggregation
  features/         feature builder + inference contract
  preprocessing/    leak-free pipeline + chronological split
  models/           metrics, baselines, training, tuning, evaluation, deployment export
  deploy/           app-data and Hugging Face Space export
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.13 is used for training; the pinned versions are in `requirements.txt`.

## Verify the checkout

Run the end-to-end verifier before deploying or submitting:

```bash
python src/verify_all.py
```

It checks the tracked model artifacts, the metadata deployment block, a service
prediction for all 59 houses, the FastAPI endpoints, notebook validity, and —
when the git-ignored `data/processed/` files are present — the split summary,
train/serve parity, and the pipeline smoke test.

Unit/API tests live in `tests/`:

```bash
pytest -q
```

On a fresh clone the data-dependent tests skip automatically because
`data/processed/` is git-ignored.

## Run the Streamlit app locally

```bash
streamlit run app/streamlit_app.py
```

Open <http://localhost:8501>. Select one of the 59 households, adjust the
current-hour weather if you want a what-if scenario, and press
**Predict next hour**. The app shows the point forecast, its 90% interval, and
the last 72 hours of usage.

The Streamlit app, the FastAPI backend and the Hugging Face Space all call the
same `app/service.py` contract, so the three deployments cannot drift apart.

## Deployments

| Target | Entry point | Status |
|---|---|---|
| Streamlit Community Cloud | `app/streamlit_app.py` | ready to deploy from GitHub |
| Hugging Face Spaces | `deploy/huggingface/app.py` | bundle generated, self-checked |
| FastAPI + HTML | `app/api.py` | verified locally (`/`, `/docs`, `/predict`) |

### 1. Streamlit Community Cloud

The app needs only these committed files (all small):

```
app/streamlit_app.py
app/streamlit_ui.py
app/service.py
.streamlit/config.toml
models/deployment_model.joblib       # HistGradientBoosting, ~1.4 MB
models/house_profiles.csv
models/usage_history.parquet          # 59 x 168 hourly values
models/latest_weather.csv
models/model_metadata.json
models/app_manifest.json
requirements.txt
src/                                  # imported by the app
```

Steps:

1. Push the repository to GitHub. **Do not commit `models/best_model.joblib`**
   (270 MB, already git-ignored) or `data/processed/` (git-ignored).
2. Go to <https://share.streamlit.io> → **New app**.
3. Repository: `muhammad-umar-218980/BijliCast`, branch `main`,
   main file path: `app/streamlit_app.py`.
4. Deploy. No secrets or API keys are needed.
5. Optional: in **Advanced settings**, select Python 3.13 to match the pins.

The first build installs `requirements.txt` (scikit-learn, pandas, numpy,
pyarrow, joblib, matplotlib, streamlit).

> **Large files:** `models/best_model.joblib` (Random Forest) is kept locally for
> report/metric alignment and is listed in `.gitignore`. GitHub rejects files
> over 100 MB, so it must not be pushed.

### 2. Hugging Face Spaces

The Space is a separate repository, so a self-contained bundle is generated:

```bash
python -m src.deploy.export_hf_space
```

It copies the UI, service, deployment artifacts, the minimal `src/` modules
needed to unpickle the pipeline, a pinned `requirements.txt`, and the HF model
card. The output is `deploy/huggingface/` (~1.6 MB) and the script runs a real
prediction inside that folder as a self-check.

Steps:

1. Create a **New Space** at <https://huggingface.co/new-space> with SDK
   **Streamlit**, hardware **CPU basic (free)** and visibility **Public**.
2. Publish the contents of `deploy/huggingface/` to the Space, either:
   - **Web UI:** *Files → Add file → Upload files/folders* and upload the
     folder contents (include the hidden `.streamlit/` directory), or
   - **git:**
     ```bash
     git clone https://huggingface.co/spaces/<user>/BijliCast hf-space
     cp -r deploy/huggingface/. hf-space/
     cd hf-space
     git add .
     git commit -m "Deploy BijliCast"
     git push
     ```
3. Wait for the build; the Space installs `requirements.txt` and opens the same
   Streamlit UI. The `README.md` front matter (`sdk: streamlit`,
   `app_file: app.py`) is already set.

If you change the UI, service, model or theme, re-run
`python -m src.deploy.export_hf_space` before pushing the Space again.

### 3. FastAPI + HTML front end

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

- UI: <http://127.0.0.1:8000/>
- Interactive docs: <http://127.0.0.1:8000/docs>
- Liveness: <http://127.0.0.1:8000/health>

The page loads the 59 households, fills the current-hour weather defaults, and
calls `POST /predict` with the selected household and edited weather.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "house_id": "House#1",
    "weather": {
      "temperature": 30, "humidity": 55, "dew": 18,
      "precipitation": 0, "wind_speed": 12, "wind_direction": 220,
      "pressure": 985, "solar_radiation": 450
    }
  }'
```

`weather` is optional; omitted fields fall back to the household's last known
hour. Responses include the point forecast, the 90% interval, the origin/target
timestamps, and the last 72 hourly values for charting. Weather ranges are
enforced by Pydantic (`422`), and unknown households return `404`.

## Reproduce the pipeline

```bash
python src/data/collect_openmeteo.py              # weather (already committed)
python src/data_prep.py                           # hourly_combined.parquet
# notebooks/04_data_integration.ipynb             # merged_dataset.parquet
python -m src.features.build_features             # 480,209 x 42 model features
python -m src.preprocessing.split                 # chronological 70/15/15
python -m src.preprocessing.pipeline              # pipeline smoke test
python -m src.features.inference                  # profiles + train/serve parity
python -m src.models.baseline                     # naive baselines
python -m src.models.train                        # untuned comparison
python -m src.models.tune                         # 5-fold CV searches
python -m src.models.evaluate                     # one-shot test evaluation + artifacts
python -m src.models.export_deployment            # compact HGB deployment model
python -m src.deploy.export_app_data              # 168 h history + latest weather
python -m src.deploy.export_hf_space              # self-contained HF Space bundle
```

Notebooks `01`–`11` are executed and contain the figures/tables used in the
report.

## Limitations

- The test period (autumn) has a lower target level than validation (summer);
  validation MAE is therefore higher for every model. This is a seasonal shift,
  not over-fitting, and selection never used the test set.
- The bundled app history ends at each household's last meter reading
  (Oct–Nov 2024), so the demo is a realistic replay rather than a live meter.
- The model is global (one model for all 59 houses) with city/static features;
  per-house models were not used.
- `House#32` is missing 15 of its 18 static numeric fields. Its 8,592 rows
  (1.79% of the dataset) are median-imputed inside the pipeline, and it has the
  second-worst test MAE (0.6045 kWh) while also having the highest mean usage
  (2.5375 kWh). This is a data-quality caveat, not a modelling result.
- KNN uses `weights="distance"`, so its training predictions are exact
  (train MAE 0.0, R² 1.0). Its validation/test metrics are the meaningful ones;
  the train row should not be read as evidence of leakage.
- The test split covers 48 of the 59 households because households end on
  different dates. Test metrics are therefore not equally weighted across all
  houses.
- Tuning uses time-ordered subsamples (KNN 20k, RF 40k, HGB 150k rows) and five
  random candidates per model. `TimeSeriesSplit(gap=120)` is expressed in rows,
  not hours. These are documented compute guards, not an exhaustive search.
- Predictions are for research/demo purposes, not billing or grid operations.
