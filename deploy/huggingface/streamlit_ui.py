
"""Streamlit UI for BijliCast (shared by local/Community Cloud and HF Spaces).

The wrapper (:mod:`app.streamlit_app`) passes the repository root; this module
only knows about ``<root>/models`` and ``<root>/src``. That keeps the exact same
UI working inside a Hugging Face Space, where the app files live at the root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from service import BijliCastError, get_service  # noqa: E402

from src.utils.config import (  # noqa: E402
    CITY_COL,
    HOUSE_COL,
    INPUT_WEATHER_FEATURES,
    STATIC_CATEGORICAL_FEATURES,
    STATIC_NUMERIC_FEATURES,
    TIME_COL,
)

WEATHER_SPECS = {
    "temperature": ("Temperature (°C)", -10.0, 50.0, 0.5, "%.1f"),
    "humidity": ("Humidity (%)", 0.0, 100.0, 1.0, "%.0f"),
    "dew": ("Dew point (°C)", -25.0, 35.0, 0.5, "%.1f"),
    "precipitation": ("Precipitation (mm)", 0.0, 25.0, 0.1, "%.1f"),
    "wind_speed": ("Wind speed (km/h)", 0.0, 50.0, 0.5, "%.1f"),
    "wind_direction": ("Wind direction (°)", 0.0, 360.0, 5.0, "%.0f"),
    "pressure": ("Pressure (hPa)", 700.0, 1050.0, 1.0, "%.0f"),
    "solar_radiation": ("Solar radiation (W/m²)", 0.0, 1100.0, 10.0, "%.0f"),
}

PROFILE_METRICS = [
    ("No. of people (Temp+Perm)", "People", "%.0f"),
    ("Air Conditioners", "Air conditioners", "%.0f"),
    ("No. of rooms", "Rooms", "%.0f"),
]

PROFILE_TABLE_COLUMNS = [
    "No. of people (Temp+Perm)",
    "No. of Adults (14-60)",
    "No. of Children (0-13)",
    "Property Area (Marla)",
    "Covered Area",
    "No of Floors",
    "No. of rooms",
    "Air Conditioners",
    "Air Coolers",
    "Refrigerators",
    "Washing Machines",
    "Celling Fans",
    "Wall Fans",
    "Water Pumps",
    "Geysers",
    "Electric heaters",
    "UPS",
    "Build year of house",
    "Owner/Rented",
]


def _fmt(value, spec: str = "%.0f") -> str:
    if value is None:
        return "n/a"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(number):
        return "n/a"

    return spec % number


def _static_dict(profile: pd.Series) -> dict:
    static = {}

    for col in STATIC_NUMERIC_FEATURES:
        static[col] = profile[col]

    for col in STATIC_CATEGORICAL_FEATURES:
        static[col] = profile[col]

    return static


def run(root: str | Path | None = None) -> None:
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()

    st.set_page_config(
        page_title="BijliCast - next-hour electricity forecast",
        page_icon="⚡",
        layout="wide",
    )

    try:
        service = get_service(str(root))
    except BijliCastError as exc:
        st.error(f"Deployment bundle error: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        st.error(f"Could not load the deployment bundle: {exc}")
        st.stop()

    model_info = service.model_info()
    houses = service.houses()
    by_house = {h["house_id"]: h for h in houses}

    st.title("⚡ BijliCast")
    st.caption(
        "Weather-aware next-hour household electricity forecast for Pakistan · "
        "LUMS REWD-P households + Open-Meteo ERA5 weather"
    )

    # -------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------
    with st.sidebar:
        st.header("Household")

        labels = {
            f"{h['house_id']} · {h['city']}": h["house_id"]
            for h in houses
        }

        choice = st.selectbox(
            "Select a household",
            list(labels),
            index=0,
        )

        house = labels[choice]
        context = by_house[house]
        city = context["city"]

        st.divider()
        st.subheader("Model")

        st.write(f"**{model_info['name']}**")

        if model_info["test_mae"] is not None:
            st.write(
                f"Held-out test MAE: **{model_info['test_mae']:.3f} kWh**"
            )

        if model_info["test_r2"] is not None:
            st.write(f"Test R²: **{model_info['test_r2']:.3f}**")

        coverage = model_info["interval_val_coverage"]

        coverage_text = (
            f"{coverage:.0%} coverage"
            if coverage is not None
            else "coverage unavailable"
        )

        st.caption(
            f"Trained on {model_info['n_train_rows']:,} rows. "
            f"Prediction interval from validation residuals ({coverage_text})."
        )

        st.divider()

        st.caption(
            "Uses only current-hour weather and the last 168 h of usage. "
            "No next-hour weather is used, so the forecast does not depend on "
            "a perfect weather forecast."
        )

    # -------------------------------------------------------------
    # Household profile
    # -------------------------------------------------------------
    profile = service.profiles[
        service.profiles[HOUSE_COL] == house
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("City", city)

    for col, (source, label, spec) in zip(
        (c2, c3, c4),
        PROFILE_METRICS,
    ):
        col.metric(label, _fmt(profile[source], spec))

    with st.expander(
        "Household profile used by the model",
        expanded=False,
    ):
        table = pd.DataFrame(
            {
                house: [
                    profile[col] if pd.notna(profile[col]) else "n/a"
                    for col in PROFILE_TABLE_COLUMNS
                ]
            },
            index=PROFILE_TABLE_COLUMNS,
        ).astype(str)

        st.dataframe(table, width="stretch")

    # -------------------------------------------------------------
    # Weather inputs
    # -------------------------------------------------------------
    origin = pd.Timestamp(context["origin_time"])

    st.subheader("Current-hour conditions")

    st.caption(
        f"Origin reading: **{origin:%Y-%m-%d %H:%M}** · "
        f"forecast target: **{origin + pd.Timedelta(hours=1):%Y-%m-%d %H:%M}**. "
        "Defaults are the last known weather for this household; adjust them to "
        "ask a what-if question."
    )

    defaults = context["defaults"]
    weather_inputs: dict[str, float] = {}

    keys = list(INPUT_WEATHER_FEATURES)

    for row_start in range(0, len(keys), 4):
        cols = st.columns(4)

        for col, key in zip(cols, keys[row_start:row_start + 4]):
            label, lo, hi, step, fmt = WEATHER_SPECS[key]

            default = min(max(defaults[key], lo), hi)

            weather_inputs[key] = col.number_input(
                label,
                min_value=lo,
                max_value=hi,
                value=default,
                step=step,
                format=fmt,
                key=f"w_{key}",
            )

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    predict = st.button(
        "⚡ Predict next hour",
        type="primary",
        width="stretch",
    )

    if predict:
        try:
            result = service.predict(house, weather_inputs)
            st.session_state["result"] = result.to_dict()

        except Exception as exc:  # noqa: BLE001 - shown to the user
            st.error(f"Prediction failed: {exc}")
            st.stop()

    result = st.session_state.get("result")

    if result and result["house_id"] == house:

        # ---------------------------------------------------------
        # Prediction metrics
        # ---------------------------------------------------------
        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Predicted next-hour energy",
            f"{result['prediction_kwh']:.3f} kWh",
        )

        r2.metric(
            "90% prediction interval",
            f"{result['interval_low_kwh']:.2f} – "
            f"{result['interval_high_kwh']:.2f} kWh",
        )

        r3.metric(
            "Change vs last hour",
            f"{result['delta_kwh']:+.3f} kWh",
        )

        # ---------------------------------------------------------
        # Historical usage + forecast chart
        # ---------------------------------------------------------
        st.subheader("Last 72 hours and the forecast")

        history = pd.DataFrame(result["history"])
        history[TIME_COL] = pd.to_datetime(history["time"])

        history = history.set_index(TIME_COL)["usage_kwh"]

        # Rendered client-side with Vega-Lite: no PNG is streamed from the
        # server's /media route, which is where Community Cloud was failing.
        history_plot = pd.DataFrame(
            {
                "time": history.index,
                "usage_kwh": history.to_numpy(),
                "series": "actual usage",
            }
        )

        target = pd.Timestamp(result["target_time"])

        forecast_plot = pd.DataFrame(
            {
                "time": [target],
                "usage_kwh": [result["prediction_kwh"]],
                "low": [result["interval_low_kwh"]],
                "high": [result["interval_high_kwh"]],
                "series": "forecast (90% interval)",
            }
        )

        series_color = alt.Color(
            "series:N",
            title=None,
            scale=alt.Scale(
                domain=["actual usage", "forecast (90% interval)"],
                range=["#1f77b4", "#d62728"],
            ),
        )

        chart = (
            alt.Chart(history_plot)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("time:T", title="datetime"),
                y=alt.Y("usage_kwh:Q", title="kWh"),
                color=series_color,
                tooltip=[
                    alt.Tooltip("time:T", title="time"),
                    alt.Tooltip("usage_kwh:Q", title="kWh", format=".3f"),
                ],
            )
            + alt.Chart(forecast_plot)
            .mark_rule(strokeWidth=2)
            .encode(
                x=alt.X("time:T"),
                y=alt.Y("low:Q", title="kWh"),
                y2=alt.Y2("high:Q"),
                color=series_color,
                tooltip=[
                    alt.Tooltip("low:Q", title="interval low", format=".3f"),
                    alt.Tooltip("high:Q", title="interval high", format=".3f"),
                ],
            )
            + alt.Chart(forecast_plot)
            .mark_point(filled=True, size=70)
            .encode(
                x=alt.X("time:T"),
                y=alt.Y("usage_kwh:Q", title="kWh"),
                color=series_color,
                tooltip=[
                    alt.Tooltip("time:T", title="time"),
                    alt.Tooltip(
                        "usage_kwh:Q", title="forecast", format=".3f"
                    ),
                ],
            )
            + alt.Chart(pd.DataFrame({"time": [origin]}))
            .mark_rule(color="grey", strokeDash=[4, 4])
            .encode(x=alt.X("time:T"))
        ).configure_axis(grid=True, gridOpacity=0.3)

        st.altair_chart(chart, width="stretch")

        # ---------------------------------------------------------
        # Model information
        # ---------------------------------------------------------
        test_mae = model_info["test_mae"]

        test_mae_text = (
            f"{test_mae:.3f} kWh"
            if test_mae is not None
            else "n/a"
        )

        coverage = model_info["interval_val_coverage"]

        coverage_text = (
            f"{coverage:.0%} coverage"
            if coverage is not None
            else "coverage unavailable"
        )

        st.caption(
            f"Model: {model_info['name']} · "
            f"test MAE {test_mae_text} · "
            f"interval calibrated on validation residuals ({coverage_text})."
        )

    else:
        st.info(
            "Choose a household, adjust the weather if you like, then press "
            "**Predict next hour**."
        )

    # -------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------
    st.divider()

    st.caption(
        "Data: LUMS REWD-P (59 households) and Open-Meteo ERA5 (6 cities). "
        "Predictions are for research/demo purposes, not billing or grid "
        "operations."
    )
