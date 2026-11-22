"""Streamlit UI for BijliCast (shared by local/Community Cloud and HF Spaces).

The wrapper (:mod:`app.streamlit_app`) passes the repository root; this module
only knows about ``<root>/models`` and ``<root>/src``. That keeps the exact same
UI working inside a Hugging Face Space, where the app files live at the root.
"""
from __future__ import annotations

import base64
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


def _get_image_base64(path: Path) -> str:
    """Return base64 data URI for embedding local images into CSS."""
    if path.exists():
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            suffix = path.suffix.lower().replace(".", "")
            mime = "jpeg" if suffix in ("jpg", "jpeg") else ("webp" if suffix == "webp" else "png")
            return f"data:image/{mime};base64,{data}"
        except Exception:
            return ""
    return ""


def _apply_theme_styles(bg_uri: str) -> None:
    """Inject dark-blue, brown/clay, and whitish custom theme CSS with clearly visible header backdrop."""
    bg_css = (
        f"""
        background-color: #0b101b;
        background-image: linear-gradient(180deg, rgba(11, 16, 27, 0.05) 0%, rgba(11, 16, 27, 0.25) 240px, rgba(11, 16, 27, 0.75) 450px, #0b101b 600px), url('{bg_uri}');
        background-repeat: no-repeat;
        background-position: top center;
        background-size: 100% auto;
        """
        if bg_uri
        else "background-color: #0b101b;"
    )

    st.markdown(
        f"""
        <style>
        /* Base page styling with ambient top background */
        .stApp {{
            {bg_css}
            color: #f1f5f9;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: #0d1424 !important;
            border-right: 1px solid #1b2842;
        }}

        /* Clean header styling without a bordered box */
        .bc-header {{
            padding: 14px 0 24px 0;
            margin-bottom: 20px;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        .bc-hero-sub {{
            font-size: 11.5px;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #8da4c4;
            margin-bottom: 6px;
        }}
        .bc-hero-title {{
            font-size: 31px;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.25;
            margin: 0 0 10px 0;
            text-shadow: 0 2px 14px rgba(0, 0, 0, 0.95), 0 0 30px rgba(0, 0, 0, 0.8);
        }}
        .bc-hero-desc {{
            font-size: 14.5px;
            color: #f1f5f9;
            margin: 0 0 18px 0;
            max-width: 860px;
            line-height: 1.5;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.95);
        }}

        /* Badges / capsules */
        .bc-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .bc-badge {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.20);
            padding: 4px 13px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #e2e8f0;
            text-transform: uppercase;
            backdrop-filter: blur(4px);
        }}
        .bc-badge-clay {{
            background: rgba(207, 100, 57, 0.20);
            border-color: #cf6439;
            color: #f8af90;
        }}

        /* Custom Cards */
        .bc-card {{
            background-color: #131b2e;
            border: 1px solid #1e2c47;
            border-radius: 14px;
            padding: 22px 24px;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.30);
        }}
        .bc-card h3 {{
            color: #ffffff;
            font-size: 17px;
            margin-top: 0;
            margin-bottom: 8px;
        }}
        .bc-card p {{
            color: #94a3b8;
            font-size: 13.5px;
            margin-bottom: 0;
            line-height: 1.6;
        }}

        /* Metric boxes */
        .bc-metric-box {{
            background-color: #131b2e;
            border: 1px solid #1f2c47;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        }}
        .bc-metric-label {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #8da4c4;
            margin-bottom: 8px;
        }}
        .bc-metric-val {{
            font-size: 28px;
            font-weight: 800;
            color: #cf6439;
            line-height: 1.1;
            margin-bottom: 4px;
        }}
        .bc-metric-sub {{
            font-size: 12px;
            color: #64748b;
        }}

        /* Steppers & inputs */
        div[data-baseweb="input"] {{
            background-color: #162035 !important;
            border-color: #243555 !important;
            border-radius: 10px !important;
        }}
        div[data-baseweb="input"] input {{
            color: #f1f5f9 !important;
            font-weight: 600 !important;
        }}
        div[data-baseweb="select"] > div {{
            background-color: #162035 !important;
            border-color: #243555 !important;
            border-radius: 10px !important;
            color: #ffffff !important;
        }}

        /* Primary Button (Clay / Brown) */
        button[kind="primary"] {{
            background-color: #cf6439 !important;
            border-color: #cf6439 !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            padding: 10px 22px !important;
            border-radius: 11px !important;
            box-shadow: 0 6px 18px rgba(207, 100, 57, 0.35) !important;
            transition: all 0.2s ease !important;
        }}
        button[kind="primary"]:hover {{
            background-color: #df7449 !important;
            border-color: #df7449 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 22px rgba(207, 100, 57, 0.45) !important;
        }}

        /* Expander styling */
        div[data-testid="stExpander"] {{
            background-color: #131b2e !important;
            border: 1px solid #1f2c47 !important;
            border-radius: 12px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home_page(root: Path, service) -> None:
    """Render the BijliCast home page without a boxed hero container."""
    model_info = service.model_info()

    # Open header seamlessly on top of background image
    st.markdown(
        """
        <div class="bc-header">
            <div class="bc-hero-sub">NATIONAL RESIDENTIAL ENERGY FORECASTING</div>
            <div class="bc-hero-title">BijliCast : Weather-Aware Household Electricity Consumption Forecasting in Pakistan</div>
            <div class="bc-hero-desc">
                Next-hour residential electricity demand forecasting for Pakistan · LUMS REWD-P smart-meter telemetry
                across 59 households and 6 major cities.
            </div>
            <div class="bc-badges">
                <span class="bc-badge bc-badge-clay">HistGradientBoosting</span>
                <span class="bc-badge">LUMS REWD-P DATASET</span>
                <span class="bc-badge">59 HOUSEHOLDS</span>
                <span class="bc-badge">6 MAJOR CITIES</span>
                <span class="bc-badge">ZERO NEXT-HOUR WEATHER DEPENDENCY</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Problem & Solution Columns
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="bc-card">
                <h3>⚠️ The Energy Challenge in Pakistan</h3>
                <p>
                    Pakistan's residential sector experiences severe grid stress, frequent unannounced load-shedding,
                    and steep peak-hour tariff multipliers (17:00 – 22:00). Extreme seasonal temperatures across
                    cities like Lahore, Multan, and Karachi cause erratic AC cooling surges that strain sub-stations
                    and inflate consumer bills without warning.
                </p>
                <br>
                <p>
                    Existing forecasting solutions typically fail because they require accurate future weather forecasts
                    or depend on high-cost industrial SCADA systems unavailable to ordinary households.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="bc-card">
                <h3>💡 How BijliCast Solves It</h3>
                <p>
                    BijliCast delivers ultra-low-latency next-hour consumption predictions using <b>only current-hour
                    ambient weather</b> and causal 168-hour (7-day) historical usage sequences. Because it uses no next-hour
                    weather predictions, the model is resilient against imperfect meteorological forecasts.
                </p>
                <br>
                <p>
                    Empirical 90% prediction intervals calibrated on validation residuals give consumers and grid
                    operators actionable bounds to optimize battery storage (UPS/solar), shift appliance usage, and
                    avoid expensive peak-slab tariff escalations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Model Highlights Row
    st.markdown("### Model & System Overview")
    m1, m2, m3, m4 = st.columns(4)

    test_mae = model_info.get("test_mae")
    test_r2 = model_info.get("test_r2")
    n_rows = model_info.get("n_train_rows", 408048)
    coverage = model_info.get("interval_val_coverage", 0.90)

    m1.markdown(
        f"""
        <div class="bc-metric-box">
            <div class="bc-metric-label">Model Architecture</div>
            <div class="bc-metric-val" style="font-size:20px; color:#ffffff;">{model_info.get('name', 'HistGradientBoosting')}</div>
            <div class="bc-metric-sub">Trained on {n_rows:,} hourly rows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m2.markdown(
        f"""
        <div class="bc-metric-box">
            <div class="bc-metric-label">Held-out Test MAE</div>
            <div class="bc-metric-val">{test_mae:.3f} <span style="font-size:16px;">kWh</span></div>
            <div class="bc-metric-sub">High precision per household</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m3.markdown(
        f"""
        <div class="bc-metric-box">
            <div class="bc-metric-label">Held-out Test R²</div>
            <div class="bc-metric-val">{test_r2:.3f}</div>
            <div class="bc-metric-sub">Strong variance explanation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m4.markdown(
        f"""
        <div class="bc-metric-box">
            <div class="bc-metric-label">Interval Coverage</div>
            <div class="bc-metric-val">{coverage:.0%}</div>
            <div class="bc-metric-sub">90% calibrated validation band</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation Call to Action
    st.markdown(
        """
        <div style="background:#131b2e; border:1px solid #cf6439; border-radius:14px; padding:22px; text-align:center;">
            <h4 style="margin:0 0 8px 0; color:#ffffff;">Ready to explore household energy forecasts?</h4>
            <p style="color:#94a3b8; margin:0 0 16px 0; font-size:14px;">
                Select any of the 59 monitored households, adjust ambient weather parameters for what-if scenarios, and inspect 3 interactive analytical charts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta = st.button("⚡ Go to Forecast & Prediction Dashboard →", type="primary", use_container_width=True)
    if cta:
        st.session_state["nav_page"] = "Predict"
        st.rerun()


def render_predict_page(service) -> None:
    """Render the BijliCast prediction dashboard with the 3 requested charts."""
    model_info = service.model_info()
    houses = service.houses()
    by_house = {h["house_id"]: h for h in houses}

    # Retrieve selected household from sidebar (managed in run())
    house = st.session_state.get("selected_house", houses[0]["house_id"])
    context = by_house[house]
    city = context["city"]
    origin = pd.Timestamp(context["origin_time"])
    target = origin + pd.Timedelta(hours=1)

    # Top Header for Predict (no boxed border)
    st.markdown(
        f"""
        <div class="bc-header">
            <div class="bc-hero-sub">MAKE A PREDICTION</div>
            <div class="bc-hero-title">Next hour for {house}, {city}</div>
            <div class="bc-hero-desc">
                Origin {origin:%d %b %Y, %H:%M} → target {target:%d %b %Y, %H:%M}. Adjust the weather inputs below to ask a what-if question.
            </div>
            <div class="bc-badges">
                <span class="bc-badge bc-badge-clay">{model_info['name']}</span>
                <span class="bc-badge">90% INTERVAL</span>
                <span class="bc-badge">8 WEATHER INPUTS</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Weather Inputs
    st.subheader("Current-hour conditions")
    st.caption("Defaults are the last known weather for this household. Adjust values to simulate weather scenarios.")

    defaults = context["defaults"]
    weather_inputs: dict[str, float] = {}
    keys = list(INPUT_WEATHER_FEATURES)

    for row_start in range(0, len(keys), 4):
        cols = st.columns(4)
        for col, key in zip(cols, keys[row_start : row_start + 4]):
            label, lo, hi, step, fmt = WEATHER_SPECS[key]
            default = min(max(defaults[key], lo), hi)
            weather_inputs[key] = col.number_input(
                label,
                min_value=lo,
                max_value=hi,
                value=default,
                step=step,
                format=fmt,
                key=f"w_{house}_{key}",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡ Predict next hour", type="primary", use_container_width=True)

    if predict_clicked:
        try:
            result = service.predict(house, weather_inputs)
            st.session_state["result"] = result.to_dict()
            st.session_state["result_house"] = house
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            st.stop()

    result = st.session_state.get("result")
    result_house = st.session_state.get("result_house")

    # Forecast and charts stay hidden until user clicks predict next hour for this house
    if result and result_house == house:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div>
                    <h2 style="margin:0; font-size:22px; color:#ffffff;">Forecast</h2>
                    <span style="color:#94a3b8; font-size:13px;">{house} · {city} · target {target:%d %b %H:%M}</span>
                </div>
                <span class="bc-badge bc-badge-clay">TYPICAL VS 72 H MEAN</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        history_df = pd.DataFrame(result["history"])
        history_df[TIME_COL] = pd.to_datetime(history_df["time"])
        mean_72h = float(history_df["usage_kwh"].mean()) if not history_df.empty else 0.585

        # 3 Metric Cards
        m1, m2, m3 = st.columns(3)

        m1.markdown(
            f"""
            <div class="bc-metric-box">
                <div class="bc-metric-label">PREDICTED NEXT-HOUR ENERGY</div>
                <div class="bc-metric-val">{result['prediction_kwh']:.3f} <span style="font-size:16px; color:#cbd5e1;">kWh</span></div>
                <div class="bc-metric-sub">72 h mean {mean_72h:.3f} kWh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m2.markdown(
            f"""
            <div class="bc-metric-box">
                <div class="bc-metric-label">90% PREDICTION INTERVAL</div>
                <div class="bc-metric-val">{result['interval_low_kwh']:.2f} – {result['interval_high_kwh']:.2f} <span style="font-size:16px; color:#cbd5e1;">kWh</span></div>
                <div class="bc-metric-sub">Calibrated on validation residuals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m3.markdown(
            f"""
            <div class="bc-metric-box">
                <div class="bc-metric-label">CHANGE VS LAST HOUR</div>
                <div class="bc-metric-val" style="color:{'#cf6439' if result['delta_kwh'] >= 0 else '#5b8db8'};">{result['delta_kwh']:+.3f} <span style="font-size:16px; color:#cbd5e1;">kWh</span></div>
                <div class="bc-metric-sub">Last reading {result['last_hour_kwh']:.3f} kWh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # CHART 1: Last 72 hours and the forecast
        # -------------------------------------------------------------
        st.markdown(
            """
            <h3 style="margin:0 0 2px 0; color:#ffffff; font-size:18px;">Last 72 hours and the forecast</h3>
            <p style="margin:0 0 14px 0; color:#8da4c4; font-size:13px;">Hover the line for hourly values · dashed clay line is the 72 h mean</p>
            """,
            unsafe_allow_html=True,
        )

        history_plot = pd.DataFrame(
            {
                "time": history_df[TIME_COL],
                "usage_kwh": history_df["usage_kwh"],
                "series": "actual usage",
            }
        )

        forecast_plot = pd.DataFrame(
            {
                "time": [target],
                "usage_kwh": [result["prediction_kwh"]],
                "low": [result["interval_low_kwh"]],
                "high": [result["interval_high_kwh"]],
                "label": [f"{result['prediction_kwh']:.2f}"],
                "series": "forecast (90% interval)",
            }
        )

        mean_line_df = pd.DataFrame(
            {
                "time": [history_plot["time"].min(), target],
                "mean_val": [mean_72h, mean_72h],
            }
        )

        actual_line = (
            alt.Chart(history_plot)
            .mark_line(strokeWidth=2, color="#5b8db8")
            .encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%d %b", gridColor="#1e2b45", labelColor="#94a3b8")),
                y=alt.Y("usage_kwh:Q", title="kWh", axis=alt.Axis(gridColor="#1e2b45", labelColor="#94a3b8", titleColor="#cbd5e1")),
                tooltip=[
                    alt.Tooltip("time:T", title="time", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("usage_kwh:Q", title="kWh", format=".3f"),
                ],
            )
        )

        mean_line = (
            alt.Chart(mean_line_df)
            .mark_line(strokeDash=[4, 4], color="#cf6439", strokeWidth=1.5)
            .encode(x=alt.X("time:T"), y=alt.Y("mean_val:Q"))
        )

        interval_rule = (
            alt.Chart(forecast_plot)
            .mark_rule(strokeWidth=3, color="#cf6439")
            .encode(
                x=alt.X("time:T"),
                y=alt.Y("low:Q"),
                y2=alt.Y2("high:Q"),
                tooltip=[
                    alt.Tooltip("low:Q", title="90% low", format=".3f"),
                    alt.Tooltip("high:Q", title="90% high", format=".3f"),
                ],
            )
        )

        forecast_point = (
            alt.Chart(forecast_plot)
            .mark_point(filled=True, size=90, color="#cf6439")
            .encode(
                x=alt.X("time:T"),
                y=alt.Y("usage_kwh:Q"),
                tooltip=[
                    alt.Tooltip("time:T", title="target time", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("usage_kwh:Q", title="forecast kWh", format=".3f"),
                ],
            )
        )

        forecast_text = (
            alt.Chart(forecast_plot)
            .mark_text(dx=18, dy=-2, color="#cf6439", fontSize=13, fontWeight="bold")
            .encode(x=alt.X("time:T"), y=alt.Y("usage_kwh:Q"), text=alt.Text("label:N"))
        )

        chart1 = (
            (actual_line + mean_line + interval_rule + forecast_point + forecast_text)
            .properties(height=320)
            .configure_view(strokeOpacity=0)
            .configure(background="transparent")
        )

        st.altair_chart(chart1, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # CHARTS 2 & 3: Typical day and Daily totals (Two columns)
        # -------------------------------------------------------------
        ch_col1, ch_col2 = st.columns(2)

        raw_history = service._history_by_house[house]

        with ch_col1:
            st.markdown(
                """
                <h3 style="margin:0 0 2px 0; color:#ffffff; font-size:17px;">Typical day</h3>
                <p style="margin:0 0 12px 0; color:#8da4c4; font-size:12.5px;">Mean kWh by hour of day · peak hours in clay</p>
                """,
                unsafe_allow_html=True,
            )

            hourly_df = pd.DataFrame(
                {"hour": raw_history.index.hour, "usage_kwh": raw_history.to_numpy()}
            )
            typical_day = hourly_df.groupby("hour", as_index=False)["usage_kwh"].mean()

            # Pakistan peak tariff window: 17:00 – 22:00
            typical_day["is_peak"] = typical_day["hour"].apply(lambda h: 17 <= h <= 22)
            typical_day["color_cat"] = typical_day["is_peak"].map(
                {True: "Peak (17-22h)", False: "Regular"}
            )

            chart2 = (
                alt.Chart(typical_day)
                .mark_bar(cornerRadius=3)
                .encode(
                    x=alt.X(
                        "hour:O",
                        title="hour of day",
                        axis=alt.Axis(labelAngle=0, labelColor="#94a3b8", titleColor="#cbd5e1"),
                    ),
                    y=alt.Y(
                        "usage_kwh:Q",
                        title="mean kWh",
                        axis=alt.Axis(gridColor="#1e2b45", labelColor="#94a3b8", titleColor="#cbd5e1"),
                    ),
                    color=alt.Color(
                        "color_cat:N",
                        scale=alt.Scale(
                            domain=["Regular", "Peak (17-22h)"],
                            range=["#5b8db8", "#cf6439"],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("hour:O", title="hour of day"),
                        alt.Tooltip("usage_kwh:Q", title="mean kWh", format=".3f"),
                        alt.Tooltip("color_cat:N", title="tariff status"),
                    ],
                )
                .properties(height=260)
                .configure_view(strokeOpacity=0)
                .configure(background="transparent")
            )
            st.altair_chart(chart2, use_container_width=True)

        with ch_col2:
            st.markdown(
                """
                <h3 style="margin:0 0 2px 0; color:#ffffff; font-size:17px;">Daily totals</h3>
                <p style="margin:0 0 12px 0; color:#8da4c4; font-size:12.5px;">Last 7 complete days · latest day in clay</p>
                """,
                unsafe_allow_html=True,
            )

            daily_df = pd.DataFrame(
                {"date": raw_history.index.date, "usage_kwh": raw_history.to_numpy()}
            )
            daily_totals = daily_df.groupby("date", as_index=False)["usage_kwh"].sum()
            daily_totals = daily_totals.tail(7).copy()
            daily_totals["day_label"] = pd.to_datetime(daily_totals["date"]).dt.strftime("%d %b")

            latest_date = daily_totals["date"].iloc[-1]
            daily_totals["is_latest"] = daily_totals["date"] == latest_date
            daily_totals["color_cat"] = daily_totals["is_latest"].map(
                {True: "Latest day", False: "Previous days"}
            )

            chart3 = (
                alt.Chart(daily_totals)
                .mark_bar(cornerRadius=3)
                .encode(
                    x=alt.X(
                        "day_label:O",
                        title=None,
                        sort=None,
                        axis=alt.Axis(labelAngle=0, labelColor="#94a3b8"),
                    ),
                    y=alt.Y(
                        "usage_kwh:Q",
                        title="kWh / day",
                        axis=alt.Axis(gridColor="#1e2b45", labelColor="#94a3b8", titleColor="#cbd5e1"),
                    ),
                    color=alt.Color(
                        "color_cat:N",
                        scale=alt.Scale(
                            domain=["Previous days", "Latest day"],
                            range=["#5b8db8", "#cf6439"],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("day_label:N", title="date"),
                        alt.Tooltip("usage_kwh:Q", title="total kWh", format=".2f"),
                        alt.Tooltip("color_cat:N", title="period"),
                    ],
                )
                .properties(height=260)
                .configure_view(strokeOpacity=0)
                .configure(background="transparent")
            )
            st.altair_chart(chart3, use_container_width=True)

        st.caption(
            f"{model_info['name']} · test MAE {model_info.get('test_mae', 0.250):.3f} kWh · "
            f"90% interval covers {model_info.get('interval_val_coverage', 0.90):.0%} of validation hours."
        )

        # Household profile expander
        profile = service.profiles[service.profiles[HOUSE_COL] == house].iloc[0]
        with st.expander("Household profile used by the model", expanded=False):
            table = pd.DataFrame(
                {
                    house: [
                        profile[col] if pd.notna(profile[col]) else "n/a"
                        for col in PROFILE_TABLE_COLUMNS
                    ]
                },
                index=PROFILE_TABLE_COLUMNS,
            ).astype(str)
            st.dataframe(table, use_container_width=True)

    else:
        st.markdown(
            f"""
            <div style="background:#131b2e; border:1px solid #1f2c47; border-radius:14px; padding:24px; text-align:center; margin-top:20px;">
                <p style="color:#cbd5e1; font-size:15px; margin:0 0 6px 0;">
                    ⚡ Ready to forecast next-hour electricity consumption for <b>{house}</b>.
                </p>
                <p style="color:#64748b; font-size:13px; margin:0;">
                    Click the <b>Predict next hour</b> button above to generate the prediction metrics and the 3 analytical charts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def run(root: str | Path | None = None) -> None:
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()

    st.set_page_config(
        page_title="BijliCast - next-hour electricity forecast",
        page_icon="⚡",
        layout="wide",
    )

    # Resolve background image URI
    bg_path = root / "BijliCast bg.png"
    if not bg_path.exists():
        bg_path = root / "app" / "static" / "hero.webp"
    bg_uri = _get_image_base64(bg_path)
    _apply_theme_styles(bg_uri)

    try:
        service = get_service(str(root))
    except BijliCastError as exc:
        st.error(f"Deployment bundle error: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the deployment bundle: {exc}")
        st.stop()

    model_info = service.model_info()
    houses = service.houses()
    by_house = {h["house_id"]: h for h in houses}

    # -------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------
    with st.sidebar:
        # Small SVG Brand Logo
        logo_path = root / "logos" / "logo-dark.svg"
        if not logo_path.exists():
            logo_path = root / "app" / "static" / "logo-dark.svg"

        if logo_path.exists():
            st.image(str(logo_path), width=120)
        else:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <span style="font-size:20px;">⚡</span>
                    <span style="font-size:18px; font-weight:800; color:#ffffff;">BijliCast</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Navigation Options
        nav_options = ["Home", "Predict"]
        default_index = 0 if st.session_state.get("nav_page", "Home") == "Home" else 1

        selected_page = st.radio(
            "Navigation",
            nav_options,
            index=default_index,
            key="nav_radio",
            label_visibility="collapsed",
        )
        st.session_state["nav_page"] = selected_page

        st.divider()

        # When on Predict page, show Household selector and Model Info
        if selected_page == "Predict":
            st.markdown(
                "<span style='font-size:11px; font-weight:700; letter-spacing:0.1em; color:#8da4c4;'>HOUSEHOLD</span>",
                unsafe_allow_html=True,
            )
            labels = {
                f"{h['house_id']} · {h['city']}": h["house_id"]
                for h in houses
            }
            choice = st.selectbox(
                "Select a household",
                list(labels),
                index=0,
                label_visibility="collapsed",
            )
            selected_house_id = labels[choice]
            st.session_state["selected_house"] = selected_house_id

            st.divider()

            st.markdown(
                "<span style='font-size:11px; font-weight:700; letter-spacing:0.1em; color:#8da4c4;'>MODEL</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{model_info['name']}**")

            test_mae = model_info.get("test_mae")
            if test_mae is not None:
                st.write(f"Test MAE · **{test_mae:.3f} kWh**")

            test_r2 = model_info.get("test_r2")
            if test_r2 is not None:
                st.write(f"Test R² · **{test_r2:.3f}**")

            coverage = model_info.get("interval_val_coverage")
            coverage_text = f"{coverage:.0%}" if coverage is not None else "90%"
            st.caption(f"{coverage_text} interval covers {coverage_text} of validation hours.")
            st.caption(f"Trained on {model_info.get('n_train_rows', 408048):,} hourly rows.")

            st.divider()
            st.caption(
                "Uses current-hour weather and the last 168 h of usage. "
                "No next-hour weather is needed."
            )
        else:
            # On Home Page
            st.markdown(
                "<span style='font-size:11px; font-weight:700; letter-spacing:0.1em; color:#8da4c4;'>SUMMARY</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                "BijliCast predicts next-hour household electricity demand across Pakistan without relying on future weather forecasts."
            )

    # -------------------------------------------------------------
    # Main Page Dispatch
    # -------------------------------------------------------------
    if selected_page == "Home":
        render_home_page(root, service)
    else:
        render_predict_page(service)

    # Footer
    st.divider()
    st.caption(
        "Data: LUMS REWD-P (59 households) and Open-Meteo ERA5 (6 cities). "
        "Predictions are for research and operational planning purposes."
    )


if __name__ == "__main__":
    run()
