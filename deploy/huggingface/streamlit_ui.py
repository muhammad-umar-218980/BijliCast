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
    """Inject premium SaaS aesthetic CSS (Navy #13213F, Clay #C2643F, Off-white #F7F8FA)."""
    hero_bg = (
        f"""
        background-color: #0B101B;
        background-image: linear-gradient(180deg, rgba(11, 16, 27, 0.04) 0%, rgba(11, 16, 27, 0.18) 220px, rgba(11, 16, 27, 0.70) 450px, #0B101B 650px), url('{bg_uri}');
        background-repeat: no-repeat;
        background-position: top center;
        background-size: 100% auto;
        """
        if bg_uri
        else "background-color: #0B101B;"
    )

    st.markdown(
        f"""
        <style>
        /* Base page styling */
        .stApp {{
            {hero_bg}
            color: #F7F8FA;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}

        /* Reduce top padding above the main heading */
        .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 2.5rem !important;
        }}

        /* Sidebar styling and small logo */
        section[data-testid="stSidebar"] {{
            background-color: #0D1424 !important;
            border-right: 1px solid #1E2C4B;
        }}
        [data-testid="stSidebar"] img {{
            max-width: 90px !important;
            width: 90px !important;
            height: auto !important;
        }}

        /* Section Eyebrow Labels */
        .saas-eyebrow {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #C2643F;
            margin-bottom: 8px;
        }}

        /* Section Headings */
        .saas-heading {{
            font-size: 28px;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.25;
            margin: 0 0 12px 0;
            letter-spacing: -0.01em;
        }}

        .saas-lead {{
            font-size: 15px;
            color: #94A3B8;
            line-height: 1.6;
            margin-bottom: 24px;
        }}

        /* Hero Container */
        .saas-hero-container {{
            padding: 2px 0 20px 0;
            margin-bottom: 24px;
        }}
        .saas-hero-title {{
            font-size: 34px;
            font-weight: 850;
            color: #FFFFFF;
            line-height: 1.2;
            letter-spacing: -0.015em;
            margin: 0 0 16px 0;
            text-shadow: 0 2px 14px rgba(0, 0, 0, 0.95), 0 0 30px rgba(0, 0, 0, 0.85);
        }}
        .saas-hero-desc {{
            font-size: 16px;
            color: #E2E8F0;
            line-height: 1.6;
            margin: 0 0 24px 0;
            max-width: 650px;
            text-shadow: 0 1px 8px rgba(0, 0, 0, 0.85);
        }}

        /* Illustrative Preview Card */
        .saas-preview-card {{
            background: #13213F;
            border: 1px solid #243555;
            border-radius: 16px;
            padding: 22px 24px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
        }}
        .saas-preview-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1E2C4B;
            padding-bottom: 12px;
            margin-bottom: 14px;
        }}
        .saas-preview-title {{
            font-size: 13px;
            font-weight: 700;
            color: #E2E8F0;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .saas-preview-badge {{
            background: rgba(194, 100, 63, 0.22);
            color: #F8AF90;
            border: 1px solid #C2643F;
            font-size: 10px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 999px;
            letter-spacing: 0.06em;
        }}

        /* KPI / Stat Box */
        .saas-kpi-card {{
            background: #13213F;
            border: 1px solid #1E2C4B;
            border-radius: 14px;
            padding: 22px 20px;
            text-align: left;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .saas-kpi-val {{
            font-size: 36px;
            font-weight: 850;
            color: #C2643F;
            line-height: 1.1;
            margin-bottom: 6px;
            letter-spacing: -0.02em;
        }}
        .saas-kpi-label {{
            font-size: 13.5px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 4px;
        }}
        .saas-kpi-sub {{
            font-size: 12px;
            color: #667085;
            line-height: 1.4;
        }}

        /* Editorial 2-Column Cards */
        .saas-editorial-card {{
            background: #13213F;
            border: 1px solid #1E2C4B;
            border-radius: 16px;
            padding: 28px 28px;
            height: 100%;
        }}
        .saas-editorial-card h4 {{
            font-size: 18px;
            font-weight: 750;
            color: #FFFFFF;
            margin-top: 0;
            margin-bottom: 12px;
        }}
        .saas-editorial-card p {{
            font-size: 14px;
            color: #94A3B8;
            line-height: 1.65;
            margin-bottom: 14px;
        }}

        /* How it Works Step Card */
        .saas-step-card {{
            background: #13213F;
            border: 1px solid #1E2C4B;
            border-radius: 14px;
            padding: 24px 22px;
            height: 100%;
            position: relative;
        }}
        .saas-step-num {{
            display: inline-block;
            font-size: 12px;
            font-weight: 800;
            color: #C2643F;
            background: rgba(194, 100, 63, 0.15);
            padding: 3px 10px;
            border-radius: 8px;
            margin-bottom: 12px;
            letter-spacing: 0.08em;
        }}
        .saas-step-title {{
            font-size: 16px;
            font-weight: 750;
            color: #FFFFFF;
            margin-bottom: 8px;
        }}
        .saas-step-desc {{
            font-size: 13px;
            color: #94A3B8;
            line-height: 1.6;
            margin: 0;
        }}

        /* Final CTA Section */
        .saas-final-cta {{
            background: linear-gradient(135deg, #13213F 0%, #1A2B50 100%);
            border: 1px solid #243555;
            border-radius: 18px;
            padding: 44px 36px;
            text-align: center;
            margin: 40px 0 28px 0;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.35);
        }}
        .saas-final-cta h2 {{
            font-size: 34px;
            font-weight: 850;
            color: #FFFFFF;
            margin: 0 0 12px 0;
            letter-spacing: -0.01em;
        }}
        .saas-final-cta p {{
            font-size: 15.5px;
            color: #CBD5E1;
            max-width: 680px;
            margin: 0 auto 24px auto;
            line-height: 1.6;
        }}

        /* Footer */
        .saas-footer {{
            border-top: 1px solid #1E2C4B;
            padding: 32px 0 16px 0;
            margin-top: 36px;
            color: #667085;
            font-size: 12.5px;
            line-height: 1.6;
        }}
        .saas-footer a {{
            color: #C2643F;
            text-decoration: none;
            font-weight: 600;
        }}
        .saas-footer a:hover {{
            text-decoration: underline;
        }}

        /* Buttons */
        button[kind="primary"] {{
            background-color: #C2643F !important;
            border-color: #C2643F !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 14.5px !important;
            padding: 10px 22px !important;
            border-radius: 10px !important;
            box-shadow: 0 6px 16px rgba(194, 100, 63, 0.35) !important;
            transition: all 0.2s ease !important;
        }}
        button[kind="primary"]:hover {{
            background-color: #D6724B !important;
            border-color: #D6724B !important;
            transform: translateY(-1px) !important;
        }}
        button[kind="secondary"] {{
            background-color: transparent !important;
            border: 1px solid #2E4166 !important;
            color: #E2E8F0 !important;
            font-weight: 600 !important;
            font-size: 14.5px !important;
            padding: 10px 20px !important;
            border-radius: 10px !important;
        }}
        button[kind="secondary"]:hover {{
            border-color: #C2643F !important;
            color: #FFFFFF !important;
        }}

        /* Prediction metric boxes */
        .bc-metric-box {{
            background-color: #13213F;
            border: 1px solid #1E2C4B;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        }}
        .bc-metric-label {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #8DA4C4;
            margin-bottom: 8px;
        }}
        .bc-metric-val {{
            font-size: 28px;
            font-weight: 800;
            color: #C2643F;
            line-height: 1.1;
            margin-bottom: 4px;
        }}
        .bc-metric-sub {{
            font-size: 12px;
            color: #64748B;
        }}

        /* Input styling */
        div[data-baseweb="input"] {{
            background-color: #162035 !important;
            border-color: #243555 !important;
            border-radius: 10px !important;
        }}
        div[data-baseweb="input"] input {{
            color: #F1F5F9 !important;
            font-weight: 600 !important;
        }}
        div[data-baseweb="select"] > div {{
            background-color: #162035 !important;
            border-color: #243555 !important;
            border-radius: 10px !important;
            color: #FFFFFF !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero_sparkline(service) -> alt.Chart:
    """Generate an illustrative Altair energy load curve for the hero section."""
    raw_hist = service._history_by_house.get("House#1")
    if raw_hist is None or len(raw_hist) < 24:
        return None

    tail = raw_hist.tail(24)
    df = pd.DataFrame({
        "time": tail.index,
        "kwh": tail.values,
    })

    line = (
        alt.Chart(df)
        .mark_area(
            line={"color": "#C2643F", "width": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(194, 100, 63, 0.35)", offset=0),
                    alt.GradientStop(color="rgba(19, 33, 63, 0.0)", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
        )
        .encode(
            x=alt.X("time:T", axis=alt.Axis(labels=False, ticks=False, title=None, grid=False)),
            y=alt.Y("kwh:Q", axis=alt.Axis(labels=False, ticks=False, title=None, grid=False)),
        )
        .properties(height=110)
        .configure_view(strokeOpacity=0)
        .configure(background="transparent")
    )
    return line


def render_home_page(root: Path, service) -> None:
    """Render the redesigned 8-section SaaS homepage."""
    model_info = service.model_info()

    # -----------------------------------------------------------------------
    # SECTION 1: HERO
    # -----------------------------------------------------------------------
    st.markdown('<div class="saas-hero-container">', unsafe_allow_html=True)
    st.markdown('<div class="saas-eyebrow">RESIDENTIAL ENERGY INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="saas-hero-title">BijliCast - Weather-Aware Household Electricity Consumption Forecasting in Pakistan</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="saas-hero-desc">
            Next-hour residential electricity demand forecasting across Pakistan.
            Powered by LUMS REWD-P smart-meter telemetry and real-time weather analytics—built to
            anticipate peak tariffs, mitigate load shedding uncertainty, and stabilize household power
            budgets without relying on future weather forecasts.
        </p>
        """,
        unsafe_allow_html=True,
    )

    cta_c1, cta_c2, _ = st.columns([1.1, 1.1, 2.2], gap="small")
    with cta_c1:
        if st.button("Explore Predictions →", type="primary", use_container_width=True, key="hero_cta_pred"):
            st.session_state["nav_page"] = "Predict"
            st.rerun()
    with cta_c2:
        st.markdown(
            """
            <a href="#how-it-works" style="display:inline-block; width:100%; text-align:center; padding:10px 18px; border:1px solid #2E4166; border-radius:10px; color:#E2E8F0; text-decoration:none; font-weight:600; font-size:14.5px; background:rgba(255,255,255,0.03);">
                Explore the Technology ↓
            </a>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # SECTION 2: PROJECT STATISTICS
    # -----------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="saas-eyebrow">EMPIRICAL SCALE</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="saas-heading">High-Resolution Telemetry at Grid Scale</h2>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        st.markdown(
            """
            <div class="saas-kpi-card">
                <div class="saas-kpi-val">408,048</div>
                <div class="saas-kpi-label">Hourly Records</div>
                <div class="saas-kpi-sub">Cleaned and temporally validated smart-meter rows across multi-year telemetry.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            """
            <div class="saas-kpi-card">
                <div class="saas-kpi-val">59</div>
                <div class="saas-kpi-label">Households</div>
                <div class="saas-kpi-sub">Granular residential profiles capturing appliance counts, area, and occupant demographics.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            """
            <div class="saas-kpi-card">
                <div class="saas-kpi-val">6</div>
                <div class="saas-kpi-label">Pakistani Cities</div>
                <div class="saas-kpi-sub">Lahore, Karachi, Islamabad, Faisalabad, Multan, and Peshawar covering diverse climate zones.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            """
            <div class="saas-kpi-card">
                <div class="saas-kpi-val">90%</div>
                <div class="saas-kpi-label">Interval Coverage</div>
                <div class="saas-kpi-sub">Calibrated empirical confidence bounds covering 90% of unseen validation hours.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # SECTION 3: ENERGY CHALLENGE
    # -----------------------------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="saas-eyebrow">THE CRISIS CONTEXT</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="saas-heading">Pakistan’s Residential Energy Challenge</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="saas-lead">Why macro-level utility forecasting fails to protect households from tariff spikes and grid instability.</p>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2, gap="large")
    with c_left:
        st.markdown(
            """
            <div class="saas-editorial-card">
                <div class="saas-eyebrow">CLIMATE & PEAK TARIFFS</div>
                <h4>Extreme Weather & Tariff Multipliers</h4>
                <p>
                    Residential electricity consumers in Pakistan face aggressive Time-of-Use (ToU) tariff schedules.
                    Between <b>17:00 and 22:00</b>, per-unit billing rates surge, penalizing unmanaged appliance operation.
                </p>
                <p>
                    Simultaneously, summer temperatures exceeding 45°C across the Indus plains drive non-linear cooling
                    surges from air conditioners. These extreme spikes frequently overload local distribution transformers,
                    causing unscheduled feeder tripping and prolonged load shedding.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown(
            """
            <div class="saas-editorial-card">
                <div class="saas-eyebrow">THE VISIBILITY GAP</div>
                <h4>The Household Telemetry Deficit</h4>
                <p>
                    Traditional energy utility analytics are aggregated at the grid substation level. Individual
                    households receive only monthly aggregate bills, leaving them blind to intra-day consumption patterns.
                </p>
                <p>
                    Furthermore, standard machine learning forecasting literature assumes access to perfect <i>next-hour</i>
                    weather forecasts. In practice, regional meteorological predictions are prone to lag and error.
                    BijliCast solves this by relying strictly on <b>current-hour ambient weather</b> and <b>causal 168-hour usage sequences</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # SECTION 4: HOW BIJLICAST WORKS
    # -----------------------------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
    st.markdown('<div class="saas-eyebrow">METHODOLOGY PIPELINE</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="saas-heading">How BijliCast Works</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="saas-lead">Three sequential, causal processing stages engineered for sub-10ms inference latency without forward weather dependence.</p>',
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        st.markdown(
            """
            <div class="saas-step-card">
                <span class="saas-step-num">STAGE 01</span>
                <div class="saas-step-title">Current Weather Ingestion</div>
                <p class="saas-step-desc">
                    Captures 8 real-time ambient inputs from ERA5: temperature, relative humidity, dew point, solar irradiance,
                    pressure, precipitation, and trigonometric wind components. No forward weather predictions are ingested.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown(
            """
            <div class="saas-step-card">
                <span class="saas-step-num">STAGE 02</span>
                <div class="saas-step-title">168-Hour Causal Sequence</div>
                <p class="saas-step-desc">
                    Extracts autoregressive lags (1h, 2h, 3h), diurnal 24-hour daily rhythm, and 168-hour weekly seasonality.
                    Rolling 3h and 24h consumption statistics capture instantaneous household volatility.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s3:
        st.markdown(
            """
            <div class="saas-step-card">
                <span class="saas-step-num">STAGE 03</span>
                <div class="saas-step-title">Inference & Confidence Bounds</div>
                <p class="saas-step-desc">
                    HistGradientBoosting trees generate point estimates in under 10 milliseconds, accompanied by
                    calibrated 90% empirical prediction intervals to quantify uncertainty before the target hour arrives.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # SECTION 5: MODEL PERFORMANCE
    # -----------------------------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="saas-eyebrow">RIGOROUS EVALUATION</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="saas-heading">Benchmark-Beating Model Performance</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="saas-lead">Evaluated strictly on chronological, unseen held-out test splits across 72,059 validation rows.</p>',
        unsafe_allow_html=True,
    )

    perf_metrics, perf_chart = st.columns([1, 1.2], gap="large")

    with perf_metrics:
        test_mae = model_info.get("test_mae", 0.250)
        test_r2 = model_info.get("test_r2", 0.729)
        test_rmse = model_info.get("test_rmse", 0.427)

        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:16px;">
                <div class="saas-kpi-card">
                    <div class="saas-kpi-val" style="font-size:30px;">{test_mae:.3f} <span style="font-size:14px; color:#8DA4C4;">kWh</span></div>
                    <div class="saas-kpi-label">Held-out Test MAE</div>
                    <div class="saas-kpi-sub">Outperforms 1h persistence baseline (0.276 kWh).</div>
                </div>
                <div class="saas-kpi-card">
                    <div class="saas-kpi-val" style="font-size:30px;">{test_r2:.3f}</div>
                    <div class="saas-kpi-label">Held-out Test R²</div>
                    <div class="saas-kpi-sub">Explains ~73% of total consumption variance.</div>
                </div>
                <div class="saas-kpi-card">
                    <div class="saas-kpi-val" style="font-size:30px;">{test_rmse:.3f} <span style="font-size:14px; color:#8DA4C4;">kWh</span></div>
                    <div class="saas-kpi-label">Held-out Test RMSE</div>
                    <div class="saas-kpi-sub">Strict penalty on large residual deviations.</div>
                </div>
                <div class="saas-kpi-card">
                    <div class="saas-kpi-val" style="font-size:30px;">90.0%</div>
                    <div class="saas-kpi-label">Interval Coverage</div>
                    <div class="saas-kpi-sub">Calibrated 5th–95th percentile validation residuals.</div>
                </div>
            </div>
            <div class="saas-editorial-card" style="padding:18px 20px;">
                <h4 style="font-size:14.5px; margin-bottom:6px;">Model Selection Rationale</h4>
                <p style="font-size:12.5px; color:#8DA4C4; margin:0;">
                    While Random Forest tied closely on validation error, HistGradientBoosting achieved the top one-shot test MAE
                    with an artifact footprint of just 1.4 MB (vs. 270 MB), enabling seamless sub-second execution on free hosting tiers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with perf_chart:
        eval_img_path = root / "reports" / "figures" / "11_test_actual_vs_predicted.png"
        if eval_img_path.exists():
            st.image(str(eval_img_path), caption="Held-out Test Evaluation: Actual vs. Predicted Next-Hour Consumption", use_container_width=True)
        else:
            st.info("Evaluation chart reports/figures/11_test_actual_vs_predicted.png will be displayed here.")

    # -----------------------------------------------------------------------
    # SECTION 6: DATASET & TECHNOLOGY
    # -----------------------------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="saas-eyebrow">FOUNDATIONAL INFRASTRUCTURE</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="saas-heading">Dataset & Technology Stack</h2>', unsafe_allow_html=True)

    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.markdown(
            """
            <div class="saas-editorial-card">
                <div class="saas-eyebrow">RESEARCH TELEMETRY</div>
                <h4>LUMS REWD-P & Open-Meteo ERA5</h4>
                <p>
                    <b>LUMS REWD-P Dataset:</b> High-resolution residential electricity dataset curated by the Energy Informatics Group
                    at Lahore University of Management Sciences (LUMS). Spans 59 individual residential homes across 6 major cities,
                    complete with granular architectural and appliance metadata.
                </p>
                <p>
                    <b>Open-Meteo ERA5 Reanalysis:</b> Geographically localized European Centre for Medium-Range Weather Forecasts (ECMWF)
                    reanalysis datasets aligned down to the coordinate grid of each monitored city.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d2:
        st.markdown(
            """
            <div class="saas-editorial-card">
                <div class="saas-eyebrow">PRODUCTION STACK</div>
                <h4>Modern, Portable ML Architecture</h4>
                <p>
                    <b>Core Frameworks:</b> Python 3.13, Scikit-Learn (<code>HistGradientBoostingRegressor</code>), NumPy, Pandas, and PyArrow Parquet storage.
                </p>
                <p>
                    <b>Serving & Analytics:</b> FastAPI asynchronous microservice, Altair & Vega-Lite declarative visualization engine,
                    and responsive Streamlit web deployment compatible with local, Community Cloud, and Hugging Face environments.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # SECTION 7: FINAL CTA
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div class="saas-final-cta">
            <div class="saas-eyebrow" style="color:#F8AF90;">READY TO FORECAST</div>
            <h2>Your next hour, better understood.</h2>
            <p>
                Explore real-time next-hour electricity consumption forecasts across 59 Pakistani households,
                simulate what-if weather variations, and inspect detailed appliance profiles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_btn_col1, cta_btn_col2, cta_btn_col3 = st.columns([1, 1.2, 1])
    with cta_btn_col2:
        if st.button("Explore Predictions Dashboard →", type="primary", use_container_width=True, key="final_cta_btn"):
            st.session_state["nav_page"] = "Predict"
            st.rerun()

    # -----------------------------------------------------------------------
    # SECTION 8: FOOTER
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div class="saas-footer">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:16px; margin-bottom:12px;">
                <div>
                    <span style="font-size:16px; font-weight:800; color:#FFFFFF;">⚡ BijliCast</span>
                    <div style="color:#667085; font-size:12px; margin-top:2px;">Weather-Aware Household Electricity Consumption Forecasting in Pakistan</div>
                </div>
                <div>
                    <a href="https://github.com/muhammad-umar-218980/BijliCast" target="_blank">GitHub Repository ↗</a>
                </div>
            </div>
            <div style="border-top:1px solid #1E2C4B; padding-top:12px; font-size:11.5px; color:#667085;">
                <b>Data Acknowledgements:</b> LUMS Energy Informatics Group (REWD-P Dataset) and Open-Meteo (ERA5 Climate Reanalysis).
                <br>
                <b>Academic Disclaimer:</b> Developed for CT-378 Machine Learning. Predictions and intervals are intended for academic research, planning, and educational demonstration purposes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        <div style="padding:14px 0 20px 0;">
            <div class="saas-eyebrow">MAKE A PREDICTION</div>
            <h2 style="font-size:32px; font-weight:850; color:#FFFFFF; margin:0 0 8px 0;">Next hour for {house}, {city}</h2>
            <div style="color:#CBD5E1; font-size:14.5px; margin-bottom:14px;">
                Origin {origin:%d %b %Y, %H:%M} → target {target:%d %b %Y, %H:%M}. Adjust the weather inputs below to ask a what-if question.
            </div>
            <div style="display:flex; gap:8px;">
                <span class="saas-preview-badge">{model_info['name']}</span>
                <span style="background:rgba(255,255,255,0.08); color:#CBD5E1; font-size:10.5px; font-weight:700; padding:2px 10px; border-radius:999px;">90% INTERVAL</span>
                <span style="background:rgba(255,255,255,0.08); color:#CBD5E1; font-size:10.5px; font-weight:700; padding:2px 10px; border-radius:999px;">8 WEATHER INPUTS</span>
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
                    <h3 style="margin:0; font-size:22px; color:#ffffff;">Forecast</h3>
                    <span style="color:#94a3b8; font-size:13px;">{house} · {city} · target {target:%d %b %H:%M}</span>
                </div>
                <span class="saas-preview-badge">TYPICAL VS 72 H MEAN</span>
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
                <div class="bc-metric-val" style="color:{'#C2643F' if result['delta_kwh'] >= 0 else '#5b8db8'};">{result['delta_kwh']:+.3f} <span style="font-size:16px; color:#cbd5e1;">kWh</span></div>
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
            .mark_line(strokeDash=[4, 4], color="#C2643F", strokeWidth=1.5)
            .encode(x=alt.X("time:T"), y=alt.Y("mean_val:Q"))
        )

        interval_rule = (
            alt.Chart(forecast_plot)
            .mark_rule(strokeWidth=3, color="#C2643F")
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
            .mark_point(filled=True, size=90, color="#C2643F")
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
            .mark_text(dx=18, dy=-2, color="#C2643F", fontSize=13, fontWeight="bold")
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
                            range=["#5b8db8", "#C2643F"],
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
                            range=["#5b8db8", "#C2643F"],
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
            <div style="background:#13213F; border:1px solid #1E2C4B; border-radius:14px; padding:24px; text-align:center; margin-top:20px;">
                <p style="color:#CBD5E1; font-size:15px; margin:0 0 6px 0;">
                    ⚡ Ready to forecast next-hour electricity consumption for <b>{house}</b>.
                </p>
                <p style="color:#667085; font-size:13px; margin:0;">
                    Click the <b>Predict next hour</b> button above to generate the prediction metrics and the 3 analytical charts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def run(root: str | Path | None = None) -> None:
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()

    st.set_page_config(
        page_title="BijliCast - Next-Hour Electricity Forecast",
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
            logo_uri = _get_image_base64(logo_path)
            if logo_uri:
                st.markdown(
                    f'<div style="margin-bottom:12px;"><img src="{logo_uri}" style="width:80px; max-width:80px; height:auto; display:block;" alt="BijliCast" /></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.image(str(logo_path), width=80)
        else:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <span style="font-size:20px;">⚡</span>
                    <span style="font-size:18px; font-weight:800; color:#FFFFFF;">BijliCast</span>
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
                "<span style='font-size:11px; font-weight:700; letter-spacing:0.1em; color:#8DA4C4;'>HOUSEHOLD</span>",
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
                "<span style='font-size:11px; font-weight:700; letter-spacing:0.1em; color:#8DA4C4;'>MODEL</span>",
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
                "<span style='font-size:11px; font-weight:700; letter-spacing:0.1em; color:#8DA4C4;'>OVERVIEW</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                "BijliCast forecasts next-hour residential electricity demand in Pakistan with calibrated 90% uncertainty intervals."
            )

    # -------------------------------------------------------------
    # Main Page Dispatch
    # -------------------------------------------------------------
    if selected_page == "Home":
        render_home_page(root, service)
    else:
        render_predict_page(service)


if __name__ == "__main__":
    run()
