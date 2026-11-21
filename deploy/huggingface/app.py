"""BijliCast — Gradio front end for Hugging Face Spaces (ZeroGPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import gradio as gr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service import BijliCastService  # noqa: E402

try:  # available inside a ZeroGPU Space; optional locally
    import spaces

    gpu = spaces.GPU(duration=10)
except Exception:  # noqa: BLE001 - local/dev fallback

    def gpu(fn):
        return fn


SERVICE = BijliCastService(ROOT)
HOUSES = SERVICE.houses()
HOUSE_IDS = [h["house_id"] for h in HOUSES]
BY_ID = {h["house_id"]: h for h in HOUSES}
INFO = SERVICE.model_info()

WEATHER_FIELDS = [
    ("temperature", "Temperature (°C)", -10.0, 55.0, 0.5),
    ("humidity", "Humidity (%)", 0.0, 100.0, 1.0),
    ("dew", "Dew point (°C)", -20.0, 40.0, 0.5),
    ("precipitation", "Precipitation (mm)", 0.0, 50.0, 0.1),
    ("wind_speed", "Wind speed (km/h)", 0.0, 120.0, 0.5),
    ("wind_direction", "Wind direction (°)", 0.0, 360.0, 1.0),
    ("pressure", "Pressure (hPa)", 900.0, 1050.0, 0.5),
    ("solar_radiation", "Solar radiation (W/m²)", 0.0, 1200.0, 5.0),
]

PROFILE_ROWS = [
    ("No. of people (Temp+Perm)", "People"),
    ("Air Conditioners", "ACs"),
    ("Air Coolers", "Coolers"),
    ("Refrigerators", "Fridges"),
    ("No. of rooms", "Rooms"),
    ("Property Area (Marla)", "Area (Marla)"),
]


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def profile_markdown(house_id: str) -> str:
    house = BY_ID[house_id]
    profile = house["profile"]
    rows = "\n".join(
        f"| {label} | {_fmt(profile.get(key))} |" for key, label in PROFILE_ROWS
    )
    return (
        f"**{house_id} · {house['city']}**\n\n"
        f"| Field | Value |\n|---|---|\n{rows}\n\n"
        f"<sub>Origin {house['origin_time']} → target {house['target_time']}</sub>"
    )


def defaults_for(house_id: str) -> list[float]:
    latest = BY_ID[house_id]["defaults"]
    return [float(latest[key]) for key, *_ in WEATHER_FIELDS]


def model_caption() -> str:
    mae = INFO["test_mae"]
    r2 = INFO["test_r2"]
    coverage = INFO["interval_val_coverage"]
    parts = [f"**Model:** {INFO['name']}"]
    parts.append(f"test MAE {mae:.3f} kWh" if mae is not None else "test MAE n/a")
    if r2 is not None:
        parts.append(f"R² {r2:.3f}")
    if coverage is not None:
        parts.append(f"90% interval calibrated on {coverage:.0%} validation coverage")
    parts.append(f"trained on {INFO['n_train_rows']:,} rows")
    return " · ".join(parts)


def chart_for(result) -> plt.Figure:
    frame = pd.DataFrame(result.history)
    frame["time"] = pd.to_datetime(frame["time"])

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(frame["time"], frame["usage_kwh"], color="#1f77b4", label="last 72 h")
    origin = frame["time"].iloc[-1]
    ax.axvline(origin, color="grey", ls="--", lw=1, label="origin")
    target = pd.Timestamp(result.target_time)
    ax.errorbar(
        [target],
        [result.prediction_kwh],
        yerr=[
            [result.prediction_kwh - result.interval_low_kwh],
            [result.interval_high_kwh - result.prediction_kwh],
        ],
        fmt="o",
        color="#d62728",
        capsize=5,
        label="forecast (90% interval)",
    )
    ax.set_ylabel("kWh")
    ax.set_xlabel("datetime")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def summary_for(result) -> str:
    delta = result.delta_kwh
    sign = "+" if delta >= 0 else "−"
    return (
        f"### {result.prediction_kwh:.3f} kWh\n"
        f"**90% interval** {result.interval_low_kwh:.3f} – "
        f"{result.interval_high_kwh:.3f} kWh  \n"
        f"**vs last hour** {sign}{abs(delta):.3f} kWh "
        f"(last reading {result.last_hour_kwh:.3f} kWh)  \n"
        f"<sub>{result.house_id} · {result.city} · "
        f"{result.origin_time} → {result.target_time}</sub>"
    )


@spaces.GPU(duration=5)  # no-op outside ZeroGPU; enables Space build
def gp_probe(x):
    return x


def predict(house_id: str, *weather_values: float):
    weather = {
        key: float(value)
        for (key, *_), value in zip(WEATHER_FIELDS, weather_values)
    }
    result = SERVICE.predict(house_id, weather)
    return summary_for(result), chart_for(result)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="BijliCast", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"## ⚡ BijliCast\n*{model_caption()}*\n\n_Research/demo purposes only_")

        with gr.Row():
            with gr.Column(scale=5):
                house = gr.Dropdown(
                    choices=HOUSE_IDS,
                    value=HOUSE_IDS[0],
                    label="Household",
                )
                profile = gr.Markdown(profile_markdown(HOUSE_IDS[0]))
                sliders = [
                    gr.Slider(
                        minimum=low,
                        maximum=high,
                        value=float(BY_ID[HOUSE_IDS[0]]["defaults"][key]),
                        step=step,
                        label=label,
                    )
                    for key, label, low, high, step in WEATHER_FIELDS
                ]
                button = gr.Button("⚡ Predict next hour", variant="primary")

            with gr.Column(scale=6):
                summary = gr.Markdown(
                    "_Pick a household and press **Predict next hour**._"
                )
                plot = gr.Plot(label="Last 72 hours + forecast")
                gr.Markdown(
                    "<sub>Research/demo project. Not for billing or grid operations.</sub>"
                )

        house.change(
            fn=lambda h: [profile_markdown(h), *defaults_for(h)],
            inputs=house,
            outputs=[profile, *sliders],
        )
        button.click(
            fn=predict,
            inputs=[house, *sliders],
            outputs=[summary, plot],
        )
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()