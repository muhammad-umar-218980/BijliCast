"""BijliCast Streamlit entry point.

Run locally:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for path in (str(ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from streamlit_ui import run  # noqa: E402

run(ROOT)
