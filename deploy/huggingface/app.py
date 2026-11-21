"""Hugging Face Space entry point (Streamlit SDK)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_ui import run  # noqa: E402

run(ROOT)
