"""Create/update the BijliCast Space on Hugging Face and upload the bundle.

The Space SDK is set to ``streamlit`` explicitly. Hugging Face's New Space
form no longer offers Streamlit, but the Hub API still accepts it.

Usage (PowerShell)::

    $env:HF_TOKEN = "hf_..."          # or run `hf auth login` once
    python -m src.deploy.push_hf_space

The token is read from the environment (or the ``hf auth login`` cache) and is
never written to disk by this script.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "deploy" / "huggingface"
SPACE_NAME = os.environ.get("HF_SPACE_NAME", "BijliCast")

# upload_folder does not read .gitignore, so filter local noise here.
IGNORE_PATTERNS = [
    ".git/*",
    "__pycache__/*",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
]

TERMINAL_STAGES = {
    "RUNNING",
    "BUILD_ERROR",
    "CONFIG_ERROR",
    "RUNTIME_ERROR",
    "PAUSED",
    "STOPPED",
}


def main() -> int:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    if not BUNDLE.exists():
        print(
            f"bundle missing: {BUNDLE}\n"
            "run `python -m src.deploy.export_hf_space` first",
            file=sys.stderr,
        )
        return 1

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as exc:  # noqa: BLE001 - auth problems should be obvious
        print(f"not authenticated: {exc}", file=sys.stderr)
        print("set HF_TOKEN or run `hf auth login`", file=sys.stderr)
        return 1

    repo_id = f"{user}/{SPACE_NAME}"
    print(f"account: {user}")
    print(f"space:   {repo_id}")

    # The Hub API rejects `space_sdk="streamlit"` at creation time (only
    # gradio/docker/static are accepted), but it still accepts a README whose
    # front matter sets `sdk: streamlit`. So create a Static Space and let the
    # bundle README switch the SDK.
    try:
        url = api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        print(f"create_repo failed: {exc}", file=sys.stderr)
        return 1
    print(f"created/reused: {url}")

    # Upload the README first: it carries `sdk: streamlit`, and the Hub
    # validates + applies the SDK switch before the rest of the bundle lands.
    try:
        api.upload_file(
            path_or_fileobj=str(BUNDLE / "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
            commit_message="Set Space SDK to streamlit",
        )
    except HfHubHTTPError as exc:
        print(f"README upload failed: {exc}", file=sys.stderr)
        return 1

    try:
        api.upload_folder(
            folder_path=str(BUNDLE),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy BijliCast Streamlit app",
            ignore_patterns=IGNORE_PATTERNS,
        )
    except HfHubHTTPError as exc:
        print(f"upload failed: {exc}", file=sys.stderr)
        return 1
    print("uploaded deploy/huggingface")

    deadline = time.time() + 900
    last = None
    while time.time() < deadline:
        info = api.space_info(repo_id)
        stage = info.runtime.stage if info.runtime else None
        if stage != last:
            print(f"stage: {stage}")
            last = stage
        if stage in TERMINAL_STAGES:
            break
        time.sleep(15)

    host = repo_id.replace("/", "-").replace("_", "-").lower()
    print(f"page: https://huggingface.co/spaces/{repo_id}")
    print(f"app:  https://{host}.hf.space")
    return 0 if last == "RUNNING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
