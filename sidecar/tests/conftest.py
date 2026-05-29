"""Pytest conftest - mirrors the Mneme + ToolPicker pattern.

Loads .env at the sidecar root so OPENAI_API_KEY-gated tests can find a
key when one is set, without forcing it everywhere.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    # .env lives at the docchat repo root (one level above sidecar/).
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(_env_path, encoding="utf-8-sig")
