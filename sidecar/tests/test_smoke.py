"""Smoke test - confirms the package imports and the WebSocket echoes.

Walking-skeleton verification: prove the FastAPI app stands up, /health
returns ok, and /chat echoes a message round-trip. No agent loop yet.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import docchat_sidecar
from docchat_sidecar.__main__ import app


def test_version_exposed() -> None:
    assert isinstance(docchat_sidecar.__version__, str)
    assert docchat_sidecar.__version__ != ""


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == docchat_sidecar.__version__


def test_chat_websocket_echoes() -> None:
    """v0.0 echo round-trip. v0.1 replaces this with a real agent assertion."""
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_text("hello")
        assert ws.receive_text() == "echo: hello"
        ws.send_text("world")
        assert ws.receive_text() == "echo: world"


def test_dogfood_imports_resolve() -> None:
    """Confirm the two PyPI dependencies actually import.

    This is the test that proves the dogfood story is real: if mneme or
    toolpicker fail to import, the rest of DocChat is fiction.
    """
    import mneme
    import toolpicker

    assert hasattr(mneme, "__version__")
    assert hasattr(toolpicker, "__version__")
