"""Smoke test - confirms the package imports, the WebSocket dispatches typed
messages, and the two PyPI dogfood libraries import.
"""

from __future__ import annotations

import json

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


def test_chat_websocket_round_trips_user_query() -> None:
    """v0.2 protocol: user_query in, assistant_text out (echo-style at v0.2)."""
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_text(json.dumps({"type": "user_query", "text": "hello"}))
        reply = json.loads(ws.receive_text())
        assert reply["type"] == "assistant_text"
        assert "hello" in reply["text"]


def test_chat_websocket_ping_pong() -> None:
    """Ping returns a Pong carrying the sidecar version."""
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        reply = json.loads(ws.receive_text())
        assert reply["type"] == "pong"
        assert reply["version"] == docchat_sidecar.__version__


def test_chat_websocket_rejects_malformed_frames() -> None:
    """Bad frames produce a single error message and keep the connection open."""
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_text("not even json")
        reply = json.loads(ws.receive_text())
        assert reply["type"] == "assistant_text"
        assert "protocol error" in reply["text"]
        # Connection still open - a follow-up ping should work.
        ws.send_text(json.dumps({"type": "ping"}))
        reply2 = json.loads(ws.receive_text())
        assert reply2["type"] == "pong"


def test_dogfood_imports_resolve() -> None:
    """Confirm the two PyPI dependencies actually import.

    This is the test that proves the dogfood story is real: if mneme or
    toolpicker fail to import, the rest of DocChat is fiction.
    """
    import mneme
    import toolpicker

    assert hasattr(mneme, "__version__")
    assert hasattr(toolpicker, "__version__")
