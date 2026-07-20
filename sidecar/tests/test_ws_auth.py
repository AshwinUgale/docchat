"""WebSocket handshake authorization for the /chat endpoint.

In-process (Starlette TestClient) — no subprocess, no network. Covers the two
gates in ``_reject_unauthorized``: browser-Origin rejection and the per-spawn
token the extension sets via DOCCHAT_WS_TOKEN.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient, WebSocketDisconnect

from docchat_sidecar.__main__ import app


def _client() -> TestClient:
    return TestClient(app)


def _ping_pong(ws) -> None:
    """A deterministic round-trip that needs no OpenAI/Qdrant."""
    ws.send_json({"type": "ping"})
    assert ws.receive_json()["type"] == "pong"


@pytest.mark.parametrize("origin", ["http://evil.example", "https://evil.example"])
def test_rejects_browser_origin(monkeypatch: pytest.MonkeyPatch, origin: str) -> None:
    # A malicious web page connects with an http(s) Origin -> rejected even
    # when no token is configured.
    monkeypatch.delenv("DOCCHAT_WS_TOKEN", raising=False)
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/chat", headers={"origin": origin}) as ws:
            _ping_pong(ws)


def test_allows_when_no_token_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    # Standalone / dev / the eval harness (which bypasses the WS): no token
    # env -> connections allowed so existing usage keeps working.
    monkeypatch.delenv("DOCCHAT_WS_TOKEN", raising=False)
    with _client().websocket_connect("/chat") as ws:
        _ping_pong(ws)


def test_missing_token_rejected_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCCHAT_WS_TOKEN", "secret-nonce")
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/chat") as ws:
            _ping_pong(ws)


def test_wrong_token_rejected_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCCHAT_WS_TOKEN", "secret-nonce")
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/chat?token=wrong") as ws:
            _ping_pong(ws)


def test_correct_token_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCCHAT_WS_TOKEN", "secret-nonce")
    with _client().websocket_connect("/chat?token=secret-nonce") as ws:
        _ping_pong(ws)
