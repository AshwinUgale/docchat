"""End-to-end sidecar lifecycle test.

Mirrors what the TypeScript extension does in production: spawn the sidecar
as a subprocess, read the port from stdout, open a WebSocket against /chat,
exchange a typed message round-trip, kill the process. If this passes, the
v0.1 walking-skeleton plumbing PLUS the v0.2 protocol upgrade are sound.

This test is slower than the in-process TestClient tests because it actually
launches uvicorn. Marked as a `live_subprocess` test so a future CI config
can skip it if subprocess + asyncio gets flaky on a particular runner.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import websockets

_PORT_LINE_RE = re.compile(rb"^DOCCHAT_SIDECAR_PORT=(\d+)\s*$")
_STARTUP_TIMEOUT_S = 10.0


pytestmark = pytest.mark.live_subprocess


@asynccontextmanager
async def _spawn_sidecar() -> AsyncIterator[tuple[subprocess.Popen[bytes], int]]:
    """Spawn the sidecar as a subprocess and yield (proc, port).

    The contextmanager guarantees the subprocess is terminated even if the
    test raises - leaving a uvicorn server bound to a random port would be
    a bad CI citizen.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "docchat_sidecar", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    try:
        port = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _read_port, proc),
            timeout=_STARTUP_TIMEOUT_S,
        )
        # Give uvicorn a moment to finish wiring the WS endpoint after
        # the port print. /health would be more authoritative but adding
        # an httpx round-trip here doubles the test's network deps for
        # marginal benefit.
        await asyncio.sleep(0.3)
        yield proc, port
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)


def _read_port(proc: subprocess.Popen[bytes]) -> int:
    """Block until the sidecar prints DOCCHAT_SIDECAR_PORT=<N>, return N."""
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            # Process exited before printing the port - capture stderr for
            # the assertion failure so debugging isn't a guessing game.
            err = proc.stderr.read() if proc.stderr is not None else b""
            raise RuntimeError(
                f"sidecar exited without printing port (exit={proc.returncode})\n"
                f"stderr:\n{err.decode(errors='replace')}"
            )
        match = _PORT_LINE_RE.match(line.rstrip())
        if match:
            return int(match.group(1))


async def test_sidecar_spawns_and_prints_port() -> None:
    """The port-line contract is what the extension parses."""
    async with _spawn_sidecar() as (proc, port):
        assert proc.poll() is None
        assert 1024 < port < 65_536


async def test_sidecar_chat_websocket_echoes() -> None:
    """End-to-end round-trip: subprocess -> WS -> typed protocol response.

    v0.3 user_query goes through the real agent loop (Mneme + ToolPicker +
    OpenAI). Without OpenAI/Qdrant configured in CI, the handler surfaces
    the error as an ``[agent error]`` AssistantText - either way, the WS
    protocol round-trip works. The Ping/Pong assertion exercises the
    typed-protocol path without any external dependencies.

    v0.6.1 raised the user_query timeout from 15s to 45s: the v0.6
    multi-iteration agent loop combined with v0.6.1's "try 2+ tools before
    refusing" prompt rule means a single user_query is now 2-3 sequential
    OpenAI calls on the real path. Real-world p95 is in the 7-15s range;
    45s gives plenty of headroom for slow Windows boxes / cold connections.
    """
    async with _spawn_sidecar() as (_proc, port):
        uri = f"ws://127.0.0.1:{port}/chat"
        async with websockets.connect(uri) as ws:
            # Ping/Pong is the deterministic protocol round-trip - no agent
            # involvement, so it works without OpenAI/Qdrant.
            await ws.send(json.dumps({"type": "ping"}))
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            assert reply["type"] == "pong"
            assert isinstance(reply["version"], str)

            # A user_query lands an AssistantText regardless of whether the
            # agent path succeeds (errors come back as a typed message, not
            # a torn connection).
            await ws.send(json.dumps({"type": "user_query", "text": "hello v0.6"}))
            reply2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=45.0))
            assert reply2["type"] == "assistant_text"
            assert isinstance(reply2["text"], str)
