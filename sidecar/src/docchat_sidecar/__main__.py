"""``python -m docchat_sidecar`` - FastAPI WebSocket entrypoint.

v0.0 ships the stub: a `/health` endpoint and a `/chat` WebSocket that
echoes back whatever the client sends. v0.1 swaps the echo for the real
agent loop.

Run standalone:

    cd sidecar
    uv run python -m docchat_sidecar --port 0

The port is read from the CLI; ``--port 0`` lets the OS pick a free one
(printed to stdout so the parent VS Code extension can capture it).
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Lazy .env load so the sidecar works under VS Code's spawned environment
# without forcing the user to set env vars at the shell level. The library
# never loads dotenv itself; this is a sidecar-only concern.
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    load_dotenv(_env_path, encoding="utf-8-sig")

from docchat_sidecar import __version__

app = FastAPI(title="DocChat sidecar", version=__version__)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe - the extension polls this before opening the panel."""
    return {"status": "ok", "version": __version__}


@app.websocket("/chat")
async def chat(ws: WebSocket) -> None:
    """v0.0: echo. v0.1: ReAct loop with Mneme + ToolPicker integration."""
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            await ws.send_text(f"echo: {msg}")
    except WebSocketDisconnect:
        return


def _pick_free_port() -> int:
    """Bind to port 0 to let the OS hand us a free port, then release it.

    Small race: another process could grab the port between this call and
    uvicorn binding. Acceptable for a single-user local sidecar; we accept
    the trade-off rather than holding a lifetime-of-process socket.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m docchat_sidecar")
    p.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to bind. 0 = pick a free one (printed to stdout).",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Default 127.0.0.1; do not change for production use.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    port = args.port or _pick_free_port()
    # Print the chosen port BEFORE uvicorn takes over stdout, so the parent
    # extension can capture it from the first line of stdout.
    print(f"DOCCHAT_SIDECAR_PORT={port}", flush=True)
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
