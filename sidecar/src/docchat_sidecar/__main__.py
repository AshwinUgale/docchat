"""``python -m docchat_sidecar`` - FastAPI WebSocket entrypoint.

v0.2 upgrades the v0.1 echo to a typed message dispatcher. The /chat
WebSocket parses each incoming frame as a ``ClientMessage`` (Pydantic
discriminated union over user_query / index_library / ping) and dispatches
to the right handler. Server-side outgoing frames are ``ServerMessage``
variants (assistant_text / index_progress / index_complete / index_error
/ pong).

The agent loop lands at v0.3; v0.2 still echoes user queries via an
AssistantText so the panel UI is exercised end-to-end. Indexing is real:
``index_library`` triggers the DocIndexer and streams progress back as
``index_progress`` until ``index_complete`` or ``index_error``.

Run standalone:

    cd sidecar
    uv run python -m docchat_sidecar --port 0
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import socket
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Lazy .env load - sidecar-only concern; the library never auto-loads dotenv.
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    load_dotenv(_env_path, encoding="utf-8-sig")

from docchat_sidecar import __version__
from docchat_sidecar.protocol import (
    AssistantText,
    IndexComplete,
    IndexError,
    IndexLibrary,
    IndexProgress,
    Ping,
    Pong,
    UserQuery,
    client_adapter,
    server_adapter,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="DocChat sidecar", version=__version__)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe - the extension polls this before opening the panel."""
    return {"status": "ok", "version": __version__}


@app.websocket("/chat")
async def chat(ws: WebSocket) -> None:
    """Typed message dispatcher.

    Each text frame is JSON-decoded + validated as a ``ClientMessage``.
    Bad frames produce a single error frame; the connection stays open so
    the panel doesn't have to reconnect on transient parse failures.
    """
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = client_adapter.validate_json(raw)
            except Exception as exc:
                await _send(ws, AssistantText(text=f"[protocol error] {exc}"))
                continue
            await _dispatch(ws, msg)
    except WebSocketDisconnect:
        return


async def _dispatch(ws: WebSocket, msg: UserQuery | IndexLibrary | Ping) -> None:
    """Route a parsed client message to the appropriate handler."""
    if isinstance(msg, UserQuery):
        await _run_agent(ws, msg.text)
        return
    if isinstance(msg, IndexLibrary):
        await _run_indexing(ws, msg.library, msg.version)
        return
    if isinstance(msg, Ping):
        await _send(ws, Pong(version=__version__))
        return


async def _run_agent(ws: WebSocket, query: str) -> None:
    """v0.3 ReAct loop: route -> tool -> retrieve -> generate -> respond.

    Constructs the Agent per query for simplicity; the cost is two SDK
    constructors plus a Mneme MemoryManager - all cheap when the OpenAI
    + Qdrant clients aren't doing network work yet. A future optimisation
    is per-connection caching, but the v0.3 demo doesn't need it.
    """
    # Lazy import - keeps the bare /health surface free of OpenAI + Qdrant
    # + Mneme deps for users who only want IPC verification.
    import os

    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient

    from docchat_sidecar.agent import Agent
    from docchat_sidecar.memory import build_memory

    try:
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        workspace_path = os.environ.get("DOCCHAT_WORKSPACE_PATH")
        memory = build_memory(workspace_path=workspace_path, qdrant_url=None)
        agent = Agent(
            openai=AsyncOpenAI(),
            qdrant=AsyncQdrantClient(url=qdrant_url),
            memory=memory,
            # v0.6: thread the workspace path through to the agent so
            # SearchWorkspaceCodeTool can scope ripgrep correctly.
            workspace_path=workspace_path,
        )
        response = await agent.answer(query)
    except Exception as exc:
        logger.exception("agent failed")
        await _send(ws, AssistantText(text=f"[agent error] {exc}"))
        return

    await _send(ws, AssistantText(text=response.text))


async def _run_indexing(ws: WebSocket, library: str, version: str) -> None:
    """Construct an indexer on the fly and stream every progress frame.

    Why per-request construction: tests can substitute their own DocIndexer
    instance; in production the cost (one OpenAI + one Qdrant client) is
    fine for a per-window sidecar that only indexes occasionally.
    """
    # Lazy import - keeps the import-time deps light for users running
    # the bare /health endpoint without OpenAI / Qdrant configured.
    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient

    from docchat_sidecar.indexer import DocIndexer

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    indexer = DocIndexer(
        qdrant=AsyncQdrantClient(url=qdrant_url),
        openai=AsyncOpenAI(),
    )
    async for frame in indexer.index(library, version):
        await _send(ws, frame)


async def _send(
    ws: WebSocket,
    msg: AssistantText | IndexProgress | IndexComplete | IndexError | Pong,
) -> None:
    """Serialise a server message and send it as a single text frame."""
    payload = server_adapter.dump_json(msg).decode("utf-8")
    await ws.send_text(payload)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m docchat_sidecar")
    p.add_argument("--port", type=int, default=0, help="Bind port. 0 = pick a free one.")
    p.add_argument("--host", default="127.0.0.1", help="Bind address. Default 127.0.0.1.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    port = args.port or _pick_free_port()
    print(f"DOCCHAT_SIDECAR_PORT={port}", flush=True)
    uvicorn.run(app, host=args.host, port=port, log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
