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
import hmac
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Lazy .env load - sidecar-only concern; the library never auto-loads dotenv.
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    load_dotenv(_env_path, encoding="utf-8-sig")

# v1.0.2 - fail-fast on missing OPENAI_API_KEY so the extension can detect
# the "no key" case via stderr and surface a one-click recovery action,
# instead of the original behavior of silently spawning and failing at the
# first embed call with an opaque OpenAI AuthenticationError.
if not os.environ.get("OPENAI_API_KEY"):
    print(
        "DOCCHAT_SIDECAR_ERROR=missing_openai_api_key",
        file=sys.stdout,
        flush=True,
    )
    print(
        "OPENAI_API_KEY is not set. DocChat uses OpenAI's embeddings API "
        'for retrieval. Run "DocChat: Set OpenAI API Key" from the '
        "command palette to store one in VS Code SecretStorage.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(2)

from docchat_sidecar import __version__
from docchat_sidecar.protocol import (
    AssistantStreamFinal,
    AssistantText,
    AssistantTextDelta,
    IndexComplete,
    IndexError,
    IndexLibrary,
    IndexProgress,
    Ping,
    Pong,
    SettingsUpdate,
    UserQuery,
    client_adapter,
    server_adapter,
)

# v0.7: runtime-mutable settings. The webview's settings drawer posts
# ``SettingsUpdate`` messages that mutate this dict; ``_run_agent`` reads
# from it on every query, falling back to defaults when a key is absent.
# Process-local; respawning the sidecar resets to env-var defaults.
_RUNTIME_SETTINGS: dict[str, Any] = {}

logger = logging.getLogger(__name__)

app = FastAPI(title="DocChat sidecar", version=__version__)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe - the extension polls this before opening the panel."""
    return {"status": "ok", "version": __version__}


async def _reject_unauthorized(ws: WebSocket) -> bool:
    """Reject an unauthorized WebSocket handshake before ``accept()``.

    Two gates. Both close the socket (a close before accept is a 403
    handshake rejection) and return ``True`` when the connection is denied:

    * **Origin** — a malicious web page in the user's browser connects with an
      ``http(s)://`` Origin. The VS Code webview uses the ``vscode-webview://``
      scheme and native clients send no Origin, so neither is affected. Reject
      any http(s) Origin outright.
    * **Token** — when the extension set ``DOCCHAT_WS_TOKEN`` (a per-spawn nonce
      it also hands the webview via postMessage), require it as the ``?token=``
      query param. The nonce lives only in this process's env and that one
      webview, so no other local client can present it. Constant-time compared.
      Unset (standalone / dev / the eval harness, which bypasses the WS
      entirely) means "no token required" — logged once so it's visible.
    """
    origin = ws.headers.get("origin", "")
    if origin.startswith(("http://", "https://")):
        logger.warning("rejecting WebSocket from browser origin %r", origin)
        await ws.close(code=1008)  # policy violation
        return True

    expected = os.environ.get("DOCCHAT_WS_TOKEN") or None
    if expected is None:
        logger.warning(
            "DOCCHAT_WS_TOKEN not set — accepting WebSocket connections without a "
            "token (fine for standalone/dev; the extension always sets one)."
        )
        return False

    presented = ws.query_params.get("token", "")
    if not presented or not hmac.compare_digest(presented, expected):
        logger.warning("rejecting WebSocket: missing or invalid token")
        await ws.close(code=1008)
        return True
    return False


@app.websocket("/chat")
async def chat(ws: WebSocket) -> None:
    """Typed message dispatcher.

    Each text frame is JSON-decoded + validated as a ``ClientMessage``.
    Bad frames produce a single error frame; the connection stays open so
    the panel doesn't have to reconnect on transient parse failures.
    """
    if await _reject_unauthorized(ws):
        return
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


async def _dispatch(ws: WebSocket, msg: UserQuery | IndexLibrary | SettingsUpdate | Ping) -> None:
    """Route a parsed client message to the appropriate handler."""
    if isinstance(msg, UserQuery):
        await _run_agent(ws, msg.text)
        return
    if isinstance(msg, IndexLibrary):
        await _run_indexing(ws, msg.library, msg.version)
        return
    if isinstance(msg, SettingsUpdate):
        _apply_settings_update(msg)
        return
    if isinstance(msg, Ping):
        await _send(ws, Pong(version=__version__))
        return


def _apply_settings_update(msg: SettingsUpdate) -> None:
    """v0.7: persist the in-process settings the agent reads on next query.

    v0.7.1 uses logger.warning rather than logger.info so the line shows
    up in the DocChat Output channel without configuring basicConfig
    (which would conflict with uvicorn's own log setup). Settings
    changes are low-frequency user actions worth surfacing.
    """
    if msg.chat_model is not None:
        _RUNTIME_SETTINGS["chat_model"] = msg.chat_model
    if msg.score_floor is not None:
        _RUNTIME_SETTINGS["score_floor"] = msg.score_floor
    if msg.max_iterations is not None:
        _RUNTIME_SETTINGS["max_iterations"] = msg.max_iterations
    logger.warning("runtime settings updated: %r", _RUNTIME_SETTINGS)


async def _run_agent(ws: WebSocket, query: str) -> None:
    """v0.7 streaming ReAct loop: AssistantTextDelta chunks then AssistantStreamFinal.

    Constructs the Agent per query (cheap - SDK constructors are
    non-network) and iterates ``answer_stream`` over the WebSocket so the
    panel renders token-by-token. Errors come back as a single
    ``AssistantText`` (not the streaming protocol) since they happen
    before the stream begins.
    """
    # Lazy import - keeps the bare /health surface free of OpenAI + Qdrant
    # + Mneme deps for users who only want IPC verification.
    import os

    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient

    from docchat_sidecar.agent import Agent
    from docchat_sidecar.lockfiles import parse_package_json
    from docchat_sidecar.memory import build_memory

    try:
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        workspace_path = os.environ.get("DOCCHAT_WORKSPACE_PATH")
        memory = build_memory(workspace_path=workspace_path, qdrant_url=None)
        # v0.7: pick up runtime settings posted via SettingsUpdate; fall
        # back to env vars and then the Agent's hard-coded defaults.
        agent_kwargs: dict[str, Any] = {
            "openai": AsyncOpenAI(),
            "qdrant": AsyncQdrantClient(url=qdrant_url),
            "memory": memory,
            "workspace_path": workspace_path,
        }
        chat_model = _RUNTIME_SETTINGS.get("chat_model") or os.environ.get("DOCCHAT_CHAT_MODEL")
        if chat_model:
            agent_kwargs["chat_model"] = chat_model
        max_iter = _RUNTIME_SETTINGS.get("max_iterations")
        if max_iter is not None:
            agent_kwargs["max_iterations"] = int(max_iter)
        agent = Agent(**agent_kwargs)
    except Exception as exc:
        logger.exception("agent construction failed")
        await _send(ws, AssistantText(text=f"[agent error] {exc}"))
        return

    # v0.9.1: read lockfile pins from the workspace so the agent routes
    # queries to the right (library, version) instead of guessing from
    # question text. Same plumbing the eval runner gained at v0.9. Only
    # package.json is supported at v0.9.1; pyproject.toml + requirements.txt
    # land at v1.0 alongside the auto-install sidecar work.
    pinned_libraries: dict[str, str] | None = None
    if workspace_path:
        from docchat_sidecar.lockfiles import (
            parse_pyproject_toml,
            parse_requirements_txt,
        )

        # Local rename to avoid shadowing the WebSocket parameter ``ws``.
        ws_dir = Path(workspace_path)
        # v1.0: try each manifest format in order; first non-empty wins.
        # package.json beats Python files if both exist (Node project).
        manifests = [
            ("package.json", ws_dir / "package.json", parse_package_json),
            ("pyproject.toml", ws_dir / "pyproject.toml", parse_pyproject_toml),
            ("requirements.txt", ws_dir / "requirements.txt", parse_requirements_txt),
        ]
        for label, candidate, parser in manifests:
            if not candidate.is_file():
                continue
            try:
                pins = parser(candidate)
            except Exception as exc:
                logger.warning("%s parse failed (trying next manifest): %s", label, exc)
                continue
            if not pins:
                continue
            pinned_libraries = {p.library.lower(): p.version for p in pins}
            logger.warning("loaded %d lockfile pins from %s", len(pinned_libraries), candidate)
            break

    try:
        async for event in agent.answer_stream(query, pinned_libraries=pinned_libraries):
            await _send(ws, event)
    except Exception as exc:
        logger.exception("agent stream failed mid-query")
        # Send a terminator so the webview can release its loading state.
        await _send(ws, AssistantText(text=f"[agent error] {exc}"))


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
    msg: AssistantText
    | AssistantTextDelta
    | AssistantStreamFinal
    | IndexProgress
    | IndexComplete
    | IndexError
    | Pong,
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
