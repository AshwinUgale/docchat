"""WebSocket message protocol between the VS Code extension and the sidecar.

v0.1 used raw text echo. v0.2 introduces typed JSON messages so the two
sides can negotiate indexing progress, agent responses, and error states
without overloading a single string channel.

Wire format: every message is a JSON object with a ``type`` discriminator
plus message-specific fields. Pydantic v2 handles serialization +
validation; FastAPI already depends on it so this adds zero new deps.

ADR-005 (`.cowork/DECISIONS.md`) captures why typed discriminated unions
over raw text or JSON-RPC: small surface, no method-name registry, easy
to extend at each milestone (v0.3 adds agent-loop messages, etc.).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

__all__ = [
    "AssistantText",
    "ClientMessage",
    "IndexComplete",
    "IndexError",
    "IndexLibrary",
    "IndexProgress",
    "Ping",
    "Pong",
    "ServerMessage",
    "UserQuery",
    "client_adapter",
    "server_adapter",
]


# ---------------------------------------------------------------------------
# Client -> Server (sent by the VS Code extension)
# ---------------------------------------------------------------------------


class UserQuery(BaseModel):
    """The user typed a question into the chat panel."""

    type: Literal["user_query"] = "user_query"
    text: str


class IndexLibrary(BaseModel):
    """Request the sidecar to index a specific (library, version) pair."""

    type: Literal["index_library"] = "index_library"
    library: str
    version: str


class Ping(BaseModel):
    """Health-check ping from the client. Server replies with Pong."""

    type: Literal["ping"] = "ping"


# ---------------------------------------------------------------------------
# Server -> Client (sent by the Python sidecar)
# ---------------------------------------------------------------------------


class AssistantText(BaseModel):
    """A response chunk (full message in v0.2; streaming chunks in v0.3)."""

    type: Literal["assistant_text"] = "assistant_text"
    text: str


class IndexProgress(BaseModel):
    """Mid-indexing update so the chat panel can render a progress bar.

    ``chunks_total`` may be ``None`` for the first updates before the full
    chunk count is known (the indexer streams as it fetches).
    """

    type: Literal["index_progress"] = "index_progress"
    library: str
    version: str
    chunks_done: int
    chunks_total: int | None = None
    note: str | None = None


class IndexComplete(BaseModel):
    """Indexing finished cleanly. ``chunks_indexed`` is the final count."""

    type: Literal["index_complete"] = "index_complete"
    library: str
    version: str
    chunks_indexed: int


class IndexError(BaseModel):
    """Indexing failed. The ``message`` is what to render in the panel."""

    type: Literal["index_error"] = "index_error"
    library: str
    version: str
    message: str


class Pong(BaseModel):
    """Reply to Ping. Carries the sidecar version so the client can detect drift."""

    type: Literal["pong"] = "pong"
    version: str


# ---------------------------------------------------------------------------
# Discriminated unions + TypeAdapter helpers
# ---------------------------------------------------------------------------

ClientMessage = Annotated[
    UserQuery | IndexLibrary | Ping,
    Field(discriminator="type"),
]
"""Anything the extension can send to the sidecar."""

ServerMessage = Annotated[
    AssistantText | IndexProgress | IndexComplete | IndexError | Pong,
    Field(discriminator="type"),
]
"""Anything the sidecar can send to the extension."""


# TypeAdapter caches the parser; cheaper than calling parse_obj on the union
# every message.
client_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
server_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
