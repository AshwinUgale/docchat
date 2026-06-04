"""Protocol round-trip tests - every Message variant serialises + parses."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docchat_sidecar.protocol import (
    AssistantStreamFinal,
    AssistantText,
    AssistantTextDelta,
    CitationRef,
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


def test_user_query_round_trip() -> None:
    original = UserQuery(text="how do I create a Suspense boundary?")
    # UserQuery is a CLIENT message; use client_adapter both directions.
    raw = client_adapter.dump_json(original).decode("utf-8")
    parsed = client_adapter.validate_json(raw)
    assert isinstance(parsed, UserQuery)
    assert parsed.text == original.text


def test_index_library_round_trip() -> None:
    original = IndexLibrary(library="react", version="18.2.0")
    raw = client_adapter.dump_json(original).decode("utf-8")
    parsed = client_adapter.validate_json(raw)
    assert isinstance(parsed, IndexLibrary)
    assert parsed.library == "react"
    assert parsed.version == "18.2.0"


def test_ping_pong_round_trip() -> None:
    parsed_ping = client_adapter.validate_json(client_adapter.dump_json(Ping()))
    assert isinstance(parsed_ping, Ping)
    parsed_pong = server_adapter.validate_json(server_adapter.dump_json(Pong(version="0.2.0")))
    assert isinstance(parsed_pong, Pong)
    assert parsed_pong.version == "0.2.0"


def test_assistant_text_round_trip() -> None:
    original = AssistantText(text="React 18.2 uses createRoot.")
    raw = server_adapter.dump_json(original).decode("utf-8")
    parsed = server_adapter.validate_json(raw)
    assert isinstance(parsed, AssistantText)
    assert parsed.text == original.text


def test_index_progress_carries_optional_total() -> None:
    progress = IndexProgress(library="react", version="18.2.0", chunks_done=5, chunks_total=None)
    parsed = server_adapter.validate_json(server_adapter.dump_json(progress))
    assert isinstance(parsed, IndexProgress)
    assert parsed.chunks_total is None


def test_index_complete_round_trip() -> None:
    complete = IndexComplete(library="react", version="18.2.0", chunks_indexed=42)
    parsed = server_adapter.validate_json(server_adapter.dump_json(complete))
    assert isinstance(parsed, IndexComplete)
    assert parsed.chunks_indexed == 42


def test_index_error_round_trip() -> None:
    err = IndexError(library="react", version="18.2.0", message="fetch failed: 404")
    parsed = server_adapter.validate_json(server_adapter.dump_json(err))
    assert isinstance(parsed, IndexError)
    assert "404" in parsed.message


def test_assistant_text_delta_round_trip() -> None:
    """v0.7: per-token streaming chunk."""
    delta = AssistantTextDelta(text="useState ", chunk_index=3)
    parsed = server_adapter.validate_json(server_adapter.dump_json(delta))
    assert isinstance(parsed, AssistantTextDelta)
    assert parsed.text == "useState "
    assert parsed.chunk_index == 3


def test_assistant_stream_final_round_trip() -> None:
    """v0.7: stream terminator with citations + tool + iteration count."""
    final = AssistantStreamFinal(
        citations=[
            CitationRef(library="react", version="18.2.0", source="useState.md"),
        ],
        tool_used="search_docs",
        iterations=2,
    )
    parsed = server_adapter.validate_json(server_adapter.dump_json(final))
    assert isinstance(parsed, AssistantStreamFinal)
    assert parsed.iterations == 2
    assert parsed.tool_used == "search_docs"
    assert len(parsed.citations) == 1
    assert parsed.citations[0].source == "useState.md"


def test_settings_update_round_trip_partial() -> None:
    """v0.7: settings_update accepts any subset of the known knobs."""
    msg = SettingsUpdate(score_floor=0.10)
    parsed = client_adapter.validate_json(client_adapter.dump_json(msg))
    assert isinstance(parsed, SettingsUpdate)
    assert parsed.score_floor == 0.10
    assert parsed.chat_model is None
    assert parsed.max_iterations is None


def test_settings_update_round_trip_full() -> None:
    msg = SettingsUpdate(chat_model="gpt-4o", score_floor=0.20, max_iterations=4)
    parsed = client_adapter.validate_json(client_adapter.dump_json(msg))
    assert isinstance(parsed, SettingsUpdate)
    assert parsed.chat_model == "gpt-4o"
    assert parsed.score_floor == 0.20
    assert parsed.max_iterations == 4


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        client_adapter.validate_json('{"type": "fly_to_the_moon"}')


def test_wrong_direction_rejected() -> None:
    """A server-only message must not parse as a client message."""
    with pytest.raises(ValidationError):
        client_adapter.validate_json('{"type": "assistant_text", "text": "hi"}')
