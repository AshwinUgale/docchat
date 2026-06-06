"""Indexer tests.

Real network and real Qdrant are out of scope here - both are mocked via
unittest.mock for the orchestration path and httpx.MockTransport for the
fetches. End-to-end with a live Qdrant container is the v0.2.x integration
test that lives outside the default `pytest` run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from docchat_sidecar.indexer import (
    DocIndexer,
    _clean_mdx,
    _split_into_chunks,
    collection_name_for,
)
from docchat_sidecar.protocol import IndexComplete, IndexError, IndexProgress

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_qdrant() -> MagicMock:
    qdrant = MagicMock()
    qdrant.collection_exists = AsyncMock(return_value=False)
    qdrant.delete_collection = AsyncMock()
    qdrant.create_collection = AsyncMock()
    qdrant.upsert = AsyncMock()
    return qdrant


def _fake_openai(dim: int = 1536) -> MagicMock:
    """OpenAI client that returns a deterministic vector per input."""

    async def create(*, model: str, input: list[str]) -> object:
        del model
        response = MagicMock()
        response.data = [MagicMock(embedding=[float(i) / dim] * dim) for i, _ in enumerate(input)]
        return response

    openai = MagicMock()
    openai.embeddings = MagicMock()
    openai.embeddings.create = create
    return openai


async def _collect(it: AsyncIterator[object]) -> list[object]:
    return [item async for item in it]


# ---------------------------------------------------------------------------
# Collection naming
# ---------------------------------------------------------------------------


def test_collection_name_for_basic() -> None:
    assert collection_name_for("react", "18.2.0") == "react_18_2_0"


def test_collection_name_for_lowercases_and_replaces_special_chars() -> None:
    assert collection_name_for("React", "18.2.0-beta") == "react_18_2_0_beta"
    assert collection_name_for("@vue/core", "3.4.0") == "vue_core_3_4_0"


# ---------------------------------------------------------------------------
# MDX cleaning
# ---------------------------------------------------------------------------


def test_clean_mdx_strips_frontmatter() -> None:
    raw = "---\ntitle: useState\n---\n\nThe useState hook.\n"
    assert _clean_mdx(raw) == "The useState hook."


def test_clean_mdx_strips_imports_and_exports() -> None:
    raw = "import Foo from 'bar';\nexport const meta = {};\n\nReal content here.\n"
    cleaned = _clean_mdx(raw)
    assert "import" not in cleaned
    assert "export" not in cleaned
    assert "Real content here" in cleaned


# ---------------------------------------------------------------------------
# Chunk splitter
# ---------------------------------------------------------------------------


def test_split_emits_single_chunk_for_short_text() -> None:
    # v0.8: yields (text, section_heading) tuples.
    chunks = list(_split_into_chunks("Single paragraph.\n\nAnother short one."))
    assert len(chunks) == 1
    text, heading = chunks[0]
    assert "Single paragraph" in text
    assert heading is None  # no H2 in the source


def test_split_emits_multiple_chunks_when_target_exceeded() -> None:
    long_para = "x" * 1500
    text = "\n\n".join([long_para, long_para, long_para])
    chunks = list(_split_into_chunks(text))
    assert len(chunks) >= 2
    assert all(isinstance(c, tuple) and len(c) == 2 for c in chunks)


def test_split_skips_empty_paragraphs() -> None:
    chunks = list(_split_into_chunks("para one\n\n\n\npara two"))
    assert chunks == [("para one\n\npara two", None)]


def test_split_empty_text_yields_nothing() -> None:
    assert list(_split_into_chunks("")) == []
    assert list(_split_into_chunks("   \n   ")) == []


def test_split_captures_h2_heading_for_chunk_metadata() -> None:
    """v0.8: chunk should carry the most recent ## heading at its start."""
    raw = "## Reference\n\nIntro paragraph under Reference.\n\nAnother paragraph.\n"
    chunks = list(_split_into_chunks(raw))
    assert len(chunks) == 1
    text, heading = chunks[0]
    assert heading == "Reference"
    assert "Intro paragraph" in text


def test_split_heading_advances_across_h2_boundary() -> None:
    """A chunk that begins after a new H2 carries the new heading."""
    long = "x" * 1800
    raw = "## First Section\n\n" + long + "\n\n" + "## Second Section\n\n" + long + "\n"
    chunks = list(_split_into_chunks(raw))
    assert len(chunks) >= 2
    headings = [h for _, h in chunks]
    assert "First Section" in headings
    assert "Second Section" in headings


# ---------------------------------------------------------------------------
# End-to-end indexer flow
# ---------------------------------------------------------------------------


async def test_index_unsupported_library_yields_error() -> None:
    qdrant = _fake_qdrant()
    openai = _fake_openai()
    indexer = DocIndexer(qdrant=qdrant, openai=openai)

    events = await _collect(indexer.index("not_a_library", "1.0.0"))
    assert len(events) == 1
    assert isinstance(events[0], IndexError)
    assert "no indexer wired" in events[0].message


async def test_index_react_emits_complete_when_fetches_succeed() -> None:
    qdrant = _fake_qdrant()
    openai = _fake_openai()

    fixture_mdx = (
        "---\ntitle: useState\n---\n"
        "import X from 'y';\n\n"
        "The useState hook returns a stateful value and a function to update it.\n\n"
        "Pass an initial value as the argument.\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=fixture_mdx)

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    indexer = DocIndexer(qdrant=qdrant, openai=openai, http=http)

    try:
        events = await _collect(indexer.index("react", "18.2.0"))
    finally:
        await http.aclose()

    assert any(isinstance(e, IndexProgress) for e in events)
    completes = [e for e in events if isinstance(e, IndexComplete)]
    assert len(completes) == 1
    assert completes[0].chunks_indexed > 0
    # Qdrant was reset (no existing collection -> create_collection called).
    qdrant.create_collection.assert_awaited_once()
    qdrant.upsert.assert_awaited()


async def test_index_handles_fetch_404_per_url(caplog: pytest.LogCaptureFixture) -> None:
    """One 404 doesn't abort the run; remaining URLs still contribute."""
    qdrant = _fake_qdrant()
    openai = _fake_openai()

    fixture_mdx = "Content for the page.\n\nSecond paragraph here."
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        # First URL 404s; rest succeed.
        if call_count["n"] == 1:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=fixture_mdx)

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    indexer = DocIndexer(qdrant=qdrant, openai=openai, http=http)

    try:
        events = await _collect(indexer.index("react", "18.2.0"))
    finally:
        await http.aclose()

    completes = [e for e in events if isinstance(e, IndexComplete)]
    assert len(completes) == 1
    assert completes[0].chunks_indexed > 0


async def test_index_yields_error_when_all_fetches_return_empty() -> None:
    qdrant = _fake_qdrant()
    openai = _fake_openai()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    indexer = DocIndexer(qdrant=qdrant, openai=openai, http=http)

    try:
        events = await _collect(indexer.index("react", "18.2.0"))
    finally:
        await http.aclose()

    errors = [e for e in events if isinstance(e, IndexError)]
    assert len(errors) == 1
    assert "0 chunks" in errors[0].message
