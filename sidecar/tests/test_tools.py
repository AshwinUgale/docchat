"""Tests for the v0.3 agent tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from docchat_sidecar.tools import (
    Citation,
    FindInChangelogTool,
    SearchDocsTool,
    SearchWorkspaceCodeTool,
    tool_schemas,
)


def _fake_openai_with_query_embedding(dim: int = 1536) -> MagicMock:
    """OpenAI client whose embeddings.create returns a single deterministic vec."""

    async def create(*, model: str, input: list[str]) -> object:
        del model
        response = MagicMock()
        response.data = [MagicMock(embedding=[float(i) / dim] * dim) for i, _ in enumerate(input)]
        return response

    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = create
    return client


def _fake_qdrant_with_hits(hits: list[dict] | None = None) -> MagicMock:
    """Qdrant client where collection_exists=True and query_points returns the given hits.

    Returns a QueryResponse-shaped object with a ``points`` attribute - matches
    the current qdrant-client API (``search`` is deprecated in 1.10+).
    """

    qdrant = MagicMock()
    qdrant.collection_exists = AsyncMock(return_value=True)

    async def query_points(**kwargs: object) -> object:
        del kwargs
        response = MagicMock()
        response.points = [
            MagicMock(payload=h, score=0.9 - i * 0.1) for i, h in enumerate(hits or [])
        ]
        return response

    qdrant.query_points = query_points
    return qdrant


# ---------------------------------------------------------------------------
# Citation rendering
# ---------------------------------------------------------------------------


def test_citation_render() -> None:
    c = Citation(library="react", version="18.2.0", source="useState.md")
    assert c.render() == "[react@18.2.0:useState.md]"


# ---------------------------------------------------------------------------
# SearchDocsTool
# ---------------------------------------------------------------------------


async def test_search_docs_returns_chunks_and_citations() -> None:
    qdrant = _fake_qdrant_with_hits(
        [
            {
                "text": "useState returns a stateful value.",
                "source_url": "https://example/react/useState.md",
            },
            {
                "text": "Pass the initial value as argument.",
                "source_url": "https://example/react/useState.md",
            },
        ]
    )
    openai = _fake_openai_with_query_embedding()
    tool = SearchDocsTool(qdrant=qdrant, openai=openai)
    result = await tool.run(library="react", version="18.2.0", query="how do I use state?")
    assert "useState returns a stateful value" in result.text
    # Two hits from the same source file -> one deduped citation.
    assert len(result.citations) == 1
    assert result.citations[0].source == "useState.md"
    assert result.citations[0].library == "react"
    assert result.citations[0].version == "18.2.0"


async def test_search_docs_returns_message_when_collection_missing() -> None:
    qdrant = MagicMock()
    qdrant.collection_exists = AsyncMock(return_value=False)
    openai = _fake_openai_with_query_embedding()
    tool = SearchDocsTool(qdrant=qdrant, openai=openai)
    result = await tool.run(library="react", version="18.2.0", query="anything")
    assert "No indexed docs" in result.text
    assert result.citations == []


async def test_search_docs_returns_message_when_no_hits() -> None:
    qdrant = _fake_qdrant_with_hits([])
    openai = _fake_openai_with_query_embedding()
    tool = SearchDocsTool(qdrant=qdrant, openai=openai)
    result = await tool.run(library="react", version="18.2.0", query="nonsense xyz123")
    assert "No relevant chunks" in result.text


# ---------------------------------------------------------------------------
# Stub tools
# ---------------------------------------------------------------------------


async def test_search_workspace_code_stub_returns_placeholder() -> None:
    tool = SearchWorkspaceCodeTool()
    result = await tool.run(query="useState")
    assert "stub" in result.text.lower()
    assert result.citations == []


async def test_find_in_changelog_stub_returns_placeholder() -> None:
    tool = FindInChangelogTool()
    result = await tool.run(library="react", version="18.2.0", query="breaking changes")
    assert "stub" in result.text.lower()
    assert result.citations == []


# ---------------------------------------------------------------------------
# Schemas (consumed by ToolPicker)
# ---------------------------------------------------------------------------


def test_tool_schemas_cover_all_three_tools() -> None:
    schemas = tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == {
        SearchDocsTool.name,
        SearchWorkspaceCodeTool.name,
        FindInChangelogTool.name,
    }


def test_tool_schemas_have_descriptions_and_param_shapes() -> None:
    for schema in tool_schemas():
        assert isinstance(schema["description"], str)
        assert schema["description"]  # non-empty
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


@pytest.mark.parametrize(
    ("tool_name", "expected_required"),
    [
        ("search_docs", {"library", "version", "query"}),
        ("search_workspace_code", {"query"}),
        ("find_in_changelog", {"library", "version", "query"}),
    ],
)
def test_tool_schemas_required_fields(tool_name: str, expected_required: set[str]) -> None:
    schema = next(s for s in tool_schemas() if s["name"] == tool_name)
    assert set(schema["parameters"]["required"]) == expected_required
