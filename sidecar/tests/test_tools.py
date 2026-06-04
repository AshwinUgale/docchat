"""Tests for the v0.6 agent tools.

SearchDocsTool tests are v0.3/v0.5 (Qdrant retrieval + cosine floor).
v0.6 adds real implementations for the other two tools; tests for those
mock the subprocess (ripgrep) + httpx (GitHub raw fetch) respectively.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
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


def _fake_qdrant_with_hits(
    hits: list[dict] | None = None, scores: list[float] | None = None
) -> MagicMock:
    """Qdrant client where collection_exists=True and query_points returns the given hits.

    Returns a QueryResponse-shaped object with a ``points`` attribute - matches
    the current qdrant-client API (``search`` is deprecated in 1.10+).

    If ``scores`` is provided it must align 1:1 with ``hits``; otherwise the
    default schedule 0.9, 0.8, ... is used so the floor in v0.5 SearchDocsTool
    doesn't trip existing tests.
    """

    qdrant = MagicMock()
    qdrant.collection_exists = AsyncMock(return_value=True)
    payloads = hits or []
    score_schedule = (
        scores if scores is not None else [0.9 - i * 0.1 for i, _ in enumerate(payloads)]
    )

    async def query_points(**kwargs: object) -> object:
        del kwargs
        response = MagicMock()
        response.points = [
            MagicMock(payload=p, score=s) for p, s in zip(payloads, score_schedule, strict=False)
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


async def test_search_docs_drops_hits_below_score_floor() -> None:
    """v0.5 floor: a hit at 0.10 is below the 0.25 default, so it's dropped."""
    qdrant = _fake_qdrant_with_hits(
        [
            {"text": "Solid React useState chunk.", "source_url": "https://x/useState.md"},
            {"text": "Off-topic Flask chunk that slipped in.", "source_url": "https://x/flask.md"},
        ],
        scores=[0.55, 0.10],
    )
    openai = _fake_openai_with_query_embedding()
    tool = SearchDocsTool(qdrant=qdrant, openai=openai)
    result = await tool.run(library="react", version="18.2.0", query="how do I use state?")
    assert "useState" in result.text
    assert "Flask" not in result.text
    assert len(result.citations) == 1
    assert result.citations[0].source == "useState.md"


async def test_search_docs_returns_no_relevant_when_all_hits_below_floor() -> None:
    """All hits below the 0.25 floor -> empty -> canned refusal text."""
    qdrant = _fake_qdrant_with_hits(
        [
            {"text": "Vague match A.", "source_url": "https://x/a.md"},
            {"text": "Vague match B.", "source_url": "https://x/b.md"},
        ],
        scores=[0.12, 0.08],
    )
    openai = _fake_openai_with_query_embedding()
    tool = SearchDocsTool(qdrant=qdrant, openai=openai)
    result = await tool.run(library="react", version="18.2.0", query="how do I deploy flask?")
    assert "No relevant chunks" in result.text
    assert result.citations == []


async def test_search_docs_score_floor_is_configurable() -> None:
    """A caller can raise the floor to be stricter."""
    qdrant = _fake_qdrant_with_hits(
        [{"text": "Mid-confidence chunk.", "source_url": "https://x/mid.md"}],
        scores=[0.30],
    )
    openai = _fake_openai_with_query_embedding()
    strict = SearchDocsTool(qdrant=qdrant, openai=openai, score_floor=0.5)
    result = await strict.run(library="react", version="18.2.0", query="anything")
    assert "No relevant chunks" in result.text


# ---------------------------------------------------------------------------
# Stub tools
# ---------------------------------------------------------------------------


async def test_search_workspace_code_no_workspace_message() -> None:
    tool = SearchWorkspaceCodeTool(workspace_path=None)
    result = await tool.run(query="useState")
    assert "no workspace" in result.text.lower()
    assert result.citations == []


async def test_search_workspace_code_invalid_dir_message() -> None:
    tool = SearchWorkspaceCodeTool(workspace_path="/this/path/does/not/exist")
    result = await tool.run(query="anything")
    assert "not a directory" in result.text.lower()


async def test_search_workspace_code_handles_missing_rg(tmp_path: Path) -> None:
    """If the rg binary is unresolvable, the tool returns a guidance message."""
    tool = SearchWorkspaceCodeTool(
        workspace_path=tmp_path, rg_binary="rg_definitely_not_installed_xyz"
    )
    result = await tool.run(query="anything")
    assert "ripgrep" in result.text.lower()
    assert "not installed" in result.text.lower()


async def test_search_workspace_code_parses_real_rg(tmp_path: Path) -> None:
    """End-to-end real rg invocation. Skipped if rg isn't on PATH."""
    import shutil

    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed on this runner")

    sample = tmp_path / "hello.py"
    sample.write_text("def greet() -> str:\n    return 'hello DocChat'\n", encoding="utf-8")
    tool = SearchWorkspaceCodeTool(workspace_path=tmp_path)
    result = await tool.run(query="DocChat")
    assert "hello.py" in result.text
    assert "DocChat" in result.text


# ---------------------------------------------------------------------------
# FindInChangelogTool - real fetch + version-section grep
# ---------------------------------------------------------------------------


_CHANGELOG_FIXTURE = """\
## 18.3.0 (April 25, 2024)

- Some 18.3 feature.

## 18.2.0 (June 14, 2022)

- React added support for useSyncExternalStore.
- Strict Mode now intentionally double-invokes effects in development.
- Concurrent rendering APIs ship.

## 18.1.0 (April 26, 2022)

- Bug fixes only.
"""


async def test_find_in_changelog_returns_matching_section() -> None:
    """Real httpx fetch via MockTransport returns the changelog text; tool
    slices out the version section the user asked about."""

    # Clear the process-level cache so this test is hermetic regardless
    # of which other tests ran before.
    from docchat_sidecar.tools import _CHANGELOG_CACHE

    _CHANGELOG_CACHE.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_CHANGELOG_FIXTURE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        tool = FindInChangelogTool(http=http)
        result = await tool.run(library="react", version="18.2.0", query="useSyncExternalStore")
    finally:
        await http.aclose()

    assert "useSyncExternalStore" in result.text
    assert "18.2.0" in result.text
    assert "18.1.0" not in result.text  # other version section dropped
    assert len(result.citations) == 1
    assert result.citations[0].source == "CHANGELOG.md"


async def test_find_in_changelog_returns_message_for_unknown_library() -> None:
    tool = FindInChangelogTool()
    result = await tool.run(library="some_random_lib", version="1.0.0", query="anything")
    assert "no changelog source" in result.text.lower()


async def test_find_in_changelog_v0_6_1_does_not_leak_adjacent_version_context() -> None:
    """v0.6.1 fix: a body-mention of the requested version in a NEIGHBOURING
    section's body must NOT qualify that section. Otherwise React 19's
    section would leak into a React 18.2 query because the React 19
    release notes often mention "fixed in 18.2.0".
    """
    from docchat_sidecar.tools import _CHANGELOG_CACHE

    _CHANGELOG_CACHE.clear()

    fixture = (
        "## 19.0.0 (December 5, 2024)\n\n"
        "- Backports the 18.2.0 fix for Strict Mode double-invocation. "
        "Introduces use(promise) for unwrapping promises.\n\n"
        "## 18.2.0 (June 14, 2022)\n\n"
        "- Adds useSyncExternalStore.\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=fixture)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        tool = FindInChangelogTool(http=http)
        result = await tool.run(library="react", version="18.2.0", query="useSyncExternalStore")
    finally:
        await http.aclose()

    # The 18.2.0 section is included; the 19.0.0 section must NOT be,
    # even though it mentions "18.2.0" in its body. This is the leak that
    # caused the v0.6 version_correctness regression.
    assert "useSyncExternalStore" in result.text
    assert "use(promise)" not in result.text  # the React 19 forbidden API


async def test_find_in_changelog_v0_6_1_does_not_prefix_match_subversion() -> None:
    """v0.6.1 fix: version "0.95.0" must NOT match heading "0.95.10". The
    \b word-boundary in the new regex prevents the prefix-match leak."""
    from docchat_sidecar.tools import _CHANGELOG_CACHE

    _CHANGELOG_CACHE.clear()

    fixture = (
        "## 0.95.10\n\n"
        "- Patch release. Added Pydantic v2 helpers like model_dump().\n\n"
        "## 0.95.0\n\n"
        "- Initial 0.95 line. Introduces Annotated[X, Depends(...)] idiom.\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=fixture)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        tool = FindInChangelogTool(http=http)
        result = await tool.run(library="fastapi", version="0.95.0", query="Annotated")
    finally:
        await http.aclose()

    # 0.95.0 section included; 0.95.10 must NOT be (no prefix-leak).
    assert "Annotated" in result.text
    assert "model_dump" not in result.text  # the Pydantic-v2 forbidden API


async def test_find_in_changelog_returns_message_when_version_missing() -> None:
    """No section for the requested version -> tool returns 'no entry' text."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_CHANGELOG_FIXTURE)

    # Bust the per-process cache so a different URL is exercised fresh.
    from docchat_sidecar.tools import _CHANGELOG_CACHE

    _CHANGELOG_CACHE.clear()

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        tool = FindInChangelogTool(http=http)
        result = await tool.run(library="react", version="99.99.0", query="anything")
    finally:
        await http.aclose()

    assert "no changelog entry" in result.text.lower()


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
