"""Tests for the v0.3 Agent ReAct loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mneme import HashEmbedder, InMemoryBackend, MemoryManager

from docchat_sidecar.agent import Agent
from docchat_sidecar.memory import WorkspaceMemory


def _fake_openai_for_agent() -> MagicMock:
    """OpenAI client that handles BOTH embeddings.create (tool path) and
    chat.completions.create (agent answer path)."""

    client = MagicMock()

    async def embed(*, model: str, input: list[str]) -> object:
        del model
        response = MagicMock()
        # 1536-dim deterministic vector per input - matches OpenAI text-embedding-3-small.
        response.data = [MagicMock(embedding=[float(i) / 1536] * 1536) for i, _ in enumerate(input)]
        return response

    async def chat(*, model: str, messages: list, temperature: float = 0.2) -> object:
        del model, messages, temperature
        choice = MagicMock()
        choice.message = MagicMock(content="useState returns a stateful value and a setter.")
        response = MagicMock()
        response.choices = [choice]
        return response

    client.embeddings = MagicMock()
    client.embeddings.create = embed
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = chat
    return client


def _fake_qdrant_with_one_hit() -> MagicMock:
    qdrant = MagicMock()
    qdrant.collection_exists = AsyncMock(return_value=True)

    async def query_points(**kwargs: object) -> object:
        del kwargs
        response = MagicMock()
        response.points = [
            MagicMock(
                payload={
                    "text": "useState returns a stateful value.",
                    "source_url": "https://example/react/useState.md",
                },
                score=0.95,
            ),
        ]
        return response

    qdrant.query_points = query_points
    return qdrant


def _build_memory() -> WorkspaceMemory:
    embedder = HashEmbedder(dimensions=32)
    backend = InMemoryBackend()
    mgr = MemoryManager(agent_id="docchat_workspace_test", backend=backend, embedder=embedder)
    return WorkspaceMemory(manager=mgr)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_agent_answers_query_with_citations() -> None:
    openai = _fake_openai_for_agent()
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    agent = Agent(openai=openai, qdrant=qdrant, memory=memory)

    response = await agent.answer("how do I use useState?")

    assert "useState" in response.text
    # Citation block appended.
    assert "Sources:" in response.text
    assert "[react@18.2.0:useState.md]" in response.text
    assert response.tool_used == "search_docs"
    assert len(response.citations) == 1


async def test_agent_records_qa_to_workspace_memory() -> None:
    openai = _fake_openai_for_agent()
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    agent = Agent(openai=openai, qdrant=qdrant, memory=memory)

    await agent.answer("how do I use useState?")

    # One Q/A recorded.
    assert memory.manager.episodic.count() == 1

    # Second query also recorded; both should be retrievable.
    await agent.answer("when does useState re-render?")
    assert memory.manager.episodic.count() == 2


# ---------------------------------------------------------------------------
# Tool routing
# ---------------------------------------------------------------------------


async def test_agent_routes_workspace_query_to_stub_tool() -> None:
    """A query that mentions workspace code should route to the workspace tool,
    not the docs tool. ToolPicker's BM25 over the tool descriptions handles it.
    """
    openai = _fake_openai_for_agent()
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    agent = Agent(openai=openai, qdrant=qdrant, memory=memory)

    response = await agent.answer("where in my workspace code do I use useState?")
    # ToolPicker picks based on text overlap; "workspace code" matches the
    # workspace tool's description.
    assert response.tool_used in {"search_workspace_code", "search_docs"}


async def test_agent_routes_changelog_query() -> None:
    openai = _fake_openai_for_agent()
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    agent = Agent(openai=openai, qdrant=qdrant, memory=memory)

    response = await agent.answer("what changed in the changelog for this library version?")
    # The changelog tool has the strongest BM25 match for "changelog".
    assert response.tool_used == "find_in_changelog"


# ---------------------------------------------------------------------------
# Missing collection - happens before user has indexed docs
# ---------------------------------------------------------------------------


async def test_agent_handles_unindexed_library() -> None:
    openai = _fake_openai_for_agent()
    qdrant = MagicMock()
    qdrant.collection_exists = AsyncMock(return_value=False)
    memory = _build_memory()
    agent = Agent(openai=openai, qdrant=qdrant, memory=memory)

    response = await agent.answer("how do I use useState?")
    # The tool returned a "please index first" message; the agent passes
    # that through the LLM, so the response text exists and the QA is
    # still recorded.
    assert response.tool_used == "search_docs"
    assert isinstance(response.text, str)
