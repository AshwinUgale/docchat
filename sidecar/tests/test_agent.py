"""Tests for the v0.6 multi-iteration ReAct Agent loop.

The agent now uses OpenAI's tool-calling API; the fake client below
mirrors the shape the real ``chat.completions.create`` returns:

* `response.choices[0].message.tool_calls` is a list of ToolCall objects,
  each with ``id``, ``function.name``, ``function.arguments`` (JSON str).
* `response.choices[0].message.content` is the final text answer once the
  model is done iterating.

We script the fake to return tool_calls on iteration 0 and content text
on iteration 1, exercising one tool dispatch + final answer per test.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from mneme import HashEmbedder, InMemoryBackend, MemoryManager

from docchat_sidecar.agent import Agent
from docchat_sidecar.memory import WorkspaceMemory


def _fake_openai_scripted(*, tool_call_args: dict[str, object], final_text: str) -> MagicMock:
    """OpenAI client that returns one tool_call then a final text message.

    Iteration 0: tool_calls=[search_docs(library=..., version=..., query=...)]
    Iteration 1+: content=final_text, tool_calls=None
    Also handles embeddings.create for SearchDocsTool's query embed.
    """
    client = MagicMock()
    call_count = {"n": 0}

    async def embed(*, model: str, input: list[str]) -> object:
        del model
        response = MagicMock()
        response.data = [MagicMock(embedding=[float(i) / 1536] * 1536) for i, _ in enumerate(input)]
        return response

    async def chat(
        *,
        model: str,
        messages: list,
        tools: list | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.2,
    ) -> object:
        del model, messages, tools, tool_choice, temperature
        call_count["n"] += 1
        choice = MagicMock()
        if call_count["n"] == 1:
            # Iteration 0: tool call. The narrowing guard in the agent
            # checks ``tc.type == "function"`` before reading ``tc.function``,
            # so the fake must set type explicitly (MagicMock-default would
            # be a MagicMock instance that fails the equality check).
            tc = MagicMock()
            tc.id = "call_1"
            tc.type = "function"
            tc.function = MagicMock()
            tc.function.name = tool_call_args["name"]
            tc.function.arguments = json.dumps(tool_call_args.get("arguments", {}))
            choice.message = MagicMock(content=None, tool_calls=[tc])
        else:
            # Iteration 1: final text.
            choice.message = MagicMock(content=final_text, tool_calls=None)
        response = MagicMock()
        response.choices = [choice]
        return response

    client.embeddings = MagicMock()
    client.embeddings.create = embed
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = chat
    return client


def _fake_openai_refuses_immediately() -> MagicMock:
    """Fake OpenAI that calls one tool then refuses with the canonical phrase."""
    return _fake_openai_scripted(
        tool_call_args={
            "name": "search_docs",
            "arguments": {"library": "react", "version": "18.2.0", "query": "anything"},
        },
        final_text="I don't have documentation for that in this workspace's indexed libraries.",
    )


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


def _fake_qdrant_empty() -> MagicMock:
    qdrant = MagicMock()
    qdrant.collection_exists = AsyncMock(return_value=True)

    async def query_points(**kwargs: object) -> object:
        del kwargs
        response = MagicMock()
        response.points = []
        return response

    qdrant.query_points = query_points
    return qdrant


def _build_memory() -> WorkspaceMemory:
    embedder = HashEmbedder(dimensions=32)
    backend = InMemoryBackend()
    mgr = MemoryManager(agent_id="docchat_workspace_test", backend=backend, embedder=embedder)
    return WorkspaceMemory(manager=mgr)


# ---------------------------------------------------------------------------
# Happy path - one tool call + one final answer
# ---------------------------------------------------------------------------


async def test_agent_answers_query_with_citations() -> None:
    openai = _fake_openai_scripted(
        tool_call_args={
            "name": "search_docs",
            "arguments": {"library": "react", "version": "18.2.0", "query": "useState"},
        },
        final_text="useState returns a stateful value and a setter.",
    )
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    agent = Agent(openai=openai, qdrant=qdrant, memory=memory)

    response = await agent.answer("how do I use useState?")

    assert "useState" in response.text
    assert "Sources:" in response.text
    assert "[react@18.2.0:useState.md]" in response.text
    assert response.tool_used == "search_docs"
    assert response.iterations == 2  # 1 tool call + 1 final
    assert len(response.citations) == 1


async def test_agent_records_qa_to_workspace_memory() -> None:
    openai = _fake_openai_scripted(
        tool_call_args={
            "name": "search_docs",
            "arguments": {"library": "react", "version": "18.2.0", "query": "useState"},
        },
        final_text="useState returns a stateful value and a setter.",
    )
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    agent = Agent(openai=openai, qdrant=qdrant, memory=memory)

    await agent.answer("how do I use useState?")
    assert memory.manager.episodic.count() == 1


# ---------------------------------------------------------------------------
# Refusal path - canonical phrase makes it through unchanged
# ---------------------------------------------------------------------------


async def test_agent_emits_canonical_refusal_phrase() -> None:
    openai = _fake_openai_refuses_immediately()
    qdrant = _fake_qdrant_empty()
    memory = _build_memory()
    agent = Agent(openai=openai, qdrant=qdrant, memory=memory)

    response = await agent.answer("how do I configure CORS in Flask?")
    # The eval's is_refusal heuristic relies on this exact substring.
    assert "i don't have" in response.text.lower()


# ---------------------------------------------------------------------------
# Iteration cap - if the model never finalizes, agent breaks out
# ---------------------------------------------------------------------------


async def test_agent_caps_at_max_iterations() -> None:
    """Model that ALWAYS returns tool_calls should hit the iteration cap."""
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()

    client = MagicMock()

    async def embed(*, model: str, input: list[str]) -> object:
        del model
        response = MagicMock()
        response.data = [MagicMock(embedding=[float(i) / 1536] * 1536) for i, _ in enumerate(input)]
        return response

    async def chat(**kwargs: object) -> object:
        del kwargs
        tc = MagicMock()
        tc.id = "call_x"
        tc.type = "function"
        tc.function = MagicMock()
        tc.function.name = "search_docs"
        tc.function.arguments = json.dumps({"library": "react", "version": "18.2.0", "query": "x"})
        choice = MagicMock()
        choice.message = MagicMock(content=None, tool_calls=[tc])
        response = MagicMock()
        response.choices = [choice]
        return response

    client.embeddings = MagicMock()
    client.embeddings.create = embed
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = chat

    agent = Agent(openai=client, qdrant=qdrant, memory=memory, max_iterations=2)
    response = await agent.answer("anything")
    assert response.iterations == 2
    assert "iteration limit" in response.text.lower()


# ---------------------------------------------------------------------------
# Workspace path threads through to SearchWorkspaceCodeTool
# ---------------------------------------------------------------------------


async def test_agent_passes_workspace_path_to_tool() -> None:
    """The Agent's workspace_path kwarg should reach the workspace tool."""
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    openai = _fake_openai_scripted(
        tool_call_args={"name": "search_workspace_code", "arguments": {"query": "useState"}},
        final_text="No workspace matches.",
    )
    agent = Agent(
        openai=openai,
        qdrant=qdrant,
        memory=memory,
        workspace_path="/tmp/some-workspace",
    )
    # We just need the tool to exist and not blow up. The real ripgrep call
    # may produce a "no workspace / no rg" message; both are valid signals.
    response = await agent.answer("where do I use useState?")
    assert response.tool_used == "search_workspace_code"
