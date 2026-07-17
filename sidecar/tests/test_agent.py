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
    agent = Agent(
        openai=openai,
        qdrant=qdrant,
        memory=memory,
        self_critique=False,
        topic_filter=False,
    )

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
    agent = Agent(
        openai=openai,
        qdrant=qdrant,
        memory=memory,
        self_critique=False,
        topic_filter=False,
    )

    await agent.answer("how do I use useState?")
    assert memory.manager.episodic.count() == 1


async def test_agent_reset_memory_clears_recorded_qa() -> None:
    openai = _fake_openai_scripted(
        tool_call_args={
            "name": "search_docs",
            "arguments": {"library": "react", "version": "18.2.0", "query": "useState"},
        },
        final_text="useState returns a stateful value and a setter.",
    )
    memory = _build_memory()
    agent = Agent(
        openai=openai,
        qdrant=_fake_qdrant_with_one_hit(),
        memory=memory,
        self_critique=False,
        topic_filter=False,
    )

    await agent.answer("how do I use useState?")
    assert memory.manager.episodic.count() == 1
    # The eval harness calls this between corpus entries to answer each cold.
    agent.reset_memory()
    assert memory.manager.episodic.count() == 0


def test_agent_forwards_floor_overrides_to_search_docs() -> None:
    # The eval harness overrides retrieval floors so calibration / untuned-
    # baseline runs don't require editing the production SearchDocsTool.
    agent = Agent(
        openai=MagicMock(),
        qdrant=MagicMock(),
        memory=_build_memory(),
        score_floor=0.30,
        floors_by_library={"fastapi": 0.12},
    )
    assert agent._search_docs._score_floor == 0.30
    assert agent._search_docs._floor_for("fastapi") == 0.12
    # Libraries without an override fall back to the (overridden) global floor.
    assert agent._search_docs._floor_for("react") == 0.30


def test_agent_default_floors_when_no_override() -> None:
    # No override -> SearchDocsTool keeps its own shipped defaults.
    agent = Agent(openai=MagicMock(), qdrant=MagicMock(), memory=_build_memory())
    assert agent._search_docs._score_floor == 0.15  # tool default


# ---------------------------------------------------------------------------
# Refusal path - canonical phrase makes it through unchanged
# ---------------------------------------------------------------------------


async def test_agent_emits_canonical_refusal_phrase() -> None:
    openai = _fake_openai_refuses_immediately()
    qdrant = _fake_qdrant_empty()
    memory = _build_memory()
    agent = Agent(
        openai=openai,
        qdrant=qdrant,
        memory=memory,
        self_critique=False,
        topic_filter=False,
    )

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

    agent = Agent(
        openai=client,
        qdrant=qdrant,
        memory=memory,
        max_iterations=2,
        topic_filter=False,
    )
    response = await agent.answer("anything")
    assert response.iterations == 2
    assert "iteration limit" in response.text.lower()


# ---------------------------------------------------------------------------
# Workspace path threads through to SearchWorkspaceCodeTool
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v0.8 self-critique - on by default in answer(); revises draft if critique
# returns non-"OK" text; falls through unchanged when critique returns "OK".
# ---------------------------------------------------------------------------


def _fake_openai_with_critique(
    *, tool_call_args: dict[str, object], draft_text: str, critique_reply: str
) -> MagicMock:
    """Scripted fake for the 3-call sequence: tool_call -> draft -> critique."""
    client = MagicMock()
    call_count = {"n": 0}

    async def embed(*, model: str, input: list[str]) -> object:
        del model
        response = MagicMock()
        response.data = [MagicMock(embedding=[float(i) / 1536] * 1536) for i, _ in enumerate(input)]
        return response

    async def chat(**kwargs: object) -> object:
        del kwargs
        call_count["n"] += 1
        choice = MagicMock()
        if call_count["n"] == 1:
            tc = MagicMock()
            tc.id = "call_1"
            tc.type = "function"
            tc.function = MagicMock()
            tc.function.name = tool_call_args["name"]
            tc.function.arguments = json.dumps(tool_call_args.get("arguments", {}))
            choice.message = MagicMock(content=None, tool_calls=[tc])
        elif call_count["n"] == 2:
            choice.message = MagicMock(content=draft_text, tool_calls=None)
        else:
            choice.message = MagicMock(content=critique_reply, tool_calls=None)
        response = MagicMock()
        response.choices = [choice]
        return response

    client.embeddings = MagicMock()
    client.embeddings.create = embed
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = chat
    return client


async def test_agent_self_critique_keeps_draft_when_ok() -> None:
    """Critique returns 'OK' -> the draft is shipped unchanged."""
    openai = _fake_openai_with_critique(
        tool_call_args={
            "name": "search_docs",
            "arguments": {"library": "react", "version": "18.2.0", "query": "useState"},
        },
        draft_text="useState returns a tuple of value + setter.",
        critique_reply="OK",
    )
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    agent = Agent(
        openai=openai,
        qdrant=qdrant,
        memory=memory,
        self_critique=True,
        topic_filter=False,
    )

    response = await agent.answer("how do I use useState?")
    assert "useState returns a tuple" in response.text


async def test_agent_self_critique_revises_when_not_ok() -> None:
    """Critique returns revised text -> the agent ships the revision, not the draft.

    The draft leaks a React-19 API (``use(promise)``). The critique reply
    is a clean rewrite that mentions neither the forbidden token nor the
    word "removed"; we verify the revision shipped by checking that
    distinct critique-only wording appears AND the forbidden token is
    absent from the body.
    """
    openai = _fake_openai_with_critique(
        tool_call_args={
            "name": "search_docs",
            "arguments": {"library": "react", "version": "18.2.0", "query": "useState"},
        },
        draft_text="useState returns a tuple. Also use the new use(promise) hook.",
        critique_reply="Call useState() at the top of your function component to get a value and setter.",
    )
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    agent = Agent(
        openai=openai,
        qdrant=qdrant,
        memory=memory,
        self_critique=True,
        topic_filter=False,
    )

    response = await agent.answer("how do I use useState?")
    body = response.text.split("Sources:")[0]
    # The revised wording shipped...
    assert "at the top of your function component" in body
    # ...and the React-19 leak that was in the draft is no longer present.
    assert "use(promise)" not in body


# ---------------------------------------------------------------------------
# v0.7 streaming - answer_stream() yields AssistantTextDelta then AssistantStreamFinal
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async iterator that yields a pre-baked list of MagicMock chunks."""

    def __init__(self, chunks: list[object]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


def _stream_chunk_tool_call(*, tool_name: str, tool_args_json: str) -> object:
    """One chunk carrying a partial tool_call (the full tool_call in one chunk
    is the simplest case OpenAI returns at low temperature)."""
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = None
    tc = MagicMock()
    tc.index = 0
    tc.id = "call_stream_1"
    tc.function = MagicMock()
    tc.function.name = tool_name
    tc.function.arguments = tool_args_json
    delta.tool_calls = [tc]
    chunk.choices = [MagicMock(delta=delta)]
    return chunk


def _stream_chunk_text(text: str) -> object:
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = None
    chunk.choices = [MagicMock(delta=delta)]
    return chunk


def _fake_openai_streaming(*, tool_args_json: str, final_chunks: list[str]) -> MagicMock:
    """Streaming OpenAI: iter 0 returns one tool_call chunk; iter 1+ stream text."""
    client = MagicMock()
    call_count = {"n": 0}

    async def embed(*, model: str, input: list[str]) -> object:
        del model
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.0] * 1536) for _ in input]
        return response

    async def chat(**kwargs: object) -> object:
        del kwargs
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeStream(
                [_stream_chunk_tool_call(tool_name="search_docs", tool_args_json=tool_args_json)]
            )
        return _FakeStream([_stream_chunk_text(t) for t in final_chunks])

    client.embeddings = MagicMock()
    client.embeddings.create = embed
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = chat
    return client


async def test_agent_answer_stream_yields_deltas_and_final() -> None:
    """v0.7 answer_stream: text chunks land as AssistantTextDelta, terminator
    as AssistantStreamFinal carrying citations + tool_used + iterations."""
    from docchat_sidecar.protocol import AssistantStreamFinal, AssistantTextDelta

    openai = _fake_openai_streaming(
        tool_args_json=json.dumps({"library": "react", "version": "18.2.0", "query": "x"}),
        final_chunks=["useState ", "returns ", "a tuple."],
    )
    qdrant = _fake_qdrant_with_one_hit()
    memory = _build_memory()
    agent = Agent(
        openai=openai,
        qdrant=qdrant,
        memory=memory,
        self_critique=False,
        topic_filter=False,
    )

    events: list[object] = [evt async for evt in agent.answer_stream("how use useState?")]

    deltas = [e for e in events if isinstance(e, AssistantTextDelta)]
    finals = [e for e in events if isinstance(e, AssistantStreamFinal)]

    # Three text chunks streamed + one citation block delta = 4.
    assert len(deltas) == 4
    text = "".join(d.text for d in deltas)
    assert "useState returns a tuple." in text
    assert "Sources:" in text
    # Monotonic chunk_index.
    assert [d.chunk_index for d in deltas] == [0, 1, 2, 3]
    # Exactly one terminator with the right shape.
    assert len(finals) == 1
    assert finals[0].tool_used == "search_docs"
    assert finals[0].iterations == 2
    assert len(finals[0].citations) == 1


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
        topic_filter=False,
    )
    # We just need the tool to exist and not blow up. The real ripgrep call
    # may produce a "no workspace / no rg" message; both are valid signals.
    response = await agent.answer("where do I use useState?")
    assert response.tool_used == "search_workspace_code"
