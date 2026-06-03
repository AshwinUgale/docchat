"""ReAct-ish agent loop with Mneme + ToolPicker integration.

v0.3 ships single-iteration: receive query -> ToolPicker routes over 3
tools -> selected tool runs and returns retrieval context -> Mneme
surfaces past Q/A pairs from the same workspace -> OpenAI chat completion
with both as context -> record the new Q/A into Mneme -> return one
AssistantText with citation tokens.

Multi-iteration ReAct (model decides whether to call another tool after
seeing the first tool's output) lands at v0.4. Self-critique pass (model
re-reads its own draft against the retrieved sources for groundedness)
lands at v1.0 per the master doc.

Streaming token deltas land at v0.3.x. v0.3 emits one ``AssistantText``
per query; the WebSocket protocol already supports streaming chunks but
the loop needs an async generator wrapping OpenAI's stream API which we
defer for now.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from qdrant_client import AsyncQdrantClient
from toolpicker import FunctionSchemaSource, ToolPicker

from docchat_sidecar.memory import WorkspaceMemory
from docchat_sidecar.tools import (
    Citation,
    FindInChangelogTool,
    SearchDocsTool,
    SearchWorkspaceCodeTool,
    ToolResult,
    tool_schemas,
)

__all__ = ["Agent", "AgentResponse"]

logger = logging.getLogger(__name__)


# v0.3 system prompt - keeps the agent honest about its scope.
_SYSTEM_PROMPT = """You are DocChat, an assistant that answers questions about \
software libraries at the exact version a user's project pins. You ground every \
answer in the retrieved context provided below. If the context does not contain \
the answer, say so explicitly - DO NOT fabricate APIs, signatures, or behaviour. \
When citing, refer to source filenames as they appear in the retrieved context. \
Keep responses concise unless the user asks for depth."""


@dataclass(kw_only=True)
class AgentResponse:
    """What the agent loop produces per query."""

    text: str
    citations: list[Citation]
    tool_used: str


class Agent:
    """Single-iteration ReAct loop for one user query.

    Constructed once per WebSocket connection; ``answer()`` is called per
    incoming user_query. The agent owns:

    * A ``ToolPicker`` configured with the three v0.3 tool schemas.
    * The three tool instances (``SearchDocsTool`` real, two stubs).
    * The per-workspace ``WorkspaceMemory`` (Mneme).
    * The OpenAI client (for chat completions).

    The ``chat_model`` default is ``gpt-4o-mini`` because the v0.3 ReAct
    work is small-context: ~5 retrieved chunks + ~3 past Q/As + a single
    user question, total well under 32k tokens. Costs ~$0.0001 per turn.
    """

    def __init__(
        self,
        *,
        openai: AsyncOpenAI,
        qdrant: AsyncQdrantClient,
        memory: WorkspaceMemory,
        chat_model: str = "gpt-4o-mini",
        default_library: str = "react",
        default_version: str = "18.2.0",
    ) -> None:
        self._openai = openai
        self._memory = memory
        self._chat_model = chat_model
        self._default_library = default_library
        self._default_version = default_version

        self._search_docs = SearchDocsTool(qdrant=qdrant, openai=openai)
        self._search_workspace = SearchWorkspaceCodeTool()
        self._find_changelog = FindInChangelogTool()

        # Build ToolPicker over the three schemas. v0.3 uses BM25-only
        # (embedder=None) because we only have 3 tools and there's no win
        # from semantic + BM25 + RRF at that scale. v0.4 with more tools
        # turns on the full hybrid stack.
        self._picker = ToolPicker(FunctionSchemaSource(tool_schemas()))

    async def answer(self, query: str) -> AgentResponse:
        """Run one query through the loop and return a full response."""
        # 1. ToolPicker route - which tool best matches this query?
        selected = self._picker.select(query, k=1)
        if not selected:
            return AgentResponse(
                text="I don't know which tool to use for that question.",
                citations=[],
                tool_used="(none)",
            )
        tool_name = selected[0].id
        logger.info("agent routing query %r -> tool %s", query, tool_name)

        # 2. Dispatch to the picked tool to gather context.
        tool_result = await self._dispatch(tool_name, query)

        # 3. Pull relevant past Q/As from Mneme - workspace-scoped memory.
        past_turns = self._memory.retrieve_relevant(query, k=3)

        # 4. Compose the LLM prompt: system + retrieved context + past turns
        #    + the user question. Then call OpenAI chat.
        messages = self._compose_messages(query, tool_result, past_turns)
        completion = await self._openai.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            temperature=0.2,
        )
        answer_text = completion.choices[0].message.content or ""

        # 5. Append citation tokens inline at the end so the panel can
        #    render them as a Sources section without parsing free text.
        if tool_result.citations:
            citation_block = " ".join(c.render() for c in tool_result.citations)
            answer_text = f"{answer_text}\n\nSources: {citation_block}"

        # 6. Record this Q/A in workspace memory so future turns can use it.
        try:
            self._memory.record_qa(
                query=query,
                answer=answer_text,
                citations=[c.render() for c in tool_result.citations],
            )
        except Exception as exc:  # pragma: no cover - memory write shouldn't kill the response
            logger.warning("failed to record Q/A in Mneme: %s", exc)

        return AgentResponse(text=answer_text, citations=tool_result.citations, tool_used=tool_name)

    async def _dispatch(self, tool_name: str, query: str) -> ToolResult:
        """Call the named tool with sensible argument defaults.

        v0.3 hardcodes the library/version to the demo target (React 18.2);
        v0.4 parses the user's lockfile pins to fill these in dynamically.
        """
        if tool_name == self._search_docs.name:
            return await self._search_docs.run(
                library=self._default_library,
                version=self._default_version,
                query=query,
            )
        if tool_name == self._search_workspace.name:
            return await self._search_workspace.run(query=query)
        if tool_name == self._find_changelog.name:
            return await self._find_changelog.run(
                library=self._default_library,
                version=self._default_version,
                query=query,
            )
        return ToolResult(text=f"[unknown tool: {tool_name}]")

    def _compose_messages(
        self,
        query: str,
        tool_result: ToolResult,
        past_turns: list[str],
    ) -> list[ChatCompletionMessageParam]:
        """Build the chat-completions message list."""
        context_sections: list[str] = []
        if tool_result.text:
            context_sections.append(f"## Retrieved context\n\n{tool_result.text}")
        if past_turns:
            past_text = "\n\n".join(f"- {t}" for t in past_turns)
            context_sections.append(f"## Past Q/A in this workspace\n\n{past_text}")
        context_block = (
            "\n\n".join(context_sections)
            if context_sections
            else "(no retrieved context for this query)"
        )
        system_with_context = f"{_SYSTEM_PROMPT}\n\n{context_block}"
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system_with_context,
        }
        user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": query}
        return [system_msg, user_msg]
