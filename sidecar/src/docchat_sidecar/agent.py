"""ReAct agent loop with Mneme + ToolPicker integration.

v0.6 ships multi-iteration ReAct via OpenAI's tool-calling API: receive
query -> Mneme surfaces past Q/As from the same workspace -> the model
gets the system prompt + past turns + user query + function-call schemas
for the three tools -> on each iteration the model either calls a tool
(we dispatch, append the tool's text to the message history, loop) or
emits a final text answer. Capped at ``max_iterations`` (default 3) so a
broken model can't pin the loop. Citations aggregate across iterations.

ToolPicker still preselects candidate tools per query - at v0.6 it always
returns all 3, but the architecture scales when v0.7+ adds more tools.
Self-critique pass (model re-reads its own draft against the retrieved
sources for groundedness) lands at v1.0 per the master doc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
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


# v0.6 system prompt (REVERTED from v0.6.1's softening attempt - that
# version made the model treat any loose tool output as "useful context"
# and synthesise wrong-version answers from base knowledge. Eval at
# v0.6.1 attempt: in_scope=29/30 but accuracy=0.000 and version=0.207
# because the model dressed up off-topic retrievals into hallucinated
# answers. Reverted to the stricter v0.6 shape. Recall stays bounded
# but precision and refusal-correctness are preserved. The prompt-vs-
# floor recall problem is a v0.7 item that needs a structural fix
# (per-collection thresholds, chunk metadata) rather than wording).
#
# Shape changed from v0.5 because the model now drives tool dispatch
# via the function-calling API rather than receiving tool output
# embedded in the system prompt. The HARD RULES still produce the
# canonical refusal phrase when retrieval is empty so the eval's
# is_refusal heuristic continues to match.
_SYSTEM_PROMPT = """You are DocChat, an assistant that answers questions about \
software libraries at the exact version a user's project pins.

You have three tools:
- search_docs(library, version, query): the canonical first move for any \
"how do I" question about an indexed library.
- search_workspace_code(query): grep the user's open VS Code workspace for \
how they're using a specific API. Use for "where in my code do I..." \
questions.
- find_in_changelog(library, version, query): fetch a library's CHANGELOG \
entry for the pinned version. Use for "what changed in" / "was X added in" \
questions.

HARD RULES (apply before generating anything else):
1. Call at least one tool before responding to any technical question. Do \
NOT answer from general knowledge.
2. After the tool runs: if its output contains "No relevant chunks", "No \
indexed docs", or is otherwise empty / off-topic for the question, reply \
EXACTLY: "I don't have documentation for that in this workspace's indexed \
libraries." Do NOT fall back to base knowledge - even if you know the \
answer, refuse.
3. If the tool returned useful context, ground your answer in it. Do NOT \
fabricate APIs, signatures, or behaviour. If the retrieved context only \
partially covers the question, answer the covered part and explicitly say \
what's not covered.

When citing, refer to source filenames as they appear in the retrieved \
context. Keep responses concise unless the user asks for depth."""


@dataclass(kw_only=True)
class AgentResponse:
    """What the agent loop produces per query."""

    text: str
    citations: list[Citation]
    tool_used: str
    iterations: int = 1  # v0.6: number of tool-call iterations actually run.


class Agent:
    """Multi-iteration ReAct loop for one user query.

    Constructed once per WebSocket connection; ``answer()`` is called per
    incoming user_query. The agent owns:

    * A ``ToolPicker`` configured with the v0.6 tool schemas (preselect; the
      model picks the actual tool per iteration).
    * The three tool instances (``SearchDocsTool`` real, two more real at
      v0.6: ``SearchWorkspaceCodeTool`` via ripgrep, ``FindInChangelogTool``
      via GitHub raw fetch).
    * The per-workspace ``WorkspaceMemory`` (Mneme).
    * The OpenAI client (for chat completions + embeddings).

    Args:
        openai: AsyncOpenAI client (shared by tools + chat).
        qdrant: AsyncQdrantClient for retrieval.
        memory: ``WorkspaceMemory`` already constructed for the current
            workspace path (the namespace hash policy lives in
            ``build_memory``).
        workspace_path: Absolute path to the open VS Code workspace, or
            ``None`` if no workspace is open. Used by
            ``SearchWorkspaceCodeTool`` to scope the ripgrep search.
        chat_model: OpenAI chat model. ``gpt-4o-mini`` default - cheap
            and handles tool-calling correctly at v0.6 scale.
        default_library / default_version: Fallback (library, version)
            pair when the model doesn't pass them as tool args. At v0.6
            the LLM infers from the user's question; v0.7 wires lockfile
            context into the prompt so the LLM picks the right pin.
        max_iterations: Hard cap on ReAct iterations. 3 is enough for
            "search docs -> didn't cover it -> search workspace" chains
            without letting a confused model spin indefinitely.
    """

    def __init__(
        self,
        *,
        openai: AsyncOpenAI,
        qdrant: AsyncQdrantClient,
        memory: WorkspaceMemory,
        workspace_path: str | None = None,
        chat_model: str = "gpt-4o-mini",
        default_library: str = "react",
        default_version: str = "18.2.0",
        max_iterations: int = 3,
    ) -> None:
        self._openai = openai
        self._memory = memory
        self._workspace_path = workspace_path
        self._chat_model = chat_model
        self._default_library = default_library
        self._default_version = default_version
        self._max_iterations = max(1, max_iterations)

        self._search_docs = SearchDocsTool(qdrant=qdrant, openai=openai)
        self._search_workspace = SearchWorkspaceCodeTool(workspace_path=workspace_path)
        self._find_changelog = FindInChangelogTool()

        self._tools_by_name = {
            self._search_docs.name: self._search_docs,
            self._search_workspace.name: self._search_workspace,
            self._find_changelog.name: self._find_changelog,
        }

        # ToolPicker over the three schemas. With 3 tools, it always returns
        # all of them and the model picks; the architecture stays in place
        # so v0.7+ can grow to N>3 tools without restructuring.
        self._picker = ToolPicker(FunctionSchemaSource(tool_schemas()))

    async def answer(self, query: str) -> AgentResponse:
        """Run one query through the multi-iteration ReAct loop."""
        # 1. ToolPicker preselects candidate tools. At v0.6 with 3 tools
        #    we always pass all of them through; the LLM picks per iteration.
        candidate_names = [s.id for s in self._picker.select(query, k=10)]
        if not candidate_names:
            candidate_names = list(self._tools_by_name.keys())
        tool_specs = self._openai_tool_specs(candidate_names)

        # 2. Pull relevant past Q/As from Mneme - workspace-scoped memory.
        past_turns = self._memory.retrieve_relevant(query, k=3)

        # 3. Build the initial chat-completion message list.
        messages: list[ChatCompletionMessageParam] = self._initial_messages(
            query=query, past_turns=past_turns
        )

        # 4. ReAct loop. On each iteration the model either:
        #    - emits tool_calls (we dispatch, append tool messages, loop)
        #    - emits content text (we return)
        citations: list[Citation] = []
        last_tool_used = "(none)"
        iterations_run = 0

        for iteration in range(self._max_iterations):
            iterations_run = iteration + 1
            completion = await self._openai.chat.completions.create(
                model=self._chat_model,
                messages=messages,
                tools=tool_specs,
                # First iteration: force the model to call a tool. After
                # that, allow it to answer when it has enough context.
                tool_choice="required" if iteration == 0 else "auto",
                temperature=0.2,
            )
            msg = completion.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                # Final answer.
                answer_text = msg.content or ""
                return self._finalize(
                    query=query,
                    answer_text=answer_text,
                    citations=citations,
                    tool_used=last_tool_used,
                    iterations=iterations_run,
                )

            # Append the assistant's tool-call request to history.
            messages.append(_assistant_tool_calls_message(msg, tool_calls))

            # Dispatch every requested tool in order and append results.
            for tc in tool_calls:
                # OpenAI's typed union covers ``custom`` tool calls too; we
                # only configured ``function`` tools, so narrow before
                # accessing ``.function`` and skip anything else.
                if tc.type != "function":
                    logger.warning("agent: ignoring non-function tool_call type=%s", tc.type)
                    continue
                tool_name = tc.function.name
                try:
                    raw_args = tc.function.arguments or "{}"
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    logger.warning(
                        "agent: malformed tool args from model: %r", tc.function.arguments
                    )
                    args = {}
                result = await self._dispatch(tool_name=tool_name, args=args, query=query)
                citations.extend(result.citations)
                last_tool_used = tool_name
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.text,
                    }
                )

        # Iteration cap hit without a final text response. Surface that as
        # an honest refusal rather than a hallucinated answer.
        logger.warning("agent hit max_iterations=%d for query %r", self._max_iterations, query)
        return self._finalize(
            query=query,
            answer_text=(
                "I reached my iteration limit looking for an answer. "
                "Try rephrasing the question or indexing the relevant library."
            ),
            citations=citations,
            tool_used=last_tool_used,
            iterations=iterations_run,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _finalize(
        self,
        *,
        query: str,
        answer_text: str,
        citations: list[Citation],
        tool_used: str,
        iterations: int,
    ) -> AgentResponse:
        """Append inline citations + record the Q/A in Mneme + return."""
        if citations:
            citation_block = " ".join(c.render() for c in _dedupe_citations(citations))
            answer_text = f"{answer_text}\n\nSources: {citation_block}"
        try:
            self._memory.record_qa(
                query=query,
                answer=answer_text,
                citations=[c.render() for c in citations],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("failed to record Q/A in Mneme: %s", exc)
        return AgentResponse(
            text=answer_text,
            citations=_dedupe_citations(citations),
            tool_used=tool_used,
            iterations=iterations,
        )

    async def _dispatch(self, *, tool_name: str, args: dict[str, Any], query: str) -> ToolResult:
        """Run the named tool with the model's args, defaulting where missing.

        v0.6 lets the LLM pass ``library``/``version``/``query`` via
        function-call args. If those are missing we fall back to the
        Agent's defaults (so the tool always gets workable inputs).
        """
        tool = self._tools_by_name.get(tool_name)
        if tool is None:
            return ToolResult(text=f"[unknown tool: {tool_name}]")
        library = str(args.get("library") or self._default_library)
        version = str(args.get("version") or self._default_version)
        tool_query = str(args.get("query") or query)
        if tool_name == self._search_docs.name:
            return await self._search_docs.run(library=library, version=version, query=tool_query)
        if tool_name == self._search_workspace.name:
            return await self._search_workspace.run(query=tool_query)
        if tool_name == self._find_changelog.name:
            return await self._find_changelog.run(
                library=library, version=version, query=tool_query
            )
        return ToolResult(text=f"[unknown tool: {tool_name}]")

    def _initial_messages(
        self, *, query: str, past_turns: list[str]
    ) -> list[ChatCompletionMessageParam]:
        """Build the starting message list: system + (past Q/A) + user."""
        system_content = _SYSTEM_PROMPT
        if past_turns:
            past_text = "\n\n".join(f"- {t}" for t in past_turns)
            system_content = f"{_SYSTEM_PROMPT}\n\n## Past Q/A in this workspace\n\n{past_text}"
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        return messages

    def _openai_tool_specs(self, names: list[str]) -> list[ChatCompletionToolParam]:
        """Wrap each tool schema in OpenAI's ``{type: function, function: {...}}`` envelope."""
        wanted = set(names)
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["parameters"],
                },
            }
            for s in tool_schemas()
            if s["name"] in wanted
        ]


def _assistant_tool_calls_message(msg: Any, tool_calls: list[Any]) -> ChatCompletionMessageParam:
    """Build the assistant message echoing the model's tool-call request.

    OpenAI's chat-completions API requires the assistant turn that produced
    the tool_calls to be present in subsequent calls so the tool-result
    messages have something to attach to.
    """
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in tool_calls
        ],
    }


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    """Same source-file in multiple iterations collapses to one citation."""
    seen: set[tuple[str, str, str]] = set()
    out: list[Citation] = []
    for c in citations:
        key = (c.library, c.version, c.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
