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
from collections.abc import AsyncIterator
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
from docchat_sidecar.protocol import (
    AssistantStreamFinal,
    AssistantTextDelta,
    CitationRef,
)
from docchat_sidecar.tools import (
    Citation,
    FindInChangelogTool,
    SearchDocsTool,
    SearchWorkspaceCodeTool,
    ToolResult,
    tool_schemas,
)

__all__ = ["Agent", "AgentResponse"]

# Server-message events the streaming agent can emit per query.
StreamEvent = AssistantTextDelta | AssistantStreamFinal

# v1.0: canonical refusal phrase shared by the topic-filter short-circuit and
# the system prompt's HARD RULE #2. Single source of truth so the eval's
# is_refusal heuristic always matches both paths.
_CANONICAL_REFUSAL = "I don't have documentation for that in this workspace's indexed libraries."

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
4. VERSION GROUNDING (v0.7): retrieved chunks are prefixed with their \
library@version (e.g. "## react@18.2.0 - useState.md"). Any API name or \
signature you put in your answer MUST appear in the retrieved chunks for \
the user's pinned version. If the right answer would require an API only \
available in a newer version (e.g. React 19's use(promise), FastAPI \
0.100+'s Pydantic-v2 model_dump), say so explicitly - do NOT silently \
use the newer API as if it were available in the pinned version.
5. GENERAL-PROGRAMMING REFUSAL (v0.9.1): if the user's question is about \
general programming concepts that don't belong to any specific library's \
documentation - variable declarations ("difference between let and \
const"), control flow, language syntax, generic algorithmic questions, \
or runtime errors that aren't tied to a pinned library - reply EXACTLY \
the canonical refusal phrase from rule 2. The user pinned a library \
because they want library-specific answers; route them elsewhere for \
language fundamentals rather than dressing up an unrelated chunk as an \
answer.

When citing, refer to source filenames as they appear in the retrieved \
context. Keep responses concise unless the user asks for depth."""


@dataclass(kw_only=True)
class AgentResponse:
    """What the agent loop produces per query."""

    text: str
    citations: list[Citation]
    tool_used: str
    iterations: int = 1  # v0.6: number of tool-call iterations actually run.
    refused: bool = False
    """Whether this response is a refusal, decided by the agent (authoritative)
    rather than by substring-sniffing the text downstream. True for the topic
    filter short-circuit, the iteration-cap bail-out, and any final answer that
    is the canonical refusal phrase. The eval harness reads this instead of its
    ``is_refusal`` heuristic so a legit answer that merely says "X is not
    covered in 18.2, but ..." isn't miscounted as a refusal."""


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
        self_critique: bool = False,
        topic_filter: bool = True,
        score_floor: float | None = None,
        floors_by_library: dict[str, float] | None = None,
    ) -> None:
        self._openai = openai
        self._memory = memory
        self._workspace_path = workspace_path
        self._chat_model = chat_model
        self._default_library = default_library
        self._default_version = default_version
        self._max_iterations = max(1, max_iterations)
        # v0.8: optional self-critique pass.
        # Originally defaulted ON in v0.8 with the hypothesis that re-reading
        # the draft against tool outputs would push precision up. The v0.8
        # eval ablation flipped that hypothesis: critique=on dropped
        # accuracy 0.882 -> 0.667 and version 0.944 -> 0.895 (the critique
        # rewrites well-grounded drafts into worse ones). Defaulting OFF;
        # the feature stays as opt-in so future prompt tuning or different
        # ablations can revisit it. See ADR-011 and CHANGELOG v0.8.0.
        self._self_critique = self_critique
        # v1.0: pre-retrieval topic classifier. One cheap LLM call before
        # the ReAct loop decides "library-specific" vs "general programming".
        # General-programming questions short-circuit to the canonical
        # refusal phrase WITHOUT retrieval (closes the v0.9 let/const oos
        # leak that HARD RULE #5 couldn't catch on its own). Cost: ~1
        # extra gpt-4o-mini call (~$0.00005, ~500ms). Defaults ON for
        # v1.0; flip OFF for ablation runs via the kwarg or
        # ``--no-topic-filter`` on the eval CLI.
        self._topic_filter = topic_filter

        # Retrieval score floors are decision thresholds. SearchDocsTool ships
        # corpus-tuned defaults; the eval harness can override them here so a
        # calibration run and an untuned-baseline run don't require editing the
        # production tool. Only forward the args the caller actually set.
        search_docs_kwargs: dict[str, Any] = {"qdrant": qdrant, "openai": openai}
        if score_floor is not None:
            search_docs_kwargs["score_floor"] = score_floor
        if floors_by_library is not None:
            search_docs_kwargs["floors_by_library"] = floors_by_library
        self._search_docs = SearchDocsTool(**search_docs_kwargs)
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

    def reset_memory(self) -> None:
        """Clear this agent's workspace memory.

        The eval harness calls this between corpus entries (when running in
        isolated mode) so each labelled probe is answered cold, without prior
        entries' Q/As leaking into the prompt. Production never calls it — a
        real workspace session wants memory to accumulate across turns.
        """
        self._memory.clear()

    async def answer(
        self,
        query: str,
        *,
        pinned_libraries: dict[str, str] | None = None,
    ) -> AgentResponse:
        """Run one query through the multi-iteration ReAct loop.

        v0.9.1: ``pinned_libraries`` is a dict mapping library names to the
        exact version the user's lockfile pins (e.g. ``{"react": "18.2.0",
        "fastapi": "0.95.0"}``). When provided, the system prompt gets a
        "Project lockfile pins" section AND ``_dispatch`` rewrites any
        tool-call ``version`` arg to match the pin for that library. This
        closes the bug where the LLM was inferring versions from question
        text and hitting collections that don't exist (fastapi_0_95_2 etc.).

        v1.0: pre-retrieval topic classifier short-circuits to the
        canonical refusal phrase when the query is general programming
        (closes the v0.9 let/const oos leak that prompt-only couldn't).
        """
        # 0. v1.0 topic filter: skip retrieval entirely for general-programming.
        if self._topic_filter and not await self._is_library_topic(
            query, pinned_libraries=pinned_libraries
        ):
            logger.info("topic filter: refusing general-programming query %r", query)
            return self._finalize(
                query=query,
                answer_text=_CANONICAL_REFUSAL,
                citations=[],
                tool_used="(topic_filter)",
                iterations=0,
                refused=True,
            )

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
            query=query, past_turns=past_turns, pinned_libraries=pinned_libraries
        )

        # 4. ReAct loop. On each iteration the model either:
        #    - emits tool_calls (we dispatch, append tool messages, loop)
        #    - emits content text (we return)
        citations: list[Citation] = []
        last_tool_used = "(none)"
        iterations_run = 0
        # v0.8: collect tool outputs so the self-critique pass can re-read
        # the same retrieved context the draft was generated from.
        tool_outputs: list[str] = []

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
                # Final answer. Optionally critique-revise before finalize.
                answer_text = msg.content or ""
                if self._self_critique and tool_outputs:
                    answer_text = await self._critique(
                        query=query, draft=answer_text, context_blocks=tool_outputs
                    )
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
                result = await self._dispatch(
                    tool_name=tool_name,
                    args=args,
                    query=query,
                    pinned_libraries=pinned_libraries,
                )
                citations.extend(result.citations)
                tool_outputs.append(result.text)
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
            refused=True,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _is_library_topic(
        self,
        query: str,
        pinned_libraries: dict[str, str] | None = None,
    ) -> bool:
        """v1.0 pre-retrieval topic classifier.

        Returns True if the query looks library-specific (proceed with
        normal ReAct), False if it looks like general programming /
        language fundamentals (short-circuit to refusal). One cheap
        ``gpt-4o-mini`` call at temperature 0. Defaults to True on any
        parse failure so we don't accidentally refuse real library
        questions just because the classifier glitched.

        v1.0 first-run learning: without project context the classifier
        called legitimate React questions (e.g. "How do I avoid
        recomputing... on every render?") GENERAL because they didn't
        mention "React" explicitly. v1.0 now passes ``pinned_libraries``
        into the prompt so the classifier leans LIBRARY for any question
        that's plausibly about one of the user's pinned libs.
        """
        pins_clause = ""
        if pinned_libraries:
            pin_summary = ", ".join(f"{lib}@{ver}" for lib, ver in sorted(pinned_libraries.items()))
            pins_clause = (
                f"\n\nThe user's project pins these libraries: {pin_summary}. "
                "If the question is plausibly about how to do something in "
                "one of those libraries (even if it doesn't name the library "
                "explicitly - words like 'render', 'component', 'route', "
                "'dependency', 'hook', 'reactive' usually indicate so), "
                "classify as LIBRARY."
            )
        prompt = (
            "You are a one-shot classifier inside a code-editor extension. "
            "The user is asking a question; their project has indexed "
            "documentation for specific software libraries (e.g. React, "
            "FastAPI, Vue). Decide which kind of question it is:\n\n"
            "- LIBRARY: about a specific library / framework / API. "
            "Examples: 'how do I use useState in React', 'what does "
            "Depends() do in FastAPI', 'computed vs watch in Vue', "
            "'avoid re-render when prop changes'.\n"
            "- GENERAL: about language fundamentals (let/const, hoisting), "
            "operating-system errors (EADDRINUSE), or generic CS concepts "
            "(closures, big-O) that aren't tied to any library."
            f"{pins_clause}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            "Reply with EXACTLY one word: LIBRARY or GENERAL."
        )
        try:
            response = await self._openai.chat.completions.create(
                model=self._chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
        except Exception as exc:
            logger.warning("topic classifier call failed (defaulting LIBRARY): %s", exc)
            return True
        reply = (response.choices[0].message.content or "").strip().upper()
        # Accept "LIBRARY", "LIBRARY.", "LIBRARY!" etc; treat anything that
        # starts with "GENERAL" as the refuse signal. Other shapes (empty,
        # whitespace, unexpected text) default to LIBRARY - safer to attempt
        # retrieval than to falsely refuse a real library question.
        return not reply.startswith("GENERAL")

    async def _critique(self, *, query: str, draft: str, context_blocks: list[str]) -> str:
        """v0.8 self-critique pass: re-read the draft against the retrieved
        context. If the draft introduced APIs / behaviour that don't appear
        in the context, return a revised answer; otherwise return the
        original draft unchanged.

        Costs one extra chat completion at temperature=0 (~$0.0001 on
        gpt-4o-mini, ~2s latency). Only runs on the non-streaming
        ``answer()`` path; the streaming path skips this so the streamed
        text doesn't get rewritten mid-flight.
        """
        joined_context = "\n\n---\n\n".join(context_blocks)[:12000]
        critique_prompt = (
            "You drafted this answer to the user's question:\n\n"
            "USER QUESTION:\n"
            f"{query}\n\n"
            "YOUR DRAFT:\n"
            f"---\n{draft}\n---\n\n"
            "Here is the retrieved documentation context the draft was supposed to "
            "be grounded in:\n\n"
            f"---\n{joined_context}\n---\n\n"
            "Re-read your draft against the context. Check: does your draft mention "
            "any API name, signature, or behaviour that is NOT present in the "
            "retrieved context? If yes, that's a leak across versions or a "
            "hallucination - revise the draft to stay strictly inside the "
            "retrieved context.\n\n"
            "Reply with exactly one of:\n"
            "- 'OK' (on a line by itself) if the draft is faithful to the context.\n"
            "- The revised full answer text if you need to fix anything. Do not "
            "  apologize; do not preface with 'Here is the revised answer'; just "
            "  output the corrected answer directly."
        )
        try:
            response = await self._openai.chat.completions.create(
                model=self._chat_model,
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("self-critique call failed: %s", exc)
            return draft
        critique_text = (response.choices[0].message.content or "").strip()
        if not critique_text:
            return draft
        # Accept "OK" / "ok" / "OK." / "OK - looks good" etc. as "no change".
        if critique_text.upper().startswith("OK"):
            return draft
        return critique_text

    def _finalize(
        self,
        *,
        query: str,
        answer_text: str,
        citations: list[Citation],
        tool_used: str,
        iterations: int,
        refused: bool | None = None,
    ) -> AgentResponse:
        """Append inline citations + record the Q/A in Mneme + return.

        ``refused`` is the authoritative refusal flag. Pass it explicitly for
        the non-answer paths (topic filter, iteration cap); leave it ``None``
        for a normal final answer and it's derived from whether the model
        emitted the canonical refusal phrase — computed on the raw text before
        the "Sources:" block is appended.
        """
        refused_flag = (
            refused if refused is not None else _CANONICAL_REFUSAL.lower() in answer_text.lower()
        )
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
            refused=refused_flag,
        )

    async def answer_stream(
        self,
        query: str,
        *,
        pinned_libraries: dict[str, str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one query and yield streaming protocol events.

        v0.7: same multi-iteration ReAct loop as ``answer()`` but uses
        OpenAI's ``stream=True`` so the final text response arrives as
        ``AssistantTextDelta`` chunks. Tool-call iterations don't emit
        deltas (the model returns tool_calls only when ``tool_choice``
        is required/auto and it decides to dispatch). The terminal frame
        is ``AssistantStreamFinal`` with citations + tool_used + iterations.

        v0.9.1: ``pinned_libraries`` parity with ``answer()``. The live
        production sidecar (__main__._run_agent) reads lockfile pins via
        lockfiles.parse_package_json and passes them through so the
        agent calls the right collection per query.

        ``answer()`` is preserved unchanged for the eval harness; both
        share the same dispatch + memory paths.
        """
        # v1.0 topic filter: short-circuit to refusal for general-programming.
        if self._topic_filter and not await self._is_library_topic(
            query, pinned_libraries=pinned_libraries
        ):
            logger.info("topic filter: refusing general-programming query %r", query)
            yield AssistantTextDelta(text=_CANONICAL_REFUSAL, chunk_index=0)
            yield AssistantStreamFinal(
                citations=[],
                tool_used="(topic_filter)",
                iterations=0,
            )
            try:
                self._memory.record_qa(query=query, answer=_CANONICAL_REFUSAL, citations=[])
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to record topic-filter refusal in Mneme: %s", exc)
            return
        candidate_names = [s.id for s in self._picker.select(query, k=10)]
        if not candidate_names:
            candidate_names = list(self._tools_by_name.keys())
        tool_specs = self._openai_tool_specs(candidate_names)
        past_turns = self._memory.retrieve_relevant(query, k=3)
        messages: list[ChatCompletionMessageParam] = self._initial_messages(
            query=query, past_turns=past_turns, pinned_libraries=pinned_libraries
        )

        citations: list[Citation] = []
        last_tool_used = "(none)"
        chunk_index = 0

        for iteration in range(self._max_iterations):
            iterations_run = iteration + 1
            # Stream the model response. Tool-call iterations come back
            # with tool_calls and no content. Final-answer iterations come
            # back with content text and no tool_calls.
            stream = await self._openai.chat.completions.create(
                model=self._chat_model,
                messages=messages,
                tools=tool_specs,
                tool_choice="required" if iteration == 0 else "auto",
                temperature=0.2,
                stream=True,
            )

            text_buffer: list[str] = []
            # OpenAI streams tool_calls as a sequence of partial chunks
            # indexed by position. Accumulate into a dict keyed by ``index``.
            tool_call_buf: dict[int, dict[str, str]] = {}

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    content_str = delta.content or ""
                    text_buffer.append(content_str)
                    yield AssistantTextDelta(text=content_str, chunk_index=chunk_index)
                    chunk_index += 1
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls or []:
                        idx = getattr(tc, "index", 0)
                        slot = tool_call_buf.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""}
                        )
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id or slot["id"]
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["name"] += fn.name or ""
                            if getattr(fn, "arguments", None):
                                slot["arguments"] += fn.arguments or ""

            if tool_call_buf:
                # Tool-call iteration. Rebuild the assistant turn + run
                # each tool, append tool messages, and continue the loop.
                # Rename to ``call`` so mypy doesn't conflate this dict[str, str]
                # with the ``tc: ChoiceDeltaToolCall`` from the streaming
                # accumulator loop above.
                ordered_calls: list[dict[str, str]] = [
                    call for _, call in sorted(tool_call_buf.items(), key=lambda kv: kv[0])
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(text_buffer) or "",
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": call["arguments"] or "{}",
                                },
                            }
                            for call in ordered_calls
                        ],
                    }
                )
                for call in ordered_calls:
                    try:
                        args = json.loads(call["arguments"] or "{}")
                    except json.JSONDecodeError:
                        logger.warning("agent_stream: malformed tool args: %r", call["arguments"])
                        args = {}
                    result = await self._dispatch(
                        tool_name=call["name"],
                        args=args,
                        query=query,
                        pinned_libraries=pinned_libraries,
                    )
                    citations.extend(result.citations)
                    last_tool_used = call["name"]
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result.text,
                        }
                    )
                continue

            # No tool_calls -> the streamed text was the final answer.
            answer_text = "".join(text_buffer)
            deduped = _dedupe_citations(citations)
            if deduped:
                citation_block = "\n\nSources: " + " ".join(c.render() for c in deduped)
                yield AssistantTextDelta(text=citation_block, chunk_index=chunk_index)
                chunk_index += 1
                answer_text = f"{answer_text}{citation_block}"
            try:
                self._memory.record_qa(
                    query=query,
                    answer=answer_text,
                    citations=[c.render() for c in deduped],
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to record Q/A in Mneme: %s", exc)
            yield AssistantStreamFinal(
                citations=[
                    CitationRef(
                        library=c.library,
                        version=c.version,
                        source=c.source,
                        source_url=c.source_url,
                    )
                    for c in deduped
                ],
                tool_used=last_tool_used,
                iterations=iterations_run,
            )
            return

        # Iteration cap hit.
        logger.warning(
            "agent_stream hit max_iterations=%d for query %r", self._max_iterations, query
        )
        cap_text = (
            "I reached my iteration limit looking for an answer. "
            "Try rephrasing the question or indexing the relevant library."
        )
        yield AssistantTextDelta(text=cap_text, chunk_index=chunk_index)
        deduped = _dedupe_citations(citations)
        yield AssistantStreamFinal(
            citations=[
                CitationRef(
                    library=c.library,
                    version=c.version,
                    source=c.source,
                    source_url=c.source_url,
                )
                for c in deduped
            ],
            tool_used=last_tool_used,
            iterations=self._max_iterations,
        )

    async def _dispatch(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        query: str,
        pinned_libraries: dict[str, str] | None = None,
    ) -> ToolResult:
        """Run the named tool with the model's args, defaulting where missing.

        v0.6 lets the LLM pass ``library``/``version``/``query`` via
        function-call args. If those are missing we fall back to the
        Agent's defaults (so the tool always gets workable inputs).
        v0.9 adds optional ``api_name`` for SearchDocsTool retrieval
        filtering.
        v0.9.1: when ``pinned_libraries`` is set and the model picked a
        library that's in the pin map, OVERRIDE the model's version with
        the pinned one. The LLM is unreliable at picking exact patch
        versions from question text; the lockfile is the source of truth.
        """
        tool = self._tools_by_name.get(tool_name)
        if tool is None:
            return ToolResult(text=f"[unknown tool: {tool_name}]")
        library = str(args.get("library") or self._default_library)
        version = str(args.get("version") or self._default_version)
        # v0.9.1: pinned version overrides whatever the LLM guessed.
        if pinned_libraries:
            pinned_ver = pinned_libraries.get(library.lower())
            if pinned_ver:
                version = pinned_ver
        tool_query = str(args.get("query") or query)
        if tool_name == self._search_docs.name:
            api_name_arg = args.get("api_name")
            api_name = str(api_name_arg) if api_name_arg else None
            return await self._search_docs.run(
                library=library, version=version, query=tool_query, api_name=api_name
            )
        if tool_name == self._search_workspace.name:
            return await self._search_workspace.run(query=tool_query)
        if tool_name == self._find_changelog.name:
            return await self._find_changelog.run(
                library=library, version=version, query=tool_query
            )
        return ToolResult(text=f"[unknown tool: {tool_name}]")

    def _initial_messages(
        self,
        *,
        query: str,
        past_turns: list[str],
        pinned_libraries: dict[str, str] | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """Build the starting message list: system + (past Q/A) + user.

        v0.9.1: when ``pinned_libraries`` is set, surface the lockfile
        pins to the LLM so it picks the right (library, version) per
        tool call instead of guessing from question text.
        """
        system_content = _SYSTEM_PROMPT
        if pinned_libraries:
            pins_text = ", ".join(f"{lib}@{ver}" for lib, ver in sorted(pinned_libraries.items()))
            system_content += (
                f"\n\n## Project lockfile pins\n\n"
                f"The user's project pins: {pins_text}. "
                f"When calling search_docs or find_in_changelog, use these "
                f"exact (library, version) pairs - do not guess versions "
                f"from question text."
            )
        if past_turns:
            past_text = "\n\n".join(f"- {t}" for t in past_turns)
            system_content = f"{system_content}\n\n## Past Q/A in this workspace\n\n{past_text}"
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
