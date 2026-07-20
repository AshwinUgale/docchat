"""Replay a corpus through the DocChat agent and produce per-entry results.

The runner sidesteps the WebSocket layer entirely - it constructs an
``Agent`` directly and calls ``answer()`` per entry. That's the eval
contract: prove the agent path (router + tools + Mneme + LLM) holds up
against a labelled corpus, independent of the IPC plumbing.

Latency is measured wall-clock end-to-end per call via ``time.perf_counter``.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from evals.judge import Judge
from evals.metrics import is_refusal, is_version_correct
from evals.schema import CorpusEntry, JudgeVerdict, RunResult

__all__ = ["AgentLike", "run_corpus", "run_one"]

logger = logging.getLogger(__name__)


class AgentLike(Protocol):
    """Anything that exposes ``async answer(query, *, pinned_libraries=...)``.

    Tests substitute a fake; production passes a real ``Agent``.
    The ``pinned_libraries`` kwarg (v0.9.1) lets the runner inject the
    corpus entry's library@version into the agent's system prompt so the
    LLM picks the right collection instead of guessing.
    """

    async def answer(
        self, query: str, *, pinned_libraries: dict[str, str] | None = None
    ) -> object: ...


async def run_one(
    *,
    agent: AgentLike,
    entry: CorpusEntry,
    judge: Judge | None,
) -> RunResult:
    """Run a single corpus entry and return its result.

    The agent's response object is duck-typed: we read ``.text``,
    ``.tool_used``, and ``.citations`` (each citation has ``.render()``).
    This keeps the runner decoupled from the concrete Agent class.
    """
    # v0.9.1: pass the corpus entry's library/version as the lockfile pin
    # so the agent calls search_docs with the right (library, version)
    # rather than guessing from question text. Closes the bug where
    # FastAPI queries hit fastapi_0_95_2 and Vue queries hit vue_3_0_0
    # (neither indexed).
    pinned = {entry.library.lower(): entry.version}
    start = time.perf_counter()
    response = await agent.answer(entry.question, pinned_libraries=pinned)
    latency_ms = (time.perf_counter() - start) * 1000.0

    answer_text = str(getattr(response, "text", ""))
    tool_used = str(getattr(response, "tool_used", ""))
    citations_obj = getattr(response, "citations", []) or []
    citation_strings = [c.render() if hasattr(c, "render") else str(c) for c in citations_obj]

    version_correct = is_version_correct(
        answer=answer_text,
        expected_apis=entry.expected_apis,
        forbidden_apis=entry.forbidden_apis,
    )
    # Prefer the agent's authoritative ``refused`` flag when it exposes one
    # (the real Agent does); fall back to the substring heuristic for
    # duck-typed agents/fakes that don't. The structured flag avoids
    # miscounting a legit answer that merely says "not covered in 18.2, but
    # ..." as a refusal.
    refused_attr = getattr(response, "refused", None)
    refused = refused_attr if isinstance(refused_attr, bool) else is_refusal(answer_text)

    verdict: JudgeVerdict | None = None
    if judge is not None and not entry.out_of_scope:
        # Out-of-scope entries are graded by ``refusal_rate``, not the judge.
        verdict = await judge.grade(
            question=entry.question,
            expected_answer=entry.expected_answer,
            actual_answer=answer_text,
        )

    return RunResult(
        entry_id=entry.id,
        question=entry.question,
        answer=answer_text,
        tool_used=tool_used,
        citations=citation_strings,
        judge=verdict,
        version_correct=version_correct,
        refused=refused,
        out_of_scope=entry.out_of_scope,
        latency_ms=latency_ms,
    )


async def run_corpus(
    *,
    agent: AgentLike,
    entries: list[CorpusEntry],
    judge: Judge | None,
    isolate_entries: bool = True,
) -> list[RunResult]:
    """Sequentially replay every entry through the agent.

    Sequential rather than parallel so we get stable, deterministic JSON
    output.

    ``isolate_entries`` (default ``True``) resets the agent's memory before
    each entry so every labelled probe is answered COLD. Each corpus entry is
    an independent, labelled question; the single persistent agent records
    each Q/A into Mneme and surfaces prior Q/As in the next prompt, so without
    a reset the metrics become order-dependent and an earlier entry's answer
    (e.g. one containing ``useState``) can leak into a later, similar entry —
    inflating accuracy/version-correctness in a way a genuinely cold query
    would not. Pass ``isolate_entries=False`` only to deliberately exercise
    cross-turn memory (a different, multi-turn kind of test).

    The reset is duck-typed: agents exposing ``reset_memory()`` get called;
    memoryless fakes are left alone.
    """
    results: list[RunResult] = []
    reset = getattr(agent, "reset_memory", None)
    for i, entry in enumerate(entries, start=1):
        logger.info("eval %d/%d: %s", i, len(entries), entry.id)
        if isolate_entries and callable(reset):
            reset()
        try:
            result = await run_one(agent=agent, entry=entry, judge=judge)
        except Exception as exc:
            logger.exception("entry %s crashed: %s", entry.id, exc)
            result = RunResult(
                entry_id=entry.id,
                question=entry.question,
                answer=f"[runner error] {exc}",
                tool_used="(error)",
                citations=[],
                judge=None,
                version_correct=False,
                refused=False,
                out_of_scope=entry.out_of_scope,
                latency_ms=0.0,
            )
        results.append(result)
    return results
