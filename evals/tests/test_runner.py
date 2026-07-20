"""End-to-end runner tests with a fake AgentLike + StaticJudge."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from evals.judge import StaticJudge
from evals.runner import run_corpus, run_one
from evals.schema import CorpusEntry


@dataclass
class _FakeCitation:
    library: str
    version: str
    source: str

    def render(self) -> str:
        return f"[{self.library}@{self.version}:{self.source}]"


@dataclass
class _FakeAgentResponse:
    text: str
    tool_used: str = "search_docs"
    citations: list[_FakeCitation] = field(default_factory=list)
    # None -> no structured flag (runner falls back to the is_refusal heuristic,
    # mirroring a duck-typed agent); a bool -> the authoritative flag.
    refused: bool | None = None


class _FakeAgent:
    """Configurable agent that returns canned answers per question.

    Exposes ``reset_memory`` and counts calls so the runner's memory-isolation
    behaviour can be asserted without a real Mneme backend.
    """

    def __init__(self, answers: dict[str, _FakeAgentResponse]) -> None:
        self._answers = answers
        self.reset_calls = 0

    async def answer(
        self,
        query: str,
        *,
        pinned_libraries: dict[str, str] | None = None,
    ) -> _FakeAgentResponse:
        # v0.9.1 added the ``pinned_libraries`` kwarg to AgentLike; the
        # fake accepts and ignores it so runner-side tests don't have to
        # know about retrieval-routing internals.
        del pinned_libraries
        return self._answers.get(query, _FakeAgentResponse(text="(no answer)"))

    def reset_memory(self) -> None:
        self.reset_calls += 1


@pytest.fixture
def react_entry() -> CorpusEntry:
    return CorpusEntry(
        id="react_useState_basic",
        library="react",
        version="18.2.0",
        question="How do I add state in React?",
        expected_answer="Call useState() at the top of the component.",
        expected_apis=["useState"],
        forbidden_apis=["use(promise)"],
    )


@pytest.fixture
def oos_entry() -> CorpusEntry:
    return CorpusEntry(
        id="oos_python",
        library="react",
        version="18.2.0",
        question="How do I decorate a Flask route?",
        expected_answer="Out of scope.",
        out_of_scope=True,
    )


# ---------------------------------------------------------------------------
# run_one
# ---------------------------------------------------------------------------


async def test_run_one_records_judge_verdict_for_in_scope(react_entry: CorpusEntry) -> None:
    agent = _FakeAgent(
        {
            react_entry.question: _FakeAgentResponse(
                text="Use useState() to add state.",
                citations=[_FakeCitation("react", "18.2.0", "useState.md")],
            )
        }
    )
    judge = StaticJudge(verdict=True)
    result = await run_one(agent=agent, entry=react_entry, judge=judge)

    assert result.entry_id == react_entry.id
    assert "useState" in result.answer
    assert result.tool_used == "search_docs"
    assert result.citations == ["[react@18.2.0:useState.md]"]
    assert result.judge is not None
    assert result.judge.correct is True
    assert result.version_correct is True
    assert result.refused is False


async def test_run_one_marks_version_incorrect_when_forbidden_appears(
    react_entry: CorpusEntry,
) -> None:
    agent = _FakeAgent(
        {react_entry.question: _FakeAgentResponse(text="Use use(promise) to do it.")}
    )
    result = await run_one(agent=agent, entry=react_entry, judge=StaticJudge(verdict=True))
    assert result.version_correct is False  # forbidden_apis matched


async def test_run_one_skips_judge_for_out_of_scope(oos_entry: CorpusEntry) -> None:
    agent = _FakeAgent({oos_entry.question: _FakeAgentResponse(text="I don't have docs for that.")})
    result = await run_one(agent=agent, entry=oos_entry, judge=StaticJudge(verdict=True))

    assert result.judge is None  # judge skipped for out-of-scope entries
    assert result.refused is True  # heuristic catches the refusal phrase


async def test_run_one_handles_no_judge(react_entry: CorpusEntry) -> None:
    agent = _FakeAgent({react_entry.question: _FakeAgentResponse(text="Use useState().")})
    result = await run_one(agent=agent, entry=react_entry, judge=None)
    assert result.judge is None
    assert result.version_correct is True


async def test_run_one_prefers_structured_refused_flag(react_entry: CorpusEntry) -> None:
    # Text that trips the substring heuristic ("not covered") but is a real,
    # partial answer -> the agent's structured refused=False must win.
    agent = _FakeAgent(
        {
            react_entry.question: _FakeAgentResponse(
                text="useState covers local state; effects are not covered here.",
                refused=False,
            )
        }
    )
    result = await run_one(agent=agent, entry=react_entry, judge=None)
    assert result.refused is False  # structured flag beats the heuristic


async def test_run_one_falls_back_to_heuristic_without_flag(react_entry: CorpusEntry) -> None:
    # No structured flag (refused=None) -> runner uses is_refusal on the text.
    agent = _FakeAgent(
        {react_entry.question: _FakeAgentResponse(text="I don't have docs for that.")}
    )
    result = await run_one(agent=agent, entry=react_entry, judge=None)
    assert result.refused is True


# ---------------------------------------------------------------------------
# run_corpus
# ---------------------------------------------------------------------------


async def test_run_corpus_returns_one_result_per_entry(
    react_entry: CorpusEntry, oos_entry: CorpusEntry
) -> None:
    agent = _FakeAgent(
        {
            react_entry.question: _FakeAgentResponse(text="Use useState()."),
            oos_entry.question: _FakeAgentResponse(text="I don't have docs for that."),
        }
    )
    results = await run_corpus(
        agent=agent, entries=[react_entry, oos_entry], judge=StaticJudge(verdict=True)
    )
    assert len(results) == 2
    assert results[0].entry_id == react_entry.id
    assert results[1].entry_id == oos_entry.id


async def test_run_corpus_isolates_memory_by_default(
    react_entry: CorpusEntry, oos_entry: CorpusEntry
) -> None:
    # Default (isolate_entries=True) resets memory once per entry so each
    # labelled probe is answered cold.
    agent = _FakeAgent(
        {
            react_entry.question: _FakeAgentResponse(text="Use useState()."),
            oos_entry.question: _FakeAgentResponse(text="I don't have docs for that."),
        }
    )
    await run_corpus(agent=agent, entries=[react_entry, oos_entry], judge=None)
    assert agent.reset_calls == 2


async def test_run_corpus_warm_memory_skips_reset(
    react_entry: CorpusEntry, oos_entry: CorpusEntry
) -> None:
    # Opt-in warm mode leaves memory intact across entries (cross-turn test).
    agent = _FakeAgent(
        {
            react_entry.question: _FakeAgentResponse(text="Use useState()."),
            oos_entry.question: _FakeAgentResponse(text="I don't have docs for that."),
        }
    )
    await run_corpus(
        agent=agent, entries=[react_entry, oos_entry], judge=None, isolate_entries=False
    )
    assert agent.reset_calls == 0


async def test_run_corpus_captures_runner_errors(react_entry: CorpusEntry) -> None:
    class _BrokenAgent:
        async def answer(
            self,
            query: str,
            *,
            pinned_libraries: dict[str, str] | None = None,
        ) -> _FakeAgentResponse:
            del query, pinned_libraries
            raise RuntimeError("agent exploded")

    results = await run_corpus(agent=_BrokenAgent(), entries=[react_entry], judge=None)
    assert len(results) == 1
    assert "runner error" in results[0].answer
    assert results[0].tool_used == "(error)"
    assert results[0].version_correct is False
