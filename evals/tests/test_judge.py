"""Tests for the LLM-as-judge wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

from evals.judge import LLMJudge, StaticJudge


def _fake_openai_returning(content: str) -> MagicMock:
    """OpenAI client whose chat.completions.create returns canned content."""
    client = MagicMock()

    async def chat(**kwargs: object) -> object:
        del kwargs
        choice = MagicMock()
        choice.message = MagicMock(content=content)
        response = MagicMock()
        response.choices = [choice]
        return response

    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = chat
    return client


async def test_static_judge_returns_configured_verdict() -> None:
    judge = StaticJudge(verdict=True, reasoning="ok")
    result = await judge.grade(question="q", expected_answer="e", actual_answer="a")
    assert result.correct is True
    assert result.reasoning == "ok"


async def test_static_judge_can_return_false() -> None:
    judge = StaticJudge(verdict=False, reasoning="mismatch")
    result = await judge.grade(question="q", expected_answer="e", actual_answer="a")
    assert result.correct is False


async def test_llm_judge_parses_clean_json() -> None:
    openai = _fake_openai_returning('{"correct": true, "reasoning": "addresses useState"}')
    judge = LLMJudge(openai=openai)
    result = await judge.grade(
        question="how do I add state?",
        expected_answer="Call useState",
        actual_answer="Use useState() at the top of the component.",
    )
    assert result.correct is True
    assert "useState" in result.reasoning


async def test_llm_judge_parses_false_verdict() -> None:
    openai = _fake_openai_returning('{"correct": false, "reasoning": "named wrong API"}')
    judge = LLMJudge(openai=openai)
    result = await judge.grade(question="q", expected_answer="e", actual_answer="a")
    assert result.correct is False
    assert "wrong" in result.reasoning


async def test_llm_judge_handles_unparseable_reply() -> None:
    openai = _fake_openai_returning("not json at all")
    judge = LLMJudge(openai=openai)
    result = await judge.grade(question="q", expected_answer="e", actual_answer="a")
    assert result.correct is False
    assert "unparseable" in result.reasoning


async def test_llm_judge_handles_openai_exception() -> None:
    openai = MagicMock()

    async def boom(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("rate limited")

    openai.chat = MagicMock()
    openai.chat.completions = MagicMock()
    openai.chat.completions.create = boom

    judge = LLMJudge(openai=openai)
    result = await judge.grade(question="q", expected_answer="e", actual_answer="a")
    assert result.correct is False
    assert "judge error" in result.reasoning
