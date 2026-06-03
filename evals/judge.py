"""LLM-as-judge for fuzzy answer accuracy.

The judge reads the labelled expected answer + the agent's actual answer
and returns ``correct: bool + reasoning: str``. Cheap model
(``gpt-4o-mini`` by default) because the task is short and the precision
target is "did the agent address the same point as the reference," not
"is this prose flawless."

ADR-007 captures: why LLM-as-judge over exact-match (semantic equivalence
matters for prose answers), why the prompt is template-constrained (so
the response is JSON-parseable), and why we don't ship a reference judge
of our own (OpenAI's chat completion is the cheapest option that gets the
quality we need at this scale).
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from openai import AsyncOpenAI

from evals.schema import JudgeVerdict

__all__ = ["Judge", "LLMJudge", "StaticJudge"]

logger = logging.getLogger(__name__)


class Judge(Protocol):
    """Protocol: anything that grades a (question, expected, actual) triple."""

    async def grade(
        self, *, question: str, expected_answer: str, actual_answer: str
    ) -> JudgeVerdict:
        """Return a verdict on whether ``actual_answer`` is correct."""
        ...


_JUDGE_PROMPT = """You are grading an AI assistant's answer to a question about \
a software library. The reference answer is what we labelled as correct for the \
library version in question. The candidate answer is what the assistant produced.

Reply with a single JSON object on one line: ``{{"correct": <true|false>, "reasoning": "<one short sentence>"}}``.

Mark ``correct: true`` if the candidate answer addresses the same technical point \
as the reference - same API, same shape, same behaviour. The wording does not have \
to match; semantic equivalence is the bar.

Mark ``correct: false`` if the candidate names a different API, gives wrong \
parameters, contradicts the reference on behaviour, or refuses to answer when the \
reference provides one.

Question: {question}

Reference answer:
{expected_answer}

Candidate answer:
{actual_answer}

Reply with only the JSON object, no preamble."""


class LLMJudge:
    """Judge backed by an OpenAI chat completion.

    Args:
        openai: An ``AsyncOpenAI`` client.
        model: Chat model name. ``gpt-4o-mini`` is plenty for this task and
            costs ~$0.0001 per judgement.
    """

    def __init__(self, *, openai: AsyncOpenAI, model: str = "gpt-4o-mini") -> None:
        self._openai = openai
        self._model = model

    async def grade(
        self, *, question: str, expected_answer: str, actual_answer: str
    ) -> JudgeVerdict:
        prompt = _JUDGE_PROMPT.format(
            question=question.strip(),
            expected_answer=expected_answer.strip(),
            actual_answer=actual_answer.strip(),
        )
        try:
            response = await self._openai.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning("judge OpenAI call failed: %s", exc)
            return JudgeVerdict(correct=False, reasoning=f"judge error: {exc}")
        content = response.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("judge returned non-JSON: %r", content)
            return JudgeVerdict(correct=False, reasoning=f"unparseable judge reply: {content[:120]}")
        return JudgeVerdict(
            correct=bool(data.get("correct", False)),
            reasoning=str(data.get("reasoning", ""))[:280],
        )


class StaticJudge:
    """Test double - returns a fixed verdict. Used in unit tests."""

    def __init__(self, *, verdict: bool = True, reasoning: str = "static") -> None:
        self._verdict = verdict
        self._reasoning = reasoning

    async def grade(
        self, *, question: str, expected_answer: str, actual_answer: str
    ) -> JudgeVerdict:
        del question, expected_answer, actual_answer
        return JudgeVerdict(correct=self._verdict, reasoning=self._reasoning)
