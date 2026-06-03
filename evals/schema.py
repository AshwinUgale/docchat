"""Schema for the DocChat eval corpus + runner output.

Pydantic models so the on-disk JSON has a typed contract and downstream
analysis (notebooks, README charts, regression diffs) reads it without
schema-divination.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "Corpus",
    "CorpusEntry",
    "JudgeVerdict",
    "RunMetrics",
    "RunResult",
    "RunSummary",
]


class CorpusEntry(BaseModel):
    """One labelled Q/A pair in the eval corpus.

    Designed for version-aware Q&A: each entry pins a library + version,
    carries the correct answer for that version, AND the answer that
    would be correct for a different version. Both are used: the correct
    one drives ``answer_accuracy``; the wrong one is what ``version_correctness``
    explicitly checks the agent did NOT regress to.
    """

    id: str
    """Stable identifier (e.g. ``react_18_useState_basic``)."""

    library: str
    """Library name as it would appear in package.json."""

    version: str
    """Pinned version we're testing against (e.g. ``18.2.0``)."""

    question: str
    """The natural-language question fed to the agent."""

    expected_answer: str
    """The labelled correct answer for the pinned version. Used as the
    reference for the LLM-judge."""

    wrong_for_other_version: str | None = None
    """What the answer would (wrongly) look like for a different version.
    Surfaces version-mixup failures in error logs; not used by the judge."""

    expected_apis: list[str] = Field(default_factory=list)
    """API names that SHOULD appear in the answer (case-insensitive substring).
    Empty -> skipped. Example: ``["useState"]``."""

    forbidden_apis: list[str] = Field(default_factory=list)
    """API names that MUST NOT appear (case-insensitive substring). Catches
    version regressions to APIs that only exist in a later version.
    Example for React 18.2: ``["use("]`` since ``use()`` is React 19+."""

    out_of_scope: bool = False
    """If true, the correct behaviour is to refuse rather than answer.
    Drives the ``refusal_rate`` metric. ``expected_answer`` for these can
    be a short "I don't have docs for..." sentinel."""

    notes: str | None = None
    """Free-form authoring notes. Not used by the runner."""


class Corpus(BaseModel):
    """The full eval corpus loaded from disk."""

    name: str
    description: str
    entries: list[CorpusEntry]


class JudgeVerdict(BaseModel):
    """LLM-as-judge output for one entry."""

    correct: bool
    """Did the answer correctly address the labelled expected answer?"""

    reasoning: str
    """Brief justification - useful for debugging false positives."""


class RunResult(BaseModel):
    """One Q/A run through the agent."""

    entry_id: str
    question: str
    answer: str
    """The text the agent produced."""

    tool_used: str
    """Which ToolPicker route fired."""

    citations: list[str]
    """Rendered citation tokens like ``[react@18.2.0:useState.md]``."""

    judge: JudgeVerdict | None = None
    """LLM-judge verdict on accuracy. ``None`` when ``--no-judge``."""

    version_correct: bool
    """All expected_apis present AND no forbidden_apis present."""

    refused: bool
    """Did the answer signal refusal (e.g. 'I don't have docs for...')?"""

    latency_ms: float


class RunMetrics(BaseModel):
    """Aggregated metrics across a run."""

    n_entries: int
    n_in_scope: int
    n_out_of_scope: int

    answer_accuracy: float
    """Fraction of in-scope entries the judge marked correct. ``None`` when
    the judge was skipped is represented as 0 here for JSON simplicity;
    inspect per-entry ``judge`` to disambiguate."""

    version_correctness: float
    """Fraction of in-scope entries where ``version_correct`` is True."""

    refusal_rate: float
    """Fraction of out-of-scope entries where ``refused`` is True. 1.0 is
    the goal - agent should refuse out-of-scope questions, not invent
    answers."""

    mean_latency_ms: float
    p95_latency_ms: float


class RunSummary(BaseModel):
    """Single-file output for one eval run."""

    corpus_name: str
    config: dict[str, str | int | bool]
    metrics: RunMetrics
    results: list[RunResult]
