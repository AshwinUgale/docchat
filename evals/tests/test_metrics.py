"""Tests for the pure metric functions."""

from __future__ import annotations

from evals.metrics import compute_metrics, is_refusal, is_version_correct
from evals.schema import JudgeVerdict, RunResult

# ---------------------------------------------------------------------------
# is_version_correct
# ---------------------------------------------------------------------------


def test_version_correct_when_required_present_and_forbidden_absent() -> None:
    assert is_version_correct(
        answer="Call useState() to manage state.",
        expected_apis=["useState"],
        forbidden_apis=["use(promise)"],
    )


def test_version_incorrect_when_required_missing() -> None:
    assert not is_version_correct(
        answer="State is complicated.",
        expected_apis=["useState"],
        forbidden_apis=[],
    )


def test_version_incorrect_when_forbidden_present() -> None:
    assert not is_version_correct(
        answer="Use `use(fetch('/data'))` to load data in render.",
        expected_apis=[],
        forbidden_apis=["use(fetch"],
    )


def test_version_check_is_case_insensitive() -> None:
    assert is_version_correct(
        answer="USESTATE works for state management.",
        expected_apis=["useState"],
        forbidden_apis=[],
    )


def test_version_check_vacuous_when_both_empty() -> None:
    assert is_version_correct(answer="anything", expected_apis=[], forbidden_apis=[])


# ---------------------------------------------------------------------------
# is_refusal
# ---------------------------------------------------------------------------


def test_refusal_detected_for_no_relevant() -> None:
    assert is_refusal("No relevant chunks were found for that query.")


def test_refusal_detected_for_dont_have() -> None:
    assert is_refusal("I don't have documentation for that library.")


def test_non_refusal_normal_answer() -> None:
    assert not is_refusal(
        "Call useState() at the top of your function component to declare state."
    )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def _r(
    *,
    entry_id: str,
    correct: bool | None,
    refused: bool,
    version_correct: bool,
    latency_ms: float = 1000.0,
) -> RunResult:
    return RunResult(
        entry_id=entry_id,
        question="q",
        answer="a",
        tool_used="search_docs",
        citations=[],
        judge=JudgeVerdict(correct=correct, reasoning="x") if correct is not None else None,
        version_correct=version_correct,
        refused=refused,
        latency_ms=latency_ms,
    )


def test_compute_metrics_in_scope_accuracy() -> None:
    results = [
        _r(entry_id="a", correct=True, refused=False, version_correct=True),
        _r(entry_id="b", correct=True, refused=False, version_correct=True),
        _r(entry_id="c", correct=False, refused=False, version_correct=False),
        _r(entry_id="d", correct=True, refused=False, version_correct=True),
    ]
    m = compute_metrics(results)
    assert m.n_entries == 4
    assert m.n_in_scope == 4
    assert m.n_out_of_scope == 0
    assert m.answer_accuracy == 0.75
    assert m.version_correctness == 0.75


def test_compute_metrics_refusal_rate_over_out_of_scope() -> None:
    # The convention: judge=None + refused=True => treated as out-of-scope target.
    results = [
        _r(entry_id="oos_1", correct=None, refused=True, version_correct=False),
        _r(entry_id="oos_2", correct=None, refused=True, version_correct=False),
        _r(entry_id="oos_3", correct=None, refused=False, version_correct=False),
        _r(entry_id="in_1", correct=True, refused=False, version_correct=True),
    ]
    m = compute_metrics(results)
    assert m.n_out_of_scope == 2  # only the refused-and-judge-none ones
    assert m.refusal_rate == 1.0  # both detected as refused
    assert m.n_in_scope == 2
    assert m.answer_accuracy == 1.0


def test_compute_metrics_empty_corpus_returns_zeros() -> None:
    m = compute_metrics([])
    assert m.n_entries == 0
    assert m.answer_accuracy == 0.0
    assert m.version_correctness == 0.0
    assert m.refusal_rate == 0.0
    assert m.mean_latency_ms == 0.0


def test_compute_metrics_latency_p95() -> None:
    results = [
        _r(entry_id=str(i), correct=True, refused=False, version_correct=True, latency_ms=float(i * 100))
        for i in range(1, 21)
    ]
    m = compute_metrics(results)
    assert m.mean_latency_ms == 1050.0
    # 95th percentile of [100..2000] by nearest-rank = 2000
    assert m.p95_latency_ms >= 1900.0
