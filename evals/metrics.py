"""Pure functions over RunResult lists. JSON-serialisable outputs."""

from __future__ import annotations

import math

from evals.schema import RunMetrics, RunResult

__all__ = [
    "compute_metrics",
    "is_refusal",
    "is_version_correct",
]


# Phrases the agent emits when it can't ground an answer in retrieved docs.
# The system prompt explicitly instructs this behaviour, so the substring
# check is reliable for v0.4. v0.5+ can swap to a structured refusal flag
# on the agent response if the heuristic gets noisy.
_REFUSAL_PHRASES: tuple[str, ...] = (
    "i don't have",
    "i do not have",
    "does not contain",
    "not in the retrieved",
    "no relevant",
    "cannot answer",
    "can't answer",
    "not covered",
)


def is_refusal(answer: str) -> bool:
    """Heuristic: did the agent refuse to answer?

    Case-insensitive substring match against a curated phrase list.
    v0.4 is good enough for a 20-pair corpus; v0.5 can move to a
    structured ``refused`` flag on the protocol if false positives bite.
    """
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def is_version_correct(
    *, answer: str, expected_apis: list[str], forbidden_apis: list[str]
) -> bool:
    """Substring-based version-correctness check.

    Returns True iff:
      * every entry in ``expected_apis`` appears in the answer (case-insensitive), AND
      * no entry in ``forbidden_apis`` appears in the answer (case-insensitive).

    Both lists may be empty - empty expected => no positive constraint;
    empty forbidden => no negative constraint. Both empty => True (the
    check is vacuously satisfied; rely on the LLM-judge for those entries).
    """
    lowered = answer.lower()
    for required in expected_apis:
        if required.lower() not in lowered:
            return False
    return all(forbidden.lower() not in lowered for forbidden in forbidden_apis)


def _percentile_ms(latencies: list[float], pct: float) -> float:
    """Nearest-rank percentile. Same shape as Mneme/ToolPicker eval harnesses."""
    if not latencies:
        return 0.0
    if pct <= 0:
        return latencies[0]
    if pct >= 100:
        return latencies[-1]
    sorted_lat = sorted(latencies)
    rank = math.ceil(pct / 100 * len(sorted_lat))
    return sorted_lat[max(0, rank - 1)]


def compute_metrics(results: list[RunResult]) -> RunMetrics:
    """Aggregate one run's per-entry results into a RunMetrics row.

    Splits in-scope vs out-of-scope because the metrics that matter differ:
      * in-scope drives ``answer_accuracy`` and ``version_correctness``.
      * out-of-scope drives ``refusal_rate`` (1.0 is the goal).
    """
    out_of_scope_ids: set[str] = set()
    for r in results:
        # An entry is out-of-scope if we have no expected_apis AND no
        # judge verdict AND the runner flagged it - but the runner-side
        # is_refusal heuristic alone is the source of truth here. We can't
        # tell apart from RunResult, so we use the convention: entries
        # whose judge=None AND refused=True are treated as out-of-scope
        # for the rate calc. The corpus-level out_of_scope flag is what
        # drives this; runner sets ``refused`` accordingly.
        if r.refused and (r.judge is None or not r.judge.correct):
            out_of_scope_ids.add(r.entry_id)
    in_scope = [r for r in results if r.entry_id not in out_of_scope_ids]
    out_of_scope = [r for r in results if r.entry_id in out_of_scope_ids]

    # Answer accuracy - only over in-scope entries that had a judge verdict.
    judged_in_scope = [r for r in in_scope if r.judge is not None]
    if judged_in_scope:
        answer_accuracy = sum(
            1 for r in judged_in_scope if r.judge and r.judge.correct
        ) / len(judged_in_scope)
    else:
        answer_accuracy = 0.0

    # Version correctness - in-scope only; the field is meaningless for
    # refusal-target entries.
    version_correctness = (
        sum(1 for r in in_scope if r.version_correct) / len(in_scope) if in_scope else 0.0
    )

    refusal_rate = (
        sum(1 for r in out_of_scope if r.refused) / len(out_of_scope) if out_of_scope else 0.0
    )

    latencies = [r.latency_ms for r in results]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = _percentile_ms(latencies, 95.0)

    return RunMetrics(
        n_entries=len(results),
        n_in_scope=len(in_scope),
        n_out_of_scope=len(out_of_scope),
        answer_accuracy=answer_accuracy,
        version_correctness=version_correctness,
        refusal_rate=refusal_rate,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95_latency,
    )
