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

    Splits in-scope vs out-of-scope on the **corpus label** (``RunResult.
    out_of_scope``, copied from ``CorpusEntry.out_of_scope`` by the runner),
    because the metrics that matter differ:
      * in-scope drives ``answer_accuracy``, ``version_correctness``, and
        ``overrefusal_rate``.
      * out-of-scope drives ``refusal_rate`` (1.0 is the goal).

    Scope MUST come from the label, not the agent's behaviour. The earlier
    behaviour-derived split (``refused and not judge.correct -> out_of_scope``)
    was circular: an in-scope question the agent wrongly refused got
    reclassified as out-of-scope, silently dropped from the accuracy
    denominator AND counted as a *successful* refusal - so the agent could
    never be penalised for over-refusing, and an out-of-scope hallucination
    (answered, not refused) escaped ``refusal_rate`` entirely.
    """
    in_scope = [r for r in results if not r.out_of_scope]
    out_of_scope = [r for r in results if r.out_of_scope]

    # Answer accuracy - only over in-scope entries that had a judge verdict.
    # An in-scope entry the agent refused is judged incorrect (the judge fails
    # a refusal when a reference answer exists), so over-refusal correctly
    # lowers this number instead of vanishing from the denominator.
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

    # Refusal rate - fraction of out-of-scope entries the agent refused.
    # An out-of-scope entry the agent *answered* lowers this, as it should.
    refusal_rate = (
        sum(1 for r in out_of_scope if r.refused) / len(out_of_scope) if out_of_scope else 0.0
    )

    # Over-refusal rate - fraction of in-scope entries the agent wrongly
    # refused. The companion the old split hid; 0.0 is the goal.
    overrefusal_rate = (
        sum(1 for r in in_scope if r.refused) / len(in_scope) if in_scope else 0.0
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
        overrefusal_rate=overrefusal_rate,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95_latency,
    )
