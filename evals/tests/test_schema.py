"""Schema validation tests for the eval corpus + run outputs."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from evals.schema import Corpus, CorpusEntry, RunMetrics


def test_bundled_corpus_loads() -> None:
    corpus_path = Path(__file__).resolve().parent.parent / "corpus.json"
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus = TypeAdapter(Corpus).validate_python(data)
    assert corpus.name
    assert corpus.entries
    assert len(corpus.entries) == 20


def test_bundled_corpus_has_in_and_out_of_scope() -> None:
    corpus_path = Path(__file__).resolve().parent.parent / "corpus.json"
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus = TypeAdapter(Corpus).validate_python(data)
    in_scope = [e for e in corpus.entries if not e.out_of_scope]
    oos = [e for e in corpus.entries if e.out_of_scope]
    # 16 in-scope + 4 out-of-scope - the documented mix in description.
    assert len(in_scope) == 16
    assert len(oos) == 4


def test_bundled_corpus_ids_are_unique() -> None:
    corpus_path = Path(__file__).resolve().parent.parent / "corpus.json"
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus = TypeAdapter(Corpus).validate_python(data)
    ids = [e.id for e in corpus.entries]
    assert len(ids) == len(set(ids))


def test_corpus_entry_defaults() -> None:
    entry = CorpusEntry(
        id="x",
        library="react",
        version="18.2.0",
        question="?",
        expected_answer="!",
    )
    assert entry.expected_apis == []
    assert entry.forbidden_apis == []
    assert entry.out_of_scope is False
    assert entry.wrong_for_other_version is None


def test_run_metrics_serialises_round_trip() -> None:
    metrics = RunMetrics(
        n_entries=20,
        n_in_scope=16,
        n_out_of_scope=4,
        answer_accuracy=0.875,
        version_correctness=0.9375,
        refusal_rate=1.0,
        mean_latency_ms=2400.0,
        p95_latency_ms=4100.0,
    )
    raw = metrics.model_dump_json()
    restored = RunMetrics.model_validate_json(raw)
    assert restored.answer_accuracy == 0.875
    assert restored.refusal_rate == 1.0
