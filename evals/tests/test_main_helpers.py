"""Tests for the pure CLI helpers in ``evals.__main__``."""

from __future__ import annotations

import pytest

from evals.__main__ import _parse_floor_overrides, _select_entries
from evals.schema import CorpusEntry


def _entry(entry_id: str, split: str | None) -> CorpusEntry:
    return CorpusEntry(
        id=entry_id,
        library="react",
        version="18.2.0",
        question="q",
        expected_answer="a",
        split=split,
    )


# ---------------------------------------------------------------------------
# _parse_floor_overrides
# ---------------------------------------------------------------------------


def test_parse_floor_overrides_empty() -> None:
    assert _parse_floor_overrides(None) == {}
    assert _parse_floor_overrides([]) == {}


def test_parse_floor_overrides_pairs_lowercased() -> None:
    assert _parse_floor_overrides(["FastAPI=0.12", "vue=0.05"]) == {
        "fastapi": 0.12,
        "vue": 0.05,
    }


def test_parse_floor_overrides_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        _parse_floor_overrides(["fastapi"])  # no '='
    with pytest.raises(ValueError):
        _parse_floor_overrides(["=0.1"])  # empty library


# ---------------------------------------------------------------------------
# _select_entries
# ---------------------------------------------------------------------------


def test_select_all_returns_everything() -> None:
    entries = [_entry("a", "calibration"), _entry("b", "test"), _entry("c", None)]
    assert len(_select_entries(entries, "all")) == 3


def test_select_split_keeps_matching_and_untagged() -> None:
    entries = [_entry("a", "calibration"), _entry("b", "test"), _entry("c", None)]
    test = _select_entries(entries, "test")
    ids = {e.id for e in test}
    assert ids == {"b", "c"}  # matching split + untagged (belongs to every split)


def test_select_untagged_corpus_is_unaffected_by_split() -> None:
    entries = [_entry("a", None), _entry("b", None)]
    # No tags anywhere -> any --split value runs the whole corpus.
    assert len(_select_entries(entries, "test")) == 2
    assert len(_select_entries(entries, "calibration")) == 2
