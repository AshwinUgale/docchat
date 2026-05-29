"""Tests for the package.json + package-lock.json parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docchat_sidecar.lockfiles import Pin, parse_package_json


def _write(path: Path, content: dict[str, object] | str) -> None:
    payload = content if isinstance(content, str) else json.dumps(content)
    path.write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Happy path: lockfile wins
# ---------------------------------------------------------------------------


def test_lockfile_v3_pins_exact_version(tmp_path: Path) -> None:
    _write(
        tmp_path / "package.json",
        {
            "name": "demo",
            "dependencies": {"react": "^18.2.0", "lodash": "^4.17.21"},
        },
    )
    _write(
        tmp_path / "package-lock.json",
        {
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "demo", "version": "0.0.0"},
                "node_modules/react": {"version": "18.2.0"},
                "node_modules/lodash": {"version": "4.17.21"},
            },
        },
    )
    pins = parse_package_json(tmp_path / "package.json")
    assert Pin(library="react", version="18.2.0", source="package-lock.json") in pins
    assert Pin(library="lodash", version="4.17.21", source="package-lock.json") in pins


def test_lockfile_v1_pins_exact_version(tmp_path: Path) -> None:
    _write(
        tmp_path / "package.json",
        {"name": "demo", "dependencies": {"react": "^17.0.0"}},
    )
    _write(
        tmp_path / "package-lock.json",
        {
            "lockfileVersion": 1,
            "dependencies": {"react": {"version": "17.0.2"}},
        },
    )
    pins = parse_package_json(tmp_path / "package.json")
    assert pins == [Pin(library="react", version="17.0.2", source="package-lock.json")]


def test_dev_dependencies_included(tmp_path: Path) -> None:
    _write(
        tmp_path / "package.json",
        {
            "name": "demo",
            "dependencies": {"react": "^18.2.0"},
            "devDependencies": {"typescript": "^5.4.0"},
        },
    )
    _write(
        tmp_path / "package-lock.json",
        {
            "lockfileVersion": 3,
            "packages": {
                "node_modules/react": {"version": "18.2.0"},
                "node_modules/typescript": {"version": "5.4.5"},
            },
        },
    )
    pins = {p.library: p for p in parse_package_json(tmp_path / "package.json")}
    assert pins["react"].version == "18.2.0"
    assert pins["typescript"].version == "5.4.5"


# ---------------------------------------------------------------------------
# Fallback: manifest range parsing when lockfile missing
# ---------------------------------------------------------------------------


def test_fallback_strips_caret(tmp_path: Path) -> None:
    _write(
        tmp_path / "package.json",
        {"name": "demo", "dependencies": {"react": "^18.2.0"}},
    )
    pins = parse_package_json(tmp_path / "package.json")
    assert pins == [Pin(library="react", version="18.2.0", source="package.json")]


def test_fallback_strips_tilde_and_gte(tmp_path: Path) -> None:
    _write(
        tmp_path / "package.json",
        {
            "name": "demo",
            "dependencies": {"a": "~1.5.4", "b": ">=2.0.0", "c": "3.1.0"},
        },
    )
    pins = {p.library: p.version for p in parse_package_json(tmp_path / "package.json")}
    assert pins == {"a": "1.5.4", "b": "2.0.0", "c": "3.1.0"}


def test_fallback_skips_unparseable_ranges(tmp_path: Path) -> None:
    _write(
        tmp_path / "package.json",
        {
            "name": "demo",
            "dependencies": {"react": "latest", "x": "github:user/repo"},
        },
    )
    pins = parse_package_json(tmp_path / "package.json")
    assert pins == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_missing_manifest_returns_empty(tmp_path: Path) -> None:
    assert parse_package_json(tmp_path / "package.json") == []


def test_malformed_json_returns_empty(tmp_path: Path) -> None:
    _write(tmp_path / "package.json", "{ not valid")
    assert parse_package_json(tmp_path / "package.json") == []


def test_non_package_json_path_rejected(tmp_path: Path) -> None:
    other = tmp_path / "manifest.txt"
    _write(other, "ignored")
    with pytest.raises(ValueError, match=r"expected a package\.json"):
        parse_package_json(other)


def test_lockfile_takes_precedence_over_manifest_range(tmp_path: Path) -> None:
    """The lockfile pins 18.2.7 even when the manifest says ^18.2.0."""
    _write(
        tmp_path / "package.json",
        {"name": "demo", "dependencies": {"react": "^18.2.0"}},
    )
    _write(
        tmp_path / "package-lock.json",
        {
            "lockfileVersion": 3,
            "packages": {"node_modules/react": {"version": "18.2.7"}},
        },
    )
    pins = parse_package_json(tmp_path / "package.json")
    assert pins[0].version == "18.2.7"
    assert pins[0].source == "package-lock.json"
