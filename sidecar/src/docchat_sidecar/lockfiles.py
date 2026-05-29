"""Lockfile parser - extract concrete (library, version) pins from a project.

v0.2 supports ``package.json`` only. Multi-format support (requirements.txt /
poetry.lock / go.mod / Cargo.toml) lands at v0.2.x. The Protocol stays
identical regardless of source format - every parser returns a ``list[Pin]``.

Why ``package.json`` ALWAYS resolves through ``package-lock.json`` when one
is present: the manifest declares ranges (``^18.2.0``); the lockfile pins
the EXACT installed version (``18.2.0`` or ``18.2.7``). DocChat indexes
docs for the exact pinned version, not the range — so the lock wins
whenever it's available. When no lockfile exists we strip ``^`` / ``~`` /
``>=`` and use the lowest concrete version we can read from the range as a
"best effort"; callers should treat that as approximate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Pin", "parse_package_json"]


# Strip the common npm range operators. We don't try to do full semver
# resolution here - if the lockfile is missing we just want a concrete-looking
# version string for display + Qdrant collection naming.
_RANGE_PREFIX_RE = re.compile(r"^[\^~>=<]+")
_VERSION_TAIL_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?)")


@dataclass(frozen=True, kw_only=True)
class Pin:
    """One concrete (library, version) pin extracted from a project lockfile.

    Attributes:
        library: Package name as it appears in the registry / on disk.
        version: Concrete version string. When sourced from the lockfile
            this is exact; when fallback-parsed from the manifest range
            it's the lowest version compatible with the range.
        source: Which file produced this pin ("package-lock.json" or
            "package.json"). Useful for the UI to flag "approximate" pins.
    """

    library: str
    version: str
    source: str


def parse_package_json(path: Path | str) -> list[Pin]:
    """Parse a ``package.json`` (and its lockfile if present) into pins.

    Args:
        path: Path to ``package.json``. If a sibling ``package-lock.json``
            exists, exact versions are resolved through it.

    Returns:
        Pins for every entry in ``dependencies`` + ``devDependencies``.
        Empty list if the manifest is missing or malformed.

    Raises:
        ValueError: If ``path`` does not end in ``package.json``.
    """
    manifest_path = Path(path)
    if manifest_path.name != "package.json":
        raise ValueError(f"expected a package.json, got {manifest_path.name!r}")
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(manifest, dict):
        return []

    lock_versions = _read_lockfile(manifest_path.parent / "package-lock.json")

    pins: list[Pin] = []
    for key in ("dependencies", "devDependencies"):
        block = manifest.get(key)
        if not isinstance(block, dict):
            continue
        for library, declared in block.items():
            if not isinstance(library, str) or not isinstance(declared, str):
                continue
            exact = lock_versions.get(library)
            if exact:
                pins.append(Pin(library=library, version=exact, source="package-lock.json"))
            else:
                fallback = _strip_range(declared)
                if fallback:
                    pins.append(Pin(library=library, version=fallback, source="package.json"))
    return pins


def _read_lockfile(lock_path: Path) -> dict[str, str]:
    """Map ``library -> exact version`` from package-lock.json.

    npm has shipped three lockfile schemas (v1 / v2 / v3). All three put
    direct deps under a ``packages`` (v2/v3) or ``dependencies`` (v1) key
    with the package name as the lookup. We handle both shapes.
    """
    if not lock_path.exists():
        return {}
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, str] = {}
    # v2 / v3: packages dict keyed by "node_modules/<name>" -> {"version": "..."}
    packages = data.get("packages")
    if isinstance(packages, dict):
        for key, info in packages.items():
            if not isinstance(info, dict):
                continue
            version = info.get("version")
            if not isinstance(version, str):
                continue
            # The root project is keyed as "" - skip it.
            if key == "":
                continue
            name = key.removeprefix("node_modules/").rsplit("/node_modules/", maxsplit=1)[-1]
            if name:
                out.setdefault(name, version)
    # v1: top-level dependencies dict
    deps = data.get("dependencies")
    if isinstance(deps, dict):
        for name, info in deps.items():
            if not isinstance(info, dict):
                continue
            version = info.get("version")
            if isinstance(name, str) and isinstance(version, str):
                out.setdefault(name, version)
    return out


def _strip_range(declared: str) -> str | None:
    """Approximate concrete version from an npm range string.

    Examples:
        ``"^18.2.0"``  -> ``"18.2.0"``
        ``"~1.5.4"``   -> ``"1.5.4"``
        ``">=2.0.0"``  -> ``"2.0.0"``
        ``"latest"``   -> ``None``
        ``"github:..."`` -> ``None``

    Returns ``None`` when the declared string doesn't contain a parseable
    semver tail; callers should treat that as "unknown version, can't index".
    """
    cleaned = _RANGE_PREFIX_RE.sub("", declared.strip())
    match = _VERSION_TAIL_RE.search(cleaned)
    return match.group(1) if match else None
