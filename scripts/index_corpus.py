"""Headless indexer: populate Qdrant with every collection the eval corpus needs.

Normally you index a (library, version) by clicking "Index ..." in the VS Code
extension. The eval harness needs the same collections but shouldn't require the
editor, so this script drives ``DocIndexer`` directly for each distinct
``library@version`` in the corpus.

Usage (from the repo root, with Qdrant up and OPENAI_API_KEY in .env):

    uv --directory sidecar run python ../scripts/index_corpus.py

Or point at a different corpus / Qdrant:

    python scripts/index_corpus.py --corpus evals/corpus.json --qdrant-url http://localhost:6333

Idempotent: DocIndexer recreates each collection, so re-running re-indexes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SIDECAR_SRC = _REPO_ROOT / "sidecar" / "src"
if _SIDECAR_SRC.is_dir() and str(_SIDECAR_SRC) not in sys.path:
    sys.path.insert(0, str(_SIDECAR_SRC))

# Load the repo-root .env the same way the sidecar + eval harness do.
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", encoding="utf-8-sig")


def _distinct_pairs(corpus_path: Path) -> list[tuple[str, str]]:
    """Every distinct (library, version) in the corpus, in first-seen order."""
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    seen: dict[tuple[str, str], None] = {}
    for entry in data["entries"]:
        seen.setdefault((entry["library"], entry["version"]), None)
    return list(seen)


async def _run(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient

    from docchat_sidecar.indexer import DocIndexer, collection_name_for
    from docchat_sidecar.protocol import IndexComplete, IndexError as IndexErr

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set (put it in .env or the environment).")
        return 2

    pairs = _distinct_pairs(args.corpus)
    print(f"Indexing {len(pairs)} collection(s) from {args.corpus.name}:")
    for lib, ver in pairs:
        print(f"  - {lib}@{ver}  ->  {collection_name_for(lib, ver)}")
    print()

    openai = AsyncOpenAI()
    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    indexer = DocIndexer(qdrant=qdrant, openai=openai)

    failures = 0
    for lib, ver in pairs:
        print(f"==> {lib}@{ver}")
        terminal = None
        async for frame in indexer.index(lib, ver):
            if isinstance(frame, (IndexComplete, IndexErr)):
                terminal = frame
            elif getattr(frame, "note", None):
                print(f"    {frame.note}")
        if isinstance(terminal, IndexComplete):
            print(f"    OK: {collection_name_for(lib, ver)} ({terminal.chunks_indexed} chunks)")
        else:
            failures += 1
            msg = getattr(terminal, "message", "unknown error")
            print(f"    FAILED: {msg}")
        print()

    print(f"Done: {len(pairs) - failures}/{len(pairs)} collections indexed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="index_corpus")
    p.add_argument(
        "--corpus",
        type=Path,
        default=_REPO_ROOT / "evals" / "corpus.json",
        help="Corpus JSON whose (library, version) pairs to index.",
    )
    p.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL. Defaults to QDRANT_URL or localhost:6333.",
    )
    args = p.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
