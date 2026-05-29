"""Doc indexer - fetch + chunk + embed + store for one (library, version).

v0.2 indexes React only, sourcing the canonical MDX files from
``raw.githubusercontent.com/reactjs/react.dev/main/src/content``. The
collection is named after the PINNED version from the user's lockfile
(e.g. ``react_18_2_0``), even though v0.2 fetches from ``main`` -
version-pinned fetching (resolving a tag or sha for ``18.2.0``) lands at
v0.3.

Streams progress as ``IndexProgress`` messages so the chat panel can
render a live progress bar; emits ``IndexComplete`` or ``IndexError``
at the terminal state. Idempotent: re-indexing the same (library,
version) recreates the collection from scratch.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from docchat_sidecar.protocol import IndexComplete, IndexError, IndexProgress

__all__ = ["DocIndexer", "collection_name_for"]

logger = logging.getLogger(__name__)


# v0.2 - hand-picked React 18.2 reference pages. The MDX files in this repo
# track the React 18.2 -> 19 transition cleanly; for a v0.2 demo this is the
# right subset to prove the end-to-end pipeline. v0.3 expands to the full
# /content/reference/react/ tree and adds version pinning via git ref.
_REACT_18_2_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useState.md",
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useEffect.md",
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useContext.md",
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useReducer.md",
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useMemo.md",
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useCallback.md",
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useRef.md",
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useId.md",
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useSyncExternalStore.md",
    "https://raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/useTransition.md",
)

# OpenAI text-embedding-3-small dimensions (default).
_DEFAULT_DIMENSIONS = 1536
# Target chunk size in characters - rough proxy for ~500 tokens.
_CHUNK_TARGET_CHARS = 2000
# MDX import + export lines we strip before chunking. The .md files in
# the react.dev repo start with frontmatter + a couple of import statements
# that aren't useful retrieval text.
_MDX_NOISE_RE = re.compile(r"^(import |export )", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def collection_name_for(library: str, version: str) -> str:
    """Qdrant collection name for a (library, version) pair.

    Lowercases the library and replaces ``.`` with ``_`` in the version so
    Qdrant's collection-name constraints are satisfied. Example::

        collection_name_for("react", "18.2.0") -> "react_18_2_0"
    """
    safe_lib = re.sub(r"[^a-z0-9]+", "_", library.lower()).strip("_")
    safe_ver = re.sub(r"[^a-z0-9]+", "_", version.lower()).strip("_")
    return f"{safe_lib}_{safe_ver}"


@dataclass(frozen=True, kw_only=True)
class _Chunk:
    """One chunk ready to embed and upsert."""

    source_url: str
    chunk_index: int
    text: str


class DocIndexer:
    """Fetch + chunk + embed + write docs for one (library, version).

    Args:
        qdrant: Connected async Qdrant client. The indexer creates /
            recreates collections as needed.
        openai: OpenAI client for embeddings.
        embed_model: Embedding model name; default ``text-embedding-3-small``.
        http: Optional pre-configured ``httpx.AsyncClient`` for fetches.
            If None, a fresh client is created per ``index()`` call.
    """

    def __init__(
        self,
        *,
        qdrant: AsyncQdrantClient,
        openai: AsyncOpenAI,
        embed_model: str = "text-embedding-3-small",
        embed_dimensions: int = _DEFAULT_DIMENSIONS,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._qdrant = qdrant
        self._openai = openai
        self._embed_model = embed_model
        self._embed_dimensions = embed_dimensions
        self._http = http

    async def index(
        self, library: str, version: str
    ) -> AsyncIterator[IndexProgress | IndexComplete | IndexError]:
        """Yield streaming progress while indexing one (library, version).

        Terminal: exactly one ``IndexComplete`` OR ``IndexError`` per call.
        """
        urls = _urls_for(library, version)
        if not urls:
            yield IndexError(
                library=library,
                version=version,
                message=f"no indexer wired for {library}@{version} yet (v0.2 supports react only)",
            )
            return

        collection = collection_name_for(library, version)
        try:
            await self._reset_collection(collection)
        except Exception as exc:  # pragma: no cover - qdrant connection issues
            logger.exception("failed to reset collection %s", collection)
            yield IndexError(
                library=library, version=version, message=f"qdrant reset failed: {exc}"
            )
            return

        owns_http = self._http is None
        http = self._http or httpx.AsyncClient(timeout=30.0, follow_redirects=True)

        chunks: list[_Chunk] = []
        try:
            for page_index, url in enumerate(urls):
                yield IndexProgress(
                    library=library,
                    version=version,
                    chunks_done=len(chunks),
                    chunks_total=None,
                    note=f"fetching {url.rsplit('/', 1)[-1]}",
                )
                try:
                    response = await http.get(url)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning("skipping %s: %s", url, exc)
                    continue
                text = _clean_mdx(response.text)
                for idx, chunk_text in enumerate(_split_into_chunks(text)):
                    chunks.append(_Chunk(source_url=url, chunk_index=idx, text=chunk_text))
                yield IndexProgress(
                    library=library,
                    version=version,
                    chunks_done=len(chunks),
                    chunks_total=None,
                    note=f"fetched page {page_index + 1}/{len(urls)}",
                )

            total = len(chunks)
            if total == 0:
                yield IndexError(
                    library=library,
                    version=version,
                    message="fetched 0 chunks - check network or source URLs",
                )
                return

            # Embed + upsert in batches so we stream progress instead of
            # blocking through the whole batch.
            BATCH = 16
            for batch_start in range(0, total, BATCH):
                batch = chunks[batch_start : batch_start + BATCH]
                vectors = await self._embed([c.text for c in batch])
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "library": library,
                            "version": version,
                            "source_url": c.source_url,
                            "chunk_index": c.chunk_index,
                            "text": c.text,
                        },
                    )
                    for c, vector in zip(batch, vectors, strict=True)
                ]
                await self._qdrant.upsert(collection_name=collection, points=points)
                yield IndexProgress(
                    library=library,
                    version=version,
                    chunks_done=min(batch_start + BATCH, total),
                    chunks_total=total,
                    note="embedding + upserting",
                )

            yield IndexComplete(library=library, version=version, chunks_indexed=total)
        finally:
            if owns_http:
                await http.aclose()

    async def _reset_collection(self, collection: str) -> None:
        """Drop + recreate the collection so re-indexing is idempotent."""
        existing = await self._qdrant.collection_exists(collection_name=collection)
        if existing:
            await self._qdrant.delete_collection(collection_name=collection)
        await self._qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=self._embed_dimensions, distance=Distance.COSINE),
        )

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """OpenAI batched embedding call."""
        response = await self._openai.embeddings.create(model=self._embed_model, input=texts)
        # Order is guaranteed by the API contract; cast for the static type.
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Helpers (module-private; tested via the public API)
# ---------------------------------------------------------------------------


def _urls_for(library: str, version: str) -> tuple[str, ...]:
    """Source URLs for a given (library, version). v0.2 only knows React."""
    if library.lower() == "react":
        return _REACT_18_2_URLS
    return ()


def _clean_mdx(raw: str) -> str:
    """Strip MDX frontmatter + import/export lines so we're left with prose."""
    no_frontmatter = _FRONTMATTER_RE.sub("", raw, count=1)
    no_imports = _MDX_NOISE_RE.sub("", no_frontmatter)
    return no_imports.strip()


def _split_into_chunks(text: str) -> Iterable[str]:
    """Split into ~500-token chunks at paragraph boundaries.

    Simple paragraph-aware splitter: accumulate paragraphs until the running
    char count exceeds the target, then emit. Avoids slicing mid-sentence.
    """
    if not text.strip():
        return
    buffer: list[str] = []
    buffer_len = 0
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        para_len = len(paragraph)
        if buffer and buffer_len + para_len > _CHUNK_TARGET_CHARS:
            yield "\n\n".join(buffer)
            buffer = [paragraph]
            buffer_len = para_len
        else:
            buffer.append(paragraph)
            buffer_len += para_len + 2  # +2 for the joining "\n\n"
    if buffer:
        # Emit the trailing buffer regardless of size. Merging would either
        # require backtracking the iterator or duplicating an emitted chunk.
        # A small last chunk is fine for retrieval - it still embeds.
        yield "\n\n".join(buffer)
