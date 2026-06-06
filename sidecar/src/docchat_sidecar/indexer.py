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
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from docchat_sidecar.protocol import IndexComplete, IndexError, IndexProgress

__all__ = ["DocIndexer", "collection_name_for"]

logger = logging.getLogger(__name__)


# v1.0: per-library config + git-ref resolution.
#
# Each library declares:
#   - repo: GitHub repo holding the docs source
#   - paths: list of doc-source paths relative to repo root
#   - ref_for(version): callable returning the git ref to fetch from
#
# For libraries where the docs site lives in the SAME repo as the
# released source (tiangolo/fastapi, pallets/flask, etc.), ``ref_for``
# returns a version tag - so fetching fastapi 0.100.0 actually serves
# the 0.100.0-era docs with Pydantic v2 idioms. For libraries where
# the docs live in a separate repo not tagged per release (reactjs/
# react.dev, vuejs/docs), ``ref_for`` returns ``"main"`` and the chunk
# metadata still surfaces the user's pinned version via the collection
# name + chunk header.
_REACT_DOC_PATHS: tuple[str, ...] = (
    "src/content/reference/react/useState.md",
    "src/content/reference/react/useEffect.md",
    "src/content/reference/react/useContext.md",
    "src/content/reference/react/useReducer.md",
    "src/content/reference/react/useMemo.md",
    "src/content/reference/react/useCallback.md",
    "src/content/reference/react/useRef.md",
    "src/content/reference/react/useId.md",
    "src/content/reference/react/useSyncExternalStore.md",
    "src/content/reference/react/useTransition.md",
)

_FASTAPI_DOC_PATHS: tuple[str, ...] = (
    "docs/en/docs/tutorial/first-steps.md",
    "docs/en/docs/tutorial/path-params.md",
    "docs/en/docs/tutorial/query-params.md",
    "docs/en/docs/tutorial/body.md",
    "docs/en/docs/tutorial/response-model.md",
    "docs/en/docs/tutorial/dependencies/index.md",
    "docs/en/docs/tutorial/background-tasks.md",
    "docs/en/docs/tutorial/middleware.md",
    "docs/en/docs/tutorial/cors.md",
    "docs/en/docs/tutorial/dependencies/dependencies-with-yield.md",
)

_VUE_DOC_PATHS: tuple[str, ...] = (
    "src/api/reactivity-core.md",
    "src/api/reactivity-utilities.md",
    "src/api/composition-api-setup.md",
    "src/api/composition-api-lifecycle.md",
    "src/api/composition-api-dependency-injection.md",
    "src/api/general.md",
    "src/api/sfc-script-setup.md",
    "src/guide/essentials/reactivity-fundamentals.md",
    "src/guide/essentials/computed.md",
    "src/guide/essentials/watchers.md",
)


@dataclass(frozen=True, kw_only=True)
class _LibraryConfig:
    """Per-library doc-source config used by _urls_for to build URLs.

    Attributes:
        repo: ``"<owner>/<name>"`` on github.com.
        paths: Doc paths relative to repo root.
        ref_for: Maps a pinned version to a git ref. Returns the version
            itself when the docs are tagged per release (FastAPI); returns
            ``"main"`` when the docs repo isn't version-tagged (React,
            Vue) - the chunk metadata still surfaces the user's pinned
            version via the collection name + chunk header.
    """

    repo: str
    paths: tuple[str, ...]
    ref_for: Callable[[str], str]


def _fastapi_ref(version: str) -> str:
    """FastAPI is tagged per release; the docs at that tag reflect the
    correct Pydantic generation (v1 for <0.100, v2 for >=0.100)."""
    return version


def _docs_repo_main(_: str) -> str:
    """React/Vue docs aren't version-tagged; always fetch from main."""
    return "main"


_LIBRARY_CONFIG: dict[str, _LibraryConfig] = {
    "react": _LibraryConfig(
        repo="reactjs/react.dev",
        paths=_REACT_DOC_PATHS,
        ref_for=_docs_repo_main,
    ),
    "fastapi": _LibraryConfig(
        repo="tiangolo/fastapi",
        paths=_FASTAPI_DOC_PATHS,
        ref_for=_fastapi_ref,
    ),
    "vue": _LibraryConfig(
        repo="vuejs/docs",
        paths=_VUE_DOC_PATHS,
        ref_for=_docs_repo_main,
    ),
}

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
    """One chunk ready to embed and upsert.

    v0.8 enriches the payload with ``api_name`` (derived from the source
    filename) and ``section_heading`` (the most recent H2 the chunk falls
    under). Both feed into ``SearchDocsTool``'s prompt header so the LLM
    sees which API a chunk is about, not just "this came from useState.md".
    """

    source_url: str
    chunk_index: int
    text: str
    api_name: str
    section_heading: str | None


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
                message=(
                    f"no indexer wired for {library}@{version} yet "
                    "(v1.0 supports react + fastapi + vue; FastAPI fetches "
                    "from the version tag, React and Vue from main)"
                ),
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
                api_name = _api_name_from_url(url)
                for idx, (chunk_text, section_heading) in enumerate(_split_into_chunks(text)):
                    chunks.append(
                        _Chunk(
                            source_url=url,
                            chunk_index=idx,
                            text=chunk_text,
                            api_name=api_name,
                            section_heading=section_heading,
                        )
                    )
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
                            # v0.8: chunk-level metadata for tighter version
                            # grounding + per-API filtering in SearchDocsTool.
                            "api_name": c.api_name,
                            "section_heading": c.section_heading,
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
    """Source URLs for a given (library, version).

    v1.0: per-library git-ref resolution via ``_LIBRARY_CONFIG``. For
    FastAPI the ref is the version tag (so 0.95.2 vs 0.100.0 actually
    fetch different content - the same-library-two-versions demo).
    React + Vue still fetch from ``main`` because their docs repos
    aren't version-tagged; the chunk metadata still surfaces the
    user's pinned version via the collection name + chunk header.
    """
    config = _LIBRARY_CONFIG.get(library.lower())
    if config is None:
        return ()
    ref = config.ref_for(version)
    base = f"https://raw.githubusercontent.com/{config.repo}/{ref}"
    return tuple(f"{base}/{path}" for path in config.paths)


def _api_name_from_url(url: str) -> str:
    """Derive a stable API name from a doc-source URL.

    Used at index time to tag each chunk so SearchDocsTool can surface it
    in the prompt header. Examples::

        ".../reference/react/useState.md"        -> "useState"
        ".../docs/tutorial/dependencies/index.md" -> "dependencies"
        ".../api/composition-api-setup.md"        -> "composition-api-setup"
    """
    tail = url.rsplit("/", 1)[-1]
    stem = tail.removesuffix(".md").removesuffix(".mdx")
    if stem == "index":
        parts = url.rstrip("/").split("/")
        if len(parts) >= 2:
            return parts[-2]
    return stem


def _clean_mdx(raw: str) -> str:
    """Strip MDX frontmatter + import/export lines so we're left with prose."""
    no_frontmatter = _FRONTMATTER_RE.sub("", raw, count=1)
    no_imports = _MDX_NOISE_RE.sub("", no_frontmatter)
    return no_imports.strip()


_H2_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")


def _split_into_chunks(text: str) -> Iterable[tuple[str, str | None]]:
    """Split into ~500-token chunks at paragraph boundaries.

    v0.8: yields ``(chunk_text, section_heading)`` pairs. The
    ``section_heading`` is the most recent H2 (``## ...``) line at the
    chunk's start, used downstream to surface "## react@18.2 - useState
    (under Reference)" headers in the SearchDocsTool output. ``None`` if
    the chunk doesn't fall under any H2 (e.g., pre-frontmatter content
    that survived the cleaner).

    Simple paragraph-aware splitter: accumulate paragraphs until the
    running char count exceeds the target, then emit. Avoids slicing
    mid-sentence.
    """
    if not text.strip():
        return
    buffer: list[str] = []
    buffer_len = 0
    current_heading: str | None = None
    # Heading active when the chunk *started* accumulating. We freeze it
    # at chunk start so a chunk that spans a heading boundary still
    # carries the heading where it began.
    chunk_start_heading: str | None = None
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # If this paragraph IS (or starts with) an H2, capture it as the
        # current heading. Markdown chunkers often see "## Foo\nbody..."
        # as a single paragraph when there's no blank line after the
        # heading; handle both shapes.
        first_line = paragraph.splitlines()[0]
        match = _H2_HEADING_RE.match(first_line)
        if match:
            current_heading = match.group(1).strip()
        para_len = len(paragraph)
        if buffer and buffer_len + para_len > _CHUNK_TARGET_CHARS:
            yield "\n\n".join(buffer), chunk_start_heading
            buffer = [paragraph]
            buffer_len = para_len
            chunk_start_heading = current_heading
        else:
            if not buffer:
                chunk_start_heading = current_heading
            buffer.append(paragraph)
            buffer_len += para_len + 2  # +2 for the joining "\n\n"
    if buffer:
        # Emit the trailing buffer regardless of size. Merging would either
        # require backtracking the iterator or duplicating an emitted chunk.
        # A small last chunk is fine for retrieval - it still embeds.
        yield "\n\n".join(buffer), chunk_start_heading
