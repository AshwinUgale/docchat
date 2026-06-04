"""Agent tools.

v0.6 makes all three tools real:
- ``SearchDocsTool`` (real since v0.3) - Qdrant retrieval over the user's
  pinned-version doc collection. Drops hits below the cosine floor (ADR-008).
- ``SearchWorkspaceCodeTool`` (real at v0.6) - shells out to ripgrep over
  the open VS Code workspace path. Returns up to 5 file:line:snippet hits.
- ``FindInChangelogTool`` (real at v0.6) - fetches the library's
  CHANGELOG.md from a hardcoded per-library raw-GitHub URL, slices out the
  version section.

Each tool advertises an OpenAI function-call schema via ``tool_schemas()``;
the schemas feed into ``FunctionSchemaSource`` so ToolPicker can rank them
per query, and into ``Agent._openai_tool_specs`` so the chat-completions
``tools`` array is the same shape.

Tools return ``ToolResult`` objects carrying both the text the LLM should
see AND the structured citation list the chat panel renders separately.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from docchat_sidecar.indexer import collection_name_for

__all__ = [
    "Citation",
    "FindInChangelogTool",
    "SearchDocsTool",
    "SearchWorkspaceCodeTool",
    "ToolResult",
    "tool_schemas",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Citation:
    """Inline citation token rendered by the chat panel.

    Format: ``[react@18.2.0:useState.md]``. v0.5+ layers click-to-open on
    top of this same shape.
    """

    library: str
    version: str
    source: str  # display label, typically the doc page filename

    def render(self) -> str:
        return f"[{self.library}@{self.version}:{self.source}]"


@dataclass(kw_only=True)
class ToolResult:
    """What a tool returns to the agent loop.

    ``text`` is the substrate the LLM sees as the tool's output. ``citations``
    is the structured list the panel renders separately so users can see
    where the answer came from.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# search_docs - Qdrant retrieval over the pinned-version doc collection
# ---------------------------------------------------------------------------


class SearchDocsTool:
    """Retrieve top-k doc chunks from the user's pinned-version collection."""

    name = "search_docs"
    description = (
        "Search the indexed documentation for a specific library and version. "
        "Use this when the user asks how to use an API, what a hook does, what "
        "arguments a function takes, or any 'how do I' question about a library "
        "the user has pinned in their project."
    )

    def __init__(
        self,
        *,
        qdrant: AsyncQdrantClient,
        openai: AsyncOpenAI,
        embed_model: str = "text-embedding-3-small",
        top_k: int = 5,
        score_floor: float = 0.15,
        floors_by_library: dict[str, float] | None = None,
    ) -> None:
        self._qdrant = qdrant
        self._openai = openai
        self._embed_model = embed_model
        self._top_k = top_k
        # v0.5: drop hits below this cosine-similarity floor. The first
        # eval pass at floor=0.25 over-pruned: 8/16 in-scope corpus entries
        # had their top hit below 0.25 (the React 18.2 doc corpus is only
        # 10 hook pages, and verbose question phrasings dilute the
        # similarity score). 0.15 keeps the legitimate-but-weak in-scope
        # hits while still blocking the four out-of-scope queries (Flask /
        # Vite / Node EADDRINUSE / let-vs-const) whose top hits clustered
        # below 0.15. See ADR-008.
        self._score_floor = score_floor
        # v0.7: per-library floor overrides. FastAPI's tutorial pages are
        # bigger / more varied than React's hook reference; eval at v0.6
        # showed many in-scope FastAPI queries landing at top-hit scores
        # of 0.08-0.12 (below the React-tuned 0.15 default), driving the
        # over-refusal that capped recall at 8/24 in-scope answered. A
        # lower default for fastapi recovers some of that without affecting
        # React's precision. Callers (eval harness, settings UI) can
        # override via the kwarg.
        self._floors_by_library: dict[str, float] = {
            "fastapi": 0.10,
            **(floors_by_library or {}),
        }

    def _floor_for(self, library: str) -> float:
        return self._floors_by_library.get(library.lower(), self._score_floor)

    async def run(self, *, library: str, version: str, query: str) -> ToolResult:
        collection = collection_name_for(library, version)
        if not await self._qdrant.collection_exists(collection_name=collection):
            return ToolResult(
                text=(
                    f"No indexed docs for {library}@{version}. "
                    f"The user needs to click 'Index {library} {version}' first."
                )
            )
        response = await self._openai.embeddings.create(model=self._embed_model, input=[query])
        query_vector = response.data[0].embedding
        # ``query_points`` is the current qdrant-client API; ``search`` is
        # deprecated in 1.10+ and dropped from the typed surface. The
        # response wraps a ``points`` list of ScoredPoint objects.
        query_response = await self._qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            limit=self._top_k,
        )
        raw_hits = query_response.points
        # v0.5: drop low-confidence hits. The agent prompt then refuses
        # cleanly on the empty result rather than hallucinating from base
        # knowledge. v0.7: per-library floor override (see __init__).
        floor = self._floor_for(library)
        hits = [h for h in raw_hits if getattr(h, "score", 0.0) >= floor]
        if not hits:
            return ToolResult(text=f"No relevant chunks found for {query!r}.")
        text_parts: list[str] = []
        citations: list[Citation] = []
        seen_sources: set[str] = set()
        # v0.7: prefix each chunk with the pinned library@version. Combined
        # with the agent's "any API you mention must appear in retrieved
        # chunks" prompt rule, this gives the LLM an explicit version
        # anchor in every retrieval block and closes the leak path where
        # the model would synthesise across versions.
        for hit in hits:
            payload = hit.payload or {}
            chunk_text = payload.get("text", "")
            source_url = payload.get("source_url", "")
            source_label = source_url.rsplit("/", 1)[-1] if source_url else "doc"
            # Use the payload's library + version (what was actually
            # indexed), not the caller's args, so a misdispatched tool
            # call still gets honest provenance.
            payload_lib = payload.get("library", library)
            payload_ver = payload.get("version", version)
            text_parts.append(f"## {payload_lib}@{payload_ver} - {source_label}\n\n{chunk_text}")
            if source_label not in seen_sources:
                citations.append(
                    Citation(library=payload_lib, version=payload_ver, source=source_label)
                )
                seen_sources.add(source_label)
        return ToolResult(text="\n\n---\n\n".join(text_parts), citations=citations)


# ---------------------------------------------------------------------------
# search_workspace_code - real at v0.6 via ripgrep
# ---------------------------------------------------------------------------


# Cap on bytes returned per rg invocation - prevents giant binary matches
# from blowing up the LLM context.
_RG_MAX_BYTES = 64 * 1024
# Max files we extract from rg's JSON output before truncating.
_RG_MAX_HITS = 5
# Context lines around each match.
_RG_CONTEXT_LINES = 2


class SearchWorkspaceCodeTool:
    """Search the user's open VS Code workspace via ripgrep.

    v0.6 shells out to ``rg --json`` over the workspace path. Graceful
    degradation: if ``rg`` is not on PATH or no workspace was opened, the
    tool returns a short text explaining the situation so the agent can
    refuse cleanly instead of hallucinating.
    """

    name = "search_workspace_code"
    description = (
        "Search the user's open VS Code workspace for code that uses a specific "
        "API or pattern. Use this when the user asks 'where in my code do I...' "
        "or 'show me how I'm using X' - questions about THEIR code, not the "
        "library's docs."
    )

    def __init__(
        self,
        *,
        workspace_path: str | Path | None = None,
        rg_binary: str = "rg",
        max_hits: int = _RG_MAX_HITS,
        context_lines: int = _RG_CONTEXT_LINES,
    ) -> None:
        self._workspace_path = Path(workspace_path) if workspace_path else None
        self._rg_binary = rg_binary
        self._max_hits = max_hits
        self._context_lines = context_lines

    async def run(self, *, query: str) -> ToolResult:
        if self._workspace_path is None:
            return ToolResult(
                text=(
                    "No workspace is open - I can't search workspace code. "
                    "Open a folder in VS Code and try again."
                )
            )
        if not self._workspace_path.is_dir():
            return ToolResult(text=f"Workspace path {self._workspace_path} is not a directory.")

        # ripgrep is the tool. We pass --json so the parser is deterministic
        # and -C for surrounding context. --max-count caps per-file hits.
        cmd = [
            self._rg_binary,
            "--json",
            "--max-count",
            "3",
            "-C",
            str(self._context_lines),
            "--",
            query,
            str(self._workspace_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=8.0)
        except FileNotFoundError:
            return ToolResult(
                text=(
                    "ripgrep (rg) is not installed on this system. "
                    "Install it via 'winget install BurntSushi.ripgrep.MSVC' "
                    "(Windows) or 'brew install ripgrep' (macOS) to enable "
                    "workspace code search."
                )
            )
        except TimeoutError:
            return ToolResult(text="Workspace search timed out after 8 seconds.")

        # Truncate to be safe before parsing.
        stdout = stdout_bytes[:_RG_MAX_BYTES].decode("utf-8", errors="replace")
        hits = _parse_rg_json(stdout, max_hits=self._max_hits)
        if not hits:
            return ToolResult(text=f"No matches for {query!r} in the workspace.")

        text_parts: list[str] = []
        for hit in hits:
            text_parts.append(f"## {hit['path']}:{hit['line']}\n\n```\n{hit['text']}\n```")
        return ToolResult(text="\n\n".join(text_parts))


def _parse_rg_json(stdout: str, *, max_hits: int) -> list[dict[str, Any]]:
    """Extract the top N matches from ``rg --json`` output.

    rg emits one JSON object per line; we want the ``type=="match"`` ones.
    """
    hits: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            evt = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "match":
            continue
        data = evt.get("data", {})
        path_info = data.get("path", {})
        line_info = data.get("line_number")
        lines_info = data.get("lines", {})
        # rg's path can be {"text": "..."} or {"bytes": "..."}; prefer text.
        path = path_info.get("text") or path_info.get("bytes") or "?"
        line_text = lines_info.get("text") or ""
        hits.append({"path": path, "line": line_info or 0, "text": line_text.rstrip()})
        if len(hits) >= max_hits:
            break
    return hits


# ---------------------------------------------------------------------------
# find_in_changelog - real at v0.6 via GitHub raw fetch
# ---------------------------------------------------------------------------


# Per-library CHANGELOG / release-notes URL on raw.githubusercontent.com.
# Same shape as the indexer's URL map - hardcoded for the libraries we
# actually index. v0.7+ can promote this to a config table.
_CHANGELOG_URLS: dict[str, str] = {
    "react": "https://raw.githubusercontent.com/facebook/react/main/CHANGELOG.md",
    "fastapi": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/release-notes.md",
}

_CHANGELOG_CACHE: dict[str, str] = {}


class FindInChangelogTool:
    """Find breaking-change notes for a library version.

    v0.6 fetches the library's CHANGELOG.md from raw.githubusercontent.com
    (cached per-library for the process lifetime), grep-style filters the
    text for paragraphs mentioning the requested version, and returns the
    matching slice.
    """

    name = "find_in_changelog"
    description = (
        "Look up entries in a library's CHANGELOG or release notes for a specific "
        "version. Use this when the user asks 'what changed in version X', "
        "'is this deprecated in Y', or 'when was Z added'."
    )

    def __init__(self, *, http: httpx.AsyncClient | None = None) -> None:
        self._http = http

    async def run(self, *, library: str, version: str, query: str) -> ToolResult:
        url = _CHANGELOG_URLS.get(library.lower())
        if url is None:
            return ToolResult(
                text=(
                    f"No CHANGELOG source configured for {library!r}. "
                    f"v0.6 supports: {', '.join(sorted(_CHANGELOG_URLS))}."
                )
            )

        try:
            text = await self._fetch_changelog(url)
        except httpx.HTTPError as exc:
            return ToolResult(text=f"Failed to fetch CHANGELOG for {library}: {exc}")

        slice_text = _extract_version_section(text, version=version, query=query)
        if not slice_text:
            return ToolResult(
                text=(f"No CHANGELOG entry mentioning {library}@{version} matched {query!r}.")
            )
        citation = Citation(library=library, version=version, source="CHANGELOG.md")
        return ToolResult(text=slice_text, citations=[citation])

    async def _fetch_changelog(self, url: str) -> str:
        if url in _CHANGELOG_CACHE:
            return _CHANGELOG_CACHE[url]
        owns_http = self._http is None
        http = self._http or httpx.AsyncClient(timeout=15.0, follow_redirects=True)
        try:
            response = await http.get(url)
            response.raise_for_status()
            _CHANGELOG_CACHE[url] = response.text
            return response.text
        finally:
            if owns_http:
                await http.aclose()


def _extract_version_section(text: str, *, version: str, query: str) -> str:
    """Return CHANGELOG sections whose HEADING matches the requested version.

    v0.6.1 fix: previously this used a plain substring match (``version in
    section.lower()``) which let adjacent-version context leak in - asking
    about React 18.2.0 would match the React 19 section because the text
    "18.2.0" also appears there in a "fixed in 18.2.0" backport note. That
    leak surfaced in the v0.6 eval as ``version_correctness=0.750`` (the
    model picked up React-19 / Pydantic-v2 APIs from supposedly-version-
    scoped context).

    v0.6.1 strategy:
    1. Split on H2 headings (``## ...``).
    2. A section qualifies only if the HEADING contains the version,
       anchored as a word boundary - so "0.95.0" no longer matches
       "0.95.10" or "0.95.0" inside a longer string.
    3. If a query is given, further require it to appear in the section
       text (body OR heading) - this filters huge release-notes pages.

    Truncated to ~6 KB so the LLM context isn't blown on a giant section.
    """
    parts = re.split(r"(?m)^(## .+)$", text)
    # parts now looks like: [pre-text, heading1, body1, heading2, body2, ...]
    section_pairs: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        heading = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        section_pairs.append((heading, body))

    # Word-boundary version match so "0.95.0" doesn't match "0.95.10".
    # Escape the version since it contains dots which are regex metachars.
    version_pattern = re.compile(r"\b" + re.escape(version) + r"\b", re.IGNORECASE)
    query_lower = (query or "").lower()

    matching: list[str] = []
    for heading, body in section_pairs:
        # Strict: the version must appear in the HEADING, not just the body.
        # Adjacent-version sections that happen to mention the requested
        # version in passing don't qualify any more.
        if not version_pattern.search(heading):
            continue
        section_text = f"{heading}\n{body}".strip()
        if query_lower and query_lower not in section_text.lower():
            continue
        matching.append(section_text)

    if not matching:
        return ""
    joined = "\n\n---\n\n".join(matching)
    if len(joined) > 6000:
        joined = joined[:6000] + "\n\n[... truncated]"
    return joined


# ---------------------------------------------------------------------------
# Tool schemas fed to ToolPicker + Agent's OpenAI function-call wrapper
# ---------------------------------------------------------------------------


def tool_schemas() -> list[dict[str, Any]]:
    """Function-call schemas describing every v0.6 tool.

    Same shape as v0.3 (``{name, description, parameters}``); the Agent
    wraps each in OpenAI's ``{"type": "function", "function": {...}}``
    envelope before passing to the chat completions API.
    """
    return [
        {
            "name": SearchDocsTool.name,
            "description": SearchDocsTool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "library": {"type": "string"},
                    "version": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["library", "version", "query"],
            },
        },
        {
            "name": SearchWorkspaceCodeTool.name,
            "description": SearchWorkspaceCodeTool.description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": FindInChangelogTool.name,
            "description": FindInChangelogTool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "library": {"type": "string"},
                    "version": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["library", "version", "query"],
            },
        },
    ]
