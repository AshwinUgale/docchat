"""Agent tools - one real, two stubs.

v0.3 ships three tools so ``ToolPicker`` has a real routing problem
(picking between them with 4+ tools is when ToolPicker starts paying for
itself; with 1 tool it's decorative). Only ``search_docs`` is real;
``search_workspace_code`` and ``find_in_changelog`` return placeholders
that v0.4+ will fill in.

Each tool advertises an OpenAI function-call schema via
``tool_schemas()``; the schemas feed into ``FunctionSchemaSource`` so
ToolPicker can rank them per query. The agent loop then dispatches to the
picked tool's ``run()`` method.

Tools return ``ToolResult`` objects carrying both the text the LLM should
see AND the citation list the chat panel renders separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass(frozen=True, kw_only=True)
class Citation:
    """Inline citation token rendered by the chat panel.

    Format: ``[react@18.2.0:useState.md]``. v0.5 layers click-to-open on
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
# search_docs - the only real tool at v0.3
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
    ) -> None:
        self._qdrant = qdrant
        self._openai = openai
        self._embed_model = embed_model
        self._top_k = top_k

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
        hits = query_response.points
        if not hits:
            return ToolResult(text=f"No relevant chunks found for {query!r}.")
        text_parts: list[str] = []
        citations: list[Citation] = []
        seen_sources: set[str] = set()
        for hit in hits:
            payload = hit.payload or {}
            chunk_text = payload.get("text", "")
            source_url = payload.get("source_url", "")
            source_label = source_url.rsplit("/", 1)[-1] if source_url else "doc"
            text_parts.append(f"## {source_label}\n\n{chunk_text}")
            if source_label not in seen_sources:
                citations.append(Citation(library=library, version=version, source=source_label))
                seen_sources.add(source_label)
        return ToolResult(text="\n\n---\n\n".join(text_parts), citations=citations)


# ---------------------------------------------------------------------------
# Stubs - ToolPicker has real tools to route between; the v0.4+ work fills
# these in with real implementations.
# ---------------------------------------------------------------------------


class SearchWorkspaceCodeTool:
    """Stub: search the user's own code in the open VS Code workspace."""

    name = "search_workspace_code"
    description = (
        "Search the user's open VS Code workspace for code that uses a specific "
        "API or pattern. Use this when the user asks 'where in my code do I...' "
        "or 'show me how I'm using X' - questions about THEIR code, not the "
        "library's docs."
    )

    async def run(self, *, query: str) -> ToolResult:
        return ToolResult(
            text=(
                f"[search_workspace_code stub] would search the workspace for "
                f"{query!r}. Real implementation lands at v0.4."
            )
        )


class FindInChangelogTool:
    """Stub: find breaking-change notes for a library version."""

    name = "find_in_changelog"
    description = (
        "Look up entries in a library's CHANGELOG or release notes for a specific "
        "version. Use this when the user asks 'what changed in version X', "
        "'is this deprecated in Y', or 'when was Z added'."
    )

    async def run(self, *, library: str, version: str, query: str) -> ToolResult:
        del query
        return ToolResult(
            text=(
                f"[find_in_changelog stub] would fetch the changelog for "
                f"{library}@{version}. Real implementation lands at v0.4."
            )
        )


# ---------------------------------------------------------------------------
# Tool schemas fed to ToolPicker
# ---------------------------------------------------------------------------


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI function-call schemas describing every v0.3 tool.

    ``ToolPicker(FunctionSchemaSource(tool_schemas()), ...)`` is the
    construction we feed to the picker. The ``name`` field is the routing
    key the agent loop uses to dispatch to the right tool.
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
