"""Workspace-scoped Mneme integration.

Each VS Code workspace gets its own namespace in Mneme - past Q/A pairs
from one project don't leak into another. The namespace is a stable hash
of the workspace path so reopening the same project resurfaces past memory.

Why this exists in DocChat (not in Mneme): Mneme is a general-purpose
multi-tier memory library; the "workspace path hash as agent_id" policy
is a DocChat-specific decision. Mneme stays framework-agnostic.

v0.3 wires episodic memory only: every successful Q/A pair is recorded
to the episodic tier with the query, the answer text, and the cited
sources. Semantic consolidation (the LLM-judge pass that promotes
episodic memories into facts) is opt-in via Mneme's scheduler and we
DON'T turn it on for the v0.3 demo - it costs LLM calls per
consolidation cycle and adds little to a single-session demo.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from mneme import InMemoryBackend, MemoryManager, OpenAIEmbeddings
from mneme.types import EpisodicMemory

__all__ = ["WorkspaceMemory", "build_memory", "workspace_namespace"]

logger = logging.getLogger(__name__)


def workspace_namespace(workspace_path: str | Path | None) -> str:
    """Stable per-workspace agent_id for Mneme.

    Hash-derived so two users on the same machine with different project
    paths get distinct namespaces, and reopening the same project lands
    on the same memory store.

    ``None`` (no workspace open) maps to a single shared scratch namespace.
    """
    if workspace_path is None:
        return "docchat_workspace_none"
    normalised = str(Path(workspace_path).expanduser().resolve()).lower()
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]
    return f"docchat_workspace_{digest}"


def build_memory(
    *,
    workspace_path: str | Path | None,
    qdrant_url: str | None = None,
) -> WorkspaceMemory:
    """Construct a ``WorkspaceMemory`` for the given workspace.

    v0.3 uses ``InMemoryBackend`` unconditionally (per ADR-006: memory is
    per-session at v0.3, Qdrant-backed persistence flips on at v0.4 along
    with the schema-name convention for the Mneme collection). The
    ``qdrant_url`` kwarg is accepted for forward-compatibility but ignored.
    """
    del qdrant_url  # v0.4 will wire this through to QdrantBackend(client=..., dimensions=...)
    embedder = OpenAIEmbeddings()
    backend = InMemoryBackend()
    manager = MemoryManager(
        agent_id=workspace_namespace(workspace_path),
        backend=backend,
        embedder=embedder,
    )
    return WorkspaceMemory(manager=manager)


@dataclass(kw_only=True)
class WorkspaceMemory:
    """Thin facade over ``MemoryManager`` for DocChat's recurring patterns.

    The bare Mneme API is correct but generic; DocChat always wants to
    (a) save a Q/A pair as one episodic record with the answer + sources
    in metadata, and (b) retrieve top-k past pairs to surface in the next
    turn's prompt. This wrapper keeps those call sites short.
    """

    manager: MemoryManager

    def record_qa(
        self,
        *,
        query: str,
        answer: str,
        citations: list[str],
    ) -> EpisodicMemory:
        """Save one user-question + assistant-answer turn as an episodic memory."""
        content = f"Q: {query}\nA: {answer}"
        metadata = {
            "query": query,
            "answer": answer,
            "citations": citations,
        }
        return self.manager.episodic.add(content, metadata=metadata)

    def retrieve_relevant(self, query: str, *, k: int = 3) -> list[str]:
        """Return up to ``k`` past Q/A snippets relevant to the current query.

        Returns just the text content for prompt-injection; callers that
        want the full record (with citations metadata) should use
        ``manager.retrieve()`` directly.
        """
        try:
            results = self.manager.retrieve(query, k=k)
        except Exception as exc:  # pragma: no cover - backend failures shouldn't kill the turn
            logger.warning("Mneme retrieve failed: %s", exc)
            return []
        return [r.record.content for r in results if hasattr(r.record, "content")]
