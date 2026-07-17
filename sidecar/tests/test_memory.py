"""Tests for the workspace-scoped Mneme wrapper."""

from __future__ import annotations

from pathlib import Path

from mneme import HashEmbedder, InMemoryBackend, MemoryManager

from docchat_sidecar.memory import WorkspaceMemory, workspace_namespace


def _manager_with_in_memory_backend(agent_id: str) -> MemoryManager:
    embedder = HashEmbedder(dimensions=32)
    backend = InMemoryBackend()
    return MemoryManager(agent_id=agent_id, backend=backend, embedder=embedder)


def test_workspace_namespace_is_stable_for_same_path(tmp_path: Path) -> None:
    a = workspace_namespace(tmp_path)
    b = workspace_namespace(tmp_path)
    assert a == b


def test_workspace_namespace_differs_for_different_paths(tmp_path: Path) -> None:
    p1 = tmp_path / "project_a"
    p2 = tmp_path / "project_b"
    p1.mkdir()
    p2.mkdir()
    assert workspace_namespace(p1) != workspace_namespace(p2)


def test_workspace_namespace_handles_none() -> None:
    ns = workspace_namespace(None)
    assert ns == "docchat_workspace_none"


def test_workspace_namespace_prefix() -> None:
    ns = workspace_namespace("/some/path")
    assert ns.startswith("docchat_workspace_")


def test_record_qa_persists_to_episodic() -> None:
    mgr = _manager_with_in_memory_backend("test_agent")
    wm = WorkspaceMemory(manager=mgr)
    record = wm.record_qa(
        query="how do I use useState?",
        answer="Call useState(initial) and destructure [value, setter].",
        citations=["[react@18.2.0:useState.md]"],
    )
    assert "Q: how do I use useState" in record.content
    assert "A: Call useState" in record.content
    assert record.metadata["citations"] == ["[react@18.2.0:useState.md]"]
    assert mgr.episodic.count() == 1


def test_clear_wipes_recorded_turns() -> None:
    mgr = _manager_with_in_memory_backend("test_agent")
    wm = WorkspaceMemory(manager=mgr)
    wm.record_qa(query="q1", answer="a1", citations=[])
    wm.record_qa(query="q2", answer="a2", citations=[])
    assert mgr.episodic.count() == 2
    wm.clear()
    assert mgr.episodic.count() == 0
    # A post-clear retrieval sees nothing from before the reset.
    assert wm.retrieve_relevant("q1", k=3) == []


def test_retrieve_relevant_surfaces_past_turns() -> None:
    mgr = _manager_with_in_memory_backend("test_agent")
    wm = WorkspaceMemory(manager=mgr)
    wm.record_qa(
        query="how do I use useState?",
        answer="Call useState(initial).",
        citations=[],
    )
    wm.record_qa(
        query="what does useEffect do?",
        answer="useEffect runs side effects after render.",
        citations=[],
    )
    results = wm.retrieve_relevant("how do I use useState?", k=2)
    assert len(results) <= 2
    # HashEmbedder is deterministic, so the exact-match query lands its
    # original record at the top.
    assert any("useState" in r for r in results)


def test_retrieve_relevant_empty_when_no_memories() -> None:
    mgr = _manager_with_in_memory_backend("test_agent")
    wm = WorkspaceMemory(manager=mgr)
    results = wm.retrieve_relevant("anything", k=3)
    assert results == []
