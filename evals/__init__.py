"""DocChat eval harness.

Replays a labelled Q/A corpus through the agent and reports
``answer_accuracy``, ``version_correctness``, ``refusal_rate``, and
latency. Not part of the sidecar package - this is repo-shaped tooling
that runs from a clone.
"""

from __future__ import annotations

__all__: list[str] = []
