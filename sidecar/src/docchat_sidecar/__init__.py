"""DocChat sidecar - FastAPI WebSocket agent loop.

Spawned by the VS Code extension on activation; killed on deactivation.
Imports `mneme` (from PyPI: smolAmem) for workspace memory and `toolpicker`
(from PyPI: toolpicker) for tool selection - the two portfolio libraries
this product dogfoods.

The public surface is the WebSocket protocol, not Python imports. The
sidecar runs as a server; nothing imports from this package directly
(except the test suite).
"""

from __future__ import annotations

__version__ = "0.0.1"
__all__ = ["__version__"]
