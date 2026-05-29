# DocChat

> A VS Code extension that answers questions about your project's libraries — using the docs for the exact versions your project pins.

[![status](https://img.shields.io/badge/status-alpha-orange)](#)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**Status:** v0.0 — scaffolding. Walking skeleton lands at v0.1.

---

## The problem

Every AI coding assistant in 2026 — Copilot, Continue, Cursor, Cody — answers library questions against latest-version docs (or against training data, which is worse). If your project pins React 18.2 and the latest is 19.1, asking "how do I create a Suspense boundary?" gets you a 19.x answer using APIs that don't exist in your version.

DocChat fixes this. It parses your project's lockfiles, indexes the docs for the exact versions you pin, and answers only against those docs. If a question can't be answered from the indexed docs at your version, DocChat says so instead of hallucinating.

---

## Architecture

Two processes:

- **VS Code extension (TypeScript).** Chat panel UI, file watching, citation jumps.
- **Python sidecar.** Agent loop, vector store, indexing, agent memory, tool routing.

The extension spawns the sidecar on activation and communicates over local WebSocket. The sidecar imports two PyPI libraries directly:

- [`smolAmem`](https://pypi.org/project/smolAmem/) (import name `mneme`) — multi-tier memory for the agent.
- [`toolpicker`](https://pypi.org/project/toolpicker/) — hybrid lexical + semantic tool routing.

Both are open-source libraries with docs sites and reproducible benchmarks. DocChat dogfoods them.

---

## Install

_Coming at v0.5 (the `.vsix` build). Marketplace publish at v1.0._

---

## Development

```powershell
# Install dependencies
cd extension && npm install && cd ..
cd sidecar && uv sync && cd ..

# Run the full check pipeline
.\scripts\check.ps1

# Bring up Qdrant (needed once you hit v0.2 doc indexing)
docker compose up -d qdrant
```

---

## License

MIT. See [LICENSE](./LICENSE).
