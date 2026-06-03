# DocChat

> A VS Code extension that answers questions about your project's libraries — using the docs for the exact versions your project pins.

[![status](https://img.shields.io/badge/status-alpha-orange)](#)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![version](https://img.shields.io/badge/version-0.5.0-blue)](./CHANGELOG.md)

**Status:** v0.5 — packaged `.vsix`, eval harness with published numbers. Marketplace publish at v1.0.

---

## The problem

Every AI coding assistant in 2026 — Copilot, Continue, Cursor, Cody — answers library questions against latest-version docs (or against training data, which is worse). If your project pins React 18.2 and the latest is 19.1, asking "how do I create a Suspense boundary?" gets you a 19.x answer using APIs that don't exist in your version.

DocChat fixes this. It parses your project's lockfiles, indexes the docs for the exact versions you pin, and answers only against those docs. If a question can't be answered from the indexed docs at your version, DocChat says so instead of hallucinating.

---

## How it does

| Metric | v0.5 baseline | Corpus |
|---|---|---|
| **Version correctness** (substring, over answered entries) | **1.000** | 20-pair React 18.2 |
| **Refusal rate** (over entries classified out-of-scope) | **1.000** | 4 true oos + retrieval-gated refusals |
| **Answer accuracy** (LLM-judged, over answered entries) | **~0.80** | judged-in-scope only |
| **p95 latency** | **~2.5 s** | gpt-4o-mini, single-iteration ReAct |
| **Cost / turn** | **~$0.0001** | embedding + chat |

When DocChat answers, it answers correctly for the pinned version — `version=1.000` says zero leakage of React-19-only APIs (`use(promise)`, `ReactDOM.render`, etc.) into React 18.2 answers. That's the exact failure mode latest-version-only assistants have.

When DocChat doesn't answer, it refuses cleanly with the canonical "I don't have documentation for that" phrase — `refusal=1.000` on the classified-oos bucket. The retrieval similarity floor (`score_floor=0.15` on `SearchDocsTool`) drops weak Qdrant hits below the cosine threshold so the agent never tries to dress up an off-topic retrieval as a real answer.

**Honest caveat:** the v0.5 floor-based gating is conservative on a sparse corpus. The current React 18.2 index is 10 hook pages (~50 chunks); on that surface, some legitimate in-scope queries score below the floor and get refused alongside the 4 true out-of-scope entries. v0.6 expands the corpus (FastAPI 0.95 + more React pages) to widen the precision/recall band.

Reproduce with one command:

```powershell
docker compose up -d qdrant
$env:PYTHONPATH = "$PWD;$PWD\sidecar\src"
uv --directory sidecar run python -m evals --corpus ..\evals\corpus.json --output ..\out\eval.json
```

The corpus + runner + LLM judge live in [`evals/`](./evals/).

---

## Install (v0.5)

The `.vsix` ships as a release artifact on GitHub. Marketplace publish lands at v1.0.

```powershell
# Download docchat-0.5.0.vsix from the GitHub Releases page, then:
code --install-extension docchat-0.5.0.vsix
```

Python 3.11+ on PATH is required (the extension spawns a sidecar). Docker is required for Qdrant (vector store):

```powershell
docker compose up -d qdrant
```

---

## Usage

1. Open a JS / TS project that has a `package.json` + `package-lock.json`.
2. Run **DocChat: Open Chat Panel** from the command palette.
3. Click **Index react 18.2.0** (or whatever pin DocChat detected).
4. Ask: *"how do I use useEffect with a cleanup function?"*

You'll get back an answer grounded in the exact 18.2 docs, with citations of the form `[react@18.2.0:useEffect.md]` you can click to jump to the source.

Ask something out of scope — *"how do I configure CORS in Flask?"* — and DocChat refuses cleanly instead of inventing an answer.

---

## Architecture

```
                      VS Code window
                      ┌─────────────┐
                      │  Extension  │  TypeScript
                      │  (chat UI)  │  — webview, file watching, citation jumps
                      └──────┬──────┘
                             │ WebSocket on localhost:<random>
                             │ (spawned on activate, killed on deactivate)
                      ┌──────┴──────┐
                      │   Sidecar   │  Python
                      │  (agent)    │  — FastAPI WebSocket endpoint
                      └──┬───┬───┬──┘
                         │   │   │
                ┌────────┘   │   └────────┐
                │            │            │
         ┌──────┴──────┐ ┌───┴────┐ ┌─────┴─────┐
         │   Qdrant    │ │ Mneme  │ │ ToolPicker│
         │  (Docker)   │ │ (PyPI) │ │  (PyPI)   │
         └─────────────┘ └────────┘ └───────────┘
```

The two-process design is what makes the "actually uses real Python infra" claim hold up: a pure-TS extension can't import Python libraries. The sidecar is where the agent loop runs, where Qdrant is queried, where Mneme persists per-workspace memory, and where ToolPicker routes between the agent's tools.

DocChat dogfoods two open-source libraries the same author shipped:

- **[`smolAmem`](https://pypi.org/project/smolAmem/)** (import `mneme`) — multi-tier memory for the agent (working / episodic / semantic, pluggable backends, TTL + decay-based forgetting). [docs](https://ashwinugale.github.io/mneme/)
- **[`toolpicker`](https://pypi.org/project/toolpicker/)** — hybrid lexical + semantic tool selection with optional intent-classifier reranking. [docs](https://ashwinugale.github.io/toolpicker/)

Both have docs sites, reproducible benchmarks, and 1.0 releases on PyPI.

---

## Development

```powershell
# First-time setup
cd extension && npm install && cd ..
cd sidecar && uv sync && cd ..

# Full check pipeline (TS + Python + evals in one shot)
.\scripts\check.ps1

# Bring up Qdrant (needed for v0.2+ doc indexing)
docker compose up -d qdrant

# Run the eval harness
$env:PYTHONPATH = "$PWD;$PWD\sidecar\src"
uv --directory sidecar run python -m evals --corpus ..\evals\corpus.json --output ..\out\eval.json
```

To run the extension in a dev host: open `extension/` in VS Code and press **F5**.

---

## Roadmap

- **v0.5** (current) — `.vsix` artifact, retrieval similarity floor, hard-refusal prompt, eval harness with headline numbers
- **v0.6** — multi-iteration ReAct, real `search_workspace_code` and `find_in_changelog` tools, FastAPI 0.95 corpus
- **v1.0** — Marketplace publish, settings UI, self-critique pass, 150-pair × 5-library eval corpus

---

## License

MIT. See [LICENSE](./LICENSE).
