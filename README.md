# DocChat

> A VS Code extension that answers questions about your project's libraries — using the docs for the exact versions your project pins.

[![status](https://img.shields.io/badge/status-alpha-orange)](#)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![version](https://img.shields.io/badge/version-0.8.0-blue)](./CHANGELOG.md)

**Status:** v0.8 — multi-iteration ReAct over three indexed libraries (React 18.2 + FastAPI 0.95 + Vue 3.4), streaming responses, chunk-level metadata grounding, marketplace-prep manifest. Marketplace publish at v1.0.

---

## The problem

Every AI coding assistant in 2026 — Copilot, Continue, Cursor, Cody — answers library questions against latest-version docs (or against training data, which is worse). If your project pins React 18.2 and the latest is 19.1, asking "how do I create a Suspense boundary?" gets you a 19.x answer using APIs that don't exist in your version.

DocChat fixes this. It parses your project's lockfiles, indexes the docs for the exact versions you pin, and answers only against those docs. If a question can't be answered from the indexed docs at your version, DocChat says so instead of hallucinating.

---

## How it does

| Metric | v0.8 baseline | Corpus |
|---|---|---|
| **Answer accuracy** (LLM-judged, over answered entries) | **0.889** | React 18.2 + FastAPI 0.95 + Vue 3.4 |
| **Version correctness** (substring, over answered entries) | **0.895** | 40-pair, 3 libraries |
| **Refusal rate** (over entries classified out-of-scope) | **1.000** | 8 true oos + retrieval-gated |
| **In-scope answered** | **19 of 32** | corpus-in-scope |
| **p95 latency** | **~12 s** | gpt-4o-mini, multi-iteration ReAct (max 3) |
| **Cost / turn** | **~$0.0003** | 1–3 streamed OpenAI calls per query |

Best accuracy in the project. v0.8's structural changes lifted accuracy from v0.7's 0.882 → 0.889 even with a third indexed library (Vue 3.4) added and the corpus grown from 30 → 40 pairs:

- **Chunk-level metadata in the Qdrant payload.** Indexer now extracts `api_name` (from source filename) and `section_heading` (most recent H2 at chunk start) into the payload. `SearchDocsTool` surfaces both in every retrieval header: `## react@18.2.0 - useState  (useState.md / Reference)`. Gives the LLM an explicit "this chunk is about API X, in section Y, pinned to version Z" signal in every block.
- **Vue 3.4 as the third indexed library.** 10 Composition-API markdown pages from `vuejs/docs`. Corpus extended with 8 Vue in-scope pairs (ref / reactive / computed / watch / lifecycle / provide / defineModel) and 2 cross-framework oos (Angular signal, Svelte rune).
- **Self-critique pass** (constructor-opt-in, OFF by default). Tried, measured, reverted: with critique ON the eval ran `accuracy=0.667 / version=0.895 / p95=23s`; with critique OFF it ran `accuracy=0.824 / version=0.947 / p95=8s`. The critique pass was rewriting well-grounded drafts into worse ones. Kept as opt-in for future prompt-tuning ablations; documented in the v0.8 CHANGELOG.

Streaming (token-by-token via OpenAI `stream=True`) masks the multi-iteration overhead so the first tokens land in <1s even when the full answer takes 12s.

Reproduce with one command:

```powershell
docker compose up -d qdrant
$env:PYTHONPATH = "$PWD;$PWD\sidecar\src"
uv --directory sidecar run python -m evals --corpus ..\evals\corpus.json --output ..\out\eval.json
```

**Engineering iteration the eval drove:**

| Version | accuracy | version | in_scope | what changed |
|---|---|---|---|---|
| v0.4 | 0.688 | 0.800 | 8 / 16 | first measurement, no refusal discipline |
| v0.5 | 0.800 | 1.000 | 7 / 16 | retrieval floor + canonical refusal phrase |
| v0.6 | 0.625 | 0.750 | 8 / 24 | multi-iter ReAct + FastAPI: regressed |
| v0.6.1 | 0.714 | 0.875 | 8 / 24 | changelog regex fix |
| v0.7 | 0.882 | 0.944 | 18 / 24 | per-library floors + version anchoring |
| **v0.8** | **0.889** | **0.895** | **19 / 32** | chunk metadata + Vue 3.4 (3rd library) + critique reverted |

The portfolio narrative is the iteration, not just the final number. Every regression got measured, named, and patched — including a v0.6.1 prompt-softening attempt that was tried and reverted with eval data ([see CHANGELOG](./CHANGELOG.md)).

Reproduce with one command:

```powershell
docker compose up -d qdrant
$env:PYTHONPATH = "$PWD;$PWD\sidecar\src"
uv --directory sidecar run python -m evals --corpus ..\evals\corpus.json --output ..\out\eval.json
```

The corpus + runner + LLM judge live in [`evals/`](./evals/).

---

## Install (v0.8)

The `.vsix` ships as a release artifact on GitHub. Marketplace publish lands at v1.0.

```powershell
# Download docchat-0.8.0.vsix from the GitHub Releases page, then:
code --install-extension docchat-0.8.0.vsix
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

- **v0.8** (current) — chunk-level metadata in Qdrant payload, Vue 3.4 as third indexed library, marketplace-prep manifest, self-critique pass (opt-in)
- **v0.9** — Vue recall investigation (7/8 Vue queries currently floor-pruned), chunk-level filtering by `api_name`, second version of an existing library (FastAPI 0.100)
- **v1.0** — Marketplace publish, auto-install sidecar via bundled venv, 150-pair × 5-library eval corpus, icon

---

## License

MIT. See [LICENSE](./LICENSE).
