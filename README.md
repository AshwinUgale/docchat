# DocChat

> A VS Code extension that answers questions about your project's libraries — using the docs for the exact versions your project pins.

[![status](https://img.shields.io/badge/status-production--ready-brightgreen)](#)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![version](https://img.shields.io/badge/version-1.0.0-blue)](./CHANGELOG.md)

**Status:** v1.0 — production release. Multi-iteration ReAct over three indexed libraries (React 18.2, FastAPI at 0.95 AND 0.100 side-by-side, Vue 3.4), per-library git-ref resolution, lockfile-pinned retrieval routing for both Node and Python projects, pre-retrieval topic classifier, streaming responses, click-to-open citations, settings drawer.

---

## The problem

Every AI coding assistant in 2026 — Copilot, Continue, Cursor, Cody — answers library questions against latest-version docs (or against training data, which is worse). If your project pins React 18.2 and the latest is 19.1, asking "how do I create a Suspense boundary?" gets you a 19.x answer using APIs that don't exist in your version.

DocChat fixes this. It parses your project's lockfiles, indexes the docs for the exact versions you pin, and answers only against those docs. If a question can't be answered from the indexed docs at your version, DocChat says so instead of hallucinating.

---

## How it does

Measured on the 48-pair corpus (React 18.2, FastAPI 0.95 **and** 0.100 side-by-side, Vue 3.4), `gpt-4o-mini`, scope split on the ground-truth `out_of_scope` label (40 in-scope / 8 out-of-scope):

| Metric | v1.0 (corrected metric) | Notes |
|---|---|---|
| **Answer accuracy** (LLM-judged, over 40 in-scope) | **0.675** | 27 / 40 |
| **Version correctness** (over 40 in-scope) | **0.850** | expected API present, wrong-version API absent |
| **Refusal rate** (over 8 out-of-scope) | **0.875** | 7 / 8 correctly refused |
| **Over-refusal rate** (in-scope wrongly refused) | **0.025** | 1 / 40 |
| **p95 latency** | **~15 s** | multi-iteration ReAct (max 3) |
| **Cost / turn** | **~$0.0003** | 1–3 streamed OpenAI calls per query |

> **The eval caught a bug in itself.** Earlier releases reported `refusal_rate = 1.000` — but the metric derived in/out-of-scope from the agent's *behaviour* (did it refuse?) rather than the corpus label. That was circular: an out-of-scope question the agent *answered* got silently reclassified as in-scope and never counted against the refusal metric, and an in-scope question it *over-refused* was counted as a success. Splitting on the ground-truth `out_of_scope` label dropped refusal to an honest **0.875** and immediately surfaced the failure the old number was hiding — a cross-framework question ("call my FastAPI backend from a React `useEffect`") the agent answered instead of refusing. `over-refusal rate` is the companion signal the old split also hid. The corrected harness (label-based scope + per-entry memory isolation) lives in [`evals/`](./evals/).

Historical iteration is below; the numbers in that table are as-reported at each version, using the then-current (pre-fix) metric. v0.9's structural changes:

- **`pinned_libraries` plumbing through `Agent.answer()` + eval runner.** v0.9's per-query score logging surfaced a bug that had been hidden since v0.6: the LLM was inferring versions from question text and hitting unindexed collections (`fastapi_0_95_2` instead of `fastapi_0_95_0`, `vue_3_0_0` instead of `vue_3_4_0`). The runner now passes the corpus entry's library@version as a lockfile pin; the agent's system prompt surfaces it and `_dispatch` overrides any wrong version the LLM guesses. 16 previously-refused queries now answer.
- **Vue floor recalibrated from 0.10 → 0.05.** With the routing fix in place, Vue queries actually hit `vue_3_4_0` and score 0.40–0.55 (well above any reasonable floor); the original 7/8 over-refusal was 100% the routing bug, not the floor.
- **`api_name` retrieval filter on `SearchDocsTool`.** New optional kwarg the LLM can pass to constrain retrieval to chunks tagged with a specific API (e.g. `api_name="useState"`). Tightens precision when the question names an API explicitly.
- **Per-query score logging.** Every `search_docs` call emits `top-scores=[...]` at INFO so retrieval failures self-document. This is what surfaced the routing bug above.

The trade-off: accuracy ticked down from v0.8's 0.889 → 0.781 because the answered set doubled and now includes harder Vue + FastAPI queries that previously refused. In absolute terms that's ~27 correct answers vs v0.8's ~17 — **more correct answers, on a larger surface, with version-correctness preserved**.

- **Chunk-level metadata in the Qdrant payload.** Indexer now extracts `api_name` (from source filename) and `section_heading` (most recent H2 at chunk start) into the payload. `SearchDocsTool` surfaces both in every retrieval header: `## react@18.2.0 - useState  (useState.md / Reference)`. Gives the LLM an explicit "this chunk is about API X, in section Y, pinned to version Z" signal in every block.
- **Vue 3.4 as the third indexed library.** 10 Composition-API markdown pages from `vuejs/docs`. Corpus extended with 8 Vue in-scope pairs (ref / reactive / computed / watch / lifecycle / provide / defineModel) and 2 cross-framework oos (Angular signal, Svelte rune).
- **Self-critique pass** (constructor-opt-in, OFF by default). Tried, measured, reverted: with critique ON the eval ran `accuracy=0.667 / version=0.895 / p95=23s`; with critique OFF it ran `accuracy=0.824 / version=0.947 / p95=8s`. The critique pass was rewriting well-grounded drafts into worse ones. Kept as opt-in for future prompt-tuning ablations; documented in the v0.8 CHANGELOG.

Streaming (token-by-token via OpenAI `stream=True`) masks the multi-iteration overhead so the first tokens land in <1s even when the full answer takes 12s.

**Engineering iteration the eval drove:**

| Version | accuracy | version | in_scope | what changed |
|---|---|---|---|---|
| v0.4 | 0.688 | 0.800 | 8 / 16 | first measurement, no refusal discipline |
| v0.5 | 0.800 | 1.000 | 7 / 16 | retrieval floor + canonical refusal phrase |
| v0.6 | 0.625 | 0.750 | 8 / 24 | multi-iter ReAct + FastAPI: regressed |
| v0.6.1 | 0.714 | 0.875 | 8 / 24 | changelog regex fix |
| v0.7 | 0.882 | 0.944 | 18 / 24 | per-library floors + version anchoring |
| v0.8 | 0.889 | 0.895 | 19 / 32 | chunk metadata + Vue 3.4 (3rd library) + critique reverted |
| v0.9 | 0.781 | 0.943 | 35 / 40 | pinned_libraries fix (routing bug) + api_name filter + score logging |
| v1.0 | 0.675 | 0.854 | 41 / 48 | git-ref resolution + FastAPI 0.95 & 0.100 + Python lockfiles + topic classifier |
| **v1.1** (unreleased) | **0.675** | **0.850** | **40 / 8** | **metric fix**: scope split on the corpus label (not agent behaviour) + memory isolation → honest `refusal_rate` 1.000 → **0.875**, new `over-refusal 0.025` |

The portfolio narrative is the iteration, not just the final number. Every regression got measured, named, and patched — including a v0.6.1 prompt-softening attempt reverted with eval data, and the v1.1 discovery that the harness itself was scoring refusals wrong ([see CHANGELOG](./CHANGELOG.md)). Rows through v1.0 used the pre-fix (behaviour-derived) scope split; `in_scope` there counts answered entries, whereas v1.1's `40 / 8` is the corpus's true in-scope / out-of-scope split.

Reproduce with one command:

```powershell
docker compose up -d qdrant
$env:PYTHONPATH = "$PWD;$PWD\sidecar\src"
uv --directory sidecar run python -m evals --corpus ..\evals\corpus.json --output ..\out\eval.json
```

The corpus + runner + LLM judge live in [`evals/`](./evals/) — including `--split` (held-out calibration/test) and `--score-floor` / `--floor` overrides so retrieval thresholds can be calibrated off the reported set.

---

## Install

Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=AshwinUgale.docchat) — search "DocChat" in the Extensions panel, or:

```powershell
code --install-extension AshwinUgale.docchat
```

After install, run **`DocChat: Set up sidecar`** from the command palette once. The setup command installs the Python sidecar into `~/.docchat/sidecar/` via `uv sync` (~200MB download, takes 10–30 seconds). It's idempotent — re-run any time to repair a broken install.

**Prerequisites the setup checks for:**

- **Python 3.11+** on PATH ([python.org/downloads](https://www.python.org/downloads/))
- **uv** package manager on PATH ([docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/))
- **Docker** for the Qdrant vector store (`docker compose up -d qdrant` after the first install, or set `QDRANT_URL` if you run Qdrant elsewhere)
- **`OPENAI_API_KEY`** in your environment (or a `.env` file in the workspace root)

The first time you open the panel, DocChat reads your project's lockfile (`package.json`, `pyproject.toml`, or `requirements.txt`) and routes queries to the right indexed version per question.

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
                             │ WebSocket on 127.0.0.1:<random>, per-spawn token
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

- **v1.0** (current) — production release. Per-library git-ref resolution; FastAPI 0.95 + 0.100 side-by-side; `pyproject.toml` + `requirements.txt` lockfile parsers; pre-retrieval topic classifier closes the general-programming-question leak; production-stable classifier.
- **v1.0.x** — auto-install sidecar (bundled venv via `uv` shipped with the `.vsix`); Marketplace publish (publisher account + icon + `vsce publish`)
- **v1.1+** — 150-pair × 5-library corpus expansion; hybrid sparse + dense retrieval if any single library's recall plateaus; more lockfile formats (Cargo.toml, go.mod, Gemfile)

---

## Related projects

Part of a 4-project portfolio of production AI engineering:

- **[mneme](https://github.com/AshwinUgale/mneme)** ([PyPI as smolAmem](https://pypi.org/project/smolAmem/)) — DocChat's per-workspace memory layer. Multi-tier memory with TTL + decay-based forgetting.
- **[toolpicker](https://github.com/AshwinUgale/toolpicker)** ([PyPI](https://pypi.org/project/toolpicker/)) — DocChat's agent tool-routing layer. Hybrid BM25 + embeddings + optional intent classifier.
- **[docchat-server](https://github.com/AshwinUgale/docchat-mcp)** ([PyPI](https://pypi.org/project/docchat-server/)) — Sibling project. The same version-pinned retrieval as an MCP server (for Claude Code / Cursor users who want the tool surface without the chat panel).

## License

MIT. See [LICENSE](./LICENSE).
