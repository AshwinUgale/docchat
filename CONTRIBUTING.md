# Contributing to DocChat

Thanks for your interest! DocChat is a VS Code extension that answers questions about your
project's libraries **at the exact versions your project pins** — a version-aware, agentic
RAG system. It's a two-process design, so there are two stacks to know:

- **`extension/`** — the TypeScript VS Code extension (chat UI, file watching, citation
  jumps). It's a thin shell that spawns and manages the sidecar.
- **`sidecar/`** — the Python FastAPI WebSocket server that runs the actual agent loop
  (retrieval, tools, memory) and holds the model/library connections.
- **`evals/`** — a Q/A corpus + runner that exercises the sidecar end to end.

The two-process split is deliberate: a pure-TS extension can't import the Python libraries
the agent depends on, so the extension is UI and the sidecar is the brain.

## Development setup

Requirements: Node.js + npm, Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and Docker
(for Qdrant).

```bash
# Extension (TypeScript)
cd extension && npm install && cd ..

# Sidecar (Python) — uv manages the environment; do NOT use pip directly
cd sidecar && uv sync && cd ..

# Vector store (needed once indexing runs)
docker compose up -d qdrant
```

Run it: open the `extension/` folder in VS Code and press **F5** to launch an Extension
Development Host. For sidecar-only debugging: `cd sidecar && uv run python -m docchat_sidecar`.

## Before you open a PR — run the full check

The combined pipeline runs both halves (fail-fast):

```powershell
.\scripts\check.ps1
```

It runs, for the **sidecar**: `uv run ruff check . --fix`, `uv run ruff format .`,
`uv run mypy src` (strict), `uv run pytest`; and for the **extension**: `tsc` type-check +
lint. If you're not on PowerShell, run those commands directly from `sidecar/` and
`extension/`.

## Good first contributions

- **Sidecar (Python):** a new agent tool in the ReAct loop, an indexer/chunking
  improvement, a lock-file parser for another ecosystem (so more projects get version-aware
  answers), or better citation metadata.
- **Extension (TypeScript):** chat-UI polish, citation-jump handling, settings/config, or
  status/error surfacing.
- **Evals:** add Q/A pairs to the corpus — especially version-sensitive questions where the
  right answer differs across pinned library versions. High-value, low-friction.
- **Docs & examples:** a walkthrough, a demo project, clearer setup notes.

Browse issues labeled **`good first issue`** and **`help wanted`**. For anything that spans
both processes or changes the extension⇄sidecar protocol, open an issue first so we agree on
the shape.

## Guidelines

- **Python:** 3.11+, `src/docchat_sidecar/` layout, ruff-formatted, **mypy strict**, type
  hints everywhere, docstrings on public functions/classes. Use `uv` — never invoke `pip`
  directly.
- **TypeScript:** strict TS, no bundler (compile with `tsc`). JSDoc on exported APIs.
- **Tests:** `pytest` for the sidecar; keep the eval corpus runnable.
- **Never commit secrets.** API keys live in a local, gitignored `.env` (read via
  `python-dotenv`); the library never auto-loads it for you.
- **Keep the two-process boundary clean** — the extension is UI, the sidecar is the agent.
- **Update `CHANGELOG.md`** for user-visible changes.

## Pull request checklist

- [ ] `.\scripts\check.ps1` passes (ruff + mypy strict + pytest + tsc), or the equivalent
      per-stack commands
- [ ] new behavior has a test (sidecar) and/or an eval case
- [ ] no secrets committed; `.env` stays local
- [ ] `CHANGELOG.md` updated (for user-visible changes)
- [ ] the PR description says *what* and *why* in a sentence or two

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating you
agree to uphold it — be kind, be constructive.
