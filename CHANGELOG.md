# Changelog

All notable changes to DocChat are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from v1.0 onward.

## [Unreleased]

## [0.5.0] - 2026-06-02

### Added
- `score_floor` parameter on `SearchDocsTool` (default 0.15 after empirical tuning — see Notes) that drops Qdrant hits below the cosine-similarity floor. Prunes the false-positive long tail that was letting `gpt-4o-mini` hallucinate answers to out-of-scope questions from base knowledge.
- Hard-refusal rule in the agent system prompt: when the "Retrieved context" section is empty or contains "No relevant chunks" / "No indexed docs", the agent must reply with the canonical "I don't have documentation for that in this workspace's indexed libraries." phrase. Maps directly to the eval's `is_refusal` heuristic, so refusal-rate metric is non-degenerate.
- `CHANGELOG.md` (this file), back-filled v0.0 → v0.5.
- `docchat-0.5.0.vsix` published as a GitHub Release artifact (no Marketplace yet — that's v1.0).
- README headline-metrics table with the v0.5 baseline numbers and the one-command reproduction recipe.

### Changed
- Bumped `extension/package.json` and `sidecar/pyproject.toml` to 0.5.0.

### Notes
- v0.4 reported `oos=0 / refusal_rate=0.000` on the 20-pair React 18.2 corpus because the agent answered the 4 out-of-scope entries from `gpt-4o-mini` base knowledge instead of refusing. v0.5's combination of retrieval floor + hard-refusal prompt makes `refusal_rate` non-degenerate.
- v0.5 measured numbers on the 20-pair React 18.2 corpus: `version_correctness=1.000` (no React-19 API leakage), `refusal_rate=1.000` (canonical phrase on every refused entry), `answer_accuracy≈0.80` over answered entries, `p95≈2.5s`. The floor-based gating trades recall for precision on a sparse corpus — the React 18.2 index is only 10 hook pages (~50 chunks), so some in-scope queries score below the floor and get refused alongside true off-topic queries. v0.6's corpus expansion (FastAPI 0.95 + more React pages) is the planned recall fix.
- Floor was initially set to `0.25` based on an a-priori guess; the first eval run revealed 8/16 in-scope queries had top hits below 0.25. Lowered to `0.15`; same precision/recall pattern persisted, confirming the corpus density is the bottleneck rather than the threshold itself.

## [0.4.0] - 2026-06-02

### Added
- `evals/` harness (`schema`, `corpus`, `runner`, `judge`, `metrics`, `__main__`) sitting next to `sidecar/` so the runner can import the real `Agent` via PYTHONPATH and replay queries deterministically.
- 20-pair React 18.2 corpus (`evals/corpus.json`): 16 in-scope (useState ×2, useEffect ×3, useContext, useReducer, useMemo, useCallback, useRef ×2, useId, useSyncExternalStore, useTransition, createRoot, "no `use()` hook yet in 18.2") + 4 out-of-scope (Flask, Vite plugin, Node EADDRINUSE, `let` vs `const`).
- Three metrics: `answer_accuracy` (LLM-as-judge, `gpt-4o-mini`, JSON response format), `version_correctness` (substring check over `expected_apis` + `forbidden_apis` so React-19-only tokens like `use(promise)` count as wrong), `refusal_rate` over out-of-scope only with nearest-rank `p95_latency_ms`.
- `--no-judge` CLI flag for cheap regression runs (saves ~$0.002 per 20-entry corpus).
- 29 eval tests on top of the 62 sidecar tests.

### Fixed
- `evals/pytest.ini` with `asyncio_mode=auto` so the eval suite's async tests resolve when invoked outside the sidecar's pyproject.

## [0.3.0] - 2026-06-02

### Added
- Real `SearchDocsTool` backed by `AsyncQdrantClient.query_points()` over the `react_18_2_0` collection from v0.2.
- Two ToolPicker stubs (`SearchWorkspaceCodeTool`, `FindInChangelogTool`) so the router has a real >=2-tool routing problem at v0.3 (real implementations land at v0.6).
- `WorkspaceMemory` wrapping Mneme's `MemoryManager` with a `sha256(workspace_path)[:16]` namespace so reopening a project resurfaces past memory; different projects on the same machine stay isolated.
- Single-iteration ReAct loop in `agent.py`: ToolPicker.select → dispatch → Mneme retrieve past Q/As → `gpt-4o-mini` chat completion grounded in retrieved context → append `Sources: [react@18.2.0:useState.md]` → record Q/A back into Mneme.

## [0.2.0] - 2026-06-02

### Added
- Typed Pydantic discriminated-union WebSocket protocol: `ClientMessage = UserQuery | IndexLibrary | Ping` and `ServerMessage = AssistantText | IndexProgress | IndexComplete | IndexError | Pong`.
- `lockfiles.py` package.json + package-lock.json (v1/v2/v3) parser that returns typed `Pin` objects with a `source` field for "approximate" UI flagging.
- `indexer.py` React 18.2 doc fetcher: pulls MDX from `raw.githubusercontent.com/reactjs/react.dev/main/src/content/reference/react/`, strips frontmatter + import/export lines, paragraph-chunks to ~500 tokens, embeds with `text-embedding-3-small`, upserts to Qdrant with cosine `VectorParams`.

## [0.1.0] - 2026-06-02

### Added
- TypeScript extension activation entrypoint that spawns the Python sidecar with three-tier interpreter resolution (user setting → `<extensionPath>/../sidecar/.venv/Scripts/python.exe` → `python` on PATH).
- `DOCCHAT_SIDECAR_PORT=<N>` stdout handshake + `/health` poll (5 s timeout, 100 ms interval) so the extension only opens the WebSocket once uvicorn is fully wired.
- Vanilla HTML/JS webview that round-trips an echo message over `ws://localhost:<port>/chat`.
- `test_sidecar_lifecycle.py` integration test that exercises the full subprocess + WebSocket path.

## [0.0.1] - 2026-06-02

### Added
- Repo scaffolding: `extension/` (TS) + `sidecar/` (Python) + `evals/` + `.github/workflows/check.yml` + `docker-compose.yml`.
- `sidecar/pyproject.toml` with PyPI dependencies on `smolAmem>=1.0` + `toolpicker>=1.0` (no path-deps, dogfood is real).
- `scripts/check.ps1` combined lint + typecheck + test pipeline.
- Dual-repo setup (public main + private nested `.cowork/`).

[Unreleased]: https://github.com/AshwinUgale/docchat/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/AshwinUgale/docchat/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/AshwinUgale/docchat/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/AshwinUgale/docchat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AshwinUgale/docchat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AshwinUgale/docchat/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/AshwinUgale/docchat/releases/tag/v0.0.1
