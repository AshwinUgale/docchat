# Changelog

All notable changes to DocChat are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from v1.0 onward.

## [Unreleased]

## [0.7.0] - 2026-06-02

### Added
- **Per-collection score floors** on `SearchDocsTool`. New `floors_by_library: dict[str, float] | None` kwarg with built-in default `fastapi=0.10` (vs the global 0.15 that works for React). FastAPI tutorial pages dilute cosine to 0.08–0.12 for in-scope queries; the per-library floor recovers ~10 in-scope answers that v0.6 over-refused.
- **Chunk-metadata version grounding.** `SearchDocsTool.run` now prefixes each retrieved chunk with `## library@version - source` so the LLM has an explicit version anchor in every block. New HARD RULE #4 in the agent system prompt: "Any API you mention must appear in the retrieved chunks for the user's pinned version." Closes the cross-version leak path that capped `version_correctness` at 0.875 through v0.6.1.
- **Streaming protocol** — three new server-message variants in `protocol.py`:
  - `AssistantTextDelta { text, chunk_index }` — one streaming token chunk (monotonic index per query).
  - `AssistantStreamFinal { citations, tool_used, iterations }` — terminator carrying structured citations + agent telemetry.
  - `CitationRef { library, version, source, source_url? }` — wire-shape mirror of internal `Citation`, with `source_url` reserved for v0.7.1's click-to-open.
- **`SettingsUpdate`** client-message — runtime knobs for `chat_model`, `score_floor`, `max_iterations`. Stored in a module-local dict on the sidecar; agent reads on every query. No respawn needed to change a setting.
- **`Agent.answer_stream(query)`** new async generator method that uses `chat.completions.create(stream=True)` and accumulates partial tool_call chunks via the OpenAI streaming protocol. `Agent.answer()` is preserved unchanged so the eval harness keeps producing comparable numbers across milestones.
- **`__main__.py`** routes `user_query` through `agent.answer_stream()` and emits each event over the WebSocket.
- **Webview** handles `assistant_text_delta` (accumulate into the current bot bubble) and `assistant_stream_final` (footer with `iterations` + `tool_used` + citation count).
- 6 new tests: `test_search_docs_fastapi_default_floor_is_lower_than_react`, `test_search_docs_floors_by_library_override_kwarg`, `test_search_docs_chunk_text_carries_library_and_version_prefix` (tools), `test_assistant_text_delta_round_trip`, `test_assistant_stream_final_round_trip`, `test_settings_update_round_trip_partial`, `test_settings_update_round_trip_full` (protocol), `test_agent_answer_stream_yields_deltas_and_final` (agent).

### Changed
- `extension/package.json` + `sidecar/pyproject.toml` bumped to `0.7.0`.
- `extension/webview/index.html` header reads `v0.7`.
- Lifecycle test accepts either `assistant_text` or `assistant_text_delta` as the first reply (depends on whether `OPENAI_API_KEY` is configured).

### Measured numbers on the 30-pair corpus
```
n=30 (in_scope=18, oos=12)  accuracy=0.882  version=0.944  refusal=1.000  p95=11252ms
```
Best across the project. `accuracy` +0.17 vs v0.6.1, `version` +0.07, `in_scope` more than doubled.

### Scope, documented
- Click-to-open citations and the settings UI drawer were on the v0.7 plan but trimmed mid-milestone after the retrieval changes landed clean numbers. The wire protocol (`CitationRef.source_url`, `SettingsUpdate`) is already in place; v0.7.1 is purely webview/extension JS+TS work. Shipping a focused v0.7 with measured wins beat a sprawling diff that risked regressing what just landed — same lesson v0.6.1 taught.

## [0.6.1] - 2026-06-02

### Fixed
- **`version_correctness` regression from v0.6 in `FindInChangelogTool`**: `_extract_version_section` in `tools.py` used a plain substring match (`version in section.lower()`), which let neighbouring-version sections leak in whenever they mentioned the requested version in passing (e.g. React 19's section saying "fixed in 18.2.0"). That leak surfaced as the model picking up React-19 / Pydantic-v2 APIs from supposedly-version-scoped context. v0.6.1 replaces the substring match with a word-boundary regex anchored to the section HEADING. The version must appear in the H2 heading itself, not just the body. "0.95.0" no longer matches "0.95.10".

### Added
- Two new `test_tools.py` tests exercising the changelog fix directly: one verifies React 19's body-mention of 18.2.0 doesn't leak into a v18.2.0 query; one verifies "0.95.0" doesn't prefix-match "0.95.10".
- `test_sidecar_chat_websocket_echoes` timeout raised from 15s to 45s to fit the v0.6 multi-iteration agent loop's real-world p95 (~7s + headroom).

### Changed
- Bumped `extension/package.json` and `sidecar/pyproject.toml` to 0.6.1.

### Tried, reverted
- Attempted to soften the HARD RULES system prompt so the model would try a second tool before refusing on an empty first-tool result. The intent was to recover v0.6's recall (16/24 in-scope entries were over-refused). The change swung too far the other way: the eval went from `in_scope=8, accuracy=0.625, version=0.750` to `in_scope=29, accuracy=0.000, version=0.207` because the model interpreted any loosely-related second-tool output as "useful context" and synthesized wrong-version answers — even off-topic Django / Vite questions got answered from base knowledge. Reverted to the v0.6 prompt; the prompt-vs-floor recall problem is a v0.7 structural item (per-collection thresholds + chunk metadata for version grounding), not a wording one.

### Notes
- v0.6.1 is a focused patch: it ships ONLY the changelog regex fix. The prompt-softening experiment is documented in this CHANGELOG as a failed iteration rather than hidden — the data showed the trade-off and the v0.7 plan accounts for it.
- Expected eval numbers post-patch: similar to v0.6's `in_scope=8, accuracy=0.625` baseline but with `version_correctness` lifted back toward 1.000 because the changelog leak that put React-19 and Pydantic-v2 strings into otherwise-correct answers is now closed.

## [0.6.0] - 2026-06-02

### Added
- **Multi-iteration ReAct loop** via OpenAI's tool-calling API. `Agent.answer()` now passes `tools=[...]` to `chat.completions.create`, forces a tool dispatch on iteration 0 with `tool_choice="required"`, then lets the model chain up to `max_iterations=3` tool calls before finalizing the answer. Citations aggregate across iterations and dedupe at finalize time.
- **Real `SearchWorkspaceCodeTool`** via `asyncio.create_subprocess_exec` over ripgrep with `--json` output, 8s timeout, top-5 hits with 2 lines of context. Graceful messages for missing workspace, missing rg binary, and subprocess timeout.
- **Real `FindInChangelogTool`** that fetches CHANGELOG.md from per-library raw-GitHub URLs (React's facebook/react, FastAPI's tiangolo/fastapi), per-process LRU-by-URL cache, regex-splits on `## <heading>` and returns sections matching both version and query, truncated to 6 KB.
- **FastAPI 0.95 as second indexed library.** `_urls_for()` extends with 10 FastAPI tutorial pages. Validates the indexer works for non-React libraries.
- **Eval corpus extends to 30 pairs across two libraries**: 16 React in-scope + 8 FastAPI in-scope + 6 oos (4 React-context + 2 FastAPI-context). `forbidden_apis` on FastAPI entries pin Pydantic v2 idioms (`model_dump`, `model_config`, `model_validate`) so the eval catches FastAPI 0.95 → 0.100+ regressions.
- `Agent` constructor accepts `workspace_path` (threaded from `DOCCHAT_WORKSPACE_PATH` env var) and `max_iterations`.
- `AgentResponse.iterations` field surfaces how many ReAct iterations actually ran.

### Changed
- System prompt rewritten for the tool-calling shape. HARD RULES section preserved with the canonical "I don't have documentation" phrase so the eval's `is_refusal` heuristic continues to match.
- `tool_schemas()` shape unchanged at the source; the Agent wraps each in OpenAI's `{"type": "function", "function": {...}}` envelope before passing to chat completions.
- Bumped `extension/package.json` and `sidecar/pyproject.toml` to 0.6.0.
- Indexer error message for unsupported libraries now reflects v0.6's two-library support.

### Notes
- ADR-009 captures the design space: why multi-iteration via OpenAI function-calling beats manual ReAct, why ripgrep over the VS Code workspace API at v0.6, why CHANGELOG fetch + regex beats Qdrant indexing for release notes, why one merged corpus over per-library files.
- `rg` is a soft dependency. Workspace-code questions degrade gracefully when it's not installed.
- **Measured numbers on the 30-pair corpus**: `refusal_rate=1.000`, `version_correctness=0.750`, `answer_accuracy=0.625`, `p95=7251ms`. 8 of 24 in-scope corpus entries answered through; the other 16 over-refused because the 0.15 retrieval floor that worked on the tighter React-only v0.5 corpus is too aggressive for the expanded FastAPI surface. 2 entries hit the `max_iterations=3` cap (useId, FastAPI Annotated) — exactly the cases where retrieval was inconclusive and the model correctly tried multiple tools. `version_correctness` dropped from v0.5's 1.000 to 0.750 because 2 of 8 answered entries leaked a next-major API; the chunk-metadata refactor at v0.7 will let the prompt enforce version-specific grounding more strictly.
- Trade-off acknowledged: v0.6 made the architecture deeper (multi-iteration ReAct, real workspace + changelog tools, second indexed library) at the cost of measured recall. Floor-based gating that worked on a single tight corpus hits its limit on a more diverse one.

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

[Unreleased]: https://github.com/AshwinUgale/docchat/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/AshwinUgale/docchat/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/AshwinUgale/docchat/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/AshwinUgale/docchat/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/AshwinUgale/docchat/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/AshwinUgale/docchat/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/AshwinUgale/docchat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AshwinUgale/docchat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AshwinUgale/docchat/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/AshwinUgale/docchat/releases/tag/v0.0.1
