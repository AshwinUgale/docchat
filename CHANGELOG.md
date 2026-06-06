# Changelog

All notable changes to DocChat are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from v1.0 onward.

## [Unreleased]

## [1.0.1] - 2026-06-06

### Fixed
- **Fresh marketplace installs are usable.** v1.0.0 shipped the TS extension to the marketplace but assumed the user's system Python already had `docchat_sidecar` installed, which is only true on the developer's box. Every fresh install hit `DocChat: sidecar exited before port line (code 1)` on first panel open. v1.0.1 bundles the sidecar source inside the `.vsix` and ships a setup command that installs it into a per-user managed venv.

### Added
- **`extension/scripts/prepackage.mjs`** runs as part of `vscode:prepublish`. Copies `sidecar/{pyproject.toml,uv.lock,src/}` into `extension/sidecar-source/`, rewrites the pyproject `readme` path so the bundled standalone install does not try to read `../README.md`, drops a stub `README.md` so hatchling has something to consume during `uv sync`'s build pass. `sidecar-source/` is .gitignored as a build artifact.
- **`DocChat: Set up sidecar` command** (`docchat.setupSidecar`). One-time per-user install: detects Python 3.11+ + `uv` on PATH, copies bundled source to `~/.docchat/sidecar/`, runs `uv sync` there, verifies the import works, writes the resulting venv python path to `docchat.sidecarPython` at Global scope. Re-runnable to repair a broken install. Missing Python or uv produces an error notification with the install URL — no auto-download (per ADR-014).
- **Recovery path on first openPanel failure.** When `spawnSidecar` rejects with `sidecar exited before port line` (the canonical "not installed yet" signal), the error notification now offers a **Run Setup** button alongside **View Logs**. One click runs `docchat.setupSidecar`.
- **`~/.docchat/sidecar/.venv` as a Python-resolution candidate** in `sidecar.ts`, ahead of the dev-workflow `<repo>/sidecar/.venv`. Users who run setup don't need to touch settings — the next `openPanel` finds the managed venv on its own.

### Changed
- `extension/package.json` → `1.0.1`, `sidecar/pyproject.toml` → `1.0.1`, `sidecar/src/docchat_sidecar/__init__.py` → `1.0.1` (was lagging at `0.0.1`; setup command's verify step now reports a meaningful version).
- Added `vscode:prepublish` script wiring `prepackage` → `compile` so `vsce package` and `vsce publish` automatically include the bundled source.
- Registered the new command in `activationEvents` and `contributes.commands`.

### Notes
- ADR-014 in `.cowork/DECISIONS.md` captures the bundle-vs-PyPI tradeoff and the deliberate non-auto-install of Python/uv.
- Docker / Qdrant prereq is still user-managed (out of v1.0.1 scope). Setup command only handles the Python sidecar piece.

## [1.0.0] - 2026-06-03

### Added
- **Per-library git-ref resolution in the indexer.** `_LIBRARY_CONFIG` maps each indexed library to a `(repo, paths, ref_for)` triple. FastAPI's `ref_for` returns the version tag (so `fastapi@0.95.2` and `fastapi@0.100.0` fetch from `tiangolo/fastapi/0.95.2/...` and `tiangolo/fastapi/0.100.0/...` respectively, getting the Pydantic-v1-era vs Pydantic-v2-era docs). React + Vue still fetch from `main` because their docs repos aren't version-tagged; the chunk metadata still surfaces the user's pinned version via the collection name + chunk header.
- **FastAPI 0.100 alongside 0.95 as the same-library-two-versions demo.** With git-ref resolution in place, both pinned versions index different content into separate Qdrant collections (`fastapi_0_95_2`, `fastapi_0_100_0`). The agent's lockfile-pinned routing (v0.9 + v0.9.1) picks the right one per query.
- **`parse_pyproject_toml` + `parse_requirements_txt`** in `lockfiles.py`. PEP 621 `[project] dependencies` + Poetry `[tool.poetry.dependencies]` for pyproject; `name==version` / `name>=version` parsing for requirements. Python-stdlib `tomllib`. Python-project workspaces now get the same lockfile-aware routing Node workspaces got at v0.9.1.
- **Production sidecar tries multiple manifest formats** (`package.json` → `pyproject.toml` → `requirements.txt`); first non-empty wins.
- **Pre-retrieval topic classifier in `Agent._is_library_topic`.** One cheap `gpt-4o-mini` call (temperature 0, max_tokens 4) classifies the query as LIBRARY or GENERAL before the ReAct loop starts. GENERAL queries short-circuit to the canonical refusal phrase WITHOUT touching retrieval. Closes the v0.9 let/const oos leak that the post-retrieval HARD RULE couldn't catch. Default `topic_filter=True`; `--no-topic-filter` flag on the eval CLI for ablation.
- **`_CANONICAL_REFUSAL` module constant** — single source of truth for the refusal phrase shared by the topic-filter short-circuit and the system prompt's HARD RULE #2.
- **Corpus extends to 48 pairs** with 8 new FastAPI 0.100 entries that pin Pydantic-v1 idioms (`.dict()`, `class Config`, `@validator`) as `forbidden_apis`. An agent that fetches from the wrong FastAPI collection now gets caught by `version_correctness`.
- 2 new schema tests: `test_bundled_corpus_loads` updated for 48 entries; `test_bundled_corpus_has_fastapi_at_two_versions` verifies both 0.95.x and 0.100.x are represented.

### Changed
- `Development Status :: 3 - Alpha` → `Development Status :: 5 - Production/Stable` on `sidecar/pyproject.toml`.
- README status badge flips from alpha to production-ready; install / roadmap / how-it-does sections refreshed for 1.0.
- Bumped `extension/package.json` and `sidecar/pyproject.toml` to `1.0.0`.

### Notes
- Marketplace publish itself stays a user-action item: register a `AshwinUgale` publisher on the Visual Studio Marketplace, design `icon.png` (128×128), run `vsce publish`. v1.0 ships the publish-ready manifest and the `.vsix`; the actual marketplace listing is signed by the user. See ADR-013 for the rationale.
- Auto-install sidecar (bundled venv + `uv` ship via the `.vsix`) deferred to v1.0.x — the cross-platform venv-creation work is bounded but real; better as its own milestone than crammed into 1.0.
- ADR-013 in `.cowork/DECISIONS.md` captures the full v1.0 design space: per-library git-ref resolution, Python lockfile parsers, topic classifier, classifier bump.

### Measured numbers on the 48-pair corpus
```
n=48 (in_scope=41, oos=7)  accuracy=0.675  version=0.854  refusal=1.000  p95=17239ms
```
- **All 40 in-scope corpus entries answered through** (100% recall on questions about pinned libraries). The 41st "in_scope" entry is the Vite oos that leaked — see below.
- **`refusal=1.000` over 7 classified oos** (Flask, Node EADDRINUSE, let/const, Django, Angular signal, Svelte rune, FastAPI-from-React-context). 6 of 7 caught by the v1.0 topic classifier BEFORE retrieval — saves the multi-iteration ReAct cost on queries we'd reject anyway.
- **`version=0.854` is best-on-this-corpus** despite the larger surface (48 pairs across 4 indexed (library, version) collections).
- Accuracy ticked down vs v0.9.1's 0.781 because the 8 new FastAPI 0.100 entries are stricter (expected exact strings like `model_dump`, `model_config`, `field_validator`) and the existing `fastapi_0_95_0` collection wasn't re-indexed from the 0.95.2 tag — it still has master-era content. v1.0.x: re-index 0.95 from the version tag.

### Honest residual leak
- **`react_18_oos_unrelated_lib`** ("How do I configure a Vite plugin for image optimisation?") still answered. The topic classifier let it through as LIBRARY because Vite IS a real library — but Vite isn't in the user's pinned set. The classifier prompt needs one extra clause: "If the question is about a library that's not in the user's pinned list, classify as GENERAL." Single-sentence prompt-tuning fix, queued for v1.0.1.

## [0.9.1] - 2026-06-03

### Added
- **Production sidecar reads lockfile pins via `lockfiles.parse_package_json`.** `_run_agent` in `__main__.py` now parses `<workspace_path>/package.json` (and the sibling `package-lock.json` for exact-version resolution) and passes the resulting `{library: version}` dict as `pinned_libraries` to `Agent.answer_stream`. Live extension demos now get the same routing fix the eval runner got at v0.9 — the agent calls the right collection per query instead of guessing from question text. Logs a `warning` line with the loaded pin count so the Output channel surfaces what was picked up.
- **`Agent.answer_stream` accepts `pinned_libraries`**, matching `Agent.answer`'s v0.9 signature. Both paths share the same system-prompt + `_dispatch` plumbing.
- **HARD RULE #5 in the agent system prompt: general-programming refusal.** Questions about language syntax / variable declarations / runtime errors that aren't tied to a pinned library refuse with the canonical phrase. Addresses the v0.9 eval leak where `react_18_oos_general_js` ("difference between let and const") routed to React docs (top score 0.34) and answered instead of refusing.

### Notes
- The pyproject.toml / requirements.txt parsers (for Python projects without package.json) stay deferred to v1.0 — they need new `lockfiles.parse_pyproject_toml` etc. work.
- **HARD RULE #5 measured outcome: no metric change.** v0.9.1 eval landed identical to v0.9: `n=40 (in_scope=35, oos=5) accuracy=0.781 version=0.943 refusal=1.000`. The `let vs const` entry still answered with React docs — score logs show retrieval succeeded (top scores 0.31–0.32, above floor) and the model produced a non-refusal answer despite the prompt rule. The structural limit of prompt-engineering refusal on `gpt-4o-mini`: it interprets "let vs const" as React-adjacent enough not to refuse. v1.0 candidates: pre-retrieval topic classifier (one-shot "is this question library-specific?") or top-1-score-gap check (refuse when score sits significantly below in-scope band for the library). Documented the attempt + measurement; default ships with the rule in place since it adds zero cost and might help on other queries.
- **Metric stability across v0.9 → v0.9.1** is itself a portfolio signal: the production-parity lockfile plumbing wasn't supposed to move the eval (already simulated at v0.9), and the refusal-rule attempt was measured to land unchanged. Both as expected; nothing snuck in unmeasured.

## [0.9.0] - 2026-06-03

### Added
- **Per-query score logging on `SearchDocsTool`**. Each retrieval call emits `search_docs <lib>@<ver> floor=<X> top-scores=[...] query=<...>` at `logger.info`. Eval CLI (`evals/__main__.py`) configures `logging.basicConfig(level=INFO)` so these surface in stdout alongside the headline metrics — turns "why did this query refuse?" from guesswork into a one-glance answer.
- **`api_name` filter on `SearchDocsTool.run`**. New optional kwarg; when set, post-filters Qdrant hits to chunks whose payload `api_name` matches (case-insensitive `startswith`). Exposed via the OpenAI function schema so the model can pass `api_name="useState"` when the user's question names a specific API. Falls through to the same canonical refusal when no chunk matches.
- **`pinned_libraries` kwarg on `Agent.answer()` + eval runner now passes it.** New `dict[str, str]` mapping library → pinned version. When provided, the agent's system prompt gets a "Project lockfile pins" section AND `_dispatch` overrides any tool-call `version` arg with the pinned one. Closes the bug surfaced by v0.9's score logging: the LLM was guessing versions from question text and hitting unindexed collections (`fastapi_0_95_2` instead of `fastapi_0_95_0`, `vue_3_0_0` instead of `vue_3_4_0`). The bug had been there since v0.6 but was invisible until score logging exposed it.
- 3 new `test_tools.py` tests covering the api_name filter (keeps matching, drops non-matching, empty-after-filter refuses, omitted-kwarg unchanged).

### Changed
- **Vue floor recalibrated from 0.10 → 0.05** after v0.8 eval showed 7/8 in-scope Vue queries still refused at 0.10. Vue's Composition-API reference pages intersperse type signatures with prose so cosine scores cluster lower than FastAPI's tutorial pages. Vue oos entries (Angular signal, Svelte rune) still refuse cleanly at 0.05 because they don't match anything in the Vue collection at any score.
- Agent `_dispatch` forwards `api_name` to `SearchDocsTool.run` when the LLM provides it; other tools ignore the kwarg.
- Bumped `extension/package.json` and `sidecar/pyproject.toml` to `0.9.0`.

### Notes
- v0.9 is intentionally smaller than v0.8 — no new library, no corpus expansion, no agent-loop changes. After v0.8's marathon-with-mid-milestone-reversion, the discipline play was shipping focused.
- ADR-012 captures the floor calibration + score logging + api_name filter + the `pinned_libraries` bug-fix that score-logging surfaced; the FastAPI-0.100-alongside-0.95 same-library-two-versions demo is deferred to v1.0 because the indexer currently fetches from `master` only — it needs the git-ref resolution work first.
- The `pinned_libraries` plumbing is wired through the eval runner at v0.9. The production sidecar (`_run_agent` in `__main__.py`) does NOT yet read lockfile pins; that requires gluing `lockfiles.py` into the agent construction path and lands at v1.0 alongside the auto-install sidecar work.

## [0.8.0] - 2026-06-03

### Added
- **Chunk-level metadata** in the Qdrant payload. Every chunk now carries `api_name` (derived from source filename via `_api_name_from_url`) and `section_heading` (most recent `## ` H2 at the chunk's start, tracked across the paragraph chunker). `SearchDocsTool` surfaces both in each header: `## react@18.2.0 - useState  (useState.md / Reference)`. Older collections without the fields still serve answers via `payload.get(...)` fallbacks.
- **Vue 3.4** as the third indexed library. `_urls_for("vue")` covers 10 `vuejs/docs` markdown pages from the Composition API surface (reactivity-core, computed, watch, lifecycle, script-setup, provide/inject, etc.).
- **Eval corpus** extends to 40 pairs (16 React + 8 FastAPI + 8 Vue in-scope + 8 oos). Vue entries pin Vue-3.5+ APIs (`useTemplateRef`), Vue 2 APIs (`Vue.observable`, `Vue.set`), and Options-API hooks as `forbidden_apis`. New oos entries cover Angular `signal()` and Svelte `$state` cross-framework leaks.
- **Self-critique pass on `Agent.answer()`** (constructor kwarg `self_critique`, defaults to `False` after eval ablation — see Tried, reverted below). After the multi-iteration ReAct loop produces a draft, one cheap `temperature=0` chat completion re-reads it against the joined tool outputs (truncated at 12 KB). Critique replies `OK` (ship draft unchanged) or returns a revised answer. Cost: ~$0.0001 + ~2s per query when on. Streaming path (`answer_stream()`) skips critique to keep the streaming UX clean.
- **`--no-self-critique` flag on `python -m evals`** for ablation runs against the same indexed collections.
- **Marketplace-prep manifest** in `extension/package.json`: marketing-grade `displayName` + `description` (with headline eval numbers), `publisher: AshwinUgale`, expanded `categories` + `keywords`, `bugs.url`, `homepage`, `qna`, `pricing`, `galleryBanner`. Five new VS Code config properties (`docchat.chatModel`, `docchat.scoreFloor`, `docchat.maxIterations`, `docchat.selfCritique`, plus the existing `sidecarPython` / `sidecarPort`) so users can tune via Settings UI in addition to the in-panel drawer.
- 3 new indexer tests covering H2-heading capture across chunk boundaries.
- 2 new agent tests covering self-critique: one verifies `OK` keeps the draft, one verifies a non-`OK` reply ships as the revised answer.
- 1 new schema test verifying the corpus covers all three indexed libraries (`react`, `fastapi`, `vue`).

### Changed
- `_split_into_chunks` signature changed from `Iterable[str]` to `Iterable[tuple[str, str | None]]`. Each chunk now carries its `section_heading` at chunk start. Existing tests updated to unpack the tuple.
- `_Chunk` dataclass gains `api_name: str` and `section_heading: str | None`.
- `Citation` and `CitationRef` already carried `source_url` (v0.7.1); v0.8 didn't expand the wire shape further — the new metadata stays internal to the chunk-header prompt.
- `SearchDocsTool.floors_by_library` adds `vue: 0.10` (mirrors the FastAPI calibration — Vue's reference + guide pages dilute cosine the same way).
- Indexer error message for unsupported libraries updated: `(v0.8 supports react + fastapi + vue)`.
- Bumped `extension/package.json` and `sidecar/pyproject.toml` to `0.8.0`.

### Tried, reverted (default flipped)
- **Self-critique defaulted ON** during initial v0.8 implementation. Eval ablation on the 40-pair corpus measured the regression vs critique-OFF:

  |  | critique ON | critique OFF |
  |---|---|---|
  | accuracy | 0.667 | **0.824** |
  | version | 0.895 | **0.947** |
  | p95 | 23.1s | **8.1s** |

  The critique pass was actively rewriting well-grounded drafts into worse ones. Default flipped to `self_critique=False` for v0.8 ship. Feature stays as opt-in via constructor kwarg + `--no-self-critique` eval flag so future prompt-tuning can revisit. Same pattern as v0.6.1's prompt-softening reversion: tried it, measured it, kept the data attached, flipped the default.

### Notes
- Marketplace publish itself is NOT done in v0.8 — author needs to register the publisher on the Visual Studio Marketplace and design `icon.png` (intentionally not bundled; placeholder would be worse than missing). v1.0 lands the actual `vsce publish` and bundled icon.

## [0.7.1] - 2026-06-02

### Added
- **Click-to-open citations.** Each citation in the streaming-final footer is now a clickable chip showing `library@version:source`. Clicking posts an `openCitation` message to the extension, which calls `vscode.env.openExternal` with the original raw-GitHub `source_url`. URL is validated to be `http(s)://` only so a malicious payload can't shell out to `file://` or `vscode://`.
- **Settings drawer in the webview** (gear icon in the header). Three knobs:
  - `chat_model` — dropdown: `gpt-4o-mini` (default) / `gpt-4o`
  - `score_floor` — slider 0.05–0.40 step 0.01
  - `max_iterations` — slider 1–5 step 1
  Changes send a `SettingsUpdate` message over the WebSocket directly to the sidecar; the runtime settings dict applies on the next `user_query`. No sidecar respawn needed. The protocol was already in place since v0.7; v0.7.1 is purely the UI.
- `Citation.source_url` field — populated by `SearchDocsTool` (from the Qdrant payload's `source_url`) and `FindInChangelogTool` (from the raw-GitHub URL it fetched). Propagates through to `CitationRef.source_url` over the wire so the webview can render click-to-open chips without re-parsing the streamed text.

### Changed
- Bumped `extension/package.json` and `sidecar/pyproject.toml` to `0.7.1`.
- Webview header reads `v0.7.1`.

### Notes
- v0.7.1 is purely webview + extension JS/TS work plus the small `source_url` plumbing in `tools.py` / `agent.py`. Eval numbers and the sidecar agent path are unchanged from v0.7.0.

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

[Unreleased]: https://github.com/AshwinUgale/docchat/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/AshwinUgale/docchat/compare/v0.9.1...v1.0.0
[0.9.1]: https://github.com/AshwinUgale/docchat/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/AshwinUgale/docchat/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/AshwinUgale/docchat/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/AshwinUgale/docchat/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/AshwinUgale/docchat/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/AshwinUgale/docchat/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/AshwinUgale/docchat/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/AshwinUgale/docchat/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/AshwinUgale/docchat/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/AshwinUgale/docchat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AshwinUgale/docchat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AshwinUgale/docchat/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/AshwinUgale/docchat/releases/tag/v0.0.1
