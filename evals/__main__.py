"""``python -m evals`` - replay a corpus through the agent + report metrics.

Examples::

    # Cheap path - no judge (saves ~$0.002 on 20 entries)
    python -m evals --corpus evals/corpus.json --output out/eval.json --no-judge

    # Full run with LLM judge
    python -m evals --corpus evals/corpus.json --output out/eval.json

Requires Qdrant running with the relevant collection indexed (e.g.
``react_18_2_0``) and ``OPENAI_API_KEY`` set.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from pathlib import Path

from pydantic import TypeAdapter

# Lazy .env load - same pattern the sidecar uses.
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path, encoding="utf-8-sig")

# Make the sidecar package importable when the eval runs from the repo
# root without ``uv run``. The sidecar lives at ``./sidecar/src``.
_SIDECAR_SRC = Path(__file__).resolve().parent.parent / "sidecar" / "src"
if _SIDECAR_SRC.is_dir() and str(_SIDECAR_SRC) not in sys.path:
    sys.path.insert(0, str(_SIDECAR_SRC))

# The four imports below run AFTER the sys.path tweak above so that
# ``from evals.X`` and ``from docchat_sidecar.X`` resolve when invoked
# from the repo root without ``uv pip install``-ing the harness. Ruff's
# E402 flags this; the deviation is intentional.
from evals.judge import LLMJudge  # noqa: E402
from evals.metrics import compute_metrics  # noqa: E402
from evals.runner import run_corpus  # noqa: E402
from evals.schema import Corpus, RunSummary  # noqa: E402

# v0.9: surface per-query retrieval-score logging from SearchDocsTool so
# eval output shows WHICH queries dropped chunks under the floor.
# Configured here (after all imports, so ruff's E402 stays clean) and
# scoped to the eval CLI only - the sidecar's uvicorn config stays the
# source of truth in the production path.
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)


def _parse_floor_overrides(items: list[str] | None) -> dict[str, float]:
    """Parse ``--floor lib=value`` pairs into a ``{library: floor}`` dict.

    Raises ``ValueError`` on a malformed pair so a typo fails loudly rather
    than silently running with the tool's default floors.
    """
    floors: dict[str, float] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--floor expects 'library=value', got {item!r}")
        floors[key.strip().lower()] = float(value)
    return floors


def _select_entries(entries: list, split: str) -> list:
    """Filter corpus entries by split tag.

    ``split == "all"`` runs everything. Otherwise keep entries whose ``split``
    matches, plus untagged (``split is None``) entries, which belong to every
    split - so a corpus with no split tags behaves identically under any
    ``--split`` value.
    """
    if split == "all":
        return list(entries)
    return [e for e in entries if e.split is None or e.split == split]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evals")
    p.add_argument("--corpus", type=Path, required=True, help="Path to a corpus JSON file.")
    p.add_argument("--output", type=Path, required=True, help="Path to write the run summary JSON.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N entries (handy for sanity checks).",
    )
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-judge pass. answer_accuracy will be 0 in the output.",
    )
    p.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL. Defaults to QDRANT_URL env var or localhost:6333.",
    )
    p.add_argument(
        "--no-self-critique",
        action="store_true",
        help=(
            "Disable the v0.8 self-critique pass. Use this to ablate critique "
            "on/off (default is on)."
        ),
    )
    p.add_argument(
        "--no-topic-filter",
        action="store_true",
        help=(
            "Disable the v1.0 pre-retrieval topic classifier. Use this to "
            "measure the classifier's contribution to refusal_rate "
            "(default is on)."
        ),
    )
    p.add_argument(
        "--warm-memory",
        action="store_true",
        help=(
            "Keep Mneme memory across corpus entries instead of resetting it "
            "between each (the default). Warm mode tests cross-turn memory but "
            "makes per-entry metrics order-dependent and can leak an earlier "
            "entry's answer into a later one; leave off for the headline "
            "accuracy/version/refusal numbers."
        ),
    )
    p.add_argument(
        "--split",
        default="all",
        help=(
            "Run only entries whose 'split' tag matches (untagged entries "
            "always run). Score floors are decision thresholds and must be "
            "calibrated on --split calibration, then reported on --split test "
            "- tuning and reporting on the same entries is train-on-test. "
            "Default 'all'."
        ),
    )
    p.add_argument(
        "--score-floor",
        type=float,
        default=None,
        help=(
            "Override SearchDocsTool's default cosine floor for every library. "
            "Use to run an untuned baseline (e.g. --score-floor 0.0) or a "
            "calibration sweep without editing the sidecar defaults."
        ),
    )
    p.add_argument(
        "--floor",
        action="append",
        metavar="LIB=VALUE",
        help=(
            "Per-library floor override, e.g. --floor fastapi=0.12. Repeatable. "
            "Overrides --score-floor for that library."
        ),
    )
    return p


async def _run(args: argparse.Namespace) -> RunSummary:
    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient

    from docchat_sidecar.agent import Agent  # type: ignore[import-not-found]
    from docchat_sidecar.memory import build_memory  # type: ignore[import-not-found]

    corpus_data = json.loads(args.corpus.read_text(encoding="utf-8"))
    corpus = TypeAdapter(Corpus).validate_python(corpus_data)
    entries = _select_entries(corpus.entries, args.split)
    entries = entries[: args.limit] if args.limit else entries

    floor_overrides = _parse_floor_overrides(args.floor)

    openai = AsyncOpenAI()
    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    memory = build_memory(workspace_path=None)
    agent = Agent(
        openai=openai,
        qdrant=qdrant,
        memory=memory,
        self_critique=not args.no_self_critique,
        topic_filter=not args.no_topic_filter,
        score_floor=args.score_floor,
        floors_by_library=floor_overrides or None,
    )
    judge = None if args.no_judge else LLMJudge(openai=openai)

    results = await run_corpus(
        agent=agent, entries=entries, judge=judge, isolate_entries=not args.warm_memory
    )
    metrics = compute_metrics(results)
    return RunSummary(
        corpus_name=corpus.name,
        config={
            "n_entries_requested": len(corpus.entries),
            "n_entries_run": len(entries),
            "judge_enabled": not args.no_judge,
            "self_critique": not args.no_self_critique,
            "topic_filter": not args.no_topic_filter,
            "memory_isolation": not args.warm_memory,
            "split": args.split,
            "score_floor_override": (
                "default" if args.score_floor is None else str(args.score_floor)
            ),
            "floor_overrides": json.dumps(floor_overrides) if floor_overrides else "none",
            "qdrant_url": args.qdrant_url,
        },
        metrics=metrics,
        results=results,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    m = summary.metrics
    print(
        f"corpus={summary.corpus_name} "
        f"n={m.n_entries} "
        f"(in_scope={m.n_in_scope}, oos={m.n_out_of_scope}) "
        f"accuracy={m.answer_accuracy:.3f} "
        f"version={m.version_correctness:.3f} "
        f"refusal={m.refusal_rate:.3f} "
        f"overrefusal={m.overrefusal_rate:.3f} "
        f"p95={m.p95_latency_ms:.0f}ms "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
