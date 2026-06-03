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
    return p


async def _run(args: argparse.Namespace) -> RunSummary:
    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient

    from docchat_sidecar.agent import Agent  # type: ignore[import-not-found]
    from docchat_sidecar.memory import build_memory  # type: ignore[import-not-found]

    corpus_data = json.loads(args.corpus.read_text(encoding="utf-8"))
    corpus = TypeAdapter(Corpus).validate_python(corpus_data)
    entries = corpus.entries[: args.limit] if args.limit else corpus.entries

    openai = AsyncOpenAI()
    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    memory = build_memory(workspace_path=None)
    agent = Agent(openai=openai, qdrant=qdrant, memory=memory)
    judge = None if args.no_judge else LLMJudge(openai=openai)

    results = await run_corpus(agent=agent, entries=entries, judge=judge)
    metrics = compute_metrics(results)
    return RunSummary(
        corpus_name=corpus.name,
        config={
            "n_entries_requested": len(corpus.entries),
            "n_entries_run": len(entries),
            "judge_enabled": not args.no_judge,
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
        f"p95={m.p95_latency_ms:.0f}ms "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
