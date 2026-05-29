# DocChat eval harness

Q/A corpus and runner. **Empty at v0.0; landing at v0.4.**

The plan (per `.cowork/PROGRESS.md`):

- 20-pair labelled corpus for React 18.2 at v0.4. Each pair has the correct-for-18.2 answer plus a wrong "this would be right for 19" counter-answer.
- `evals/run.py` script that replays the corpus through the sidecar and computes:
  - **Answer accuracy** — does the answer match the labelled correct answer?
  - **Version-correctness** — does the answer reference APIs that exist in the pinned version?
- Results published in the top-level `README.md` as the headline numbers.

Expansion to 50 pairs × 2 libraries lands at v1.0.
