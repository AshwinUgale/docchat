# DocChat eval harness

Replays a labelled Q/A corpus through the agent (router + tools + Mneme + LLM)
and reports metrics. Each corpus entry pins a `library@version`, carries the
correct-for-that-version answer, and may carry APIs that would be wrong for a
different version.

## Run

```powershell
docker compose up -d qdrant                       # retrieval backend
$env:PYTHONPATH = "$PWD;$PWD\sidecar\src"
uv --directory sidecar run python -m evals `
    --corpus ..\evals\corpus.json --output ..\out\eval.json
```

Needs Qdrant up with the relevant collections indexed and `OPENAI_API_KEY` set.
Add `--no-judge` for a cheap run (skips the LLM-as-judge; `answer_accuracy`
reports 0). `--limit N` runs the first N entries.

## Metrics

Split **in-scope vs out-of-scope on the corpus label** (`CorpusEntry.out_of_scope`),
never on the agent's behaviour — deriving scope from `refused` is circular and
hides the two failures that matter most.

- **answer_accuracy** — fraction of judged in-scope entries the LLM-judge marked
  correct. An in-scope entry the agent *refused* is judged incorrect, so
  over-refusal lowers this instead of vanishing from the denominator.
- **version_correctness** — fraction of in-scope entries whose answer contains
  every `expected_apis` and no `forbidden_apis` (substring, case-insensitive).
- **refusal_rate** — fraction of out-of-scope entries the agent refused. An
  out-of-scope entry the agent *answered* (hallucinated) lowers this, as it
  should. 1.0 is the goal.
- **overrefusal_rate** — fraction of in-scope entries the agent wrongly refused.
  0.0 is the goal. The companion to `refusal_rate`; surfaces the failure the old
  behaviour-derived scope split silently hid.

## Memory isolation

The harness runs one persistent agent over the whole corpus, and each answer
records the Q/A into Mneme and surfaces prior Q/As in the next prompt. By default
(`isolate_entries=True`) memory is **reset before each entry** so every labelled
probe is answered cold — otherwise metrics become order-dependent and an earlier
entry's answer (e.g. one containing `useState`) can leak into a later, similar
entry. Pass `--warm-memory` only to deliberately exercise cross-turn memory (a
different, multi-turn kind of test).

## Score floors & held-out splits

`SearchDocsTool` drops retrieved chunks below a per-library cosine floor. Those
floors are **decision thresholds** — tuning them on the same entries you report
on is train-on-test. The shipped defaults were tuned against this corpus and
should be treated as provisional until calibrated on held-out data.

The harness makes calibration and reporting separable:

- **`--split NAME`** runs only entries tagged `split == NAME` (untagged entries
  belong to every split, so an untagged corpus is unaffected). Tag some entries
  `"calibration"` and the rest `"test"`, then calibrate on one and report on the
  other.
- **`--score-floor FLOAT`** overrides the global floor for a run — e.g.
  `--score-floor 0.0` for an untuned baseline that shows how much the tuning
  flatters the numbers.
- **`--floor LIB=VALUE`** (repeatable) overrides one library's floor, e.g.
  `--floor fastapi=0.12`.

Overrides are passed through to `SearchDocsTool` without editing its production
defaults, and the chosen split + floors are recorded in the run summary's
`config` block for provenance.

Suggested workflow: sweep `--score-floor` / `--floor` on `--split calibration`,
pick the floors that maximise calibration metrics, then do a single
`--split test` run with those floors for the headline numbers.
