# Evaluate PaperTrail

Run the scripted fixture checks from the project root.

```sh
uv run --locked python scripts/run_eval.py
```

Each run writes `results.json` and `summary.md` in a new directory under
`eval_outputs/`. Use `--output-dir` to choose a different parent directory.
The scripted run exercises the workflow with four fixed cases. Its timings
measure local workflow overhead with predetermined model outputs.

## Gold cases and live evaluation

The bundled [case set](../data/eval_cases.jsonl) has four tuning questions
and 16 holdout questions. The holdout has eight answerable, four partial
and four unsupported questions. Its labels were drafted against the six
synthetic passages and are ready for human review. Keep the holdout fixed
when changing prompts and compare runs with the same case and corpus hashes.

Set `OPENAI_API_KEY` in the environment and opt in with `--live`. The default
split is `holdout`. Use `--split tune` to test a change on the tuning set.
`OPENAI_MODEL` sets the same model for both paths.

```sh
uv run --locked python scripts/run_eval.py --live
```

Both paths use the same corpus, TF-IDF retrieval, six-passage first-pass cap,
drafting adapter and citation checks. The baseline searches the original
question and passes it as the evidence request for one draft. The graph can
plan queries and perform one follow-up. Latency includes each path's search,
model calls and citation checks. Graph compilation and export are outside the
timer. Token usage lists the counts returned by the provider for available calls.

First-pass Recall@6 averages labeled passage recall across cases with relevant
passages and an attempted search. Final missing IDs mark retrieval misses.
Uncited retrieved IDs and distractor citations flag evidence-use decisions
for review. An error before retrieval has unavailable recall. Provider errors
have their own status and count.

## Human claim-support review template

**Run directory** ___  
**Reviewer and date** ___  
**Model and split** ___  
**Corpus and case-set hashes** ___

Copy one row per returned finding from `results.json`. Read the cited full
passage in the corpus alongside the quote. Check that the source supports
the claim's direction, conditions, quantities and units.

| Case ID | Path | Claim | Passage ID | Quote | Rating | Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| ___ | Baseline / Graph | ___ | ___ | ___ | Supported / Partly supported / Unsupported / Unreviewed | ___ |

Use **Supported** when the passage supports the full claim, **Partly
supported** when it supports only part, and **Unsupported** when it
contradicts the claim or supplies no support.

For each missed expected finding or unresolved point, record the case ID,
path, needed passage ID, and diagnosis. Use **retrieval miss** when the
needed passage was absent, **evidence-use issue** when the available passage
was omitted or misread, **citation rejected** when the checker flagged the
finding, or **justified gap** when the collection lacks the evidence.

For each path, record supported, partly supported, unsupported and
unreviewed counts. Report full support as **supported / reviewed** with
both counts and the percentage. Record the number of unreviewed claims
alongside it.
