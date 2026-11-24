# PaperTrail: Codex build plan

**Implementation guide · VS Code and Git · Five build sessions**  
Use this alongside `PaperTrail_Project_Brief.md`.

## Before writing code

1. Create a new folder named `papertrail-agent` and open that folder in VS Code.
2. Put these two Markdown documents under `docs/` and start a Git repository.
3. Use Codex on **one session at a time**. Review the diff and run the checks yourself before committing. Ask Codex to list which checks it ran and any failures or gaps.
4. Keep provider credentials in your local environment or an ignored `.env` file. Commit only an empty `.env.example`. The demo and CI should work without a provider key.

### Target repo layout

```text
papertrail-agent/
├── app.py                     # Streamlit entry point
├── src/papertrail/
│   ├── schemas.py             # Passage, finding, response
│   ├── retrieval.py           # Local search and stable ranking
│   ├── graph.py               # LangGraph nodes and routing
│   └── models.py              # Fake and live provider adapters
├── data/
│   ├── corpus.jsonl           # Searchable passages and provenance
│   └── eval_cases.jsonl        # Human-labeled gold cases and split
├── scripts/run_eval.py        # Baseline vs. graph evaluation
├── tests/                     # Offline tests
├── docs/                      # Brief, plan and benchmark report
├── .github/workflows/ci.yml
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

The commands below are targets for the files you’ll build. Once Codex creates the project configuration, use `uv sync --locked` to install from the committed lockfile and select the project's `.venv` interpreter in VS Code.

## Session 1 — Corpus and retrieval

Build local corpus loading and search before calling an LLM. Give each record a stable ID, title, source URL, section or page, source terms and text. Start with a handful of synthetic passages covering similar terms, conflicting details and an unrelated question. Use TF-IDF; discard zero-score matches and sort ties by stable ID so tests are repeatable.

**Done when:** A question retrieves a known relevant passage in the top results; an unrelated query can return no results. Missing or duplicate passage IDs are rejected. `uv run pytest -q` passes with networking off.

**Paste into Codex:**

> Read the project brief and action plan in docs/. Implement Session 1 only. Create the Python 3.12 project with uv, a small synthetic JSONL corpus with stable IDs, and a typed TF-IDF search function. Add focused tests for relevant retrieval, an unrelated query, and invalid corpus records. Run the tests and show me the files changed. Do not add an LLM provider or UI yet.

## Session 2 — LangGraph and citation checks

Implement plan, retrieve, draft and check as graph nodes. Define the data contract up front: a finding contains `claim`, `passage_id` and `quote`; the response includes findings, unresolved points and a status. Inject the model adapter so tests can force a bad citation or a missing result without making a network call.

**Done when:** A scripted run can answer on the first pass, repair once, or stop with evidence missing. Tests reject an invented ID, an ID outside the retrieved set, a fabricated quote and malformed model output. A forced provider timeout reports an error; no run exceeds the search or model-call limits.

**Paste into Codex:**

> Implement Session 2 only. Build the LangGraph StateGraph with a scripted model adapter and Pydantic outputs. Put quote and passage checks in plain Python. Add conditional routing for one repair search, then return answered, partial, insufficient_evidence or error. Test the failure paths and the retry cap. Keep the retrieval interface from Session 1 and leave the UI for the next step.

## Session 3 — Dashboard and live mode

Build one Streamlit screen with example questions, findings, source links and run details. Use a form so editing the input does not make new model calls. Wire in one provider behind the existing adapter and document how to configure it. Keep the scripted offline screen available by default; the three demo questions should exercise different outcomes.

**Done when:** `uv run streamlit run app.py` opens a demo from a fresh checkout without credentials. You can click or expand each source, see any unresolved parts and tell whether the result is scripted or live. With a key, try one real request and confirm that a provider failure produces a useful message without exposing secrets.

**Paste into Codex:**

> Implement Session 3 only. Build a one-page Streamlit UI that calls the existing graph and displays the answer, linked sources, exact quotes, unresolved items and run statistics. Default to the three scripted demo questions and label that mode. Add one optional live model adapter with structured outputs and a timeout. Keep API keys out of code and logs. Give me a short manual smoke-check list.

## Session 4 — Run a fair evaluation

Create **20 labeled questions** as a starting target: 4 for tuning and 16 held out (8 fully answerable, 4 partially answerable and 4 unsupported). For each, record expected findings or gaps and relevant passage IDs. You write or review these labels against the source passages. Lock the holdout before adjusting prompts. Include a paraphrase with weak lexical overlap, a question that needs two passages, and one with an attractive but irrelevant citation.

Implement the baseline and graph comparison using the rules in the brief. Export case-level JSON and create a short Markdown summary with the sample size, model, corpus version, metrics and two inspected examples. For a quick offline check, the same runner can use scripted responses; label those scores as fixture checks, not model performance.

**Done when:** Both modes can run against the same holdout. The report splits retrieval misses from poor use of retrieved evidence, and lists quality changes next to the extra calls and latency. Review claim support by hand before publishing anything.

**Paste into Codex:**

> Implement Session 4 only. Add a gold-case schema and a runner that compares single-query retrieval with the graph using the same corpus, model, first-pass context cap and citation validator. Export per-case results, latency, calls, and any available token counts. Use deterministic fixtures for automated checks and keep live runs opt-in. Give me a template for human claim-support review. Do not fill in benchmark results until a run has produced them.

## Session 5 — Ship the portfolio version

Add GitHub Actions for offline tests and lint checks. Clone the repo into a fresh folder and follow your own README. Capture a short demo of an answered question, then show a case where the agent returns partial or insufficient evidence. Include one actual before/after improvement from evaluation in the README or benchmark report.

**Done when:**

- [ ] A reviewer can start the demo, open the evidence and understand the limitations without asking you for a key.
- [ ] The README explains the problem, the graph, setup, evaluation, provenance, and what you built personally.
- [ ] The committed corpus is appropriate to distribute. Credentials, local data and private traces are absent from Git.
- [ ] CI runs `uv run ruff check .` and `uv run pytest -q` without accessing the model provider.
- [ ] The report separates deterministic tests, real model results and ideas for future work.

**Paste into Codex:**

> Implement Session 5 only. Add GitHub Actions for offline Ruff and pytest, and write a reviewer-focused README using the commands the project actually supports. Include a small Mermaid architecture diagram and links to the benchmark and corpus provenance. Check that the offline demo can run from a clean install. Do not add feature scope or publish the repository.

## What you should be able to explain in an interview

- **Why LangGraph?** Show the evidence check and the conditional retry. Explain where the model makes a decision and where Python enforces a rule.
- **Why start with TF-IDF?** Explain the small corpus and show a real paraphrase miss. If those misses are common, compare embeddings as a follow-up using the same holdout.
- **What failed and what changed?** Pick one case from your results; show the retrieved passages, the output and the evidence for the change you made.
- **How would you take it further?** Tie the next feature to a measured problem. For example, add semantic retrieval if retrieval is the limiting step, or deploy a small pilot if real usage feedback is the missing evidence.
