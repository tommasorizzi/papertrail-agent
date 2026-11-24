import argparse
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel

from papertrail.eval_fixtures import FIXTURE_CASES, FIXTURE_MODELS
from papertrail.eval_report import render_report, summarize
from papertrail.evaluation import evaluate, load_gold_cases
from papertrail.graph import MAX_PASSAGES_PER_PASS
from papertrail.live import DEFAULT_OPENAI_MODEL, OpenAIModel
from papertrail.models import ModelAdapter
from papertrail.retrieval import load_corpus
from papertrail.schemas import TokenUsage

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fingerprint(records: Sequence[BaseModel]) -> str:
    data = [record.model_dump(mode="json") for record in records]
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare single-query retrieval and the PaperTrail graph."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use OpenAI for both paths. Requires OPENAI_API_KEY.",
    )
    parser.add_argument("--split", choices=("tune", "holdout"), default="holdout")
    parser.add_argument(
        "--cases",
        type=Path,
        help="Gold JSONL. Live runs default to data/eval_cases.jsonl.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=PROJECT_ROOT / "data" / "corpus.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_outputs",
        help="Parent directory for the JSON and Markdown from each run.",
    )
    args = parser.parse_args()

    if args.live and not os.environ.get("OPENAI_API_KEY", "").strip():
        parser.error("--live requires OPENAI_API_KEY in the environment.")

    cases_path = args.cases
    if args.live and cases_path is None:
        cases_path = PROJECT_ROOT / "data" / "eval_cases.jsonl"

    try:
        corpus = load_corpus(args.corpus)
        all_cases = (
            load_gold_cases(cases_path) if cases_path is not None else FIXTURE_CASES
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    cases = [case for case in all_cases if case.split == args.split]
    if not cases:
        parser.error(f"The selected case set contains no {args.split} cases.")
    if not args.live:
        unsupported = [
            case.case_id for case in cases if case.question not in FIXTURE_MODELS
        ]
        if unsupported:
            parser.error(
                f"No scripted model for these cases: {', '.join(unsupported)}."
            )

    model_name = os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL

    def make_model(
        question: str, on_usage: Callable[[TokenUsage], None]
    ) -> ModelAdapter:
        if args.live:
            return OpenAIModel(model_name, on_usage=on_usage)
        return FIXTURE_MODELS[question]

    try:
        results = evaluate(cases, corpus, make_model)
    except ValueError as error:
        parser.error(str(error))

    mode = "live" if args.live else "scripted"
    model_label = model_name if args.live else "ScriptedModel fixtures"
    corpus_version = fingerprint(corpus)
    cases_version = fingerprint(all_cases)
    baseline = summarize(results, "baseline")
    graph = summarize(results, "graph")
    export = {
        "mode": mode,
        "model": model_label,
        "split": args.split,
        "corpus_sha256": corpus_version,
        "case_set_sha256": cases_version,
        "first_pass_context_cap": MAX_PASSAGES_PER_PASS,
        "case_count": len(results),
        "summary": {"baseline": baseline, "graph": graph},
        "cases": [result.model_dump(mode="json") for result in results],
    }
    report = render_report(
        results,
        baseline,
        graph,
        run_type="live evaluation" if args.live else "scripted fixture checks",
        model=model_label,
        split=args.split,
        corpus_version=corpus_version,
        cases_version=cases_version,
        context_cap=MAX_PASSAGES_PER_PASS,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix=f"{mode}-{args.split}-", dir=args.output_dir)
    )
    (directory / "results.json").write_text(
        json.dumps(export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "summary.md").write_text(report, encoding="utf-8")
    print(f"Exported {len(results)} paired cases to {directory}")


if __name__ == "__main__":
    main()
