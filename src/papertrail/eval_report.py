from collections import Counter
from html import escape
from statistics import fmean, median
from typing import Literal, TypedDict

from papertrail.evaluation import CaseResult


class RunSummary(TypedDict):
    first_pass_recall_at_6: float | None
    recall_case_count: int
    cases_with_retrieval_miss: int
    cases_with_retry_recovery: int
    cases_with_uncited_relevant_evidence: int
    cases_with_cited_distractors: int
    expected_status_counts: dict[str, int]
    correct_status_counts: dict[str, int]
    response_status_counts: dict[str, int]
    false_abstentions: dict[str, int]
    median_latency_ms: float
    total_model_calls: int
    model_calls_per_case: float
    token_usage: dict[str, int] | None


def summarize(
    results: list[CaseResult], path: Literal["baseline", "graph"]
) -> RunSummary:
    runs = [
        result.baseline if path == "baseline" else result.graph for result in results
    ]
    pairs = list(zip((result.case for result in results), runs, strict=True))
    recalls = [
        recall
        for run in runs
        if (recall := run.evidence.first_pass_recall_at_6) is not None
    ]
    correct = Counter(
        case.expected_status
        for case, run in pairs
        if case.expected_status == run.response.status
    )
    false_abstentions = Counter(
        case.expected_status
        for case, run in pairs
        if run.response.status == "insufficient_evidence"
    )
    model_calls = sum(run.model_calls for run in runs)
    usage = [tokens for run in runs for tokens in run.token_usage]

    return RunSummary(
        first_pass_recall_at_6=fmean(recalls) if recalls else None,
        recall_case_count=len(recalls),
        cases_with_retrieval_miss=sum(
            bool(run.evidence.missing_final_ids) for run in runs
        ),
        cases_with_retry_recovery=sum(bool(run.evidence.recovered_ids) for run in runs),
        cases_with_uncited_relevant_evidence=sum(
            bool(run.evidence.uncited_retrieved_relevant_ids) for run in runs
        ),
        cases_with_cited_distractors=sum(
            bool(run.evidence.cited_distractor_ids) for run in runs
        ),
        expected_status_counts=dict(
            Counter(result.case.expected_status for result in results)
        ),
        correct_status_counts=dict(correct),
        response_status_counts=dict(Counter(run.response.status for run in runs)),
        false_abstentions={
            "answered": false_abstentions["answered"],
            "partial": false_abstentions["partial"],
        },
        median_latency_ms=median(run.latency_ms for run in runs),
        total_model_calls=model_calls,
        model_calls_per_case=model_calls / len(runs),
        token_usage={
            "calls_with_usage": len(usage),
            "input_tokens": sum(tokens.input_tokens for tokens in usage),
            "output_tokens": sum(tokens.output_tokens for tokens in usage),
            "total_tokens": sum(tokens.total_tokens for tokens in usage),
        }
        if usage
        else None,
    )


def _recall(summary: RunSummary) -> str:
    recall = summary["first_pass_recall_at_6"]
    return (
        "Unavailable"
        if recall is None
        else f"{recall:.1%} ({summary['recall_case_count']})"
    )


def _status_rate(
    summary: RunSummary, status: str, *, false_abstentions: bool = False
) -> str:
    counts = (
        summary["false_abstentions"]
        if false_abstentions
        else summary["correct_status_counts"]
    )
    return f"{counts.get(status, 0)}/{summary['expected_status_counts'].get(status, 0)}"


def _token_count(summary: RunSummary) -> str:
    usage = summary["token_usage"]
    if usage is None:
        return "Unavailable"
    return f"{usage['total_tokens']} across {usage['calls_with_usage']} calls"


def _format_metrics(summary: RunSummary) -> dict[str, str | int]:
    return {
        "Mean first-pass Recall@6 (cases)": _recall(summary),
        "Answered on fully answerable cases": _status_rate(summary, "answered"),
        "Correct abstentions on unsupported cases": _status_rate(
            summary, "insufficient_evidence"
        ),
        "Correct partial statuses": _status_rate(summary, "partial"),
        "False abstentions on fully answerable cases": _status_rate(
            summary, "answered", false_abstentions=True
        ),
        "False abstentions on partially answerable cases": _status_rate(
            summary, "partial", false_abstentions=True
        ),
        "Cases missing labeled passages after search": summary[
            "cases_with_retrieval_miss"
        ],
        "Cases with passages recovered on retry": summary["cases_with_retry_recovery"],
        "Cases with retrieved labeled passages left uncited": summary[
            "cases_with_uncited_relevant_evidence"
        ],
        "Cases citing a labeled distractor": summary["cases_with_cited_distractors"],
        "Errors": summary["response_status_counts"].get("error", 0),
        "Median latency (ms)": f"{summary['median_latency_ms']:.2f}",
        "Model calls per case": f"{summary['model_calls_per_case']:.2f}",
        "Total tokens reported": _token_count(summary),
    }


def _ids(ids: list[str] | None) -> str:
    if ids is None:
        return "Unavailable"
    return ", ".join(escape(value).replace("|", "&#124;") for value in ids) or "None"


def _example_delta(result: CaseResult) -> int:
    relevant = set(result.case.relevant_passage_ids)
    counts = []
    for run in (result.baseline, result.graph):
        retrieved = {
            passage_id
            for attempt in run.search_passes
            for passage_id in attempt["passage_ids"]
        }
        counts.append(len(relevant & retrieved))
    baseline_count, graph_count = counts
    return graph_count - baseline_count


def render_report(
    results: list[CaseResult],
    baseline: RunSummary,
    graph: RunSummary,
    *,
    run_type: str,
    model: str,
    split: str,
    corpus_version: str,
    cases_version: str,
    context_cap: int,
) -> str:
    lines = [
        f"# PaperTrail {run_type}",
        "",
        f"{len(results)} paired cases · {split} split · {escape(model)} · context cap {context_cap}",
        "",
        f"Corpus SHA-256 `{corpus_version}`",
        "",
        f"Case-set SHA-256 `{cases_version}`",
        "",
        "| Metric | Single query | Graph |",
        "| --- | ---: | ---: |",
    ]
    graph_metrics = _format_metrics(graph)
    lines.extend(
        f"| {metric} | {value} | {graph_metrics[metric]} |"
        for metric, value in _format_metrics(baseline).items()
    )
    lines.extend(
        [
            "",
            (
                "Recall averages cases with relevant labels and a completed first search. "
                "Uncited passages and distractor citations flag cases for human review. "
                "Claim support requires a separate human rating."
            ),
            "",
            "## Example cases",
        ]
    )
    largest_gain = max(results, key=_example_delta)
    examples = [largest_gain]
    if len(results) > 1:
        examples.append(
            min(
                (result for result in results if result is not largest_gain),
                key=_example_delta,
            )
        )
    for result in examples:
        lines.extend(
            [
                "",
                f"### {escape(result.case.case_id)}",
                "",
                escape(result.case.question),
                "",
                "| Path | Status | Missing after first search | Missing after final search | Recovered on retry | Uncited retrieved labels | Calls |",
                "| --- | --- | --- | --- | --- | --- | ---: |",
            ]
        )
        for name, run in (("Single query", result.baseline), ("Graph", result.graph)):
            score = run.evidence
            lines.append(
                f"| {name} | {run.response.status} | {_ids(score.missing_first_pass_ids)} | "
                f"{_ids(score.missing_final_ids)} | {_ids(score.recovered_ids)} | "
                f"{_ids(score.uncited_retrieved_relevant_ids)} | {run.model_calls} |"
            )
    return "\n".join(lines) + "\n"
