from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from papertrail.citations import CitationFailure, check_draft
from papertrail.graph import MAX_PASSAGES_PER_PASS, SearchPass, build_graph
from papertrail.models import ModelAdapter, call_model
from papertrail.retrieval import search
from papertrail.schemas import (
    Draft,
    NonEmptyString,
    Passage,
    Response,
    SearchPlan,
    TokenUsage,
)

CaseSplit = Literal["tune", "holdout"]
ExpectedStatus = Literal["answered", "partial", "insufficient_evidence"]
type ModelFactory = Callable[[str, Callable[[TokenUsage], None]], ModelAdapter]


class GoldCase(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    case_id: NonEmptyString
    split: CaseSplit
    question: NonEmptyString
    expected_status: ExpectedStatus
    expected_findings: list[NonEmptyString]
    expected_unresolved: list[NonEmptyString]
    relevant_passage_ids: list[NonEmptyString]
    distractor_passage_ids: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        relevant = set(self.relevant_passage_ids)
        distractors = set(self.distractor_passage_ids)
        if len(relevant) != len(self.relevant_passage_ids) or len(distractors) != len(
            self.distractor_passage_ids
        ):
            raise ValueError("passage IDs must be unique within each list")
        if relevant & distractors:
            raise ValueError("a passage cannot be both relevant and a distractor")
        if self.expected_status == "insufficient_evidence":
            if self.expected_findings or relevant:
                raise ValueError(
                    "unsupported cases must have empty findings and relevant IDs"
                )
        elif not self.expected_findings or not relevant:
            raise ValueError(
                "answerable cases must have findings and relevant passage IDs"
            )
        if bool(self.expected_unresolved) != (self.expected_status != "answered"):
            raise ValueError(
                "partial and unsupported cases must list their unresolved points"
            )
        return self


def load_gold_cases(path: str | Path) -> list[GoldCase]:
    """Load labeled JSONL and reject repeated cases or questions across splits."""
    cases: list[GoldCase] = []
    ids: set[str] = set()
    questions: set[str] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                case = GoldCase.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid gold case\n{error}"
                ) from error
            question = " ".join(case.question.casefold().split())
            if case.case_id in ids or question in questions:
                raise ValueError(f"{path}:{line_number}: duplicate case ID or question")
            cases.append(case)
            ids.add(case.case_id)
            questions.add(question)
    return cases


class EvidenceScores(BaseModel):
    first_pass_recall_at_6: float | None
    missing_first_pass_ids: list[str] | None
    missing_final_ids: list[str] | None
    recovered_ids: list[str]
    uncited_retrieved_relevant_ids: list[str] | None
    cited_distractor_ids: list[str]


class EvalRun(BaseModel):
    response: Response
    search_passes: list[SearchPass]
    citation_failures: list[CitationFailure]
    latency_ms: float
    model_calls: int
    retry_count: int
    token_usage: list[TokenUsage]
    evidence: EvidenceScores


class CaseResult(BaseModel):
    case: GoldCase
    baseline: EvalRun
    graph: EvalRun


def _score_evidence(
    case: GoldCase, passes: list[SearchPass], response: Response
) -> EvidenceScores:
    relevant = set(case.relevant_passage_ids)
    first = set(passes[0]["passage_ids"]) if passes else set()
    retrieved = {
        passage_id for attempt in passes for passage_id in attempt["passage_ids"]
    }
    cited = {finding.passage_id for finding in response.findings}

    return EvidenceScores(
        first_pass_recall_at_6=len(first & relevant) / len(relevant)
        if relevant and passes
        else None,
        missing_first_pass_ids=sorted(relevant - first) if passes else None,
        missing_final_ids=sorted(relevant - retrieved) if passes else None,
        recovered_ids=sorted((relevant & retrieved) - first),
        uncited_retrieved_relevant_ids=(
            sorted((retrieved & relevant) - cited)
            if response.status != "error"
            else None
        ),
        cited_distractor_ids=sorted(cited.intersection(case.distractor_passage_ids)),
    )


def _run_baseline(
    case: GoldCase,
    corpus: Sequence[Passage],
    model: ModelAdapter,
    usage: list[TokenUsage],
) -> EvalRun:
    started = perf_counter()
    question = case.question
    context = tuple(
        hit.passage for hit in search(question, corpus, limit=MAX_PASSAGES_PER_PASS)
    )
    plan = SearchPlan(queries=[question], evidence_needed=[question])
    output = call_model(
        lambda: model.draft(question, plan, context), Draft, "drafting findings"
    )
    response, failures = (
        (output, []) if isinstance(output, Response) else check_draft(output, context)
    )
    elapsed_ms = (perf_counter() - started) * 1000
    passes: list[SearchPass] = [
        {
            "queries": [question],
            "passage_ids": [passage.passage_id for passage in context],
        }
    ]

    return EvalRun(
        response=response,
        search_passes=passes,
        citation_failures=failures,
        latency_ms=elapsed_ms,
        model_calls=1,
        retry_count=0,
        token_usage=usage,
        evidence=_score_evidence(case, passes, response),
    )


def _run_graph(
    case: GoldCase,
    corpus: Sequence[Passage],
    model: ModelAdapter,
    usage: list[TokenUsage],
) -> EvalRun:
    graph = build_graph(corpus, model)
    started = perf_counter()
    state = graph.invoke({"question": case.question})
    elapsed_ms = (perf_counter() - started) * 1000
    response = state["response"]
    passes = state["search_passes"]
    return EvalRun(
        response=response,
        search_passes=passes,
        citation_failures=state["citation_failures"],
        latency_ms=elapsed_ms,
        model_calls=state["model_calls"],
        retry_count=state["retry_count"],
        token_usage=usage,
        evidence=_score_evidence(case, passes, response),
    )


def evaluate(
    cases: Sequence[GoldCase], corpus: Sequence[Passage], make_model: ModelFactory
) -> list[CaseResult]:
    """Run both paths on each case with the supplied model configuration."""
    corpus_ids = {passage.passage_id for passage in corpus}
    for case in cases:
        labeled_ids = set(case.relevant_passage_ids) | set(case.distractor_passage_ids)
        missing = labeled_ids - corpus_ids
        if missing:
            raise ValueError(
                f"{case.case_id} refers to unknown passages: {', '.join(sorted(missing))}"
            )

    results: list[CaseResult] = []
    for case in cases:
        baseline_usage: list[TokenUsage] = []
        graph_usage: list[TokenUsage] = []
        baseline_model = make_model(case.question, baseline_usage.append)
        graph_model = make_model(case.question, graph_usage.append)
        results.append(
            CaseResult(
                case=case,
                baseline=_run_baseline(case, corpus, baseline_model, baseline_usage),
                graph=_run_graph(case, corpus, graph_model, graph_usage),
            )
        )
    return results
