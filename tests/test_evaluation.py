from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from papertrail.eval_fixtures import FIXTURE_CASES, FIXTURE_MODELS
from papertrail.eval_report import summarize
from papertrail.evaluation import GoldCase, evaluate, load_gold_cases
from papertrail.live import OpenAIModel
from papertrail.models import ScriptedModel
from papertrail.retrieval import load_corpus
from papertrail.schemas import Draft, Finding, Passage, SearchPlan, TokenUsage

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def corpus() -> list[Passage]:
    return load_corpus(DATA_DIR / "corpus.jsonl")


def test_bundled_gold_set_has_the_planned_holdout_split(corpus: list[Passage]) -> None:
    cases = load_gold_cases(DATA_DIR / "eval_cases.jsonl")
    assert sum(case.split == "tune" for case in cases) == 4
    assert Counter(
        case.expected_status for case in cases if case.split == "holdout"
    ) == {
        "answered": 8,
        "partial": 4,
        "insufficient_evidence": 4,
    }
    corpus_ids = {passage.passage_id for passage in corpus}
    assert all(
        corpus_ids.issuperset(case.relevant_passage_ids + case.distractor_passage_ids)
        for case in cases
    )


def test_gold_loader_rejects_same_question_across_splits(tmp_path: Path) -> None:
    case = FIXTURE_CASES[0]
    duplicate = case.model_copy(
        update={
            "case_id": "tuning-copy",
            "split": "tune",
            "question": case.question.upper(),
        }
    )
    path = tmp_path / "cases.jsonl"
    path.write_text(
        f"{case.model_dump_json()}\n{duplicate.model_dump_json()}\n", encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match=r"cases\.jsonl:2: duplicate case ID or question"
    ):
        load_gold_cases(path)


def test_fixture_run_measures_a_first_pass_miss_and_retry_recovery(
    corpus: list[Passage],
) -> None:
    results = evaluate(
        FIXTURE_CASES, corpus, lambda question, _: FIXTURE_MODELS[question]
    )
    paraphrase = next(
        result for result in results if result.case.case_id == "fixture-paraphrase"
    )

    baseline = paraphrase.baseline
    assert len(baseline.search_passes) == baseline.model_calls == 1
    assert baseline.search_passes[0]["queries"] == [paraphrase.case.question]
    assert baseline.evidence.first_pass_recall_at_6 == 0
    assert baseline.citation_failures[0].reason == "passage_not_retrieved"
    assert baseline.response.status == "insufficient_evidence"

    graph = paraphrase.graph
    assert graph.evidence.first_pass_recall_at_6 == 0.5
    assert graph.evidence.recovered_ids == ["pcm-copper-mesh-02"]
    assert graph.evidence.missing_final_ids == []
    assert graph.response.status == "answered"
    assert graph.model_calls == 3
    assert graph.retry_count == 1

    assert summarize(results, "baseline")["token_usage"] is None
    assert summarize(results, "graph")["cases_with_retry_recovery"] == 1


def test_both_paths_supply_the_same_six_passage_cap(corpus: list[Passage]) -> None:
    passages = [
        corpus[0].model_copy(
            update={"passage_id": f"copper-{number}", "text": "Copper mesh."}
        )
        for number in range(8)
    ]
    case = GoldCase(
        case_id="cap",
        split="holdout",
        question="copper",
        expected_status="answered",
        expected_findings=["Copper mesh."],
        expected_unresolved=[],
        relevant_passage_ids=["copper-0"],
    )
    script = ScriptedModel(
        SearchPlan(queries=["copper"], evidence_needed=["Copper mesh"]),
        Draft(
            findings=[
                Finding(
                    claim="Copper mesh.", passage_id="copper-0", quote="Copper mesh."
                )
            ],
            unresolved=[],
        ),
    )
    models = [Mock(wraps=script), Mock(wraps=script)]
    factory = Mock(side_effect=models)
    (result,) = evaluate([case], passages, factory)

    baseline, graph = models
    baseline.plan.assert_not_called()
    graph.plan.assert_called_once_with(case.question)
    baseline.draft.assert_called_once()
    graph.draft.assert_called_once()
    assert baseline.draft.call_args.args[2] == graph.draft.call_args.args[2]
    assert len(baseline.draft.call_args.args[2]) == 6
    assert result.baseline.response == result.graph.response


def test_labels_flag_uncited_relevant_evidence_and_an_accepted_distractor(
    corpus: list[Passage],
) -> None:
    cases = load_gold_cases(DATA_DIR / "eval_cases.jsonl")
    case = next(case for case in cases if case.case_id == "holdout-03")
    model = ScriptedModel(
        SearchPlan(
            queries=["copper mesh melting time"], evidence_needed=["Contact conditions"]
        ),
        Draft(
            findings=[
                Finding(
                    claim="Adding copper mesh consistently shortened melting time.",
                    passage_id="pcm-copper-mesh-02",
                    quote="The conductive copper mesh insert shortened melting time relative to plain paraffin.",
                )
            ],
            unresolved=[],
        ),
    )
    (result,) = evaluate([case], corpus, lambda *_: model)

    for run in (result.baseline, result.graph):
        assert run.evidence.missing_final_ids == []
        assert run.evidence.uncited_retrieved_relevant_ids == ["pcm-mesh-contact-01"]
        assert run.evidence.cited_distractor_ids == ["pcm-copper-mesh-02"]
        assert run.citation_failures == []


def test_openai_usage_is_captured_separately_for_each_path(
    corpus: list[Passage], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = FIXTURE_CASES[0]
    script = FIXTURE_MODELS[case.question]
    session = MagicMock()
    client = session.__enter__.return_value
    details = SimpleNamespace(cached_tokens=5, cache_write_tokens=2)
    usage = SimpleNamespace(
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
        input_tokens_details=details,
        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )

    def complete(
        *, text_format: type[SearchPlan] | type[Draft], **_: object
    ) -> SimpleNamespace:
        parsed = (
            script.plan_output if text_format is SearchPlan else script.draft_output
        )
        return SimpleNamespace(output_parsed=parsed, usage=usage)

    client.responses.parse.side_effect = complete
    monkeypatch.setattr("papertrail.live.OpenAI", Mock(return_value=session))
    (result,) = evaluate(
        [case], corpus, lambda _, on_usage: OpenAIModel("fixture-model", on_usage)
    )

    expected = TokenUsage(
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
        cached_input_tokens=5,
        cache_write_input_tokens=2,
        reasoning_output_tokens=3,
    )
    assert result.baseline.token_usage == [expected]
    assert result.graph.token_usage == [expected, expected]
    assert summarize([result], "graph")["token_usage"] == {
        "calls_with_usage": 2,
        "input_tokens": 40,
        "output_tokens": 20,
        "total_tokens": 60,
    }
