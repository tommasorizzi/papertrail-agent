from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock

import pytest

from papertrail.citations import CitationFailure
from papertrail.graph import build_graph
from papertrail.models import ScriptedModel
from papertrail.retrieval import SearchHit, load_corpus, search
from papertrail.schemas import Draft, Finding, Passage, Response, SearchPlan

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.jsonl"
QUESTION = "What did the copper mesh insert change?"
COPPER_PLAN = SearchPlan(
    queries=["copper mesh"], evidence_needed=["Effect of the mesh on melting time"]
)
COPPER_FINDING = Finding(
    claim="The copper mesh shortened melting time.",
    passage_id="pcm-copper-mesh-02",
    quote="The conductive copper mesh insert shortened melting time relative to plain paraffin.",
)
GRAPHITE_FINDING = Finding(
    claim="Graphite increased conductivity.",
    passage_id="pcm-graphite-01",
    quote="Expanded graphite increased the effective thermal conductivity of the paraffin composite.",
)
UNSUPPORTED_FINDING = Finding(
    claim="The insert tripled the latent heat.",
    passage_id=COPPER_FINDING.passage_id,
    quote="The insert tripled the latent heat.",
)


@pytest.fixture
def passages() -> list[Passage]:
    return load_corpus(CORPUS_PATH)


def test_answers_on_first_pass_with_whitespace_normalized_quote(
    passages: list[Passage],
) -> None:
    finding = COPPER_FINDING.model_copy(
        update={"quote": "The conductive\tcopper mesh insert\nshortened melting time"}
    )
    output = Draft(findings=[finding], unresolved=[], follow_up_query="graphite")
    model = ScriptedModel(COPPER_PLAN.model_dump(), output.model_dump())
    graph = build_graph(passages, model)

    state = graph.invoke({"question": QUESTION})

    assert state["response"] == Response(status="answered", findings=[finding])
    assert state["citation_failures"] == []
    assert state["retrieved"][finding.passage_id] in passages
    assert len(state["search_passes"]) == 1
    assert state["retry_count"] == 0
    assert state["model_calls"] == 2
    assert [timing["node"] for timing in state["node_timings"]] == [
        "plan",
        "retrieve",
        "draft",
        "check",
    ]
    assert all(timing["elapsed_ms"] >= 0 for timing in state["node_timings"])

    next_run = graph.invoke({"question": QUESTION})
    assert next_run["response"] == state["response"]
    assert next_run["model_calls"] == 2
    assert len(next_run["search_passes"]) == 1


def test_follow_up_retrieves_missing_evidence_and_answers(
    passages: list[Passage],
) -> None:
    plan = SearchPlan(
        queries=["graphite"],
        evidence_needed=["Effects of graphite and copper mesh"],
    )
    initial = Draft(
        findings=[GRAPHITE_FINDING],
        unresolved=["Effect of the copper mesh"],
        follow_up_query="copper mesh",
    )
    repaired = Draft(findings=[GRAPHITE_FINDING, COPPER_FINDING], unresolved=[])
    graph = build_graph(passages, ScriptedModel(plan, initial, repaired))

    state = graph.invoke(
        {"question": "What effects did graphite and copper mesh have?"}
    )

    first_pass, repair_pass = state["search_passes"]
    assert first_pass["queries"] == ["graphite"]
    assert first_pass["passage_ids"] == [GRAPHITE_FINDING.passage_id]
    assert repair_pass["queries"] == ["copper mesh"]
    assert COPPER_FINDING.passage_id in repair_pass["passage_ids"]
    assert GRAPHITE_FINDING.passage_id in repair_pass["passage_ids"]
    assert state["repair_feedback"].status == "partial"
    assert state["response"] == Response(
        status="answered", findings=[GRAPHITE_FINDING, COPPER_FINDING]
    )
    assert state["retry_count"] == 1
    assert state["model_calls"] == 3


@pytest.mark.parametrize(
    ("bad_finding", "reason"),
    [
        (
            COPPER_FINDING.model_copy(update={"passage_id": "invented-id"}),
            "passage_not_retrieved",
        ),
        (
            Finding(
                claim="Salt settled during cycling.",
                passage_id="pcm-cycling-01",
                quote="salt crystals to settle",
            ),
            "passage_not_retrieved",
        ),
        (UNSUPPORTED_FINDING, "quote_not_found"),
    ],
)
def test_drops_unverified_findings_and_returns_partial(
    bad_finding: Finding, reason: str, passages: list[Passage]
) -> None:
    draft = Draft(findings=[COPPER_FINDING, bad_finding], unresolved=[])
    graph = build_graph(passages, ScriptedModel(COPPER_PLAN, draft))

    state = graph.invoke({"question": QUESTION})

    response = state["response"]
    assert response.status == "partial"
    assert response.findings == [COPPER_FINDING]
    assert bad_finding.claim in response.unresolved[0]
    assert state["citation_failures"] == [CitationFailure(bad_finding, reason)]
    assert state["retry_count"] == 0


def test_unrelated_question_returns_insufficient_evidence(
    passages: list[Passage],
) -> None:
    plan = SearchPlan(
        queries=["orbital period Europa"], evidence_needed=["Orbital period"]
    )
    graph = build_graph(
        passages, ScriptedModel(plan, Draft(findings=[], unresolved=[]))
    )

    state = graph.invoke({"question": "What is the orbital period of Europa?"})

    assert state["retrieved"] == {}
    assert state["response"] == Response(
        status="insufficient_evidence",
        unresolved=["This collection does not provide enough evidence."],
    )
    assert state["retry_count"] == 0


def test_caps_searches_model_calls_and_context_on_persistent_citation_failure(
    passages: list[Passage], monkeypatch: pytest.MonkeyPatch
) -> None:
    texts = [
        "copper",
        "copper mesh",
        "copper graphite",
        "silver fins",
        "silver graphite",
        "silver enclosure",
        "graphite conductivity",
        "graphite composite",
        *["wax discharge"] * 6,
    ]
    corpus = [
        passages[0].model_copy(update={"passage_id": f"sample-{i:02d}", "text": text})
        for i, text in enumerate(texts)
    ]
    queries = ["copper", "silver", "graphite"]
    plan = SearchPlan(
        queries=queries, evidence_needed=["Materials used in the fixtures"]
    )
    earlier_finding = Finding(
        claim="A fixture mentions copper.", passage_id="sample-00", quote="copper"
    )
    initial = Draft(
        findings=[UNSUPPORTED_FINDING], unresolved=[], follow_up_query="wax"
    )
    repaired = Draft(
        findings=[earlier_finding, UNSUPPORTED_FINDING],
        unresolved=[],
        follow_up_query="salt",
    )
    model = Mock(wraps=ScriptedModel(plan, initial, repaired))
    search_calls: list[str] = []

    def record_search(
        query: str, passages: Sequence[Passage], *, limit: int
    ) -> list[SearchHit]:
        search_calls.append(query)
        assert limit == 6
        return search(query, passages, limit=limit)

    monkeypatch.setattr("papertrail.graph.search", record_search)
    state = build_graph(corpus, model).invoke(
        {"question": "Which materials were used?"}
    )

    assert search_calls == [*queries, "wax"]
    first_ids, repair_ids = [
        attempt["passage_ids"] for attempt in state["search_passes"]
    ]
    assert len(first_ids) == len(set(first_ids)) == 6
    assert len(repair_ids) == len(set(repair_ids)) == 6
    assert earlier_finding.passage_id in first_ids
    assert earlier_finding.passage_id not in repair_ids
    assert model.plan.call_count == 1
    assert model.draft.call_count == 2
    assert [len(call.args[2]) for call in model.draft.call_args_list] == [6, 6]
    assert state["model_calls"] == 3
    assert state["retry_count"] == 1
    assert len(state["citation_failures"]) == 2
    assert state["response"].status == "partial"
    assert state["response"].findings == [earlier_finding]


@pytest.mark.parametrize(
    ("step", "scripted_output", "message", "calls"),
    [
        (
            "plan_output",
            COPPER_PLAN.model_copy(
                update={"queries": ["copper", "wax", "graphite", "salt"]}
            ),
            "invalid output",
            1,
        ),
        (
            "draft_output",
            {"findings": [{"claim": "No citation fields"}], "unresolved": []},
            "invalid output",
            2,
        ),
        (
            "draft_output",
            {
                "findings": [
                    {
                        **COPPER_FINDING.model_dump(),
                        "source_url": "https://example.invalid/invented-source",
                    }
                ],
                "unresolved": [],
            },
            "invalid output",
            2,
        ),
        ("plan_output", TimeoutError("private detail"), "timed out", 1),
        ("draft_output", TimeoutError("private detail"), "timed out", 2),
        ("repair_output", TimeoutError("private detail"), "timed out", 3),
        ("draft_output", RuntimeError("private detail"), "call failed", 2),
    ],
)
def test_model_failures_return_error_and_stop(
    step: str,
    scripted_output: object,
    message: str,
    calls: int,
    passages: list[Passage],
) -> None:
    script: dict[str, object] = {
        "plan_output": COPPER_PLAN,
        "draft_output": Draft(
            findings=[], unresolved=["Heat storage effect"], follow_up_query="graphite"
        ),
    }
    script[step] = scripted_output
    graph = build_graph(passages, ScriptedModel(**script))

    state = graph.invoke({"question": QUESTION})

    response = state["response"]
    assert response.status == "error"
    assert response.error is not None and message in response.error
    assert "private detail" not in response.error
    assert response.findings == []
    assert state["model_calls"] == calls
    assert len(state["search_passes"]) == calls - 1
    assert state["retry_count"] == (1 if step == "repair_output" else 0)
