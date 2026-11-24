import json
from pathlib import Path

import pytest

from papertrail.retrieval import load_corpus, search
from papertrail.schemas import Passage

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.jsonl"


@pytest.fixture
def passages() -> list[Passage]:
    return load_corpus(CORPUS_PATH)


def test_search_ranks_relevant_passage_first(passages: list[Passage]) -> None:
    hits = search(
        "What trade-off did the conductive copper mesh insert report for melting time "
        "and latent heat storage?",
        passages,
        limit=2,
    )

    assert len(hits) == 2
    assert hits[0].passage.passage_id == "pcm-copper-mesh-02"
    assert hits[0].passage.section == "Results"
    assert hits[0].score >= hits[1].score > 0


@pytest.mark.parametrize(
    "query", ["What is the orbital period of Europa?", " ", "the and"]
)
def test_search_returns_no_hits_without_matching_terms(
    query: str, passages: list[Passage]
) -> None:
    assert search(query, passages) == []


def test_search_breaks_ties_by_id_and_discards_zero_scores(
    passages: list[Passage],
) -> None:
    passage = passages[0]
    corpus = [
        passage.model_copy(update={"passage_id": "b", "text": "copper mesh"}),
        passage.model_copy(update={"passage_id": "a", "text": "copper mesh"}),
        passage.model_copy(update={"passage_id": "c", "text": "paraffin wax"}),
    ]

    hits = search("copper", corpus)

    assert [hit.passage.passage_id for hit in hits] == ["a", "b"]
    assert search("copper", corpus, limit=1) == hits[:1]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("passage_id", None),
        ("passage_id", " "),
        ("passage_id", 1),
        ("source_url", "invalid"),
        ("text", " "),
    ],
)
def test_load_corpus_rejects_invalid_fields(
    field: str, invalid_value: object, passages: list[Passage], tmp_path: Path
) -> None:
    record = passages[0].model_dump(mode="json")
    if invalid_value is None:
        del record[field]
    else:
        record[field] = invalid_value
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"invalid\.jsonl:1: invalid passage record"
    ) as error:
        load_corpus(path)
    assert field in str(error.value).splitlines()


def test_load_corpus_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text("{\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"malformed\.jsonl:1: invalid passage record"):
        load_corpus(path)


def test_load_corpus_rejects_duplicate_ids(
    passages: list[Passage], tmp_path: Path
) -> None:
    passage = passages[0]
    duplicate = passage.model_copy(update={"text": "Different text, same passage ID."})
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        f"{passage.model_dump_json()}\n{duplicate.model_dump_json()}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate\.jsonl:2: duplicate passage_id 'pcm-copper-mesh-01'",
    ):
        load_corpus(path)
