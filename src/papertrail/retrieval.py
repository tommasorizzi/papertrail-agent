from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from papertrail.schemas import Passage


@dataclass(frozen=True, slots=True)
class SearchHit:
    passage: Passage
    score: float


def load_corpus(path: str | Path) -> list[Passage]:
    """Load and validate JSONL records, requiring each passage ID to be unique."""
    passages: list[Passage] = []
    seen_ids: set[str] = set()

    with Path(path).open(encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            try:
                passage = Passage.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid passage record\n{error}"
                ) from error

            if passage.passage_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate passage_id {passage.passage_id!r}"
                )
            seen_ids.add(passage.passage_id)
            passages.append(passage)

    return passages


def search(
    query: str, passages: Sequence[Passage], *, limit: int = 6
) -> list[SearchHit]:
    """Rank passage text by TF-IDF cosine similarity, breaking ties by passage ID."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0 or not passages:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    analyze = vectorizer.build_analyzer()
    if not analyze(query) or not any(analyze(passage.text) for passage in passages):
        return []

    passage_vectors = vectorizer.fit_transform(passage.text for passage in passages)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, passage_vectors)[0]

    hits = [
        SearchHit(passage, float(score))
        for passage, score in zip(passages, scores, strict=True)
        if score > 0
    ]
    hits.sort(key=lambda hit: (-hit.score, hit.passage.passage_id))
    return hits[:limit]
