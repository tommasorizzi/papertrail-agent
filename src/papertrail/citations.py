import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from papertrail.schemas import Draft, Finding, Passage, Response, ResponseStatus


@dataclass(frozen=True, slots=True)
class CitationFailure:
    finding: Finding
    reason: Literal["passage_not_retrieved", "quote_not_found"]


def exact_quote(text: str, quote: str) -> str | None:
    """Return the matching source excerpt, preserving its original whitespace."""
    words = quote.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, text)
    return match.group() if match else None


def check_citations(
    findings: Iterable[Finding], retrieved: Iterable[Passage]
) -> tuple[list[Finding], list[CitationFailure]]:
    """Check retrieved passage membership and verbatim quote text."""
    passage_texts = {passage.passage_id: passage.text for passage in retrieved}
    verified: list[Finding] = []
    failures: list[CitationFailure] = []

    for finding in findings:
        text = passage_texts.get(finding.passage_id)
        if text is None:
            failures.append(CitationFailure(finding, "passage_not_retrieved"))
        elif exact_quote(text, finding.quote) is None:
            failures.append(CitationFailure(finding, "quote_not_found"))
        else:
            verified.append(finding)

    return verified, failures


def check_draft(
    draft: Draft, retrieved: Iterable[Passage]
) -> tuple[Response, list[CitationFailure]]:
    """Filter failed citations and assign a status from the remaining findings."""
    findings, failures = check_citations(draft.findings, retrieved)
    unresolved = list(draft.unresolved)
    for failure in failures:
        unresolved.append(
            f"Could not verify the citation for “{failure.finding.claim}”."
        )

    status: ResponseStatus
    if not findings:
        status = "insufficient_evidence"
        if not unresolved:
            unresolved.append("This collection does not provide enough evidence.")
    elif unresolved:
        status = "partial"
    else:
        status = "answered"

    return Response(status=status, findings=findings, unresolved=unresolved), failures
