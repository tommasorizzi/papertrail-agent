from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ResponseStatus = Literal["answered", "partial", "insufficient_evidence", "error"]

_MODEL_OUTPUT_CONFIG = ConfigDict(
    strict=True, extra="forbid", revalidate_instances="always"
)


class Passage(BaseModel):
    """A corpus passage with its source metadata."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    passage_id: NonEmptyString
    title: NonEmptyString
    source_url: HttpUrl
    section: NonEmptyString
    source_terms: NonEmptyString
    text: str

    @field_validator("text")
    @classmethod
    def require_nonblank_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("passage text must contain non-whitespace characters")
        return text


class SearchPlan(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    queries: list[NonEmptyString] = Field(min_length=1, max_length=3)
    evidence_needed: list[NonEmptyString] = Field(min_length=1)


class Finding(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    claim: NonEmptyString
    passage_id: NonEmptyString
    quote: NonEmptyString


class Draft(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    findings: list[Finding]
    unresolved: list[NonEmptyString]
    follow_up_query: NonEmptyString | None = None


class Response(BaseModel):
    status: ResponseStatus
    findings: list[Finding] = Field(default_factory=list)
    unresolved: list[NonEmptyString] = Field(default_factory=list)
    error: NonEmptyString | None = None


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
