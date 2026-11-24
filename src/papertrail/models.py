from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ValidationError

from papertrail.schemas import Passage, Response, SearchPlan


def call_model[Output: BaseModel](
    call: Callable[[], object], schema: type[Output], stage: str
) -> Output | Response:
    """Validate a model call and expose a sanitized error on failure."""
    try:
        raw = call()
    except TimeoutError:
        message = f"The model timed out while {stage}."
    except Exception:  # noqa: BLE001 - adapters can raise provider-specific errors.
        message = f"The model call failed while {stage}."
    else:
        try:
            return schema.model_validate(raw)
        except ValidationError:
            message = f"The model returned invalid output while {stage}."
    return Response(status="error", error=message)


class ModelAdapter(Protocol):
    """Return parsed model data for the graph to validate."""

    def plan(self, question: str) -> object: ...

    def draft(
        self,
        question: str,
        plan: SearchPlan,
        passages: Sequence[Passage],
        *,
        repair_feedback: Response | None = None,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class ScriptedModel:
    """Replay the configured outputs on each run, raising any scripted exception."""

    plan_output: object
    draft_output: object
    repair_output: object = None

    def plan(self, question: str) -> object:
        return self._replay(self.plan_output)

    def draft(
        self,
        question: str,
        plan: SearchPlan,
        passages: Sequence[Passage],
        *,
        repair_feedback: Response | None = None,
    ) -> object:
        output = self.draft_output if repair_feedback is None else self.repair_output
        return self._replay(output)

    @staticmethod
    def _replay(output: object) -> object:
        if isinstance(output, Exception):
            raise output
        return output
