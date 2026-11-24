from papertrail.demo import (
    COPPER_QUESTION,
    DEMO_MODELS,
    EUROPA_QUESTION,
    GRAPHITE_QUESTION,
)
from papertrail.evaluation import GoldCase
from papertrail.models import ScriptedModel
from papertrail.schemas import Draft, Finding, SearchPlan

_PARAPHRASE = (
    "How was the comparison kept fair, and what was the cost of speeding up "
    "the transition from solid to liquid?"
)
_CONTROL_FINDING = Finding(
    claim="The control used plain paraffin at the same wall temperature.",
    passage_id="pcm-copper-mesh-01",
    quote="The control enclosure used plain paraffin at the same wall temperature.",
)
_TRADEOFF_FINDING = Finding(
    claim="Displacing wax reduced the latent heat stored per unit volume.",
    passage_id="pcm-copper-mesh-02",
    quote="Because the insert displaced wax, the enclosure stored less latent heat per unit volume.",
)

FIXTURE_MODELS = {
    **DEMO_MODELS,
    _PARAPHRASE: ScriptedModel(
        plan_output=SearchPlan(
            queries=["control wall temperature"],
            evidence_needed=["Comparison conditions", "Heat storage trade-off"],
        ),
        draft_output=Draft(
            findings=[_CONTROL_FINDING],
            unresolved=["The cost of faster melting."],
            follow_up_query="displaced wax latent heat",
        ),
        repair_output=Draft(
            findings=[_CONTROL_FINDING, _TRADEOFF_FINDING], unresolved=[]
        ),
    ),
}

FIXTURE_CASES = [
    GoldCase(
        case_id="fixture-copper",
        split="holdout",
        question=COPPER_QUESTION,
        expected_status="answered",
        expected_findings=[
            "Conductive copper mesh was embedded in paraffin.",
            "The mesh shortened melting time but reduced latent heat stored per unit volume.",
        ],
        expected_unresolved=[],
        relevant_passage_ids=["pcm-copper-mesh-01", "pcm-copper-mesh-02"],
    ),
    GoldCase(
        case_id="fixture-graphite",
        split="holdout",
        question=GRAPHITE_QUESTION,
        expected_status="partial",
        expected_findings=[
            "Expanded graphite increased effective thermal conductivity."
        ],
        expected_unresolved=["Long-term cycling durability of the graphite composite."],
        relevant_passage_ids=["pcm-graphite-01"],
    ),
    GoldCase(
        case_id="fixture-europa",
        split="holdout",
        question=EUROPA_QUESTION,
        expected_status="insufficient_evidence",
        expected_findings=[],
        expected_unresolved=["The orbital period of Europa."],
        relevant_passage_ids=[],
    ),
    GoldCase(
        case_id="fixture-paraphrase",
        split="holdout",
        question=_PARAPHRASE,
        expected_status="answered",
        expected_findings=[_CONTROL_FINDING.claim, _TRADEOFF_FINDING.claim],
        expected_unresolved=[],
        relevant_passage_ids=["pcm-copper-mesh-01", "pcm-copper-mesh-02"],
        distractor_passage_ids=["pcm-aluminium-fins-01"],
    ),
]
