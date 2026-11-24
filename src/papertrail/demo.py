from papertrail.models import ScriptedModel
from papertrail.schemas import Draft, Finding, SearchPlan

COPPER_QUESTION = "Which passages describe a conductive insert in paraffin, and what trade-off do they report?"
GRAPHITE_QUESTION = "How does expanded graphite affect paraffin conductivity and long-term cycling durability?"
EUROPA_QUESTION = "What is the orbital period of Europa?"

_GRAPHITE_FINDING = Finding(
    claim="Expanded graphite increased the thermal conductivity of the paraffin composite.",
    passage_id="pcm-graphite-01",
    quote="Expanded graphite increased the effective thermal conductivity of the paraffin composite.",
)
_GRAPHITE_GAP = (
    "How many charge and discharge cycles the graphite composite can sustain."
)

DEMO_MODELS = {
    COPPER_QUESTION: (
        ScriptedModel(
            plan_output=SearchPlan(
                queries=["conductive copper mesh insert", "displaced wax latent heat"],
                evidence_needed=[
                    "How the insert was used",
                    "Melting and heat storage trade-off",
                ],
            ),
            draft_output=Draft(
                findings=[
                    Finding(
                        claim="A synthetic study embedded a conductive copper mesh insert in paraffin.",
                        passage_id="pcm-copper-mesh-01",
                        quote=(
                            "A conductive copper mesh insert was embedded in paraffin "
                            "phase change material inside a heated enclosure."
                        ),
                    ),
                    Finding(
                        claim=(
                            "The insert shortened melting time but reduced latent heat "
                            "storage per unit volume by displacing wax."
                        ),
                        passage_id="pcm-copper-mesh-02",
                        quote=(
                            "The conductive copper mesh insert shortened melting time relative "
                            "to plain paraffin. Because the insert displaced wax, the enclosure "
                            "stored less latent heat per unit volume."
                        ),
                    ),
                ],
                unresolved=[],
            ),
        )
    ),
    GRAPHITE_QUESTION: (
        ScriptedModel(
            plan_output=SearchPlan(
                queries=["expanded graphite thermal conductivity"],
                evidence_needed=[
                    "Conductivity change",
                    "Cycling durability of the composite",
                ],
            ),
            draft_output=Draft(
                findings=[_GRAPHITE_FINDING],
                unresolved=[_GRAPHITE_GAP],
                follow_up_query="graphite composite charge discharge cycling durability",
            ),
            repair_output=Draft(
                findings=[_GRAPHITE_FINDING], unresolved=[_GRAPHITE_GAP]
            ),
        )
    ),
    EUROPA_QUESTION: ScriptedModel(
        plan_output=SearchPlan(
            queries=["Europa orbital period"], evidence_needed=["Orbital period"]
        ),
        draft_output=Draft(findings=[], unresolved=["The orbital period of Europa."]),
    ),
}
