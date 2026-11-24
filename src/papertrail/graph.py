from collections.abc import Callable, Sequence
from functools import wraps
from operator import add
from time import perf_counter
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from papertrail.citations import CitationFailure, check_draft
from papertrail.models import ModelAdapter, call_model
from papertrail.retrieval import SearchHit, search
from papertrail.schemas import Draft, Passage, Response, SearchPlan

MAX_PASSAGES_PER_PASS = 6


class GraphInput(TypedDict):
    question: str


class SearchPass(TypedDict):
    queries: list[str]
    passage_ids: list[str]


class NodeTiming(TypedDict):
    node: str
    elapsed_ms: float


class GraphState(TypedDict, total=False):
    question: str
    search_plan: SearchPlan
    queries: list[str]
    retrieved: dict[str, Passage]
    search_passes: Annotated[list[SearchPass], add]
    latest_draft: Draft
    citation_failures: Annotated[list[CitationFailure], add]
    repair_feedback: Response
    retry_count: int
    model_calls: Annotated[int, add]
    node_timings: Annotated[list[NodeTiming], add]
    response: Response


def _timed(
    node: Callable[[GraphState], GraphState],
) -> Callable[[GraphState], GraphState]:
    @wraps(node)
    def run(state: GraphState) -> GraphState:
        started = perf_counter()
        update = node(state)
        elapsed_ms = (perf_counter() - started) * 1000
        update["node_timings"] = [{"node": node.__name__, "elapsed_ms": elapsed_ms}]
        return update

    return run


def _has_response(state: GraphState) -> bool:
    return "response" in state


def build_graph(
    passages: Sequence[Passage], model: ModelAdapter
) -> CompiledStateGraph[GraphState, None, GraphInput, GraphState]:
    """Compile the workflow. Invoke with a question and read the state's response."""
    corpus = tuple(passages)

    @_timed
    def plan(state: GraphState) -> GraphState:
        question = state.get("question")
        if not isinstance(question, str) or not question.strip():
            return {
                "response": Response(
                    status="error", error="Enter a non-blank question."
                ),
                "retry_count": 0,
            }

        output = call_model(
            lambda: model.plan(question), SearchPlan, "planning the search"
        )
        if isinstance(output, Response):
            return {"response": output, "retry_count": 0, "model_calls": 1}
        return {
            "search_plan": output,
            "queries": list(output.queries),
            "retry_count": 0,
            "model_calls": 1,
        }

    @_timed
    def retrieve(state: GraphState) -> GraphState:
        queries = state["queries"]
        best_hits: dict[str, SearchHit] = {}
        for query in queries:
            for hit in search(query, corpus, limit=MAX_PASSAGES_PER_PASS):
                passage_id = hit.passage.passage_id
                previous = best_hits.get(passage_id)
                if previous is None or hit.score > previous.score:
                    best_hits[passage_id] = hit

        ranked = sorted(
            best_hits.values(), key=lambda hit: (-hit.score, hit.passage.passage_id)
        )
        retrieved = {
            hit.passage.passage_id: hit.passage
            for hit in ranked[:MAX_PASSAGES_PER_PASS]
        }
        earlier = state.get("retrieved", {})
        # Give follow-up hits priority and use earlier passages to fill the context.
        passage_ids = list(dict.fromkeys([*retrieved, *earlier]))[
            :MAX_PASSAGES_PER_PASS
        ]
        return {
            "retrieved": earlier | retrieved,
            "search_passes": [{"queries": queries, "passage_ids": passage_ids}],
        }

    @_timed
    def draft(state: GraphState) -> GraphState:
        passage_ids = state["search_passes"][-1]["passage_ids"]
        retrieved = state["retrieved"]
        context = tuple(retrieved[passage_id] for passage_id in passage_ids)
        output = call_model(
            lambda: model.draft(
                state["question"],
                state["search_plan"],
                context,
                repair_feedback=state.get("repair_feedback"),
            ),
            Draft,
            "drafting findings",
        )
        if isinstance(output, Response):
            return {"response": output, "model_calls": 1}
        return {"latest_draft": output, "model_calls": 1}

    @_timed
    def check(state: GraphState) -> GraphState:
        latest_draft = state["latest_draft"]
        response, failures = check_draft(latest_draft, state["retrieved"].values())
        update: GraphState = {"citation_failures": failures}
        if (
            response.status != "answered"
            and state["retry_count"] == 0
            and latest_draft.follow_up_query is not None
        ):
            update["queries"] = [latest_draft.follow_up_query]
            update["retry_count"] = 1
            update["repair_feedback"] = response
        else:
            update["response"] = response
        return update

    graph = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("draft", draft)
    graph.add_node("check", check)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _has_response, {True: END, False: "retrieve"})
    graph.add_edge("retrieve", "draft")
    graph.add_conditional_edges("draft", _has_response, {True: END, False: "check"})
    graph.add_conditional_edges("check", _has_response, {True: END, False: "retrieve"})
    return graph.compile()
