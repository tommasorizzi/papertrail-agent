import os
from html import escape
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import streamlit as st

from papertrail.citations import exact_quote
from papertrail.demo import DEMO_MODELS
from papertrail.graph import MAX_PASSAGES_PER_PASS, GraphState, build_graph
from papertrail.live import DEFAULT_OPENAI_MODEL, OpenAIModel
from papertrail.models import ModelAdapter
from papertrail.retrieval import load_corpus
from papertrail.schemas import Finding, Passage

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus.jsonl"
DEMO_MODE = "Scripted demo"
LIVE_MODE = "Live"

STYLES = """
<style>
.stMainBlockContainer {
    max-width: 54rem;
    padding-top: 3.75rem;
}
.papertrail-question, .papertrail-finding, .papertrail-source-text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.papertrail-question {
    font-size: 1.13rem;
    font-weight: 600;
}
.papertrail-finding p {
    margin-bottom: .15rem;
}
.papertrail-finding a, .papertrail-meta {
    font-size: .875rem;
}
.papertrail-finding a {
    color: #21685a;
}
.papertrail-meta {
    color: #53695f;
}
.papertrail-source {
    scroll-margin-top: 4rem;
}
.papertrail-quote {
    border-inline-start: .2rem solid #749b86;
    margin-inline: 0;
    padding-inline-start: .9rem;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
</style>
"""


class RunResult(TypedDict):
    state: GraphState
    elapsed_ms: float


def run_question(question: str, model: ModelAdapter) -> RunResult:
    graph = build_graph(load_corpus(CORPUS_PATH), model)
    started = perf_counter()
    state = graph.invoke({"question": question})
    return {"state": state, "elapsed_ms": (perf_counter() - started) * 1000}


def demo_form() -> RunResult:
    st.caption("Choose one of three scripted questions from the synthetic corpus demo.")
    with st.form("demo"):
        question = st.selectbox(
            "Example question", list(DEMO_MODELS), key="demo_question"
        )
        submitted = st.form_submit_button("Run demo", type="primary")

    if submitted or "demo_result" not in st.session_state:
        st.session_state["demo_result"] = run_question(question, DEMO_MODELS[question])
    return st.session_state["demo_result"]


def live_form() -> RunResult | None:
    st.caption(
        "Live mode sends the question and selected synthetic passages to OpenAI."
    )
    configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if not configured:
        st.info(
            "Set OPENAI_API_KEY in your shell and restart PaperTrail to enable live mode."
        )

    with st.form("live"):
        question = st.text_area(
            "Research question",
            placeholder="What does the collection report about heat transfer?",
            key="live_question",
        )
        submitted = st.form_submit_button(
            "Search collection", type="primary", disabled=not configured
        )

    if submitted and configured:
        if question.strip():
            name = os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
            with st.spinner("Searching the collection…"):
                st.session_state["live_result"] = run_question(
                    question.strip(), OpenAIModel(name)
                )
        else:
            st.warning("Enter a research question to search the collection.")
    return st.session_state.get("live_result")


def render_sources(
    retrieved: dict[str, Passage], findings: list[Finding], anchors: dict[str, str]
) -> None:
    st.subheader("Evidence", anchor=False)
    for passage_id, anchor in anchors.items():
        passage = retrieved[passage_id]
        quotes = dict.fromkeys(
            exact_quote(passage.text, finding.quote)
            for finding in findings
            if finding.passage_id == passage_id
        )
        with st.container(border=True):
            st.html(
                f'<h3 class="papertrail-source" id="{anchor}">{escape(passage.title)}</h3>'
                f'<p class="papertrail-meta">{escape(passage.section)} · {escape(passage_id)}</p>'
            )
            for quote in quotes:
                if quote is not None:
                    st.html(
                        f'<blockquote class="papertrail-quote">{escape(quote)}</blockquote>'
                    )
            with st.expander("View full passage"):
                st.html(f'<p class="papertrail-source-text">{escape(passage.text)}</p>')
                st.html(
                    f'<p class="papertrail-meta">{escape(passage.source_terms)}</p>'
                )
                host = passage.source_url.host or ""
                if host != "invalid" and not host.endswith(".invalid"):
                    st.link_button("Open source", str(passage.source_url))


def render_run_details(result: RunResult) -> None:
    state = result["state"]
    with st.expander("Run details"):
        passages, calls, retries, elapsed = st.columns(4)
        passages.metric("Passages", len(state.get("retrieved", {})))
        calls.metric("Model calls", state.get("model_calls", 0))
        retries.metric("Retries", state.get("retry_count", 0))
        elapsed.metric("Elapsed", f"{result['elapsed_ms']:.0f} ms")

        for number, attempt in enumerate(state.get("search_passes", [])):
            label = "Initial search" if number == 0 else "Follow-up search"
            count = len(attempt["passage_ids"])
            st.markdown(f"**{label}** · {count} passages")
            st.text("\n".join(attempt["queries"]))

        if state.get("node_timings"):
            st.table(
                [
                    {
                        "Step": timing["node"].capitalize(),
                        "Time (ms)": round(timing["elapsed_ms"], 1),
                    }
                    for timing in state["node_timings"]
                ]
            )


def render_result(result: RunResult, mode: str) -> None:
    state = result["state"]
    response = state["response"]
    st.subheader("Answer", anchor=False)
    st.caption(f"{mode} result")
    st.html(f'<p class="papertrail-question">{escape(state["question"])}</p>')
    if response.status == "answered":
        st.success("Answered")
    elif response.status == "partial":
        st.warning("Partial answer")
    elif response.status == "insufficient_evidence":
        st.info("Insufficient evidence")
    else:
        st.error(response.error or "The search could not complete.")
        if mode == LIVE_MODE:
            st.caption(
                "Check the API key, model configuration, or connection and try again."
            )

    retrieved = state.get("retrieved", {})
    cited_ids = list(dict.fromkeys(finding.passage_id for finding in response.findings))
    anchors = {
        passage_id: f"passage-{number}"
        for number, passage_id in enumerate(cited_ids[:MAX_PASSAGES_PER_PASS], start=1)
    }
    for finding in response.findings:
        if finding.passage_id in anchors:
            passage = retrieved[finding.passage_id]
            st.html(
                f'<div class="papertrail-finding"><p>{escape(finding.claim)}</p>'
                f'<a href="#{anchors[finding.passage_id]}">'
                f"{escape(passage.title)} · {escape(passage.section)}</a></div>"
            )
    if len(anchors) < len(cited_ids):
        st.caption(
            f"Showing findings from {len(anchors)} of {len(cited_ids)} cited passages."
        )

    if response.unresolved:
        st.subheader("Unresolved", anchor=False)
        items = "".join(f"<li>{escape(item)}</li>" for item in response.unresolved)
        st.html(f"<ul>{items}</ul>")

    if anchors:
        render_sources(retrieved, response.findings, anchors)
    render_run_details(result)


def main() -> None:
    st.set_page_config(page_title="PaperTrail")
    st.html(STYLES)
    st.title("PaperTrail", anchor=False)
    st.write(
        "Search the phase change materials collection and inspect the evidence behind each finding."
    )
    mode = st.radio("Mode", (DEMO_MODE, LIVE_MODE), horizontal=True, key="mode")
    result = demo_form() if mode == DEMO_MODE else live_form()
    if result is not None:
        render_result(result, mode)


if __name__ == "__main__":
    main()
