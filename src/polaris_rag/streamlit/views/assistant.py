from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
from typing import Any, Mapping

import streamlit as st

from polaris_rag.streamlit.api_client import (
    ApiClientConfig,
    ApiClientError,
    ApiTimeoutError,
    NormalizedApiError,
    QueryResponseData,
    RetrievedContextItem,
    query_backend,
)


EXAMPLE_QUERIES: tuple[str, ...] = (
    "We need to renew RDS and transfer ownership. How should this be handled?",
    "Can you confirm the exact new path for my project data?",
    "What is the latest GROMACS version available on CCLake?",
    "How do I check why my Slurm job is stuck in the queue?",
)

_SECTION_LABELS: dict[str, str] = {
    "CLASSIFICATION": "Classification",
    "QUICK ASSESSMENT": "Quick Assessment",
    "ACTION": "Action",
    "ACTION STEPS (HELPDESK)": "Action Steps (Helpdesk)",
    "QUESTIONS TO ASK (ONLY IF NEEDED)": "Questions to Ask",
    "EXAMPLE CUSTOMER REPLY": "Example Customer Reply",
    "SAFETY / POLICY NOTES": "Safety / Policy Notes",
    "REFERENCE KEY": "Reference Key",
}


@dataclass(frozen=True)
class AnswerSection:
    key: str
    heading: str
    body: str


def parse_answer_sections(answer: str) -> list[AnswerSection]:
    text = str(answer or "").strip()
    if not text:
        return []

    sections: list[AnswerSection] = []
    current_key: str | None = None
    current_lines: list[str] = []

    def _flush_section() -> None:
        nonlocal current_key, current_lines
        if current_key is None:
            return
        sections.append(
            AnswerSection(
                key=current_key,
                heading=_SECTION_LABELS.get(current_key, current_key.title()),
                body="\n".join(current_lines).strip(),
            )
        )
        current_key = None
        current_lines = []

    for raw_line in text.splitlines():
        candidate = raw_line.strip().rstrip(":")
        if candidate in _SECTION_LABELS:
            _flush_section()
            current_key = candidate
            continue

        if current_key is None:
            current_key = "RESPONSE"
        current_lines.append(raw_line)

    _flush_section()
    if not sections:
        return [AnswerSection(key="RESPONSE", heading="Response", body=text)]
    return sections


def render_view(
    config: ApiClientConfig,
    *,
    manual_query_constraints: Mapping[str, Any] | None,
    debug_mode: bool,
) -> None:
    st.markdown("## Assistant")
    st.caption("Evidence-first support assistance with source inspection, query diagnostics, and explicit weak-answer states.")

    _render_example_queries()

    prompt_from_buttons = st.session_state.pop("assistant_pending_prompt", None)
    prompt_from_input = st.chat_input("Ask Polaris about HPC support, policies, or documentation...")
    prompt = prompt_from_buttons or prompt_from_input
    if prompt:
        _submit_prompt(
            config,
            prompt=prompt,
            manual_query_constraints=manual_query_constraints,
            debug_mode=debug_mode,
        )

    left_col, right_col = st.columns([7, 3], gap="large")

    with left_col:
        _render_chat_history()

    with right_col:
        _render_diagnostics_for_latest_message(debug_mode=debug_mode)


def render_answer_block(response: QueryResponseData) -> None:
    _render_status_badges(response.answer_status.code, response.answer_status.detail, len(response.context))
    if response.answer_status.code == "no_evidence":
        st.warning(response.answer_status.detail)
    elif response.answer_status.code == "limited_evidence":
        st.info(response.answer_status.detail)

    sections = parse_answer_sections(response.answer)
    if not sections:
        st.markdown("_No answer text was returned._")
        return

    for section in sections:
        if section.key == "RESPONSE":
            st.markdown(section.body)
            continue

        st.markdown(f"#### {section.heading}")
        if section.key in {"EXAMPLE CUSTOMER REPLY", "REFERENCE KEY"}:
            st.code(section.body or "No content returned.", language="text", wrap_lines=True)
        else:
            st.markdown(section.body or "_No content returned._")


def render_error_block(error: NormalizedApiError) -> None:
    if error.kind == "timeout":
        st.error("The request reached the API deadline before a full answer was returned.")
        st.caption("Try a shorter query, disable debug mode, or increase the request timeout.")
    elif error.kind == "network_error":
        st.error("The Streamlit frontend could not reach the backend service.")
        st.caption("Check the API base URL and whether the backend container is running.")
    else:
        st.error("The backend returned an error before the answer could be completed.")
    st.code(error.message, language="text", wrap_lines=True)
    if error.failure_class:
        st.caption(f"Failure class: `{error.failure_class}`")


def render_diagnostics_panel(
    response: QueryResponseData | None,
    error: NormalizedApiError | None,
    *,
    debug_mode: bool,
) -> None:
    st.markdown("### Diagnostics")
    if response is None and error is None:
        st.info("Run a query to inspect evidence, timings, and query interpretation.")
        return

    if response is not None:
        st.metric("Evidence Chunks", len(response.context))
        retrieval_ms = response.timings.retrieval_elapsed_ms if response.timings.retrieval_elapsed_ms is not None else "n/a"
        generation_ms = response.timings.generation_elapsed_ms if response.timings.generation_elapsed_ms is not None else "n/a"
        metric_cols = st.columns(2)
        metric_cols[0].metric("Retrieval", retrieval_ms)
        metric_cols[1].metric("Generation", generation_ms)
        _render_constraints(response.query_constraints)
        _render_context_items(response.context)
        if debug_mode and response.evaluation_metadata:
            with st.expander("Debug Metadata", expanded=False):
                st.json(response.evaluation_metadata)
        return

    if error is not None:
        st.metric("Evidence Chunks", 0)
        st.warning(error.message)
        if error.detail is not None:
            with st.expander("Error Detail", expanded=False):
                st.json(error.detail) if isinstance(error.detail, Mapping) else st.code(str(error.detail), language="text")


def _submit_prompt(
    config: ApiClientConfig,
    *,
    prompt: str,
    manual_query_constraints: Mapping[str, Any] | None,
    debug_mode: bool,
) -> None:
    user_message = {
        "role": "user",
        "content": prompt,
        "created_at": _now_iso(),
    }
    st.session_state.assistant_messages.append(user_message)

    try:
        with st.spinner("Retrieving evidence and drafting an answer..."):
            response = query_backend(
                config,
                prompt,
                query_constraints=manual_query_constraints,
                include_evaluation_metadata=debug_mode,
            )
        assistant_message = {
            "role": "assistant",
            "content": response.answer,
            "query": prompt,
            "response": response,
            "error": None,
            "created_at": _now_iso(),
        }
    except ApiTimeoutError as exc:
        assistant_message = {
            "role": "assistant",
            "content": "",
            "query": prompt,
            "response": None,
            "error": exc.error,
            "created_at": _now_iso(),
        }
    except ApiClientError as exc:
        assistant_message = {
            "role": "assistant",
            "content": "",
            "query": prompt,
            "response": None,
            "error": exc.error,
            "created_at": _now_iso(),
        }

    st.session_state.assistant_messages.append(assistant_message)
    st.session_state.assistant_latest_message = assistant_message


def _render_chat_history() -> None:
    messages = st.session_state.assistant_messages
    if not messages:
        st.markdown("### Start Here")
        st.write(
            "Use the example prompts or ask your own question. The answer panel is structured for quick reading, "
            "while the diagnostics panel keeps the grounding evidence visible."
        )
        return

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
                continue

            response = message.get("response")
            error = message.get("error")
            if response is not None:
                render_answer_block(response)
            elif error is not None:
                render_error_block(error)
            else:
                st.markdown("_No assistant content available._")


def _render_diagnostics_for_latest_message(*, debug_mode: bool) -> None:
    latest_message = st.session_state.get("assistant_latest_message")
    if latest_message is None:
        render_diagnostics_panel(None, None, debug_mode=debug_mode)
        return
    render_diagnostics_panel(
        latest_message.get("response"),
        latest_message.get("error"),
        debug_mode=debug_mode,
    )


def _render_constraints(query_constraints: Mapping[str, Any] | None) -> None:
    st.markdown("#### Interpreted Query Constraints")
    if not query_constraints:
        st.caption("No explicit query constraints were returned for this answer.")
        return

    labels = {
        "query_type": "Query type",
        "system_names": "Systems",
        "partition_names": "Partitions",
        "service_names": "Services",
        "scope_family_names": "Scope families",
        "software_names": "Software",
        "software_versions": "Software versions",
        "module_names": "Modules",
        "toolchain_names": "Toolchains",
        "toolchain_versions": "Toolchain versions",
        "scope_required": "Scope required",
        "version_sensitive_guess": "Version sensitive",
    }

    for key, label in labels.items():
        value = query_constraints.get(key)
        if value in (None, [], ""):
            continue
        if isinstance(value, list):
            display_value = ", ".join(str(item) for item in value)
        else:
            display_value = str(value)
        st.markdown(f"**{label}:** {display_value}")


def _render_context_items(context_items: list[RetrievedContextItem]) -> None:
    st.markdown("#### Retrieved Evidence")
    if not context_items:
        st.caption("No supporting context was returned.")
        return

    for item in context_items:
        score_text = f"{item.score:.4f}" if isinstance(item.score, float) else "n/a"
        source_text = item.source or "unknown"
        with st.expander(f"[{item.rank}] {item.doc_id}", expanded=False):
            _render_inline_badges(
                [
                    (f"Source: {source_text}", "accent"),
                    (f"Score: {score_text}", "muted"),
                ]
            )
            st.code(item.text or "No chunk text returned.", language="text", wrap_lines=True)


def _render_example_queries() -> None:
    action_cols = st.columns([4, 1], gap="small")
    with action_cols[0]:
        st.markdown("### Demo Prompts")
    with action_cols[1]:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.assistant_messages = []
            st.session_state.assistant_latest_message = None
            st.rerun()

    button_cols = st.columns(2, gap="small")
    for idx, query in enumerate(EXAMPLE_QUERIES):
        column = button_cols[idx % 2]
        if column.button(query, key=f"assistant-example-{idx}", use_container_width=True):
            st.session_state.assistant_pending_prompt = query
            st.rerun()


def _render_status_badges(status_code: str, detail: str, evidence_count: int) -> None:
    labels = {
        "grounded": "Grounded",
        "limited_evidence": "Limited Evidence",
        "no_evidence": "No Evidence",
    }
    status_label = labels.get(status_code, status_code.replace("_", " ").title())
    _render_inline_badges(
        [
            (status_label, status_code),
            (f"{evidence_count} evidence chunks", "muted"),
        ]
    )
    st.caption(detail)


def _render_inline_badges(values: list[tuple[str, str]]) -> None:
    rendered = "".join(
        f"<span class='polaris-badge polaris-badge--{html.escape(variant)}'>{html.escape(label)}</span>"
        for label, variant in values
    )
    st.markdown(f"<div class='polaris-badge-row'>{rendered}</div>", unsafe_allow_html=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
