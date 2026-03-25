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


def quick_prompt_cards() -> tuple[str, str]:
    return EXAMPLE_QUERIES[:2]


def assistant_view_state(messages: list[dict[str, Any]]) -> str:
    _, latest_assistant, _ = latest_exchange(messages)
    return "active" if latest_assistant is not None else "landing"


def latest_exchange(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    if not messages:
        return None, None, []

    latest_assistant_idx: int | None = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            latest_assistant_idx = idx
            break

    if latest_assistant_idx is None:
        latest_user_idx = next(
            (idx for idx in range(len(messages) - 1, -1, -1) if messages[idx].get("role") == "user"),
            None,
        )
        if latest_user_idx is None:
            return None, None, list(messages)
        return messages[latest_user_idx], None, messages[:latest_user_idx]

    latest_user_idx: int | None = None
    for idx in range(latest_assistant_idx - 1, -1, -1):
        if messages[idx].get("role") == "user":
            latest_user_idx = idx
            break

    latest_user = messages[latest_user_idx] if latest_user_idx is not None else None
    older_messages = messages[: latest_user_idx if latest_user_idx is not None else latest_assistant_idx]
    return latest_user, messages[latest_assistant_idx], older_messages


def render_view(
    config: ApiClientConfig,
    *,
    manual_query_constraints: Mapping[str, Any] | None,
    debug_mode: bool,
    display_name: str,
    sidebar_open: bool,
) -> None:
    pending_prompt = st.session_state.pop("assistant_pending_prompt", None)
    if pending_prompt:
        _submit_prompt(
            config,
            prompt=pending_prompt,
            manual_query_constraints=manual_query_constraints,
            debug_mode=debug_mode,
        )
        st.rerun()

    messages = list(st.session_state.assistant_messages)
    current_state = assistant_view_state(messages)

    if current_state == "landing":
        submitted_prompt = _render_landing_state()
    else:
        submitted_prompt = _render_active_state(
            display_name=display_name,
            debug_mode=debug_mode,
            sidebar_open=sidebar_open,
        )

    if submitted_prompt:
        _submit_prompt(
            config,
            prompt=submitted_prompt,
            manual_query_constraints=manual_query_constraints,
            debug_mode=debug_mode,
        )
        st.session_state.assistant_current_prompt = ""
        st.rerun()


def render_answer_block(response: QueryResponseData) -> None:
    st.markdown(_build_answer_card_html(response), unsafe_allow_html=True)


def render_error_block(error: NormalizedApiError) -> None:
    st.markdown(_build_error_card_html(error), unsafe_allow_html=True)


def render_diagnostics_panel(
    response: QueryResponseData | None,
    error: NormalizedApiError | None,
    *,
    debug_mode: bool,
) -> None:
    st.markdown(_build_diagnostics_html(response, error), unsafe_allow_html=True)

    if response is not None:
        if response.context:
            with st.expander("Retrieved Evidence", expanded=False):
                _render_context_items(response.context)
        if debug_mode and response.evaluation_metadata:
            with st.expander("Debug Metadata", expanded=False):
                st.json(response.evaluation_metadata)
        return

    if error is not None and error.detail is not None:
        with st.expander("Error Detail", expanded=False):
            if isinstance(error.detail, Mapping):
                st.json(error.detail)
            else:
                st.code(str(error.detail), language="text", wrap_lines=True)


def _render_landing_state() -> str | None:
    st.markdown("<div class='polaris-landing-spacer'></div>", unsafe_allow_html=True)
    left_margin, content_col, right_margin = st.columns([0.25, 0.6, 0.15], gap="small")
    with content_col:
        st.markdown("<div class='polaris-quick-prompts-label'>Quick Prompts</div>", unsafe_allow_html=True)
        prompt_cols = st.columns(2, gap="medium")
        for idx, prompt in enumerate(quick_prompt_cards()):
            if prompt_cols[idx].button(
                prompt,
                key=f"assistant-quick-prompt-{idx}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.assistant_pending_prompt = prompt
                st.rerun()
        return _render_prompt_composer(
            form_key="assistant-landing-form",
            input_key="assistant_current_prompt",
            placeholder="Next quick insights...",
            submit_key="assistant-landing-submit",
        )


def _render_active_state(*, display_name: str, debug_mode: bool, sidebar_open: bool) -> str | None:
    latest_user, latest_assistant, older_messages = latest_exchange(list(st.session_state.assistant_messages))
    if latest_assistant is None:
        return _render_landing_state()

    if latest_user is not None:
        st.markdown(
            f"<div class='polaris-user-label'>{html.escape(display_name)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='polaris-query-pill'>{html.escape(str(latest_user.get('content') or ''))}</div>",
            unsafe_allow_html=True,
        )

    left_col, right_col = st.columns([7, 4], gap="large")
    submitted_prompt: str | None = None

    with left_col:
        st.markdown("<div class='polaris-assistant-label'>Polaris</div>", unsafe_allow_html=True)
        response = latest_assistant.get("response")
        error = latest_assistant.get("error")
        if response is not None:
            render_answer_block(response)
        elif error is not None:
            render_error_block(error)
        else:
            st.markdown(
                "<div class='polaris-answer-card'><div class='polaris-answer-empty'>No assistant content available.</div></div>",
                unsafe_allow_html=True,
            )

        _render_history_section(older_messages)
        submitted_prompt = _render_prompt_composer(
            form_key="assistant-followup-form",
            input_key="assistant_current_prompt",
            placeholder="Ask a follow-up question...",
            submit_key="assistant-followup-submit",
        )

    with right_col:
        render_diagnostics_panel(
            latest_assistant.get("response"),
            latest_assistant.get("error"),
            debug_mode=debug_mode,
        )
    return submitted_prompt


def _render_prompt_composer(
    *,
    form_key: str,
    input_key: str,
    placeholder: str,
    submit_key: str,
) -> str | None:
    with st.form(form_key):
        input_col, action_col = st.columns([12, 1], gap="small")
        with input_col:
            prompt = st.text_input(
                "Prompt",
                key=input_key,
                label_visibility="collapsed",
                placeholder=placeholder,
            )
        with action_col:
            submitted = st.form_submit_button("➜", key=submit_key, type="secondary", use_container_width=True)

    text = str(prompt or "").strip()
    if submitted and text:
        return text
    return None


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


def _render_history_section(older_messages: list[dict[str, Any]]) -> None:
    if not older_messages:
        return

    button_label = "Hide Conversation History" if bool(st.session_state.assistant_history_open) else "Conversation History"
    if st.button(button_label, key="assistant-history-toggle", type="secondary", use_container_width=False):
        st.session_state.assistant_history_open = not bool(st.session_state.assistant_history_open)
        st.rerun()

    if not bool(st.session_state.assistant_history_open):
        return

    rows: list[str] = []
    for message in older_messages:
        role = str(message.get("role") or "assistant").title()
        content = str(message.get("content") or "")
        rows.append(
            "<div class='polaris-history-row'>"
            f"<div class='polaris-history-row__role'>{html.escape(role)}</div>"
            f"<div class='polaris-history-row__content'>{_html_text(content)}</div>"
            "</div>",
        )
    st.markdown(
        f"<div class='polaris-history-panel'>{''.join(rows)}</div>",
        unsafe_allow_html=True,
    )


def _render_context_items(context_items: list[RetrievedContextItem]) -> None:
    for item in context_items:
        score_text = f"{item.score:.4f}" if isinstance(item.score, float) else "n/a"
        source_text = item.source or "unknown"
        with st.expander(f"[{item.rank}] {item.doc_id} · {source_text} · {score_text}", expanded=False):
            st.code(item.text or "No chunk text returned.", language="text", wrap_lines=True)


def _build_answer_card_html(response: QueryResponseData) -> str:
    badges = _build_status_badges_html(response.answer_status.code, len(response.context))
    sections_html = "".join(_build_answer_section_html(section) for section in parse_answer_sections(response.answer))
    if not sections_html:
        sections_html = "<div class='polaris-answer-empty'>No answer text was returned.</div>"

    return (
        "<div class='polaris-answer-card'>"
        f"{badges}"
        f"<div class='polaris-answer-detail'>{html.escape(response.answer_status.detail)}</div>"
        f"{sections_html}"
        "</div>"
    )


def _build_error_card_html(error: NormalizedApiError) -> str:
    if error.kind == "timeout":
        title = "The request reached the API deadline before a full answer was returned."
        body = "Try a shorter query, disable debug mode, or increase the request timeout."
    elif error.kind == "network_error":
        title = "The Streamlit frontend could not reach the backend service."
        body = "Check the API base URL and whether the backend container is running."
    else:
        title = "The backend returned an error before the answer could be completed."
        body = error.message

    failure_class = (
        f"<div class='polaris-answer-detail'>Failure class: {html.escape(error.failure_class)}</div>"
        if error.failure_class
        else ""
    )
    return (
        "<div class='polaris-answer-card polaris-answer-card--error'>"
        f"<div class='polaris-answer-section__title'>{html.escape(title)}</div>"
        f"<div class='polaris-answer-section__body'>{html.escape(body)}</div>"
        f"{failure_class}"
        "</div>"
    )


def _build_diagnostics_html(response: QueryResponseData | None, error: NormalizedApiError | None) -> str:
    if response is None and error is None:
        return (
            "<div class='polaris-diagnostics'>"
            "<div class='polaris-diagnostics__title'>Diagnostics</div>"
            "<div class='polaris-diagnostics__message'>Run a query to inspect evidence, timings, and query interpretation.</div>"
            "</div>"
        )

    if response is not None:
        retrieval_value = _format_timing_value(response.timings.retrieval_elapsed_ms)
        generation_value = _format_timing_value(response.timings.generation_elapsed_ms)
        constraints_html = _build_constraints_html(response.query_constraints)
        return (
            "<div class='polaris-diagnostics'>"
            "<div class='polaris-diagnostics__title'>Diagnostics</div>"
            "<div class='polaris-stat-block'>"
            "<div class='polaris-stat-block__label'>Evidence Chunks</div>"
            f"<div class='polaris-stat-block__value'>{len(response.context)}</div>"
            "</div>"
            "<div class='polaris-stat-grid'>"
            "<div class='polaris-stat-grid__item'>"
            "<div class='polaris-stat-grid__label'>Retrieval</div>"
            f"<div class='polaris-stat-grid__value'>{html.escape(retrieval_value)}</div>"
            "</div>"
            "<div class='polaris-stat-grid__item'>"
            "<div class='polaris-stat-grid__label'>Generation</div>"
            f"<div class='polaris-stat-grid__value'>{html.escape(generation_value)}</div>"
            "</div>"
            "</div>"
            "<div class='polaris-diagnostics__section-title'>Interpreted Query Constraints</div>"
            f"{constraints_html}"
            "</div>"
        )

    return (
        "<div class='polaris-diagnostics'>"
        "<div class='polaris-diagnostics__title'>Diagnostics</div>"
        "<div class='polaris-stat-block'>"
        "<div class='polaris-stat-block__label'>Evidence Chunks</div>"
        "<div class='polaris-stat-block__value'>0</div>"
        "</div>"
        f"<div class='polaris-diagnostics__message'>{html.escape(error.message if error is not None else 'No diagnostics available.')}</div>"
        "</div>"
    )


def _build_constraints_html(query_constraints: Mapping[str, Any] | None) -> str:
    if not query_constraints:
        return "<div class='polaris-constraints__empty'>No explicit query constraints were returned for this answer.</div>"

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
    rows: list[str] = []
    for key, label in labels.items():
        value = query_constraints.get(key)
        if value in (None, [], ""):
            continue
        if isinstance(value, list):
            display_value = ", ".join(str(item) for item in value)
        else:
            display_value = str(value)
        rows.append(
            "<div class='polaris-constraint-row'>"
            f"<span class='polaris-constraint-row__label'>{html.escape(label)}:</span> "
            f"<span class='polaris-constraint-row__value'>{html.escape(display_value)}</span>"
            "</div>"
        )
    if not rows:
        return "<div class='polaris-constraints__empty'>No explicit query constraints were returned for this answer.</div>"
    return "".join(rows)


def _build_status_badges_html(status_code: str, evidence_count: int) -> str:
    labels = {
        "grounded": "Grounded",
        "limited_evidence": "Limited Evidence",
        "no_evidence": "No Evidence",
    }
    status_label = labels.get(status_code, status_code.replace("_", " ").title())
    return (
        "<div class='polaris-status-row'>"
        f"<span class='polaris-status-badge polaris-status-badge--{html.escape(status_code)}'>{html.escape(status_label)}</span>"
        f"<span class='polaris-status-badge polaris-status-badge--muted'>{evidence_count} evidence chunks</span>"
        "</div>"
    )


def _build_answer_section_html(section: AnswerSection) -> str:
    if section.key == "RESPONSE":
        return f"<div class='polaris-answer-section__body'>{_html_text(section.body)}</div>"

    body_class = "polaris-answer-section__body polaris-answer-section__body--mono" if section.key in {"EXAMPLE CUSTOMER REPLY", "REFERENCE KEY"} else "polaris-answer-section__body"
    return (
        "<div class='polaris-answer-section'>"
        f"<div class='polaris-answer-section__title'>{html.escape(section.heading)}</div>"
        f"<div class='{body_class}'>{_html_text(section.body or 'No content returned.')}</div>"
        "</div>"
    )


def _format_timing_value(value_ms: int | None) -> str:
    if value_ms is None:
        return "n/a"
    if value_ms >= 1000:
        return f"{value_ms / 1000.0:.1f} s"
    return f"{value_ms} ms"


def _html_text(text: str) -> str:
    escaped = html.escape(str(text or ""))
    return escaped.replace("\n", "<br>")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
