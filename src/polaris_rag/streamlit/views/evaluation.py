from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from polaris_rag.streamlit.api_client import ApiClientConfig, ApiClientError, ApiTimeoutError, query_backend
from polaris_rag.streamlit.demo_catalog import DemoScenario, demo_scenarios
from polaris_rag.streamlit.feedback import (
    FeedbackRecord,
    append_feedback_record,
    compute_response_fingerprint,
    feedback_summary,
)
from polaris_rag.streamlit.shell import render_page_intro
from polaris_rag.streamlit.views.assistant import render_answer_block, render_diagnostics_panel, render_error_block


def render_view(
    config: ApiClientConfig,
    *,
    feedback_log_path: str,
    debug_mode: bool,
) -> None:
    render_page_intro(
        "Evaluation",
        "Curated live scenarios for your screencast, report screenshots, and lightweight usability evidence.",
    )

    _render_feedback_summary(feedback_log_path)

    for scenario in demo_scenarios():
        _render_scenario(config, scenario, feedback_log_path=feedback_log_path, debug_mode=debug_mode)


def _render_scenario(
    config: ApiClientConfig,
    scenario: DemoScenario,
    *,
    feedback_log_path: str,
    debug_mode: bool,
) -> None:
    with st.container():
        header_cols = st.columns([4, 1], gap="small")
        header_cols[0].markdown(f"<div class='polaris-scenario-title'>{scenario.title}</div>", unsafe_allow_html=True)
        run_clicked = header_cols[1].button(
            "Run",
            key=f"scenario-run-{scenario.scenario_id}",
            type="secondary",
            use_container_width=True,
        )
        st.write(scenario.description)
        st.caption(f"Focus: {scenario.focus}")
        st.code(scenario.query, language="text", wrap_lines=True)

        if run_clicked:
            _run_scenario(config, scenario, debug_mode=debug_mode)

        stored = st.session_state.evaluation_results.get(scenario.scenario_id)
        if not stored:
            st.divider()
            return

        left_col, right_col = st.columns([7, 3], gap="large")
        with left_col:
            if stored.get("response") is not None:
                render_answer_block(stored["response"])
            else:
                render_error_block(stored["error"])
        with right_col:
            render_diagnostics_panel(stored.get("response"), stored.get("error"), debug_mode=debug_mode)

        _render_feedback_form(
            scenario=scenario,
            feedback_log_path=feedback_log_path,
            result=stored,
        )
        st.divider()


def _run_scenario(config: ApiClientConfig, scenario: DemoScenario, *, debug_mode: bool) -> None:
    try:
        with st.spinner(f"Running scenario: {scenario.title}"):
            response = query_backend(
                config,
                scenario.query,
                query_constraints=scenario.query_constraints,
                include_evaluation_metadata=debug_mode or scenario.include_evaluation_metadata,
                server_timeout_ms=scenario.server_timeout_ms,
            )
        stored = {
            "response": response,
            "error": None,
            "query": scenario.query,
            "scenario_id": scenario.scenario_id,
        }
    except ApiTimeoutError as exc:
        stored = {
            "response": None,
            "error": exc.error,
            "query": scenario.query,
            "scenario_id": scenario.scenario_id,
        }
    except ApiClientError as exc:
        stored = {
            "response": None,
            "error": exc.error,
            "query": scenario.query,
            "scenario_id": scenario.scenario_id,
        }

    st.session_state.evaluation_results[scenario.scenario_id] = stored


def _render_feedback_form(*, scenario: DemoScenario, feedback_log_path: str, result: dict[str, Any]) -> None:
    response = result.get("response")
    error = result.get("error")
    if response is not None:
        response_fingerprint = compute_response_fingerprint(
            result["query"],
            response.answer,
            context_doc_ids=[item.doc_id for item in response.context],
            scenario_id=scenario.scenario_id,
        )
        answer_status_code = response.answer_status.code
        evidence_count = len(response.context)
    else:
        response_fingerprint = compute_response_fingerprint(
            result["query"],
            error.message if error is not None else "",
            context_doc_ids=[],
            scenario_id=scenario.scenario_id,
        )
        answer_status_code = "no_evidence"
        evidence_count = 0

    if response_fingerprint in st.session_state.feedback_submission_ids:
        st.info("Feedback for this exact response has already been submitted in this session.")
        return

    with st.form(f"feedback-form-{scenario.scenario_id}"):
        st.markdown("### Record Evaluation Feedback")
        helpful = st.radio("Helpful?", options=["yes", "partly", "no"], horizontal=True, key=f"helpful-{scenario.scenario_id}")
        grounded = st.radio("Grounded?", options=["yes", "partly", "no"], horizontal=True, key=f"grounded-{scenario.scenario_id}")
        citation_quality = st.selectbox(
            "Citation quality",
            options=["strong", "adequate", "weak"],
            key=f"citation-quality-{scenario.scenario_id}",
        )
        failure_type = st.selectbox(
            "Failure type",
            options=["none", "retrieval_gap", "ambiguous_question", "stale_or_version_risk", "timeout", "backend_error"],
            key=f"failure-type-{scenario.scenario_id}",
        )
        notes = st.text_area("Notes", key=f"notes-{scenario.scenario_id}", height=120)
        submitted = st.form_submit_button("Save Feedback", type="secondary", use_container_width=True)

    if not submitted:
        return

    record = FeedbackRecord(
        created_at=_now_iso(),
        response_fingerprint=response_fingerprint,
        query=result["query"],
        scenario_id=scenario.scenario_id,
        answer_status_code=answer_status_code,
        evidence_count=evidence_count,
        helpful=helpful,
        grounded=grounded,
        citation_quality=citation_quality,
        failure_type=failure_type,
        notes=notes.strip(),
    )
    append_feedback_record(Path(feedback_log_path), record)
    st.session_state.feedback_submission_ids.add(response_fingerprint)
    st.success("Feedback saved.")


def _render_feedback_summary(feedback_log_path: str) -> None:
    summary = feedback_summary(Path(feedback_log_path))
    cols = st.columns(3)
    cols[0].metric("Saved Feedback", summary["total"])
    cols[1].metric("Helpful = yes", summary["helpful_yes"])
    cols[2].metric("Grounded = yes", summary["grounded_yes"])

    if summary["total"] <= 0:
        st.caption("No persistent evaluation feedback has been recorded yet.")
        return

    detail_cols = st.columns(2)
    detail_cols[0].markdown("#### By Scenario")
    detail_cols[0].dataframe(summary["by_scenario"], use_container_width=True, hide_index=True)
    detail_cols[1].markdown("#### Failure Types")
    detail_cols[1].dataframe(summary["failure_types"], use_container_width=True, hide_index=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
