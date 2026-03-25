from __future__ import annotations

import html

import streamlit as st

from polaris_rag.streamlit.api_client import ApiClientConfig, probe_endpoint
from polaris_rag.streamlit.shell import render_page_intro


def render_view(
    config: ApiClientConfig,
    *,
    debug_mode: bool,
    feedback_log_path: str,
) -> None:
    render_page_intro(
        "System",
        "System overview, source legend, and live backend readiness checks.",
    )

    _render_architecture()
    _render_source_legend()
    _render_health_checks(config)
    _render_runtime_summary(config, debug_mode=debug_mode, feedback_log_path=feedback_log_path)


def _render_architecture() -> None:
    st.markdown("### Architecture Flow")
    cols = st.columns(5, gap="small")
    stages = [
        ("Corpus", "Jira tickets and official documentation"),
        ("Retrieval", "Vector search plus query interpretation"),
        ("Reranking", "Validity-aware ranking and source filtering"),
        ("Prompting", "Structured ticket-style answer template"),
        ("Answer", "Grounded response with evidence inspection"),
    ]
    for column, (title, body) in zip(cols, stages):
        column.markdown(
            "<div class='polaris-stage-card'>"
            f"<div class='polaris-stage-title'>{html.escape(title)}</div>"
            f"<div class='polaris-stage-body'>{html.escape(body)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_source_legend() -> None:
    st.markdown("### Corpus Legend")
    legend_cols = st.columns(3, gap="small")
    legend_cols[0].markdown(
        "<div class='polaris-stage-card'><div class='polaris-stage-title'>Docs</div><div class='polaris-stage-body'>Official documentation and service pages.</div></div>",
        unsafe_allow_html=True,
    )
    legend_cols[1].markdown(
        "<div class='polaris-stage-card'><div class='polaris-stage-title'>Tickets</div><div class='polaris-stage-body'>Historical support tickets used as operational memory.</div></div>",
        unsafe_allow_html=True,
    )
    legend_cols[2].markdown(
        "<div class='polaris-stage-card'><div class='polaris-stage-title'>Multi-source</div><div class='polaris-stage-body'>Merged retrieval across official and experiential evidence.</div></div>",
        unsafe_allow_html=True,
    )


def _render_health_checks(config: ApiClientConfig) -> None:
    st.markdown("### Live Backend Checks")
    if st.button("Refresh Status", key="system-refresh-status", type="secondary"):
        st.session_state.system_refresh_counter = st.session_state.get("system_refresh_counter", 0) + 1

    health = probe_endpoint(config, "/health")
    ready = probe_endpoint(config, "/ready")

    cols = st.columns(2, gap="large")
    for column, title, probe in [(cols[0], "Health", health), (cols[1], "Readiness", ready)]:
        column.markdown(f"#### {title}")
        if probe.ok:
            column.success(f"{title} check passed.")
        else:
            column.error(probe.message or f"{title} check failed.")
        column.caption(f"`{probe.url}`")
        if probe.payload is not None:
            with column.expander(f"{title} Payload", expanded=False):
                if isinstance(probe.payload, dict):
                    st.json(probe.payload)
                else:
                    st.code(str(probe.payload), language="text", wrap_lines=True)


def _render_runtime_summary(config: ApiClientConfig, *, debug_mode: bool, feedback_log_path: str) -> None:
    st.markdown("### Frontend Runtime Summary")
    st.markdown(f"**API base URL:** `{config.base_url}`")
    st.markdown(f"**Query endpoint:** `{config.endpoint_path}`")
    st.markdown(f"**HTTP timeout:** `{config.timeout_s}` seconds")
    st.markdown(f"**Debug mode:** `{debug_mode}`")
    st.markdown(f"**Feedback log:** `{feedback_log_path}`")
