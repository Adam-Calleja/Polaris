from __future__ import annotations

from pathlib import Path

import streamlit as st

from polaris_rag.streamlit.api_client import (
    ApiClientConfig,
    DEFAULT_API_BASE_URL,
    DEFAULT_API_ENDPOINT_PATH,
    DEFAULT_FEEDBACK_LOG_PATH,
    DEFAULT_TIMEOUT_S,
)
from polaris_rag.streamlit.views import assistant, evaluation, system


st.set_page_config(
    page_title="Polaris",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    _bootstrap_session_state()
    _inject_theme()
    workspace, config, manual_query_constraints, debug_mode, feedback_log_path = _render_sidebar()

    st.markdown("# Polaris")
    st.caption("Validity-aware RAG for HPC support, with evidence inspection, evaluation scenarios, and live system checks.")

    if workspace == "Assistant":
        assistant.render_view(
            config,
            manual_query_constraints=manual_query_constraints,
            debug_mode=debug_mode,
        )
    elif workspace == "Evaluation":
        evaluation.render_view(
            config,
            feedback_log_path=feedback_log_path,
            debug_mode=debug_mode,
        )
    else:
        system.render_view(
            config,
            debug_mode=debug_mode,
            feedback_log_path=feedback_log_path,
        )


def _bootstrap_session_state() -> None:
    defaults = {
        "ui_workspace": "Assistant",
        "ui_api_base_url": DEFAULT_API_BASE_URL,
        "ui_api_path": DEFAULT_API_ENDPOINT_PATH,
        "ui_timeout_s": int(DEFAULT_TIMEOUT_S),
        "ui_debug_mode": False,
        "ui_query_type": "auto",
        "ui_scope_family_names": "",
        "ui_service_names": "",
        "ui_software_names": "",
        "ui_software_versions": "",
        "assistant_messages": [],
        "assistant_latest_message": None,
        "assistant_pending_prompt": None,
        "evaluation_results": {},
        "feedback_submission_ids": set(),
        "system_refresh_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _inject_theme() -> None:
    css_path = Path(__file__).with_name("theme.css")
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _render_sidebar() -> tuple[str, ApiClientConfig, dict[str, object] | None, bool, str]:
    with st.sidebar:
        st.markdown("## Control Panel")
        workspace = st.radio(
            "Workspace",
            options=["Assistant", "Evaluation", "System"],
            key="ui_workspace",
            label_visibility="collapsed",
        )

        st.markdown("### Backend")
        st.text_input("API base URL", key="ui_api_base_url")
        st.text_input("Query endpoint", key="ui_api_path")
        st.number_input(
            "HTTP timeout (s)",
            min_value=1,
            max_value=600,
            step=1,
            key="ui_timeout_s",
        )
        st.toggle("Debug mode", key="ui_debug_mode")
        st.caption(
            "Debug mode requests evaluation metadata from the API and exposes raw diagnostic payloads in the interface."
        )

        with st.expander("Manual Query Constraints", expanded=False):
            st.selectbox(
                "Query type",
                options=["auto", "local_operational", "software_version", "general_how_to"],
                key="ui_query_type",
            )
            st.text_input("Scope families", key="ui_scope_family_names", help="Comma-separated, for example: cclake")
            st.text_input("Services", key="ui_service_names", help="Comma-separated service names.")
            st.text_input("Software", key="ui_software_names", help="Comma-separated software names.")
            st.text_input("Software versions", key="ui_software_versions", help="Comma-separated version strings.")
            st.caption("These constraints are optional and mainly intended for demos or controlled evaluation runs.")

    config = ApiClientConfig(
        base_url=str(st.session_state.ui_api_base_url),
        endpoint_path=str(st.session_state.ui_api_path),
        timeout_s=float(st.session_state.ui_timeout_s),
    )
    manual_constraints = _build_manual_query_constraints()
    return workspace, config, manual_constraints, bool(st.session_state.ui_debug_mode), DEFAULT_FEEDBACK_LOG_PATH


def _build_manual_query_constraints() -> dict[str, object] | None:
    def _split_csv(text: str) -> list[str]:
        return [item.strip() for item in str(text or "").split(",") if item.strip()]

    query_type = None if st.session_state.ui_query_type == "auto" else str(st.session_state.ui_query_type)
    payload = {
        "query_type": query_type,
        "scope_family_names": _split_csv(st.session_state.ui_scope_family_names),
        "service_names": _split_csv(st.session_state.ui_service_names),
        "software_names": _split_csv(st.session_state.ui_software_names),
        "software_versions": _split_csv(st.session_state.ui_software_versions),
    }
    if all(value in (None, []) for value in payload.values()):
        return None
    return payload


main()
