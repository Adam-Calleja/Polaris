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
from polaris_rag.streamlit.shell import WORKSPACE_META, current_display_name, render_active_nav_row, render_brand
from polaris_rag.streamlit.views import assistant, evaluation, system


st.set_page_config(
    page_title="Polaris",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="collapsed",
)


SESSION_DEFAULTS: dict[str, object] = {
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
    "ui_sidebar_open": False,
    "ui_manual_constraints_open": False,
    "assistant_messages": [],
    "assistant_latest_message": None,
    "assistant_pending_prompt": None,
    "assistant_current_prompt": "",
    "assistant_history_open": False,
    "evaluation_results": {},
    "feedback_submission_ids": set(),
    "system_refresh_counter": 0,
}


def main() -> None:
    _bootstrap_session_state()
    _inject_theme()

    config = ApiClientConfig(
        base_url=str(st.session_state.ui_api_base_url),
        endpoint_path=str(st.session_state.ui_api_path),
        timeout_s=float(st.session_state.ui_timeout_s),
    )
    manual_constraints = _build_manual_query_constraints()
    display_name = current_display_name()

    if bool(st.session_state.ui_sidebar_open):
        drawer_col, main_col = st.columns([1.08, 4.1], gap="medium")
        with drawer_col:
            _render_drawer()
        with main_col:
            _render_main_canvas(
                config=config,
                manual_query_constraints=manual_constraints,
                display_name=display_name,
                feedback_log_path=DEFAULT_FEEDBACK_LOG_PATH,
            )
        return

    left_margin, main_col, right_margin = st.columns([0.08, 0.84, 0.08], gap="small")
    with main_col:
        _render_main_canvas(
            config=config,
            manual_query_constraints=manual_constraints,
            display_name=display_name,
            feedback_log_path=DEFAULT_FEEDBACK_LOG_PATH,
        )


def _bootstrap_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not str(st.session_state.ui_api_base_url or "").strip():
        st.session_state.ui_api_base_url = DEFAULT_API_BASE_URL
    if not str(st.session_state.ui_api_path or "").strip():
        st.session_state.ui_api_path = DEFAULT_API_ENDPOINT_PATH


def _inject_theme() -> None:
    css_path = Path(__file__).with_name("theme.css")
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _render_main_canvas(
    *,
    config: ApiClientConfig,
    manual_query_constraints: dict[str, object] | None,
    display_name: str,
    feedback_log_path: str,
) -> None:
    _render_main_header(sidebar_open=bool(st.session_state.ui_sidebar_open))

    workspace = str(st.session_state.ui_workspace)
    if workspace == "Assistant":
        assistant.render_view(
            config,
            manual_query_constraints=manual_query_constraints,
            debug_mode=bool(st.session_state.ui_debug_mode),
            display_name=display_name,
            sidebar_open=bool(st.session_state.ui_sidebar_open),
        )
    elif workspace == "Evaluation":
        evaluation.render_view(
            config,
            feedback_log_path=feedback_log_path,
            debug_mode=bool(st.session_state.ui_debug_mode),
        )
    else:
        system.render_view(
            config,
            debug_mode=bool(st.session_state.ui_debug_mode),
            feedback_log_path=feedback_log_path,
        )


def _render_main_header(*, sidebar_open: bool) -> None:
    if sidebar_open:
        spacer_col, brand_col = st.columns([4.4, 1.1], gap="small")
        with brand_col:
            render_brand()
        return

    toggle_col, spacer_col, brand_col = st.columns([0.8, 4.0, 1.3], gap="small")
    with toggle_col:
        if st.button("☰", key="shell-open-drawer", type="secondary", use_container_width=True):
            st.session_state.ui_sidebar_open = True
            st.rerun()
    with brand_col:
        render_brand()


def _render_drawer() -> None:
    with st.container():
        st.markdown("<div class='polaris-drawer-sentinel'></div>", unsafe_allow_html=True)

        header_cols = st.columns([4, 1], gap="small")
        with header_cols[0]:
            st.markdown("<div class='polaris-drawer-title'>Menu</div>", unsafe_allow_html=True)
        with header_cols[1]:
            with st.container():
                st.markdown("<div class='polaris-drawer-close-sentinel'></div>", unsafe_allow_html=True)
                if st.button("×", key="shell-close-drawer", type="secondary", use_container_width=True):
                    st.session_state.ui_sidebar_open = False
                    st.rerun()

        st.markdown("<div class='polaris-drawer-divider'></div>", unsafe_allow_html=True)
        _render_drawer_navigation()
        st.markdown("<div class='polaris-drawer-section-title'>Backend</div>", unsafe_allow_html=True)
        st.markdown("<div class='polaris-drawer-divider'></div>", unsafe_allow_html=True)
        _render_drawer_backend_controls()


def _render_drawer_navigation() -> None:
    current_workspace = str(st.session_state.ui_workspace)
    for workspace_key, label, icon in WORKSPACE_META:
        if current_workspace == workspace_key:
            render_active_nav_row(icon, label)
            continue
        with st.container():
            st.markdown("<div class='polaris-drawer-nav-sentinel'></div>", unsafe_allow_html=True)
            if st.button(
                f"{icon}  {label}",
                key=f"nav-{workspace_key.lower()}",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state.ui_workspace = workspace_key
                st.rerun()


def _render_drawer_backend_controls() -> None:
    _render_drawer_blue_field("API base URL", "ui_api_base_url", "API base URL")
    _render_drawer_blue_field("Query Endpoint", "ui_api_path", "Query endpoint")

    st.markdown("<div class='polaris-field-label'>HTTP Timeout</div>", unsafe_allow_html=True)
    timeout_cols = st.columns([1.4, 0.8, 0.8], gap="small")
    with timeout_cols[0]:
        st.markdown(
            f"<div class='polaris-timeout-pill polaris-timeout-pill--value'>{int(st.session_state.ui_timeout_s)}</div>",
            unsafe_allow_html=True,
        )
    with timeout_cols[1]:
        with st.container():
            st.markdown("<div class='polaris-timeout-button-sentinel'></div>", unsafe_allow_html=True)
            if st.button("−", key="timeout-dec", type="secondary", use_container_width=True):
                st.session_state.ui_timeout_s = max(1, int(st.session_state.ui_timeout_s) - 5)
                st.rerun()
    with timeout_cols[2]:
        with st.container():
            st.markdown("<div class='polaris-timeout-button-sentinel'></div>", unsafe_allow_html=True)
            if st.button("+", key="timeout-inc", type="secondary", use_container_width=True):
                st.session_state.ui_timeout_s = min(600, int(st.session_state.ui_timeout_s) + 5)
                st.rerun()

    st.toggle("Debug mode", key="ui_debug_mode")
    st.caption(
        "Debug mode requests evaluation metadata from the API and exposes raw diagnostic payloads in the interface."
    )

    toggle_label = "Hide Manual Query Constraints" if bool(st.session_state.ui_manual_constraints_open) else "Manual Query Constraints"
    if st.button(toggle_label, key="toggle-manual-constraints", type="secondary", use_container_width=True):
        st.session_state.ui_manual_constraints_open = not bool(st.session_state.ui_manual_constraints_open)
        st.rerun()

    if not bool(st.session_state.ui_manual_constraints_open):
        return

    st.selectbox(
        "Query type",
        options=["auto", "local_operational", "software_version", "general_how_to"],
        key="ui_query_type",
    )
    st.text_input("Scope families", key="ui_scope_family_names", help="Comma-separated, for example: cclake")
    st.text_input("Services", key="ui_service_names", help="Comma-separated service names.")
    st.text_input("Software", key="ui_software_names", help="Comma-separated software names.")
    st.text_input("Software versions", key="ui_software_versions", help="Comma-separated version strings.")
    st.caption("Optional constraints for demos and controlled evaluation runs.")


def _render_drawer_blue_field(label: str, key: str, input_label: str) -> None:
    st.markdown(f"<div class='polaris-field-label'>{label}</div>", unsafe_allow_html=True)
    chip_col, spacer_col = st.columns([1.55, 1], gap="small")
    with chip_col:
        with st.container():
            st.markdown("<div class='polaris-drawer-blue-field-sentinel'></div>", unsafe_allow_html=True)
            st.text_input(
                input_label,
                key=key,
                label_visibility="collapsed",
            )


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


if __name__ == "__main__":
    main()
