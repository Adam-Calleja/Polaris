from __future__ import annotations

from polaris_rag.streamlit.polaris_interface import SESSION_DEFAULTS


def test_session_defaults_include_shell_state() -> None:
    assert SESSION_DEFAULTS["ui_sidebar_open"] is False
    assert SESSION_DEFAULTS["assistant_history_open"] is False
    assert SESSION_DEFAULTS["assistant_current_prompt"] == ""
    assert SESSION_DEFAULTS["ui_workspace"] == "Assistant"
