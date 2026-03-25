from __future__ import annotations

from pathlib import Path
import html
import os

import streamlit as st


WORKSPACE_META: tuple[tuple[str, str, str], ...] = (
    ("Assistant", "AI Assistant", "✳"),
    ("Evaluation", "Evaluation", "◔"),
    ("System", "System", "⚙"),
)


def current_display_name() -> str:
    env_name = os.getenv("POLARIS_DISPLAY_NAME")
    if isinstance(env_name, str) and env_name.strip():
        return env_name.strip()

    home_name = Path.home().name.strip().replace("-", " ").replace("_", " ")
    if not home_name:
        return "You"
    return home_name[:1].upper() + home_name[1:]


def render_brand() -> None:
    st.markdown(
        "<div class='polaris-brand'>"
        "<span class='polaris-brand__icon'>✳</span>"
        "<span class='polaris-brand__text'>Polaris</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_page_intro(title: str, description: str) -> None:
    st.markdown(
        "<div class='polaris-page-intro'>"
        f"<div class='polaris-page-intro__title'>{html.escape(title)}</div>"
        f"<div class='polaris-page-intro__body'>{html.escape(description)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_active_nav_row(icon: str, label: str) -> None:
    st.markdown(
        "<div class='polaris-nav-row polaris-nav-row--active'>"
        f"<span class='polaris-nav-row__icon'>{html.escape(icon)}</span>"
        f"<span class='polaris-nav-row__label'>{html.escape(label)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
