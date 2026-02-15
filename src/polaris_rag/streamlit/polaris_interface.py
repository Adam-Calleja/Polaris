import os
from typing import Any, Dict, Optional

import requests
import streamlit as st

st.title("Polaris")

# -----------------------------
# Backend configuration
# -----------------------------
# You can override these via environment variables (recommended for Docker Compose).
DEFAULT_API_BASE_URL = os.getenv("POLARIS_API_BASE_URL", "http://rag-api:8000")
# FastAPI exposes the query endpoint at /v1/query
DEFAULT_API_ENDPOINT_PATH = os.getenv("POLARIS_API_ENDPOINT_PATH", "/v1/query")
DEFAULT_TIMEOUT_S = float(os.getenv("POLARIS_API_TIMEOUT_S", "60"))

with st.sidebar:
    st.header("Backend")
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)
    api_path = st.text_input("Endpoint path", value=DEFAULT_API_ENDPOINT_PATH)
    timeout_s = st.number_input("Timeout (s)", min_value=1, max_value=600, value=int(DEFAULT_TIMEOUT_S))

    st.caption(
        "Tip: when running via Docker Compose, `http://rag-api:8000` should resolve from the Streamlit container. "
        "If running Streamlit locally, you may want `http://localhost:8000`."
    )


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> requests.Response:
    """POST JSON to the backend with a sensible default header."""
    return requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


def call_backend(prompt: str) -> str:
    """Call the FastAPI backend and extract a text answer.

    This is intentionally tolerant to different response shapes so you can evolve the API
    without having to constantly update the UI.
    """
    base = api_base_url.rstrip("/")
    path = api_path if api_path.startswith("/") else f"/{api_path}"
    url = f"{base}{path}"

    # API expects the QueryRequest schema: {"query": "..."}
    payload: Dict[str, Any] = {"query": prompt}

    try:
        resp = _post_json(url, payload, timeout=float(timeout_s))
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach API at {url}: {e}") from e

    if resp.status_code >= 400:
        # Try to include any JSON error detail if present.
        detail: Optional[str] = None
        try:
            j = resp.json()
            # FastAPI often returns {"detail": ...}
            if isinstance(j, dict) and "detail" in j:
                detail = str(j["detail"])
        except Exception:
            detail = None

        if detail:
            raise RuntimeError(f"API error {resp.status_code}: {detail}")
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")

    # Parse response
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"API returned non-JSON response: {resp.text[:500]}") from e

    # Accept a few common shapes:
    # 1) {"answer": "..."}
    # 2) {"response": "..."}
    # 3) {"text": "..."}
    # 4) {"output": "..."}
    # 5) {"result": {"answer": "..."}} etc.
    # Polaris' /v1/query commonly returns a QueryResponse-like object.
    if isinstance(data, dict):
        for key in ("answer", "response", "text", "output"):
            if key in data and isinstance(data[key], str):
                return data[key]

        # Nested common patterns
        for outer_key in ("result", "data"):
            if outer_key in data and isinstance(data[outer_key], dict):
                inner = data[outer_key]
                for key in ("answer", "response", "text", "output"):
                    if key in inner and isinstance(inner[key], str):
                        return inner[key]

        # If nothing obvious, fall back to pretty-ish string.
        return str(data)

    # If backend returns a raw JSON string/list
    return str(data)


# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Polaris..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_text = call_backend(prompt)
            except Exception as e:
                response_text = f"⚠️ {e}"
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})