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


def _normalise_context_items(raw_context: Any) -> list[dict[str, Any]]:
    """Normalise backend context payload into a predictable list for rendering."""
    if not isinstance(raw_context, list):
        return []

    items: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_context, start=1):
        if not isinstance(item, dict):
            items.append(
                {
                    "rank": idx,
                    "doc_id": "<unknown-doc-id>",
                    "text": str(item),
                    "score": None,
                }
            )
            continue

        rank_raw = item.get("rank", idx)
        try:
            rank = int(rank_raw)
        except Exception:
            rank = idx

        score_raw = item.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else None

        items.append(
            {
                "rank": rank,
                "doc_id": str(item.get("doc_id") or item.get("id") or item.get("node_id") or "<unknown-doc-id>"),
                "text": str(item.get("text") or item.get("content") or ""),
                "score": score,
            }
        )

    return items


def _render_retrieved_context(context_items: list[dict[str, Any]]) -> None:
    if not context_items:
        return

    with st.expander(f"Retrieved context ({len(context_items)})", expanded=False):
        for item in context_items:
            rank = item.get("rank", "?")
            doc_id = item.get("doc_id", "<unknown-doc-id>")
            score = item.get("score")
            header = f"[{rank}] {doc_id}"
            if isinstance(score, float):
                header += f" (score: {score:.4f})"

            st.markdown(f"**{header}**")
            text = item.get("text", "")
            if text.strip():
                st.code(text, language="text", wrap_lines=True)
            else:
                st.caption("No text returned for this chunk.")


def call_backend(prompt: str) -> Dict[str, Any]:
    """Call the FastAPI backend and extract answer + retrieved context.

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
    # 1) {"answer": "...", "context": [...]}
    # 2) {"response": "..."}
    # 3) {"text": "..."}
    # 4) {"output": "..."}
    # 5) {"result": {"answer": "..."}} etc.
    if isinstance(data, dict):
        answer: Optional[str] = None
        for key in ("answer", "response", "text", "output"):
            if key in data and isinstance(data[key], str):
                answer = data[key]
                break

        # Nested common patterns
        if answer is None:
            for outer_key in ("result", "data"):
                if outer_key in data and isinstance(data[outer_key], dict):
                    inner = data[outer_key]
                    for key in ("answer", "response", "text", "output"):
                        if key in inner and isinstance(inner[key], str):
                            answer = inner[key]
                            break
                if answer is not None:
                    break

        context_items = _normalise_context_items(
            data.get("context")
            or data.get("contexts")
            or data.get("retrieved_context")
            or []
        )

        if answer is None:
            answer = str(data)

        return {"answer": answer, "context": context_items}

    # If backend returns a raw JSON string/list
    return {"answer": str(data), "context": []}


# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_retrieved_context(message.get("context", []))

if prompt := st.chat_input("Ask Polaris..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                backend_response = call_backend(prompt)
                response_text = backend_response["answer"]
                context_items = backend_response.get("context", [])
            except Exception as e:
                response_text = f"⚠️ {e}"
                context_items = []
        st.markdown(response_text)
        _render_retrieved_context(context_items)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "context": context_items}
    )
