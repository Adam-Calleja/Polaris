from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping

import requests

from polaris_rag.common.request_budget import POLARIS_TIMEOUT_HEADER


DEFAULT_API_BASE_URL = os.getenv("POLARIS_API_BASE_URL") or os.getenv("RAG_API_BASE") or "http://rag-api:8000"
DEFAULT_API_ENDPOINT_PATH = os.getenv("POLARIS_API_ENDPOINT_PATH", "/v1/query")
DEFAULT_TIMEOUT_S = float(os.getenv("POLARIS_API_TIMEOUT_S", "60"))
DEFAULT_FEEDBACK_LOG_PATH = os.getenv("POLARIS_FEEDBACK_LOG_PATH", "/app/data/ui_feedback/feedback.jsonl")


@dataclass(frozen=True)
class ApiClientConfig:
    base_url: str = DEFAULT_API_BASE_URL
    endpoint_path: str = DEFAULT_API_ENDPOINT_PATH
    timeout_s: float = DEFAULT_TIMEOUT_S


@dataclass(frozen=True)
class RetrievedContextItem:
    rank: int
    doc_id: str
    text: str
    score: float | None = None
    source: str | None = None


@dataclass(frozen=True)
class QueryTimings:
    retrieval_elapsed_ms: int | None = None
    generation_elapsed_ms: int | None = None


@dataclass(frozen=True)
class AnswerStatus:
    code: str
    detail: str


@dataclass(frozen=True)
class QueryResponseData:
    answer: str
    context: list[RetrievedContextItem] = field(default_factory=list)
    query_constraints: dict[str, Any] | None = None
    evaluation_metadata: dict[str, Any] | None = None
    answer_status: AnswerStatus = field(default_factory=lambda: AnswerStatus(code="no_evidence", detail="No retrieved context was returned for this answer."))
    timings: QueryTimings = field(default_factory=QueryTimings)


@dataclass(frozen=True)
class ApiProbeResult:
    ok: bool
    url: str
    status_code: int | None
    payload: Any = None
    message: str | None = None


@dataclass(frozen=True)
class NormalizedApiError:
    kind: str
    message: str
    status_code: int | None = None
    failure_class: str | None = None
    detail: Any = None


class ApiClientError(RuntimeError):
    def __init__(self, error: NormalizedApiError):
        super().__init__(error.message)
        self.error = error


class ApiTimeoutError(ApiClientError):
    pass


def _join_url(base_url: str, path: str) -> str:
    base = str(base_url or "").rstrip("/")
    suffix = path if str(path).startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def _post_json(url: str, payload: Mapping[str, Any], timeout: float, headers: Mapping[str, str]) -> requests.Response:
    return requests.post(url, json=dict(payload), headers=dict(headers), timeout=timeout)


def _get_json(url: str, timeout: float) -> requests.Response:
    return requests.get(url, timeout=timeout)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_answer_status(context_items: list[RetrievedContextItem]) -> AnswerStatus:
    if len(context_items) <= 0:
        return AnswerStatus(code="no_evidence", detail="No retrieved context was returned for this answer.")
    if len(context_items) == 1:
        return AnswerStatus(code="limited_evidence", detail="Only one supporting context item was retrieved for this answer.")
    return AnswerStatus(code="grounded", detail="Multiple supporting context items were retrieved for this answer.")


def _normalise_context_items(raw_context: Any) -> list[RetrievedContextItem]:
    if not isinstance(raw_context, list):
        return []

    items: list[RetrievedContextItem] = []
    for idx, item in enumerate(raw_context, start=1):
        if not isinstance(item, Mapping):
            items.append(
                RetrievedContextItem(
                    rank=idx,
                    doc_id="<unknown-doc-id>",
                    text=str(item),
                )
            )
            continue

        rank = _optional_int(item.get("rank")) or idx
        items.append(
            RetrievedContextItem(
                rank=rank,
                doc_id=str(item.get("doc_id") or item.get("id") or item.get("node_id") or "<unknown-doc-id>"),
                text=str(item.get("text") or item.get("content") or ""),
                score=_optional_float(item.get("score")),
                source=(str(item.get("source")) if item.get("source") is not None else None),
            )
        )

    return items


def _normalise_query_constraints(raw_constraints: Any) -> dict[str, Any] | None:
    if not isinstance(raw_constraints, Mapping):
        return None

    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    def _optional_bool(value: Any) -> bool | None:
        if value is None or isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return None

    query_type_raw = raw_constraints.get("query_type")
    query_type = str(query_type_raw).strip() if query_type_raw is not None and str(query_type_raw).strip() else None

    return {
        "query_type": query_type,
        "system_names": _string_list(raw_constraints.get("system_names")),
        "partition_names": _string_list(raw_constraints.get("partition_names")),
        "service_names": _string_list(raw_constraints.get("service_names")),
        "scope_family_names": _string_list(raw_constraints.get("scope_family_names")),
        "software_names": _string_list(raw_constraints.get("software_names")),
        "software_versions": _string_list(raw_constraints.get("software_versions")),
        "module_names": _string_list(raw_constraints.get("module_names")),
        "toolchain_names": _string_list(raw_constraints.get("toolchain_names")),
        "toolchain_versions": _string_list(raw_constraints.get("toolchain_versions")),
        "scope_required": _optional_bool(raw_constraints.get("scope_required")),
        "version_sensitive_guess": _optional_bool(raw_constraints.get("version_sensitive_guess")),
    }


def _normalise_timings(raw_timings: Any) -> QueryTimings:
    if not isinstance(raw_timings, Mapping):
        return QueryTimings()
    return QueryTimings(
        retrieval_elapsed_ms=_optional_int(raw_timings.get("retrieval_elapsed_ms")),
        generation_elapsed_ms=_optional_int(raw_timings.get("generation_elapsed_ms")),
    )


def _normalise_answer_status(raw_status: Any, context_items: list[RetrievedContextItem]) -> AnswerStatus:
    if isinstance(raw_status, Mapping):
        code_raw = raw_status.get("code")
        detail_raw = raw_status.get("detail")
        code = str(code_raw).strip() if code_raw is not None and str(code_raw).strip() else None
        detail = str(detail_raw).strip() if detail_raw is not None and str(detail_raw).strip() else None
        if code and detail:
            return AnswerStatus(code=code, detail=detail)
    return _derive_answer_status(context_items)


def _extract_answer(data: Any) -> str:
    if not isinstance(data, Mapping):
        return str(data)

    for key in ("answer", "response", "text", "output"):
        if key in data and isinstance(data[key], str):
            return data[key]

    for outer_key in ("result", "data"):
        nested = data.get(outer_key)
        if not isinstance(nested, Mapping):
            continue
        for key in ("answer", "response", "text", "output"):
            if key in nested and isinstance(nested[key], str):
                return nested[key]

    return str(data)


def _extract_error_detail(response: requests.Response) -> Any:
    try:
        payload = response.json()
    except Exception:
        text = response.text[:500]
        return text or None

    if isinstance(payload, Mapping) and "detail" in payload:
        return payload["detail"]
    return payload


def _error_from_response(response: requests.Response) -> ApiClientError:
    detail = _extract_error_detail(response)
    failure_class = detail.get("failure_class") if isinstance(detail, Mapping) else None
    if isinstance(detail, Mapping):
        message = str(detail.get("error") or detail.get("message") or f"API error {response.status_code}")
    else:
        message = str(detail or f"API error {response.status_code}")
    kind = "timeout" if response.status_code == 504 else "server_error" if response.status_code >= 500 else "api_error"
    error = NormalizedApiError(
        kind=kind,
        message=message,
        status_code=response.status_code,
        failure_class=str(failure_class) if failure_class is not None else None,
        detail=detail,
    )
    if kind == "timeout":
        return ApiTimeoutError(error)
    return ApiClientError(error)


def query_backend(
    config: ApiClientConfig,
    prompt: str,
    *,
    query_constraints: Mapping[str, Any] | None = None,
    include_evaluation_metadata: bool = False,
    server_timeout_ms: int | None = None,
) -> QueryResponseData:
    url = _join_url(config.base_url, config.endpoint_path)
    payload: dict[str, Any] = {"query": prompt}
    if query_constraints:
        payload["query_constraints"] = dict(query_constraints)
    if include_evaluation_metadata:
        payload["include_evaluation_metadata"] = True

    headers = {"Content-Type": "application/json"}
    if server_timeout_ms is not None:
        headers[POLARIS_TIMEOUT_HEADER] = str(server_timeout_ms)

    try:
        response = _post_json(url, payload, timeout=float(config.timeout_s), headers=headers)
    except requests.Timeout as exc:
        raise ApiTimeoutError(
            NormalizedApiError(
                kind="timeout",
                message=f"Timed out while waiting for the API at {url}.",
                detail=str(exc),
            )
        ) from exc
    except requests.RequestException as exc:
        raise ApiClientError(
            NormalizedApiError(
                kind="network_error",
                message=f"Failed to reach the API at {url}.",
                detail=str(exc),
            )
        ) from exc

    if response.status_code >= 400:
        raise _error_from_response(response)

    try:
        data = response.json()
    except Exception as exc:
        raise ApiClientError(
            NormalizedApiError(
                kind="invalid_response",
                message="The API returned a non-JSON response.",
                status_code=response.status_code,
                detail=response.text[:500],
            )
        ) from exc

    context_items = _normalise_context_items(data.get("context") if isinstance(data, Mapping) else [])
    answer = _extract_answer(data)
    query_constraints_payload = _normalise_query_constraints(data.get("query_constraints") if isinstance(data, Mapping) else None)
    evaluation_metadata = data.get("evaluation_metadata") if isinstance(data, Mapping) and isinstance(data.get("evaluation_metadata"), Mapping) else None
    answer_status = _normalise_answer_status(data.get("answer_status") if isinstance(data, Mapping) else None, context_items)
    timings = _normalise_timings(data.get("timings") if isinstance(data, Mapping) else None)

    return QueryResponseData(
        answer=answer,
        context=context_items,
        query_constraints=query_constraints_payload,
        evaluation_metadata=dict(evaluation_metadata) if evaluation_metadata is not None else None,
        answer_status=answer_status,
        timings=timings,
    )


def probe_endpoint(config: ApiClientConfig, path: str) -> ApiProbeResult:
    url = _join_url(config.base_url, path)
    try:
        response = _get_json(url, timeout=float(config.timeout_s))
    except requests.Timeout as exc:
        return ApiProbeResult(ok=False, url=url, status_code=None, message=f"Timed out: {exc}")
    except requests.RequestException as exc:
        return ApiProbeResult(ok=False, url=url, status_code=None, message=f"Request failed: {exc}")

    try:
        payload = response.json()
    except Exception:
        payload = response.text[:500]

    return ApiProbeResult(
        ok=response.status_code < 400,
        url=url,
        status_code=response.status_code,
        payload=payload,
        message=None if response.status_code < 400 else f"Endpoint returned {response.status_code}.",
    )
