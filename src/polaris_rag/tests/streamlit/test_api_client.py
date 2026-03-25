from __future__ import annotations

import pytest

from polaris_rag.streamlit import api_client


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_query_backend_normalizes_success_response(monkeypatch) -> None:
    captured = {"payload": None, "headers": None}

    def _fake_post_json(url, payload, timeout, headers):  # noqa: ANN001
        captured["payload"] = payload
        captured["headers"] = headers
        return _FakeResponse(
            200,
            {
                "answer": "Grounded answer",
                "context": [
                    {"rank": 1, "doc_id": "doc-1", "text": "chunk text", "score": 0.9, "source": "docs"}
                ],
                "query_constraints": {
                    "query_type": "software_version",
                    "scope_family_names": ["cclake"],
                    "software_names": ["GROMACS"],
                    "software_versions": ["2024.4"],
                },
                "answer_status": {
                    "code": "limited_evidence",
                    "detail": "Only one supporting context item was retrieved for this answer.",
                },
                "timings": {
                    "retrieval_elapsed_ms": 5,
                    "generation_elapsed_ms": 9,
                },
            },
        )

    monkeypatch.setattr(api_client, "_post_json", _fake_post_json)

    result = api_client.query_backend(
        api_client.ApiClientConfig(base_url="http://example", endpoint_path="/v1/query", timeout_s=5),
        "hello",
        query_constraints={"software_names": ["GROMACS"]},
        include_evaluation_metadata=True,
        server_timeout_ms=1234,
    )

    assert result.answer == "Grounded answer"
    assert result.context[0].doc_id == "doc-1"
    assert result.query_constraints["software_names"] == ["GROMACS"]
    assert result.answer_status.code == "limited_evidence"
    assert result.timings.retrieval_elapsed_ms == 5
    assert captured["payload"]["include_evaluation_metadata"] is True
    assert captured["payload"]["query_constraints"] == {"software_names": ["GROMACS"]}
    assert captured["headers"][api_client.POLARIS_TIMEOUT_HEADER] == "1234"


def test_query_backend_normalizes_timeout_response(monkeypatch) -> None:
    def _fake_post_json(url, payload, timeout, headers):  # noqa: ANN001
        return _FakeResponse(
            504,
            {
                "detail": {
                    "error": "generation timed out",
                    "failure_class": "generation_timeout",
                }
            },
        )

    monkeypatch.setattr(api_client, "_post_json", _fake_post_json)

    with pytest.raises(api_client.ApiTimeoutError) as exc_info:
        api_client.query_backend(
            api_client.ApiClientConfig(base_url="http://example", endpoint_path="/v1/query", timeout_s=5),
            "hello",
        )

    assert exc_info.value.error.kind == "timeout"
    assert exc_info.value.error.failure_class == "generation_timeout"
    assert exc_info.value.error.message == "generation timed out"


def test_query_backend_normalizes_server_error_response(monkeypatch) -> None:
    def _fake_post_json(url, payload, timeout, headers):  # noqa: ANN001
        return _FakeResponse(
            500,
            {
                "detail": {
                    "error": "boom",
                    "failure_class": "api_internal_error",
                }
            },
        )

    monkeypatch.setattr(api_client, "_post_json", _fake_post_json)

    with pytest.raises(api_client.ApiClientError) as exc_info:
        api_client.query_backend(
            api_client.ApiClientConfig(base_url="http://example", endpoint_path="/v1/query", timeout_s=5),
            "hello",
        )

    assert exc_info.value.error.kind == "server_error"
    assert exc_info.value.error.failure_class == "api_internal_error"
    assert exc_info.value.error.message == "boom"
