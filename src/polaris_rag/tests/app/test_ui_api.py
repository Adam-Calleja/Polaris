from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from polaris_rag.app import api


def test_configure_cors_allows_frontend_origin_and_headers(monkeypatch) -> None:
    monkeypatch.setenv(
        "POLARIS_UI_CORS_ALLOWED_ORIGINS",
        "http://localhost:8501,http://example.test:9000",
    )

    test_app = FastAPI()

    @test_app.post("/echo")
    def echo() -> dict[str, bool]:
        return {"ok": True}

    api.configure_cors(test_app)
    client = TestClient(test_app)

    response = client.options(
        "/echo",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": ", ".join(api.UI_CORS_ALLOWED_HEADERS),
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
    allowed_headers = response.headers["access-control-allow-headers"].lower()
    for header_name in api.UI_CORS_ALLOWED_HEADERS:
        assert header_name.lower() in allowed_headers


def test_ui_runtime_returns_feedback_log_path(monkeypatch, tmp_path) -> None:
    feedback_log_path = tmp_path / "feedback" / "feedback.jsonl"
    monkeypatch.setenv("POLARIS_FEEDBACK_LOG_PATH", str(feedback_log_path))

    response = api.ui_runtime()

    assert response.feedback_log_path == str(feedback_log_path)
    assert response.query_endpoint_path == "/v1/query"
    assert response.health_endpoint_path == "/health"
    assert response.ready_endpoint_path == "/ready"


def test_submit_feedback_and_summary_round_trip(monkeypatch, tmp_path) -> None:
    feedback_log_path = tmp_path / "feedback" / "feedback.jsonl"
    monkeypatch.setenv("POLARIS_FEEDBACK_LOG_PATH", str(feedback_log_path))

    first = api.UiFeedbackSubmissionRequest(
        response_fingerprint="fingerprint-1",
        query="q1",
        scenario_id="strong_answer",
        answer_status_code="grounded",
        evidence_count=3,
        helpful="yes",
        grounded="yes",
        citation_quality="strong",
        failure_type="none",
        notes=" clear answer ",
    )
    second = api.UiFeedbackSubmissionRequest(
        response_fingerprint="fingerprint-2",
        query="q2",
        scenario_id="timeout_case",
        answer_status_code="no_evidence",
        evidence_count=0,
        helpful="no",
        grounded="no",
        citation_quality="weak",
        failure_type="timeout",
        notes="deadline hit",
    )

    first_response = api.submit_ui_feedback(first)
    second_response = api.submit_ui_feedback(second)
    summary = api.ui_feedback_summary()

    assert first_response.response_fingerprint == "fingerprint-1"
    assert second_response.response_fingerprint == "fingerprint-2"
    assert summary.total == 2
    assert summary.helpful_yes == 1
    assert summary.grounded_yes == 1
    assert summary.by_scenario == [
        {"scenario_id": "strong_answer", "count": 1},
        {"scenario_id": "timeout_case", "count": 1},
    ]
    assert summary.failure_types == [
        {"failure_type": "none", "count": 1},
        {"failure_type": "timeout", "count": 1},
    ]
