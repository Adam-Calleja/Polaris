from __future__ import annotations

from contextlib import contextmanager

import pytest

Request = pytest.importorskip("starlette.requests").Request

from polaris_rag.app import api


class _Span:
    def __init__(self, name: str):
        self.name = name
        self.outputs = None


class _FakePipeline:
    def run(self, query: str, **kwargs):
        return {
            "response": f"resp::{query}",
            "source_nodes": [],
        }


class _FakeContainer:
    def __init__(self):
        self.pipeline = _FakePipeline()



def test_query_adds_trace_tag_from_mlflow_header(monkeypatch) -> None:
    captured = {"tags": None, "outputs": None}

    @contextmanager
    def _fake_start_span(name: str, **kwargs):  # noqa: ANN001
        captured["tags"] = kwargs.get("tags")
        yield _Span(name)

    def _fake_set_span_outputs(span: _Span, outputs):  # noqa: ANN001
        captured["outputs"] = outputs

    monkeypatch.setattr(api, "start_span", _fake_start_span)
    monkeypatch.setattr(api, "set_span_outputs", _fake_set_span_outputs)

    api.app.state.container = _FakeContainer()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [
                (api.TRACE_PARENT_RUN_HEADER.lower().encode("utf-8"), b"run-123"),
            ],
        }
    )

    response = api.query(api.QueryRequest(query="hello"), request)

    assert response.answer == "resp::hello"
    assert captured["tags"] == {
        "polaris.source": "api",
        "polaris.eval_request": "true",
        "polaris.eval_policy": "interactive",
        "mlflow.parent_run_id": "run-123",
        "polaris.parent_run_id": "run-123",
    }
    assert captured["outputs"] is not None
    assert captured["outputs"]["answer"] == "resp::hello"


def test_query_adds_child_run_and_stage_tags(monkeypatch) -> None:
    captured = {"tags": None}

    @contextmanager
    def _fake_start_span(name: str, **kwargs):  # noqa: ANN001
        captured["tags"] = kwargs.get("tags")
        yield _Span(name)

    monkeypatch.setattr(api, "start_span", _fake_start_span)

    api.app.state.container = _FakeContainer()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [
                (api.TRACE_PARENT_RUN_HEADER.lower().encode("utf-8"), b"run-parent"),
                (api.TRACE_CHILD_RUN_HEADER.lower().encode("utf-8"), b"run-child"),
                (api.TRACE_STAGE_HEADER.lower().encode("utf-8"), b"dataset_preparation"),
            ],
        }
    )

    api.query(api.QueryRequest(query="hello"), request)

    assert captured["tags"] == {
        "polaris.source": "api",
        "polaris.eval_request": "true",
        "polaris.eval_policy": "interactive",
        "mlflow.parent_run_id": "run-parent",
        "polaris.parent_run_id": "run-parent",
        "polaris.child_run_id": "run-child",
        "polaris.stage": "dataset_preparation",
    }


def test_query_maps_generation_timeout_to_504(monkeypatch) -> None:
    captured = {"outputs": None}

    class _TimeoutPipeline:
        def run(self, query: str, **kwargs):  # noqa: ANN001
            raise api.GenerationTimeoutError("generation timed out")

    @contextmanager
    def _fake_start_span(name: str, **kwargs):  # noqa: ANN001
        yield _Span(name)

    def _fake_set_span_outputs(span: _Span, outputs):  # noqa: ANN001
        captured["outputs"] = outputs

    monkeypatch.setattr(api, "start_span", _fake_start_span)
    monkeypatch.setattr(api, "set_span_outputs", _fake_set_span_outputs)

    api.app.state.container = _FakeContainer()
    api.app.state.container.pipeline = _TimeoutPipeline()
    api.app.state.container.config = type(
        "_Cfg",
        (),
        {
            "raw": {
                "evaluation": {
                    "generation": {
                        "deadlines": {
                            "official": {
                                "client_total_seconds": 120,
                                "server_total_seconds": 110,
                                "retrieval_cap_seconds": 10,
                                "cleanup_reserve_seconds": 5,
                            }
                        }
                    }
                }
            }
        },
    )()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [
                (api.POLARIS_TIMEOUT_HEADER.lower().encode("utf-8"), b"110000"),
                (api.POLARIS_EVAL_POLICY_HEADER.lower().encode("utf-8"), b"official"),
            ],
        }
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.query(api.QueryRequest(query="hello"), request)

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail["failure_class"] == "generation_timeout"
    assert exc_info.value.detail["failure_stage"] == "generation"
    assert captured["outputs"] is not None
    assert captured["outputs"]["failure_class"] == "generation_timeout"


def test_query_maps_retrieval_timeout_to_504(monkeypatch) -> None:
    captured = {"outputs": None}

    class _TimeoutPipeline:
        def run(self, query: str, **kwargs):  # noqa: ANN001
            raise api.RetrievalTimeoutError("retrieval timed out")

    @contextmanager
    def _fake_start_span(name: str, **kwargs):  # noqa: ANN001
        yield _Span(name)

    def _fake_set_span_outputs(span: _Span, outputs):  # noqa: ANN001
        captured["outputs"] = outputs

    monkeypatch.setattr(api, "start_span", _fake_start_span)
    monkeypatch.setattr(api, "set_span_outputs", _fake_set_span_outputs)

    api.app.state.container = _FakeContainer()
    api.app.state.container.pipeline = _TimeoutPipeline()
    api.app.state.container.config = type(
        "_Cfg",
        (),
        {
            "raw": {
                "evaluation": {
                    "generation": {
                        "deadlines": {
                            "official": {
                                "client_total_seconds": 120,
                                "server_total_seconds": 110,
                                "retrieval_cap_seconds": 10,
                                "cleanup_reserve_seconds": 5,
                            }
                        }
                    }
                }
            }
        },
    )()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [
                (api.POLARIS_TIMEOUT_HEADER.lower().encode("utf-8"), b"110000"),
                (api.POLARIS_EVAL_POLICY_HEADER.lower().encode("utf-8"), b"official"),
            ],
        }
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.query(api.QueryRequest(query="hello"), request)

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail["failure_class"] == "retrieval_timeout"
    assert exc_info.value.detail["failure_stage"] == "retrieval"
    assert captured["outputs"] is not None
    assert captured["outputs"]["failure_class"] == "retrieval_timeout"
