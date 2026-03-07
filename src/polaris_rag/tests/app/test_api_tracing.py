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
    def run(self, query: str):
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
        "mlflow.parent_run_id": "run-parent",
        "polaris.parent_run_id": "run-parent",
        "polaris.child_run_id": "run-child",
        "polaris.stage": "dataset_preparation",
    }
