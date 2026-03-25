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


class _ContextPipeline:
    def run(self, query: str, **kwargs):
        node = type(
            "_Node",
            (),
            {
                "id_": "ticket-1",
                "text": "FULL-TICKET",
                "metadata": {
                    "retrieval_source": "tickets",
                    "source_authority": "ticket_memory",
                    "authority_tier": 1,
                    "validity_status": "unknown",
                    "doc_title": "Ticket Memory",
                    "private_email": "secret@example.com",
                },
            },
        )()
        raw_node = type("_Node", (), {"id_": "ticket-1::chunk::0001", "text": "chunk"})()
        source = type("_Source", (), {"node": node, "score": 0.9})()
        raw_source = type("_Source", (), {"node": raw_node, "score": 0.8})()
        return {
            "response": f"resp::{query}",
            "source_nodes": [source],
            "raw_source_nodes": [raw_source],
            "retrieval_trace": [
                {
                    "rank": 1,
                    "doc_id": "ticket-1",
                    "source": "tickets",
                    "source_authority": "ticket_memory",
                    "validity_status": "unknown",
                    "rerank_trace": {
                        "reranker_type": "validity_aware",
                        "final_score": 0.7,
                    },
                }
            ],
            "reranker_profile": {"type": "validity_aware"},
            "reranker_fingerprint": "fingerprint-123",
        }


class _QueryConstraintPipeline:
    def run(self, query: str, **kwargs):
        return {
            "response": f"resp::{query}",
            "source_nodes": [],
            "query_constraints": {
                "query_type": "software_version",
                "system_names": [],
                "partition_names": [],
                "service_names": [],
                "scope_family_names": ["cclake"],
                "software_names": ["GROMACS"],
                "software_versions": ["2024.4"],
                "module_names": [],
                "toolchain_names": [],
                "toolchain_versions": [],
                "scope_required": None,
                "version_sensitive_guess": True,
            },
        }


class _TimedPipeline:
    def run(self, query: str, **kwargs):
        node_one = type("_Node", (), {"id_": "doc-1", "text": "chunk-1", "metadata": {"retrieval_source": "docs"}})()
        node_two = type("_Node", (), {"id_": "doc-2", "text": "chunk-2", "metadata": {"retrieval_source": "tickets"}})()
        return {
            "response": f"resp::{query}",
            "source_nodes": [
                type("_Source", (), {"node": node_one, "score": 0.9})(),
                type("_Source", (), {"node": node_two, "score": 0.8})(),
            ],
            "timings": {
                "retrieval_elapsed_ms": 12,
                "generation_elapsed_ms": 34,
            },
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


def test_query_returns_resolved_context_not_raw_chunks(monkeypatch) -> None:
    api.app.state.container = _FakeContainer()
    api.app.state.container.pipeline = _ContextPipeline()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [],
        }
    )

    response = api.query(api.QueryRequest(query="hello"), request)

    assert response.context[0].doc_id == "ticket-1"
    assert response.context[0].text == "FULL-TICKET"


def test_query_returns_query_constraints_when_present() -> None:
    api.app.state.container = _FakeContainer()
    api.app.state.container.pipeline = _QueryConstraintPipeline()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [],
        }
    )

    response = api.query(api.QueryRequest(query="hello"), request)

    assert response.answer == "resp::hello"
    assert response.query_constraints is not None
    assert response.query_constraints.query_type == "software_version"
    assert response.query_constraints.software_names == ["GROMACS"]
    assert response.query_constraints.scope_family_names == ["cclake"]


def test_query_forwards_body_query_constraints_to_pipeline() -> None:
    captured = {"kwargs": None}

    class _CapturingPipeline:
        def run(self, query: str, **kwargs):  # noqa: ANN001
            captured["kwargs"] = kwargs
            return {"response": f"resp::{query}", "source_nodes": []}

    api.app.state.container = _FakeContainer()
    api.app.state.container.pipeline = _CapturingPipeline()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [],
        }
    )

    response = api.query(
        api.QueryRequest(
            query="hello",
            query_constraints=api.QueryConstraintsPayload(
                query_type="software_version",
                scope_family_names=["cclake"],
                software_names=["GROMACS"],
                software_versions=["2024.4"],
            ),
        ),
        request,
    )

    assert response.answer == "resp::hello"
    assert captured["kwargs"]["query_constraints"] == {
        "query_type": "software_version",
        "system_names": [],
        "partition_names": [],
        "service_names": [],
        "scope_family_names": ["cclake"],
        "software_names": ["GROMACS"],
        "software_versions": ["2024.4"],
        "module_names": [],
        "toolchain_names": [],
        "toolchain_versions": [],
        "scope_required": None,
        "version_sensitive_guess": None,
    }


def test_query_includes_evaluation_metadata_when_requested() -> None:
    api.app.state.container = _FakeContainer()
    api.app.state.container.pipeline = _ContextPipeline()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [
                (api.POLARIS_EVAL_INCLUDE_METADATA_HEADER.lower().encode("utf-8"), b"true"),
            ],
        }
    )

    response = api.query(api.QueryRequest(query="hello"), request)

    assert response.evaluation_metadata is not None
    assert response.evaluation_metadata["reranker_profile"] == {"type": "validity_aware"}
    assert response.evaluation_metadata["reranker_fingerprint"] == "fingerprint-123"
    assert response.evaluation_metadata["retrieval_trace"][0]["doc_id"] == "ticket-1"
    assert response.evaluation_metadata["ranked_context_metadata"][0]["doc_title"] == "Ticket Memory"
    assert "private_email" not in response.evaluation_metadata["ranked_context_metadata"][0]
    assert "text" not in response.evaluation_metadata["ranked_context_metadata"][0]


def test_query_includes_evaluation_metadata_when_requested_in_body() -> None:
    api.app.state.container = _FakeContainer()
    api.app.state.container.pipeline = _ContextPipeline()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [],
        }
    )

    response = api.query(
        api.QueryRequest(query="hello", include_evaluation_metadata=True),
        request,
    )

    assert response.evaluation_metadata is not None
    assert response.evaluation_metadata["reranker_profile"] == {"type": "validity_aware"}


def test_query_returns_answer_status_and_timings() -> None:
    api.app.state.container = _FakeContainer()
    api.app.state.container.pipeline = _TimedPipeline()

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/query",
            "headers": [],
        }
    )

    response = api.query(api.QueryRequest(query="hello"), request)

    assert response.answer_status.code == "grounded"
    assert response.answer_status.detail == "Multiple supporting context items were retrieved for this answer."
    assert response.timings.retrieval_elapsed_ms == 12
    assert response.timings.generation_elapsed_ms == 34


def test_query_request_model_accepts_legacy_body_shape() -> None:
    request_model = api.QueryRequest.model_validate({"query": "hello"})

    assert request_model.query == "hello"
    assert request_model.query_constraints is None
    assert request_model.include_evaluation_metadata is False


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
