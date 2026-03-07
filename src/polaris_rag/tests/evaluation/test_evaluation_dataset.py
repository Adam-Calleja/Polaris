import json

import pytest

from polaris_rag.evaluation.evaluation_dataset import (
    PrepProgressEvent,
    build_prepared_rows,
    build_prepared_rows_from_api,
    load_raw_examples,
)


class _Node:
    def __init__(self, node_id: str, text: str):
        self.id_ = node_id
        self.text = text


class _Source:
    def __init__(self, node: _Node):
        self.node = node


class _FakePipeline:
    def run(self, query: str, **kwargs):  # noqa: ANN001
        return {
            "response": f"resp::{query}",
            "source_nodes": [
                _Source(_Node("doc-1", "ctx-1")),
                _Source(_Node("doc-2", "ctx-2")),
            ],
        }


class _FlakyPipeline:
    def run(self, query: str, **kwargs):  # noqa: ANN001
        if query == "boom":
            raise RuntimeError("simulated pipeline failure")
        return {
            "response": f"resp::{query}",
            "source_nodes": [_Source(_Node("doc-1", f"ctx::{query}"))],
        }


def _ok_requester(
    api_url: str, query: str, timeout_seconds: float, headers  # noqa: ANN001
) -> dict[str, object]:
    return {
        "answer": f"answer::{query}",
        "context": [
            {"doc_id": f"doc::{query}", "text": f"ctx::{query}"},
        ],
    }


def _flaky_requester(
    api_url: str, query: str, timeout_seconds: float, headers  # noqa: ANN001
) -> dict[str, object]:
    if query == "boom":
        raise RuntimeError("simulated api failure")
    return _ok_requester(api_url, query, timeout_seconds, headers)


def test_load_raw_examples_supports_json_array_in_jsonl_suffix(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    payload = [{"query": "q1", "expected_answer": "a1"}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows = load_raw_examples(path)
    assert rows == payload


def test_build_prepared_rows_maps_query_and_expected_answer() -> None:
    raw_examples = [
        {"id": "ex-1", "query": "Q1", "expected_answer": "A1", "metadata": {"k": 1}},
        {"id": "ex-2", "query": "Q2", "expected_answer": "A2", "metadata": {"k": 2}},
    ]

    rows = build_prepared_rows(
        raw_examples=raw_examples,
        pipeline=_FakePipeline(),
        generation_workers=2,
    )

    assert [row["id"] for row in rows] == ["ex-1", "ex-2"]
    assert [row["user_input"] for row in rows] == ["Q1", "Q2"]
    assert [row["reference"] for row in rows] == ["A1", "A2"]
    assert [row["response"] for row in rows] == ["resp::Q1", "resp::Q2"]
    assert rows[0]["retrieved_contexts"] == ["ctx-1", "ctx-2"]
    assert rows[0]["retrieved_context_ids"] == ["doc-1", "doc-2"]


def test_build_prepared_rows_from_api_maps_answer_and_context() -> None:
    raw_examples = [
        {"id": "ex-1", "query": "Q1", "expected_answer": "A1"},
        {"id": "ex-2", "query": "Q2", "expected_answer": "A2"},
    ]

    def requester(api_url: str, query: str, timeout: float, headers):  # noqa: ANN001, ANN202
        assert api_url == "http://127.0.0.1:8000/v1/query"
        assert timeout == 30.0
        assert headers == {"X-Test": "1"}
        return {
            "answer": f"ans::{query}",
            "context": [
                {"doc_id": "doc-1", "text": "ctx-1"},
                {"doc_id": "doc-2", "text": "ctx-2"},
            ],
        }

    rows = build_prepared_rows_from_api(
        raw_examples=raw_examples,
        api_url="http://127.0.0.1:8000/v1/query",
        generation_workers=2,
        timeout_seconds=30.0,
        headers={"X-Test": "1"},
        requester=requester,
    )

    assert [row["id"] for row in rows] == ["ex-1", "ex-2"]
    assert [row["user_input"] for row in rows] == ["Q1", "Q2"]
    assert [row["reference"] for row in rows] == ["A1", "A2"]
    assert [row["response"] for row in rows] == ["ans::Q1", "ans::Q2"]
    assert rows[0]["retrieved_contexts"] == ["ctx-1", "ctx-2"]
    assert rows[0]["retrieved_context_ids"] == ["doc-1", "doc-2"]


def test_build_prepared_rows_from_api_fail_soft_captures_source_error() -> None:
    raw_examples = [{"id": "ex-1", "query": "Q1", "expected_answer": "A1"}]

    def failing_requester(api_url: str, query: str, timeout: float, headers):  # noqa: ANN001, ANN202
        raise RuntimeError("boom")

    rows = build_prepared_rows_from_api(
        raw_examples=raw_examples,
        api_url="http://127.0.0.1:8000/v1/query",
        raise_exceptions=False,
        requester=failing_requester,
    )

    assert len(rows) == 1
    assert rows[0]["response"] == ""
    assert rows[0]["retrieved_contexts"] == []
    assert rows[0]["retrieved_context_ids"] == []
    assert "source_error" in rows[0]["metadata"]
    assert "boom" in rows[0]["metadata"]["source_error"]


def test_build_prepared_rows_emits_monotonic_progress_events() -> None:
    raw_examples = [
        {"id": "ex-1", "query": "Q1", "expected_answer": "A1"},
        {"id": "ex-2", "query": "Q2", "expected_answer": "A2"},
        {"id": "ex-3", "query": "Q3", "expected_answer": "A3"},
    ]
    events: list[PrepProgressEvent] = []

    rows = build_prepared_rows(
        raw_examples=raw_examples,
        pipeline=_FakePipeline(),
        generation_workers=1,
        progress_callback=events.append,
    )

    assert len(rows) == 3
    assert [event.completed for event in events] == [1, 2, 3]
    assert all(event.total == 3 for event in events)
    assert events[-1].successes == 3
    assert events[-1].failures == 0
    assert events[-1].mode == "pipeline"


def test_build_prepared_rows_fail_soft_counts_failures() -> None:
    raw_examples = [
        {"id": "ex-1", "query": "Q1", "expected_answer": "A1"},
        {"id": "ex-2", "query": "boom", "expected_answer": "A2"},
        {"id": "ex-3", "query": "Q3", "expected_answer": "A3"},
    ]
    events: list[PrepProgressEvent] = []

    rows = build_prepared_rows(
        raw_examples=raw_examples,
        pipeline=_FlakyPipeline(),
        generation_workers=2,
        raise_exceptions=False,
        progress_callback=events.append,
    )

    assert len(rows) == 3
    assert events[-1].completed == 3
    assert events[-1].successes == 2
    assert events[-1].failures == 1
    failing = [row for row in rows if row["id"] == "ex-2"][0]
    assert "source_error" in failing["metadata"]


def test_build_prepared_rows_fail_fast_preserves_exception() -> None:
    raw_examples = [{"id": "ex-1", "query": "boom", "expected_answer": "A1"}]
    events: list[PrepProgressEvent] = []

    with pytest.raises(RuntimeError, match="simulated pipeline failure"):
        build_prepared_rows(
            raw_examples=raw_examples,
            pipeline=_FlakyPipeline(),
            generation_workers=1,
            raise_exceptions=True,
            progress_callback=events.append,
        )

    assert len(events) == 1
    assert events[0].completed == 1
    assert events[0].failures == 1
    assert events[0].last_error is not None
    assert "RuntimeError" in events[0].last_error


def test_build_prepared_rows_from_api_emits_progress_events() -> None:
    raw_examples = [
        {"id": "ex-1", "query": "Q1", "expected_answer": "A1"},
        {"id": "ex-2", "query": "Q2", "expected_answer": "A2"},
        {"id": "ex-3", "query": "Q3", "expected_answer": "A3"},
        {"id": "ex-4", "query": "Q4", "expected_answer": "A4"},
    ]
    events: list[PrepProgressEvent] = []

    rows = build_prepared_rows_from_api(
        raw_examples=raw_examples,
        api_url="http://unused.local/v1/query",
        generation_workers=2,
        requester=_ok_requester,
        progress_callback=events.append,
    )

    assert len(rows) == 4
    assert events[-1].completed == 4
    assert events[-1].successes == 4
    assert events[-1].failures == 0
    assert events[-1].mode == "api"
    assert rows[0]["retrieved_contexts"] == ["ctx::Q1"]
    assert rows[0]["retrieved_context_ids"] == ["doc::Q1"]


def test_build_prepared_rows_from_api_fail_soft_counts_failures() -> None:
    raw_examples = [
        {"id": "ex-1", "query": "Q1", "expected_answer": "A1"},
        {"id": "ex-2", "query": "boom", "expected_answer": "A2"},
    ]
    events: list[PrepProgressEvent] = []

    rows = build_prepared_rows_from_api(
        raw_examples=raw_examples,
        api_url="http://unused.local/v1/query",
        generation_workers=1,
        raise_exceptions=False,
        requester=_flaky_requester,
        progress_callback=events.append,
    )

    assert len(rows) == 2
    assert events[-1].completed == 2
    assert events[-1].successes == 1
    assert events[-1].failures == 1
    assert events[-1].last_error is not None
    assert "simulated api failure" in events[-1].last_error
    assert "id=ex-2" in events[-1].last_error
    failing = [row for row in rows if row["id"] == "ex-2"][0]
    assert "source_error" in failing["metadata"]
    assert "id=ex-2" in failing["metadata"]["source_error"]


def test_build_prepared_rows_retries_fail_soft_then_succeeds() -> None:
    raw_examples = [{"id": "ex-1", "query": "Q1", "expected_answer": "A1"}]
    calls = {"count": 0}

    class _RetryingPipeline:
        def run(self, query: str, **kwargs):  # noqa: ANN001
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("transient")
            return {
                "response": "final-response",
                "source_nodes": [_Source(_Node("doc-1", "ctx-1"))],
            }

    rows = build_prepared_rows(
        raw_examples=raw_examples,
        pipeline=_RetryingPipeline(),
        generation_workers=1,
        raise_exceptions=False,
        retry_policy={
            "max_attempts": 3,
            "initial_backoff_seconds": 0.0,
            "max_backoff_seconds": 0.0,
            "jitter_seconds": 0.0,
            "retry_on_empty_response": True,
        },
    )

    assert calls["count"] == 3
    assert rows[0]["response"] == "final-response"
    assert "source_error" not in rows[0]["metadata"]


def test_build_prepared_rows_from_api_retries_fail_soft_then_succeeds() -> None:
    raw_examples = [{"id": "ex-1", "query": "Q1", "expected_answer": "A1"}]
    calls = {"count": 0}

    def requester(api_url: str, query: str, timeout_seconds: float, headers):  # noqa: ANN001, ANN202
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("transient-api")
        return {
            "answer": "final-answer",
            "context": [{"doc_id": "doc-1", "text": "ctx-1"}],
        }

    rows = build_prepared_rows_from_api(
        raw_examples=raw_examples,
        api_url="http://unused.local/v1/query",
        generation_workers=1,
        raise_exceptions=False,
        requester=requester,
        retry_policy={
            "max_attempts": 3,
            "initial_backoff_seconds": 0.0,
            "max_backoff_seconds": 0.0,
            "jitter_seconds": 0.0,
            "retry_on_empty_response": True,
        },
    )

    assert calls["count"] == 3
    assert rows[0]["response"] == "final-answer"
    assert "source_error" not in rows[0]["metadata"]


def test_build_prepared_rows_retries_on_empty_response_then_succeeds() -> None:
    raw_examples = [{"id": "ex-1", "query": "Q1", "expected_answer": "A1"}]
    calls = {"count": 0}

    class _EmptyThenSuccessPipeline:
        def run(self, query: str, **kwargs):  # noqa: ANN001
            calls["count"] += 1
            if calls["count"] == 1:
                return {
                    "response": "",
                    "source_nodes": [_Source(_Node("doc-1", "ctx-1"))],
                }
            return {
                "response": "non-empty",
                "source_nodes": [_Source(_Node("doc-1", "ctx-1"))],
            }

    rows = build_prepared_rows(
        raw_examples=raw_examples,
        pipeline=_EmptyThenSuccessPipeline(),
        generation_workers=1,
        retry_policy={
            "max_attempts": 2,
            "initial_backoff_seconds": 0.0,
            "max_backoff_seconds": 0.0,
            "jitter_seconds": 0.0,
            "retry_on_empty_response": True,
        },
    )

    assert calls["count"] == 2
    assert rows[0]["response"] == "non-empty"
    assert "source_error" not in rows[0]["metadata"]


def test_build_prepared_rows_exhausted_empty_response_sets_source_error() -> None:
    raw_examples = [{"id": "ex-1", "query": "Q1", "expected_answer": "A1"}]

    class _AlwaysEmptyPipeline:
        def run(self, query: str, **kwargs):  # noqa: ANN001
            return {
                "response": "",
                "source_nodes": [_Source(_Node("doc-1", "ctx-1"))],
            }

    rows = build_prepared_rows(
        raw_examples=raw_examples,
        pipeline=_AlwaysEmptyPipeline(),
        generation_workers=1,
        raise_exceptions=False,
        retry_policy={
            "max_attempts": 2,
            "initial_backoff_seconds": 0.0,
            "max_backoff_seconds": 0.0,
            "jitter_seconds": 0.0,
            "retry_on_empty_response": True,
        },
    )

    assert rows[0]["response"] == ""
    assert "source_error" in rows[0]["metadata"]
    assert "response is empty after 2 attempt(s)" in rows[0]["metadata"]["source_error"]


def test_build_prepared_rows_fail_fast_retries_then_raises() -> None:
    raw_examples = [{"id": "ex-1", "query": "Q1", "expected_answer": "A1"}]
    calls = {"count": 0}

    class _AlwaysFailPipeline:
        def run(self, query: str, **kwargs):  # noqa: ANN001
            calls["count"] += 1
            raise RuntimeError("always-fail")

    with pytest.raises(RuntimeError, match="always-fail"):
        build_prepared_rows(
            raw_examples=raw_examples,
            pipeline=_AlwaysFailPipeline(),
            generation_workers=1,
            raise_exceptions=True,
            retry_policy={
                "max_attempts": 3,
                "initial_backoff_seconds": 0.0,
                "max_backoff_seconds": 0.0,
                "jitter_seconds": 0.0,
                "retry_on_empty_response": True,
            },
        )

    assert calls["count"] == 3
