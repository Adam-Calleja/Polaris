import json

from polaris_rag.evaluation.evaluation_dataset import (
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
