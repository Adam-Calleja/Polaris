import json

from polaris_rag.evaluation.evaluation_dataset import (
    build_prepared_rows,
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
