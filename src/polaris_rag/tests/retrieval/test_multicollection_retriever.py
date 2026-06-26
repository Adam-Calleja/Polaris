from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

pytest.importorskip("llama_index.retrievers.bm25")
schema = pytest.importorskip("llama_index.core.schema")

NodeWithScore = schema.NodeWithScore
TextNode = schema.TextNode

from polaris_rag.common.request_budget import RetrievalTimeoutError
from polaris_rag.retrieval.retriever import MultiCollectionRetriever


class _RecordingRetriever:
    def __init__(self, results):
        self._results = list(results)
        self.calls: list[float | None] = []

    def retrieve(self, query: str, *, timeout_seconds: float | None = None, **kwargs):
        self.calls.append(timeout_seconds)
        return list(self._results)


class _TimeoutRetriever:
    def retrieve(self, query: str, *, timeout_seconds: float | None = None, **kwargs):
        raise TimeoutError("source timed out")


def _node(node_id: str, *, source: str) -> TextNode:
    return TextNode(text=f"text::{node_id}", id_=node_id, metadata={"source": source})


def test_multi_collection_merges_candidates_and_stamps_source_metadata() -> None:
    docs_shared = _node("shared", source="docs")
    docs_unique = _node("docs-only", source="docs")
    tickets_shared = _node("shared", source="tickets")
    tickets_unique = _node("tickets-only", source="tickets")

    retriever = MultiCollectionRetriever(
        source_retrievers={
            "docs": _RecordingRetriever(
                [
                    NodeWithScore(node=docs_shared, score=0.9),
                    NodeWithScore(node=docs_unique, score=0.7),
                ]
            ),
            "tickets": _RecordingRetriever(
                [
                    NodeWithScore(node=tickets_shared, score=0.8),
                    NodeWithScore(node=tickets_unique, score=0.6),
                ]
            ),
        },
        source_settings={
            "docs": {"weight": 1.0},
            "tickets": {"weight": 1.0},
        },
        final_top_k=3,
        rerank={"type": "rrf", "rrf_k": 60},
    )

    results = retriever.retrieve("where is my ticket?")

    assert [item.node.id_ for item in results] == ["shared", "docs-only", "tickets-only"]
    shared_metadata = results[0].node.metadata
    assert shared_metadata["retrieval_source"] == "multi"
    assert shared_metadata["retrieval_sources"] == ["docs", "tickets"]
    assert shared_metadata["retrieval_source_ranks"] == {"docs": 1, "tickets": 1}
    assert results[0].score > results[1].score


def test_multi_collection_shares_timeout_budget_across_sources(monkeypatch) -> None:
    first = _RecordingRetriever([NodeWithScore(node=_node("a", source="docs"), score=1.0)])
    second = _RecordingRetriever([NodeWithScore(node=_node("b", source="tickets"), score=0.9)])

    monotonic_values = iter([100.0, 100.01, 100.03])
    monkeypatch.setattr(
        "polaris_rag.retrieval.retriever.time.monotonic",
        lambda: next(monotonic_values),
    )

    retriever = MultiCollectionRetriever(
        source_retrievers={"docs": first, "tickets": second},
        final_top_k=2,
    )

    retriever.retrieve("query", timeout_seconds=0.1)

    assert first.calls[0] == pytest.approx(0.09, abs=1e-6)
    assert second.calls[0] == pytest.approx(0.07, abs=1e-6)


def test_multi_collection_raises_retrieval_timeout_when_source_times_out() -> None:
    retriever = MultiCollectionRetriever(
        source_retrievers={
            "docs": _TimeoutRetriever(),
        },
        final_top_k=1,
    )

    with pytest.raises(RetrievalTimeoutError) as exc_info:
        retriever.retrieve("query", timeout_seconds=0.5)

    assert "docs" in str(exc_info.value)
