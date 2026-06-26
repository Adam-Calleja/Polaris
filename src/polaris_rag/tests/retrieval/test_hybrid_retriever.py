from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

schema = pytest.importorskip("llama_index.core.schema")

NodeWithScore = schema.NodeWithScore
TextNode = schema.TextNode

from polaris_rag.authority import RegistryEntity
from polaris_rag.retrieval.query_constraints import QueryConstraints
from polaris_rag.retrieval.retriever import HybridRetriever
from polaris_rag.retrieval.sparse_query import DeterministicSparseQueryExpander


def _node(node_id: str) -> TextNode:
    return TextNode(id_=node_id, text=f"text::{node_id}", metadata={})


class _FakeVectorStore:
    def __init__(self, *, dense_results=None, sparse_results=None):  # noqa: ANN001
        self._dense_results = dense_results or []
        self._sparse_results = sparse_results or []
        self.dense_queries: list[dict[str, object]] = []
        self.sparse_queries: list[dict[str, object]] = []

    def profile(self) -> dict[str, object]:
        return {
            "backend": "fake",
            "collection_name": "test_collection",
            "dense_model": "dense-model",
            "sparse_model": {"type": "fastembed", "model_name": "splade"},
        }

    def query_dense_nodes(self, query: str, **kwargs):  # noqa: ANN003
        self.dense_queries.append({"query": query, **kwargs})
        if callable(self._dense_results):
            return list(self._dense_results(query))
        return list(self._dense_results)

    def query_sparse_nodes(self, query: str, **kwargs):  # noqa: ANN003
        self.sparse_queries.append({"query": query, **kwargs})
        if callable(self._sparse_results):
            return list(self._sparse_results(query))
        return list(self._sparse_results)


def _hybrid_profile(rrf_k: int = 60) -> dict[str, object]:
    return {
        "dense_top_k": 3,
        "sparse_top_k": 3,
        "top_k": 3,
        "fusion": {
            "type": "rrf",
            "rrf_k": rrf_k,
            "signal_weights": {"dense": 1.0, "sparse": 1.0},
        },
    }


def test_deterministic_sparse_query_expander_adds_aliases_and_versions() -> None:
    entity = RegistryEntity(
        entity_id="software::openfoam",
        entity_type="software",
        canonical_name="OpenFOAM",
        aliases=["foam"],
        source_scope="local_official",
        status="current",
        known_versions=["2306"],
        doc_id="doc-1",
        doc_title="OpenFOAM docs",
        heading_path=[],
        evidence_spans=[],
        extraction_method="manual",
        review_state="auto_verified",
    )
    expander = DeterministicSparseQueryExpander(SimpleNamespace(entities=(entity,)))

    expansion = expander.expand(
        "how do I load it?",
        QueryConstraints(
            software_names=["OpenFOAM"],
            software_versions=["2306"],
        ),
    )

    assert "OpenFOAM" in expansion.expansion_terms
    assert "foam" in expansion.expansion_terms
    assert "2306" in expansion.expansion_terms
    assert expander.profile()["alias_group_count"] == 1


def test_hybrid_retriever_fuses_dense_and_sparse_results_and_stamps_trace() -> None:
    shared = NodeWithScore(node=_node("shared"), score=0.9)
    dense_only = NodeWithScore(node=_node("dense-only"), score=0.7)
    sparse_only = NodeWithScore(node=_node("sparse-only"), score=0.8)

    store = _FakeVectorStore(
        dense_results=[shared, dense_only],
        sparse_results=[NodeWithScore(node=_node("shared"), score=0.4), sparse_only],
    )
    retriever = HybridRetriever(
        storage_context=SimpleNamespace(vector_store=None),
        vector_store=store,
        retrieval_profile=_hybrid_profile(),
    )

    results = retriever.retrieve("where is the module?")

    assert [item.node.id_ for item in results] == ["shared", "sparse-only", "dense-only"]
    trace = results[0].node.metadata["retrieval_signal_trace"]
    assert trace["dense_rank"] == 1
    assert trace["sparse_rank"] == 1
    assert trace["fusion_type"] == "rrf"
    assert trace["fusion_score"] > results[1].score
    assert retriever.retriever_profile()["fusion"]["rrf_k"] == 60


def test_hybrid_retriever_expands_sparse_query_for_alias_and_version_hits() -> None:
    entity = RegistryEntity(
        entity_id="software::openfoam",
        entity_type="software",
        canonical_name="OpenFOAM",
        aliases=["foam"],
        source_scope="local_official",
        status="current",
        known_versions=["2306"],
        doc_id="doc-1",
        doc_title="OpenFOAM docs",
        heading_path=[],
        evidence_spans=[],
        extraction_method="manual",
        review_state="auto_verified",
    )
    expander = DeterministicSparseQueryExpander(SimpleNamespace(entities=(entity,)))
    store = _FakeVectorStore(
        dense_results=[],
        sparse_results=lambda query: [
            NodeWithScore(node=_node("openfoam-doc"), score=0.8),
        ] if "foam" in query and "2306" in query else [],
    )
    retriever = HybridRetriever(
        storage_context=SimpleNamespace(vector_store=None),
        vector_store=store,
        retrieval_profile=_hybrid_profile(),
        sparse_query_expander=expander,
    )

    results = retriever.retrieve(
        "Why does my build fail?",
        query_constraints=QueryConstraints(
            software_names=["OpenFOAM"],
            software_versions=["2306"],
        ),
    )

    assert [item.node.id_ for item in results] == ["openfoam-doc"]
    assert "foam" in store.sparse_queries[0]["query"]
    assert "2306" in store.sparse_queries[0]["query"]
    trace = results[0].node.metadata["retrieval_signal_trace"]
    assert "foam" in trace["expansion_terms"]
    assert "2306" in trace["expansion_terms"]


def test_hybrid_retriever_fingerprint_changes_when_fusion_profile_changes() -> None:
    store = _FakeVectorStore(dense_results=[], sparse_results=[])
    retriever_a = HybridRetriever(
        storage_context=SimpleNamespace(vector_store=None),
        vector_store=store,
        retrieval_profile=_hybrid_profile(rrf_k=60),
    )
    retriever_b = HybridRetriever(
        storage_context=SimpleNamespace(vector_store=None),
        vector_store=store,
        retrieval_profile=_hybrid_profile(rrf_k=30),
    )

    assert retriever_a.retriever_fingerprint() != retriever_b.retriever_fingerprint()
