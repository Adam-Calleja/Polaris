"""Retriever implementations for the Polaris retrieval layer."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Optional

from llama_index.core import StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from polaris_rag.common.request_budget import RetrievalTimeoutError, is_timeout_exception
from polaris_rag.retrieval.query_constraints import QueryConstraints
from polaris_rag.retrieval.reranker import MergedCandidate, create_reranker
from polaris_rag.retrieval.vector_store import QdrantIndexStore

try:
    from polaris_rag.retrieval.sparse_query import (
        DeterministicSparseQueryExpander,
        SparseQueryExpansion,
    )
except Exception:  # pragma: no cover - defensive import for lighter test contexts
    DeterministicSparseQueryExpander = Any  # type: ignore[assignment]
    SparseQueryExpansion = Any  # type: ignore[assignment]


def retriever_fingerprint(profile: Mapping[str, Any] | None) -> str:
    """Return a stable SHA256 fingerprint for a retriever profile."""
    payload = json.dumps(
        _stable_json_value(profile or {}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VectorIndexRetriever:
    """Dense vector retriever over a configured source collection."""

    def __init__(
        self,
        *,
        storage_context: StorageContext,
        filters: Optional[MetadataFilters] = None,
        top_k: Optional[int] = 5,
        vector_store: QdrantIndexStore | None = None,
        embedder: Any | None = None,
        retrieval_profile: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self._top_k = max(1, int(top_k or 5))
        self._filters = filters
        self._embedder = embedder
        self._vector_store = _resolve_native_vector_store(storage_context, vector_store)
        self._retriever = None
        if self._vector_store is None:
            vector_store_impl = getattr(storage_context, "vector_store", None)
            vector_index = getattr(vector_store_impl, "_index", None)
            if vector_index is None:
                raise ValueError(
                    "VectorIndexRetriever requires a native vector store wrapper "
                    "or a LlamaIndex vector index."
                )
            self._retriever = vector_index.as_retriever(
                similarity_top_k=self._top_k,
                filters=filters,
            )
        self._profile = {
            "type": "vector",
            "top_k": self._top_k,
            "filters": _metadata_filters_profile(filters),
            "vector_store": _vector_store_profile(self._vector_store),
            "retrieval_profile": _stable_json_value(dict(retrieval_profile or {})) or None,
        }

    def retrieve(
        self,
        query: str,
        *,
        timeout_seconds: float | None = None,
        query_constraints: Any | None = None,
        **kwargs: Any,
    ) -> list[NodeWithScore]:
        _ = query_constraints, kwargs
        if self._vector_store is not None:
            results = self._vector_store.query_dense_nodes(
                query,
                top_k=self._top_k,
                filters=self._filters,
                timeout_seconds=timeout_seconds,
            )
        else:
            results = self._retriever.retrieve(query)

        return _stamp_dense_results(results)

    def retriever_profile(self) -> dict[str, Any]:
        """Return the active retriever profile."""
        return dict(self._profile)

    def retriever_fingerprint(self) -> str:
        """Return the active retriever fingerprint."""
        return retriever_fingerprint(self._profile)


class SparseIndexRetriever:
    """Sparse-only retriever over a configured source collection."""

    def __init__(
        self,
        *,
        storage_context: StorageContext,
        filters: Optional[MetadataFilters] = None,
        top_k: Optional[int] = 5,
        vector_store: QdrantIndexStore | None = None,
        retrieval_profile: Mapping[str, Any] | None = None,
        sparse_query_expander: DeterministicSparseQueryExpander | None = None,
        **_: Any,
    ) -> None:
        self._filters = filters
        self._vector_store = _resolve_native_vector_store(storage_context, vector_store)
        if self._vector_store is None:
            raise ValueError(
                "SparseIndexRetriever requires a native vector store wrapper with sparse query support."
            )
        profile = dict(retrieval_profile or {})
        self._top_k = max(1, int(profile.get("sparse_top_k", profile.get("top_k", top_k or 5))))
        self._expander = sparse_query_expander
        self._profile = {
            "type": "sparse",
            "top_k": self._top_k,
            "filters": _metadata_filters_profile(filters),
            "vector_store": _vector_store_profile(self._vector_store),
            "sparse_query_expander": _sparse_query_expander_profile(self._expander),
        }

    def retrieve(
        self,
        query: str,
        *,
        timeout_seconds: float | None = None,
        query_constraints: Any | None = None,
        **kwargs: Any,
    ) -> list[NodeWithScore]:
        _ = kwargs
        expansion = _expand_sparse_query(self._expander, query, query_constraints)
        results = self._vector_store.query_sparse_nodes(
            expansion.query_text,
            top_k=self._top_k,
            filters=self._filters,
            timeout_seconds=timeout_seconds,
        )
        return _stamp_sparse_results(results, expansion=expansion)

    def retriever_profile(self) -> dict[str, Any]:
        """Return the active retriever profile."""
        return dict(self._profile)

    def retriever_fingerprint(self) -> str:
        """Return the active retriever fingerprint."""
        return retriever_fingerprint(self._profile)


class BM25IndexRetriever:
    """BM25 retriever over a configured source chunk docstore."""

    def __init__(
        self,
        *,
        storage_context: StorageContext,
        filters: Optional[MetadataFilters] = None,
        top_k: Optional[int] = 5,
        retrieval_profile: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self._filters = filters
        self._docstore = getattr(storage_context, "docstore", None)
        if self._docstore is None:
            raise ValueError("BM25IndexRetriever requires a storage context with a docstore.")

        profile = dict(retrieval_profile or {})
        bm25_cfg = _as_mapping(profile.get("bm25", {}))
        self._top_k = max(1, int(profile.get("bm25_top_k", profile.get("top_k", top_k or 5))))
        self._k1 = _coerce_float(bm25_cfg.get("k1"), 1.5)
        self._b = _coerce_float(bm25_cfg.get("b"), 0.75)
        self._corpus = _BM25Corpus.from_docstore(
            self._docstore,
            k1=self._k1,
            b=self._b,
        )
        self._profile = {
            "type": "bm25",
            "top_k": self._top_k,
            "filters": _metadata_filters_profile(filters),
            "docstore": _docstore_profile(self._docstore, node_count=self._corpus.document_count),
            "bm25": {
                "k1": self._k1,
                "b": self._b,
            },
        }

    def retrieve(
        self,
        query: str,
        *,
        timeout_seconds: float | None = None,
        query_constraints: Any | None = None,
        **kwargs: Any,
    ) -> list[NodeWithScore]:
        _ = query_constraints, kwargs
        results = self._corpus.search(
            query,
            top_k=self._top_k,
            filters=self._filters,
            timeout_seconds=timeout_seconds,
        )
        return _stamp_bm25_results(results)

    def retriever_profile(self) -> dict[str, Any]:
        """Return the active retriever profile."""
        return dict(self._profile)

    def retriever_fingerprint(self) -> str:
        """Return the active retriever fingerprint."""
        return retriever_fingerprint(self._profile)


class DenseBM25HybridRetriever:
    """Hybrid dense+BM25 retriever with client-side within-source fusion."""

    def __init__(
        self,
        *,
        storage_context: StorageContext,
        filters: Optional[MetadataFilters] = None,
        top_k: Optional[int] = 5,
        vector_store: QdrantIndexStore | None = None,
        embedder: Any | None = None,
        retrieval_profile: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        profile = dict(retrieval_profile or {})
        fusion_cfg = _as_mapping(profile.get("fusion", {}))
        signal_weights = _as_mapping(fusion_cfg.get("signal_weights", {}))

        self._filters = filters
        self._dense_top_k = max(1, int(profile.get("dense_top_k", top_k or 5)))
        self._bm25_top_k = max(
            1,
            int(profile.get("bm25_top_k", profile.get("sparse_top_k", profile.get("dense_top_k", top_k or 5)))),
        )
        self._top_k = max(1, int(profile.get("top_k", top_k or 5)))
        self._fusion_type = str(fusion_cfg.get("type", "rrf") or "rrf").strip().lower()
        if self._fusion_type != "rrf":
            raise ValueError(
                f"Unsupported hybrid fusion type {self._fusion_type!r}. Supported fusion types: ['rrf']."
            )
        self._rrf_k = max(1, int(fusion_cfg.get("rrf_k", 60) or 60))
        self._signal_weights = {
            "dense": _coerce_float(signal_weights.get("dense"), 1.0),
            "bm25": _coerce_float(signal_weights.get("bm25", signal_weights.get("sparse")), 1.0),
        }

        self._dense_retriever = VectorIndexRetriever(
            storage_context=storage_context,
            filters=filters,
            top_k=self._dense_top_k,
            vector_store=vector_store,
            embedder=embedder,
            retrieval_profile=profile,
        )
        self._bm25_retriever = BM25IndexRetriever(
            storage_context=storage_context,
            filters=filters,
            top_k=self._bm25_top_k,
            retrieval_profile=profile,
        )
        self._profile = {
            "type": "dense_bm25_hybrid",
            "dense_top_k": self._dense_top_k,
            "bm25_top_k": self._bm25_top_k,
            "top_k": self._top_k,
            "filters": _metadata_filters_profile(filters),
            "fusion": {
                "type": self._fusion_type,
                "rrf_k": self._rrf_k,
                "signal_weights": dict(self._signal_weights),
            },
            "vector_store": _retriever_profile(self._dense_retriever).get("vector_store") if _retriever_profile(self._dense_retriever) else None,
            "docstore": _retriever_profile(self._bm25_retriever).get("docstore") if _retriever_profile(self._bm25_retriever) else None,
            "bm25": _retriever_profile(self._bm25_retriever).get("bm25") if _retriever_profile(self._bm25_retriever) else None,
        }

    def retrieve(
        self,
        query: str,
        *,
        timeout_seconds: float | None = None,
        query_constraints: Any | None = None,
        **kwargs: Any,
    ) -> list[NodeWithScore]:
        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))

        dense_results = self._dense_retriever.retrieve(
            query,
            timeout_seconds=_remaining_timeout(deadline, "dense"),
            query_constraints=query_constraints,
            **kwargs,
        )
        bm25_results = self._bm25_retriever.retrieve(
            query,
            timeout_seconds=_remaining_timeout(deadline, "bm25"),
            query_constraints=query_constraints,
            **kwargs,
        )
        return self._fuse_results(dense_results=dense_results, bm25_results=bm25_results)

    def retriever_profile(self) -> dict[str, Any]:
        """Return the active retriever profile."""
        return dict(self._profile)

    def retriever_fingerprint(self) -> str:
        """Return the active retriever fingerprint."""
        return retriever_fingerprint(self._profile)

    def _fuse_results(
        self,
        *,
        dense_results: list[NodeWithScore],
        bm25_results: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        merged: dict[str, _FusedCandidate] = {}

        for rank, item in enumerate(dense_results, start=1):
            node_with_score = _coerce_node_with_score(item)
            node = node_with_score.node
            node_id = _node_id(node)
            entry = merged.setdefault(node_id, _FusedCandidate(node=node))
            entry.node = _preferred_node(entry.node, node, existing_score=entry.best_score, candidate_score=node_with_score.score)
            entry.best_score = _better_score(entry.best_score, node_with_score.score)
            entry.dense_rank = min(entry.dense_rank, rank) if entry.dense_rank is not None else rank
            entry.dense_score = _better_score(entry.dense_score, node_with_score.score)

        for rank, item in enumerate(bm25_results, start=1):
            node_with_score = _coerce_node_with_score(item)
            node = node_with_score.node
            node_id = _node_id(node)
            entry = merged.setdefault(node_id, _FusedCandidate(node=node))
            entry.node = _preferred_node(entry.node, node, existing_score=entry.best_score, candidate_score=node_with_score.score)
            entry.best_score = _better_score(entry.best_score, node_with_score.score)
            entry.bm25_rank = min(entry.bm25_rank, rank) if entry.bm25_rank is not None else rank
            entry.bm25_score = _better_score(entry.bm25_score, node_with_score.score)

        items: list[tuple[float, float, str, NodeWithScore]] = []
        for node_id, entry in merged.items():
            fusion_score = 0.0
            if entry.dense_rank is not None:
                fusion_score += self._signal_weights["dense"] / float(self._rrf_k + entry.dense_rank)
            if entry.bm25_rank is not None:
                fusion_score += self._signal_weights["bm25"] / float(self._rrf_k + entry.bm25_rank)
            entry.fusion_score = fusion_score
            _stamp_signal_trace(
                entry.node,
                {
                    "signal_type": "hybrid_rrf",
                    "hybrid_kind": "dense_bm25",
                    "fusion_type": self._fusion_type,
                    "rrf_k": self._rrf_k,
                    "signal_weights": dict(self._signal_weights),
                    "dense_rank": entry.dense_rank,
                    "dense_score": _float_or_none(entry.dense_score),
                    "bm25_rank": entry.bm25_rank,
                    "bm25_score": _float_or_none(entry.bm25_score),
                    "fusion_score": fusion_score,
                },
            )
            items.append(
                (
                    fusion_score,
                    _tie_break_score(entry),
                    node_id,
                    NodeWithScore(node=entry.node, score=fusion_score),
                )
            )

        items.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [item[3] for item in items[: self._top_k]]


class HybridRetriever:
    """Hybrid dense+sparse retriever with client-side within-source fusion."""

    def __init__(
        self,
        *,
        storage_context: StorageContext,
        filters: Optional[MetadataFilters] = None,
        top_k: Optional[int] = 5,
        vector_store: QdrantIndexStore | None = None,
        retrieval_profile: Mapping[str, Any] | None = None,
        sparse_query_expander: DeterministicSparseQueryExpander | None = None,
        **_: Any,
    ) -> None:
        profile = dict(retrieval_profile or {})
        fusion_cfg = _as_mapping(profile.get("fusion", {}))
        signal_weights = _as_mapping(fusion_cfg.get("signal_weights", {}))

        self._filters = filters
        self._vector_store = _resolve_native_vector_store(storage_context, vector_store)
        if self._vector_store is None:
            raise ValueError(
                "HybridRetriever requires a native vector store wrapper with dense and sparse query support."
            )

        self._dense_top_k = max(1, int(profile.get("dense_top_k", top_k or 5)))
        self._sparse_top_k = max(
            1,
            int(profile.get("sparse_top_k", profile.get("dense_top_k", top_k or 5))),
        )
        self._top_k = max(1, int(profile.get("top_k", top_k or 5)))
        self._fusion_type = str(fusion_cfg.get("type", "rrf") or "rrf").strip().lower()
        if self._fusion_type != "rrf":
            raise ValueError(
                f"Unsupported hybrid fusion type {self._fusion_type!r}. Supported fusion types: ['rrf']."
            )
        self._rrf_k = max(1, int(fusion_cfg.get("rrf_k", 60) or 60))
        self._signal_weights = {
            "dense": _coerce_float(signal_weights.get("dense"), 1.0),
            "sparse": _coerce_float(signal_weights.get("sparse"), 1.0),
        }
        self._expander = sparse_query_expander
        self._profile = {
            "type": "hybrid",
            "dense_top_k": self._dense_top_k,
            "sparse_top_k": self._sparse_top_k,
            "top_k": self._top_k,
            "filters": _metadata_filters_profile(filters),
            "fusion": {
                "type": self._fusion_type,
                "rrf_k": self._rrf_k,
                "signal_weights": dict(self._signal_weights),
            },
            "vector_store": _vector_store_profile(self._vector_store),
            "sparse_query_expander": _sparse_query_expander_profile(self._expander),
        }

    def retrieve(
        self,
        query: str,
        *,
        timeout_seconds: float | None = None,
        query_constraints: Any | None = None,
        **kwargs: Any,
    ) -> list[NodeWithScore]:
        _ = kwargs
        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))

        dense_results = self._vector_store.query_dense_nodes(
            query,
            top_k=self._dense_top_k,
            filters=self._filters,
            timeout_seconds=_remaining_timeout(deadline, "dense"),
        )
        expansion = _expand_sparse_query(self._expander, query, query_constraints)
        sparse_results = self._vector_store.query_sparse_nodes(
            expansion.query_text,
            top_k=self._sparse_top_k,
            filters=self._filters,
            timeout_seconds=_remaining_timeout(deadline, "sparse"),
        )
        return self._fuse_results(dense_results=dense_results, sparse_results=sparse_results, expansion=expansion)

    def retriever_profile(self) -> dict[str, Any]:
        """Return the active retriever profile."""
        return dict(self._profile)

    def retriever_fingerprint(self) -> str:
        """Return the active retriever fingerprint."""
        return retriever_fingerprint(self._profile)

    def _fuse_results(
        self,
        *,
        dense_results: list[NodeWithScore],
        sparse_results: list[NodeWithScore],
        expansion: SparseQueryExpansion,
    ) -> list[NodeWithScore]:
        merged: dict[str, _FusedCandidate] = {}

        for rank, item in enumerate(dense_results, start=1):
            node_with_score = _coerce_node_with_score(item)
            node = node_with_score.node
            node_id = _node_id(node)
            entry = merged.setdefault(node_id, _FusedCandidate(node=node))
            entry.node = _preferred_node(entry.node, node, existing_score=entry.best_score, candidate_score=node_with_score.score)
            entry.best_score = _better_score(entry.best_score, node_with_score.score)
            entry.dense_rank = min(entry.dense_rank, rank) if entry.dense_rank is not None else rank
            entry.dense_score = _better_score(entry.dense_score, node_with_score.score)

        for rank, item in enumerate(sparse_results, start=1):
            node_with_score = _coerce_node_with_score(item)
            node = node_with_score.node
            node_id = _node_id(node)
            entry = merged.setdefault(node_id, _FusedCandidate(node=node))
            entry.node = _preferred_node(entry.node, node, existing_score=entry.best_score, candidate_score=node_with_score.score)
            entry.best_score = _better_score(entry.best_score, node_with_score.score)
            entry.sparse_rank = min(entry.sparse_rank, rank) if entry.sparse_rank is not None else rank
            entry.sparse_score = _better_score(entry.sparse_score, node_with_score.score)

        items: list[tuple[float, float, str, NodeWithScore]] = []
        for node_id, entry in merged.items():
            fusion_score = 0.0
            if entry.dense_rank is not None:
                fusion_score += self._signal_weights["dense"] / float(self._rrf_k + entry.dense_rank)
            if entry.sparse_rank is not None:
                fusion_score += self._signal_weights["sparse"] / float(self._rrf_k + entry.sparse_rank)
            entry.fusion_score = fusion_score
            _stamp_signal_trace(
                entry.node,
                {
                    "signal_type": "hybrid_rrf",
                    "fusion_type": self._fusion_type,
                    "rrf_k": self._rrf_k,
                    "signal_weights": dict(self._signal_weights),
                    "dense_rank": entry.dense_rank,
                    "dense_score": _float_or_none(entry.dense_score),
                    "sparse_rank": entry.sparse_rank,
                    "sparse_score": _float_or_none(entry.sparse_score),
                    "fusion_score": fusion_score,
                    "expansion_terms": list(expansion.expansion_terms),
                    "expanded_sparse_query": expansion.query_text,
                },
            )
            items.append(
                (
                    fusion_score,
                    _tie_break_score(entry),
                    node_id,
                    NodeWithScore(node=entry.node, score=fusion_score),
                )
            )

        items.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [item[3] for item in items[: self._top_k]]


class MultiCollectionRetriever:
    """Retrieve from multiple source retrievers and rerank merged candidates."""

    def __init__(
        self,
        *,
        source_retrievers: Mapping[str, Any],
        source_settings: Mapping[str, Mapping[str, Any]] | None = None,
        final_top_k: int = 10,
        rerank: Mapping[str, Any] | None = None,
        config_base_dir: str | Path | None = None,
    ) -> None:
        if not source_retrievers:
            raise ValueError("'source_retrievers' must define at least one source retriever.")

        self.source_retrievers = dict(source_retrievers)
        self.source_settings = {
            str(k): dict(v or {})
            for k, v in (source_settings or {}).items()
        }
        self.final_top_k = max(1, int(final_top_k))
        self._reranker = create_reranker(
            config=rerank,
            source_settings=self.source_settings,
            config_base_dir=config_base_dir,
        )
        self._profile = {
            "type": "multi_collection",
            "final_top_k": self.final_top_k,
            "sources": {
                source_name: {
                    "settings": _stable_json_value(dict(self.source_settings.get(source_name, {}))),
                    "retriever_profile": _retriever_profile(self.source_retrievers[source_name]),
                    "retriever_fingerprint": _retriever_fingerprint(self.source_retrievers[source_name]),
                }
                for source_name in sorted(self.source_retrievers.keys())
            },
        }

    def retrieve(
        self,
        query: str,
        *,
        timeout_seconds: float | None = None,
        query_constraints: Any | None = None,
        **kwargs: Any,
    ) -> list[NodeWithScore]:
        candidates = self._collect_candidates(
            query,
            timeout_seconds=timeout_seconds,
            query_constraints=query_constraints,
            **kwargs,
        )
        if not candidates:
            return []

        for candidate in candidates:
            self._stamp_source_metadata(
                node=candidate.node,
                source_ranks=candidate.source_ranks,
            )

        reranked = self._reranker.rerank(
            candidates,
            query_constraints=query_constraints,
        )
        return reranked[: self.final_top_k]

    def retriever_profile(self) -> dict[str, Any]:
        """Return the active retriever profile."""
        return dict(self._profile)

    def retriever_fingerprint(self) -> str:
        """Return the active retriever fingerprint."""
        return retriever_fingerprint(self._profile)

    def reranker_profile(self) -> dict[str, Any]:
        """Return the active reranker profile."""
        return self._reranker.profile()

    def reranker_fingerprint(self) -> str:
        """Return the active reranker fingerprint."""
        return self._reranker.fingerprint()

    def _collect_candidates(
        self,
        query: str,
        *,
        timeout_seconds: float | None = None,
        query_constraints: Any | None = None,
        **kwargs: Any,
    ) -> list[MergedCandidate]:
        merged: dict[str, MergedCandidate] = {}
        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))

        for source_name, retriever in self.source_retrievers.items():
            remaining_timeout = _remaining_timeout(deadline, source_name)
            try:
                source_results = retriever.retrieve(
                    query,
                    timeout_seconds=remaining_timeout,
                    query_constraints=query_constraints,
                    **kwargs,
                )
            except Exception as exc:
                if is_timeout_exception(exc):
                    timeout_value = remaining_timeout if remaining_timeout is not None else float(timeout_seconds or 0.0)
                    raise RetrievalTimeoutError(
                        f"multi-collection source {source_name!r} timed out after {max(timeout_value, 0.0):.3f}s"
                    ) from exc
                raise

            for rank, item in enumerate(source_results, start=1):
                node_with_score = _coerce_node_with_score(item)
                node = node_with_score.node
                node_id = _node_id(node)
                raw_score = _float_or_none(node_with_score.score)

                entry = merged.setdefault(
                    node_id,
                    MergedCandidate(
                        node=node,
                        best_score=raw_score,
                        source_ranks={},
                    ),
                )

                if entry.best_score is None or (
                    raw_score is not None and raw_score > entry.best_score
                ):
                    entry.best_score = raw_score
                    entry.node = node

                prev_rank = entry.source_ranks.get(source_name)
                if prev_rank is None or rank < prev_rank:
                    entry.source_ranks[source_name] = rank

        return list(merged.values())

    @staticmethod
    def _stamp_source_metadata(node: Any, source_ranks: Mapping[str, int]) -> None:
        metadata = getattr(node, "metadata", None)
        if not isinstance(metadata, dict):
            return
        sorted_sources = sorted(source_ranks.keys())
        metadata["retrieval_sources"] = sorted_sources
        metadata["retrieval_source_ranks"] = dict(source_ranks)
        metadata["retrieval_source"] = (
            sorted_sources[0] if len(sorted_sources) == 1 else "multi"
        )


class _BM25Corpus:
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")

    def __init__(
        self,
        *,
        nodes: list[Any],
        postings: Mapping[str, list[tuple[int, int]]],
        document_frequencies: Mapping[str, int],
        document_lengths: list[int],
        avg_document_length: float,
        k1: float,
        b: float,
    ) -> None:
        self._nodes = list(nodes)
        self._postings = {str(term): list(entries) for term, entries in postings.items()}
        self._document_frequencies = {str(term): int(freq) for term, freq in document_frequencies.items()}
        self._document_lengths = [int(length) for length in document_lengths]
        self._avg_document_length = max(float(avg_document_length), 1.0)
        self._k1 = max(0.0, float(k1))
        self._b = max(0.0, float(b))

    @classmethod
    def from_docstore(
        cls,
        docstore: Any,
        *,
        k1: float,
        b: float,
    ) -> "_BM25Corpus":
        nodes = _docstore_nodes(docstore)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequencies: dict[str, int] = defaultdict(int)
        document_lengths: list[int] = []

        for doc_index, node in enumerate(nodes):
            tokens = cls.tokenize(_node_text(node))
            document_lengths.append(len(tokens))
            term_counts = Counter(tokens)
            for term, freq in term_counts.items():
                postings[term].append((doc_index, int(freq)))
                document_frequencies[term] += 1

        avg_document_length = (
            sum(document_lengths) / float(len(document_lengths))
            if document_lengths else 1.0
        )
        return cls(
            nodes=nodes,
            postings=postings,
            document_frequencies=document_frequencies,
            document_lengths=document_lengths,
            avg_document_length=avg_document_length,
            k1=k1,
            b=b,
        )

    @property
    def document_count(self) -> int:
        return len(self._nodes)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return [match.group(0).lower() for match in cls.TOKEN_PATTERN.finditer(str(text or ""))]

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: MetadataFilters | dict | None = None,
        timeout_seconds: float | None = None,
    ) -> list[NodeWithScore]:
        if not self._nodes:
            return []

        query_terms = Counter(self.tokenize(query))
        if not query_terms:
            return []

        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))

        scores: dict[int, float] = defaultdict(float)
        document_count = len(self._nodes)
        for term, query_term_frequency in query_terms.items():
            _ensure_bm25_time_remaining(deadline)
            postings = self._postings.get(term)
            if not postings:
                continue
            document_frequency = self._document_frequencies.get(term, 0)
            if document_frequency <= 0:
                continue
            idf = math.log(1.0 + ((document_count - document_frequency + 0.5) / (document_frequency + 0.5)))
            for doc_index, term_frequency in postings:
                node = self._nodes[doc_index]
                if not _node_matches_filters(node, filters):
                    continue
                doc_length = self._document_lengths[doc_index] if doc_index < len(self._document_lengths) else 0
                norm = self._k1 * (1.0 - self._b + self._b * (float(doc_length) / self._avg_document_length))
                denominator = float(term_frequency) + norm
                if denominator <= 0.0:
                    continue
                scores[doc_index] += (
                    idf
                    * ((float(term_frequency) * (self._k1 + 1.0)) / denominator)
                    * float(query_term_frequency)
                )

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], _node_id(self._nodes[item[0]])),
        )
        results: list[NodeWithScore] = []
        for doc_index, score in ranked[: max(1, int(top_k))]:
            results.append(
                NodeWithScore(
                    node=_clone_node(self._nodes[doc_index]),
                    score=float(score),
                )
            )
        return results


@dataclass
class _FusedCandidate:
    node: Any
    best_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None


def _resolve_native_vector_store(
    storage_context: StorageContext,
    vector_store: QdrantIndexStore | None,
) -> QdrantIndexStore | None:
    for candidate in [vector_store, getattr(storage_context, "vector_store", None)]:
        if hasattr(candidate, "query_dense_nodes"):
            return candidate  # type: ignore[return-value]
    return None


def _expand_sparse_query(
    expander: DeterministicSparseQueryExpander | None,
    query_text: str,
    query_constraints: QueryConstraints | Mapping[str, Any] | None,
) -> SparseQueryExpansion:
    if expander is None:
        return SparseQueryExpansion(query_text=str(query_text or ""), expansion_terms=[])
    return expander.expand(query_text, query_constraints)


def _stamp_dense_results(results: list[NodeWithScore]) -> list[NodeWithScore]:
    stamped: list[NodeWithScore] = []
    for rank, item in enumerate(results, start=1):
        node_with_score = _coerce_node_with_score(item)
        _stamp_signal_trace(
            node_with_score.node,
            {
                "signal_type": "dense",
                "dense_rank": rank,
                "dense_score": _float_or_none(node_with_score.score),
            },
        )
        stamped.append(node_with_score)
    return stamped


def _stamp_sparse_results(
    results: list[NodeWithScore],
    *,
    expansion: SparseQueryExpansion,
) -> list[NodeWithScore]:
    stamped: list[NodeWithScore] = []
    for rank, item in enumerate(results, start=1):
        node_with_score = _coerce_node_with_score(item)
        _stamp_signal_trace(
            node_with_score.node,
            {
                "signal_type": "sparse",
                "sparse_rank": rank,
                "sparse_score": _float_or_none(node_with_score.score),
                "expansion_terms": list(expansion.expansion_terms),
                "expanded_sparse_query": expansion.query_text,
            },
        )
        stamped.append(node_with_score)
    return stamped


def _stamp_bm25_results(results: list[NodeWithScore]) -> list[NodeWithScore]:
    stamped: list[NodeWithScore] = []
    for rank, item in enumerate(results, start=1):
        node_with_score = _coerce_node_with_score(item)
        _stamp_signal_trace(
            node_with_score.node,
            {
                "signal_type": "bm25",
                "bm25_rank": rank,
                "bm25_score": _float_or_none(node_with_score.score),
            },
        )
        stamped.append(node_with_score)
    return stamped


def _stamp_signal_trace(node: Any, trace: Mapping[str, Any]) -> None:
    metadata = getattr(node, "metadata", None)
    if not isinstance(metadata, dict):
        return
    existing = metadata.get("retrieval_signal_trace")
    merged = dict(existing) if isinstance(existing, Mapping) else {}
    merged.update(_stable_json_value(dict(trace)))
    metadata["retrieval_signal_trace"] = merged


def _remaining_timeout(deadline: float | None, stage: str) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RetrievalTimeoutError(f"retrieval timed out before querying {stage!r}")
    return remaining


def _ensure_bm25_time_remaining(deadline: float | None) -> None:
    if deadline is None:
        return
    if deadline - time.monotonic() <= 0.0:
        raise RetrievalTimeoutError("retrieval timed out during 'bm25' scoring")


def _coerce_node_with_score(item: Any) -> NodeWithScore:
    if isinstance(item, NodeWithScore):
        return item
    node = getattr(item, "node", item)
    return NodeWithScore(node=node, score=_float_or_none(getattr(item, "score", None)))


def _node_id(node: Any) -> str:
    for attr in ("id_", "node_id", "id"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value:
            return value
    return f"<anon-node:{id(node)}>"


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _better_score(existing: float | None, candidate: float | None) -> float | None:
    if existing is None:
        return candidate
    if candidate is None:
        return existing
    return candidate if candidate > existing else existing


def _preferred_node(
    existing_node: Any,
    candidate_node: Any,
    *,
    existing_score: float | None,
    candidate_score: float | None,
) -> Any:
    if existing_node is None:
        return candidate_node
    better = _better_score(existing_score, candidate_score)
    if better is not None and candidate_score == better:
        return candidate_node
    return existing_node


def _tie_break_score(entry: _FusedCandidate) -> float:
    return max(
        score
        for score in [entry.best_score, entry.dense_score, entry.sparse_score, entry.bm25_score]
        if score is not None
    ) if any(
        score is not None
        for score in [entry.best_score, entry.dense_score, entry.sparse_score, entry.bm25_score]
    ) else 0.0


def _clone_node(node: Any) -> Any:
    try:
        return copy.deepcopy(node)
    except Exception:
        return node


def _node_text(node: Any) -> str:
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text
    getter = getattr(node, "get_content", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            value = None
        if isinstance(value, str):
            return value
    return ""


def _docstore_nodes(docstore: Any) -> list[Any]:
    docs = getattr(docstore, "docs", None)
    if not isinstance(docs, Mapping):
        return []
    items = sorted(docs.items(), key=lambda item: str(item[0]))
    return [node for _, node in items if node is not None]


def _vector_store_profile(vector_store: QdrantIndexStore | None) -> dict[str, Any] | None:
    if vector_store is None:
        return None
    getter = getattr(vector_store, "profile", None)
    if callable(getter):
        try:
            profile = getter()
        except Exception:
            return None
        return _stable_json_value(profile)
    return None


def _docstore_profile(docstore: Any, *, node_count: int | None = None) -> dict[str, Any] | None:
    if docstore is None:
        return None
    profile = {
        "backend": type(docstore).__name__,
        "node_count": int(node_count) if node_count is not None else len(_docstore_nodes(docstore)),
    }
    return _stable_json_value(profile)


def _retriever_profile(retriever: Any) -> dict[str, Any] | None:
    getter = getattr(retriever, "retriever_profile", None)
    if callable(getter):
        try:
            return _stable_json_value(getter())
        except Exception:
            return None
    return None


def _retriever_fingerprint(retriever: Any) -> str | None:
    getter = getattr(retriever, "retriever_fingerprint", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            return None
        text = str(value or "").strip()
        return text or None
    return None


def _sparse_query_expander_profile(
    expander: DeterministicSparseQueryExpander | None,
) -> dict[str, Any] | None:
    if expander is None:
        return None
    getter = getattr(expander, "profile", None)
    if callable(getter):
        try:
            return _stable_json_value(getter())
        except Exception:
            return None
    return {"type": type(expander).__name__}


def _metadata_filters_profile(filters: MetadataFilters | None) -> dict[str, Any] | None:
    if filters is None:
        return None
    if hasattr(filters, "model_dump"):
        try:
            return _stable_json_value(filters.model_dump())  # type: ignore[return-value]
        except Exception:
            return None
    result: dict[str, Any] = {}
    condition = getattr(filters, "condition", None)
    if condition is not None:
        result["condition"] = str(condition)
    items: list[dict[str, Any]] = []
    for item in list(getattr(filters, "filters", []) or []):
        if isinstance(item, MetadataFilter):
            items.append(
                {
                    "key": str(getattr(item, "key", "") or ""),
                    "value": _stable_json_value(getattr(item, "value", None)),
                    "operator": str(getattr(item, "operator", "") or ""),
                }
            )
    result["filters"] = items
    return result


def _node_matches_filters(
    node: Any,
    filters: MetadataFilters | dict | None,
) -> bool:
    if filters is None:
        return True
    metadata = getattr(node, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if isinstance(filters, dict):
        return all(metadata.get(key) == value for key, value in filters.items())
    if not isinstance(filters, MetadataFilters):
        return True

    child_results = [
        _node_matches_filters(node, item) if isinstance(item, MetadataFilters)
        else _metadata_filter_matches(metadata, item)
        for item in list(getattr(filters, "filters", []) or [])
    ]
    raw_condition = getattr(filters, "condition", "and")
    condition = str(getattr(raw_condition, "value", raw_condition) or "and").lower()
    if condition == "or":
        return any(child_results)
    if condition == "not":
        return not any(child_results)
    return all(child_results)


def _metadata_filter_matches(metadata: Mapping[str, Any], metadata_filter: MetadataFilter) -> bool:
    key = str(getattr(metadata_filter, "key", "") or "")
    raw_operator = getattr(metadata_filter, "operator", "==")
    operator = str(getattr(raw_operator, "value", raw_operator) or "==").lower()
    expected = getattr(metadata_filter, "value", None)
    actual = metadata.get(key)

    if operator == "==":
        return actual == expected
    if operator == "!=":
        return actual != expected
    if operator == ">":
        lhs, rhs = _ordered_values(actual, expected)
        return lhs > rhs
    if operator == ">=":
        lhs, rhs = _ordered_values(actual, expected)
        return lhs >= rhs
    if operator == "<":
        lhs, rhs = _ordered_values(actual, expected)
        return lhs < rhs
    if operator == "<=":
        lhs, rhs = _ordered_values(actual, expected)
        return lhs <= rhs
    if operator == "in":
        return actual in expected if isinstance(expected, (list, tuple, set, frozenset)) else False
    if operator == "nin":
        return actual not in expected if isinstance(expected, (list, tuple, set, frozenset)) else True
    if operator == "any":
        if not isinstance(actual, (list, tuple, set, frozenset)):
            return False
        expected_values = expected if isinstance(expected, (list, tuple, set, frozenset)) else [expected]
        return any(value in actual for value in expected_values)
    if operator == "all":
        if not isinstance(actual, (list, tuple, set, frozenset)):
            return False
        expected_values = expected if isinstance(expected, (list, tuple, set, frozenset)) else [expected]
        return all(value in actual for value in expected_values)
    return actual == expected


def _ordered_values(left: Any, right: Any) -> tuple[Any, Any]:
    try:
        return float(left), float(right)
    except Exception:
        return str(left), str(right)


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_json_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int, float, bool)):
        return getattr(value, "value")
    return str(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


__all__ = [
    "BM25IndexRetriever",
    "DenseBM25HybridRetriever",
    "HybridRetriever",
    "MultiCollectionRetriever",
    "SparseIndexRetriever",
    "VectorIndexRetriever",
    "retriever_fingerprint",
]
