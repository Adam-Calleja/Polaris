"""polaris_rag.retrieval.retriever

Retriever implementations for the Polaris RAG system.

This module provides concrete retriever classes that wrap LlamaIndex
retrievers to fetch relevant document nodes for a natural-language query.
Both vector-only and hybrid (vector + BM25) retrieval strategies are
supported.

Classes
-------
VectorIndexRetriever
    Vector-based retriever backed by a LlamaIndex VectorStoreIndex.
HybridRetriever
    Hybrid retriever combining vector similarity and BM25 keyword search
    using query fusion.
MultiCollectionRetriever
    Orchestrates retrieval from multiple source retrievers, then reranks.
"""

from pathlib import Path
from queue import Queue
import threading
import time

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.core.schema import NodeWithScore
from llama_index.core import StorageContext

from typing import Any, Mapping, Optional
from polaris_rag.common.request_budget import RetrievalTimeoutError, is_timeout_exception
from polaris_rag.generation.llm_interface import BaseLLM
from polaris_rag.retrieval.embedder import BaseEmbedder
from polaris_rag.retrieval.reranker import MergedCandidate, create_reranker
from polaris_rag.retrieval.vector_store import QdrantIndexStore

class VectorIndexRetriever:
    """Vector-based retriever over a vector store index.

    This class wraps a LlamaIndex ``VectorStoreIndex`` retriever to provide a
    simple interface for retrieving nodes based on semantic similarity and
    optional metadata filters.

    Parameters
    ----------
    storage_context : StorageContext
        Storage context providing access to the underlying vector store.
    filters : MetadataFilters, optional
        Metadata filters applied during retrieval. Defaults to ``None``.
    top_k : int, optional
        Maximum number of results to return. Defaults to ``5``.

    Attributes
    ----------
    _retriever : Any
        Internal LlamaIndex retriever instance.
    """

    def __init__(
            self,
            *,
            storage_context: StorageContext,
            filters: Optional[MetadataFilters] = None,
            top_k: Optional[int] = 5,
            vector_store: QdrantIndexStore | None = None,
            embedder: BaseEmbedder | None = None,
        ):
        """Initialise a VectorIndexRetriever.

        Parameters
        ----------
        storage_context : StorageContext
            Storage context providing access to the underlying vector store.
        filters : MetadataFilters, optional
            Metadata filters applied during retrieval. Defaults to ``None``.
        top_k : int, optional
            Maximum number of results to return. Defaults to ``5``.
        """
        vector_store_impl = storage_context.vector_store
        vector_index = vector_store_impl._index

        self._retriever = vector_index.as_retriever(
            similarity_top_k=top_k,
            filters=filters
        )
        self._top_k = int(top_k or 5)
        self._filters = filters
        self._vector_store_wrapper = vector_store if hasattr(vector_store, "query_nodes") else None
        self._embedder = embedder

    def retrieve(
            self, 
            query: str,
            *,
            timeout_seconds: float | None = None,
            query_constraints: Any | None = None,
            **kwargs,
        ) -> list[NodeWithScore]:
        """Retrieve nodes for a query.

        Parameters
        ----------
        query : str
            Natural-language query string.

        Returns
        -------
        list[NodeWithScore]
            Retrieved nodes with associated similarity scores.
        """
        _ = query_constraints
        if self._vector_store_wrapper is not None:
            return self._vector_store_wrapper.query_nodes(
                query,
                top_k=self._top_k,
                filters=self._filters,
                timeout_seconds=timeout_seconds,
            )
        return self._retriever.retrieve(query)
    
class HybridRetriever:
    """Hybrid retriever combining vector and keyword search.

    This retriever fuses results from a vector-based retriever and a BM25
    retriever using LlamaIndex's :class:`QueryFusionRetriever` to improve
    recall and ranking quality.

    Parameters
    ----------
    storage_context : StorageContext
        Storage context providing access to vector and document stores.
    filters : MetadataFilters, optional
        Metadata filters applied during retrieval. Defaults to ``None``.
    top_k : int, optional
        Maximum number of results to return. Defaults to ``5``.
    llm : BaseLLM or None, optional
        Language model used for query fusion. If ``None``, query fusion
        operates without LLM assistance.

    Attributes
    ----------
    _retriever : QueryFusionRetriever
        Internal fused retriever instance.
    """

    def __init__(
            self,
            *,
            storage_context: StorageContext,
            filters: Optional[MetadataFilters] = None,
            top_k: Optional[int] = 5,
            llm: BaseLLM = None,
            vector_store: QdrantIndexStore | None = None,
            embedder: BaseEmbedder | None = None,
        ):
        """Initialise a HybridRetriever.

        Parameters
        ----------
        storage_context : StorageContext
            Storage context providing access to vector and document stores.
        filters : MetadataFilters, optional
            Metadata filters applied during retrieval. Defaults to ``None``.
        top_k : int, optional
            Maximum number of results to return. Defaults to ``5``.
        llm : BaseLLM or None, optional
            Language model used for query fusion. Defaults to ``None``.
        """
        vector_store = storage_context.vector_store
        vector_index = vector_store._index
        docstore = storage_context.docstore

        vector_retriever = vector_index.as_retriever(
            similarity_top_k=top_k,
            filters=filters
        )

        bm25_retriever = BM25Retriever.from_defaults(
            docstore=docstore, 
            similarity_top_k=top_k,
        )

        fusion_llm = llm

        self._retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            llm=fusion_llm,
            similarity_top_k=top_k,
            num_queries=4,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
        )

    def retrieve(
            self, 
            query: str,
            *,
            timeout_seconds: float | None = None,
            query_constraints: Any | None = None,
            **kwargs,
        ) -> list[NodeWithScore]:
        """Retrieve nodes for a query.

        Parameters
        ----------
        query : str
            Natural-language query string.

        Returns
        -------
        list[NodeWithScore]
            Retrieved nodes with associated scores.
        """
        _ = query_constraints
        if timeout_seconds is None:
            return self._retriever.retrieve(query)

        result_queue: Queue[tuple[bool, object]] = Queue(maxsize=1)

        def _worker() -> None:
            try:
                result_queue.put((True, self._retriever.retrieve(query)))
            except Exception as exc:  # pragma: no cover - exercised via caller behavior
                result_queue.put((False, exc))

        worker = threading.Thread(
            target=_worker,
            name="polaris-hybrid-retrieval",
            daemon=True,
        )
        worker.start()
        worker.join(timeout=float(timeout_seconds))
        if worker.is_alive():
            raise RetrievalTimeoutError(
                f"hybrid retrieval timed out after {float(timeout_seconds):.3f}s"
            )

        succeeded, payload = result_queue.get_nowait()
        if not succeeded:
            raise payload  # type: ignore[misc]
        return payload  # type: ignore[return-value]


class MultiCollectionRetriever:
    """Retrieve from multiple source retrievers and rerank merged candidates.

    Parameters
    ----------
    source_retrievers : Mapping[str, Any]
        Mapping from source name to retriever instance. Each retriever must
        implement ``retrieve(query: str, *, timeout_seconds: float | None = None)``.
    source_settings : Mapping[str, Mapping[str, Any]] or None, optional
        Optional per-source settings map. Currently reads ``weight`` for RRF.
    final_top_k : int, optional
        Number of nodes to return after reranking. Defaults to ``10``.
    rerank : Mapping[str, Any] or None, optional
        Reranker configuration. Supported:
        - ``{"type": "rrf", "rrf_k": 60}`` (default)

    Notes
    -----
    - Candidate deduplication is performed by node id.
    - Source provenance is stamped into node metadata:
      - ``retrieval_source``
      - ``retrieval_sources``
      - ``retrieval_source_ranks``
    - When ``timeout_seconds`` is provided, the timeout is treated as a shared
      budget across all source retrievers rather than a fresh timeout per source.
    """

    def __init__(
            self,
            *,
            source_retrievers: Mapping[str, Any],
            source_settings: Mapping[str, Mapping[str, Any]] | None = None,
            final_top_k: int = 10,
            rerank: Mapping[str, Any] | None = None,
            config_base_dir: str | Path | None = None,
        ):
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

    def retrieve(
            self,
            query: str,
            *,
            timeout_seconds: float | None = None,
            query_constraints: Any | None = None,
            **kwargs,
        ) -> list[NodeWithScore]:
        """Retrieve and rerank nodes for a query."""
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
            **kwargs,
        ) -> list[MergedCandidate]:
        """Collect and deduplicate candidates returned by each source retriever."""
        merged: dict[str, MergedCandidate] = {}
        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))

        for source_name, retriever in self.source_retrievers.items():
            remaining_timeout = self._remaining_timeout(deadline, source_name)
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
                node_with_score = self._coerce_node_with_score(item)
                node = node_with_score.node
                node_id = self._node_id(node)
                raw_score = self._to_float_or_none(node_with_score.score)

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
    def _remaining_timeout(deadline: float | None, source_name: str) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RetrievalTimeoutError(
                f"multi-collection retrieval timed out before querying source {source_name!r}"
            )
        return remaining

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _coerce_node_with_score(item: Any) -> NodeWithScore:
        """Coerce retrieval output item into NodeWithScore."""
        if isinstance(item, NodeWithScore):
            return item

        node = getattr(item, "node", item)
        score = getattr(item, "score", None)
        score_val = MultiCollectionRetriever._to_float_or_none(score)
        return NodeWithScore(node=node, score=score_val)

    @staticmethod
    def _node_id(node: Any) -> str:
        """Extract a stable identifier for a node for deduplication."""
        for attr in ("id_", "node_id", "id"):
            value = getattr(node, attr, None)
            if isinstance(value, str) and value:
                return value
        return f"<anon-node:{id(node)}>"

    @staticmethod
    def _stamp_source_metadata(node: Any, source_ranks: Mapping[str, int]) -> None:
        """Annotate node metadata with retrieval provenance."""
        metadata = getattr(node, "metadata", None)
        if not isinstance(metadata, dict):
            return

        sorted_sources = sorted(source_ranks.keys())
        metadata["retrieval_sources"] = sorted_sources
        metadata["retrieval_source_ranks"] = dict(source_ranks)
        metadata["retrieval_source"] = (
            sorted_sources[0] if len(sorted_sources) == 1 else "multi"
        )
