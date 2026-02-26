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

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.core.schema import NodeWithScore
from llama_index.core import StorageContext

from typing import Any, Mapping, Optional
from polaris_rag.generation.llm_interface import BaseLLM

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
        vector_store = storage_context.vector_store
        vector_index = vector_store._index

        self._retriever = vector_index.as_retriever(
            similarity_top_k=top_k,
            filters=filters
        )

    def retrieve(
            self, 
            query: str,
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
        return self._retriever.retrieve(query)


class MultiCollectionRetriever:
    """Retrieve from multiple source retrievers and rerank merged candidates.

    Parameters
    ----------
    source_retrievers : Mapping[str, Any]
        Mapping from source name to retriever instance. Each retriever must
        implement ``retrieve(query: str) -> list[NodeWithScore]``.
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
    """

    def __init__(
            self,
            *,
            source_retrievers: Mapping[str, Any],
            source_settings: Mapping[str, Mapping[str, Any]] | None = None,
            final_top_k: int = 10,
            rerank: Mapping[str, Any] | None = None,
        ):
        if not source_retrievers:
            raise ValueError("'source_retrievers' must define at least one source retriever.")

        self.source_retrievers = dict(source_retrievers)
        self.source_settings = {
            str(k): dict(v or {})
            for k, v in (source_settings or {}).items()
        }

        self.final_top_k = max(1, int(final_top_k))
        self.rerank = dict(rerank or {})
        self.rerank_type = str(self.rerank.get("type", "rrf")).lower().strip()

        if self.rerank_type != "rrf":
            raise ValueError(
                f"Unsupported rerank type {self.rerank_type!r}. "
                "Supported rerankers: ['rrf']."
            )

        self.rrf_k = int(self.rerank.get("rrf_k", 60))
        if self.rrf_k <= 0:
            raise ValueError("'rerank.rrf_k' must be a positive integer.")

    def retrieve(
            self,
            query: str,
        ) -> list[NodeWithScore]:
        """Retrieve and rerank nodes for a query.

        Parameters
        ----------
        query : str
            Natural-language query string.

        Returns
        -------
        list[NodeWithScore]
            Final reranked list of nodes across all sources.
        """
        candidates = self._collect_candidates(query)
        if not candidates:
            return []

        reranked = self._rerank_rrf(candidates)
        return reranked[: self.final_top_k]

    def _collect_candidates(self, query: str) -> dict[str, dict[str, Any]]:
        """Collect and deduplicate candidates returned by each source retriever."""
        merged: dict[str, dict[str, Any]] = {}

        for source_name, retriever in self.source_retrievers.items():
            source_results = retriever.retrieve(query)
            for rank, item in enumerate(source_results, start=1):
                node_with_score = self._coerce_node_with_score(item)
                node = node_with_score.node
                node_id = self._node_id(node)

                entry = merged.setdefault(
                    node_id,
                    {
                        "node": node,
                        "best_score": self._to_float_or_none(node_with_score.score),
                        "source_ranks": {},
                    },
                )

                # Track the best raw retrieval score seen for stable tie-breaks.
                raw_score = self._to_float_or_none(node_with_score.score)
                if entry["best_score"] is None or (
                    raw_score is not None and raw_score > entry["best_score"]
                ):
                    entry["best_score"] = raw_score
                    entry["node"] = node

                prev_rank = entry["source_ranks"].get(source_name)
                if prev_rank is None or rank < prev_rank:
                    entry["source_ranks"][source_name] = rank

        return merged

    def _rerank_rrf(self, candidates: dict[str, dict[str, Any]]) -> list[NodeWithScore]:
        """Apply reciprocal-rank fusion over merged candidates."""
        scored: list[tuple[float, float, NodeWithScore]] = []

        for entry in candidates.values():
            source_ranks = entry["source_ranks"]
            rrf_score = 0.0
            for source_name, rank in source_ranks.items():
                weight = self._source_weight(source_name)
                rrf_score += weight / (self.rrf_k + int(rank))

            node = entry["node"]
            self._stamp_source_metadata(node=node, source_ranks=source_ranks)

            raw_score = entry["best_score"]
            tie_break = raw_score if raw_score is not None else float("-inf")
            scored.append((rrf_score, tie_break, NodeWithScore(node=node, score=rrf_score)))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored]

    def _source_weight(self, source_name: str) -> float:
        """Return configured source weight for RRF."""
        source_cfg = self.source_settings.get(source_name, {})
        weight_raw = source_cfg.get("weight", 1.0)
        try:
            return float(weight_raw)
        except Exception:
            return 1.0

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

        if len(sorted_sources) == 1:
            metadata["retrieval_source"] = sorted_sources[0]
        else:
            metadata["retrieval_source"] = "multi"
