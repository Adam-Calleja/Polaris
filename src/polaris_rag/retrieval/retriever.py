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

from queue import Queue
import threading

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.core.schema import NodeWithScore
from llama_index.core import StorageContext

from typing import Optional
from polaris_rag.common.request_budget import RetrievalTimeoutError
from polaris_rag.generation.llm_interface import BaseLLM
from polaris_rag.retrieval.embedder import BaseEmbedder
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
