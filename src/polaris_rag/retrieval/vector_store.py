"""polaris_rag.retrieval.vector_store

Vector store interfaces and factories for the retrieval layer.

This module defines a small wrapper interface around vector-store backends and
provides a concrete implementation backed by Qdrant and LlamaIndex. The main
responsibilities are:
- indexing :class:`~polaris_rag.common.schemas.DocumentChunk` objects into a vector store
- running similarity queries via a LlamaIndex query engine

Classes
-------
BaseVectorStore
    Abstract interface for vector store wrappers.
QdrantIndexStore
    Qdrant-backed vector store wrapper and LlamaIndex index manager.

Functions
---------
create_vector_store
    Create a vector store implementation from a configuration mapping.
"""
import asyncio
import math
import time
from typing import TYPE_CHECKING, Any, Iterable, Optional
from uuid import UUID, NAMESPACE_URL, uuid5
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as rest
from os import environ
import yaml
from abc import ABC, abstractmethod
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.response.schema import Response
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, VectorStoreQuery

from polaris_rag.common.schemas import DocumentChunk
from polaris_rag.common.request_budget import RetrievalTimeoutError, is_timeout_exception
from polaris_rag.retrieval.node_utils import chunk_to_text_node

if TYPE_CHECKING:
    from polaris_rag.generation.llm_interface import BaseLLM
    from polaris_rag.retrieval.embedder import BaseEmbedder
else:
    BaseLLM = Any
    BaseEmbedder = Any

QDRANT_POINT_ID_NAMESPACE = uuid5(NAMESPACE_URL, "polaris-rag/qdrant-point-id")


def qdrant_point_id_from_node_id(node_id: Any) -> int | str:
    """Return a Qdrant-compatible point id for an arbitrary logical node id."""
    if isinstance(node_id, int):
        if node_id < 0:
            raise ValueError("Qdrant point ids must be unsigned integers or UUID strings.")
        return node_id

    raw_node_id = str(node_id or "").strip()
    if not raw_node_id:
        raise ValueError("Node id must be non-empty.")

    try:
        return str(UUID(raw_node_id))
    except ValueError:
        return str(uuid5(QDRANT_POINT_ID_NAMESPACE, raw_node_id))


def _translate_node_ids_for_qdrant(node_ids: list[str] | None) -> list[int | str] | None:
    if node_ids is None:
        return None
    return [qdrant_point_id_from_node_id(node_id) for node_id in node_ids]


class PolarisQdrantVectorStore(QdrantVectorStore):
    """Qdrant adapter that maps arbitrary logical node ids to valid point ids."""

    def _build_points(self, nodes, sparse_vector_name):
        points, ids = super()._build_points(nodes, sparse_vector_name)
        for point, node_id in zip(points, ids):
            point.id = qdrant_point_id_from_node_id(node_id)
        return points, ids

    def get_nodes(self, node_ids=None, filters=None, limit=None, shard_identifier=None):
        return super().get_nodes(
            node_ids=_translate_node_ids_for_qdrant(node_ids),
            filters=filters,
            limit=limit,
            shard_identifier=shard_identifier,
        )

    async def aget_nodes(self, node_ids=None, filters=None, limit=None, shard_identifier=None):
        return await super().aget_nodes(
            node_ids=_translate_node_ids_for_qdrant(node_ids),
            filters=filters,
            limit=limit,
            shard_identifier=shard_identifier,
        )

    def delete_nodes(self, node_ids=None, filters=None, shard_identifier=None, **delete_kwargs):
        return super().delete_nodes(
            node_ids=_translate_node_ids_for_qdrant(node_ids),
            filters=filters,
            shard_identifier=shard_identifier,
            **delete_kwargs,
        )

    async def adelete_nodes(self, node_ids=None, filters=None, shard_identifier=None, **delete_kwargs):
        return await super().adelete_nodes(
            node_ids=_translate_node_ids_for_qdrant(node_ids),
            filters=filters,
            shard_identifier=shard_identifier,
            **delete_kwargs,
        )


class BaseVectorStore(ABC):
    """Abstract interface for vector store wrappers.

    Concrete implementations encapsulate a vector store backend plus any
    indexing/query orchestration required by the retrieval layer.
    """

    @classmethod
    @abstractmethod
    def from_config(
            cls, 
            config_path: str
        ) -> "BaseVectorStore":
        """Create a vector store instance from a YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.

        Returns
        -------
        BaseVectorStore
            Initialised vector store instance.

        Raises
        ------
        FileNotFoundError
            If ``config_path`` does not exist.
        ValueError
            If the configuration is invalid for the concrete implementation.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config_dict(
            cls, 
            config: dict
        ) -> "BaseVectorStore":
        """Create a vector store instance from a configuration mapping.

        Parameters
        ----------
        config : dict
            Configuration mapping (typically loaded from YAML).

        Returns
        -------
        BaseVectorStore
            Initialised vector store instance.

        Raises
        ------
        ValueError
            If required configuration keys are missing or invalid.
        """
        pass

class QdrantIndexStore(BaseVectorStore):
    """Qdrant-backed vector store wrapper and LlamaIndex index manager.

    This class wraps a Qdrant collection via :class:`qdrant_client.QdrantClient`
    and exposes methods to build an index from chunks and query it using a
    LlamaIndex query engine.

    Parameters
    ----------
    llm : BaseLLM
        Generator LLM wrapper used by the query engine to synthesise responses.
    embedder : BaseEmbedder
        Embedder wrapper used to generate embeddings for indexing and querying.
    host : str, optional
        Qdrant host address. Defaults to ``"localhost"``.
    port : int, optional
        Qdrant port number. Defaults to ``6333``.
    collection_name : str, optional
        Name of the Qdrant collection. Defaults to ``"default_collection"``.
    token : str or None, optional
        Optional token used to configure downstream authentication (e.g., sets
        ``HF_TOKEN``). Defaults to ``None``.
    """

    @classmethod
    def from_config_dict(
            cls, 
            config: dict,
            llm: "BaseLLM",
            embedder: BaseEmbedder,
        ) -> "QdrantIndexStore":
        """Create a QdrantIndexStore instance from a configuration mapping.

        Parameters
        ----------
        config : dict
            Configuration mapping. Expected keys include:
            - ``host`` (str, optional): Qdrant host address (default ``"localhost"``)
            - ``port`` (int, optional): Qdrant port (default ``6333``)
            - ``collection_name`` (str, optional): collection name
              (default ``"default_collection"``)
            - ``token`` (str, optional): optional token used to set ``HF_TOKEN``
        llm : BaseLLM
            Generator LLM wrapper used by the query engine.
        embedder : BaseEmbedder
            Embedder wrapper used to generate embeddings.

        Returns
        -------
        QdrantIndexStore
            Initialised QdrantIndexStore instance.
        """
        host = config.get("host", "localhost")
        port = config.get("port", 6333)
        collection_name = config.get("collection_name", "default_collection")
        token = config.get("token")
        return cls(
            llm=llm,
            embedder=embedder,
            host=host,
            port=port,
            collection_name=collection_name,
            token=token,
        )

    @classmethod
    def from_config(
        cls, 
        config_path: str, 
        *, 
        llm: "BaseLLM",
        embedder: BaseEmbedder,
    ) -> "QdrantIndexStore":
        """Load YAML configuration and create a QdrantIndexStore.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        llm : BaseLLM
            Generator LLM wrapper used by the query engine.
        embedder : BaseEmbedder
            Embedder wrapper used to generate embeddings.

        Returns
        -------
        QdrantIndexStore
            Initialised QdrantIndexStore instance.

        Notes
        -----
        This method loads the YAML file and delegates to
        :meth:`~polaris_rag.retrieval.vector_store.QdrantIndexStore.from_config_dict`.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cls.from_config_dict(cfg, llm=llm, embedder=embedder)

    def __init__(
        self, 
        *,
        llm: "BaseLLM" = None,
        embedder: BaseEmbedder = None,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "default_collection",
        token: Optional[str] = None,
    ):
        """Initialise a Qdrant-backed vector store index.

        Parameters
        ----------
        llm : BaseLLM
            Generator LLM wrapper used by the query engine.
        embedder : BaseEmbedder
            Embedder wrapper used to generate embeddings.
        host : str, optional
            Qdrant host address. Defaults to ``"localhost"``.
        port : int, optional
            Qdrant port number. Defaults to ``6333``.
        collection_name : str, optional
            Name of the Qdrant collection. Defaults to ``"default_collection"``.
        token : str or None, optional
            Optional token used to configure downstream authentication (e.g., sets
            ``HF_TOKEN``). Defaults to ``None``.

        Raises
        ------
        ValueError
            If ``embedder`` or ``llm`` is not provided.
        """

        if embedder is None:
            raise ValueError("QdrantIndexStore requires an embedder instance. Provide it via the container.")
        if llm is None:
            raise ValueError("QdrantIndexStore requires an LLM instance. Provide it via the container.")

        self.embedder = embedder
        self.llm = llm

        self._embed_model = self.embedder.get_embedder()
        self._llm = self.llm.get_llm()

        if token:
            environ["HF_TOKEN"] = token

        self.client = QdrantClient(host=host, port=port)
        self.aclient = AsyncQdrantClient(host=host, port=port)
        self.vector_store = PolarisQdrantVectorStore(
            client=self.client,
            aclient=self.aclient,
            collection_name=collection_name,
            dense_vector_name=""
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        self._index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self._embed_model,
        )

    def create_index(
        self,
        chunks: list[DocumentChunk],
        batch_size: int,
    ) -> None:
        """Build (or rebuild) the vector index from document chunks.

        This method embeds the provided chunks and stores the resulting vectors
        in the configured Qdrant collection by constructing a fresh
        :class:`llama_index.core.VectorStoreIndex`.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Document chunks to index.
        batch_size : int
            Insert batch size forwarded to LlamaIndex.

        Returns
        -------
        None
        """
        nodes = self._build_nodes(chunks)

        self._index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            show_progress=True,
            insert_batch_size=batch_size,
            embed_model=self._embed_model,
        ) 

    def insert_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int,
        *,
        use_async: bool = False,
    ) -> None:
        """Insert document chunks into an existing vector index.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Document chunks to insert.
        batch_size : int
            Batch size for insertion. If ``batch_size`` is not a positive integer,
            all chunks are inserted in a single call.
        use_async : bool, optional
            If ``True``, use async insertion to allow concurrent embedding
            requests when the configured embedder supports it.

        Returns
        -------
        None
        """

        if use_async:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.ainsert_chunks(chunks, batch_size=batch_size))
                return
            raise RuntimeError(
                "insert_chunks(use_async=True) cannot run inside an active event loop; "
                "use `await ainsert_chunks(...)` instead."
            )

        nodes = self._build_nodes(chunks)
        self._insert_nodes_sync(nodes=nodes, batch_size=batch_size)

    async def ainsert_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int,
    ) -> None:
        """Asynchronously insert document chunks into the vector index."""
        nodes = self._build_nodes(chunks)

        if not batch_size or batch_size <= 0 or not isinstance(batch_size, int):
            await self._index.ainsert_nodes(nodes)
        else:
            for start in range(0, len(nodes), batch_size):
                batch = nodes[start:start + batch_size]
                await self._index.ainsert_nodes(batch)

    def _insert_nodes_sync(
        self,
        *,
        nodes: list[TextNode],
        batch_size: int,
    ) -> None:
        if not batch_size or batch_size <= 0 or not isinstance(batch_size, int):
            self._index.insert_nodes(nodes)
        else:
            for start in range(0, len(nodes), batch_size):
                batch = nodes[start:start + batch_size]
                self._index.insert_nodes(batch)

    def _build_nodes(self, chunks: list[DocumentChunk]) -> list[TextNode]:
        return [chunk_to_text_node(chunk) for chunk in chunks]

    def delete_ref_doc(self, ref_doc_id: str) -> None:
        """Delete all chunks associated with a parent/source document id."""
        ref_doc_id = str(ref_doc_id or "").strip()
        if not ref_doc_id:
            return

        self.vector_store.delete(ref_doc_id)
        self.client.delete(
            collection_name=self.vector_store.collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="parent_id",
                        match=rest.MatchValue(value=ref_doc_id),
                    )
                ]
            ),
        )

    def delete_ref_docs(self, ref_doc_ids: Iterable[str]) -> int:
        """Delete all chunks associated with the provided parent/source ids."""
        deleted = 0
        seen: set[str] = set()
        for ref_doc_id in ref_doc_ids:
            ref_doc_id = str(ref_doc_id or "").strip()
            if not ref_doc_id or ref_doc_id in seen:
                continue
            seen.add(ref_doc_id)
            self.delete_ref_doc(ref_doc_id)
            deleted += 1
        return deleted

    @staticmethod
    def _coerce_metadata_filters(filters: MetadataFilters | dict | None) -> MetadataFilters | None:
        if isinstance(filters, MetadataFilters):
            return filters
        if isinstance(filters, dict):
            if not filters:
                return None
            return MetadataFilters(
                filters=[MetadataFilter(key=str(key), value=value) for key, value in filters.items()]
            )
        return None

    def query_nodes(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        filters: MetadataFilters | dict | None = None,
        timeout_seconds: float | None = None,
    ) -> list[NodeWithScore]:
        """Run a vector retrieval query and return LlamaIndex ``NodeWithScore`` items."""

        normalized_filters = self._coerce_metadata_filters(filters)
        retrieval_started_at = time.perf_counter()
        query_embedding = self.embedder.embed_query(query_text, timeout_seconds=timeout_seconds)
        query_spec = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=max(1, int(top_k)),
            filters=normalized_filters,
        )
        query_filter = self.vector_store._build_query_filter(query_spec)
        qdrant_timeout = None
        if timeout_seconds is not None:
            embed_elapsed = max(0.0, time.perf_counter() - retrieval_started_at)
            remaining_timeout = float(timeout_seconds) - embed_elapsed
            if remaining_timeout <= 0.0:
                raise RetrievalTimeoutError(
                    f"retrieval budget exhausted before vector-store query started ({float(timeout_seconds):.3f}s)"
                )
            qdrant_timeout = max(1, int(math.ceil(remaining_timeout)))

        try:
            response = self.client.query_points(
                collection_name=self.vector_store.collection_name,
                query=query_embedding,
                using=self.vector_store.dense_vector_name,
                limit=max(1, int(top_k)),
                query_filter=query_filter,
                with_payload=True,
                timeout=qdrant_timeout,
            )
        except Exception as exc:
            if timeout_seconds is not None and is_timeout_exception(exc):
                raise RetrievalTimeoutError(
                    f"vector-store query timed out after {float(timeout_seconds):.3f}s"
                ) from exc
            raise

        result = self.vector_store.parse_to_query_result(response.points)
        scored_nodes: list[NodeWithScore] = []
        similarities = list(result.similarities or [])
        for idx, node in enumerate(result.nodes or []):
            score = similarities[idx] if idx < len(similarities) else None
            scored_nodes.append(NodeWithScore(node=node, score=score))
        return scored_nodes

    def query(
            self, 
            query_text: str,
            top_k: int = 5,
            *,
            filter: Optional[dict] = None,
            llm: Optional["BaseLLM"] = None,
    ) -> Response:
        """Run a similarity query against the vector index.

        Parameters
        ----------
        query_text : str
            Natural-language query string.
        top_k : int, optional
            Number of top matching nodes to retrieve. Defaults to ``5``.
        filter : dict or None, optional
            Optional metadata filter applied to the vector store query.
        llm : BaseLLM or None, optional
            Optional LLM override. If provided, it is used instead of the instance
            LLM for this call only.

        Returns
        -------
        Response
            LlamaIndex response containing the generated answer and retrieved context.

        Raises
        ------
        ValueError
            If the index is empty.
        """
        vector_count = self.client.count(collection_name=self.vector_store.collection_name)
        if vector_count.count == 0:
            raise ValueError("Index is empty. Call create_index() first.")
        
        chosen_llm = llm or self._llm
        
        query_engine = self._index.as_query_engine(
            llm = chosen_llm,
            similarity_top_k=top_k,
            vector_store_kwargs={"filter": filter} if filter else {},
        )

        return query_engine.query(query_text)
    
    def persist(self, **kwargs):
        """Persist vector store state.

        For Qdrant, vectors are already persisted by the database itself.
        StorageContext still calls ``vector_store.persist(...)`` during overall
        persistence, so this method intentionally acts as a no-op.
        """
        return None


def _get_vector_store_kind(cfg):
    """Extract the vector store kind/type/provider discriminator from a config mapping.

    Parameters
    ----------
    cfg : dict
        Configuration mapping.

    Returns
    -------
    str or None
        Discriminator value if present, otherwise ``None``.
    """
    for key in ("kind", "type", "provider", "backend", "impl"):
        val = cfg.get(key)
        if val is not None:
            return val
    return None

def _normalize_vector_store_kind(kind):
    """Normalise a vector store kind/type string to a stable registry key.

    Parameters
    ----------
    kind : Any
        Provider/type discriminator value.

    Returns
    -------
    str
        Normalised key (defaults to ``"qdrant"`` when ``kind`` is falsy).
    """
    if not kind:
        return "qdrant"
    k = str(kind).lower()
    if k in {"qdrant", "qdrantindexstore", "qdrant_index_store"}:
        return "qdrant"
    return k

def create_vector_store(config: dict, llm: BaseLLM | None = None, embedder: BaseEmbedder | None = None) -> BaseVectorStore:
    """Create a vector store implementation from a configuration mapping.

    Parameters
    ----------
    config : dict
        Configuration mapping used to construct the vector store.
    llm : BaseLLM or None, optional
        Generator LLM wrapper required by some backends (e.g., QdrantIndexStore).
    embedder : BaseEmbedder or None, optional
        Embedder wrapper required by some backends (e.g., QdrantIndexStore).

    Returns
    -------
    BaseVectorStore
        Initialised vector store implementation.

    Raises
    ------
    ValueError
        If the requested backend kind is not supported.

    Notes
    -----
    The backend kind is selected using one of the discriminator keys:
    ``kind``, ``type``, ``provider``, ``backend``, or ``impl``. If none are
    provided, the default is Qdrant.
    """
    kind = _get_vector_store_kind(config)
    kind = _normalize_vector_store_kind(kind)
    if not kind or kind == "qdrant":
        return QdrantIndexStore.from_config_dict(config, llm=llm, embedder=embedder)
    raise ValueError(f"Unknown vector store kind: {kind!r}")


__all__ = [
    "BaseVectorStore",
    "QdrantIndexStore",
    "create_vector_store",
]
