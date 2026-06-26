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
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Optional
from uuid import UUID, NAMESPACE_URL, uuid5
import httpx
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http import models as rest
from os import environ
import yaml
from abc import ABC, abstractmethod
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeRelationship, NodeWithScore, ObjectType, RelatedNodeInfo, TextNode
from llama_index.core.base.response.schema import Response
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, VectorStoreQuery

from polaris_rag.common.schemas import DocumentChunk
from polaris_rag.common.request_budget import RetrievalTimeoutError, is_timeout_exception
from polaris_rag.retrieval.node_utils import chunk_to_text_node

if TYPE_CHECKING:
    from polaris_rag.generation.llm_interface import BaseLLM
    from polaris_rag.retrieval.embedder import BaseEmbedder
    from polaris_rag.retrieval.sparse_encoder import BaseSparseEncoder, SparseEmbedding
else:
    BaseLLM = Any
    BaseEmbedder = Any
    BaseSparseEncoder = Any
    SparseEmbedding = Any

QDRANT_POINT_ID_NAMESPACE = uuid5(NAMESPACE_URL, "polaris-rag/qdrant-point-id")
QDRANT_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
OPENAI_RETRYABLE_STATUS_CODES = frozenset({408, 424, 429, 500, 502, 503, 504})


def qdrant_point_id_from_node_id(node_id: Any) -> int | str:
    """Return a Qdrant-compatible point id for an arbitrary logical node id.
    
    Parameters
    ----------
    node_id : Any
        Stable identifier for node.
    
    Returns
    -------
    int or str
        Result of the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
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
    """Translate Node IDs For Qdrant.
    
    Parameters
    ----------
    node_ids : list[str] or None, optional
        Logical node identifiers to translate or fetch.
    
    Returns
    -------
    list[int or str] or None
        Collected results from the operation.
    """
    if node_ids is None:
        return None
    return [qdrant_point_id_from_node_id(node_id) for node_id in node_ids]


def _is_retryable_qdrant_write_error(exc: Exception) -> bool:
    if isinstance(exc, (ResponseHandlingException, httpx.TimeoutException, httpx.TransportError, TimeoutError)):
        return True

    if isinstance(exc, UnexpectedResponse):
        status_code = getattr(exc, "status_code", None)
        return status_code in QDRANT_RETRYABLE_STATUS_CODES

    return False


def _is_retryable_embedding_error(exc: Exception) -> bool:
    if isinstance(exc, (OpenAIAPITimeoutError, OpenAIAPIConnectionError, httpx.TimeoutException, TimeoutError)):
        return True

    if isinstance(exc, OpenAIAPIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code in OPENAI_RETRYABLE_STATUS_CODES

    return False


def _is_retryable_insert_error(exc: Exception) -> bool:
    return _is_retryable_qdrant_write_error(exc) or _is_retryable_embedding_error(exc)


class PolarisQdrantVectorStore(QdrantVectorStore):
    """Qdrant adapter that maps arbitrary logical node ids to valid point ids.
    
    Methods
    -------
    get_nodes
        Return nodes.
    aget_nodes
        Aget Nodes.
    delete_nodes
        Delete Nodes.
    adelete_nodes
        Adelete Nodes.
    """

    def _build_points(self, nodes, sparse_vector_name):
        """Build points.
        
        Parameters
        ----------
        nodes : Any
            Value for nodes.
        sparse_vector_name : Any
            Value for sparse Vector Name.
        """
        points, ids = super()._build_points(nodes, sparse_vector_name)
        for point, node_id in zip(points, ids):
            point.id = qdrant_point_id_from_node_id(node_id)
        return points, ids

    def get_nodes(self, node_ids=None, filters=None, limit=None, shard_identifier=None):
        """Return nodes.
        
        Parameters
        ----------
        node_ids : Any, optional
            Logical node identifiers to translate or fetch.
        filters : Any, optional
            Value for filters.
        limit : Any, optional
            Maximum number of records or nodes to return.
        shard_identifier : Any, optional
            Value for shard Identifier.
        """
        return super().get_nodes(
            node_ids=_translate_node_ids_for_qdrant(node_ids),
            filters=filters,
            limit=limit,
            shard_identifier=shard_identifier,
        )

    async def aget_nodes(self, node_ids=None, filters=None, limit=None, shard_identifier=None):
        """Aget Nodes.
        
        Parameters
        ----------
        node_ids : Any, optional
            Logical node identifiers to translate or fetch.
        filters : Any, optional
            Value for filters.
        limit : Any, optional
            Maximum number of records or nodes to return.
        shard_identifier : Any, optional
            Value for shard Identifier.
        """
        return await super().aget_nodes(
            node_ids=_translate_node_ids_for_qdrant(node_ids),
            filters=filters,
            limit=limit,
            shard_identifier=shard_identifier,
        )

    def delete_nodes(self, node_ids=None, filters=None, shard_identifier=None, **delete_kwargs):
        """Delete Nodes.
        
        Parameters
        ----------
        node_ids : Any, optional
            Logical node identifiers to translate or fetch.
        filters : Any, optional
            Value for filters.
        shard_identifier : Any, optional
            Value for shard Identifier.
        **delete_kwargs : Any
            Value for delete Kwargs.
        """
        return super().delete_nodes(
            node_ids=_translate_node_ids_for_qdrant(node_ids),
            filters=filters,
            shard_identifier=shard_identifier,
            **delete_kwargs,
        )

    async def adelete_nodes(self, node_ids=None, filters=None, shard_identifier=None, **delete_kwargs):
        """Adelete Nodes.
        
        Parameters
        ----------
        node_ids : Any, optional
            Logical node identifiers to translate or fetch.
        filters : Any, optional
            Value for filters.
        shard_identifier : Any, optional
            Value for shard Identifier.
        **delete_kwargs : Any
            Value for delete Kwargs.
        """
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
    
    Methods
    -------
    from_config
        Create a vector store instance from a YAML configuration file.
    from_config_dict
        Create a vector store instance from a configuration mapping.
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
    """Qdrant-backed vector store wrapper with native dense+sparse retrieval."""

    DEFAULT_DENSE_VECTOR_NAME = "dense"
    DEFAULT_SPARSE_VECTOR_NAME = "sparse"

    @classmethod
    def from_config_dict(
        cls,
        config: dict,
        llm: "BaseLLM",
        embedder: BaseEmbedder,
        sparse_encoder: BaseSparseEncoder | None = None,
    ) -> "QdrantIndexStore":
        host = config.get("host", "localhost")
        port = config.get("port", 6333)
        collection_name = config.get("collection_name", "default_collection")
        token = config.get("token")
        return cls(
            llm=llm,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
            host=host,
            port=port,
            collection_name=collection_name,
            token=token,
            dense_vector_name=str(config.get("dense_vector_name", cls.DEFAULT_DENSE_VECTOR_NAME)),
            sparse_vector_name=str(config.get("sparse_vector_name", cls.DEFAULT_SPARSE_VECTOR_NAME)),
            write_max_attempts=int(config.get("write_max_attempts", 5)),
            write_base_backoff_seconds=float(config.get("write_base_backoff_seconds", 1.0)),
            reduce_batch_on_failure=bool(config.get("reduce_batch_on_failure", True)),
            min_insertion_batch_size=int(config.get("min_insertion_batch_size", 1)),
        )

    @classmethod
    def from_config(
        cls,
        config_path: str,
        *,
        llm: "BaseLLM",
        embedder: BaseEmbedder,
        sparse_encoder: BaseSparseEncoder | None = None,
    ) -> "QdrantIndexStore":
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cls.from_config_dict(cfg, llm=llm, embedder=embedder, sparse_encoder=sparse_encoder)

    def __init__(
        self,
        *,
        llm: "BaseLLM" = None,
        embedder: BaseEmbedder = None,
        sparse_encoder: BaseSparseEncoder | None = None,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "default_collection",
        token: Optional[str] = None,
        dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME,
        sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
        write_max_attempts: int = 5,
        write_base_backoff_seconds: float = 1.0,
        reduce_batch_on_failure: bool = True,
        min_insertion_batch_size: int = 1,
    ):
        if embedder is None:
            raise ValueError("QdrantIndexStore requires an embedder instance. Provide it via the container.")
        if llm is None:
            raise ValueError("QdrantIndexStore requires an LLM instance. Provide it via the container.")

        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.llm = llm
        self.collection_name = str(collection_name)
        self.dense_vector_name = str(dense_vector_name or self.DEFAULT_DENSE_VECTOR_NAME)
        self.sparse_vector_name = str(sparse_vector_name or self.DEFAULT_SPARSE_VECTOR_NAME)
        self.write_max_attempts = max(1, int(write_max_attempts))
        self.write_base_backoff_seconds = max(0.0, float(write_base_backoff_seconds))
        self.reduce_batch_on_failure = bool(reduce_batch_on_failure)
        self.min_insertion_batch_size = max(1, int(min_insertion_batch_size))
        self._embed_model = self.embedder.get_embedder()
        self._llm = self.llm.get_llm()

        if token:
            environ["HF_TOKEN"] = token

        self.client = QdrantClient(host=host, port=port)
        self.aclient = AsyncQdrantClient(host=host, port=port)
        self.vector_store = PolarisQdrantVectorStore(
            client=self.client,
            aclient=self.aclient,
            collection_name=self.collection_name,
            dense_vector_name=self.dense_vector_name,
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self._index = None

    def profile(self) -> dict[str, Any]:
        """Return a stable profile for retrieval fingerprinting."""
        return {
            "backend": "qdrant",
            "collection_name": self.collection_name,
            "dense_vector_name": self.dense_vector_name,
            "sparse_vector_name": self.sparse_vector_name,
            "dense_model": getattr(self.embedder, "model_name", type(self.embedder).__name__),
            "sparse_model": (
                self.sparse_encoder.profile() if self.sparse_encoder is not None else None
            ),
        }

    def _resolved_write_max_attempts(self) -> int:
        return max(1, int(getattr(self, "write_max_attempts", 5)))

    def _resolved_write_base_backoff_seconds(self) -> float:
        return max(0.0, float(getattr(self, "write_base_backoff_seconds", 1.0)))

    def _resolved_reduce_batch_on_failure(self) -> bool:
        return bool(getattr(self, "reduce_batch_on_failure", True))

    def _resolved_min_insertion_batch_size(self) -> int:
        return max(1, int(getattr(self, "min_insertion_batch_size", 1)))

    def ensure_collection_exists(self, *, sample_text: str = "polaris collection bootstrap") -> None:
        """Create the backing Qdrant collection if it does not already exist."""
        if self.client.collection_exists(self.collection_name):
            return

        dense_vectors = self.embedder.embed_documents([str(sample_text)])
        if not dense_vectors or not dense_vectors[0]:
            raise ValueError(
                "Unable to infer embedding dimension for Qdrant collection creation: "
                "embedder returned no vectors."
            )

        self._ensure_collection(
            dense_dim=len(dense_vectors[0]),
            enable_sparse=self.sparse_encoder is not None,
        )

    def clear_collection(self) -> None:
        """Delete the backing Qdrant collection if it exists."""
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete_collection(collection_name=self.collection_name)

    def recreate_collection(self, *, sample_text: str = "polaris collection bootstrap") -> None:
        """Recreate the backing Qdrant collection from scratch."""
        self.clear_collection()
        self.ensure_collection_exists(sample_text=sample_text)

    def create_index(self, chunks: list[DocumentChunk], batch_size: int) -> None:
        """Compatibility wrapper: recreate by reinserting chunks."""
        self.insert_chunks(chunks, batch_size=batch_size, use_async=False)

    def insert_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int,
        *,
        use_async: bool = False,
    ) -> None:
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

        if not chunks:
            return
        step = self._insertion_batch_size(batch_size=batch_size, total_chunks=len(chunks))
        for chunk_batch in self._iter_chunk_batches(chunks, batch_size=step):
            self._insert_chunk_batch(chunk_batch, batch_size=step)

    async def ainsert_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int,
    ) -> None:
        if not chunks:
            return
        step = self._insertion_batch_size(batch_size=batch_size, total_chunks=len(chunks))
        for chunk_batch in self._iter_chunk_batches(chunks, batch_size=step):
            await self._ainsert_chunk_batch(chunk_batch, batch_size=step)

    def _insert_chunk_batch(
        self,
        chunks: list[DocumentChunk],
        *,
        batch_size: int,
    ) -> None:
        if not chunks:
            return

        last_exc: Exception | None = None
        write_max_attempts = self._resolved_write_max_attempts()
        write_base_backoff_seconds = self._resolved_write_base_backoff_seconds()
        reduce_batch_on_failure = self._resolved_reduce_batch_on_failure()
        min_insertion_batch_size = self._resolved_min_insertion_batch_size()

        for attempt in range(1, write_max_attempts + 1):
            try:
                texts = [str(chunk.text or "") for chunk in chunks]
                dense_vectors = self.embedder.embed_documents(texts)
                sparse_vectors = self._encode_sparse_documents(texts)
                self._upsert_chunks(
                    chunks=chunks,
                    dense_vectors=dense_vectors,
                    sparse_vectors=sparse_vectors,
                    batch_size=max(1, int(batch_size)),
                )
                return
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_insert_error(exc):
                    raise
                if attempt < write_max_attempts:
                    retry_delay = write_base_backoff_seconds * (2 ** (attempt - 1))
                    print(
                        f"Chunk insert batch of {len(chunks)} failed "
                        f"(attempt {attempt}/{write_max_attempts}): {exc}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                    continue

        if (
            reduce_batch_on_failure
            and len(chunks) > min_insertion_batch_size
        ):
            reduced_batch_size = max(
                min_insertion_batch_size,
                min(len(chunks) - 1, max(1, len(chunks) // 2)),
            )
            print(
                f"Chunk insert batch of {len(chunks)} exhausted retries. "
                f"Reducing batch size to {reduced_batch_size} and retrying."
            )
            for chunk_batch in self._iter_chunk_batches(chunks, batch_size=reduced_batch_size):
                self._insert_chunk_batch(chunk_batch, batch_size=reduced_batch_size)
            return

        if last_exc is None:
            raise RuntimeError("Chunk insert retry loop exited without success or exception.")
        raise last_exc

    async def _ainsert_chunk_batch(
        self,
        chunks: list[DocumentChunk],
        *,
        batch_size: int,
    ) -> None:
        if not chunks:
            return

        last_exc: Exception | None = None
        write_max_attempts = self._resolved_write_max_attempts()
        write_base_backoff_seconds = self._resolved_write_base_backoff_seconds()
        reduce_batch_on_failure = self._resolved_reduce_batch_on_failure()
        min_insertion_batch_size = self._resolved_min_insertion_batch_size()

        for attempt in range(1, write_max_attempts + 1):
            try:
                texts = [str(chunk.text or "") for chunk in chunks]
                dense_vectors = await self.embedder.aembed_documents(texts)
                sparse_vectors = self._encode_sparse_documents(texts)
                self._upsert_chunks(
                    chunks=chunks,
                    dense_vectors=dense_vectors,
                    sparse_vectors=sparse_vectors,
                    batch_size=max(1, int(batch_size)),
                )
                return
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_insert_error(exc):
                    raise
                if attempt < write_max_attempts:
                    retry_delay = write_base_backoff_seconds * (2 ** (attempt - 1))
                    print(
                        f"Async chunk insert batch of {len(chunks)} failed "
                        f"(attempt {attempt}/{write_max_attempts}): {exc}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

        if (
            reduce_batch_on_failure
            and len(chunks) > min_insertion_batch_size
        ):
            reduced_batch_size = max(
                min_insertion_batch_size,
                min(len(chunks) - 1, max(1, len(chunks) // 2)),
            )
            print(
                f"Async chunk insert batch of {len(chunks)} exhausted retries. "
                f"Reducing batch size to {reduced_batch_size} and retrying."
            )
            for chunk_batch in self._iter_chunk_batches(chunks, batch_size=reduced_batch_size):
                await self._ainsert_chunk_batch(chunk_batch, batch_size=reduced_batch_size)
            return

        if last_exc is None:
            raise RuntimeError("Async chunk insert retry loop exited without success or exception.")
        raise last_exc

    def _upsert_chunks(
        self,
        *,
        chunks: list[DocumentChunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseEmbedding | None],
        batch_size: int,
    ) -> None:
        if not dense_vectors:
            return
        dense_dim = len(dense_vectors[0])
        enable_sparse = self.sparse_encoder is not None or any(
            vector is not None and not vector.is_empty() for vector in sparse_vectors
        )
        self._ensure_collection(dense_dim=dense_dim, enable_sparse=enable_sparse)

        points = [
            rest.PointStruct(
                id=qdrant_point_id_from_node_id(chunk.id),
                vector=self._build_point_vectors(dense=dense, sparse=sparse),
                payload=self._chunk_payload(chunk),
            )
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
        ]
        step = batch_size if batch_size and batch_size > 0 else len(points)
        for start in range(0, len(points), max(1, step)):
            batch = points[start:start + max(1, step)]
            self._upsert_points_batch(batch)

    def _upsert_points_batch(self, points: list[rest.PointStruct]) -> None:
        last_exc: Exception | None = None
        write_max_attempts = self._resolved_write_max_attempts()
        write_base_backoff_seconds = self._resolved_write_base_backoff_seconds()

        for attempt in range(1, write_max_attempts + 1):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                return
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_qdrant_write_error(exc):
                    raise
                if attempt < write_max_attempts:
                    retry_delay = write_base_backoff_seconds * (2 ** (attempt - 1))
                    print(
                        f"Qdrant upsert of {len(points)} point(s) failed "
                        f"(attempt {attempt}/{write_max_attempts}): {exc}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                    continue

        if last_exc is None:
            raise RuntimeError("Qdrant upsert retry loop exited without success or exception.")
        raise last_exc

    @staticmethod
    def _insertion_batch_size(*, batch_size: int, total_chunks: int) -> int:
        if batch_size and batch_size > 0:
            return int(batch_size)
        return max(1, int(total_chunks))

    @staticmethod
    def _iter_chunk_batches(
        chunks: list[DocumentChunk],
        *,
        batch_size: int,
    ) -> Iterable[list[DocumentChunk]]:
        step = max(1, int(batch_size))
        for start in range(0, len(chunks), step):
            yield chunks[start:start + step]

    def _chunk_payload(self, chunk: DocumentChunk) -> dict[str, Any]:
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        payload = dict(metadata)
        payload["_node_id"] = str(getattr(chunk, "id", "") or "")
        payload["_node_text"] = str(getattr(chunk, "text", "") or "")
        payload["_node_metadata"] = dict(metadata)
        payload["document_type"] = str(getattr(chunk, "document_type", "") or metadata.get("document_type", ""))
        parent_id = str(getattr(chunk, "parent_id", "") or metadata.get("parent_id", ""))
        if parent_id:
            payload["parent_id"] = parent_id
        return payload

    def _build_point_vectors(
        self,
        *,
        dense: list[float],
        sparse: SparseEmbedding | None,
    ) -> dict[str, Any]:
        vectors: dict[str, Any] = {
            self.dense_vector_name: [float(value) for value in dense],
        }
        if sparse is not None and not sparse.is_empty():
            vectors[self.sparse_vector_name] = rest.SparseVector(
                indices=[int(item) for item in sparse.indices],
                values=[float(item) for item in sparse.values],
            )
        return vectors

    def _encode_sparse_documents(self, texts: list[str]) -> list[SparseEmbedding | None]:
        if self.sparse_encoder is None:
            return [None for _ in texts]
        embeddings = self.sparse_encoder.encode_documents(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                "Sparse encoder returned a different number of embeddings than requested "
                f"({len(embeddings)} != {len(texts)})."
            )
        return list(embeddings)

    def _ensure_collection(self, *, dense_dim: int, enable_sparse: bool) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        vectors_config = {
            self.dense_vector_name: rest.VectorParams(
                size=max(1, int(dense_dim)),
                distance=rest.Distance.COSINE,
            )
        }
        kwargs: dict[str, Any] = {}
        if enable_sparse:
            kwargs["sparse_vectors_config"] = {
                self.sparse_vector_name: rest.SparseVectorParams()
            }
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            **kwargs,
        )

    def delete_ref_doc(self, ref_doc_id: str) -> None:
        ref_doc_id = str(ref_doc_id or "").strip()
        if not ref_doc_id:
            return
        if not self.client.collection_exists(self.collection_name):
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                should=[
                    rest.FieldCondition(
                        key="_node_id",
                        match=rest.MatchValue(value=ref_doc_id),
                    ),
                    rest.FieldCondition(
                        key="parent_id",
                        match=rest.MatchValue(value=ref_doc_id),
                    ),
                ]
            ),
        )

    def delete_ref_docs(self, ref_doc_ids: Iterable[str]) -> int:
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

    def _query_filter(self, filters: MetadataFilters | dict | None) -> Any:
        normalized_filters = self._coerce_metadata_filters(filters)
        query_spec = VectorStoreQuery(query_embedding=None, similarity_top_k=1, filters=normalized_filters)
        return self.vector_store._build_query_filter(query_spec)

    def query_nodes(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        filters: MetadataFilters | dict | None = None,
        timeout_seconds: float | None = None,
    ) -> list[NodeWithScore]:
        return self.query_dense_nodes(
            query_text,
            top_k=top_k,
            filters=filters,
            timeout_seconds=timeout_seconds,
        )

    def query_dense_nodes(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        filters: MetadataFilters | dict | None = None,
        timeout_seconds: float | None = None,
    ) -> list[NodeWithScore]:
        if not self.client.collection_exists(self.collection_name):
            return []

        retrieval_started_at = time.perf_counter()
        query_embedding = self.embedder.embed_query(query_text, timeout_seconds=timeout_seconds)
        qdrant_timeout = self._remaining_qdrant_timeout(
            timeout_seconds=timeout_seconds,
            retrieval_started_at=retrieval_started_at,
        )
        query_filter = self._query_filter(filters)
        response = self._query_points(
            query=query_embedding,
            using=self.dense_vector_name,
            top_k=top_k,
            query_filter=query_filter,
            qdrant_timeout=qdrant_timeout,
            allow_legacy_dense_fallback=True,
            timeout_seconds=timeout_seconds,
        )
        return self._points_to_scored_nodes(response.points)

    def query_sparse_nodes(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        filters: MetadataFilters | dict | None = None,
        timeout_seconds: float | None = None,
    ) -> list[NodeWithScore]:
        if not self.client.collection_exists(self.collection_name) or self.sparse_encoder is None:
            return []
        retrieval_started_at = time.perf_counter()
        sparse_query = self.sparse_encoder.encode_query(query_text)
        if sparse_query.is_empty():
            return []
        qdrant_timeout = self._remaining_qdrant_timeout(
            timeout_seconds=timeout_seconds,
            retrieval_started_at=retrieval_started_at,
        )
        query_filter = self._query_filter(filters)
        response = self._query_points(
            query=rest.SparseVector(
                indices=[int(item) for item in sparse_query.indices],
                values=[float(item) for item in sparse_query.values],
            ),
            using=self.sparse_vector_name,
            top_k=top_k,
            query_filter=query_filter,
            qdrant_timeout=qdrant_timeout,
            allow_legacy_dense_fallback=False,
            timeout_seconds=timeout_seconds,
        )
        return self._points_to_scored_nodes(response.points)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        *,
        filter: Optional[dict] = None,
        llm: Optional["BaseLLM"] = None,
    ) -> Response:
        vector_count = self.client.count(collection_name=self.collection_name)
        if vector_count.count == 0:
            raise ValueError("Index is empty. Call create_index() first.")

        chosen_llm = llm or self._llm
        results = self.query_nodes(query_text, top_k=top_k, filters=filter)
        prompt_lines = [query_text, "", "CONTEXT:"]
        for item in results:
            prompt_lines.append(str(getattr(item.node, "text", "") or ""))
        response_text = chosen_llm.complete("\n".join(prompt_lines))
        return Response(response=response_text, source_nodes=results)

    def persist(self, **kwargs):
        return None

    def iter_payload_nodes(
        self,
        *,
        batch_size: int = 512,
        filters: MetadataFilters | dict | None = None,
        timeout_seconds: float | None = None,
    ) -> Iterator[TextNode]:
        """Yield stored nodes reconstructed from Qdrant payloads.

        Parameters
        ----------
        batch_size : int, optional
            Number of points to fetch per Qdrant scroll request.
        filters : MetadataFilters or dict or None, optional
            Optional metadata filters forwarded to the Qdrant scroll query.
        timeout_seconds : float or None, optional
            Optional overall timeout budget for the scroll operation.

        Yields
        ------
        TextNode
            Reconstructed node from the stored payload.
        """
        if not self.client.collection_exists(self.collection_name):
            return

        retrieval_started_at = time.perf_counter()
        qdrant_timeout = self._remaining_qdrant_timeout(
            timeout_seconds=timeout_seconds,
            retrieval_started_at=retrieval_started_at,
        ) if timeout_seconds is not None else None
        scroll_filter = self._query_filter(filters)
        offset: Any | None = None
        step = max(1, int(batch_size))

        while True:
            try:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=step,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    timeout=qdrant_timeout,
                )
            except Exception as exc:
                if timeout_seconds is not None and is_timeout_exception(exc):
                    raise RetrievalTimeoutError(
                        f"vector-store payload scroll timed out after {float(timeout_seconds):.3f}s"
                    ) from exc
                raise

            for point in list(points or []):
                yield self._point_to_node(point)

            if offset is None:
                break

    def _remaining_qdrant_timeout(
        self,
        *,
        timeout_seconds: float | None,
        retrieval_started_at: float,
    ) -> int | None:
        if timeout_seconds is None:
            return None
        elapsed = max(0.0, time.perf_counter() - retrieval_started_at)
        remaining_timeout = float(timeout_seconds) - elapsed
        if remaining_timeout <= 0.0:
            raise RetrievalTimeoutError(
                f"retrieval budget exhausted before vector-store query started ({float(timeout_seconds):.3f}s)"
            )
        return max(1, int(math.ceil(remaining_timeout)))

    def _query_points(
        self,
        *,
        query: Any,
        using: str | None,
        top_k: int,
        query_filter: Any,
        qdrant_timeout: int | None,
        allow_legacy_dense_fallback: bool,
        timeout_seconds: float | None,
    ) -> Any:
        try:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query,
                using=using,
                limit=max(1, int(top_k)),
                query_filter=query_filter,
                with_payload=True,
                timeout=qdrant_timeout,
            )
        except Exception as exc:
            if allow_legacy_dense_fallback and using:
                try:
                    return self.client.query_points(
                        collection_name=self.collection_name,
                        query=query,
                        using=None,
                        limit=max(1, int(top_k)),
                        query_filter=query_filter,
                        with_payload=True,
                        timeout=qdrant_timeout,
                    )
                except Exception:
                    pass
            if timeout_seconds is not None and is_timeout_exception(exc):
                raise RetrievalTimeoutError(
                    f"vector-store query timed out after {float(timeout_seconds):.3f}s"
                ) from exc
            raise

    @staticmethod
    def _metadata_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        metadata = {}
        raw_metadata = payload.get("_node_metadata")
        if isinstance(raw_metadata, Mapping):
            metadata.update(dict(raw_metadata))
        for key, value in payload.items():
            if key.startswith("_node_"):
                continue
            if key in {"_node_id", "_node_text"}:
                continue
            metadata.setdefault(key, value)
        return metadata

    @staticmethod
    def _source_relationship(parent_id: str | None) -> dict[NodeRelationship, RelatedNodeInfo]:
        if not parent_id:
            return {}
        return {
            NodeRelationship.SOURCE: RelatedNodeInfo(
                node_id=str(parent_id),
                node_type=ObjectType.DOCUMENT,
                metadata={},
                hash=None,
            )
        }

    def _point_to_node(self, point: Any) -> TextNode:
        payload = getattr(point, "payload", None) or {}
        if not isinstance(payload, Mapping):
            payload = {}
        metadata = self._metadata_from_payload(payload)
        node_id = str(payload.get("_node_id") or getattr(point, "id", ""))
        node_text = str(payload.get("_node_text", "") or "")
        parent_id = str(metadata.get("parent_id", "") or "")
        return TextNode(
            id_=node_id,
            text=node_text,
            metadata=metadata,
            relationships=self._source_relationship(parent_id if parent_id else None),
        )

    def _points_to_scored_nodes(self, points: list[Any]) -> list[NodeWithScore]:
        results: list[NodeWithScore] = []
        for point in list(points or []):
            score = getattr(point, "score", None)
            results.append(
                NodeWithScore(
                    node=self._point_to_node(point),
                    score=float(score) if isinstance(score, (int, float)) else None,
                )
            )
        return results


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

def create_vector_store(
    config: dict,
    llm: BaseLLM | None = None,
    embedder: BaseEmbedder | None = None,
    sparse_encoder: BaseSparseEncoder | None = None,
) -> BaseVectorStore:
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
        return QdrantIndexStore.from_config_dict(
            config,
            llm=llm,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
        )
    raise ValueError(f"Unknown vector store kind: {kind!r}")


__all__ = [
    "BaseVectorStore",
    "QdrantIndexStore",
    "create_vector_store",
]
