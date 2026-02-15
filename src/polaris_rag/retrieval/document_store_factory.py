"""polaris_rag.retrieval.document_store_factory

Document store and storage-context factory utilities.

This module provides small factory helpers for constructing and persisting
LlamaIndex document stores and storage contexts used by the retrieval layer.
It centralises backend selection and persistence logic so that higher-level
components do not need to depend directly on LlamaIndex constructors.

Functions
---------
create_docstore
    Create a LlamaIndex document store by backend kind.
build_storage_context
    Build a :class:`llama_index.core.StorageContext` with docstore and optional vector store.
persist_storage
    Persist storage context state to disk.
load_storage
    Load a storage context from disk.
add_chunks_to_docstore
    Backfill document-store entries from chunk objects.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, Iterable, List

from llama_index.core.storage.docstore import (
    BaseDocumentStore,
    SimpleDocumentStore,
)
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode

from polaris_rag.retrieval.vector_store import BaseVectorStore

def create_docstore(
        kind: str = "simple",
        **kwargs: Any
    ) -> BaseDocumentStore:
    """Create a LlamaIndex document store by backend kind.

    Parameters
    ----------
    kind : {"simple"}, optional
        Docstore backend to use. ``"simple"`` stores JSON on disk via
        :class:`llama_index.core.storage.docstore.SimpleDocumentStore`.
        Defaults to ``"simple"``.
    **kwargs : Any
        Backend-specific keyword arguments (currently unused).

    Returns
    -------
    BaseDocumentStore
        Instantiated document store backend.

    Raises
    ------
    ValueError
        If ``kind`` does not correspond to a supported backend.
    """
    k = (kind or "").lower()
    if k == "simple":
        return SimpleDocumentStore()
    
    raise ValueError(f"Unknown docstore kind: {kind!r}. Use 'simple'.")


def build_storage_context(
        *,
        vector_store: Optional[BaseVectorStore] = None,
        docstore: Optional[BaseDocumentStore] = None,
        docstore_kind: str = "simple",
        persist_dir: Optional[str] = None,
        **docstore_kwargs: Any,
    ) -> StorageContext:
    """Build a LlamaIndex storage context.

    This function constructs a :class:`llama_index.core.StorageContext` that
    includes a document store and optionally a vector store. If ``persist_dir``
    is provided and contains an existing persisted context, it is loaded from
    disk; otherwise a fresh context is created.

    Parameters
    ----------
    vector_store : BaseVectorStore or None, optional
        Vector store instance to attach to the storage context.
    docstore : BaseDocumentStore or None, optional
        Explicit document store instance. If provided, ``docstore_kind`` is ignored.
    docstore_kind : str, optional
        Backend kind used when creating a new document store. Defaults to
        ``"simple"``.
    persist_dir : str or None, optional
        Directory path used for loading or persisting storage context state.
    **docstore_kwargs : Any
        Additional keyword arguments forwarded to the document-store factory.

    Returns
    -------
    StorageContext
        Constructed storage context instance.
    """
    if persist_dir:
        try:
            return StorageContext.from_defaults(
                persist_dir=persist_dir,
                vector_store=vector_store,
            )
        except Exception:
            pass

    ds = docstore or create_docstore(kind=docstore_kind, **docstore_kwargs)
    
    return StorageContext.from_defaults(vector_store=vector_store, docstore=ds)

def persist_storage(
        storage: StorageContext, 
        *, 
        persist_dir: str = "storage"
    ) -> None:
    """Persist a storage context to disk.

    This persists both document-store state and any vector-store state supported
    by the attached backends.

    Parameters
    ----------
    storage : StorageContext
        Storage context to persist.
    persist_dir : str, optional
        Target directory for persisted state. Defaults to ``"storage"``.
    """
    storage.persist(persist_dir=persist_dir)


def load_storage(
        *, 
        persist_dir: str, 
        vector_store: Optional[BaseVectorStore] = None
    ) -> StorageContext:
    """Load a storage context from disk.

    Parameters
    ----------
    persist_dir : str
        Directory containing a persisted storage context.
    vector_store : BaseVectorStore or None, optional
        Optional vector store instance to attach to the loaded context.

    Returns
    -------
    StorageContext
        Loaded storage context.
    """
    return StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)


def add_chunks_to_docstore(
        storage: StorageContext, 
        chunks: Iterable[Any]
    ) -> int:
    """Ensure chunk nodes are present in the document store.

    This helper converts chunk-like objects into LlamaIndex ``TextNode`` objects
    and adds them to the document store. It is primarily used when chunks have
    already been inserted into a vector store and the document store needs to be
    backfilled so that keyword/BM25 retrieval can function correctly.

    Parameters
    ----------
    storage : StorageContext
        Storage context whose document store should be populated.
    chunks : Iterable[Any]
        Iterable of chunk objects. Each chunk is expected to expose ``text``,
        ``id``, ``document_type``, ``parent_id`` (optional), and ``metadata``
        attributes.

    Returns
    -------
    int
        Number of nodes added to the document store.
    """
    ds = storage.docstore

    nodes = []

    for chunk in chunks:
        text = chunk.text
        document_type = chunk.document_type
        id = chunk.id
        parent_id = chunk.parent_id
        metadata = chunk.metadata
        
        if parent_id:
            metadata.update({'parent_id': parent_id})
        
        metadata.update({'document_type': document_type})

        node = TextNode(text=text, id_=id, metadata=metadata)
        nodes.append(node)

    ds.add_documents(nodes)
    return len(nodes)


__all__ = [
    "create_docstore",
    "build_storage_context",
    "persist_storage",
    "load_storage",
    "add_chunks_to_docstore",
]