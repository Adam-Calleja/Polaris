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
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, Iterable, Sequence

from llama_index.core.storage.docstore import (
    BaseDocumentStore,
    SimpleDocumentStore,
)
from llama_index.core import StorageContext

from polaris_rag.retrieval.node_utils import chunk_to_text_node, document_to_text_node

if TYPE_CHECKING:
    from polaris_rag.retrieval.vector_store import BaseVectorStore

SOURCE_DOCUMENT_STORE_FILENAME = "source_docstore.json"

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
        vector_store: Optional["BaseVectorStore"] = None,
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


def source_document_store_path(persist_dir: str | Path) -> str:
    """Return the path used for the persisted source-document store."""
    return str(Path(persist_dir).expanduser().resolve() / SOURCE_DOCUMENT_STORE_FILENAME)


def load_or_create_source_document_store(
        *,
        persist_dir: str | Path | None,
    ) -> BaseDocumentStore:
    """Load a persisted source-document store or create a fresh one."""
    if persist_dir is None:
        return create_docstore("simple")

    persist_path = Path(source_document_store_path(persist_dir))
    if persist_path.exists():
        return SimpleDocumentStore.from_persist_path(str(persist_path))

    return create_docstore("simple")

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


def persist_docstore(
        docstore: BaseDocumentStore,
        *,
        persist_path: str | Path,
    ) -> None:
    """Persist a standalone docstore to a specific JSON path."""
    target = Path(persist_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    docstore.persist(persist_path=str(target))


def load_storage(
        *, 
        persist_dir: str, 
        vector_store: Optional["BaseVectorStore"] = None
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
        nodes.append(chunk_to_text_node(chunk))

    ds.add_documents(nodes)
    return len(nodes)


def add_documents_to_docstore(
        docstore: BaseDocumentStore,
        documents: Iterable[Any],
    ) -> int:
    """Store full source documents in a docstore."""
    nodes = [document_to_text_node(document) for document in documents]
    docstore.add_documents(nodes, allow_update=True)
    return len(nodes)


def delete_ref_docs_from_docstore(
        docstore: BaseDocumentStore,
        ref_doc_ids: Sequence[str],
    ) -> int:
    """Delete all nodes associated with each ref-doc id from a docstore."""
    deleted = 0
    seen: set[str] = set()
    for ref_doc_id in ref_doc_ids:
        ref_doc_id = str(ref_doc_id or "").strip()
        if not ref_doc_id or ref_doc_id in seen:
            continue
        seen.add(ref_doc_id)
        docstore.delete_ref_doc(ref_doc_id, raise_error=False)
        docs = getattr(docstore, "docs", {})
        if isinstance(docs, dict):
            matching_node_ids = []
            for node_id, node in docs.items():
                metadata = getattr(node, "metadata", None)
                metadata = metadata if isinstance(metadata, dict) else {}
                node_parent_id = str(metadata.get("parent_id") or "")
                node_ref_doc_id = str(getattr(node, "ref_doc_id", "") or "")
                if node_parent_id == ref_doc_id or node_ref_doc_id == ref_doc_id:
                    matching_node_ids.append(str(node_id))
            for node_id in matching_node_ids:
                docstore.delete_document(node_id, raise_error=False)
        deleted += 1
    return deleted


__all__ = [
    "create_docstore",
    "build_storage_context",
    "source_document_store_path",
    "load_or_create_source_document_store",
    "persist_storage",
    "persist_docstore",
    "load_storage",
    "add_chunks_to_docstore",
    "add_documents_to_docstore",
    "delete_ref_docs_from_docstore",
]
