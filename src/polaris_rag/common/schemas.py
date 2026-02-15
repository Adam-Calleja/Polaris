"""polaris_rag.common.schemas

Core data schemas shared across the RAG pipeline.

These lightweight dataclasses describe the canonical shapes for raw source
documents and their chunked derivatives. They are passed between ingestion,
chunking, embedding, retrieval, and generation components.

Classes
-------
Document
    Represents a full, un-split source document.
DocumentChunk
    A chunk produced from a parent :class:`~polaris_rag.common.schemas.Document`,
    suitable for embedding/retrieval.

Notes
-----
``metadata`` is intentionally untyped (``dict[str, Any]``/``Dict[str, Any]``) to
allow arbitrary key–value pairs (e.g., source, page number, author). Downstream
code should treat missing keys defensively.
"""

from dataclasses import dataclass, field
from typing import Any, Dict
from uuid import uuid4

@dataclass
class Document:
    """Container for a raw source document.

    Attributes
    ----------
    text : str
        Full textual content of the document, prior to chunking.
    document_type : str
        A coarse document/category label used by downstream components
        (e.g., "jira", "html", "pdf").
    id : str
        Unique identifier for the document. Defaults to a random UUID4 string.
    node_id : str or None
        Optional identifier used for graph/node-based integrations. If not
        provided, it is set to ``id`` in ``__post_init__``.
    metadata : Dict[str, Any]
        Arbitrary metadata associated with the document (e.g., ``{"source": "pdf", "title": "Whitepaper"}``).
        Defaults to an empty dict.
    """
    text: str
    document_type: str
    id: str = field(default_factory=lambda: str(uuid4()))
    node_id: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.node_id is None:
            self.node_id = self.id


@dataclass
class DocumentChunk:
    """A contiguous chunk of text derived from a parent :class:`~polaris_rag.common.schemas.Document`.

    Attributes
    ----------
    parent_id : str
        Identifier of the source :class:`~polaris_rag.common.schemas.Document` from
        which this chunk was produced.
    prev_id : str
        Identifier of the previous chunk in a linked sequence.
    next_id : str
        Identifier of the next chunk in a linked sequence.
    text : str
        Chunk text content (typically capped by a character/token limit).
    document_type : str
        A coarse document/category label (mirrors the parent document type).
    id : str
        Unique identifier for the chunk. Defaults to a random UUID4 string.
    node_id : str or None
        Optional identifier used for graph/node-based integrations. If not
        provided, it is set to ``id`` in ``__post_init__``.
    metadata : Dict[str, Any]
        Metadata propagated from the parent or added during chunking
        (e.g., ``{"chunk_index": 3, "overlap": 32}``). Defaults to an empty dict.
    source_node : Document or None
        Optional reference to the originating :class:`~polaris_rag.common.schemas.Document`.
    """
    parent_id: str
    prev_id: str
    next_id: str
    text: str
    document_type: str
    id: str = field(default_factory=lambda: str(uuid4()))
    node_id: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_node: Document = None

    def __post_init__(self):
        if self.node_id is None:
            self.node_id = self.id