"""Helpers for converting Polaris documents and chunks into LlamaIndex nodes."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeRelationship, ObjectType, RelatedNodeInfo, TextNode


def _source_relationship(parent_id: str | None) -> dict[NodeRelationship, RelatedNodeInfo]:
    if not isinstance(parent_id, str) or not parent_id:
        return {}

    return {
        NodeRelationship.SOURCE: RelatedNodeInfo(
            node_id=parent_id,
            node_type=ObjectType.DOCUMENT,
            metadata={},
            hash=None,
        )
    }


def chunk_to_text_node(chunk: Any) -> TextNode:
    """Convert a chunk-like object into a ``TextNode`` with source linkage."""
    text = getattr(chunk, "text", "") or ""
    chunk_id = str(getattr(chunk, "id", "") or "")
    document_type = str(getattr(chunk, "document_type", "") or "")
    parent_id = getattr(chunk, "parent_id", None)
    metadata = dict(getattr(chunk, "metadata", {}) or {})

    if parent_id:
        metadata["parent_id"] = parent_id
    if document_type:
        metadata["document_type"] = document_type

    return TextNode(
        text=text,
        id_=chunk_id,
        metadata=metadata,
        relationships=_source_relationship(str(parent_id) if parent_id is not None else None),
    )


def document_to_text_node(document: Any) -> TextNode:
    """Convert a document-like object into a ``TextNode``."""
    text = getattr(document, "text", "") or ""
    document_id = str(getattr(document, "id", "") or "")
    document_type = str(getattr(document, "document_type", "") or "")
    metadata = dict(getattr(document, "metadata", {}) or {})

    if document_type:
        metadata["document_type"] = document_type

    return TextNode(text=text, id_=document_id, metadata=metadata)
