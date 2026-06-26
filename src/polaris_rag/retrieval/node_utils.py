"""Helpers for converting Polaris documents/chunks to nodes and serializing trace metadata.

This module exposes public helper functions used by the surrounding Polaris subsystem.

Functions
---------
chunk_to_text_node
    Convert a chunk-like object into a ``TextNode`` with source linkage.
document_to_text_node
    Convert a document-like object into a ``TextNode``.
extract_doc_id
    Extract a stable document identifier from a node-like object.
extract_text
    Extract text content from a node-like object.
serialize_source_nodes
    Serialize source nodes into a deterministic, analysis-safe trace shape.
trace_metadata_keys
    Return metadata keys included in trace serialization.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeRelationship, ObjectType, RelatedNodeInfo, TextNode


_TRACE_METADATA_KEYS: tuple[str, ...] = (
    "retrieval_source",
    "retrieval_sources",
    "retrieval_source_ranks",
    "retrieval_signal_trace",
    "source_authority",
    "authority_tier",
    "validity_status",
    "freshness_hint",
    "title",
    "doc_title",
    "system_names",
    "partition_names",
    "service_names",
    "scope_family_names",
    "software_names",
    "software_versions",
    "module_names",
    "toolchain_names",
    "toolchain_versions",
    "official_doc_matches",
)


def _source_relationship(parent_id: str | None) -> dict[NodeRelationship, RelatedNodeInfo]:
    """Source Relationship.
    
    Parameters
    ----------
    parent_id : str or None, optional
        Stable identifier for parent.
    
    Returns
    -------
    dict[NodeRelationship, RelatedNodeInfo]
        Structured result of the operation.
    """
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
    """Convert a chunk-like object into a ``TextNode`` with source linkage.
    
    Parameters
    ----------
    chunk : Any
        Value for chunk.
    
    Returns
    -------
    TextNode
        Result of the operation.
    """
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
    """Convert a document-like object into a ``TextNode``.
    
    Parameters
    ----------
    document : Any
        Value for document.
    
    Returns
    -------
    TextNode
        Result of the operation.
    """
    text = getattr(document, "text", "") or ""
    document_id = str(getattr(document, "id", "") or "")
    document_type = str(getattr(document, "document_type", "") or "")
    metadata = dict(getattr(document, "metadata", {}) or {})

    if document_type:
        metadata["document_type"] = document_type

    return TextNode(text=text, id_=document_id, metadata=metadata)


def extract_doc_id(node: Any) -> str:
    """Extract a stable document identifier from a node-like object.
    
    Parameters
    ----------
    node : Any
        Value for node.
    
    Returns
    -------
    str
        Resulting string value.
    """

    for attr in ("id_", "node_id", "id"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value:
            return value
    return "<unknown-doc-id>"


def extract_text(node: Any) -> str:
    """Extract text content from a node-like object.
    
    Parameters
    ----------
    node : Any
        Value for node.
    
    Returns
    -------
    str
        Resulting string value.
    """

    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text
    if hasattr(node, "get_content"):
        try:
            content = node.get_content()
            return content if isinstance(content, str) else str(content)
        except Exception:
            return ""
    return ""


def serialize_source_nodes(
    source_nodes: list[Any],
    *,
    include_text: bool = True,
) -> list[dict[str, Any]]:
    """Serialize source nodes into a deterministic, analysis-safe trace shape.
    
    Parameters
    ----------
    source_nodes : list[Any]
        Retrieved nodes or node wrappers to serialize.
    include_text : bool, optional
        Whether to include text.
    
    Returns
    -------
    list[dict[str, Any]]
        Serialized source Nodes.
    """

    records: list[dict[str, Any]] = []
    for idx, source in enumerate(source_nodes, start=1):
        node = getattr(source, "node", source)
        score_raw = getattr(source, "score", None)
        score = float(score_raw) if isinstance(score_raw, (int, float)) else None
        metadata = _node_metadata(node)
        rerank_trace = metadata.get("rerank_trace")

        record: dict[str, Any] = {
            "rank": idx,
            "doc_id": extract_doc_id(node),
            "score": score,
            "source": _optional_string(metadata.get("retrieval_source")),
            "retrieval_sources": _string_list(metadata.get("retrieval_sources")),
            "retrieval_source_ranks": _normalized_value(metadata.get("retrieval_source_ranks")),
            "retrieval_signal_trace": _normalized_value(metadata.get("retrieval_signal_trace")),
            "source_authority": _optional_string(metadata.get("source_authority")),
            "authority_tier": _optional_int(metadata.get("authority_tier")),
            "validity_status": _optional_string(metadata.get("validity_status")),
            "freshness_hint": _optional_string(metadata.get("freshness_hint")),
            "title": _optional_string(metadata.get("title")),
            "doc_title": _optional_string(metadata.get("doc_title")),
            "system_names": _string_list(metadata.get("system_names")),
            "partition_names": _string_list(metadata.get("partition_names")),
            "service_names": _string_list(metadata.get("service_names")),
            "scope_family_names": _string_list(metadata.get("scope_family_names")),
            "software_names": _string_list(metadata.get("software_names")),
            "software_versions": _string_list(metadata.get("software_versions")),
            "module_names": _string_list(metadata.get("module_names")),
            "toolchain_names": _string_list(metadata.get("toolchain_names")),
            "toolchain_versions": _string_list(metadata.get("toolchain_versions")),
            "official_doc_matches": _normalize_matches(metadata.get("official_doc_matches")),
            "rerank_trace": _normalized_value(rerank_trace),
        }
        if include_text:
            record["text"] = extract_text(node)
        records.append(record)
    return records


def trace_metadata_keys() -> tuple[str, ...]:
    """Return metadata keys included in trace serialization.
    
    Returns
    -------
    tuple[str, ...]
        Result of the operation.
    """

    return _TRACE_METADATA_KEYS


def _node_metadata(node: Any) -> dict[str, Any]:
    """Node Metadata.
    
    Parameters
    ----------
    node : Any
        Value for node.
    
    Returns
    -------
    dict[str, Any]
        Structured result of the operation.
    """
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _optional_string(value: Any) -> str | None:
    """Optional String.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    str or None
        Result of the operation.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    """Optional Int.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    int or None
        Result of the operation.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    """String List.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalize_matches(value: Any) -> list[dict[str, Any]]:
    """Normalize matches.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    list[dict[str, Any]]
        Collected results from the operation.
    """
    if not isinstance(value, list):
        return []
    matches: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized = {
                "entity_id": _optional_string(item.get("entity_id")),
                "entity_type": _optional_string(item.get("entity_type")),
                "canonical_name": _optional_string(item.get("canonical_name")),
                "doc_id": _optional_string(item.get("doc_id")),
                "doc_title": _optional_string(item.get("doc_title")),
                "source_scope": _optional_string(item.get("source_scope")),
                "status": _optional_string(item.get("status")),
                "match_methods": _string_list(item.get("match_methods")),
                "matched_aliases": _string_list(item.get("matched_aliases")),
                "matched_versions": _string_list(item.get("matched_versions")),
            }
            matches.append(normalized)
    return matches


def _normalized_value(value: Any) -> Any:
    """Normalized Value.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    Any
        Result of the operation.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalized_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_normalized_value(item) for item in value]
    return str(value)
