"""Resolve raw retrieved chunks into the final context passed to generation."""

from __future__ import annotations

from typing import Any, Iterable

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.storage.docstore import BaseDocumentStore


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_node_with_score(item: Any) -> NodeWithScore:
    if isinstance(item, NodeWithScore):
        return item
    return NodeWithScore(node=getattr(item, "node", item), score=_to_float_or_none(getattr(item, "score", None)))


def _extract_text(node: Any) -> str:
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text
    if hasattr(node, "get_content"):
        try:
            content = node.get_content()
            return content if isinstance(content, str) else str(content)
        except Exception:
            return ""
    return str(text or "")


class SupportTicketContextResolver:
    """Expand retrieved ticket chunks to full tickets and deduplicate them."""

    def __init__(
            self,
            *,
            source_document_store: BaseDocumentStore | None = None,
            ticket_document_types: Iterable[str] | None = None,
        ) -> None:
        self._source_document_store = source_document_store
        self._ticket_document_types = frozenset(ticket_document_types or {"helpdesk_ticket"})

    def resolve(self, source_nodes: list[Any]) -> list[NodeWithScore]:
        if not source_nodes:
            return []

        resolved: list[NodeWithScore] = []
        seen_ticket_ids: set[str] = set()

        for item in source_nodes:
            node_with_score = _coerce_node_with_score(item)
            node = node_with_score.node
            metadata = getattr(node, "metadata", None)
            metadata = dict(metadata or {}) if isinstance(metadata, dict) else {}

            document_type = str(metadata.get("document_type") or "")
            parent_id = str(metadata.get("parent_id") or "")

            if document_type in self._ticket_document_types and parent_id:
                if parent_id in seen_ticket_ids:
                    continue
                seen_ticket_ids.add(parent_id)

                full_ticket_node = self._load_full_ticket_node(parent_id)
                if full_ticket_node is not None:
                    resolved.append(NodeWithScore(node=full_ticket_node, score=node_with_score.score))
                    continue

            resolved.append(node_with_score)

        return resolved

    def _load_full_ticket_node(self, ticket_id: str) -> TextNode | None:
        if self._source_document_store is None:
            return None

        try:
            stored = self._source_document_store.get_document(ticket_id, raise_error=False)
        except Exception:
            return None

        if stored is None:
            return None

        metadata = getattr(stored, "metadata", None)
        metadata = dict(metadata or {}) if isinstance(metadata, dict) else {}
        metadata.setdefault("document_type", metadata.get("document_type") or "helpdesk_ticket")
        metadata.setdefault("parent_id", ticket_id)

        return TextNode(
            text=_extract_text(stored),
            id_=ticket_id,
            metadata=metadata,
        )
