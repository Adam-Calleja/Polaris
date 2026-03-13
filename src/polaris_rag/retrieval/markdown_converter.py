"""Conversion helpers for normalizing sources to Markdown."""

from __future__ import annotations

import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from bs4 import BeautifulSoup

from polaris_rag.common import Document, MarkdownDocument
from polaris_rag.retrieval.document_preprocessor import convert_jira_ticket_to_markdown
from polaris_rag.retrieval.ingestion_settings import (
    MARKITDOWN_CONVERSION_ENGINE,
    NATIVE_JIRA_CONVERSION_ENGINE,
)

_SUFFIX_BY_DOCUMENT_TYPE = {
    "html": ".html",
    "pdf": ".pdf",
    "docx": ".docx",
    "pptx": ".pptx",
    "xlsx": ".xlsx",
    "csv": ".csv",
    "xml": ".xml",
    "json": ".json",
    "txt": ".txt",
}


class MarkdownConverter(Protocol):
    """Convert a source object into a markdown-normalized document."""

    def convert(self, source: Any) -> MarkdownDocument:
        """Return markdown-normalized content for ``source``."""


def _normalise_markdown(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_title(html: str) -> str | None:
    soup = BeautifulSoup(html or "", "html.parser")
    title = soup.title.string if soup.title and soup.title.string else ""
    title = str(title or "").strip()
    return title or None


def _document_suffix(document: Document) -> str:
    metadata = dict(getattr(document, "metadata", {}) or {})
    source_path = metadata.get("source_path")
    if source_path:
        suffix = Path(str(source_path)).suffix
        if suffix:
            return suffix
    return _SUFFIX_BY_DOCUMENT_TYPE.get(document.document_type, ".txt")


def _markitdown_convert(document: Document, *, options: dict[str, Any] | None = None) -> MarkdownDocument:
    try:
        from markitdown import MarkItDown  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "MarkItDown is required for the 'markitdown' conversion engine. "
            "Install the ingestion dependencies for this project."
        ) from exc

    options = dict(options or {})
    converter = MarkItDown(**options)
    metadata = dict(document.metadata or {})
    title = _html_title(document.text) if document.document_type == "html" else None
    source_path = metadata.get("source_path")

    if source_path:
        result = converter.convert(str(source_path))
    else:
        suffix = _document_suffix(document)
        with NamedTemporaryFile(suffix=suffix, mode="w", encoding="utf-8") as handle:
            handle.write(document.text)
            handle.flush()
            result = converter.convert(handle.name)

    markdown_text = _normalise_markdown(getattr(result, "text_content", ""))
    if title and not metadata.get("title"):
        metadata["title"] = title

    metadata["conversion_engine"] = MARKITDOWN_CONVERSION_ENGINE
    metadata["source_format"] = document.document_type
    metadata["content_format"] = "markdown"

    return MarkdownDocument(
        text=markdown_text,
        document_type=document.document_type,
        id=document.id,
        metadata=metadata,
        source_node=document,
    )


class MarkItDownDocumentConverter:
    """Markdown converter backed by Microsoft's MarkItDown."""

    def __init__(self, *, options: dict[str, Any] | None = None) -> None:
        self.options = dict(options or {})

    def convert(self, source: Document) -> MarkdownDocument:
        return _markitdown_convert(source, options=self.options)


class NativeJiraMarkdownConverter:
    """Markdown converter for structured Jira ticket payloads."""

    def convert(self, source: dict[str, Any]) -> MarkdownDocument:
        return convert_jira_ticket_to_markdown(source)


def build_markdown_converter(
    engine: str,
    *,
    options: dict[str, Any] | None = None,
) -> MarkdownConverter:
    normalized = (engine or "").strip().lower().replace("-", "_")
    if normalized == MARKITDOWN_CONVERSION_ENGINE:
        return MarkItDownDocumentConverter(options=options)
    if normalized == NATIVE_JIRA_CONVERSION_ENGINE:
        return NativeJiraMarkdownConverter()
    raise ValueError(f"Unsupported markdown conversion engine: {engine!r}")


def convert_documents_to_markdown(
    documents: list[Document],
    *,
    engine: str = MARKITDOWN_CONVERSION_ENGINE,
    options: dict[str, Any] | None = None,
) -> list[MarkdownDocument]:
    converter = build_markdown_converter(engine, options=options)
    return [converter.convert(document) for document in documents]


def convert_tickets_to_markdown(
    tickets: list[dict[str, Any]],
    *,
    engine: str = NATIVE_JIRA_CONVERSION_ENGINE,
    options: dict[str, Any] | None = None,
) -> list[MarkdownDocument]:
    converter = build_markdown_converter(engine, options=options)
    return [converter.convert(ticket) for ticket in tickets]


__all__ = [
    "MarkdownConverter",
    "MarkItDownDocumentConverter",
    "NativeJiraMarkdownConverter",
    "build_markdown_converter",
    "convert_documents_to_markdown",
    "convert_tickets_to_markdown",
]
