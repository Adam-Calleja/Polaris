from __future__ import annotations

import sys
import types
from pathlib import Path

from polaris_rag.common import Document
from polaris_rag.retrieval.document_preprocessor import preprocess_html_documents
from polaris_rag.retrieval.markdown_converter import (
    convert_documents_to_markdown,
    convert_tickets_to_markdown,
)


def _install_fake_markitdown(monkeypatch, capture: dict[str, str]) -> None:
    class FakeMarkItDown:
        def __init__(self, **kwargs):
            capture["kwargs"] = kwargs

        def convert(self, path: str):
            capture["path"] = path
            capture["input_text"] = Path(path).read_text(encoding="utf-8")
            return types.SimpleNamespace(text_content="# Converted\n\nBody text")

    monkeypatch.setitem(sys.modules, "markitdown", types.SimpleNamespace(MarkItDown=FakeMarkItDown))


def _make_ticket() -> dict:
    return {
        "key": "HPCSSUP-1",
        "fields": {
            "summary": "Cannot access docs",
            "description": {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Initial problem description."}],
            },
            "status": {"name": "Resolved"},
            "created": "2025-01-01T09:00:00.000+0000",
            "updated": "2025-01-01T10:00:00.000+0000",
            "resolutionDate": "2025-01-01T11:00:00.000+0000",
            "comment": {
                "comments": [
                    {
                        "author": {"emailAddress": "creator@example.com", "displayName": "Creator"},
                        "body": {"type": "paragraph", "content": [{"type": "text", "text": "First comment"}]},
                    },
                    {
                        "author": {"emailAddress": "assignee@example.com", "displayName": "Support"},
                        "body": {"type": "paragraph", "content": [{"type": "text", "text": "Reply from support"}]},
                    },
                ]
            },
            "assignee": {"emailAddress": "assignee@example.com"},
            "creator": {"emailAddress": "creator@example.com"},
            "customfield_10042": [{"emailAddress": "pi@example.com"}],
            "reporter": {"emailAddress": "reporter@example.com"},
        },
    }


def test_convert_documents_to_markdown_uses_markitdown_and_preserves_metadata(monkeypatch):
    capture: dict[str, str] = {}
    _install_fake_markitdown(monkeypatch, capture)

    html = """
    <html>
      <head><title>Example Guide</title></head>
      <body>
        <nav>navigation</nav>
        <main><h1>Guide</h1><p>Useful body text</p></main>
      </body>
    </html>
    """
    processed_documents = preprocess_html_documents(
        [Document(id="https://docs.example.org/guide", document_type="html", text=html, metadata={"source": "https://docs.example.org/guide"})],
        tags=["nav"],
        conditions=[],
    )

    markdown_documents = convert_documents_to_markdown(processed_documents, engine="markitdown")

    assert len(markdown_documents) == 1
    markdown_document = markdown_documents[0]
    assert markdown_document.id == "https://docs.example.org/guide"
    assert markdown_document.text == "# Converted\n\nBody text"
    assert markdown_document.metadata["title"] == "Example Guide"
    assert markdown_document.metadata["conversion_engine"] == "markitdown"
    assert markdown_document.metadata["content_format"] == "markdown"
    assert "navigation" not in capture["input_text"]


def test_convert_tickets_to_markdown_preserves_ticket_structure():
    documents = convert_tickets_to_markdown([_make_ticket()], engine="native_jira")

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "HPCSSUP-1"
    assert document.metadata["conversion_engine"] == "native_jira"
    assert document.metadata["content_format"] == "markdown"
    assert "# Ticket HPCSSUP-1" in document.text
    assert "## Initial Description" in document.text
    assert "### Message 0001 (TICKET_CREATOR)" in document.text
    assert "### Message 0002 (HELPDESK_ASSIGNEE)" in document.text
