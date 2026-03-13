from __future__ import annotations

import sys
import types

sys.modules.setdefault("atlassian", types.SimpleNamespace(Jira=object))

from polaris_rag.retrieval.document_loader import canonicalize_url, load_website_docs


def test_canonicalize_url_removes_fragment_and_trailing_slash():
    assert canonicalize_url("HTTPS://Docs.Example.org/guide/#section") == "https://docs.example.org/guide"


def test_load_website_docs_uses_canonical_url_as_document_id(monkeypatch):
    response = types.SimpleNamespace(
        content=b"<html><head><meta charset='utf-8'></head><body><h1>Guide</h1></body></html>",
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr("polaris_rag.retrieval.document_loader.requests.get", lambda url, timeout=10: response)

    documents = load_website_docs(["https://docs.example.org/guide/#intro"])

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "https://docs.example.org/guide"
    assert document.metadata["source"] == "https://docs.example.org/guide"
