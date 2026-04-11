from __future__ import annotations

import sys
import types

sys.modules.setdefault("atlassian", types.SimpleNamespace(Jira=object))

from polaris_rag.retrieval import document_loader
from polaris_rag.retrieval.document_loader import canonicalize_url, iter_support_ticket_batches, load_website_docs


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


def test_iter_support_ticket_batches_streams_paged_results_into_requested_batch_size(monkeypatch):
    responses = iter(
        [
            {
                "issues": [{"key": "HPCSSUP-1"}, {"key": "HPCSSUP-2"}],
                "nextPageToken": "page-2",
            },
            {
                "issues": [{"key": "HPCSSUP-3"}, {"key": "HPCSSUP-4"}],
                "nextPageToken": None,
            },
        ]
    )
    seen_tokens: list[str | None] = []

    class FakeJira:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def enhanced_jql(self, jql_query, nextPageToken=None, expand=None):  # noqa: N803
            seen_tokens.append(nextPageToken)
            return next(responses)

    monkeypatch.setitem(sys.modules, "atlassian", types.SimpleNamespace(Jira=FakeJira))

    batches = list(
        iter_support_ticket_batches(
            cfg=types.SimpleNamespace(jira_api_credentials={"username": "user", "password": "token"}),
            start_date="2024-01-01",
            end_date="2024-02-01",
            limit=None,
            batch_size=3,
        )
    )

    assert [[ticket["key"] for ticket in batch] for batch in batches] == [
        ["HPCSSUP-1", "HPCSSUP-2", "HPCSSUP-3"],
        ["HPCSSUP-4"],
    ]
    assert seen_tokens == [None, "page-2"]


def test_load_support_tickets_flattens_streamed_batches(monkeypatch):
    monkeypatch.setattr(
        document_loader,
        "iter_support_ticket_batches",
        lambda **kwargs: iter(
            [
                [{"key": "HPCSSUP-1"}],
                [{"key": "HPCSSUP-2"}, {"key": "HPCSSUP-3"}],
            ]
        ),
    )

    tickets = document_loader.load_support_tickets(
        cfg=types.SimpleNamespace(jira_api_credentials={"username": "user", "password": "token"}),
        limit=None,
    )

    assert [ticket["key"] for ticket in tickets] == ["HPCSSUP-1", "HPCSSUP-2", "HPCSSUP-3"]
