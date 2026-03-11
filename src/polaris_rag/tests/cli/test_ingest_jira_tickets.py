from __future__ import annotations

import sys
from datetime import datetime
from types import SimpleNamespace

from polaris_rag.cli import ingest_jira_tickets


def test_build_source_storage_context_uses_named_store(monkeypatch):
    called: dict[str, object] = {}

    def fake_build_storage_context(*, vector_store, docstore, persist_dir):
        called["vector_store"] = vector_store
        called["docstore"] = docstore
        called["persist_dir"] = persist_dir
        return "storage-context"

    monkeypatch.setattr(ingest_jira_tickets, "build_storage_context", fake_build_storage_context)

    container = SimpleNamespace(
        vector_stores={"tickets": "ticket-store"},
        doc_store="chunk-docstore",
    )

    result = ingest_jira_tickets._build_source_storage_context(container, "tickets")

    assert result == "storage-context"
    assert called == {
        "vector_store": "ticket-store",
        "docstore": "chunk-docstore",
        "persist_dir": None,
    }


def test_override_qdrant_collection_name_updates_selected_source():
    cfg = SimpleNamespace(raw={"vector_stores": {"tickets": {"collection_name": "old"}}})

    ingest_jira_tickets._override_qdrant_collection_name(cfg, "tickets", "new-collection")

    assert cfg.raw["vector_stores"]["tickets"]["collection_name"] == "new-collection"


def test_resolve_dates_and_limit_use_current_defaults():
    cfg = SimpleNamespace(ingestion={})

    start_date, end_date = ingest_jira_tickets._resolve_dates(cfg, None, None)

    assert start_date == "2024-01-01"
    assert end_date == datetime.now().strftime("%Y-%m-%d")
    assert ingest_jira_tickets._resolve_limit(cfg, None) is None


def test_parse_args_supports_exclusion_and_batching_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_jira_tickets.py",
            "-c",
            "config/config.yaml",
            "--exclude-keys-file",
            "excluded.txt",
            "--vector-batch-size",
            "8",
            "--embedding-workers",
            "3",
            "--source",
            "tickets",
        ],
    )

    args = ingest_jira_tickets.parse_args()

    assert args.exclude_keys_file == "excluded.txt"
    assert args.vector_batch_size == 8
    assert args.embedding_workers == 3
    assert args.source == "tickets"
