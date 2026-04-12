from __future__ import annotations

import importlib
import sys
from datetime import datetime
from types import SimpleNamespace

from polaris_rag.common import DocumentChunk, MarkdownDocument

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
            "--fetch-batch-size",
            "25",
            "--embedding-workers",
            "3",
            "--source",
            "tickets",
            "--chunking-strategy",
            "markdown_token",
            "--conversion-engine",
            "native_jira",
            "--chunk-size-tokens",
            "512",
            "--chunk-overlap-tokens",
            "64",
            "--clear-collection",
            "--index-only",
        ],
    )

    args = ingest_jira_tickets.parse_args()

    assert args.exclude_keys_file == "excluded.txt"
    assert args.vector_batch_size == 8
    assert args.fetch_batch_size == 25
    assert args.embedding_workers == 3
    assert args.source == "tickets"
    assert args.chunking_strategy == "markdown_token"
    assert args.conversion_engine == "native_jira"
    assert args.chunk_size_tokens == 512
    assert args.chunk_overlap_tokens == 64
    assert args.clear_collection is True
    assert args.index_only is True


def test_main_index_only_creates_collection_without_loading_tickets(monkeypatch):
    ensured: list[str] = []

    fake_cfg = SimpleNamespace(
        raw={"vector_stores": {"tickets": {"collection_name": "tickets"}}},
        embedder={},
    )
    fake_container = SimpleNamespace(
        vector_stores={"tickets": "ticket-store"},
        doc_store="chunk-docstore",
    )
    storage_context = SimpleNamespace(
        vector_store=SimpleNamespace(
            ensure_collection_exists=lambda: ensured.append("tickets"),
        ),
    )

    monkeypatch.setattr(ingest_jira_tickets.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(ingest_jira_tickets, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(ingest_jira_tickets, "_build_source_storage_context", lambda container, source: storage_context)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_jira_tickets.py",
            "-c",
            "config/config.yaml",
            "--index-only",
        ],
    )

    ingest_jira_tickets.main()

    assert ensured == ["tickets"]


def test_main_markdown_chunking_path_persists_markdown_tickets(monkeypatch):
    inserted: dict[str, object] = {}
    deleted_ids: list[str] = []
    deleted_from_docstore: list[str] = []
    persisted_docs: list[MarkdownDocument] = []

    fake_cfg = SimpleNamespace(
        raw={"vector_stores": {"tickets": {"collection_name": "tickets"}}},
        ingestion={
            "conversion": {"sources": {"tickets": {"engine": "native_jira"}}},
            "chunking": {"sources": {"tickets": {"strategy": "markdown_token", "chunk_size_tokens": 64, "overlap_tokens": 8}}},
            "jira": {},
        },
        storage_context={"persist_dir": "data/storage/local"},
        embedder={},
    )
    fake_container = SimpleNamespace(
        token_counter=SimpleNamespace(),
        vector_stores={"tickets": "ticket-store"},
        doc_store="chunk-docstore",
    )
    storage_context = SimpleNamespace(
        vector_store=SimpleNamespace(
            delete_ref_docs=lambda ids: deleted_ids.extend(ids),
            insert_chunks=lambda chunks, batch_size, use_async: inserted.update({"chunks": chunks, "batch_size": batch_size, "use_async": use_async}),
        ),
        docstore="docstore",
    )
    source_docstore = SimpleNamespace()
    markdown_documents = [
        MarkdownDocument(
            id="HPCSSUP-1",
            document_type="helpdesk_ticket",
            text="# Ticket HPCSSUP-1",
            metadata={"ticket_key": "HPCSSUP-1", "summary": "Summary"},
        )
    ]
    chunks = [
        DocumentChunk(
            id="HPCSSUP-1::chunk::0000",
            parent_id="HPCSSUP-1",
            prev_id=None,
            next_id=None,
            text="# Ticket HPCSSUP-1",
            document_type="helpdesk_ticket",
            metadata={},
        )
    ]

    monkeypatch.setattr(ingest_jira_tickets.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(ingest_jira_tickets, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(ingest_jira_tickets, "_build_source_storage_context", lambda container, source: storage_context)
    monkeypatch.setattr(ingest_jira_tickets, "load_or_create_source_document_store", lambda persist_dir: source_docstore)
    monkeypatch.setattr(ingest_jira_tickets, "persist_storage", lambda storage, persist_dir: None)
    monkeypatch.setattr(ingest_jira_tickets, "persist_docstore", lambda docstore, persist_path: None)
    monkeypatch.setattr(ingest_jira_tickets, "source_document_store_path", lambda persist_dir: "source_docstore.json")
    monkeypatch.setattr(ingest_jira_tickets, "add_chunks_to_docstore", lambda storage, chunks: len(chunks))
    monkeypatch.setattr(ingest_jira_tickets, "delete_ref_docs_from_docstore", lambda docstore, ids: deleted_from_docstore.extend(ids))
    monkeypatch.setattr(ingest_jira_tickets, "add_documents_to_docstore", lambda docstore, documents: persisted_docs.extend(documents))
    monkeypatch.setattr(
        ingest_jira_tickets,
        "_resolve_dates",
        lambda cfg, start_cli, end_cli: ("2024-01-01", "2024-02-01"),
    )
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_limit", lambda cfg, limit_cli: None)
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_unwanted_summaries", lambda cfg: [])
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_exclude_keys", lambda cfg, cli_value: [])
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_persist_dir", lambda cfg, cli_value: "data/storage/local")
    monkeypatch.setattr(
        ingest_jira_tickets,
        "resolve_authority_registry_artifact_path",
        lambda cfg: "data/authority/registry.local_official.v1.json",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_jira_tickets.py",
            "-c",
            "config/config.yaml",
            "--chunking-strategy",
            "markdown_token",
        ],
    )

    sys.modules.setdefault("atlassian", SimpleNamespace(Jira=object))
    document_loader = importlib.import_module("polaris_rag.retrieval.document_loader")
    monkeypatch.setattr(
        document_loader,
        "iter_support_ticket_batches",
        lambda start_date, end_date, limit, cfg, exclude_keys, batch_size: iter(
            [[{"key": "HPCSSUP-1", "fields": {"summary": "Summary"}}]]
        ),
    )
    monkeypatch.setattr(
        ingest_jira_tickets,
        "prepare_jira_tickets_for_chunking",
        lambda tickets, chunking_strategy, conversion_engine, conversion_options, registry_artifact_path, source_name: markdown_documents,
    )
    monkeypatch.setattr(
        ingest_jira_tickets,
        "chunk_processed_jira_tickets",
        lambda processed_tickets, chunking_strategy, token_counter, chunk_size, overlap: chunks,
    )

    ingest_jira_tickets.main()

    assert deleted_ids == ["HPCSSUP-1"]
    assert deleted_from_docstore == ["HPCSSUP-1"]
    assert inserted["chunks"] == chunks
    assert inserted["batch_size"] == 16
    assert inserted["use_async"] is False
    assert persisted_docs == markdown_documents


def test_main_clear_collection_resets_target_once_and_skips_batch_deletes(monkeypatch, tmp_path):
    recreate_calls: list[str] = []
    deleted_ids: list[str] = []
    deleted_from_docstore: list[str] = []
    persist_dir = tmp_path / "storage"
    persist_dir.mkdir(parents=True)
    stale_file = persist_dir / "stale.json"
    stale_file.write_text("stale", encoding="utf-8")

    fake_cfg = SimpleNamespace(
        raw={"vector_stores": {"tickets": {"collection_name": "tickets"}}},
        ingestion={
            "conversion": {"sources": {"tickets": {"engine": "native_jira"}}},
            "chunking": {"sources": {"tickets": {"strategy": "markdown_token", "chunk_size_tokens": 64, "overlap_tokens": 8}}},
            "jira": {},
        },
        storage_context={"persist_dir": str(persist_dir)},
        embedder={},
    )
    fake_container = SimpleNamespace(
        token_counter=SimpleNamespace(),
        vector_stores={"tickets": "ticket-store"},
        doc_store="chunk-docstore",
    )
    storage_context = SimpleNamespace(
        vector_store=SimpleNamespace(
            recreate_collection=lambda: recreate_calls.append("tickets"),
            delete_ref_docs=lambda ids: deleted_ids.extend(ids),
            insert_chunks=lambda chunks, batch_size, use_async: None,
        ),
        docstore="docstore",
    )
    source_docstore = SimpleNamespace()

    monkeypatch.setattr(ingest_jira_tickets.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(ingest_jira_tickets, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(ingest_jira_tickets, "_build_source_storage_context", lambda container, source: storage_context)
    monkeypatch.setattr(ingest_jira_tickets, "load_or_create_source_document_store", lambda persist_dir: source_docstore)
    monkeypatch.setattr(ingest_jira_tickets, "persist_storage", lambda storage, persist_dir: None)
    monkeypatch.setattr(ingest_jira_tickets, "persist_docstore", lambda docstore, persist_path: None)
    monkeypatch.setattr(ingest_jira_tickets, "source_document_store_path", lambda persist_dir: "source_docstore.json")
    monkeypatch.setattr(ingest_jira_tickets, "add_chunks_to_docstore", lambda storage, chunks: len(chunks))
    monkeypatch.setattr(ingest_jira_tickets, "delete_ref_docs_from_docstore", lambda docstore, ids: deleted_from_docstore.extend(ids))
    monkeypatch.setattr(
        ingest_jira_tickets,
        "add_documents_to_docstore",
        lambda docstore, documents: len(documents),
    )
    monkeypatch.setattr(
        ingest_jira_tickets,
        "_resolve_dates",
        lambda cfg, start_cli, end_cli: ("2024-01-01", "2024-02-01"),
    )
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_limit", lambda cfg, limit_cli: None)
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_unwanted_summaries", lambda cfg: [])
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_exclude_keys", lambda cfg, cli_value: [])
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_persist_dir", lambda cfg, cli_value: str(persist_dir))
    monkeypatch.setattr(
        ingest_jira_tickets,
        "resolve_authority_registry_artifact_path",
        lambda cfg: "data/authority/registry.local_official.v1.json",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_jira_tickets.py",
            "-c",
            "config/config.yaml",
            "--chunking-strategy",
            "markdown_token",
            "--clear-collection",
        ],
    )

    sys.modules.setdefault("atlassian", SimpleNamespace(Jira=object))
    document_loader = importlib.import_module("polaris_rag.retrieval.document_loader")
    monkeypatch.setattr(
        document_loader,
        "iter_support_ticket_batches",
        lambda start_date, end_date, limit, cfg, exclude_keys, batch_size: iter(
            [[{"key": "HPCSSUP-1", "fields": {"summary": "Summary"}}]]
        ),
    )
    monkeypatch.setattr(
        ingest_jira_tickets,
        "prepare_jira_tickets_for_chunking",
        lambda tickets, chunking_strategy, conversion_engine, conversion_options, registry_artifact_path, source_name: [
            MarkdownDocument(
                id="HPCSSUP-1",
                document_type="helpdesk_ticket",
                text="# Ticket HPCSSUP-1",
                metadata={"ticket_key": "HPCSSUP-1", "summary": "Summary"},
            )
        ],
    )
    monkeypatch.setattr(
        ingest_jira_tickets,
        "chunk_processed_jira_tickets",
        lambda processed_tickets, chunking_strategy, token_counter, chunk_size, overlap: [
            DocumentChunk(
                id="HPCSSUP-1::chunk::0000",
                parent_id="HPCSSUP-1",
                prev_id=None,
                next_id=None,
                text="# Ticket HPCSSUP-1",
                document_type="helpdesk_ticket",
                metadata={},
            )
        ],
    )

    ingest_jira_tickets.main()

    assert recreate_calls == ["tickets"]
    assert deleted_ids == []
    assert deleted_from_docstore == []
    assert not stale_file.exists()


def test_main_processes_jira_tickets_in_fetch_batches(monkeypatch):
    inserted_batches: list[list[str]] = []
    deleted_ids: list[str] = []
    deleted_from_docstore: list[str] = []
    persisted_docs: list[str] = []
    persist_storage_calls: list[str] = []
    persist_docstore_calls: list[str] = []

    fake_cfg = SimpleNamespace(
        raw={"vector_stores": {"tickets": {"collection_name": "tickets"}}},
        ingestion={
            "conversion": {"sources": {"tickets": {"engine": "native_jira"}}},
            "chunking": {"sources": {"tickets": {"strategy": "markdown_token", "chunk_size_tokens": 64, "overlap_tokens": 8}}},
            "jira": {},
        },
        storage_context={"persist_dir": "data/storage/local"},
        embedder={},
    )
    fake_container = SimpleNamespace(
        token_counter=SimpleNamespace(),
        vector_stores={"tickets": "ticket-store"},
        doc_store="chunk-docstore",
    )
    storage_context = SimpleNamespace(
        vector_store=SimpleNamespace(
            delete_ref_docs=lambda ids: deleted_ids.extend(ids),
            insert_chunks=lambda chunks, batch_size, use_async: inserted_batches.append(
                [chunk.parent_id for chunk in chunks]
            ),
        ),
        docstore="docstore",
    )
    source_docstore = SimpleNamespace()

    monkeypatch.setattr(ingest_jira_tickets.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(ingest_jira_tickets, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(ingest_jira_tickets, "_build_source_storage_context", lambda container, source: storage_context)
    monkeypatch.setattr(ingest_jira_tickets, "load_or_create_source_document_store", lambda persist_dir: source_docstore)
    monkeypatch.setattr(ingest_jira_tickets, "persist_storage", lambda storage, persist_dir: persist_storage_calls.append(persist_dir))
    monkeypatch.setattr(ingest_jira_tickets, "persist_docstore", lambda docstore, persist_path: persist_docstore_calls.append(persist_path))
    monkeypatch.setattr(ingest_jira_tickets, "source_document_store_path", lambda persist_dir: "source_docstore.json")
    monkeypatch.setattr(ingest_jira_tickets, "add_chunks_to_docstore", lambda storage, chunks: len(chunks))
    monkeypatch.setattr(ingest_jira_tickets, "delete_ref_docs_from_docstore", lambda docstore, ids: deleted_from_docstore.extend(ids))
    monkeypatch.setattr(
        ingest_jira_tickets,
        "add_documents_to_docstore",
        lambda docstore, documents: persisted_docs.extend([document.id for document in documents]),
    )
    monkeypatch.setattr(
        ingest_jira_tickets,
        "_resolve_dates",
        lambda cfg, start_cli, end_cli: ("2024-01-01", "2024-02-01"),
    )
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_limit", lambda cfg, limit_cli: None)
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_unwanted_summaries", lambda cfg: [])
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_exclude_keys", lambda cfg, cli_value: [])
    monkeypatch.setattr(ingest_jira_tickets, "_resolve_persist_dir", lambda cfg, cli_value: "data/storage/local")
    monkeypatch.setattr(
        ingest_jira_tickets,
        "resolve_authority_registry_artifact_path",
        lambda cfg: "data/authority/registry.local_official.v1.json",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_jira_tickets.py",
            "-c",
            "config/config.yaml",
            "--chunking-strategy",
            "markdown_token",
            "--fetch-batch-size",
            "2",
        ],
    )

    sys.modules.setdefault("atlassian", SimpleNamespace(Jira=object))
    document_loader = importlib.import_module("polaris_rag.retrieval.document_loader")
    monkeypatch.setattr(
        document_loader,
        "iter_support_ticket_batches",
        lambda start_date, end_date, limit, cfg, exclude_keys, batch_size: iter(
            [
                [
                    {"key": "HPCSSUP-1", "fields": {"summary": "Summary 1"}},
                    {"key": "HPCSSUP-2", "fields": {"summary": "Summary 2"}},
                ],
                [
                    {"key": "HPCSSUP-3", "fields": {"summary": "Summary 3"}},
                ],
            ]
        ),
    )
    monkeypatch.setattr(
        ingest_jira_tickets,
        "prepare_jira_tickets_for_chunking",
        lambda tickets, chunking_strategy, conversion_engine, conversion_options, registry_artifact_path, source_name: [
            MarkdownDocument(
                id=ticket["key"],
                document_type="helpdesk_ticket",
                text=f"# Ticket {ticket['key']}",
                metadata={"ticket_key": ticket["key"], "summary": ticket["fields"]["summary"]},
            )
            for ticket in tickets
        ],
    )
    monkeypatch.setattr(
        ingest_jira_tickets,
        "chunk_processed_jira_tickets",
        lambda processed_tickets, chunking_strategy, token_counter, chunk_size, overlap: [
            DocumentChunk(
                id=f"{document.id}::chunk::0000",
                parent_id=document.id,
                prev_id=None,
                next_id=None,
                text=document.text,
                document_type=document.document_type,
                metadata={},
            )
            for document in processed_tickets
        ],
    )

    ingest_jira_tickets.main()

    assert inserted_batches == [["HPCSSUP-1", "HPCSSUP-2"], ["HPCSSUP-3"]]
    assert deleted_ids == ["HPCSSUP-1", "HPCSSUP-2", "HPCSSUP-3"]
    assert deleted_from_docstore == ["HPCSSUP-1", "HPCSSUP-2", "HPCSSUP-3"]
    assert persisted_docs == ["HPCSSUP-1", "HPCSSUP-2", "HPCSSUP-3"]
    assert persist_storage_calls == ["data/storage/local", "data/storage/local"]
    assert persist_docstore_calls == ["source_docstore.json", "source_docstore.json"]
