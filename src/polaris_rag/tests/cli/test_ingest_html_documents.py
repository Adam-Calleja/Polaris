from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import types

from polaris_rag.common import Document, DocumentChunk, MarkdownDocument


_MODULE_PATH = Path(__file__).resolve().parents[4] / "scripts" / "ingest_html_documents.py"
_SPEC = spec_from_file_location("test_ingest_html_documents_module", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
sys.modules.setdefault("atlassian", types.SimpleNamespace(Jira=object))
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ingest_html_documents = _MODULE


def test_build_source_storage_context_uses_named_store(monkeypatch):
    called: dict[str, object] = {}

    def fake_build_storage_context(*, vector_store, docstore, persist_dir):
        called["vector_store"] = vector_store
        called["docstore"] = docstore
        called["persist_dir"] = persist_dir
        return "storage-context"

    monkeypatch.setattr(ingest_html_documents, "build_storage_context", fake_build_storage_context)

    container = SimpleNamespace(
        vector_stores={"docs": "docs-store"},
        doc_store="chunk-docstore",
    )

    result = ingest_html_documents._build_source_storage_context(container, "docs")

    assert result == "storage-context"
    assert called == {
        "vector_store": "docs-store",
        "docstore": "chunk-docstore",
        "persist_dir": None,
    }


def test_override_qdrant_collection_name_updates_selected_source():
    cfg = SimpleNamespace(raw={"vector_stores": {"docs": {"collection_name": "old"}}})

    ingest_html_documents._override_qdrant_collection_name(cfg, "docs", "new-collection")

    assert cfg.raw["vector_stores"]["docs"]["collection_name"] == "new-collection"


def test_parse_args_supports_source_and_batching_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_html_documents.py",
            "-c",
            "config/config.yaml",
            "-p",
            "https://docs.example.org",
            "--source",
            "docs",
            "--vector-batch-size",
            "8",
            "--embedding-workers",
            "3",
            "--chunking-strategy",
            "markdown_token",
            "--conversion-engine",
            "markitdown",
            "--chunk-size-tokens",
            "256",
            "--chunk-overlap-tokens",
            "32",
        ],
    )

    args = ingest_html_documents.parse_args()

    assert args.source == "docs"
    assert args.vector_batch_size == 8
    assert args.embedding_workers == 3
    assert args.chunking_strategy == "markdown_token"
    assert args.conversion_engine == "markitdown"
    assert args.chunk_size_tokens == 256
    assert args.chunk_overlap_tokens == 32


def test_main_markdown_chunking_path_replaces_existing_docs(monkeypatch):
    inserted: dict[str, object] = {}
    deleted_ids: list[str] = []
    deleted_from_docstore: list[str] = []

    fake_cfg = SimpleNamespace(
        raw={"vector_stores": {"docs": {"collection_name": "docs"}}},
        ingestion={
            "conversion": {"sources": {"docs": {"engine": "markitdown"}}},
            "chunking": {"sources": {"docs": {"strategy": "markdown_token", "chunk_size_tokens": 64, "overlap_tokens": 8}}},
        },
        storage_context={"persist_dir": "data/storage/local"},
        document_preprocess_html_conditions=[],
        document_preprocess_html_tags=[],
        document_preprocess_html_link_classes=[],
        embedder={},
    )
    fake_container = SimpleNamespace(
        token_counter=SimpleNamespace(),
        vector_stores={"docs": "docs-store"},
        doc_store="chunk-docstore",
    )
    storage_context = SimpleNamespace(
        vector_store=SimpleNamespace(
            delete_ref_docs=lambda ids: deleted_ids.extend(ids),
            insert_chunks=lambda chunks, batch_size, use_async: inserted.update({"chunks": chunks, "batch_size": batch_size, "use_async": use_async}),
        ),
        docstore="docstore",
    )
    html_documents = [
        Document(
            id="https://docs.example.org/guide",
            document_type="html",
            text="<html><body><h1>Guide</h1></body></html>",
            metadata={"source": "https://docs.example.org/guide"},
        )
    ]
    markdown_documents = [
        MarkdownDocument(
            id="https://docs.example.org/guide",
            document_type="html",
            text="# Guide",
            metadata={"source": "https://docs.example.org/guide"},
        )
    ]
    chunks = [
        DocumentChunk(
            id="https://docs.example.org/guide::chunk::0000",
            parent_id="https://docs.example.org/guide",
            prev_id=None,
            next_id=None,
            text="# Guide",
            document_type="html",
            metadata={},
        )
    ]

    monkeypatch.setattr(ingest_html_documents.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(ingest_html_documents, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(ingest_html_documents, "_build_source_storage_context", lambda container, source: storage_context)
    monkeypatch.setattr(ingest_html_documents, "load_website_docs", lambda links: html_documents)
    monkeypatch.setattr(ingest_html_documents, "preprocess_html_documents", lambda documents, tags, conditions: documents)
    monkeypatch.setattr(ingest_html_documents, "convert_documents_to_markdown", lambda documents, engine, options: markdown_documents)
    monkeypatch.setattr(ingest_html_documents, "get_chunks_from_markdown_documents", lambda documents, token_counter, chunk_size, overlap: chunks)
    monkeypatch.setattr(ingest_html_documents, "add_chunks_to_docstore", lambda storage, chunks: len(chunks))
    monkeypatch.setattr(ingest_html_documents, "delete_ref_docs_from_docstore", lambda docstore, ids: deleted_from_docstore.extend(ids))
    monkeypatch.setattr(ingest_html_documents, "persist_storage", lambda storage, persist_dir: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_html_documents.py",
            "-c",
            "config/config.yaml",
            "-p",
            "https://docs.example.org",
            "--chunking-strategy",
            "markdown_token",
        ],
    )

    ingest_html_documents.main()

    assert deleted_ids == ["https://docs.example.org/guide"]
    assert deleted_from_docstore == ["https://docs.example.org/guide"]
    assert inserted["chunks"] == chunks
    assert inserted["batch_size"] == 16
    assert inserted["use_async"] is False
