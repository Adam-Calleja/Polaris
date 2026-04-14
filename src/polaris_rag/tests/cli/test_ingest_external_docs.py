from __future__ import annotations

import sys
from types import SimpleNamespace
import types

sys.modules.setdefault("atlassian", types.SimpleNamespace(Jira=object))

from polaris_rag.cli import ingest_external_docs
from polaris_rag.common import Document, DocumentChunk, MarkdownDocument


def test_parse_args_supports_external_ingestion_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_external_docs.py",
            "-c",
            "config/config.yaml",
            "--source-register-file",
            "data/authority/source_register.external_v1.yaml",
            "--source",
            "external_docs",
            "--vector-batch-size",
            "8",
            "--embedding-workers",
            "3",
            "--index-only",
        ],
    )

    args = ingest_external_docs.parse_args()

    assert args.source_register_file.endswith("source_register.external_v1.yaml")
    assert args.source == "external_docs"
    assert args.vector_batch_size == 8
    assert args.embedding_workers == 3
    assert args.index_only is True


def test_main_index_only_creates_collection_without_loading_external_docs(monkeypatch):
    ensured: list[str] = []

    fake_cfg = SimpleNamespace(
        raw={"vector_stores": {"external_docs": {"collection_name": "external_docs"}}},
        embedder={},
    )
    fake_container = SimpleNamespace(
        vector_stores={"external_docs": "external-store"},
        doc_store="chunk-docstore",
    )
    storage_context = SimpleNamespace(
        vector_store=SimpleNamespace(
            ensure_collection_exists=lambda: ensured.append("external_docs"),
        ),
    )

    def _unexpected_load(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("External documents should not be loaded in index-only mode.")

    monkeypatch.setattr(ingest_external_docs.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(ingest_external_docs, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(ingest_external_docs, "_build_source_storage_context", lambda container, source, persist_dir=None: storage_context)
    monkeypatch.setattr(ingest_external_docs, "_load_external_documents", _unexpected_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_external_docs.py",
            "-c",
            "config/config.yaml",
            "--index-only",
        ],
    )

    ingest_external_docs.main()

    assert ensured == ["external_docs"]


def test_main_markdown_chunking_path_ingests_registered_external_docs(monkeypatch):
    inserted: dict[str, object] = {}
    deleted_ids: list[str] = []
    deleted_from_docstore: list[str] = []
    localized_calls: list[dict[str, object]] = []

    fake_cfg = SimpleNamespace(
        raw={"vector_stores": {"external_docs": {"collection_name": "external_docs"}}},
        ingestion={
            "conversion": {"sources": {"external_docs": {"engine": "markitdown"}}},
            "chunking": {"sources": {"external_docs": {"strategy": "markdown_token", "chunk_size_tokens": 64, "overlap_tokens": 8}}},
            "metadata_enrichment": {"authority_registry_path": "data/authority/registry.official_combined.v1.json"},
        },
        storage_context={"persist_dir": "data/storage/local"},
        document_preprocess_html_conditions=[],
        document_preprocess_html_tags=[],
        document_preprocess_html_link_classes=[],
        embedder={},
    )
    fake_container = SimpleNamespace(
        token_counter=SimpleNamespace(),
        vector_stores={"external_docs": "external-store"},
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
            id="https://manual.example.org/gromacs/install.html",
            document_type="html",
            text="<html><body><h1>Installing GROMACS</h1></body></html>",
            metadata={"source": "https://manual.example.org/gromacs/install.html"},
        )
    ]
    markdown_documents = [
        MarkdownDocument(
            id="https://manual.example.org/gromacs/install.html",
            document_type="html",
            text="# Installing GROMACS",
            metadata={"source": "https://manual.example.org/gromacs/install.html"},
        )
    ]
    chunks = [
        DocumentChunk(
            id="https://manual.example.org/gromacs/install.html::chunk::0000",
            parent_id="https://manual.example.org/gromacs/install.html",
            prev_id=None,
            next_id=None,
            text="# Installing GROMACS",
            document_type="html",
            metadata={},
        )
    ]

    monkeypatch.setattr(ingest_external_docs.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(ingest_external_docs, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(ingest_external_docs, "_build_source_storage_context", lambda container, source, persist_dir=None: storage_context)
    monkeypatch.setattr(
        ingest_external_docs,
        "_load_external_documents",
        lambda register_path: (SimpleNamespace(sources=[SimpleNamespace(source_id="gromacs")]), html_documents),
    )
    monkeypatch.setattr(ingest_external_docs, "preprocess_html_documents", lambda documents, tags, conditions: documents)
    monkeypatch.setattr(ingest_external_docs, "convert_documents_to_markdown", lambda documents, engine, options: markdown_documents)
    monkeypatch.setattr(
        ingest_external_docs,
        "enrich_documents_with_authority_metadata",
        lambda documents, registry_artifact_path, source_name: documents,
    )
    monkeypatch.setattr(ingest_external_docs, "get_chunks_from_markdown_documents", lambda documents, token_counter, chunk_size, overlap: chunks)
    monkeypatch.setattr(
        ingest_external_docs,
        "localize_doc_chunk_scope_family_metadata",
        lambda chunks, registry_artifact_path: localized_calls.append(
            {"chunks": chunks, "registry_artifact_path": registry_artifact_path}
        ) or chunks,
    )
    monkeypatch.setattr(ingest_external_docs, "add_chunks_to_docstore", lambda storage, chunks: len(chunks))
    monkeypatch.setattr(ingest_external_docs, "delete_ref_docs_from_docstore", lambda docstore, ids: deleted_from_docstore.extend(ids))
    monkeypatch.setattr(ingest_external_docs, "persist_docstore", lambda docstore, persist_path: None)
    monkeypatch.setattr(
        ingest_external_docs,
        "chunk_document_store_path",
        lambda persist_dir, source: "chunk_docstore.external_docs.json",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_external_docs.py",
            "-c",
            "config/config.yaml",
            "--source-register-file",
            "data/authority/source_register.external_v1.yaml",
        ],
    )

    ingest_external_docs.main()

    assert deleted_ids == ["https://manual.example.org/gromacs/install.html"]
    assert deleted_from_docstore == ["https://manual.example.org/gromacs/install.html"]
    assert localized_calls == [
        {
            "chunks": chunks,
            "registry_artifact_path": "data/authority/registry.official_combined.v1.json",
        }
    ]
    assert inserted["chunks"] == chunks
    assert inserted["batch_size"] == 16
    assert inserted["use_async"] is False
