from __future__ import annotations

import json
from pathlib import Path

from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

from polaris_rag.common import Document, DocumentChunk, MarkdownDocument
from polaris_rag.retrieval.document_store_factory import add_chunks_to_docstore, add_documents_to_docstore
from polaris_rag.retrieval.markdown_chunker import MarkdownTokenChunker
from polaris_rag.retrieval.metadata_enricher import (
    enrich_documents_with_authority_metadata,
    localize_doc_chunk_scope_family_metadata,
)
from polaris_rag.retrieval.node_utils import chunk_to_text_node
from polaris_rag.retrieval.text_splitter import get_chunks_from_jira_ticket


class DummyTokenCounter:
    def count(self, text: str) -> int:
        return len((text or "").split())

    def tail(self, text: str, n_tokens: int) -> str:
        tokens = (text or "").split()
        return " ".join(tokens[-n_tokens:])

    def split(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
        tokens = (text or "").split()
        if not tokens:
            return []
        step = max_tokens - overlap_tokens
        return [
            " ".join(tokens[start : start + max_tokens])
            for start in range(0, len(tokens), step)
            if tokens[start : start + max_tokens]
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.inserted_nodes = []

    def insert_chunks(self, chunks, batch_size: int, *, use_async: bool = False) -> None:
        self.inserted_nodes = [chunk_to_text_node(chunk) for chunk in chunks]


def _entity(
    *,
    entity_id: str,
    entity_type: str,
    canonical_name: str,
    doc_id: str,
    aliases: list[str] | None = None,
    known_versions: list[str] | None = None,
    status: str = "unknown",
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "aliases": list(aliases or [canonical_name]),
        "source_scope": "local_official",
        "status": status,
        "known_versions": list(known_versions or []),
        "doc_id": doc_id,
        "doc_title": canonical_name,
        "heading_path": [canonical_name],
        "evidence_spans": [],
        "extraction_method": "test_fixture",
        "review_state": "auto_verified",
    }


def _write_registry(tmp_path: Path, *, entities: list[dict[str, object]], source_urls: list[str]) -> Path:
    path = tmp_path / "registry.local_official.v1.json"
    payload = {
        "build": {
            "homepage": "https://docs.example.org/hpc/index.html",
            "source_scope": "local_official",
            "extraction_version": "authority_registry_v1",
        },
        "source_urls": source_urls,
        "entities": entities,
        "summary": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_markdown_docs_propagate_enriched_authority_metadata_to_chunks(tmp_path: Path) -> None:
    doc_id = "https://docs.example.org/hpc/gromacs.html"
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-gromacs",
                entity_type="software",
                canonical_name="GROMACS",
                aliases=["GROMACS", "gromacs"],
                known_versions=["2021.3", "2024.4"],
                status="current",
                doc_id=doc_id,
            ),
            _entity(
                entity_id="module-gromacs-2024.4",
                entity_type="module",
                canonical_name="gromacs/2024.4",
                aliases=["gromacs/2024.4"],
                known_versions=["2024.4"],
                status="current",
                doc_id=doc_id,
            ),
            _entity(
                entity_id="module-rhel8-cclake-base",
                entity_type="module",
                canonical_name="rhel8/cclake/base",
                aliases=["rhel8/cclake/base"],
                status="current",
                doc_id=doc_id,
            ),
        ],
        source_urls=[doc_id],
    )
    document = MarkdownDocument(
        id=doc_id,
        document_type="html",
        text="# GROMACS\n\nUse `module load rhel8/cclake/base gromacs/2024.4` to begin.",
        metadata={"source": doc_id, "title": "GROMACS"},
    )

    [enriched_document] = enrich_documents_with_authority_metadata(
        [document],
        registry_artifact_path=registry_path,
        source_name="docs",
    )
    chunks = MarkdownTokenChunker(
        token_counter=DummyTokenCounter(),
        chunk_size_tokens=6,
        overlap_tokens=1,
    ).chunk(enriched_document)

    assert enriched_document.metadata["source_authority"] == "local_official"
    assert enriched_document.metadata["authority_tier"] == 3
    assert enriched_document.metadata["software_names"] == ["GROMACS"]
    assert enriched_document.metadata["module_names"] == ["gromacs/2024.4", "rhel8/cclake/base"]
    assert enriched_document.metadata["scope_family_names"] == ["cclake"]
    assert enriched_document.metadata["software_versions"] == ["2021.3", "2024.4"]
    assert enriched_document.metadata["validity_status"] == "current"
    assert len(enriched_document.metadata["official_doc_matches"]) == 3

    assert chunks
    assert chunks[0].metadata["source_authority"] == "local_official"
    assert chunks[0].metadata["software_names"] == ["GROMACS"]
    assert chunks[0].metadata["module_names"] == ["gromacs/2024.4", "rhel8/cclake/base"]
    assert chunks[0].metadata["scope_family_names"] == ["cclake"]
    assert chunks[0].metadata["validity_status"] == "current"
    assert len(chunks[0].metadata["official_doc_matches"]) == 3


def test_service_catalog_docs_still_enrich_as_local_official(tmp_path: Path) -> None:
    doc_id = "https://www.example.org/secure-research-computing"
    registry_path = _write_registry(
        tmp_path,
        entities=[
            {
                **_entity(
                    entity_id="service-secure-research-computing",
                    entity_type="service",
                    canonical_name="Secure Research Computing",
                    aliases=["Secure Research Computing"],
                    status="current",
                    doc_id=doc_id,
                ),
                "source_scope": "local_official_services",
            }
        ],
        source_urls=[doc_id],
    )
    document = MarkdownDocument(
        id=doc_id,
        document_type="html",
        text="# Secure Research Computing\n\nSecure Research Computing is available.",
        metadata={"source": doc_id, "title": "Secure Research Computing"},
    )

    [enriched_document] = enrich_documents_with_authority_metadata(
        [document],
        registry_artifact_path=registry_path,
    )

    assert enriched_document.metadata["source_authority"] == "local_official"
    assert enriched_document.metadata["authority_tier"] == 3
    assert enriched_document.metadata["service_names"] == ["Secure Research Computing"]
    assert len(enriched_document.metadata["official_doc_matches"]) == 1


def test_ticket_metadata_extraction_populates_authority_versions_and_privacy_flags(tmp_path: Path) -> None:
    doc_id = "https://docs.example.org/hpc/gromacs.html"
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-gromacs",
                entity_type="software",
                canonical_name="GROMACS",
                aliases=["GROMACS", "gromacs"],
                known_versions=["2021.3", "2024.4"],
                status="current",
                doc_id=doc_id,
            ),
            _entity(
                entity_id="module-gromacs-2021.3",
                entity_type="module",
                canonical_name="gromacs/2021.3",
                aliases=["gromacs/2021.3"],
                known_versions=["2021.3"],
                status="current",
                doc_id=doc_id,
            ),
            _entity(
                entity_id="toolchain-cuda-12.1",
                entity_type="toolchain",
                canonical_name="cuda/12.1",
                aliases=["cuda/12.1"],
                known_versions=["12.1"],
                status="current",
                doc_id="https://docs.example.org/hpc/cuda.html",
            ),
            _entity(
                entity_id="module-rhel8-cclake-base",
                entity_type="module",
                canonical_name="rhel8/cclake/base",
                aliases=["rhel8/cclake/base"],
                status="current",
                doc_id="https://docs.example.org/hpc/cclake.html",
            ),
        ],
        source_urls=[doc_id, "https://docs.example.org/hpc/cuda.html", "https://docs.example.org/hpc/cclake.html"],
    )
    ticket = Document(
        id="HPCSSUP-1",
        document_type="helpdesk_ticket",
        text=(
            "[INITIAL_DESCRIPTION]\n"
            "module load rhel8/cclake/base gromacs/2021.3 cuda/12.1\n"
            "Please contact me at user@example.com.\n"
            "The failing input is under /home/abc123/project/run.\n"
            "The compute node was 131.111.8.42.\n"
        ),
        metadata={
            "summary": "GROMACS run failure",
            "created_at": "2025-01-01T09:00:00.000+0000",
            "updated_at": "2025-01-01T10:00:00.000+0000",
            "resolved_at": "2025-01-01T11:00:00.000+0000",
        },
    )

    [enriched_ticket] = enrich_documents_with_authority_metadata(
        [ticket],
        registry_artifact_path=registry_path,
        source_name="tickets",
    )

    assert enriched_ticket.metadata["source_authority"] == "ticket_memory"
    assert enriched_ticket.metadata["authority_tier"] == 1
    assert enriched_ticket.metadata["software_names"] == ["GROMACS"]
    assert enriched_ticket.metadata["software_versions"] == ["2021.3"]
    assert enriched_ticket.metadata["scope_family_names"] == ["cclake"]
    assert enriched_ticket.metadata["module_names"] == ["gromacs/2021.3", "rhel8/cclake/base"]
    assert enriched_ticket.metadata["toolchain_names"] == ["cuda/12.1"]
    assert enriched_ticket.metadata["toolchain_versions"] == ["12.1"]
    assert enriched_ticket.metadata["validity_status"] == "current"
    assert enriched_ticket.metadata["freshness_hint"] == "2025-01-01T11:00:00.000+0000"
    assert enriched_ticket.metadata["privacy_flags"] == [
        "contains_email_address",
        "contains_filesystem_path",
        "contains_ipv4_address",
    ]
    assert [match["canonical_name"] for match in enriched_ticket.metadata["official_doc_matches"]] == [
        "gromacs/2021.3",
        "rhel8/cclake/base",
        "GROMACS",
        "cuda/12.1",
    ]


def test_jira_turn_chunker_preserves_enriched_ticket_metadata(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-gromacs",
                entity_type="software",
                canonical_name="GROMACS",
                aliases=["gromacs"],
                known_versions=["2021.3"],
                status="current",
                doc_id="https://docs.example.org/hpc/gromacs.html",
            ),
            _entity(
                entity_id="module-rhel8-cclake-base",
                entity_type="module",
                canonical_name="rhel8/cclake/base",
                aliases=["rhel8/cclake/base"],
                status="current",
                doc_id="https://docs.example.org/hpc/cclake.html",
            ),
        ],
        source_urls=["https://docs.example.org/hpc/gromacs.html", "https://docs.example.org/hpc/cclake.html"],
    )
    ticket = Document(
        id="HPCSSUP-2",
        document_type="helpdesk_ticket",
        text=(
            "[INITIAL_DESCRIPTION]\n"
            "module load rhel8/cclake/base gromacs/2021.3\n"
            "[CONVERSATION]\n"
            "<MESSAGE id=0001 role=TICKET_CREATOR>\n"
            "Still failing after reload.\n"
            "</MESSAGE>\n"
        ),
        metadata={
            "summary": "GROMACS ticket",
            "created_at": "2025-01-01T09:00:00.000+0000",
            "updated_at": "2025-01-01T10:00:00.000+0000",
            "resolved_at": "2025-01-01T11:00:00.000+0000",
        },
    )

    [enriched_ticket] = enrich_documents_with_authority_metadata(
        [ticket],
        registry_artifact_path=registry_path,
        source_name="tickets",
    )
    chunks = get_chunks_from_jira_ticket(
        enriched_ticket,
        token_counter=DummyTokenCounter(),
        chunk_size=80,
        overlap=10,
    )

    assert len(chunks) == 2
    assert all(chunk.metadata["source_authority"] == "ticket_memory" for chunk in chunks)
    assert all(chunk.metadata["software_names"] == ["GROMACS"] for chunk in chunks)
    assert all(chunk.metadata["scope_family_names"] == ["cclake"] for chunk in chunks)
    assert all(chunk.metadata["validity_status"] == "current" for chunk in chunks)


def test_doc_chunk_scope_family_localization_uses_chunk_text_not_parent_doc_scope(tmp_path: Path) -> None:
    doc_id = "https://docs.example.org/hpc/mixed.html"
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="module-rhel8-cclake-base",
                entity_type="module",
                canonical_name="rhel8/cclake/base",
                aliases=["rhel8/cclake/base"],
                status="current",
                doc_id=doc_id,
            ),
            _entity(
                entity_id="module-rhel8-default-amp",
                entity_type="module",
                canonical_name="rhel8/default-amp",
                aliases=["rhel8/default-amp"],
                status="current",
                doc_id=doc_id,
            ),
        ],
        source_urls=[doc_id],
    )

    chunks = [
        DocumentChunk(
            id=f"{doc_id}::chunk::0000",
            parent_id=doc_id,
            prev_id=None,
            next_id=f"{doc_id}::chunk::0001",
            text="module load rhel8/cclake/base",
            document_type="html",
            metadata={"scope_family_names": ["ampere", "cclake"]},
        ),
        DocumentChunk(
            id=f"{doc_id}::chunk::0001",
            parent_id=doc_id,
            prev_id=f"{doc_id}::chunk::0000",
            next_id=None,
            text="module load rhel8/default-amp",
            document_type="html",
            metadata={"scope_family_names": ["ampere", "cclake"]},
        ),
    ]

    localized = localize_doc_chunk_scope_family_metadata(
        chunks,
        registry_artifact_path=registry_path,
    )

    assert localized[0].metadata["scope_family_names"] == ["cclake"]
    assert localized[1].metadata["scope_family_names"] == ["ampere"]


def test_unmatched_ticket_mentions_remain_neutral(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-gromacs",
                entity_type="software",
                canonical_name="GROMACS",
                aliases=["gromacs"],
                known_versions=["2021.3"],
                status="current",
                doc_id="https://docs.example.org/hpc/gromacs.html",
            )
        ],
        source_urls=["https://docs.example.org/hpc/gromacs.html"],
    )
    ticket = Document(
        id="HPCSSUP-3",
        document_type="helpdesk_ticket",
        text="[INITIAL_DESCRIPTION]\nunknownsoft 9.9 fails on startup.\n",
        metadata={"summary": "Unknown package problem", "updated_at": "2025-01-01T10:00:00.000+0000"},
    )

    [enriched_ticket] = enrich_documents_with_authority_metadata(
        [ticket],
        registry_artifact_path=registry_path,
        source_name="tickets",
    )

    assert enriched_ticket.metadata["source_authority"] == "ticket_memory"
    assert enriched_ticket.metadata["software_names"] == []
    assert enriched_ticket.metadata["software_versions"] == []
    assert enriched_ticket.metadata["official_doc_matches"] == []
    assert enriched_ticket.metadata["validity_status"] == "unknown"
    assert enriched_ticket.metadata["validity_hint"] is None


def test_enriched_metadata_survives_vector_and_docstore_round_trip(tmp_path: Path) -> None:
    doc_id = "https://docs.example.org/hpc/gromacs.html"
    registry_path = _write_registry(
        tmp_path,
        entities=[
            _entity(
                entity_id="software-gromacs",
                entity_type="software",
                canonical_name="GROMACS",
                aliases=["gromacs"],
                known_versions=["2024.4"],
                status="current",
                doc_id=doc_id,
            ),
            _entity(
                entity_id="module-rhel8-cclake-base",
                entity_type="module",
                canonical_name="rhel8/cclake/base",
                aliases=["rhel8/cclake/base"],
                status="current",
                doc_id=doc_id,
            ),
        ],
        source_urls=[doc_id],
    )
    document = MarkdownDocument(
        id=doc_id,
        document_type="html",
        text="# GROMACS\n\nmodule load rhel8/cclake/base\n\nReference text for persistence validation.",
        metadata={"source": doc_id},
    )
    [enriched_document] = enrich_documents_with_authority_metadata(
        [document],
        registry_artifact_path=registry_path,
        source_name="docs",
    )
    chunks = MarkdownTokenChunker(
        token_counter=DummyTokenCounter(),
        chunk_size_tokens=12,
        overlap_tokens=2,
    ).chunk(enriched_document)

    vector_store = FakeVectorStore()
    vector_store.insert_chunks(chunks, batch_size=4, use_async=False)
    assert vector_store.inserted_nodes
    assert vector_store.inserted_nodes[0].metadata["source_authority"] == "local_official"
    assert vector_store.inserted_nodes[0].metadata["scope_family_names"] == ["cclake"]
    assert vector_store.inserted_nodes[0].metadata["validity_status"] == "current"

    chunk_docstore = SimpleDocumentStore()
    chunk_storage = StorageContext.from_defaults(docstore=chunk_docstore)
    add_chunks_to_docstore(storage=chunk_storage, chunks=chunks)
    chunk_path = tmp_path / "chunk_docstore.json"
    chunk_docstore.persist(persist_path=str(chunk_path))
    reloaded_chunk_docstore = SimpleDocumentStore.from_persist_path(str(chunk_path))

    source_docstore = SimpleDocumentStore()
    add_documents_to_docstore(source_docstore, [enriched_document])
    source_path = tmp_path / "source_docstore.json"
    source_docstore.persist(persist_path=str(source_path))
    reloaded_source_docstore = SimpleDocumentStore.from_persist_path(str(source_path))

    stored_chunk = reloaded_chunk_docstore.get_document(chunks[0].id)
    stored_document = reloaded_source_docstore.get_document(doc_id)

    assert stored_chunk.metadata["source_authority"] == "local_official"
    assert stored_chunk.metadata["scope_family_names"] == ["cclake"]
    assert stored_chunk.metadata["software_names"] == ["GROMACS"]
    assert stored_document.metadata["source_authority"] == "local_official"
    assert stored_document.metadata["scope_family_names"] == ["cclake"]
    assert stored_document.metadata["validity_status"] == "current"
