from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.authority import (
    SOURCE_SCOPE_LOCAL_OFFICIAL,
    SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES,
    build_registry_artifact,
)
from polaris_rag.authority.service_catalog import extract_service_catalog_candidates
from polaris_rag.common import MarkdownDocument


def _doc(*, doc_id: str, source: str, title: str, text: str) -> MarkdownDocument:
    return MarkdownDocument(
        id=doc_id,
        document_type="html",
        text=text,
        metadata={"source": source, "title": title},
    )


def test_extract_service_catalog_candidates_emits_top_level_services_and_subservices() -> None:
    documents = [
        _doc(
            doc_id="https://www.example.org/services",
            source="https://www.example.org/services",
            title="Research Computing Services",
            text=(
                "# Research Computing Services\n\n"
                "High Performance Computing\n"
                "Dawn\n"
                "Research Software Engineering\n"
                "Data Storage\n"
                "Secure Research Computing\n"
                "Cambridge Research Cloud\n"
                "AI & Data Analysis\n"
                "Visualisation\n"
                "Training & Consultancy\n"
            ),
        ),
        _doc(
            doc_id="https://www.example.org/data-storage",
            source="https://www.example.org/data-storage",
            title="Data Storage",
            text=(
                "# Data Storage\n\n"
                "Research File Share (RFS)\n"
                "Research Cold Store (RCS)\n"
                "Research Data Store (RDS)\n"
            ),
        ),
    ]

    candidates, review_rows = extract_service_catalog_candidates(documents)

    assert review_rows == []
    by_key = {(candidate.entity_type, candidate.canonical_name): candidate for candidate in candidates}
    assert ("service", "High Performance Computing") in by_key
    assert ("service", "Research Software Engineering") in by_key
    assert ("service", "Data Storage") in by_key
    assert ("service", "Secure Research Computing") in by_key
    assert ("service", "Cambridge Research Cloud") in by_key
    assert ("service", "AI & Data Analysis") in by_key
    assert ("service", "Visualisation") in by_key
    assert ("service", "Training & Consultancy") in by_key
    assert ("service", "Research File Share") in by_key
    assert ("service", "Research Cold Store") in by_key
    assert ("service", "Research Data Store") in by_key
    assert ("system", "Dawn - Intel GPU (PVC) Nodes") in by_key
    assert "Secure Research Computing Platform (SRCP)" in by_key[("service", "Secure Research Computing")].aliases
    assert "SRCP" in by_key[("service", "Secure Research Computing")].aliases
    assert all(candidate.canonical_name != "Research Computing Services" for candidate in candidates)


def test_service_catalog_merge_prefers_docs_scope_for_overlapping_dawn_entity() -> None:
    docs_document = _doc(
        doc_id="https://docs.example.org/hpc/user-guide/pvc.html",
        source="https://docs.example.org/hpc/user-guide/pvc.html",
        title="Dawn - Intel GPU (PVC) Nodes",
        text="# Dawn - Intel GPU (PVC) Nodes\n\nThese nodes are current.\n",
    )
    service_document = _doc(
        doc_id="https://www.example.org/d-w-n",
        source="https://www.example.org/d-w-n",
        title="Dawn",
        text="# Dawn\n\nDawn is one of the Research Computing Services.\n",
    )

    service_candidates, service_review_rows = extract_service_catalog_candidates([service_document])
    artifact, review_rows = build_registry_artifact(
        [docs_document],
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[str(docs_document.metadata["source"])],
        additional_candidates=service_candidates,
        additional_review_rows=service_review_rows,
        additional_source_urls=[str(service_document.metadata["source"])],
        build_metadata={
            "docs_homepage": "https://docs.example.org/hpc/index.html",
            "services_homepage": "https://www.example.org/services",
            "service_catalog_included": True,
        },
    )

    assert review_rows == []
    dawn_entities = [
        entity
        for entity in artifact.entities
        if entity.entity_type == "system" and entity.canonical_name == "Dawn - Intel GPU (PVC) Nodes"
    ]
    assert len(dawn_entities) == 1
    dawn = dawn_entities[0]
    assert dawn.source_scope == SOURCE_SCOPE_LOCAL_OFFICIAL
    assert dawn.doc_id == "https://docs.example.org/hpc/user-guide/pvc.html"
    assert "Dawn" in dawn.aliases
    assert {span["doc_id"] for span in dawn.evidence_spans} == {
        "https://docs.example.org/hpc/user-guide/pvc.html",
        "https://www.example.org/d-w-n",
    }


def test_service_only_entities_keep_service_catalog_scope() -> None:
    service_document = _doc(
        doc_id="https://www.example.org/secure-research-computing",
        source="https://www.example.org/secure-research-computing",
        title="Secure Research Computing",
        text="# Secure Research Computing\n\nSecure Research Computing is an official service.\n",
    )

    service_candidates, service_review_rows = extract_service_catalog_candidates([service_document])
    artifact, review_rows = build_registry_artifact(
        [],
        homepage="https://docs.example.org/hpc/index.html",
        source_urls=[],
        additional_candidates=service_candidates,
        additional_review_rows=service_review_rows,
        additional_source_urls=[str(service_document.metadata["source"])],
        build_metadata={
            "docs_homepage": "https://docs.example.org/hpc/index.html",
            "services_homepage": "https://www.example.org/services",
            "service_catalog_included": True,
        },
    )

    assert review_rows == []
    secure = next(
        entity for entity in artifact.entities if entity.entity_type == "service" and entity.canonical_name == "Secure Research Computing"
    )
    assert secure.source_scope == SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES
    assert artifact.summary["counts_by_source_scope"][SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES] >= 1
