"""Deterministic extractor for the official RCS services catalog.

This module exposes public helper functions used by the surrounding Polaris subsystem.

Functions
---------
service_catalog_normalized_path
    Service Catalog Normalized Path.
is_allowed_service_catalog_url
    Return whether allowed Service Catalog URL.
extract_service_catalog_candidates
    Extract service Catalog Candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from urllib.parse import urlsplit

from polaris_rag.common import MarkdownDocument

from .registry_builder import (
    SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES,
    ReviewQueueRow,
    _Candidate,
    _build_candidate,
    _clean_title,
    _document_slug,
    _extract_sections,
    _normalize_spaces,
    _primary_heading_title,
    _slug_aliases,
    _sorted_unique,
)

SERVICE_CATALOG_DEFAULT_HOMEPAGE = "https://www.hpc.cam.ac.uk/services"
SERVICE_CATALOG_MAX_DEPTH = 2


@dataclass(frozen=True)
class _ServiceCatalogSpec:
    entity_type: str
    canonical_name: str
    explicit_aliases: tuple[str, ...] = ()


_SERVICE_CATALOG_PATH_SPECS: dict[str, _ServiceCatalogSpec] = {
    "high-performance-computing": _ServiceCatalogSpec("service", "High Performance Computing"),
    "d-w-n": _ServiceCatalogSpec(
        "system",
        "Dawn - Intel GPU (PVC) Nodes",
        explicit_aliases=("Dawn",),
    ),
    "research-software-engineering": _ServiceCatalogSpec("service", "Research Software Engineering"),
    "data-storage": _ServiceCatalogSpec("service", "Data Storage"),
    "research-file-share": _ServiceCatalogSpec(
        "service",
        "Research File Share",
        explicit_aliases=("Research File Share (RFS)", "RFS"),
    ),
    "research-cold-store": _ServiceCatalogSpec(
        "service",
        "Research Cold Store",
        explicit_aliases=("Research Cold Store (RCS)", "RCS"),
    ),
    "research-data-store": _ServiceCatalogSpec(
        "service",
        "Research Data Store",
        explicit_aliases=("Research Data Store (RDS)", "RDS"),
    ),
    "secure-research-computing": _ServiceCatalogSpec(
        "service",
        "Secure Research Computing",
        explicit_aliases=("Secure Research Computing Platform (SRCP)", "SRCP"),
    ),
    "cambridge-research-cloud": _ServiceCatalogSpec("service", "Cambridge Research Cloud"),
    "ai-data-analysis": _ServiceCatalogSpec("service", "AI & Data Analysis"),
    "visualisation": _ServiceCatalogSpec("service", "Visualisation"),
    "training-consultancy": _ServiceCatalogSpec("service", "Training & Consultancy"),
}

_SERVICE_CATALOG_LANDING_PATHS = {
    "",
    "services",
    "index.php/services",
}
_SERVICE_CATALOG_ALLOWED_PATHS = frozenset(
    {
        *_SERVICE_CATALOG_LANDING_PATHS,
        *_SERVICE_CATALOG_PATH_SPECS.keys(),
    }
)

_TOP_LEVEL_SERVICE_LABELS = (
    "High Performance Computing",
    "Dawn",
    "Research Software Engineering",
    "Data Storage",
    "Secure Research Computing",
    "Cambridge Research Cloud",
    "AI & Data Analysis",
    "Visualisation",
    "Training & Consultancy",
)
_DATA_STORAGE_SUBSERVICE_LABELS = (
    "Research File Share",
    "Research File Share (RFS)",
    "Research Cold Store",
    "Research Cold Store (RCS)",
    "Research Data Store",
    "Research Data Store (RDS)",
)


def service_catalog_normalized_path(url: str) -> str:
    """Service Catalog Normalized Path.
    
    Parameters
    ----------
    url : str
        URL used by the operation.
    
    Returns
    -------
    str
        Resulting string value.
    """
    path = urlsplit(str(url or "")).path.strip("/").lower()
    if path in _SERVICE_CATALOG_LANDING_PATHS:
        return path
    for prefix in ("services/", "index.php/services/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path or ""


def is_allowed_service_catalog_url(homepage: str, url: str) -> bool:
    """Return whether allowed Service Catalog URL.
    
    Parameters
    ----------
    homepage : str
        Value for homepage.
    url : str
        URL used by the operation.
    
    Returns
    -------
    bool
        `True` if allowed Service Catalog URL; otherwise `False`.
    """
    homepage_parts = urlsplit(homepage)
    candidate_parts = urlsplit(url)
    if not candidate_parts.scheme or not candidate_parts.netloc:
        return False
    if candidate_parts.scheme != homepage_parts.scheme or candidate_parts.netloc != homepage_parts.netloc:
        return False

    normalized_path = service_catalog_normalized_path(url)
    if normalized_path not in _SERVICE_CATALOG_ALLOWED_PATHS:
        return False

    candidate_path_lower = candidate_parts.path.lower()
    if candidate_path_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf", ".ico", ".css", ".js", ".xml", ".txt")):
        return False
    return True


def extract_service_catalog_candidates(
    markdown_documents: Sequence[MarkdownDocument],
) -> tuple[list[_Candidate], list[ReviewQueueRow]]:
    """Extract service Catalog Candidates.
    
    Parameters
    ----------
    markdown_documents : Sequence[MarkdownDocument]
        Value for markdown Documents.
    
    Returns
    -------
    tuple[list[_Candidate], list[ReviewQueueRow]]
        Result of the operation.
    """
    candidates: list[_Candidate] = []

    for document in sorted(markdown_documents, key=lambda item: str(item.id)):
        sections = _extract_sections(document.text)
        normalized_path = service_catalog_normalized_path(str(document.id))
        doc_title = _resolve_service_catalog_title(document, sections)

        if normalized_path in _SERVICE_CATALOG_LANDING_PATHS:
            candidates.extend(_extract_landing_page_candidates(document, doc_title))
            continue

        spec = _SERVICE_CATALOG_PATH_SPECS.get(normalized_path)
        if spec is None:
            continue

        aliases = _sorted_unique(
            [
                spec.canonical_name,
                doc_title,
                *_slug_aliases(str(document.id)),
                *spec.explicit_aliases,
                *_explicit_aliases_from_text(str(document.text or ""), spec.canonical_name),
            ]
        )
        candidate = _build_candidate(
            entity_type=spec.entity_type,
            canonical_name=spec.canonical_name,
            aliases=aliases,
            source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES,
            status="current",
            known_versions=(),
            doc_id=str(document.id),
            doc_title=doc_title or spec.canonical_name,
            heading_path=(doc_title,) if doc_title else (),
            evidence_text=_service_catalog_evidence_text(document),
            extraction_method="service_catalog_page",
        )
        if candidate is not None:
            candidates.append(candidate)
        if normalized_path == "data-storage":
            candidates.extend(_extract_data_storage_subservice_candidates(document, doc_title))

    return candidates, []


def _resolve_service_catalog_title(document: MarkdownDocument, sections: Sequence[object]) -> str:
    """Resolve service Catalog Title.
    
    Parameters
    ----------
    document : MarkdownDocument
        Value for document.
    sections : Sequence[object]
        Value for sections.
    
    Returns
    -------
    str
        Resulting string value.
    """
    heading_title = _primary_heading_title(sections)
    if heading_title:
        return heading_title
    metadata_title = _clean_title(str((document.metadata or {}).get("title") or ""))
    if metadata_title:
        return metadata_title
    slug = _document_slug(document)
    return _normalize_spaces(slug.replace("-", " "))


def _service_catalog_evidence_text(document: MarkdownDocument) -> str:
    """Service Catalog Evidence Text.
    
    Parameters
    ----------
    document : MarkdownDocument
        Value for document.
    
    Returns
    -------
    str
        Resulting string value.
    """
    return str(document.text or "").strip()[:400]


def _extract_landing_page_candidates(document: MarkdownDocument, doc_title: str) -> list[_Candidate]:
    """Extract landing Page Candidates.
    
    Parameters
    ----------
    document : MarkdownDocument
        Value for document.
    doc_title : str
        Value for doc Title.
    
    Returns
    -------
    list[_Candidate]
        Collected results from the operation.
    """
    text = str(document.text or "")
    candidates: list[_Candidate] = []

    for label in _TOP_LEVEL_SERVICE_LABELS:
        spec = _service_spec_for_label(label)
        if spec is None or label not in text:
            continue
        candidate = _build_candidate(
            entity_type=spec.entity_type,
            canonical_name=spec.canonical_name,
            aliases=_sorted_unique([spec.canonical_name, label, *spec.explicit_aliases]),
            source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES,
            status="current",
            known_versions=(),
            doc_id=str(document.id),
            doc_title=doc_title or "Services",
            heading_path=(doc_title,) if doc_title else (),
            evidence_text=text[:400],
            extraction_method="service_catalog_landing",
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _extract_data_storage_subservice_candidates(document: MarkdownDocument, doc_title: str) -> list[_Candidate]:
    """Extract data Storage Subservice Candidates.
    
    Parameters
    ----------
    document : MarkdownDocument
        Value for document.
    doc_title : str
        Value for doc Title.
    
    Returns
    -------
    list[_Candidate]
        Collected results from the operation.
    """
    text = str(document.text or "")
    candidates: list[_Candidate] = []
    for label in _DATA_STORAGE_SUBSERVICE_LABELS:
        spec = _service_spec_for_label(label)
        if spec is None or label not in text:
            continue
        candidate = _build_candidate(
            entity_type=spec.entity_type,
            canonical_name=spec.canonical_name,
            aliases=_sorted_unique([spec.canonical_name, label, *spec.explicit_aliases]),
            source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES,
            status="current",
            known_versions=(),
            doc_id=str(document.id),
            doc_title=doc_title or "Data Storage",
            heading_path=(doc_title,) if doc_title else (),
            evidence_text=text[:400],
            extraction_method="service_catalog_page",
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _service_spec_for_label(label: str) -> _ServiceCatalogSpec | None:
    """Service Spec For Label.
    
    Parameters
    ----------
    label : str
        Value for label.
    
    Returns
    -------
    _ServiceCatalogSpec or None
        Result of the operation.
    """
    normalized = _normalize_spaces(label)
    for spec in _SERVICE_CATALOG_PATH_SPECS.values():
        aliases = {spec.canonical_name, *spec.explicit_aliases}
        if normalized in aliases:
            return spec
        if spec.canonical_name == "Dawn - Intel GPU (PVC) Nodes" and normalized == "Dawn":
            return spec
    return None


def _explicit_aliases_from_text(text: str, canonical_name: str) -> list[str]:
    """Explicit Aliases From Text.
    
    Parameters
    ----------
    text : str
        Text value to inspect, tokenize, or encode.
    canonical_name : str
        Value for canonical Name.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
    body = str(text or "")
    aliases: list[str] = []
    explicit_patterns = {
        "Research File Share": ("Research File Share (RFS)", "RFS"),
        "Research Data Store": ("Research Data Store (RDS)", "RDS"),
        "Research Cold Store": ("Research Cold Store (RCS)", "RCS"),
    }
    for alias in explicit_patterns.get(canonical_name, ()):
        if alias in body:
            aliases.append(alias)
    return aliases


__all__ = [
    "SERVICE_CATALOG_DEFAULT_HOMEPAGE",
    "SERVICE_CATALOG_MAX_DEPTH",
    "SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES",
    "extract_service_catalog_candidates",
    "is_allowed_service_catalog_url",
    "service_catalog_normalized_path",
]
