"""Authority-registry utilities for offline source-of-truth artifacts."""

from __future__ import annotations

from .registry_builder import (
    EXTRACTION_VERSION,
    REGISTRY_ENTITY_TYPES,
    REVIEW_STATE_AUTO_VERIFIED,
    REVIEW_STATE_NEEDS_REVIEW,
    SOURCE_SCOPE_EXTERNAL_OFFICIAL,
    SOURCE_SCOPE_LOCAL_OFFICIAL,
    SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES,
    STATUS_VALUES,
    RegistryArtifact,
    RegistryEntity,
    RegistrySourceDocument,
    ReviewQueueRow,
    build_registry_artifact,
    extract_registry_candidates,
    load_registry_artifact,
    merge_registry_artifacts,
    persist_registry_artifact,
    persist_review_rows,
)

__all__ = [
    "EXTRACTION_VERSION",
    "REGISTRY_ENTITY_TYPES",
    "REVIEW_STATE_AUTO_VERIFIED",
    "REVIEW_STATE_NEEDS_REVIEW",
    "SOURCE_SCOPE_EXTERNAL_OFFICIAL",
    "SOURCE_SCOPE_LOCAL_OFFICIAL",
    "SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES",
    "STATUS_VALUES",
    "RegistryArtifact",
    "RegistryEntity",
    "RegistrySourceDocument",
    "ReviewQueueRow",
    "build_registry_artifact",
    "extract_registry_candidates",
    "load_registry_artifact",
    "merge_registry_artifacts",
    "persist_registry_artifact",
    "persist_review_rows",
]
