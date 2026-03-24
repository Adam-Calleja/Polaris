"""Authority-aware metadata enrichment for documents and ticket corpora."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from polaris_rag.authority import RegistryEntity
from polaris_rag.retrieval.scope_family import ScopeFamilyResolver

ENRICHMENT_VERSION = "authority_metadata_v1"
DEFAULT_REGISTRY_ARTIFACT_PATH = "data/authority/registry.local_official.v1.json"

SOURCE_AUTHORITY_LOCAL_OFFICIAL = "local_official"
SOURCE_AUTHORITY_EXTERNAL_OFFICIAL = "external_official"
SOURCE_AUTHORITY_TICKET_MEMORY = "ticket_memory"
SOURCE_AUTHORITY_UNKNOWN = "unknown"

AUTHORITY_TIER_BY_SOURCE_AUTHORITY = {
    SOURCE_AUTHORITY_LOCAL_OFFICIAL: 3,
    SOURCE_AUTHORITY_EXTERNAL_OFFICIAL: 2,
    SOURCE_AUTHORITY_TICKET_MEMORY: 1,
    SOURCE_AUTHORITY_UNKNOWN: 0,
}

VALIDITY_UNKNOWN = "unknown"

_BOUNDARY_CHARS = r"A-Za-z0-9"
_MODULE_LOAD_PATTERN = re.compile(
    r"(?:^|&&|;)\s*(?:\$+\s*)?module\s+load\s+([^\n#;]+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_GENERIC_VERSION_PATTERN = re.compile(
    r"\b(?:\d+\.\d+(?:\.\d+){0,2}(?:[-+_][A-Za-z0-9][A-Za-z0-9.+_-]*)?|r\d{4}[a-z])\b",
    flags=re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
_IPV4_PATTERN = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"
)
_PATH_PATTERN = re.compile(r"(?<!\w)/(?:[A-Za-z0-9._-]+/){1,}[A-Za-z0-9._-]+")

DocumentT = TypeVar("DocumentT")


@dataclass(frozen=True)
class _AliasEntry:
    alias: str
    entity: RegistryEntity
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class _VersionEntry:
    version: str
    pattern: re.Pattern[str]


@dataclass
class _EntityMatch:
    entity: RegistryEntity
    match_methods: set[str]
    matched_aliases: set[str]
    matched_versions: set[str]


@dataclass(frozen=True)
class AuthorityRegistryIndex:
    artifact_path: str
    build: dict[str, Any]
    source_urls: frozenset[str]
    entities: tuple[RegistryEntity, ...]
    entities_by_doc_id: Mapping[str, tuple[RegistryEntity, ...]]
    entities_by_alias: Mapping[str, tuple[RegistryEntity, ...]]
    alias_entries: tuple[_AliasEntry, ...]
    version_entries: tuple[_VersionEntry, ...]

    @classmethod
    def load(cls, path: str | Path) -> "AuthorityRegistryIndex":
        resolved = Path(path).expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))

        entities = tuple(
            RegistryEntity(
                entity_id=str(entity["entity_id"]),
                entity_type=str(entity["entity_type"]),
                canonical_name=str(entity["canonical_name"]),
                aliases=[str(alias) for alias in entity.get("aliases", [])],
                source_scope=str(entity["source_scope"]),
                status=str(entity["status"]),
                known_versions=[str(version) for version in entity.get("known_versions", [])],
                doc_id=str(entity["doc_id"]),
                doc_title=str(entity["doc_title"]),
                heading_path=[str(part) for part in entity.get("heading_path", [])],
                evidence_spans=list(entity.get("evidence_spans", [])),
                extraction_method=str(entity["extraction_method"]),
                review_state=str(entity["review_state"]),
            )
            for entity in payload.get("entities", [])
        )

        entities_by_doc_id: dict[str, list[RegistryEntity]] = defaultdict(list)
        entities_by_alias: dict[str, list[RegistryEntity]] = defaultdict(list)
        alias_entries: list[_AliasEntry] = []
        version_entries: list[_VersionEntry] = []
        seen_versions: set[str] = set()

        for entity in entities:
            entities_by_doc_id[str(entity.doc_id)].append(entity)

            for alias in _sorted_unique([entity.canonical_name, *entity.aliases]):
                entities_by_alias[alias.lower()].append(entity)
                pattern = _compile_boundary_pattern(alias)
                if pattern is not None:
                    alias_entries.append(_AliasEntry(alias=alias, entity=entity, pattern=pattern))

            for version in _sorted_unique(entity.known_versions):
                version_key = version.lower()
                if version_key in seen_versions:
                    continue
                seen_versions.add(version_key)
                pattern = _compile_boundary_pattern(version)
                if pattern is not None:
                    version_entries.append(_VersionEntry(version=version, pattern=pattern))

        alias_entries.sort(key=lambda entry: (-len(entry.alias), entry.alias.lower(), entry.entity.entity_id))
        version_entries.sort(key=lambda entry: (-len(entry.version), entry.version.lower()))

        return cls(
            artifact_path=str(resolved),
            build=dict(payload.get("build", {})),
            source_urls=frozenset(str(url) for url in payload.get("source_urls", [])),
            entities=entities,
            entities_by_doc_id={key: tuple(value) for key, value in entities_by_doc_id.items()},
            entities_by_alias={key: tuple(value) for key, value in entities_by_alias.items()},
            alias_entries=tuple(alias_entries),
            version_entries=tuple(version_entries),
        )


class AuthorityMetadataEnricher:
    """Attach registry-backed authority metadata to documents prior to chunking."""

    def __init__(
        self,
        registry_index: AuthorityRegistryIndex,
        *,
        source_name: str | None = None,
    ) -> None:
        self.registry_index = registry_index
        self.source_name = str(source_name or "").strip().lower() or None
        self.scope_family_resolver = ScopeFamilyResolver(registry_index.entities)

    @classmethod
    def from_registry_artifact(
        cls,
        path: str | Path,
        *,
        source_name: str | None = None,
    ) -> "AuthorityMetadataEnricher":
        return cls(AuthorityRegistryIndex.load(path), source_name=source_name)

    def enrich_documents(self, documents: Sequence[DocumentT]) -> list[DocumentT]:
        return [self.enrich_document(document) for document in documents]

    def enrich_document(self, document: DocumentT) -> DocumentT:
        metadata = dict(getattr(document, "metadata", {}) or {})
        matches = (
            self._match_ticket_entities(document)
            if str(getattr(document, "document_type", "") or "") == "helpdesk_ticket"
            else self._match_document_entities(document)
        )
        source_authority = self._source_authority_for(document)
        authority_tier = AUTHORITY_TIER_BY_SOURCE_AUTHORITY.get(source_authority, 0)

        enriched_metadata = {
            "source_authority": source_authority,
            "authority_tier": authority_tier,
            "system_names": self._names_for_type(matches, "system"),
            "partition_names": self._names_for_type(matches, "partition"),
            "service_names": self._names_for_type(matches, "service"),
            "scope_family_names": self._scope_family_names(matches, document=document),
            "software_names": self._names_for_type(matches, "software"),
            "software_versions": self._versions_for_type(matches, "software", document=document),
            "module_names": self._names_for_type(matches, "module"),
            "toolchain_names": self._names_for_type(matches, "toolchain"),
            "toolchain_versions": self._versions_for_type(matches, "toolchain", document=document),
            "official_doc_matches": self._serialise_matches(matches),
            "validity_status": self._aggregate_validity_status(matches),
            "validity_hint": self._build_validity_hint(matches),
            "freshness_hint": self._freshness_hint(metadata),
            "privacy_flags": self._privacy_flags(document, metadata),
            "provenance": self._build_provenance(matches),
        }

        metadata.update(enriched_metadata)
        return replace(document, metadata=metadata)

    def _source_authority_for(self, document: Any) -> str:
        document_type = str(getattr(document, "document_type", "") or "")
        if document_type == "helpdesk_ticket" or self.source_name == "tickets":
            return SOURCE_AUTHORITY_TICKET_MEMORY
        if self.source_name == "docs":
            return SOURCE_AUTHORITY_LOCAL_OFFICIAL

        source = self._document_source(document)
        if source and (source in self.registry_index.source_urls or source in self.registry_index.entities_by_doc_id):
            return SOURCE_AUTHORITY_LOCAL_OFFICIAL

        return SOURCE_AUTHORITY_UNKNOWN

    def _match_document_entities(self, document: Any) -> list[_EntityMatch]:
        source = self._document_source(document)
        matches: dict[str, _EntityMatch] = {}
        for entity in self.registry_index.entities_by_doc_id.get(source, ()):
            self._record_match(matches, entity, match_method="doc_id_join")
        return self._sorted_matches(matches.values())

    def _match_ticket_entities(self, document: Any) -> list[_EntityMatch]:
        text = str(getattr(document, "text", "") or "")
        matches: dict[str, _EntityMatch] = {}
        observed_versions = self._extract_version_mentions(text)

        for module_token in self._extract_module_tokens(text):
            entity_list = self.registry_index.entities_by_alias.get(module_token.lower(), ())
            for entity in entity_list:
                self._record_match(
                    matches,
                    entity,
                    match_method="module_load_parse",
                    matched_alias=module_token,
                    matched_versions=self._matched_versions(entity, observed_versions),
                )

        for alias_entry in self.registry_index.alias_entries:
            if not alias_entry.pattern.search(text):
                continue
            self._record_match(
                matches,
                alias_entry.entity,
                match_method="alias_scan",
                matched_alias=alias_entry.alias,
                matched_versions=self._matched_versions(alias_entry.entity, observed_versions),
            )

        return self._sorted_matches(matches.values())

    def _matched_versions(self, entity: RegistryEntity, observed_versions: Sequence[str]) -> set[str]:
        observed_by_key = {version.lower(): version for version in observed_versions}
        matched: set[str] = set()
        for version in entity.known_versions:
            key = version.lower()
            if key in observed_by_key:
                matched.add(version)
        return matched

    def _versions_for_type(
        self,
        matches: Sequence[_EntityMatch],
        entity_type: str,
        *,
        document: Any,
    ) -> list[str]:
        document_type = str(getattr(document, "document_type", "") or "")
        if document_type == "helpdesk_ticket":
            versions = []
            for match in matches:
                if match.entity.entity_type != entity_type:
                    continue
                versions.extend(match.matched_versions)
            return _sorted_unique(versions)

        versions = []
        for match in matches:
            if match.entity.entity_type != entity_type:
                continue
            versions.extend(match.entity.known_versions)
        return _sorted_unique(versions)

    def _scope_family_names(
        self,
        matches: Sequence[_EntityMatch],
        *,
        document: Any,
    ) -> list[str]:
        families = set(self.scope_family_resolver.families_for_entities(match.entity for match in matches))
        text = str(getattr(document, "text", "") or "")
        for module_token in self._extract_module_tokens(text):
            families.update(self.scope_family_resolver.families_for_text(module_token))
        return _sorted_unique(families)

    def _names_for_type(self, matches: Sequence[_EntityMatch], entity_type: str) -> list[str]:
        return _sorted_unique(
            match.entity.canonical_name
            for match in matches
            if match.entity.entity_type == entity_type
        )

    def _aggregate_validity_status(self, matches: Sequence[_EntityMatch]) -> str:
        explicit_statuses = sorted(
            {
                match.entity.status
                for match in matches
                if str(match.entity.status or "").strip() and match.entity.status != VALIDITY_UNKNOWN
            }
        )
        if len(explicit_statuses) == 1:
            return explicit_statuses[0]
        return VALIDITY_UNKNOWN

    def _build_validity_hint(self, matches: Sequence[_EntityMatch]) -> dict[str, Any] | None:
        if not matches:
            return None

        status_counts: dict[str, int] = defaultdict(int)
        for match in matches:
            status_counts[str(match.entity.status or VALIDITY_UNKNOWN)] += 1

        return {
            "status_source": "registry",
            "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
            "matched_entity_ids": [match.entity.entity_id for match in self._sorted_matches(matches)],
        }

    def _privacy_flags(self, document: Any, metadata: Mapping[str, Any]) -> list[str]:
        flags: list[str] = []
        text = str(getattr(document, "text", "") or "")
        summary = str(metadata.get("summary", "") or "")
        combined = "\n".join(part for part in (text, summary) if part)

        if _EMAIL_PATTERN.search(combined):
            flags.append("contains_email_address")
        if _IPV4_PATTERN.search(combined):
            flags.append("contains_ipv4_address")
        if _PATH_PATTERN.search(combined):
            flags.append("contains_filesystem_path")

        return _sorted_unique(flags)

    def _freshness_hint(self, metadata: Mapping[str, Any]) -> str | None:
        for key in ("resolved_at", "updated_at", "created_at", "last_modified", "modified_at", "published_at"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return None

    def _build_provenance(self, matches: Sequence[_EntityMatch]) -> dict[str, Any]:
        return {
            "enrichment_version": ENRICHMENT_VERSION,
            "registry_artifact_path": self.registry_index.artifact_path,
            "registry_extraction_version": self.registry_index.build.get("extraction_version"),
            "matched_entity_ids": [match.entity.entity_id for match in self._sorted_matches(matches)],
        }

    def _serialise_matches(self, matches: Sequence[_EntityMatch]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for match in self._sorted_matches(matches):
            records.append(
                {
                    "entity_id": match.entity.entity_id,
                    "entity_type": match.entity.entity_type,
                    "canonical_name": match.entity.canonical_name,
                    "doc_id": match.entity.doc_id,
                    "doc_title": match.entity.doc_title,
                    "source_scope": match.entity.source_scope,
                    "status": match.entity.status,
                    "match_methods": sorted(match.match_methods),
                    "matched_aliases": _sorted_unique(match.matched_aliases),
                    "matched_versions": _sorted_unique(match.matched_versions),
                }
            )
        return records

    def _extract_version_mentions(self, text: str) -> list[str]:
        observed = {match.group(0) for match in _GENERIC_VERSION_PATTERN.finditer(text or "")}
        for version_entry in self.registry_index.version_entries:
            if version_entry.pattern.search(text or ""):
                observed.add(version_entry.version)
        return _sorted_unique(observed)

    @staticmethod
    def _extract_module_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        for match in _MODULE_LOAD_PATTERN.finditer(text or ""):
            raw_segment = match.group(1) or ""
            for candidate in raw_segment.split():
                cleaned = candidate.strip().strip("`'\",")
                if not cleaned or cleaned in {"&&", "\\"}:
                    continue
                tokens.append(cleaned)
        return _sorted_unique(tokens)

    def _record_match(
        self,
        matches: dict[str, _EntityMatch],
        entity: RegistryEntity,
        *,
        match_method: str,
        matched_alias: str | None = None,
        matched_versions: Iterable[str] = (),
    ) -> None:
        current = matches.get(entity.entity_id)
        if current is None:
            current = _EntityMatch(
                entity=entity,
                match_methods=set(),
                matched_aliases=set(),
                matched_versions=set(),
            )
            matches[entity.entity_id] = current

        current.match_methods.add(str(match_method))
        if matched_alias:
            current.matched_aliases.add(str(matched_alias))
        for version in matched_versions:
            current.matched_versions.add(str(version))

    def _document_source(self, document: Any) -> str:
        metadata = dict(getattr(document, "metadata", {}) or {})
        source = str(metadata.get("source") or getattr(document, "id", "") or "").strip()
        return source

    @staticmethod
    def _sorted_matches(matches: Iterable[_EntityMatch]) -> list[_EntityMatch]:
        return sorted(
            matches,
            key=lambda match: (
                match.entity.entity_type,
                match.entity.canonical_name.lower(),
                match.entity.entity_id,
            ),
        )


def _compile_boundary_pattern(value: str) -> re.Pattern[str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    return re.compile(rf"(?<![{_BOUNDARY_CHARS}]){re.escape(text)}(?![{_BOUNDARY_CHARS}])", flags=re.IGNORECASE)


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value or "").strip()})


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def resolve_authority_registry_artifact_path(cfg: Any) -> str:
    ingestion_cfg = _as_mapping(getattr(cfg, "ingestion", None))
    enrichment_cfg = _as_mapping(ingestion_cfg.get("metadata_enrichment"))
    configured = enrichment_cfg.get("authority_registry_path") or DEFAULT_REGISTRY_ARTIFACT_PATH
    return str(configured)


def enrich_documents_with_authority_metadata(
    documents: Sequence[DocumentT],
    *,
    registry_artifact_path: str | Path = DEFAULT_REGISTRY_ARTIFACT_PATH,
    source_name: str | None = None,
) -> list[DocumentT]:
    enricher = AuthorityMetadataEnricher.from_registry_artifact(
        registry_artifact_path,
        source_name=source_name,
    )
    return enricher.enrich_documents(documents)


def localize_doc_chunk_scope_family_metadata(
    chunks: Sequence[DocumentT],
    *,
    registry_artifact_path: str | Path = DEFAULT_REGISTRY_ARTIFACT_PATH,
) -> list[DocumentT]:
    registry_index = AuthorityRegistryIndex.load(registry_artifact_path)
    scope_family_resolver = ScopeFamilyResolver(registry_index.entities)

    localized: list[DocumentT] = []
    for chunk in chunks:
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        text = str(getattr(chunk, "text", "") or "")
        families = set(scope_family_resolver.families_for_text(text))
        for module_token in AuthorityMetadataEnricher._extract_module_tokens(text):
            families.update(scope_family_resolver.families_for_text(module_token))
        metadata["scope_family_names"] = _sorted_unique(families)
        localized.append(replace(chunk, metadata=metadata))

    return localized


__all__ = [
    "AUTHORITY_TIER_BY_SOURCE_AUTHORITY",
    "AuthorityMetadataEnricher",
    "AuthorityRegistryIndex",
    "DEFAULT_REGISTRY_ARTIFACT_PATH",
    "ENRICHMENT_VERSION",
    "SOURCE_AUTHORITY_EXTERNAL_OFFICIAL",
    "SOURCE_AUTHORITY_LOCAL_OFFICIAL",
    "SOURCE_AUTHORITY_TICKET_MEMORY",
    "SOURCE_AUTHORITY_UNKNOWN",
    "enrich_documents_with_authority_metadata",
    "localize_doc_chunk_scope_family_metadata",
    "resolve_authority_registry_artifact_path",
]
