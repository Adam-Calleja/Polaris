"""Deterministic runtime query constraint extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from polaris_rag.authority import RegistryEntity
from polaris_rag.retrieval.metadata_enricher import AuthorityRegistryIndex
from polaris_rag.retrieval.scope_family import ScopeFamilyResolver

QUERY_TYPE_LOCAL_OPERATIONAL = "local_operational"
QUERY_TYPE_SOFTWARE_VERSION = "software_version"
QUERY_TYPE_GENERAL_HOW_TO = "general_how_to"
QUERY_TYPE_VALUES: tuple[str, ...] = (
    QUERY_TYPE_LOCAL_OPERATIONAL,
    QUERY_TYPE_SOFTWARE_VERSION,
    QUERY_TYPE_GENERAL_HOW_TO,
)

_BOUNDARY_CHARS = r"A-Za-z0-9"
_MODULE_LOAD_PATTERN = re.compile(
    r"(?:^|&&|;)\s*(?:\$+\s*)?module\s+load\s+([^\n#;]+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_GENERIC_VERSION_PATTERN = re.compile(
    r"\b(?:\d+\.\d+(?:\.\d+){0,2}(?:[-+_][A-Za-z0-9][A-Za-z0-9.+_-]*)?|r\d{4}[a-z])\b",
    flags=re.IGNORECASE,
)
_PARTITION_ASSIGNMENT_PATTERN = re.compile(
    r"(?:--partition(?:=|\s+)|-p\s+)([A-Za-z0-9][A-Za-z0-9_-]*)\b",
    flags=re.IGNORECASE,
)
_PARTITION_CONTEXT_PATTERN = re.compile(
    r"\b(?:partition|queue)\s*(?:=|:)?\s*([A-Za-z0-9][A-Za-z0-9_-]*)\b",
    flags=re.IGNORECASE,
)
_ON_PARTITION_PATTERN = re.compile(
    r"\bon\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9_-]*)\s+partition\b",
    flags=re.IGNORECASE,
)
_OPERATIONAL_ON_SCOPE_PATTERN = re.compile(
    r"\bon\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9_-]*)\b",
    flags=re.IGNORECASE,
)
_MODULE_CONTEXT_PATTERN = re.compile(r"\b(?:module|modules|spack)\b", flags=re.IGNORECASE)
_PARTITION_CONTEXT_HINT_PATTERN = re.compile(
    r"\b(?:partition|queue|sbatch|srun|salloc)\b|--partition|(?<!\w)-p\b",
    flags=re.IGNORECASE,
)
_RUN_CONTEXT_HINT_PATTERN = re.compile(
    r"\b(?:run|running|submit|submitting|launch|launching|execute|executing|compile|compiling|build|building|install|installing|use|using|develop|development|job|jobs)\b",
    flags=re.IGNORECASE,
)
_SYSTEM_CONTEXT_HINT_PATTERN = re.compile(
    r"\b(?:node|nodes|system|cluster)\b",
    flags=re.IGNORECASE,
)
_LOCAL_OPERATIONAL_QUERY_PATTERN = re.compile(
    r"\b(?:slurm|sbatch|srun|salloc|ondemand|ssh|login|quota|mybalance|allocation|account|mfa|job|queue|partition)\b",
    flags=re.IGNORECASE,
)
_VERSION_SENSITIVE_HINT_PATTERN = re.compile(
    r"\b(?:version|versions|release|releases|latest|newest|older|oldest|upgrade|downgrade|update(?:d)?|compatible|compatibility)\b",
    flags=re.IGNORECASE,
)

@dataclass(frozen=True)
class QueryConstraints:
    """Structured query-side constraints extracted from a runtime query."""

    query_type: str | None = None
    system_names: list[str] = field(default_factory=list)
    partition_names: list[str] = field(default_factory=list)
    service_names: list[str] = field(default_factory=list)
    scope_family_names: list[str] = field(default_factory=list)
    software_names: list[str] = field(default_factory=list)
    software_versions: list[str] = field(default_factory=list)
    module_names: list[str] = field(default_factory=list)
    toolchain_names: list[str] = field(default_factory=list)
    toolchain_versions: list[str] = field(default_factory=list)
    scope_required: bool | None = None
    version_sensitive_guess: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "query_type": self.query_type,
            "system_names": list(self.system_names),
            "partition_names": list(self.partition_names),
            "service_names": list(self.service_names),
            "scope_family_names": list(self.scope_family_names),
            "software_names": list(self.software_names),
            "software_versions": list(self.software_versions),
            "module_names": list(self.module_names),
            "toolchain_names": list(self.toolchain_names),
            "toolchain_versions": list(self.toolchain_versions),
            "scope_required": self.scope_required,
            "version_sensitive_guess": self.version_sensitive_guess,
        }

    @classmethod
    def from_value(cls, value: "QueryConstraints | Mapping[str, Any] | None") -> "QueryConstraints | None":
        """Normalize supported query-constraint payloads."""

        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return None

        query_type = _normalize_optional_text(value.get("query_type"))
        if query_type not in QUERY_TYPE_VALUES:
            query_type = None

        return cls(
            query_type=query_type,
            system_names=_normalize_text_list(value.get("system_names")),
            partition_names=_normalize_text_list(value.get("partition_names")),
            service_names=_normalize_text_list(value.get("service_names")),
            scope_family_names=_normalize_text_list(value.get("scope_family_names")),
            software_names=_normalize_text_list(value.get("software_names")),
            software_versions=_normalize_text_list(value.get("software_versions")),
            module_names=_normalize_text_list(value.get("module_names")),
            toolchain_names=_normalize_text_list(value.get("toolchain_names")),
            toolchain_versions=_normalize_text_list(value.get("toolchain_versions")),
            scope_required=_normalize_optional_bool(value.get("scope_required")),
            version_sensitive_guess=_normalize_optional_bool(value.get("version_sensitive_guess")),
        )


def serialize_query_constraints(
    value: QueryConstraints | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize query constraints into a stable serializable mapping."""

    normalized = QueryConstraints.from_value(value)
    if normalized is None:
        return None
    return normalized.to_dict()


@dataclass(frozen=True)
class _AliasGroup:
    alias: str
    pattern: re.Pattern[str]
    entities: tuple[RegistryEntity, ...]


@dataclass
class _EntityMatch:
    entity: RegistryEntity
    match_methods: set[str]
    matched_aliases: set[str]
    matched_versions: set[str]


class AuthorityQueryConstraintParser:
    """Extract deterministic query constraints from runtime text."""

    def __init__(self, registry_index: AuthorityRegistryIndex) -> None:
        self.registry_index = registry_index
        self.entities_by_alias = self._build_entities_by_alias(registry_index.entities)
        self.alias_groups = self._build_alias_groups(self.entities_by_alias)
        self.scope_family_resolver = ScopeFamilyResolver(registry_index.entities)

    @classmethod
    def from_registry_artifact(cls, path: str | Path) -> "AuthorityQueryConstraintParser":
        return cls(AuthorityRegistryIndex.load(path))

    def parse(self, query: str) -> QueryConstraints:
        text = str(query or "")
        if not text.strip():
            return QueryConstraints()

        observed_versions = self._extract_version_mentions(text)
        matches: dict[str, _EntityMatch] = {}
        scope_family_names: set[str] = set()

        for partition_token in self._extract_partition_tokens(text):
            for entity in self.entities_by_alias.get(partition_token, ()):
                if entity.entity_type != "partition":
                    continue
                self._record_match(
                    matches,
                    entity,
                    match_method="partition_syntax",
                    matched_alias=partition_token,
                    matched_versions=self._matched_versions(entity, observed_versions),
                )
                scope_family_names.update(self.scope_family_resolver.families_for_entity(entity))

        for module_token in self._extract_module_tokens(text):
            scope_family_names.update(self.scope_family_resolver.families_for_text(module_token))
            for entity in self.entities_by_alias.get(module_token.lower(), ()):
                self._record_match(
                    matches,
                    entity,
                    match_method="module_load_parse",
                    matched_alias=module_token,
                    matched_versions=self._matched_versions(entity, observed_versions),
                )
                scope_family_names.update(self.scope_family_resolver.families_for_entity(entity))

        for alias_group in self.alias_groups:
            if not alias_group.pattern.search(text):
                continue

            resolved_entities = self._resolve_alias_group(alias_group, text)
            for entity in resolved_entities:
                self._record_match(
                    matches,
                    entity,
                    match_method="alias_scan",
                    matched_alias=alias_group.alias,
                    matched_versions=self._matched_versions(entity, observed_versions),
                )
                scope_family_names.update(self.scope_family_resolver.families_for_entity(entity))
            if resolved_entities:
                continue

            scope_family_names.update(self._family_hint_for_alias_group(alias_group, text))

        return self._build_constraints(
            text,
            matches.values(),
            explicit_version_mention=bool(observed_versions),
            scope_family_names=scope_family_names,
        )

    @staticmethod
    def _build_entities_by_alias(
        entities: Sequence[RegistryEntity],
    ) -> dict[str, tuple[RegistryEntity, ...]]:
        grouped: dict[str, dict[str, RegistryEntity]] = {}

        for entity in entities:
            for alias in _entity_aliases(entity):
                if not _should_index_alias(alias):
                    continue
                alias_key = alias.lower()
                by_entity = grouped.setdefault(alias_key, {})
                by_entity[entity.entity_id] = entity

        return {
            alias_key: tuple(sorted(value.values(), key=_entity_sort_key))
            for alias_key, value in grouped.items()
        }

    @staticmethod
    def _build_alias_groups(
        entities_by_alias: Mapping[str, Sequence[RegistryEntity]],
    ) -> tuple[_AliasGroup, ...]:
        groups: list[_AliasGroup] = []
        for alias, entities in entities_by_alias.items():
            pattern = _compile_boundary_pattern(alias)
            if pattern is None:
                continue
            groups.append(
                _AliasGroup(
                    alias=alias,
                    pattern=pattern,
                    entities=tuple(sorted(entities, key=_entity_sort_key)),
                )
            )
        groups.sort(key=lambda group: (-len(group.alias), group.alias))
        return tuple(groups)

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

    @staticmethod
    def _extract_partition_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        for pattern in (_PARTITION_ASSIGNMENT_PATTERN, _PARTITION_CONTEXT_PATTERN, _ON_PARTITION_PATTERN):
            for match in pattern.finditer(text or ""):
                token = str(match.group(1) or "").strip().lower()
                if token:
                    tokens.append(token)
        return _sorted_unique(tokens)

    def _resolve_alias_group(
        self,
        alias_group: _AliasGroup,
        text: str,
    ) -> tuple[RegistryEntity, ...]:
        if len(alias_group.entities) == 1:
            return alias_group.entities

        entity_types = {entity.entity_type for entity in alias_group.entities}
        match = alias_group.pattern.search(text or "")
        if match is None:
            return ()

        context = self._local_context(text, match.start(), match.end())

        if entity_types <= {"partition", "system"}:
            if _PARTITION_CONTEXT_HINT_PATTERN.search(context):
                return tuple(entity for entity in alias_group.entities if entity.entity_type == "partition")
            if _SYSTEM_CONTEXT_HINT_PATTERN.search(context):
                return tuple(entity for entity in alias_group.entities if entity.entity_type == "system")
            return ()

        if entity_types <= {"module", "toolchain"}:
            return ()

        if "software" in entity_types and entity_types <= {"software", "module", "toolchain"}:
            if _MODULE_CONTEXT_PATTERN.search(context):
                return tuple(entity for entity in alias_group.entities if entity.entity_type in {"module", "toolchain"})
            return tuple(entity for entity in alias_group.entities if entity.entity_type == "software")

        return ()

    def _family_hint_for_alias_group(
        self,
        alias_group: _AliasGroup,
        text: str,
    ) -> tuple[str, ...]:
        entity_types = {entity.entity_type for entity in alias_group.entities}
        if not entity_types <= {"partition", "system"}:
            return ()

        match = alias_group.pattern.search(text or "")
        if match is None:
            return ()

        context = self._local_context(text, match.start(), match.end(), radius=32)
        if _PARTITION_CONTEXT_HINT_PATTERN.search(context) or _SYSTEM_CONTEXT_HINT_PATTERN.search(context):
            return ()
        if not _RUN_CONTEXT_HINT_PATTERN.search(context):
            return ()

        return tuple(self.scope_family_resolver.families_for_entities(alias_group.entities))

    @staticmethod
    def _local_context(text: str, start: int, end: int, *, radius: int = 28) -> str:
        left = max(0, int(start) - radius)
        right = min(len(text), int(end) + radius)
        return text[left:right]

    @staticmethod
    def _matched_versions(entity: RegistryEntity, observed_versions: Sequence[str]) -> set[str]:
        observed = {version.lower(): version for version in observed_versions}
        matched: set[str] = set()
        for version in entity.known_versions:
            if version.lower() in observed:
                matched.add(version)
        return matched

    @staticmethod
    def _record_match(
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

    def _build_constraints(
        self,
        text: str,
        matches: Iterable[_EntityMatch],
        *,
        explicit_version_mention: bool,
        scope_family_names: Iterable[str],
    ) -> QueryConstraints:
        sorted_matches = sorted(matches, key=lambda item: _entity_sort_key(item.entity))
        system_names = _names_for_type(sorted_matches, "system")
        partition_names = _names_for_type(sorted_matches, "partition")
        service_names = _names_for_type(sorted_matches, "service")
        scope_family_names_list = _sorted_unique(scope_family_names)
        software_names = _names_for_type(sorted_matches, "software")
        module_names = _names_for_type(sorted_matches, "module")
        toolchain_names = _names_for_type(sorted_matches, "toolchain")
        software_versions = _versions_for_type(sorted_matches, "software")
        toolchain_versions = _versions_for_type(sorted_matches, "toolchain")

        scope_required = True if system_names or partition_names or service_names or scope_family_names_list else None

        has_version_targets = bool(software_names or module_names or toolchain_names)
        has_explicit_version_match = bool(software_versions or toolchain_versions)
        has_version_hint = has_version_targets and bool(_VERSION_SENSITIVE_HINT_PATTERN.search(text or ""))
        version_sensitive_guess = True if has_explicit_version_match or has_version_hint else None
        if version_sensitive_guess is None and has_version_targets and explicit_version_mention:
            version_sensitive_guess = True

        query_type = None
        if version_sensitive_guess and has_version_targets:
            query_type = QUERY_TYPE_SOFTWARE_VERSION
        elif (
            system_names
            or partition_names
            or service_names
            or scope_family_names_list
            or _LOCAL_OPERATIONAL_QUERY_PATTERN.search(text or "")
        ):
            query_type = QUERY_TYPE_LOCAL_OPERATIONAL
        elif software_names or module_names or toolchain_names:
            query_type = QUERY_TYPE_GENERAL_HOW_TO

        return QueryConstraints(
            query_type=query_type,
            system_names=system_names,
            partition_names=partition_names,
            service_names=service_names,
            scope_family_names=scope_family_names_list,
            software_names=software_names,
            software_versions=software_versions,
            module_names=module_names,
            toolchain_names=toolchain_names,
            toolchain_versions=toolchain_versions,
            scope_required=scope_required,
            version_sensitive_guess=version_sensitive_guess,
        )


def _names_for_type(matches: Sequence[_EntityMatch], entity_type: str) -> list[str]:
    return _sorted_unique(
        match.entity.canonical_name
        for match in matches
        if match.entity.entity_type == entity_type
    )


def _versions_for_type(matches: Sequence[_EntityMatch], entity_type: str) -> list[str]:
    values: list[str] = []
    for match in matches:
        if match.entity.entity_type != entity_type:
            continue
        values.extend(match.matched_versions)
    return _sorted_unique(values)


def _entity_aliases(entity: RegistryEntity) -> list[str]:
    raw_values = [entity.canonical_name, *entity.aliases]
    dedup: dict[str, str] = {}
    for raw_value in raw_values:
        text = str(raw_value or "").strip()
        if not text:
            continue
        dedup.setdefault(text.lower(), text)
    return sorted(dedup.values(), key=lambda value: (len(value), value.lower()))


def _compile_boundary_pattern(value: str) -> re.Pattern[str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    return re.compile(
        rf"(?<![{_BOUNDARY_CHARS}]){re.escape(text)}(?![{_BOUNDARY_CHARS}])",
        flags=re.IGNORECASE,
    )


def _entity_sort_key(entity: RegistryEntity) -> tuple[str, str, str]:
    return (entity.entity_type, entity.canonical_name.lower(), entity.entity_id)


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _sorted_unique(value)


def _should_index_alias(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False

    alnum_count = sum(1 for char in text if char.isalnum())
    if alnum_count < 2 and "/" not in text and "-" not in text:
        return False

    if len(text) < 3 and not any(char.isdigit() for char in text) and "/" not in text and "-" not in text:
        return False

    return True


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value or "").strip()})


__all__ = [
    "AuthorityQueryConstraintParser",
    "QUERY_TYPE_GENERAL_HOW_TO",
    "QUERY_TYPE_LOCAL_OPERATIONAL",
    "QUERY_TYPE_SOFTWARE_VERSION",
    "QUERY_TYPE_VALUES",
    "QueryConstraints",
    "serialize_query_constraints",
]
