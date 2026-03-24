"""Deterministic scope-family normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from polaris_rag.authority import RegistryEntity


@dataclass(frozen=True)
class _ScopeFamilyRule:
    family: str
    patterns: tuple[re.Pattern[str], ...]


def _compile_patterns(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns)


_SCOPE_FAMILY_RULES: tuple[_ScopeFamilyRule, ...] = (
    _ScopeFamilyRule(
        family="cclake",
        patterns=_compile_patterns(
            (
                r"\bcclake(?:-(?:himem|long))?\b",
                r"\bcascade\s+lake\b",
                r"/cclake/",
                r"\blogin-cascadelake\b",
            )
        ),
    ),
    _ScopeFamilyRule(
        family="icelake",
        patterns=_compile_patterns(
            (
                r"\bicelake(?:-(?:himem|long))?\b",
                r"\bice\s+lake\b",
                r"/icelake/",
                r"\bdefault-icl\b",
                r"\blogin-icelake\b",
            )
        ),
    ),
    _ScopeFamilyRule(
        family="sapphire",
        patterns=_compile_patterns(
            (
                r"\bsapphire(?:-(?:hbm|long))?\b",
                r"\bsapphire\s+rapid(?:s)?\b",
                r"/sapphire/",
            )
        ),
    ),
    _ScopeFamilyRule(
        family="ampere",
        patterns=_compile_patterns(
            (
                r"\bampere(?:-long)?\b",
                r"\ba100\b",
                r"/ampere/",
                r"\bwilkes3\b",
            )
        ),
    ),
    _ScopeFamilyRule(
        family="pvc",
        patterns=_compile_patterns(
            (
                r"\bpvc\b",
                r"\bdawn\b",
            )
        ),
    ),
    _ScopeFamilyRule(
        family="wbic",
        patterns=_compile_patterns(
            (
                r"\bwbic\b",
            )
        ),
    ),
)


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value or "").strip()})


def _families_for_text(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []

    families: list[str] = []
    for rule in _SCOPE_FAMILY_RULES:
        if any(pattern.search(value) for pattern in rule.patterns):
            families.append(rule.family)
    return _sorted_unique(families)


class ScopeFamilyResolver:
    """Resolve normalized scope-family tokens from registry-backed entities and text."""

    def __init__(self, entities: Sequence[RegistryEntity]) -> None:
        families_by_entity_id: dict[str, list[str]] = {}
        families_by_alias: dict[str, set[str]] = {}

        for entity in entities:
            families = _families_for_text(" ".join([entity.canonical_name, *entity.aliases]))
            if families:
                families_by_entity_id[entity.entity_id] = list(families)

            for value in [entity.canonical_name, *entity.aliases]:
                key = str(value or "").strip().lower()
                if not key:
                    continue
                alias_families = _families_for_text(value)
                if not alias_families:
                    continue
                families_by_alias.setdefault(key, set()).update(alias_families)

        self._families_by_entity_id = {
            entity_id: _sorted_unique(families)
            for entity_id, families in families_by_entity_id.items()
        }
        self._families_by_alias = {
            alias: _sorted_unique(families)
            for alias, families in families_by_alias.items()
        }

    def families_for_entity(self, entity: RegistryEntity) -> list[str]:
        families = self._families_by_entity_id.get(entity.entity_id)
        if families:
            return list(families)
        return _families_for_text(" ".join([entity.canonical_name, *entity.aliases]))

    def families_for_entities(self, entities: Iterable[RegistryEntity]) -> list[str]:
        values: list[str] = []
        for entity in entities:
            values.extend(self.families_for_entity(entity))
        return _sorted_unique(values)

    def families_for_alias(self, alias: str) -> list[str]:
        key = str(alias or "").strip().lower()
        if not key:
            return []
        families = self._families_by_alias.get(key)
        if families:
            return list(families)
        return _families_for_text(alias)

    def families_for_text(self, text: str) -> list[str]:
        return _families_for_text(text)


__all__ = ["ScopeFamilyResolver"]
