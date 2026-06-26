"""Deterministic scope-family normalization helpers.

This module combines public functions and classes used by the surrounding Polaris
subsystem.

Classes
-------
ScopeFamilyResolver
    Resolve normalized scope-family tokens from registry-backed entities and text.

Functions
---------
specialized_families_for_text
    Specialized Families For Text.
specialized_families_for_values
    Specialized Families For Values.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from polaris_rag.authority import RegistryEntity


@dataclass(frozen=True)
class _ScopeFamilyRule:
    family: str
    patterns: tuple[re.Pattern[str], ...]
    specialized_patterns: tuple[re.Pattern[str], ...] = ()


def _compile_patterns(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """Compile patterns.
    
    Parameters
    ----------
    patterns : Sequence[str]
        Value for patterns.
    
    Returns
    -------
    tuple[re.Pattern[str], ...]
        Collected results from the operation.
    """
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
        specialized_patterns=_compile_patterns(
            (
                r"\bcclake-(?:himem|long)\b",
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
        specialized_patterns=_compile_patterns(
            (
                r"\bicelake-(?:himem|long)\b",
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
        specialized_patterns=_compile_patterns(
            (
                r"\bsapphire-(?:hbm|long)\b",
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
                r"\bdefault-amp\b",
                r"\bwilkes3\b",
            )
        ),
        specialized_patterns=_compile_patterns(
            (
                r"\bampere-long\b",
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
    """Sorted Unique.
    
    Parameters
    ----------
    values : Iterable[str]
        Value for values.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
    return sorted({str(value).strip() for value in values if str(value or "").strip()})


def _families_for_text(text: str) -> list[str]:
    """Families For Text.
    
    Parameters
    ----------
    text : str
        Text value to inspect, tokenize, or encode.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
    value = str(text or "").strip()
    if not value:
        return []

    families: list[str] = []
    for rule in _SCOPE_FAMILY_RULES:
        if any(pattern.search(value) for pattern in rule.patterns):
            families.append(rule.family)
    return _sorted_unique(families)


def _specialized_families_for_text(text: str) -> list[str]:
    """Specialized Families For Text.
    
    Parameters
    ----------
    text : str
        Text value to inspect, tokenize, or encode.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
    value = str(text or "").strip()
    if not value:
        return []

    families: list[str] = []
    for rule in _SCOPE_FAMILY_RULES:
        if any(pattern.search(value) for pattern in rule.specialized_patterns):
            families.append(rule.family)
    return _sorted_unique(families)


def specialized_families_for_text(text: str) -> list[str]:
    """Specialized Families For Text.
    
    Parameters
    ----------
    text : str
        Text value to inspect, tokenize, or encode.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
    return _specialized_families_for_text(text)


def specialized_families_for_values(values: Iterable[str]) -> list[str]:
    """Specialized Families For Values.
    
    Parameters
    ----------
    values : Iterable[str]
        Value for values.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
    families: list[str] = []
    for value in values:
        families.extend(_specialized_families_for_text(str(value or "")))
    return _sorted_unique(families)


class ScopeFamilyResolver:
    """Resolve normalized scope-family tokens from registry-backed entities and text.
    
    Parameters
    ----------
    entities : Sequence[RegistryEntity]
        Value for entities.
    
    Methods
    -------
    families_for_entity
        Families For Entity.
    families_for_entities
        Families For Entities.
    families_for_alias
        Families For Alias.
    families_for_text
        Families For Text.
    specialized_families_for_text
        Specialized Families For Text.
    specialized_families_for_values
        Specialized Families For Values.
    """

    def __init__(self, entities: Sequence[RegistryEntity]) -> None:
        """Initialize the instance.
        
        Parameters
        ----------
        entities : Sequence[RegistryEntity]
            Value for entities.
        """
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
        """Families For Entity.
        
        Parameters
        ----------
        entity : RegistryEntity
            Value for entity.
        
        Returns
        -------
        list[str]
            Collected results from the operation.
        """
        families = self._families_by_entity_id.get(entity.entity_id)
        if families:
            return list(families)
        return _families_for_text(" ".join([entity.canonical_name, *entity.aliases]))

    def families_for_entities(self, entities: Iterable[RegistryEntity]) -> list[str]:
        """Families For Entities.
        
        Parameters
        ----------
        entities : Iterable[RegistryEntity]
            Value for entities.
        
        Returns
        -------
        list[str]
            Collected results from the operation.
        """
        values: list[str] = []
        for entity in entities:
            values.extend(self.families_for_entity(entity))
        return _sorted_unique(values)

    def families_for_alias(self, alias: str) -> list[str]:
        """Families For Alias.
        
        Parameters
        ----------
        alias : str
            Value for alias.
        
        Returns
        -------
        list[str]
            Collected results from the operation.
        """
        key = str(alias or "").strip().lower()
        if not key:
            return []
        families = self._families_by_alias.get(key)
        if families:
            return list(families)
        return _families_for_text(alias)

    def families_for_text(self, text: str) -> list[str]:
        """Families For Text.
        
        Parameters
        ----------
        text : str
            Text value to inspect, tokenize, or encode.
        
        Returns
        -------
        list[str]
            Collected results from the operation.
        """
        return _families_for_text(text)

    def specialized_families_for_text(self, text: str) -> list[str]:
        """Specialized Families For Text.
        
        Parameters
        ----------
        text : str
            Text value to inspect, tokenize, or encode.
        
        Returns
        -------
        list[str]
            Collected results from the operation.
        """
        return specialized_families_for_text(text)

    def specialized_families_for_values(self, values: Iterable[str]) -> list[str]:
        """Specialized Families For Values.
        
        Parameters
        ----------
        values : Iterable[str]
            Value for values.
        
        Returns
        -------
        list[str]
            Collected results from the operation.
        """
        return specialized_families_for_values(values)


__all__ = ["ScopeFamilyResolver", "specialized_families_for_text", "specialized_families_for_values"]
