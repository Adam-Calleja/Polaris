"""Deterministic sparse-query expansion for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from polaris_rag.authority import RegistryEntity
from polaris_rag.retrieval.metadata_enricher import AuthorityRegistryIndex
from polaris_rag.retrieval.query_constraints import QueryConstraints


@dataclass(frozen=True)
class SparseQueryExpansion:
    """Expanded sparse-query payload."""

    query_text: str
    expansion_terms: list[str]


class DeterministicSparseQueryExpander:
    """Expand sparse queries with authority aliases and exact version tokens."""

    def __init__(self, registry_index: AuthorityRegistryIndex) -> None:
        self.registry_index = registry_index
        self._canonical_aliases = self._build_canonical_aliases(registry_index.entities)
        payload = json.dumps(self._canonical_aliases, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self._fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_registry_artifact(cls, path: str | Path) -> "DeterministicSparseQueryExpander":
        return cls(AuthorityRegistryIndex.load(path))

    @staticmethod
    def _build_canonical_aliases(
        entities: tuple[RegistryEntity, ...],
    ) -> dict[str, list[str]]:
        values: dict[str, list[str]] = {}
        for entity in entities:
            if entity.entity_type not in {"software", "module", "toolchain"}:
                continue
            canonical = str(entity.canonical_name or "").strip()
            if not canonical:
                continue
            key = canonical.lower()
            aliases = values.setdefault(key, [])
            seen = {item.lower() for item in aliases}
            for alias in [canonical, *entity.aliases]:
                text = str(alias or "").strip()
                lower = text.lower()
                if not text or lower in seen:
                    continue
                aliases.append(text)
                seen.add(lower)
        return values

    def expand(
        self,
        query_text: str,
        query_constraints: QueryConstraints | dict[str, Any] | None,
    ) -> SparseQueryExpansion:
        normalized = QueryConstraints.from_value(query_constraints)
        if normalized is None:
            return SparseQueryExpansion(query_text=str(query_text or ""), expansion_terms=[])

        expansion_terms: list[str] = []
        seen: set[str] = set()

        def _append(value: str) -> None:
            text = str(value or "").strip()
            key = text.lower()
            if not text or key in seen:
                return
            seen.add(key)
            expansion_terms.append(text)

        names = [
            *normalized.software_names,
            *normalized.module_names,
            *normalized.toolchain_names,
        ]
        for name in names:
            aliases = self._canonical_aliases.get(str(name).strip().lower())
            if aliases:
                for alias in aliases:
                    _append(alias)
            else:
                _append(str(name))

        for version in [*normalized.software_versions, *normalized.toolchain_versions]:
            _append(str(version))

        expanded_query = str(query_text or "").strip()
        if expansion_terms:
            expanded_query = " ".join([expanded_query, *expansion_terms]).strip()
        return SparseQueryExpansion(
            query_text=expanded_query,
            expansion_terms=expansion_terms,
        )

    def profile(self) -> dict[str, Any]:
        """Return a stable profile for retrieval fingerprinting."""
        return {
            "type": "deterministic_alias_version",
            "entity_types": ["software", "module", "toolchain"],
            "alias_group_count": len(self._canonical_aliases),
            "alias_registry_fingerprint": self._fingerprint,
        }


__all__ = [
    "DeterministicSparseQueryExpander",
    "SparseQueryExpansion",
]
