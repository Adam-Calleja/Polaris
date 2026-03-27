"""Deterministic external source-register loading and crawl helpers.

This module combines public functions and classes used by the surrounding Polaris
subsystem.

Classes
-------
ExternalSourceCrawlConfig
    Configuration for external Source Crawl.
ExternalSourceDefinition
    Structured definition for external Source.
ExternalSourceRegister
    Collection of registered external Source.

Functions
---------
load_external_source_register
    Load and validate an external source register.
discover_external_source_urls
    Crawl one registered external source deterministically.
discover_all_external_source_urls
    Return discovered URLs keyed by registered source id.
source_url_allowed
    Return whether a URL is allowed by a registered source definition.
attach_source_register_metadata
    Attach deterministic source-register metadata to loaded documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, TypeVar
from urllib.parse import urlsplit, urlunsplit

import yaml

from polaris_rag.authority.registry_builder import (
    REGISTRY_ENTITY_TYPES,
    SOURCE_SCOPE_EXTERNAL_OFFICIAL,
)


EXTERNAL_SOURCE_REGISTER_VERSION = "external_source_register_v1"
_COMMUNITY_PATH_HINTS = (
    "/blog",
    "/blogs",
    "/community",
    "/communities",
    "/discuss",
    "/discussion",
    "/forum",
    "/forums",
    "/issues",
    "/pull",
    "/pulls",
    "/news",
)

DocumentT = TypeVar("DocumentT")


@dataclass(frozen=True)
class ExternalSourceCrawlConfig:
    """Configuration for external Source Crawl.
    
    Attributes
    ----------
    max_depth : int
        Value for max Depth.
    max_pages : int
        Value for max Pages.
    """
    max_depth: int
    max_pages: int


@dataclass(frozen=True)
class ExternalSourceDefinition:
    """Structured definition for external Source.
    
    Attributes
    ----------
    source_id : str
        Stable identifier for the registered source.
    canonical_name : str
        Value for canonical Name.
    entity_type : str
        Value for entity Type.
    homepage : str
        Value for homepage.
    allowed_domains : tuple[str, ...]
        Value for allowed Domains.
    include_url_prefixes : tuple[str, ...]
        Whether to include URL Prefixes.
    exclude_url_patterns : tuple[str, ...]
        Value for exclude URL Patterns.
    aliases : tuple[str, ...]
        Value for aliases.
    relevance_tags : tuple[str, ...]
        Value for relevance Tags.
    crawl : ExternalSourceCrawlConfig
        Value for crawl.
    """
    source_id: str
    canonical_name: str
    entity_type: str
    homepage: str
    allowed_domains: tuple[str, ...]
    include_url_prefixes: tuple[str, ...]
    exclude_url_patterns: tuple[str, ...]
    aliases: tuple[str, ...]
    relevance_tags: tuple[str, ...]
    crawl: ExternalSourceCrawlConfig


@dataclass(frozen=True)
class ExternalSourceRegister:
    """Collection of registered external Source.
    
    Attributes
    ----------
    version : str
        Value for version.
    sources : tuple[ExternalSourceDefinition, ...]
        Value for sources.
    """
    version: str
    sources: tuple[ExternalSourceDefinition, ...]


def load_external_source_register(path: str | Path) -> ExternalSourceRegister:
    """Load and validate an external source register.
    
    Parameters
    ----------
    path : str or Path
        Filesystem path used by the operation.
    
    Returns
    -------
    ExternalSourceRegister
        Loaded external Source Register.
    
    Raises
    ------
    TypeError
        If the provided value has an unexpected type.
    ValueError
        If the provided value is invalid for the operation.
    """

    resolved = Path(path).expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"External source register {resolved} must contain a mapping.")

    version = str(payload.get("version") or "").strip()
    if version != EXTERNAL_SOURCE_REGISTER_VERSION:
        raise ValueError(
            f"Unsupported external source register version {version!r}. "
            f"Expected {EXTERNAL_SOURCE_REGISTER_VERSION!r}."
        )

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("External source register must define a non-empty 'sources' list.")

    sources = tuple(_normalize_source_definition(item) for item in raw_sources)
    _validate_source_prefix_overlaps(sources)
    return ExternalSourceRegister(version=version, sources=tuple(sorted(sources, key=lambda item: item.source_id)))


def discover_external_source_urls(
    source: ExternalSourceDefinition,
    *,
    get_internal_links: Callable[[str], Sequence[str]],
) -> list[str]:
    """Crawl one registered external source deterministically.
    
    Parameters
    ----------
    source : ExternalSourceDefinition
        Source definition, source name, or source identifier to process.
    get_internal_links : Callable[[str], Sequence[str]]
        Callable used to discover internal links for a source homepage.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """

    queue: list[tuple[str, int]] = [(source.homepage, 0)]
    seen: set[str] = set()
    ordered: list[str] = []

    while queue and len(seen) < source.crawl.max_pages:
        current, depth = queue.pop(0)
        current_normalized = _canonicalize_url(current)
        if current_normalized in seen or not source_url_allowed(source, current_normalized):
            continue
        seen.add(current_normalized)
        ordered.append(current_normalized)

        if depth >= source.crawl.max_depth:
            continue

        discovered = get_internal_links(current_normalized) or []
        for raw_link in discovered:
            link = _canonicalize_url(raw_link)
            if link in seen or not source_url_allowed(source, link):
                continue
            if link not in [candidate for candidate, _ in queue]:
                if len(seen) + len(queue) >= source.crawl.max_pages:
                    break
                queue.append((link, depth + 1))

    if source.homepage not in ordered and source_url_allowed(source, source.homepage):
        ordered.insert(0, source.homepage)
    return ordered[: source.crawl.max_pages]


def discover_all_external_source_urls(
    register: ExternalSourceRegister,
    *,
    get_internal_links: Callable[[str], Sequence[str]],
) -> dict[str, list[str]]:
    """Return discovered URLs keyed by registered source id.
    
    Parameters
    ----------
    register : ExternalSourceRegister
        Value for register.
    get_internal_links : Callable[[str], Sequence[str]]
        Callable used to discover internal links for a source homepage.
    
    Returns
    -------
    dict[str, list[str]]
        Structured result of the operation.
    """

    return {
        source.source_id: discover_external_source_urls(source, get_internal_links=get_internal_links)
        for source in register.sources
    }


def source_url_allowed(source: ExternalSourceDefinition, url: str) -> bool:
    """Return whether a URL is allowed by a registered source definition.
    
    Parameters
    ----------
    source : ExternalSourceDefinition
        Source definition, source name, or source identifier to process.
    url : str
        URL used by the operation.
    
    Returns
    -------
    bool
        `True` if source URL Allowed; otherwise `False`.
    """

    normalized = _canonicalize_url(url)
    parts = urlsplit(normalized)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return False

    hostname = (parts.hostname or "").lower()
    if hostname not in set(source.allowed_domains):
        return False

    if not any(normalized.startswith(prefix) for prefix in source.include_url_prefixes):
        return False

    for pattern in source.exclude_url_patterns:
        if re.search(pattern, normalized):
            return False

    if _has_unstable_path_hint(normalized):
        return False
    return True


def attach_source_register_metadata(
    documents: Sequence[DocumentT],
    *,
    source: ExternalSourceDefinition,
) -> list[DocumentT]:
    """Attach deterministic source-register metadata to loaded documents.
    
    Parameters
    ----------
    documents : Sequence[DocumentT]
        Document objects to enrich, convert, or inspect.
    source : ExternalSourceDefinition
        Source definition, source name, or source identifier to process.
    
    Returns
    -------
    list[DocumentT]
        Collected results from the operation.
    """

    enriched: list[DocumentT] = []
    for document in documents:
        metadata = dict(getattr(document, "metadata", {}) or {})
        metadata.update(
            {
                "source_scope": SOURCE_SCOPE_EXTERNAL_OFFICIAL,
                "source_register_id": source.source_id,
                "source_register_canonical_name": source.canonical_name,
                "source_register_entity_type": source.entity_type,
                "source_register_aliases": list(source.aliases),
                "source_register_relevance_tags": list(source.relevance_tags),
            }
        )
        setattr(document, "metadata", metadata)
        enriched.append(document)
    return enriched


def _normalize_source_definition(value: Any) -> ExternalSourceDefinition:
    """Normalize source Definition.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    ExternalSourceDefinition
        Result of the operation.
    
    Raises
    ------
    TypeError
        If the provided value has an unexpected type.
    ValueError
        If the provided value is invalid for the operation.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"Each source definition must be a mapping, got {type(value)!r}.")

    source_id = _normalized_non_empty_text(value.get("source_id"), "source_id")
    canonical_name = _normalized_non_empty_text(value.get("canonical_name"), f"sources[{source_id}].canonical_name")
    entity_type = _normalized_non_empty_text(value.get("entity_type"), f"sources[{source_id}].entity_type").lower()
    if entity_type not in REGISTRY_ENTITY_TYPES:
        raise ValueError(
            f"sources[{source_id}].entity_type must be one of {sorted(REGISTRY_ENTITY_TYPES)}, got {entity_type!r}."
        )

    homepage = _canonicalize_url(
        _normalized_non_empty_text(value.get("homepage"), f"sources[{source_id}].homepage")
    )
    allowed_domains = _normalize_domain_list(value.get("allowed_domains"), source_id=source_id)
    homepage_domain = (urlsplit(homepage).hostname or "").lower()
    if homepage_domain not in allowed_domains:
        raise ValueError(
            f"sources[{source_id}].homepage host {homepage_domain!r} must be present in allowed_domains."
        )

    include_url_prefixes = _normalize_url_prefixes(value.get("include_url_prefixes"), source_id=source_id)
    if not any(homepage.startswith(prefix) for prefix in include_url_prefixes):
        raise ValueError(
            f"sources[{source_id}].homepage must fall under one of its include_url_prefixes."
        )

    exclude_url_patterns = _normalize_regex_list(value.get("exclude_url_patterns"), source_id=source_id)
    aliases = tuple(
        sorted(
            {
                canonical_name,
                *(
                    str(alias).strip()
                    for alias in value.get("aliases", [])
                    if str(alias or "").strip()
                ),
            },
            key=lambda item: (len(item), item.lower()),
        )
    )
    relevance_tags = tuple(
        sorted(
            {
                str(tag).strip()
                for tag in value.get("relevance_tags", [])
                if str(tag or "").strip()
            }
        )
    )
    crawl = _normalize_crawl_config(value.get("crawl"), source_id=source_id)

    if _has_unstable_path_hint(homepage) or any(_has_unstable_path_hint(prefix) for prefix in include_url_prefixes):
        raise ValueError(
            f"sources[{source_id}] points at an unstable/community path. Register only stable official docs."
        )

    return ExternalSourceDefinition(
        source_id=source_id,
        canonical_name=canonical_name,
        entity_type=entity_type,
        homepage=homepage,
        allowed_domains=allowed_domains,
        include_url_prefixes=include_url_prefixes,
        exclude_url_patterns=exclude_url_patterns,
        aliases=aliases,
        relevance_tags=relevance_tags,
        crawl=crawl,
    )


def _normalize_crawl_config(value: Any, *, source_id: str) -> ExternalSourceCrawlConfig:
    """Normalize crawl Config.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    source_id : str
        Stable identifier for the registered source.
    
    Returns
    -------
    ExternalSourceCrawlConfig
        Result of the operation.
    
    Raises
    ------
    TypeError
        If the provided value has an unexpected type.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"sources[{source_id}].crawl must be a mapping.")
    max_depth = _normalized_non_negative_int(value.get("max_depth"), f"sources[{source_id}].crawl.max_depth")
    max_pages = _normalized_positive_int(value.get("max_pages"), f"sources[{source_id}].crawl.max_pages")
    return ExternalSourceCrawlConfig(max_depth=max_depth, max_pages=max_pages)


def _normalize_domain_list(value: Any, *, source_id: str) -> tuple[str, ...]:
    """Normalize domain List.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    source_id : str
        Stable identifier for the registered source.
    
    Returns
    -------
    tuple[str, ...]
        Collected results from the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"sources[{source_id}].allowed_domains must be a non-empty list.")
    domains = []
    for item in value:
        domain = str(item or "").strip().lower()
        if not domain:
            continue
        if "/" in domain:
            raise ValueError(f"sources[{source_id}].allowed_domains entries must be bare hostnames.")
        domains.append(domain)
    normalized = tuple(sorted(set(domains)))
    if not normalized:
        raise ValueError(f"sources[{source_id}].allowed_domains must contain at least one hostname.")
    return normalized


def _normalize_url_prefixes(value: Any, *, source_id: str) -> tuple[str, ...]:
    """Normalize URL Prefixes.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    source_id : str
        Stable identifier for the registered source.
    
    Returns
    -------
    tuple[str, ...]
        Collected results from the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"sources[{source_id}].include_url_prefixes must be a non-empty list.")
    prefixes = tuple(sorted({_canonicalize_url(str(item or "").strip()) for item in value if str(item or "").strip()}))
    if not prefixes:
        raise ValueError(f"sources[{source_id}].include_url_prefixes must contain at least one URL prefix.")
    return prefixes


def _normalize_regex_list(value: Any, *, source_id: str) -> tuple[str, ...]:
    """Normalize regex List.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    source_id : str
        Stable identifier for the registered source.
    
    Returns
    -------
    tuple[str, ...]
        Collected results from the operation.
    
    Raises
    ------
    TypeError
        If the provided value has an unexpected type.
    """
    if value is None:
        return tuple()
    if not isinstance(value, list):
        raise TypeError(f"sources[{source_id}].exclude_url_patterns must be a list.")
    patterns: list[str] = []
    for item in value:
        pattern = str(item or "").strip()
        if not pattern:
            continue
        re.compile(pattern)
        patterns.append(pattern)
    return tuple(sorted(set(patterns)))


def _validate_source_prefix_overlaps(sources: Sequence[ExternalSourceDefinition]) -> None:
    """Validate source Prefix Overlaps.
    
    Parameters
    ----------
    sources : Sequence[ExternalSourceDefinition]
        Value for sources.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    for idx, left in enumerate(sources):
        left_prefixes = set(left.include_url_prefixes)
        for right in sources[idx + 1 :]:
            for left_prefix in left_prefixes:
                for right_prefix in right.include_url_prefixes:
                    if left_prefix.startswith(right_prefix) or right_prefix.startswith(left_prefix):
                        raise ValueError(
                            "External source register includes overlapping include_url_prefixes for "
                            f"{left.source_id!r} and {right.source_id!r}: {left_prefix!r} vs {right_prefix!r}."
                        )


def _canonicalize_url(url: str) -> str:
    """Canonicalize URL.
    
    Parameters
    ----------
    url : str
        URL used by the operation.
    
    Returns
    -------
    str
        Resulting string value.
    """
    parts = urlsplit(str(url or "").strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _has_unstable_path_hint(url: str) -> bool:
    """Return whether unstable Path Hint.
    
    Parameters
    ----------
    url : str
        URL used by the operation.
    
    Returns
    -------
    bool
        `True` if has Unstable Path Hint; otherwise `False`.
    """
    lower = _canonicalize_url(url).lower()
    return any(hint in lower for hint in _COMMUNITY_PATH_HINTS)


def _normalized_non_empty_text(value: Any, field_name: str) -> str:
    """Normalized Non Empty Text.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    field_name : str
        Value for field Name.
    
    Returns
    -------
    str
        Resulting string value.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _normalized_positive_int(value: Any, field_name: str) -> int:
    """Normalized Positive Int.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    field_name : str
        Value for field Name.
    
    Returns
    -------
    int
        Computed integer value.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a positive integer.") from None
    if resolved <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return resolved


def _normalized_non_negative_int(value: Any, field_name: str) -> int:
    """Normalized Non Negative Int.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    field_name : str
        Value for field Name.
    
    Returns
    -------
    int
        Computed integer value.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a non-negative integer.") from None
    if resolved < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return resolved


__all__ = [
    "EXTERNAL_SOURCE_REGISTER_VERSION",
    "ExternalSourceCrawlConfig",
    "ExternalSourceDefinition",
    "ExternalSourceRegister",
    "attach_source_register_metadata",
    "discover_all_external_source_urls",
    "discover_external_source_urls",
    "load_external_source_register",
    "source_url_allowed",
]
