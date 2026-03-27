"""Helpers for resolving ingestion conversion and chunking settings.

This module combines public functions and classes used by the surrounding Polaris
subsystem.

Classes
-------
ChunkingSettings
    Chunking Settings.
ConversionSettings
    Conversion Settings.

Functions
---------
resolve_chunking_settings
    Resolve chunking Settings.
resolve_conversion_settings
    Resolve conversion Settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

MARKDOWN_TOKEN_CHUNKING_STRATEGY = "markdown_token"
HTML_HIERARCHICAL_CHUNKING_STRATEGY = "html_hierarchical"
JIRA_TURNS_TOKEN_CHUNKING_STRATEGY = "jira_turns_token"

MARKITDOWN_CONVERSION_ENGINE = "markitdown"
NATIVE_JIRA_CONVERSION_ENGINE = "native_jira"

DEFAULT_CHUNK_SIZE_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 80


@dataclass(frozen=True)
class ChunkingSettings:
    """Chunking Settings.
    
    Attributes
    ----------
    strategy : str
        Value for strategy.
    chunk_size_tokens : int
        Value for chunk Size Tokens.
    overlap_tokens : int
        Number of tokens to overlap between adjacent chunks.
    """
    strategy: str
    chunk_size_tokens: int
    overlap_tokens: int


@dataclass(frozen=True)
class ConversionSettings:
    """Conversion Settings.
    
    Attributes
    ----------
    engine : str
        Value for engine.
    options : dict[str, Any]
        Value for options.
    """
    engine: str
    options: dict[str, Any] = field(default_factory=dict)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """As Mapping.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    Mapping[str, Any]
        Result of the operation.
    """
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _resolve_int(value: Any, default: int) -> int:
    """Resolve int.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    default : int
        Fallback value to use when normalization fails.
    
    Returns
    -------
    int
        Computed integer value.
    """
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved >= 0 else default


def _default_chunking_strategy(source: str) -> str:
    """Default Chunking Strategy.
    
    Parameters
    ----------
    source : str
        Source definition, source name, or source identifier to process.
    
    Returns
    -------
    str
        Resulting string value.
    """
    return MARKDOWN_TOKEN_CHUNKING_STRATEGY


def _default_conversion_engine(source: str) -> str:
    """Default Conversion Engine.
    
    Parameters
    ----------
    source : str
        Source definition, source name, or source identifier to process.
    
    Returns
    -------
    str
        Resulting string value.
    """
    if source == "tickets":
        return NATIVE_JIRA_CONVERSION_ENGINE
    return MARKITDOWN_CONVERSION_ENGINE


def resolve_chunking_settings(
    cfg: Any,
    *,
    source: str,
    strategy_override: str | None = None,
    chunk_size_override: int | None = None,
    overlap_override: int | None = None,
) -> ChunkingSettings:
    """Resolve chunking Settings.
    
    Parameters
    ----------
    cfg : Any
        Configuration object or mapping used to resolve runtime settings.
    source : str
        Source definition, source name, or source identifier to process.
    strategy_override : str or None, optional
        Value for strategy Override.
    chunk_size_override : int or None, optional
        Value for chunk Size Override.
    overlap_override : int or None, optional
        Value for overlap Override.
    
    Returns
    -------
    ChunkingSettings
        Resolved chunking Settings.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    ingestion_cfg = _as_mapping(getattr(cfg, "ingestion", None))
    chunking_cfg = _as_mapping(ingestion_cfg.get("chunking"))
    default_cfg = _as_mapping(chunking_cfg.get("default"))
    source_cfg = _as_mapping(_as_mapping(chunking_cfg.get("sources")).get(source))

    strategy = (
        strategy_override
        or source_cfg.get("strategy")
        or default_cfg.get("strategy")
        or _default_chunking_strategy(source)
    )
    chunk_size_tokens = (
        chunk_size_override
        if chunk_size_override is not None
        else source_cfg.get("chunk_size_tokens", default_cfg.get("chunk_size_tokens", DEFAULT_CHUNK_SIZE_TOKENS))
    )
    overlap_tokens = (
        overlap_override
        if overlap_override is not None
        else source_cfg.get("overlap_tokens", default_cfg.get("overlap_tokens", DEFAULT_OVERLAP_TOKENS))
    )

    resolved_chunk_size = _resolve_int(chunk_size_tokens, DEFAULT_CHUNK_SIZE_TOKENS)
    resolved_overlap = _resolve_int(overlap_tokens, DEFAULT_OVERLAP_TOKENS)
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size_tokens must be > 0")
    if resolved_overlap >= resolved_chunk_size:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    return ChunkingSettings(
        strategy=str(strategy).strip() or _default_chunking_strategy(source),
        chunk_size_tokens=resolved_chunk_size,
        overlap_tokens=resolved_overlap,
    )


def resolve_conversion_settings(
    cfg: Any,
    *,
    source: str,
    engine_override: str | None = None,
) -> ConversionSettings:
    """Resolve conversion Settings.
    
    Parameters
    ----------
    cfg : Any
        Configuration object or mapping used to resolve runtime settings.
    source : str
        Source definition, source name, or source identifier to process.
    engine_override : str or None, optional
        Value for engine Override.
    
    Returns
    -------
    ConversionSettings
        Resolved conversion Settings.
    """
    ingestion_cfg = _as_mapping(getattr(cfg, "ingestion", None))
    conversion_cfg = _as_mapping(ingestion_cfg.get("conversion"))
    default_cfg = _as_mapping(conversion_cfg.get("default"))
    source_cfg = _as_mapping(_as_mapping(conversion_cfg.get("sources")).get(source))

    options = {}
    options.update(_as_mapping(default_cfg.get("options")))
    options.update(_as_mapping(source_cfg.get("options")))

    engine = (
        engine_override
        or source_cfg.get("engine")
        or default_cfg.get("engine")
        or _default_conversion_engine(source)
    )

    return ConversionSettings(
        engine=str(engine).strip() or _default_conversion_engine(source),
        options=dict(options),
    )


__all__ = [
    "ChunkingSettings",
    "ConversionSettings",
    "DEFAULT_CHUNK_SIZE_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "HTML_HIERARCHICAL_CHUNKING_STRATEGY",
    "JIRA_TURNS_TOKEN_CHUNKING_STRATEGY",
    "MARKDOWN_TOKEN_CHUNKING_STRATEGY",
    "MARKITDOWN_CONVERSION_ENGINE",
    "NATIVE_JIRA_CONVERSION_ENGINE",
    "resolve_chunking_settings",
    "resolve_conversion_settings",
]
