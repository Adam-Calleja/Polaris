"""polaris_rag.retrieval.reranker.

Reranker abstractions and implementations for multi-source retrieval.

This module provides: - a normalized merged-candidate container - an abstract reranker
interface - a reciprocal-rank-fusion (RRF) reranker - a validity-aware reranker built on
top of RRF - profile/fingerprint helpers for experiment-safe configuration changes

Classes
-------
MergedCandidate
    Merged retrieval candidate across one or more sources.
ValidityRerankerConfig
    Resolved validity-aware reranker configuration.
BaseReranker
    Abstract interface for reranking merged multi-source candidates.
RRFReranker
    Reciprocal-rank-fusion reranker.
ValidityAwareReranker
    Validity-aware reranker using RRF as the semantic base signal.

Functions
---------
reranker_fingerprint
    Return a stable SHA256 fingerprint for a reranker profile.
create_reranker
    Create a reranker from configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import yaml
from llama_index.core.schema import NodeWithScore
from polaris_rag.retrieval.scope_family import specialized_families_for_values

try:
    from polaris_rag.retrieval.query_constraints import QueryConstraints
except Exception:  # pragma: no cover - defensive import for lighter test contexts
    QueryConstraints = Any  # type: ignore[assignment]


_TIMESTAMP_OFFSET_SUFFIX = re.compile(r"([+-]\d{2})(\d{2})$")

_DEFAULT_VALIDITY_WEIGHTS: dict[str, float] = {
    "authority": 0.04,
    "scope": 0.04,
    "software": 0.0,
    "scope_family": 0.0,
    "version": 0.04,
    "status": 0.04,
    "freshness": 0.01,
}
_DEFAULT_AUTHORITY_VALUES: dict[str, float] = {
    "local_official": 1.0,
    "external_official": 0.5,
    "ticket_memory": 0.0,
    "unknown": 0.0,
}
_DEFAULT_STATUS_VALUES: dict[str, float] = {
    "current": 1.0,
    "maintenance": 0.25,
    "legacy": -0.5,
    "eol": -1.0,
    "unknown": 0.0,
}


@dataclass
class MergedCandidate:
    """Merged retrieval candidate across one or more sources.
    
    Attributes
    ----------
    node : Any
        Value for node.
    best_score : float or None
        Value for best Score.
    source_ranks : dict[str, int]
        Value for source Ranks.
    """

    node: Any
    best_score: float | None
    source_ranks: dict[str, int]


@dataclass(frozen=True)
class ValidityRerankerConfig:
    """Resolved validity-aware reranker configuration.
    
    Attributes
    ----------
    trace_enabled : bool
        Value for trace Enabled.
    semantic_base_type : str
        Value for semantic Base Type.
    rrf_k : int
        Value for RRF K.
    source_weights : dict[str, float]
        Value for source Weights.
    weights : dict[str, float]
        Weight values to evaluate or persist.
    authority_values : dict[str, float]
        Value for authority Values.
    status_values : dict[str, float]
        Value for status Values.
    freshness_mode : str
        Value for freshness Mode.
    weights_source : dict[str, Any]
        Value for weights Source.
    """

    trace_enabled: bool
    semantic_base_type: str
    rrf_k: int
    source_weights: dict[str, float]
    weights: dict[str, float]
    authority_values: dict[str, float]
    status_values: dict[str, float]
    freshness_mode: str
    weights_source: dict[str, Any]

    def profile(self) -> dict[str, Any]:
        """Profile.
        
        Returns
        -------
        dict[str, Any]
            Structured result of the operation.
        """
        return {
            "type": "validity_aware",
            "trace_enabled": self.trace_enabled,
            "semantic_base": {
                "type": self.semantic_base_type,
                "rrf_k": self.rrf_k,
                "source_weights": dict(self.source_weights),
            },
            "weights": dict(self.weights),
            "authority_values": dict(self.authority_values),
            "status_values": dict(self.status_values),
            "freshness": {"mode": self.freshness_mode},
            "weights_source": dict(self.weights_source),
        }


class BaseReranker(ABC):
    """Abstract interface for reranking merged multi-source candidates.
    
    Methods
    -------
    rerank
        Return reranked candidates as ``NodeWithScore`` items.
    profile
        Return a stable JSON-serializable reranker profile.
    fingerprint
        Return a stable fingerprint for experiment compatibility checks.
    """

    @abstractmethod
    def rerank(
        self,
        candidates: Sequence[MergedCandidate],
        *,
        query_constraints: QueryConstraints | Mapping[str, Any] | None = None,
    ) -> list[NodeWithScore]:
        """Return reranked candidates as ``NodeWithScore`` items.
        
        Parameters
        ----------
        candidates : Sequence[MergedCandidate]
            Value for candidates.
        query_constraints : QueryConstraints or Mapping[str, Any] or None, optional
            Optional structured retrieval constraints.
        
        Returns
        -------
        list[NodeWithScore]
            Collected results from the operation.
        
        Raises
        ------
        NotImplementedError
            If `NotImplementedError` is raised while executing the operation.
        """
        raise NotImplementedError

    @abstractmethod
    def profile(self) -> dict[str, Any]:
        """Return a stable JSON-serializable reranker profile.
        
        Returns
        -------
        dict[str, Any]
            Structured result of the operation.
        
        Raises
        ------
        NotImplementedError
            If `NotImplementedError` is raised while executing the operation.
        """
        raise NotImplementedError

    def fingerprint(self) -> str:
        """Return a stable fingerprint for experiment compatibility checks.
        
        Returns
        -------
        str
            Resulting string value.
        """

        return reranker_fingerprint(self.profile())


class RRFReranker(BaseReranker):
    """Reciprocal-rank-fusion reranker.
    
    Parameters
    ----------
    rrf_k : int, optional
        Value for RRF K.
    source_weights : Mapping[str, float] or None, optional
        Value for source Weights.
    trace_enabled : bool, optional
        Value for trace Enabled.
    
    Methods
    -------
    rerank
        Rerank merged candidates using reciprocal-rank fusion.
    profile
        Profile.
    """

    def __init__(
        self,
        *,
        rrf_k: int = 60,
        source_weights: Mapping[str, float] | None = None,
        trace_enabled: bool = False,
    ):
        """Initialize the instance.
        
        Parameters
        ----------
        rrf_k : int, optional
            Value for RRF K.
        source_weights : Mapping[str, float] or None, optional
            Value for source Weights.
        trace_enabled : bool, optional
            Value for trace Enabled.
        
        Raises
        ------
        ValueError
            If the provided value is invalid for the operation.
        """
        self.rrf_k = int(rrf_k)
        if self.rrf_k <= 0:
            raise ValueError("'rerank.rrf_k' must be a positive integer.")
        self.source_weights = {
            str(source_name): _coerce_float(weight_raw, 1.0)
            for source_name, weight_raw in dict(source_weights or {}).items()
        }
        self.trace_enabled = bool(trace_enabled)

    def rerank(
        self,
        candidates: Sequence[MergedCandidate],
        *,
        query_constraints: QueryConstraints | Mapping[str, Any] | None = None,
    ) -> list[NodeWithScore]:
        """Rerank merged candidates using reciprocal-rank fusion.
        
        Parameters
        ----------
        candidates : Sequence[MergedCandidate]
            Value for candidates.
        query_constraints : QueryConstraints or Mapping[str, Any] or None, optional
            Optional structured retrieval constraints.
        
        Returns
        -------
        list[NodeWithScore]
            Collected results from the operation.
        """
        _ = query_constraints
        scored = _score_candidates_with_rrf(
            candidates,
            rrf_k=self.rrf_k,
            source_weights=self.source_weights,
        )

        items: list[tuple[float, float, str, NodeWithScore]] = []
        for score_data in scored:
            if self.trace_enabled:
                _stamp_rerank_trace(
                    score_data.candidate.node,
                    {
                        "reranker_type": "rrf",
                        "rrf_k": self.rrf_k,
                        "source_weights": dict(self.source_weights),
                        "source_ranks": dict(score_data.candidate.source_ranks),
                        "rrf_score": score_data.rrf_score,
                    },
                )

            items.append(
                (
                    score_data.rrf_score,
                    score_data.tie_break,
                    score_data.node_id,
                    NodeWithScore(node=score_data.candidate.node, score=score_data.rrf_score),
                )
            )

        items.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [item[3] for item in items]

    def profile(self) -> dict[str, Any]:
        """Profile.
        
        Returns
        -------
        dict[str, Any]
            Structured result of the operation.
        """
        return {
            "type": "rrf",
            "rrf_k": self.rrf_k,
            "source_weights": dict(self.source_weights),
            "trace_enabled": self.trace_enabled,
        }


class ValidityAwareReranker(BaseReranker):
    """Validity-aware reranker using RRF as the semantic base signal.
    
    Parameters
    ----------
    config : ValidityRerankerConfig
        Configuration object or mapping used by the operation.
    
    Methods
    -------
    profile
        Profile.
    rerank
        Rerank.
    """

    def __init__(self, config: ValidityRerankerConfig) -> None:
        """Initialize the instance.
        
        Parameters
        ----------
        config : ValidityRerankerConfig
            Configuration object or mapping used by the operation.
        
        Raises
        ------
        ValueError
            If the provided value is invalid for the operation.
        """
        if config.semantic_base_type != "rrf":
            raise ValueError(
                "Unsupported validity-aware semantic base "
                f"{config.semantic_base_type!r}. Supported semantic bases: ['rrf']."
            )
        self.config = config

    def profile(self) -> dict[str, Any]:
        """Profile.
        
        Returns
        -------
        dict[str, Any]
            Structured result of the operation.
        """
        return self.config.profile()

    def rerank(
        self,
        candidates: Sequence[MergedCandidate],
        *,
        query_constraints: QueryConstraints | Mapping[str, Any] | None = None,
    ) -> list[NodeWithScore]:
        """Rerank.
        
        Parameters
        ----------
        candidates : Sequence[MergedCandidate]
            Value for candidates.
        query_constraints : QueryConstraints or Mapping[str, Any] or None, optional
            Optional structured retrieval constraints.
        
        Returns
        -------
        list[NodeWithScore]
            Collected results from the operation.
        """
        normalized_constraints = _normalize_query_constraints(query_constraints)
        scored = _score_candidates_with_rrf(
            candidates,
            rrf_k=self.config.rrf_k,
            source_weights=self.config.source_weights,
        )
        if not scored:
            return []

        semantic_base_by_id = _normalized_score_map(
            (
                (item.node_id, item.rrf_score)
                for item in scored
            ),
            default_when_uniform=1.0,
        )
        freshness_by_id = _freshness_feature_map(scored)

        items: list[tuple[float, float, int, str, NodeWithScore]] = []
        for base_rank, score_data in enumerate(scored, start=1):
            candidate = score_data.candidate
            node = candidate.node
            metadata = _node_metadata(node)
            node_id = score_data.node_id

            semantic_base = semantic_base_by_id.get(node_id, 1.0)
            authority_feature = _authority_feature(metadata, self.config.authority_values)
            scope_feature = _scope_feature(normalized_constraints, metadata)
            software_feature = _software_feature(normalized_constraints, metadata)
            scope_family_feature = _scope_family_feature(normalized_constraints, metadata)
            (
                scope_family_effective_feature,
                scope_family_gate_reason,
            ) = _effective_scope_family_feature(
                normalized_constraints,
                metadata,
                raw_scope_family_feature=scope_family_feature,
                software_feature=software_feature,
            )
            version_feature = _version_feature(normalized_constraints, metadata)
            status_feature = _status_feature(metadata, self.config.status_values)
            freshness_feature = freshness_by_id.get(node_id, 0.0)

            contributions = {
                "authority_feature": self.config.weights["authority"] * authority_feature,
                "scope_feature": self.config.weights["scope"] * scope_feature,
                "software_feature": self.config.weights["software"] * software_feature,
                "scope_family_feature": self.config.weights["scope_family"] * scope_family_effective_feature,
                "version_feature": self.config.weights["version"] * version_feature,
                "status_feature": self.config.weights["status"] * status_feature,
                "freshness_feature": self.config.weights["freshness"] * freshness_feature,
            }
            final_score = semantic_base + sum(contributions.values())

            trace_payload = {
                "reranker_type": "validity_aware",
                "base_rank": base_rank,
                "base_rrf_score": score_data.rrf_score,
                "semantic_base": semantic_base,
                "authority_feature": authority_feature,
                "scope_feature": scope_feature,
                "software_feature": software_feature,
                "scope_family_feature": scope_family_feature,
                "scope_family_effective_feature": scope_family_effective_feature,
                "scope_family_gate_reason": scope_family_gate_reason,
                "version_feature": version_feature,
                "status_feature": status_feature,
                "freshness_feature": freshness_feature,
                "weighted_contributions": contributions,
                "final_score": final_score,
                "source_ranks": dict(candidate.source_ranks),
                "query_constraints": _stable_json_value(normalized_constraints),
            }

            if self.config.trace_enabled:
                _stamp_rerank_trace(node, trace_payload)

            items.append(
                (
                    final_score,
                    semantic_base,
                    base_rank,
                    node_id,
                    NodeWithScore(node=node, score=final_score),
                )
            )

        items.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        return [item[4] for item in items]


@dataclass(frozen=True)
class _RRFScore:
    candidate: MergedCandidate
    node_id: str
    rrf_score: float
    tie_break: float


def reranker_fingerprint(profile: Mapping[str, Any] | None) -> str:
    """Return a stable SHA256 fingerprint for a reranker profile.
    
    Parameters
    ----------
    profile : Mapping[str, Any] or None, optional
        Value for profile.
    
    Returns
    -------
    str
        Resulting string value.
    """

    payload = json.dumps(_stable_json_value(profile or {}), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_reranker(
    *,
    config: Mapping[str, Any] | None,
    source_settings: Mapping[str, Mapping[str, Any]] | None = None,
    config_base_dir: str | Path | None = None,
) -> BaseReranker:
    """Create a reranker from configuration.
    
    Parameters
    ----------
    config : Mapping[str, Any] or None, optional
        Configuration object or mapping used by the operation.
    source_settings : Mapping[str, Mapping[str, Any]] or None, optional
        Value for source Settings.
    config_base_dir : str or Path or None, optional
        Value for config Base Dir.
    
    Returns
    -------
    BaseReranker
        Created reranker.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """

    cfg = dict(config or {})
    kind = str(cfg.get("type", "rrf")).lower().strip()

    source_weights = _source_weights_from_settings(source_settings)

    if kind == "rrf":
        return RRFReranker(
            rrf_k=int(cfg.get("rrf_k", 60)),
            source_weights=source_weights,
            trace_enabled=_coerce_bool(cfg.get("trace_enabled"), False),
        )

    if kind == "validity_aware":
        resolved = _resolve_validity_reranker_config(
            cfg,
            source_weights=source_weights,
            config_base_dir=config_base_dir,
        )
        return ValidityAwareReranker(resolved)

    raise ValueError(f"Unsupported rerank type {kind!r}. Supported rerankers: ['rrf', 'validity_aware'].")


def _resolve_validity_reranker_config(
    cfg: Mapping[str, Any],
    *,
    source_weights: Mapping[str, float],
    config_base_dir: str | Path | None,
) -> ValidityRerankerConfig:
    """Resolve validity Reranker Config.
    
    Parameters
    ----------
    cfg : Mapping[str, Any]
        Configuration object or mapping used to resolve runtime settings.
    source_weights : Mapping[str, float]
        Value for source Weights.
    config_base_dir : str or Path or None, optional
        Value for config Base Dir.
    
    Returns
    -------
    ValidityRerankerConfig
        Result of the operation.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    FileNotFoundError
        If the requested file does not exist.
    TypeError
        If the provided value has an unexpected type.
    """
    semantic_base_cfg = _as_mapping(cfg.get("semantic_base"))
    semantic_base_type = str(semantic_base_cfg.get("type", "rrf")).strip().lower()
    if semantic_base_type != "rrf":
        raise ValueError(
            "Unsupported validity-aware semantic base "
            f"{semantic_base_type!r}. Supported semantic bases: ['rrf']."
        )

    rrf_k = int(semantic_base_cfg.get("rrf_k", cfg.get("rrf_k", 60)))
    if rrf_k <= 0:
        raise ValueError("'rerank.semantic_base.rrf_k' must be a positive integer.")

    weights_source: dict[str, Any] = {"type": "defaults"}
    merged_cfg: dict[str, Any] = {}

    weights_path_raw = cfg.get("weights_path")
    if weights_path_raw is not None and str(weights_path_raw).strip():
        resolved_path = _resolve_config_relative_path(weights_path_raw, config_base_dir=config_base_dir)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Validity reranker weights file not found: {resolved_path}")
        payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise TypeError(f"Validity reranker weights file {resolved_path} must contain a mapping.")
        merged_cfg.update(payload)
        weights_source = {
            "type": "file",
            "path": str(resolved_path),
            "sha256": _sha256_text(resolved_path.read_text(encoding="utf-8")),
        }

    merged_cfg.update({key: value for key, value in cfg.items() if key != "weights_path"})

    weights = dict(_DEFAULT_VALIDITY_WEIGHTS)
    weights.update(
        {
            str(name): _coerce_float(value, weights.get(str(name), 0.0))
            for name, value in _as_mapping(merged_cfg.get("weights")).items()
        }
    )
    authority_values = dict(_DEFAULT_AUTHORITY_VALUES)
    authority_values.update(
        {
            str(name): _coerce_float(value, authority_values.get(str(name), 0.0))
            for name, value in _as_mapping(merged_cfg.get("authority_values")).items()
        }
    )
    status_values = dict(_DEFAULT_STATUS_VALUES)
    status_values.update(
        {
            str(name): _coerce_float(value, status_values.get(str(name), 0.0))
            for name, value in _as_mapping(merged_cfg.get("status_values")).items()
        }
    )
    freshness_cfg = _as_mapping(merged_cfg.get("freshness"))
    freshness_mode = str(freshness_cfg.get("mode", "relative_recency")).strip().lower() or "relative_recency"
    if freshness_mode != "relative_recency":
        raise ValueError(
            f"Unsupported validity reranker freshness mode {freshness_mode!r}. "
            "Supported freshness modes: ['relative_recency']."
        )

    return ValidityRerankerConfig(
        trace_enabled=_coerce_bool(merged_cfg.get("trace_enabled"), True),
        semantic_base_type=semantic_base_type,
        rrf_k=rrf_k,
        source_weights={str(k): _coerce_float(v, 1.0) for k, v in dict(source_weights).items()},
        weights=weights,
        authority_values=authority_values,
        status_values=status_values,
        freshness_mode=freshness_mode,
        weights_source=weights_source,
    )


def _score_candidates_with_rrf(
    candidates: Sequence[MergedCandidate],
    *,
    rrf_k: int,
    source_weights: Mapping[str, float],
) -> list[_RRFScore]:
    """Score Candidates With RRF.
    
    Parameters
    ----------
    candidates : Sequence[MergedCandidate]
        Value for candidates.
    rrf_k : int
        Value for RRF K.
    source_weights : Mapping[str, float]
        Value for source Weights.
    
    Returns
    -------
    list[_RRFScore]
        Collected results from the operation.
    """
    scored: list[_RRFScore] = []
    for candidate in candidates:
        node_id = _node_id(candidate.node)
        rrf_score = 0.0
        for source_name, rank in candidate.source_ranks.items():
            weight = _coerce_float(source_weights.get(source_name), 1.0)
            rrf_score += weight / (int(rrf_k) + int(rank))
        tie_break = candidate.best_score if candidate.best_score is not None else float("-inf")
        scored.append(
            _RRFScore(
                candidate=candidate,
                node_id=node_id,
                rrf_score=rrf_score,
                tie_break=tie_break,
            )
        )

    scored.sort(key=lambda item: (item.rrf_score, item.tie_break, item.node_id), reverse=True)
    return scored


def _normalized_score_map(
    items: Iterable[tuple[str, float]],
    *,
    default_when_uniform: float,
) -> dict[str, float]:
    """Normalized Score Map.
    
    Parameters
    ----------
    items : Iterable[tuple[str, float]]
        Value for items.
    default_when_uniform : float
        Value for default When Uniform.
    
    Returns
    -------
    dict[str, float]
        Structured result of the operation.
    """
    values = list(items)
    if not values:
        return {}
    raw_scores = [score for _, score in values]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if max_score - min_score <= 1e-12:
        return {key: float(default_when_uniform) for key, _ in values}
    return {
        key: (score - min_score) / (max_score - min_score)
        for key, score in values
    }


def _authority_feature(metadata: Mapping[str, Any], values: Mapping[str, float]) -> float:
    """Authority Feature.
    
    Parameters
    ----------
    metadata : Mapping[str, Any]
        Metadata mapping to extend or stamp.
    values : Mapping[str, float]
        Value for values.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    authority = str(metadata.get("source_authority", "unknown") or "unknown").strip().lower()
    return _coerce_float(values.get(authority), 0.0)


def _scope_feature(
    constraints: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> float:
    """Scope Feature.
    
    Parameters
    ----------
    constraints : Mapping[str, Any] or None, optional
        Value for constraints.
    metadata : Mapping[str, Any]
        Metadata mapping to extend or stamp.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    if not constraints:
        return 0.0

    fields = ("system_names", "partition_names", "service_names")
    component_scores: list[float] = []
    for field in fields:
        query_values = _normalized_text_set(constraints.get(field))
        if not query_values:
            continue
        candidate_values = _normalized_text_set(metadata.get(field))
        component_scores.append(_match_component_score(query_values, candidate_values))
    return _average_or_zero(component_scores)


def _scope_family_feature(
    constraints: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> float:
    """Scope Family Feature.
    
    Parameters
    ----------
    constraints : Mapping[str, Any] or None, optional
        Value for constraints.
    metadata : Mapping[str, Any]
        Metadata mapping to extend or stamp.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    if not constraints:
        return 0.0

    query_values = _normalized_text_set(constraints.get("scope_family_names"))
    if not query_values:
        return 0.0
    candidate_values = _normalized_text_set(metadata.get("scope_family_names"))
    return _match_component_score(query_values, candidate_values)


def _software_feature(
    constraints: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> float:
    """Software Feature.
    
    Parameters
    ----------
    constraints : Mapping[str, Any] or None, optional
        Value for constraints.
    metadata : Mapping[str, Any]
        Metadata mapping to extend or stamp.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    if not constraints:
        return 0.0

    query_values = _normalized_text_set(constraints.get("software_names"))
    if not query_values:
        return 0.0
    candidate_values = _normalized_text_set(metadata.get("software_names"))
    return _match_component_score(query_values, candidate_values)


def _effective_scope_family_feature(
    constraints: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
    *,
    raw_scope_family_feature: float,
    software_feature: float,
) -> tuple[float, str]:
    """Effective Scope Family Feature.
    
    Parameters
    ----------
    constraints : Mapping[str, Any] or None, optional
        Value for constraints.
    metadata : Mapping[str, Any]
        Metadata mapping to extend or stamp.
    raw_scope_family_feature : float
        Raw scope Family Feature value to normalize.
    software_feature : float
        Value for software Feature.
    
    Returns
    -------
    tuple[float, str]
        Collected results from the operation.
    """
    if raw_scope_family_feature <= 0.0:
        return raw_scope_family_feature, "applied"
    if not constraints or not _query_is_software_constrained(constraints):
        return raw_scope_family_feature, "applied"
    if software_feature <= 0.0:
        return 0.0, "blocked_no_software_match"
    if _query_has_exact_scope(constraints):
        return raw_scope_family_feature, "applied"
    if _candidate_has_specialized_family_variant(constraints, metadata):
        return 0.0, "blocked_specialized_variant"
    return raw_scope_family_feature, "applied"


def _version_feature(
    constraints: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> float:
    """Version Feature.
    
    Parameters
    ----------
    constraints : Mapping[str, Any] or None, optional
        Value for constraints.
    metadata : Mapping[str, Any]
        Metadata mapping to extend or stamp.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    if not constraints:
        return 0.0

    fields = ("module_names", "software_versions", "toolchain_versions")
    component_scores: list[float] = []
    for field in fields:
        query_values = _normalized_text_set(constraints.get(field))
        if not query_values:
            continue
        candidate_values = _normalized_text_set(metadata.get(field))
        component_scores.append(_match_component_score(query_values, candidate_values))
    return _average_or_zero(component_scores)


def _status_feature(metadata: Mapping[str, Any], values: Mapping[str, float]) -> float:
    """Status Feature.
    
    Parameters
    ----------
    metadata : Mapping[str, Any]
        Metadata mapping to extend or stamp.
    values : Mapping[str, float]
        Value for values.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    status = str(metadata.get("validity_status", "unknown") or "unknown").strip().lower()
    return _coerce_float(values.get(status), 0.0)


def _freshness_feature_map(scored_candidates: Sequence[_RRFScore]) -> dict[str, float]:
    """Freshness Feature Map.
    
    Parameters
    ----------
    scored_candidates : Sequence[_RRFScore]
        Value for scored Candidates.
    
    Returns
    -------
    dict[str, float]
        Structured result of the operation.
    """
    by_authority: dict[str, list[tuple[str, float]]] = {}
    for item in scored_candidates:
        metadata = _node_metadata(item.candidate.node)
        authority = str(metadata.get("source_authority", "unknown") or "unknown").strip().lower()
        timestamp = _parse_timestamp(metadata.get("freshness_hint"))
        if timestamp is None:
            continue
        by_authority.setdefault(authority, []).append((item.node_id, timestamp))

    feature_map: dict[str, float] = {}
    for entries in by_authority.values():
        entries.sort(key=lambda item: (item[1], item[0]), reverse=True)
        total = len(entries)
        for index, (node_id, _timestamp) in enumerate(entries):
            if total <= 1:
                feature_map[node_id] = 1.0
            else:
                feature_map[node_id] = 1.0 - (index / (total - 1))
    return feature_map


def _parse_timestamp(value: Any) -> float | None:
    """Parse timestamp.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    float or None
        Result of the operation.
    """
    text = str(value or "").strip()
    if not text:
        return None

    normalized = text
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    if _TIMESTAMP_OFFSET_SUFFIX.search(normalized):
        normalized = _TIMESTAMP_OFFSET_SUFFIX.sub(r"\1:\2", normalized)

    for candidate in (normalized, text):
        try:
            return datetime.fromisoformat(candidate).astimezone(timezone.utc).timestamp()
        except Exception:
            continue

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).timestamp()
        except Exception:
            continue

    return None


def _normalized_text_set(value: Any) -> set[str]:
    """Normalized Text Set.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    set[str]
        Collected results from the operation.
    """
    if not isinstance(value, (list, tuple, set)):
        return set()
    normalized: set[str] = set()
    for item in value:
        text = str(item or "").strip().lower()
        if text:
            normalized.add(text)
    return normalized


def _normalized_text_list(value: Any) -> list[str]:
    """Normalized Text List.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
    if not isinstance(value, (list, tuple, set)):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _match_component_score(query_values: set[str], candidate_values: set[str]) -> float:
    """Match Component Score.
    
    Parameters
    ----------
    query_values : set[str]
        Value for query Values.
    candidate_values : set[str]
        Value for candidate Values.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    if not query_values:
        return 0.0
    if not candidate_values:
        return 0.0
    if query_values & candidate_values:
        return 1.0
    return -1.0


def _average_or_zero(values: Sequence[float]) -> float:
    """Average Or Zero.
    
    Parameters
    ----------
    values : Sequence[float]
        Value for values.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def _query_is_software_constrained(constraints: Mapping[str, Any]) -> bool:
    """Query Is Software Constrained.
    
    Parameters
    ----------
    constraints : Mapping[str, Any]
        Value for constraints.
    
    Returns
    -------
    bool
        `True` if query Is Software Constrained; otherwise `False`.
    """
    return bool(_normalized_text_set(constraints.get("software_names")))


def _query_has_exact_scope(constraints: Mapping[str, Any]) -> bool:
    """Query Has Exact Scope.
    
    Parameters
    ----------
    constraints : Mapping[str, Any]
        Value for constraints.
    
    Returns
    -------
    bool
        `True` if query Has Exact Scope; otherwise `False`.
    """
    return any(
        _normalized_text_set(constraints.get(field))
        for field in ("system_names", "partition_names", "service_names")
    )


def _scope_variant_values(payload: Mapping[str, Any]) -> list[str]:
    """Scope Variant Values.
    
    Parameters
    ----------
    payload : Mapping[str, Any]
        Structured payload for the operation.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
    values: list[str] = []
    for field in ("system_names", "partition_names", "service_names", "module_names"):
        values.extend(_normalized_text_list(payload.get(field)))
    return values


def _candidate_has_specialized_family_variant(
    constraints: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bool:
    """Candidate Has Specialized Family Variant.
    
    Parameters
    ----------
    constraints : Mapping[str, Any]
        Value for constraints.
    metadata : Mapping[str, Any]
        Metadata mapping to extend or stamp.
    
    Returns
    -------
    bool
        `True` if candidate Has Specialized Family Variant; otherwise `False`.
    """
    query_families = _normalized_text_set(constraints.get("scope_family_names"))
    if not query_families:
        return False

    candidate_specialized_families = set(specialized_families_for_values(_scope_variant_values(metadata)))
    if not candidate_specialized_families:
        return False

    query_specialized_families = set(specialized_families_for_values(_scope_variant_values(constraints)))
    return bool((query_families & candidate_specialized_families) - query_specialized_families)


def _normalize_query_constraints(
    value: QueryConstraints | Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Normalize query Constraints.
    
    Parameters
    ----------
    value : QueryConstraints or Mapping[str, Any] or None, optional
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    Mapping[str, Any] or None
        Result of the operation.
    """
    if value is None:
        return None
    from_value = getattr(QueryConstraints, "from_value", None)
    if callable(from_value):
        normalized = from_value(value)
        if normalized is None:
            return None
        to_dict = getattr(normalized, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        if isinstance(normalized, Mapping):
            return dict(normalized)
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _stamp_rerank_trace(node: Any, payload: Mapping[str, Any]) -> None:
    """Stamp rerank Trace.
    
    Parameters
    ----------
    node : Any
        Value for node.
    payload : Mapping[str, Any]
        Structured payload for the operation.
    """
    metadata = getattr(node, "metadata", None)
    if not isinstance(metadata, dict):
        return
    metadata["rerank_trace"] = _stable_json_value(payload)


def _node_metadata(node: Any) -> dict[str, Any]:
    """Node Metadata.
    
    Parameters
    ----------
    node : Any
        Value for node.
    
    Returns
    -------
    dict[str, Any]
        Structured result of the operation.
    """
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def _node_id(node: Any) -> str:
    """Node ID.
    
    Parameters
    ----------
    node : Any
        Value for node.
    
    Returns
    -------
    str
        Resulting string value.
    """
    for attr in ("id_", "node_id", "id"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value:
            return value
    return f"<anon-node:{id(node)}>"


def _source_weights_from_settings(
    source_settings: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, float]:
    """Source Weights From Settings.
    
    Parameters
    ----------
    source_settings : Mapping[str, Mapping[str, Any]] or None, optional
        Value for source Settings.
    
    Returns
    -------
    dict[str, float]
        Structured result of the operation.
    """
    weights: dict[str, float] = {}
    for source_name, source_cfg in (source_settings or {}).items():
        weights[str(source_name)] = _coerce_float(_as_mapping(source_cfg).get("weight"), 1.0)
    return weights


def _resolve_config_relative_path(
    path_value: Any,
    *,
    config_base_dir: str | Path | None,
) -> Path:
    """Resolve config Relative Path.
    
    Parameters
    ----------
    path_value : Any
        Value for path Value.
    config_base_dir : str or Path or None, optional
        Value for config Base Dir.
    
    Returns
    -------
    Path
        Result of the operation.
    """
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if config_base_dir is not None:
        return (Path(config_base_dir).expanduser().resolve() / path).resolve()
    return path.resolve()


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
    return {}


def _coerce_float(value: Any, default: float) -> float:
    """Coerce float.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    default : float
        Fallback value to use when normalization fails.
    
    Returns
    -------
    float
        Computed floating-point value.
    """
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_bool(value: Any, default: bool) -> bool:
    """Coerce bool.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    default : bool
        Fallback value to use when normalization fails.
    
    Returns
    -------
    bool
        `True` if coerce Bool; otherwise `False`.
    """
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _stable_json_value(value: Any) -> Any:
    """Stable JSON Value.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    Any
        Result of the operation.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _stable_json_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_stable_json_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _stable_json_value(value.to_dict())
    if hasattr(value, "dict") and callable(value.dict):
        return _stable_json_value(value.dict())
    return str(value)


def _sha256_text(text: str) -> str:
    """Sha 256 Text.
    
    Parameters
    ----------
    text : str
        Text value to inspect, tokenize, or encode.
    
    Returns
    -------
    str
        Resulting string value.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "BaseReranker",
    "MergedCandidate",
    "RRFReranker",
    "ValidityAwareReranker",
    "create_reranker",
    "reranker_fingerprint",
]
