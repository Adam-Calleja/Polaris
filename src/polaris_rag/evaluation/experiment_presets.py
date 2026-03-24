"""Evaluation preset resolution for reproducible experiment conditions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from polaris_rag.config import GlobalConfig
from polaris_rag.retrieval.reranker import create_reranker


DEFAULT_VALIDITY_WEIGHTS_PATH = "config/weights/validity_reranker.dev_v3.yaml"
FEATURE_WEIGHT_KEYS: tuple[str, ...] = (
    "authority",
    "scope",
    "software",
    "scope_family",
    "version",
    "status",
    "freshness",
)


@dataclass(frozen=True)
class PresetContext:
    """Resolved preset metadata embedded into run artifacts."""

    preset_name: str | None
    preset_description: str | None
    condition_summary: dict[str, Any]
    condition_fingerprint: str

    def manifest_fields(self) -> dict[str, Any]:
        return {
            "preset_name": self.preset_name,
            "preset_description": self.preset_description,
            "condition_summary": dict(self.condition_summary),
            "condition_fingerprint": self.condition_fingerprint,
        }


def list_preset_names() -> list[str]:
    return [
        "docs_only",
        "tickets_only",
        "naive_combined",
        "source_aware",
        "freshness_only",
        "validity_aware",
        "all_docs_validity_aware",
    ]


def apply_evaluation_preset(cfg: GlobalConfig, preset_name: str | None) -> tuple[GlobalConfig, PresetContext]:
    """Apply an evaluation preset and return the effective config plus manifest metadata."""

    raw = deepcopy(_as_mapping(getattr(cfg, "raw", {})))
    if preset_name:
        handler = _PRESET_HANDLERS.get(preset_name)
        if handler is None:
            available = ", ".join(list_preset_names())
            raise ValueError(f"Unknown evaluation preset {preset_name!r}. Available presets: {available}")
        description = handler(raw, config_path=getattr(cfg, "config_path", None))
    else:
        description = None

    effective_cfg = GlobalConfig(raw=raw, config_path=getattr(cfg, "config_path", None))
    summary = resolve_condition_summary(effective_cfg)
    fingerprint = _stable_fingerprint(summary)
    return effective_cfg, PresetContext(
        preset_name=preset_name,
        preset_description=description,
        condition_summary=summary,
        condition_fingerprint=fingerprint,
    )


def resolve_condition_summary(cfg: GlobalConfig) -> dict[str, Any]:
    """Return a stable summary of the active experiment condition."""

    raw = _as_mapping(getattr(cfg, "raw", {}))
    retriever_cfg = _as_mapping(raw.get("retriever", {}))
    source_settings = _source_settings_from_raw(raw)
    active_sources = list(source_settings.keys())
    rerank_cfg = _as_mapping(retriever_cfg.get("rerank", {}))

    reranker_profile: dict[str, Any] | None = None
    if rerank_cfg:
        reranker = create_reranker(
            config=rerank_cfg,
            source_settings=source_settings,
            config_base_dir=_config_base_dir(getattr(cfg, "config_path", None)),
        )
        reranker_profile = dict(reranker.profile())

    vector_stores = _as_mapping(raw.get("vector_stores", {}))
    sources_summary: list[dict[str, Any]] = []
    for source_name in active_sources:
        source_store_cfg = _as_mapping(vector_stores.get(source_name, {}))
        source_cfg = dict(source_settings.get(source_name, {}))
        source_cfg["name"] = source_name
        source_cfg["collection_name"] = source_store_cfg.get("collection_name")
        sources_summary.append(source_cfg)

    return {
        "retriever_type": str(retriever_cfg.get("type", "") or ""),
        "source_type": str(retriever_cfg.get("source_type", "") or ""),
        "final_top_k": retriever_cfg.get("final_top_k"),
        "sources": sources_summary,
        "reranker_profile": reranker_profile,
    }


def _apply_docs_only(raw: dict[str, Any], *, config_path: Path | None) -> str:
    _configure_sources(raw, ["docs"])
    _configure_rrf_reranker(raw)
    return "Docs-only baseline with RRF reranking."


def _apply_tickets_only(raw: dict[str, Any], *, config_path: Path | None) -> str:
    _configure_sources(raw, ["tickets"])
    _configure_rrf_reranker(raw)
    return "Tickets-only baseline with RRF reranking."


def _apply_naive_combined(raw: dict[str, Any], *, config_path: Path | None) -> str:
    _configure_sources(raw, ["docs", "tickets"])
    _configure_rrf_reranker(raw)
    return "Naive docs+tickets baseline with RRF reranking."


def _apply_source_aware(raw: dict[str, Any], *, config_path: Path | None) -> str:
    weights = _target_feature_weights(raw, config_path=config_path, enabled_feature="authority")
    _configure_sources(raw, ["docs", "tickets"])
    _configure_validity_reranker(raw, weights=weights, config_path=config_path)
    return "Authority-only validity-aware reranker over docs+tickets."


def _apply_freshness_only(raw: dict[str, Any], *, config_path: Path | None) -> str:
    weights = _target_feature_weights(raw, config_path=config_path, enabled_feature="freshness")
    _configure_sources(raw, ["docs", "tickets"])
    _configure_validity_reranker(raw, weights=weights, config_path=config_path)
    return "Freshness-only validity-aware reranker over docs+tickets."


def _apply_validity_aware(raw: dict[str, Any], *, config_path: Path | None) -> str:
    _configure_sources(raw, ["docs", "tickets"])
    _configure_validity_reranker(raw, weights=None, config_path=config_path)
    return "Frozen validity-aware reranker over docs+tickets."


def _apply_all_docs_validity_aware(raw: dict[str, Any], *, config_path: Path | None) -> str:
    external_source_names = _configured_external_source_names(raw)
    if not external_source_names:
        raise ValueError(
            "Preset 'all_docs_validity_aware' requires an external official docs retriever source. "
            "No vector_stores entry with authority_scope=external_official is configured yet."
        )
    _configure_sources(raw, ["docs", *external_source_names, "tickets"])
    _configure_validity_reranker(raw, weights=None, config_path=config_path)
    return "Frozen validity-aware reranker over local docs, external docs, and tickets."


_PRESET_HANDLERS = {
    "docs_only": _apply_docs_only,
    "tickets_only": _apply_tickets_only,
    "naive_combined": _apply_naive_combined,
    "source_aware": _apply_source_aware,
    "freshness_only": _apply_freshness_only,
    "validity_aware": _apply_validity_aware,
    "all_docs_validity_aware": _apply_all_docs_validity_aware,
}


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _configured_source_names(raw: Mapping[str, Any]) -> list[str]:
    retriever_cfg = _as_mapping(raw.get("retriever", {}))
    raw_sources = retriever_cfg.get("sources")
    if not isinstance(raw_sources, list):
        return []
    names: list[str] = []
    for item in raw_sources:
        item_map = _as_mapping(item)
        name = str(item_map.get("name", "") or "").strip()
        if name:
            names.append(name)
    return names


def _configured_external_source_names(raw: Mapping[str, Any]) -> list[str]:
    vector_stores = _as_mapping(raw.get("vector_stores", {}))
    return sorted(
        source_name
        for source_name, store_cfg in vector_stores.items()
        if str(_as_mapping(store_cfg).get("authority_scope", "") or "").strip().lower() == "external_official"
    )


def _configure_sources(raw: dict[str, Any], source_names: list[str]) -> None:
    retriever_cfg = _as_mapping(raw.get("retriever", {}))
    existing_sources = retriever_cfg.get("sources")
    if not isinstance(existing_sources, list):
        raise ValueError("evaluation preset resolution requires retriever.sources to be configured as a list.")
    existing_by_name = {
        str(_as_mapping(item).get("name", "") or "").strip(): dict(_as_mapping(item))
        for item in existing_sources
        if str(_as_mapping(item).get("name", "") or "").strip()
    }
    missing = [name for name in source_names if name not in existing_by_name]
    if missing:
        vector_stores = _as_mapping(raw.get("vector_stores", {}))
        default_top_k = retriever_cfg.get("top_k")
        default_filters = retriever_cfg.get("filters")
        for name in list(missing):
            if name not in vector_stores:
                continue
            existing_by_name[name] = {
                "name": name,
                "top_k": default_top_k,
                "filters": default_filters,
                "weight": 1.0,
            }
            missing.remove(name)
    if missing:
        raise ValueError(
            "evaluation preset resolution references unconfigured retriever sources: "
            + ", ".join(missing)
        )
    retriever_cfg["sources"] = [existing_by_name[name] for name in source_names]
    raw["retriever"] = retriever_cfg


def _configure_rrf_reranker(raw: dict[str, Any]) -> None:
    retriever_cfg = _as_mapping(raw.get("retriever", {}))
    current_rerank = _as_mapping(retriever_cfg.get("rerank", {}))
    semantic_base_cfg = _as_mapping(current_rerank.get("semantic_base", {}))
    rrf_k = int(semantic_base_cfg.get("rrf_k", current_rerank.get("rrf_k", 60)) or 60)
    retriever_cfg["rerank"] = {
        "type": "rrf",
        "rrf_k": rrf_k,
        "trace_enabled": True,
    }
    raw["retriever"] = retriever_cfg


def _configure_validity_reranker(
    raw: dict[str, Any],
    *,
    weights: Mapping[str, float] | None,
    config_path: Path | None,
) -> None:
    retriever_cfg = _as_mapping(raw.get("retriever", {}))
    current_rerank = _as_mapping(retriever_cfg.get("rerank", {}))
    semantic_base_cfg = _as_mapping(current_rerank.get("semantic_base", {}))
    rrf_k = int(semantic_base_cfg.get("rrf_k", current_rerank.get("rrf_k", 60)) or 60)
    weights_path = _resolved_weights_path(raw, config_path=config_path)
    rerank_cfg: dict[str, Any] = {
        "type": "validity_aware",
        "trace_enabled": True,
        "semantic_base": {
            "type": "rrf",
            "rrf_k": rrf_k,
        },
        "weights_path": weights_path,
    }
    if weights is not None:
        rerank_cfg["weights"] = {key: float(value) for key, value in weights.items()}
    retriever_cfg["rerank"] = rerank_cfg
    raw["retriever"] = retriever_cfg


def _target_feature_weights(
    raw: Mapping[str, Any],
    *,
    config_path: Path | None,
    enabled_feature: str,
) -> dict[str, float]:
    tuned_weights = _load_tuned_feature_weights(raw, config_path=config_path)
    if enabled_feature not in FEATURE_WEIGHT_KEYS:
        raise ValueError(f"Unsupported feature weight {enabled_feature!r}.")
    return {
        key: float(tuned_weights[key]) if key == enabled_feature else 0.0
        for key in FEATURE_WEIGHT_KEYS
    }


def _load_tuned_feature_weights(raw: Mapping[str, Any], *, config_path: Path | None) -> dict[str, float]:
    resolved = _resolve_weights_file_path(raw, config_path=config_path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"Validity reranker weights file {resolved} must contain a mapping.")
    weights = _as_mapping(payload.get("weights", {}))
    missing = [key for key in FEATURE_WEIGHT_KEYS if key not in weights]
    if missing:
        raise ValueError(
            f"Validity reranker weights file {resolved} is missing feature weights: {', '.join(missing)}"
        )
    return {key: float(weights[key]) for key in FEATURE_WEIGHT_KEYS}


def _resolved_weights_path(raw: Mapping[str, Any], *, config_path: Path | None) -> str:
    retriever_cfg = _as_mapping(raw.get("retriever", {}))
    rerank_cfg = _as_mapping(retriever_cfg.get("rerank", {}))
    configured = str(rerank_cfg.get("weights_path", "") or "").strip()
    if configured:
        return configured
    return DEFAULT_VALIDITY_WEIGHTS_PATH


def _resolve_weights_file_path(raw: Mapping[str, Any], *, config_path: Path | None) -> Path:
    configured = _resolved_weights_path(raw, config_path=config_path)
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        base_dir = _config_base_dir(config_path)
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Validity-aware evaluation presets require a frozen Stage 4 weights file. "
            f"Expected to find one at {candidate}."
        )
    return candidate


def _source_settings_from_raw(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    retriever_cfg = _as_mapping(raw.get("retriever", {}))
    raw_sources = retriever_cfg.get("sources")
    if not isinstance(raw_sources, list):
        return {}
    default_top_k = retriever_cfg.get("top_k")
    default_filters = retriever_cfg.get("filters")
    settings: dict[str, dict[str, Any]] = {}
    for item in raw_sources:
        item_map = _as_mapping(item)
        name = str(item_map.get("name", "") or "").strip()
        if not name:
            continue
        settings[name] = {
            "top_k": item_map.get("top_k", default_top_k),
            "filters": item_map.get("filters", default_filters),
            "weight": item_map.get("weight", 1.0),
        }
    return settings


def _config_base_dir(config_path: Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).expanduser().resolve().parent


def _stable_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_VALIDITY_WEIGHTS_PATH",
    "PresetContext",
    "apply_evaluation_preset",
    "list_preset_names",
    "resolve_condition_summary",
]
