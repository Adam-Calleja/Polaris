from __future__ import annotations

from pathlib import Path

import pytest

from polaris_rag.config import GlobalConfig
from polaris_rag.evaluation import experiment_presets


class _FakeReranker:
    def __init__(self, config):  # noqa: ANN001
        self._config = dict(config)

    def profile(self) -> dict[str, object]:
        return {
            "type": self._config.get("type"),
            "weights": dict(self._config.get("weights", {})),
        }


def _base_cfg(tmp_path: Path, *, include_external: bool = False) -> GlobalConfig:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("retriever: {}\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[4]
    sources = [
        {"name": "docs", "weight": 1.0},
        {"name": "tickets", "weight": 1.0},
    ]
    vector_stores = {
        "docs": {"collection_name": "docs_collection", "authority_scope": "local_official"},
        "tickets": {"collection_name": "tickets_collection", "authority_scope": "ticket_memory"},
    }
    if include_external:
        vector_stores["external_docs"] = {
            "collection_name": "external_collection",
            "authority_scope": "external_official",
        }
    return GlobalConfig(
        raw={
            "retriever": {
                "type": "multi_collection",
                "source_type": "hybrid",
                "final_top_k": 8,
                "sources": sources,
                "rerank": {
                    "type": "validity_aware",
                    "semantic_base": {"type": "rrf", "rrf_k": 60},
                    "weights_path": str(repo_root / "config/weights/validity_reranker.dev_v3.yaml"),
                },
            },
            "vector_stores": vector_stores,
        },
        config_path=config_path,
    )


def test_apply_evaluation_preset_docs_only_uses_docs_rrf(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(experiment_presets, "create_reranker", lambda **kwargs: _FakeReranker(kwargs["config"]))

    cfg, context = experiment_presets.apply_evaluation_preset(_base_cfg(tmp_path), "docs_only")

    assert [source["name"] for source in cfg.raw["retriever"]["sources"]] == ["docs"]
    assert cfg.raw["retriever"]["rerank"]["type"] == "rrf"
    assert cfg.raw["retriever"]["rerank"]["trace_enabled"] is True
    assert context.preset_name == "docs_only"
    assert context.condition_summary["sources"][0]["collection_name"] == "docs_collection"


def test_apply_evaluation_preset_source_aware_zeros_non_authority_weights(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(experiment_presets, "create_reranker", lambda **kwargs: _FakeReranker(kwargs["config"]))

    cfg, context = experiment_presets.apply_evaluation_preset(_base_cfg(tmp_path), "source_aware")

    weights = cfg.raw["retriever"]["rerank"]["weights"]
    assert [source["name"] for source in cfg.raw["retriever"]["sources"]] == ["docs", "tickets"]
    assert cfg.raw["retriever"]["rerank"]["type"] == "validity_aware"
    assert weights["authority"] > 0.0
    assert all(
        weights[key] == 0.0
        for key in experiment_presets.FEATURE_WEIGHT_KEYS
        if key != "authority"
    )
    assert context.condition_fingerprint


def test_apply_evaluation_preset_requires_external_source_for_all_docs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(experiment_presets, "create_reranker", lambda **kwargs: _FakeReranker(kwargs["config"]))

    with pytest.raises(ValueError, match="requires an external official docs retriever source"):
        experiment_presets.apply_evaluation_preset(_base_cfg(tmp_path, include_external=False), "all_docs_validity_aware")


def test_apply_evaluation_preset_all_docs_includes_external_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(experiment_presets, "create_reranker", lambda **kwargs: _FakeReranker(kwargs["config"]))

    cfg, context = experiment_presets.apply_evaluation_preset(
        _base_cfg(tmp_path, include_external=True),
        "all_docs_validity_aware",
    )

    assert [source["name"] for source in cfg.raw["retriever"]["sources"]] == [
        "docs",
        "external_docs",
        "tickets",
    ]
    assert context.condition_summary["sources"][1]["collection_name"] == "external_collection"


def test_apply_evaluation_preset_requires_explicit_external_scope_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(experiment_presets, "create_reranker", lambda **kwargs: _FakeReranker(kwargs["config"]))
    cfg = _base_cfg(tmp_path, include_external=True)
    cfg.raw["vector_stores"]["external_docs"]["authority_scope"] = "unknown"

    with pytest.raises(ValueError, match="authority_scope=external_official"):
        experiment_presets.apply_evaluation_preset(cfg, "all_docs_validity_aware")


def test_apply_evaluation_preset_hybrid_rrf_sets_hybrid_source_type(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(experiment_presets, "create_reranker", lambda **kwargs: _FakeReranker(kwargs["config"]))
    cfg = _base_cfg(tmp_path)
    cfg.raw["retriever"]["hybrid_profile"] = {
        "dense_top_k": 30,
        "sparse_top_k": 30,
        "top_k": 20,
        "fusion": {"type": "rrf", "rrf_k": 60},
    }

    resolved, context = experiment_presets.apply_evaluation_preset(cfg, "hybrid_rrf")

    assert resolved.raw["retriever"]["source_type"] == "hybrid"
    assert context.condition_summary["hybrid_profile"]["fusion"]["rrf_k"] == 60
    assert context.condition_summary["retriever_profile"]["source_type"] == "hybrid"
    assert context.condition_summary["retriever_fingerprint"]


def test_apply_evaluation_preset_sparse_only_sets_sparse_source_type(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(experiment_presets, "create_reranker", lambda **kwargs: _FakeReranker(kwargs["config"]))

    resolved, context = experiment_presets.apply_evaluation_preset(_base_cfg(tmp_path), "sparse_only")

    assert resolved.raw["retriever"]["source_type"] == "sparse"
    assert context.condition_summary["source_type"] == "sparse"


def test_condition_fingerprint_changes_between_dense_and_hybrid(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(experiment_presets, "create_reranker", lambda **kwargs: _FakeReranker(kwargs["config"]))

    _, dense_context = experiment_presets.apply_evaluation_preset(_base_cfg(tmp_path), "dense_only")
    _, hybrid_context = experiment_presets.apply_evaluation_preset(_base_cfg(tmp_path), "hybrid_rrf")

    assert dense_context.condition_fingerprint != hybrid_context.condition_fingerprint
