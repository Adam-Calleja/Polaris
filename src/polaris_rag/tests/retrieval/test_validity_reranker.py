from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

pytest.importorskip("llama_index.core.schema")
schema = pytest.importorskip("llama_index.core.schema")

from polaris_rag.retrieval.reranker import MergedCandidate, create_reranker

TextNode = schema.TextNode


class _Node:
    def __init__(self, node_id: str, metadata: dict[str, object]) -> None:
        self.node = TextNode(id_=node_id, text=f"text::{node_id}", metadata=dict(metadata))

    @property
    def id_(self) -> str:
        return self.node.id_

    @property
    def metadata(self) -> dict[str, object]:
        return self.node.metadata


def test_validity_aware_reranker_promotes_exact_local_official_match() -> None:
    reranker = create_reranker(
        config={
            "type": "validity_aware",
            "trace_enabled": True,
            "semantic_base": {"type": "rrf", "rrf_k": 60},
            "weights": {
                "authority": 0.08,
                "scope": 0.08,
                "software": 0.08,
                "scope_family": 0.02,
                "version": 0.08,
                "status": 0.08,
                "freshness": 0.01,
            },
        },
        source_settings={
            "docs": {"weight": 1.0},
            "tickets": {"weight": 1.0},
        },
    )

    official = _Node(
        "official-doc",
        {
            "source_authority": "local_official",
            "system_names": ["cclake"],
            "software_names": ["GROMACS"],
            "software_versions": ["2024.4"],
            "validity_status": "current",
        },
    )
    ticket = _Node(
        "ticket-note",
        {
            "source_authority": "ticket_memory",
            "system_names": ["icelake"],
            "software_names": ["LAMMPS"],
            "software_versions": ["2021.3"],
            "validity_status": "current",
            "freshness_hint": "2025-01-01T11:00:00.000+0000",
        },
    )

    candidates = [
        MergedCandidate(node=official.node, best_score=0.8, source_ranks={"docs": 1}),
        MergedCandidate(node=ticket.node, best_score=0.8, source_ranks={"tickets": 1}),
    ]
    results = reranker.rerank(
        candidates,
        query_constraints={
            "query_type": "software_version",
            "system_names": ["cclake"],
            "partition_names": [],
            "service_names": [],
            "scope_family_names": ["cclake"],
            "software_names": ["GROMACS"],
            "software_versions": ["2024.4"],
            "module_names": [],
            "toolchain_names": [],
            "toolchain_versions": [],
            "scope_required": True,
            "version_sensitive_guess": True,
        },
    )

    assert [item.node.id_ for item in results] == ["official-doc", "ticket-note"]
    trace = official.metadata.get("rerank_trace")
    assert isinstance(trace, dict)
    assert trace["reranker_type"] == "validity_aware"
    assert trace["scope_feature"] == 1.0
    assert trace["scope_family_feature"] == 0.0
    assert trace["scope_family_effective_feature"] == 0.0
    assert trace["scope_family_gate_reason"] == "applied"
    assert trace["software_feature"] == 1.0
    assert trace["version_feature"] == 1.0
    assert trace["authority_feature"] == 1.0
    assert trace["final_score"] > results[1].score


def test_validity_aware_reranker_gates_positive_family_boost_without_software_match() -> None:
    reranker = create_reranker(
        config={
            "type": "validity_aware",
            "trace_enabled": True,
            "semantic_base": {"type": "rrf", "rrf_k": 60},
            "weights": {
                "authority": 0.0,
                "scope": 0.0,
                "software": 0.08,
                "scope_family": 0.04,
                "version": 0.0,
                "status": 0.0,
                "freshness": 0.0,
            },
        },
        source_settings={
            "docs": {"weight": 1.0},
            "tickets": {"weight": 1.0},
        },
    )

    family_match = _Node(
        "family-doc",
        {
            "source_authority": "local_official",
            "software_names": ["GROMACS"],
            "scope_family_names": ["cclake"],
        },
    )
    family_irrelevant = _Node(
        "family-ticket",
        {
            "source_authority": "ticket_memory",
            "scope_family_names": ["cclake"],
        },
    )

    candidates = [
        MergedCandidate(node=family_match.node, best_score=0.8, source_ranks={"docs": 1}),
        MergedCandidate(node=family_irrelevant.node, best_score=0.8, source_ranks={"tickets": 1}),
    ]
    results = reranker.rerank(
        candidates,
        query_constraints={
            "query_type": "software_version",
            "system_names": [],
            "partition_names": [],
            "service_names": [],
            "scope_family_names": ["cclake"],
            "software_names": ["GROMACS"],
            "software_versions": ["2024.4"],
            "module_names": [],
            "toolchain_names": [],
            "toolchain_versions": [],
            "scope_required": True,
            "version_sensitive_guess": None,
        },
    )

    assert [item.node.id_ for item in results] == ["family-doc", "family-ticket"]
    trace = family_match.metadata.get("rerank_trace")
    assert isinstance(trace, dict)
    assert trace["scope_feature"] == 0.0
    assert trace["scope_family_feature"] == 1.0
    assert trace["scope_family_effective_feature"] == 1.0
    blocked_trace = family_irrelevant.metadata.get("rerank_trace")
    assert isinstance(blocked_trace, dict)
    assert blocked_trace["scope_family_feature"] == 1.0
    assert blocked_trace["scope_family_effective_feature"] == 0.0
    assert blocked_trace["scope_family_gate_reason"] == "blocked_no_software_match"


def test_validity_aware_reranker_uses_scope_family_for_non_software_queries() -> None:
    reranker = create_reranker(
        config={
            "type": "validity_aware",
            "trace_enabled": True,
            "semantic_base": {"type": "rrf", "rrf_k": 60},
            "weights": {
                "authority": 0.0,
                "scope": 0.0,
                "software": 0.0,
                "scope_family": 0.04,
                "version": 0.0,
                "status": 0.0,
                "freshness": 0.0,
            },
        },
        source_settings={
            "docs": {"weight": 1.0},
            "tickets": {"weight": 1.0},
        },
    )

    family_match = _Node(
        "family-doc",
        {
            "source_authority": "local_official",
            "scope_family_names": ["cclake"],
        },
    )
    family_mismatch = _Node(
        "wrong-family-doc",
        {
            "source_authority": "local_official",
            "scope_family_names": ["icelake"],
        },
    )

    candidates = [
        MergedCandidate(node=family_match.node, best_score=0.8, source_ranks={"docs": 1}),
        MergedCandidate(node=family_mismatch.node, best_score=0.8, source_ranks={"tickets": 1}),
    ]
    results = reranker.rerank(
        candidates,
        query_constraints={
            "query_type": "local_operational",
            "system_names": [],
            "partition_names": [],
            "service_names": [],
            "scope_family_names": ["cclake"],
            "software_names": [],
            "software_versions": [],
            "module_names": [],
            "toolchain_names": [],
            "toolchain_versions": [],
            "scope_required": True,
            "version_sensitive_guess": None,
        },
    )

    assert [item.node.id_ for item in results] == ["family-doc", "wrong-family-doc"]
    trace = family_match.metadata.get("rerank_trace")
    assert isinstance(trace, dict)
    assert trace["scope_family_feature"] == 1.0
    assert trace["scope_family_effective_feature"] == 1.0
    assert trace["scope_family_gate_reason"] == "applied"


def test_validity_aware_reranker_blocks_specialized_variant_family_boost_without_exact_variant() -> None:
    reranker = create_reranker(
        config={
            "type": "validity_aware",
            "trace_enabled": True,
            "semantic_base": {"type": "rrf", "rrf_k": 60},
            "weights": {
                "authority": 0.0,
                "scope": 0.0,
                "software": 0.08,
                "scope_family": 0.04,
                "version": 0.0,
                "status": 0.0,
                "freshness": 0.0,
            },
        },
        source_settings={
            "docs": {"weight": 1.0},
        },
    )

    specialized = _Node(
        "cclake-himem-doc",
        {
            "source_authority": "local_official",
            "software_names": ["GROMACS"],
            "scope_family_names": ["cclake"],
            "partition_names": ["cclake-himem"],
        },
    )

    generic = _Node(
        "cclake-doc",
        {
            "source_authority": "local_official",
            "software_names": ["GROMACS"],
            "scope_family_names": ["cclake"],
        },
    )

    results = reranker.rerank(
        [
            MergedCandidate(node=specialized.node, best_score=0.8, source_ranks={"docs": 2}),
            MergedCandidate(node=generic.node, best_score=0.8, source_ranks={"docs": 1}),
        ],
        query_constraints={
            "query_type": "software_version",
            "system_names": [],
            "partition_names": [],
            "service_names": [],
            "scope_family_names": ["cclake"],
            "software_names": ["GROMACS"],
            "software_versions": ["2024.4"],
            "module_names": [],
            "toolchain_names": [],
            "toolchain_versions": [],
            "scope_required": True,
            "version_sensitive_guess": True,
        },
    )

    assert [item.node.id_ for item in results] == ["cclake-doc", "cclake-himem-doc"]
    trace = specialized.metadata.get("rerank_trace")
    assert isinstance(trace, dict)
    assert trace["scope_family_feature"] == 1.0
    assert trace["scope_family_effective_feature"] == 0.0
    assert trace["scope_family_gate_reason"] == "blocked_specialized_variant"


def test_create_validity_reranker_resolves_relative_weights_path(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.yaml"
    weights_path.write_text(
        "\n".join(
            [
                "weights:",
                "  authority: 0.08",
                "  scope: 0.04",
                "  software: 0.08",
                "  scope_family: 0.02",
                "  version: 0.04",
                "  status: 0.04",
                "  freshness: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    reranker = create_reranker(
        config={
            "type": "validity_aware",
            "weights_path": "weights.yaml",
            "semantic_base": {"type": "rrf", "rrf_k": 60},
        },
        source_settings={"docs": {"weight": 1.0}},
        config_base_dir=tmp_path,
    )

    profile = reranker.profile()
    assert profile["weights"]["authority"] == 0.08
    assert profile["weights"]["software"] == 0.08
    assert profile["weights"]["scope_family"] == 0.02
    assert profile["weights_source"]["type"] == "file"
    assert profile["weights_source"]["path"] == str(weights_path.resolve())
    assert reranker.fingerprint()
