"""polaris_rag.retrieval.reranker

Reranker abstractions and implementations for multi-source retrieval.

This module defines:
- a normalized merged-candidate container
- an abstract reranker interface
- a concrete reciprocal-rank-fusion (RRF) reranker
- a small reranker factory for configuration-driven construction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from llama_index.core.schema import NodeWithScore


@dataclass
class MergedCandidate:
    """Merged retrieval candidate across one or more sources."""

    node: Any
    best_score: float | None
    source_ranks: dict[str, int]


class BaseReranker(ABC):
    """Abstract interface for reranking merged multi-source candidates."""

    @abstractmethod
    def rerank(self, candidates: Sequence[MergedCandidate]) -> list[NodeWithScore]:
        """Return reranked candidates as ``NodeWithScore`` items."""
        raise NotImplementedError


class RRFReranker(BaseReranker):
    """Reciprocal-rank-fusion reranker."""

    def __init__(
            self,
            *,
            rrf_k: int = 60,
            source_weights: Mapping[str, float] | None = None,
        ):
        self.rrf_k = int(rrf_k)
        if self.rrf_k <= 0:
            raise ValueError("'rerank.rrf_k' must be a positive integer.")
        self.source_weights = dict(source_weights or {})

    def rerank(self, candidates: Sequence[MergedCandidate]) -> list[NodeWithScore]:
        """Rerank merged candidates using reciprocal-rank fusion."""
        scored: list[tuple[float, float, NodeWithScore]] = []

        for candidate in candidates:
            rrf_score = 0.0
            for source_name, rank in candidate.source_ranks.items():
                weight = self._source_weight(source_name)
                rrf_score += weight / (self.rrf_k + int(rank))

            tie_break = candidate.best_score if candidate.best_score is not None else float("-inf")
            scored.append(
                (
                    rrf_score,
                    tie_break,
                    NodeWithScore(node=candidate.node, score=rrf_score),
                )
            )

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored]

    def _source_weight(self, source_name: str) -> float:
        weight_raw = self.source_weights.get(source_name, 1.0)
        try:
            return float(weight_raw)
        except Exception:
            return 1.0


def create_reranker(
        *,
        config: Mapping[str, Any] | None,
        source_settings: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> BaseReranker:
    """Create a reranker from configuration."""
    cfg = dict(config or {})
    kind = str(cfg.get("type", "rrf")).lower().strip()

    if kind == "rrf":
        source_weights: dict[str, float] = {}
        for source_name, source_cfg in (source_settings or {}).items():
            weight_raw = (source_cfg or {}).get("weight", 1.0)
            try:
                source_weights[str(source_name)] = float(weight_raw)
            except Exception:
                source_weights[str(source_name)] = 1.0
        return RRFReranker(rrf_k=int(cfg.get("rrf_k", 60)), source_weights=source_weights)

    raise ValueError(f"Unsupported rerank type {kind!r}. Supported rerankers: ['rrf'].")


__all__ = [
    "BaseReranker",
    "MergedCandidate",
    "RRFReranker",
    "create_reranker",
]
