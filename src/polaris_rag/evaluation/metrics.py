"""polaris_rag.evaluation.metrics.

Modern RAGAS metric registry for Polaris evaluation.

This module exposes a small factory/registry API for resolving and building RAGAS 0.4
metric instances from ``ragas.metrics.collections``.

Design goals ------------ - Use modern metric names only. - Keep metric configuration
declarative and explicit. - Auto-gate metrics based on available dataset columns. -
Avoid legacy/deprecated ``ragas.metrics`` import surfaces.

Classes
-------
MetricSpec
    Configuration and constructor for one evaluation metric.

Functions
---------
list_available_metric_names
    Return all metric names known to the registry.
resolve_metric_specs
    Resolve requested metric names into buildable metric specs.
instantiate_metrics
    Instantiate concrete RAGAS metric objects from resolved specs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping

from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecisionWithoutReference,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
    NoiseSensitivity,
    SemanticSimilarity,
    SummaryScore,
)


MetricBuilder = Callable[[Any, Any, Mapping[str, Any] | None], Any]


@dataclass(frozen=True)
class MetricSpec:
    """Configuration and constructor for one evaluation metric.
    
    Attributes
    ----------
    name : str
        Human-readable name for the resource or tracing span.
    required_columns : frozenset[str]
        Value for required Columns.
    builder : MetricBuilder
        Value for builder.
    """

    name: str
    required_columns: frozenset[str]
    builder: MetricBuilder


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping view for optional metric configuration."""

    if isinstance(value, Mapping):
        return value
    return {}


def _string_option(options: Mapping[str, Any], key: str, default: str) -> str:
    """Resolve a string metric option with a stable fallback."""

    value = str(options.get(key, default) or "").strip()
    return value or default


def _build_factual_correctness_metric(
    llm: Any,
    _embeddings: Any,
    options: Mapping[str, Any] | None = None,
) -> Any:
    """Build the factual-correctness metric with optional config overrides."""

    config = _as_mapping(options)
    return FactualCorrectness(
        llm=llm,
        mode=_string_option(config, "mode", "f1"),
        atomicity=_string_option(config, "atomicity", "high"),
        coverage=_string_option(config, "coverage", "high"),
    )


# Default metric set for single-turn RAG evaluation.
DEFAULT_METRIC_ORDER: tuple[str, ...] = (
    "retrieval_recall_at_k",
    "retrieval_ndcg_at_10",
    "context_precision_without_reference",
    "context_recall",
    "context_entity_recall",
    "faithfulness",
    "answer_relevancy",
    "factual_correctness",
    "semantic_similarity",
    "noise_sensitivity",
    "summary_score",
)


METRIC_REGISTRY: dict[str, MetricSpec] = {
    "retrieval_recall_at_k": MetricSpec(
        name="retrieval_recall_at_k",
        required_columns=frozenset({"retrieved_context_ids", "reference_context_ids"}),
        builder=lambda llm, embeddings: _AsyncRetrievalMetric(
            name="retrieval_recall_at_k",
            scorer=_retrieval_recall_at_k,
        ),
    ),
    "retrieval_ndcg_at_10": MetricSpec(
        name="retrieval_ndcg_at_10",
        required_columns=frozenset({"retrieved_context_ids", "reference_context_ids"}),
        builder=lambda llm, embeddings: _AsyncRetrievalMetric(
            name="retrieval_ndcg_at_10",
            scorer=_retrieval_ndcg_at_10,
        ),
    ),
    "context_precision_without_reference": MetricSpec(
        name="context_precision_without_reference",
        required_columns=frozenset({"user_input", "response", "retrieved_contexts"}),
        builder=lambda llm, embeddings, options=None: ContextPrecisionWithoutReference(llm=llm),
    ),
    "context_recall": MetricSpec(
        name="context_recall",
        required_columns=frozenset({"user_input", "retrieved_contexts", "reference"}),
        builder=lambda llm, embeddings, options=None: ContextRecall(llm=llm),
    ),
    "context_entity_recall": MetricSpec(
        name="context_entity_recall",
        required_columns=frozenset({"retrieved_contexts", "reference"}),
        builder=lambda llm, embeddings, options=None: ContextEntityRecall(llm=llm),
    ),
    "faithfulness": MetricSpec(
        name="faithfulness",
        required_columns=frozenset({"user_input", "response", "retrieved_contexts"}),
        builder=lambda llm, embeddings, options=None: Faithfulness(llm=llm),
    ),
    "answer_relevancy": MetricSpec(
        name="answer_relevancy",
        required_columns=frozenset({"user_input", "response"}),
        builder=lambda llm, embeddings, options=None: AnswerRelevancy(llm=llm, embeddings=embeddings),
    ),
    "factual_correctness": MetricSpec(
        name="factual_correctness",
        required_columns=frozenset({"response", "reference"}),
        builder=_build_factual_correctness_metric,
    ),
    "semantic_similarity": MetricSpec(
        name="semantic_similarity",
        required_columns=frozenset({"response", "reference"}),
        builder=lambda llm, embeddings, options=None: SemanticSimilarity(embeddings=embeddings),
    ),
    "noise_sensitivity": MetricSpec(
        name="noise_sensitivity",
        required_columns=frozenset({"user_input", "response", "reference", "retrieved_contexts"}),
        builder=lambda llm, embeddings, options=None: NoiseSensitivity(llm=llm, mode="relevant"),
    ),
    "summary_score": MetricSpec(
        name="summary_score",
        required_columns=frozenset({"response", "reference_contexts"}),
        builder=lambda llm, embeddings, options=None: SummaryScore(llm=llm),
    ),
}


class _AsyncRetrievalMetric:
    """Small async-compatible metric wrapper for retrieval-only scores."""

    def __init__(self, *, name: str, scorer: Callable[..., float | None]) -> None:
        self.name = name
        self._scorer = scorer

    async def ascore(self, **kwargs: Any) -> float | None:
        """Return the metric score for one prepared row."""
        return self._scorer(**kwargs)


def _normalized_context_id_list(value: Any) -> list[str]:
    """Normalize arbitrary context-id payloads into a deduplicated list."""
    if not isinstance(value, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def _retrieval_recall_at_k(**kwargs: Any) -> float | None:
    """Binary recall over the retrieved context id list."""
    retrieved = _normalized_context_id_list(kwargs.get("retrieved_context_ids"))
    reference = _normalized_context_id_list(kwargs.get("reference_context_ids"))
    if not reference:
        return None
    if not retrieved:
        return 0.0
    reference_set = set(reference)
    hits = sum(1 for item in retrieved if item in reference_set)
    return hits / float(len(reference_set))


def _retrieval_ndcg_at_10(**kwargs: Any) -> float | None:
    """nDCG@10 over retrieved and reference context ids."""
    retrieved = _normalized_context_id_list(kwargs.get("retrieved_context_ids"))[:10]
    reference = _normalized_context_id_list(kwargs.get("reference_context_ids"))
    if not reference:
        return None
    reference_set = set(reference)
    if not retrieved:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(retrieved, start=1):
        relevance = 1.0 if item in reference_set else 0.0
        if relevance <= 0.0:
            continue
        dcg += relevance / math.log2(rank + 1.0)

    ideal_hits = min(len(reference_set), 10)
    if ideal_hits <= 0:
        return None
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return None
    return dcg / idcg


def list_available_metric_names() -> list[str]:
    """Return all metric names known to the registry.
    
    Returns
    -------
    list[str]
        Available available Metric Names.
    """

    return sorted(METRIC_REGISTRY.keys())


def resolve_metric_specs(
    *,
    dataset_columns: set[str],
    requested_metrics: Iterable[str] | None,
    auto_gate: bool = True,
) -> tuple[list[MetricSpec], list[tuple[str, str]]]:
    """Resolve requested metric names into buildable metric specs.

    Parameters
    ----------
    dataset_columns : set[str]
        Available columns in the prepared dataset.
    requested_metrics : Iterable[str] or None
        Explicit requested metric names. If None, defaults are used.
    auto_gate : bool, default True
        If True, metrics missing required columns are skipped.

    Returns
    -------
    tuple[list[MetricSpec], list[tuple[str, str]]]
        Resolved metric specs and list of skipped metrics ``(name, reason)``.

    Raises
    ------
    KeyError
        If a requested metric is unknown.
    ValueError
        If no metrics remain after gating.
    """

    names = list(requested_metrics) if requested_metrics is not None else list(DEFAULT_METRIC_ORDER)

    for name in names:
        if name not in METRIC_REGISTRY:
            available = ", ".join(list_available_metric_names())
            raise KeyError(f"Unknown metric '{name}'. Available metrics: {available}")

    resolved: list[MetricSpec] = []
    skipped: list[tuple[str, str]] = []

    for name in names:
        spec = METRIC_REGISTRY[name]
        missing = sorted(spec.required_columns - dataset_columns)

        if auto_gate and missing:
            skipped.append(
                (
                    name,
                    f"missing required columns: {', '.join(missing)}",
                )
            )
            continue

        if not auto_gate and missing:
            raise ValueError(
                f"Metric '{name}' requires missing columns: {', '.join(missing)}"
            )

        resolved.append(spec)

    if not resolved:
        raise ValueError(
            "No metrics available after gating. Ensure the prepared dataset contains required columns."
        )

    return resolved, skipped


def instantiate_metrics(
    specs: list[MetricSpec],
    *,
    llm: Any,
    embeddings: Any,
    metric_config: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Any]:
    """Instantiate concrete RAGAS metric objects from resolved specs.
    
    Parameters
    ----------
    specs : list[MetricSpec]
        Value for specs.
    llm : Any
        Value for LLM.
    embeddings : Any
        Value for embeddings.
    
    Returns
    -------
    list[Any]
        Collected results from the operation.
    """

    config_by_name = _as_mapping(metric_config)
    return [
        spec.builder(llm, embeddings, _as_mapping(config_by_name.get(spec.name)))
        for spec in specs
    ]


__all__ = [
    "DEFAULT_METRIC_ORDER",
    "METRIC_REGISTRY",
    "MetricSpec",
    "instantiate_metrics",
    "list_available_metric_names",
    "resolve_metric_specs",
]
