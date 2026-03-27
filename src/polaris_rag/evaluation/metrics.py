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
from typing import Any, Callable, Iterable

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


MetricBuilder = Callable[[Any, Any], Any]


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


# Default metric set for single-turn RAG evaluation.
DEFAULT_METRIC_ORDER: tuple[str, ...] = (
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
    "context_precision_without_reference": MetricSpec(
        name="context_precision_without_reference",
        required_columns=frozenset({"user_input", "response", "retrieved_contexts"}),
        builder=lambda llm, embeddings: ContextPrecisionWithoutReference(llm=llm),
    ),
    "context_recall": MetricSpec(
        name="context_recall",
        required_columns=frozenset({"user_input", "retrieved_contexts", "reference"}),
        builder=lambda llm, embeddings: ContextRecall(llm=llm),
    ),
    "context_entity_recall": MetricSpec(
        name="context_entity_recall",
        required_columns=frozenset({"retrieved_contexts", "reference"}),
        builder=lambda llm, embeddings: ContextEntityRecall(llm=llm),
    ),
    "faithfulness": MetricSpec(
        name="faithfulness",
        required_columns=frozenset({"user_input", "response", "retrieved_contexts"}),
        builder=lambda llm, embeddings: Faithfulness(llm=llm),
    ),
    "answer_relevancy": MetricSpec(
        name="answer_relevancy",
        required_columns=frozenset({"user_input", "response"}),
        builder=lambda llm, embeddings: AnswerRelevancy(llm=llm, embeddings=embeddings),
    ),
    "factual_correctness": MetricSpec(
        name="factual_correctness",
        required_columns=frozenset({"response", "reference"}),
        builder=lambda llm, embeddings: FactualCorrectness(
            llm=llm,
            mode="f1",
            atomicity="high",
            coverage="high",
        ),
    ),
    "semantic_similarity": MetricSpec(
        name="semantic_similarity",
        required_columns=frozenset({"response", "reference"}),
        builder=lambda llm, embeddings: SemanticSimilarity(embeddings=embeddings),
    ),
    "noise_sensitivity": MetricSpec(
        name="noise_sensitivity",
        required_columns=frozenset({"user_input", "response", "reference", "retrieved_contexts"}),
        builder=lambda llm, embeddings: NoiseSensitivity(llm=llm, mode="relevant"),
    ),
    "summary_score": MetricSpec(
        name="summary_score",
        required_columns=frozenset({"response", "reference_contexts"}),
        builder=lambda llm, embeddings: SummaryScore(llm=llm),
    ),
}


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


def instantiate_metrics(specs: list[MetricSpec], *, llm: Any, embeddings: Any) -> list[Any]:
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

    return [spec.builder(llm, embeddings) for spec in specs]


__all__ = [
    "DEFAULT_METRIC_ORDER",
    "METRIC_REGISTRY",
    "MetricSpec",
    "instantiate_metrics",
    "list_available_metric_names",
    "resolve_metric_specs",
]
