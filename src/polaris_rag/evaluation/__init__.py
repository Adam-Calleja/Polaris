"""Evaluation utilities for Polaris RAG."""

from polaris_rag.evaluation.evaluation_dataset import (
    build_prepared_rows,
    load_prepared_rows,
    load_raw_examples,
    persist_prepared_rows,
    to_evaluation_dataset,
)
from polaris_rag.evaluation.evaluator import (
    AdaptiveConcurrencySettings,
    ConcurrencyTrial,
    EvaluationRunResult,
    Evaluator,
    executor_results_to_dataframe,
    write_outputs,
)
from polaris_rag.evaluation.metrics import (
    DEFAULT_METRIC_ORDER,
    METRIC_REGISTRY,
    MetricSpec,
    instantiate_metrics,
    list_available_metric_names,
    resolve_metric_specs,
)

__all__ = [
    "AdaptiveConcurrencySettings",
    "ConcurrencyTrial",
    "DEFAULT_METRIC_ORDER",
    "EvaluationRunResult",
    "Evaluator",
    "METRIC_REGISTRY",
    "MetricSpec",
    "build_prepared_rows",
    "executor_results_to_dataframe",
    "instantiate_metrics",
    "list_available_metric_names",
    "load_prepared_rows",
    "load_raw_examples",
    "persist_prepared_rows",
    "resolve_metric_specs",
    "to_evaluation_dataset",
    "write_outputs",
]
