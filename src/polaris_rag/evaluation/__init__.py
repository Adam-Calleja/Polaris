"""Evaluation utilities for Polaris RAG."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AdaptiveConcurrencySettings": ("polaris_rag.evaluation.evaluator", "AdaptiveConcurrencySettings"),
    "ApiRequester": ("polaris_rag.evaluation.evaluation_dataset", "ApiRequester"),
    "ConcurrencyTrial": ("polaris_rag.evaluation.evaluator", "ConcurrencyTrial"),
    "DEFAULT_METRIC_ORDER": ("polaris_rag.evaluation.metrics", "DEFAULT_METRIC_ORDER"),
    "EvaluationRunResult": ("polaris_rag.evaluation.evaluator", "EvaluationRunResult"),
    "Evaluator": ("polaris_rag.evaluation.evaluator", "Evaluator"),
    "METRIC_REGISTRY": ("polaris_rag.evaluation.metrics", "METRIC_REGISTRY"),
    "MetricSpec": ("polaris_rag.evaluation.metrics", "MetricSpec"),
    "PrepProgressCallback": ("polaris_rag.evaluation.evaluation_dataset", "PrepProgressCallback"),
    "PrepProgressEvent": ("polaris_rag.evaluation.evaluation_dataset", "PrepProgressEvent"),
    "PrepRetryPolicy": ("polaris_rag.evaluation.evaluation_dataset", "PrepRetryPolicy"),
    "build_prepared_rows": ("polaris_rag.evaluation.evaluation_dataset", "build_prepared_rows"),
    "build_prepared_rows_from_api": ("polaris_rag.evaluation.evaluation_dataset", "build_prepared_rows_from_api"),
    "executor_results_to_dataframe": ("polaris_rag.evaluation.evaluator", "executor_results_to_dataframe"),
    "instantiate_metrics": ("polaris_rag.evaluation.metrics", "instantiate_metrics"),
    "list_available_metric_names": ("polaris_rag.evaluation.metrics", "list_available_metric_names"),
    "load_prepared_rows": ("polaris_rag.evaluation.evaluation_dataset", "load_prepared_rows"),
    "load_raw_examples": ("polaris_rag.evaluation.evaluation_dataset", "load_raw_examples"),
    "load_sample_categories": ("polaris_rag.evaluation.evaluation_dataset", "load_sample_categories"),
    "load_sample_ids": ("polaris_rag.evaluation.evaluation_dataset", "load_sample_ids"),
    "persist_prepared_rows": ("polaris_rag.evaluation.evaluation_dataset", "persist_prepared_rows"),
    "resolve_metric_specs": ("polaris_rag.evaluation.metrics", "resolve_metric_specs"),
    "stratified_split_raw_examples_by_categories": ("polaris_rag.evaluation.evaluation_dataset", "stratified_split_raw_examples_by_categories"),
    "split_raw_examples_by_ids": ("polaris_rag.evaluation.evaluation_dataset", "split_raw_examples_by_ids"),
    "to_evaluation_dataset": ("polaris_rag.evaluation.evaluation_dataset", "to_evaluation_dataset"),
    "write_outputs": ("polaris_rag.evaluation.evaluator", "write_outputs"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORTS.keys()))


__all__ = [
    "AdaptiveConcurrencySettings",
    "ApiRequester",
    "ConcurrencyTrial",
    "DEFAULT_METRIC_ORDER",
    "EvaluationRunResult",
    "Evaluator",
    "METRIC_REGISTRY",
    "MetricSpec",
    "PrepProgressCallback",
    "PrepProgressEvent",
    "PrepRetryPolicy",
    "build_prepared_rows",
    "build_prepared_rows_from_api",
    "executor_results_to_dataframe",
    "instantiate_metrics",
    "list_available_metric_names",
    "load_prepared_rows",
    "load_raw_examples",
    "load_sample_categories",
    "load_sample_ids",
    "persist_prepared_rows",
    "resolve_metric_specs",
    "stratified_split_raw_examples_by_categories",
    "split_raw_examples_by_ids",
    "to_evaluation_dataset",
    "write_outputs",
]
