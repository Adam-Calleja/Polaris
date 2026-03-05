"""Evaluation utilities for Polaris RAG."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ApiRequester": ("polaris_rag.evaluation.evaluation_dataset", "ApiRequester"),
    "PrepProgressCallback": ("polaris_rag.evaluation.evaluation_dataset", "PrepProgressCallback"),
    "PrepProgressEvent": ("polaris_rag.evaluation.evaluation_dataset", "PrepProgressEvent"),
    "build_prepared_rows": ("polaris_rag.evaluation.evaluation_dataset", "build_prepared_rows"),
    "build_prepared_rows_from_api": ("polaris_rag.evaluation.evaluation_dataset", "build_prepared_rows_from_api"),
    "load_prepared_rows": ("polaris_rag.evaluation.evaluation_dataset", "load_prepared_rows"),
    "load_raw_examples": ("polaris_rag.evaluation.evaluation_dataset", "load_raw_examples"),
    "persist_prepared_rows": ("polaris_rag.evaluation.evaluation_dataset", "persist_prepared_rows"),
    "to_evaluation_dataset": ("polaris_rag.evaluation.evaluation_dataset", "to_evaluation_dataset"),
    "AdaptiveConcurrencySettings": ("polaris_rag.evaluation.evaluator", "AdaptiveConcurrencySettings"),
    "ConcurrencyTrial": ("polaris_rag.evaluation.evaluator", "ConcurrencyTrial"),
    "EvaluationRunResult": ("polaris_rag.evaluation.evaluator", "EvaluationRunResult"),
    "Evaluator": ("polaris_rag.evaluation.evaluator", "Evaluator"),
    "executor_results_to_dataframe": ("polaris_rag.evaluation.evaluator", "executor_results_to_dataframe"),
    "write_outputs": ("polaris_rag.evaluation.evaluator", "write_outputs"),
    "DEFAULT_METRIC_ORDER": ("polaris_rag.evaluation.metrics", "DEFAULT_METRIC_ORDER"),
    "METRIC_REGISTRY": ("polaris_rag.evaluation.metrics", "METRIC_REGISTRY"),
    "MetricSpec": ("polaris_rag.evaluation.metrics", "MetricSpec"),
    "instantiate_metrics": ("polaris_rag.evaluation.metrics", "instantiate_metrics"),
    "list_available_metric_names": ("polaris_rag.evaluation.metrics", "list_available_metric_names"),
    "resolve_metric_specs": ("polaris_rag.evaluation.metrics", "resolve_metric_specs"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = export
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
