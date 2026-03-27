"""Evaluation utilities for Polaris RAG.

This package groups the public helpers and types that belong to this subsystem of the
Polaris RAG codebase.

See Also
--------
benchmark_analysis
    Related module for benchmark Analysis.
benchmark_annotations
    Related module for benchmark Annotations.
evaluation_dataset
    Related module for evaluation Dataset.
evaluator
    Related module for evaluator.
experiment_presets
    Related module for experiment Presets.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ANALYSIS_LABEL_COLUMNS": ("polaris_rag.evaluation.benchmark_annotations", "ANALYSIS_LABEL_COLUMNS"),
    "ANNOTATION_COLUMNS": ("polaris_rag.evaluation.benchmark_annotations", "ANNOTATION_COLUMNS"),
    "ANNOTATION_METADATA_KEY": ("polaris_rag.evaluation.benchmark_annotations", "ANNOTATION_METADATA_KEY"),
    "AdaptiveConcurrencySettings": ("polaris_rag.evaluation.evaluator", "AdaptiveConcurrencySettings"),
    "ApiRequester": ("polaris_rag.evaluation.evaluation_dataset", "ApiRequester"),
    "AnnotationValidationError": ("polaris_rag.evaluation.benchmark_annotations", "AnnotationValidationError"),
    "ConcurrencyTrial": ("polaris_rag.evaluation.evaluator", "ConcurrencyTrial"),
    "CORE_ANALYSIS_COLUMNS": ("polaris_rag.evaluation.benchmark_annotations", "CORE_ANALYSIS_COLUMNS"),
    "DEFAULT_VALIDITY_WEIGHTS_PATH": ("polaris_rag.evaluation.experiment_presets", "DEFAULT_VALIDITY_WEIGHTS_PATH"),
    "DEFAULT_METRIC_ORDER": ("polaris_rag.evaluation.metrics", "DEFAULT_METRIC_ORDER"),
    "DOCS_SCOPE_VALUES": ("polaris_rag.evaluation.benchmark_annotations", "DOCS_SCOPE_VALUES"),
    "EvaluationRunResult": ("polaris_rag.evaluation.evaluator", "EvaluationRunResult"),
    "Evaluator": ("polaris_rag.evaluation.evaluator", "Evaluator"),
    "LABEL_VALUE_ORDERS": ("polaris_rag.evaluation.benchmark_annotations", "LABEL_VALUE_ORDERS"),
    "METRIC_REGISTRY": ("polaris_rag.evaluation.metrics", "METRIC_REGISTRY"),
    "MetricSpec": ("polaris_rag.evaluation.metrics", "MetricSpec"),
    "PrepProgressCallback": ("polaris_rag.evaluation.evaluation_dataset", "PrepProgressCallback"),
    "PrepProgressEvent": ("polaris_rag.evaluation.evaluation_dataset", "PrepProgressEvent"),
    "PrepRetryPolicy": ("polaris_rag.evaluation.evaluation_dataset", "PrepRetryPolicy"),
    "build_prepared_rows": ("polaris_rag.evaluation.evaluation_dataset", "build_prepared_rows"),
    "build_prepared_rows_from_api": ("polaris_rag.evaluation.evaluation_dataset", "build_prepared_rows_from_api"),
    "build_analysis_rows": ("polaris_rag.evaluation.run_analysis", "build_analysis_rows"),
    "build_split_lookup": ("polaris_rag.evaluation.benchmark_annotations", "build_split_lookup"),
    "build_condition_summary_rows": ("polaris_rag.evaluation.run_analysis", "build_condition_summary_rows"),
    "build_manual_eval_outputs": ("polaris_rag.evaluation.run_analysis", "build_manual_eval_outputs"),
    "build_query_review_rows": ("polaris_rag.evaluation.run_analysis", "build_query_review_rows"),
    "build_source_distribution_rows": ("polaris_rag.evaluation.run_analysis", "build_source_distribution_rows"),
    "build_subgroup_metric_rows": ("polaris_rag.evaluation.run_analysis", "build_subgroup_metric_rows"),
    "executor_results_to_dataframe": ("polaris_rag.evaluation.evaluator", "executor_results_to_dataframe"),
    "instantiate_metrics": ("polaris_rag.evaluation.metrics", "instantiate_metrics"),
    "join_annotations_into_rows": ("polaris_rag.evaluation.benchmark_annotations", "join_annotations_into_rows"),
    "list_preset_names": ("polaris_rag.evaluation.experiment_presets", "list_preset_names"),
    "list_available_metric_names": ("polaris_rag.evaluation.metrics", "list_available_metric_names"),
    "load_analysis_rows": ("polaris_rag.evaluation.run_analysis", "load_analysis_rows"),
    "load_annotation_rows": ("polaris_rag.evaluation.benchmark_annotations", "load_annotation_rows"),
    "load_legacy_audit_labels": ("polaris_rag.evaluation.benchmark_annotations", "load_legacy_audit_labels"),
    "load_prepared_rows": ("polaris_rag.evaluation.evaluation_dataset", "load_prepared_rows"),
    "load_raw_examples": ("polaris_rag.evaluation.evaluation_dataset", "load_raw_examples"),
    "load_run_input": ("polaris_rag.evaluation.run_analysis", "load_run_input"),
    "load_sample_categories": ("polaris_rag.evaluation.evaluation_dataset", "load_sample_categories"),
    "load_sample_ids": ("polaris_rag.evaluation.evaluation_dataset", "load_sample_ids"),
    "persist_annotation_rows": ("polaris_rag.evaluation.benchmark_annotations", "persist_annotation_rows"),
    "persist_analysis_rows": ("polaris_rag.evaluation.run_analysis", "persist_analysis_rows"),
    "persist_prepared_rows": ("polaris_rag.evaluation.evaluation_dataset", "persist_prepared_rows"),
    "PresetContext": ("polaris_rag.evaluation.experiment_presets", "PresetContext"),
    "apply_evaluation_preset": ("polaris_rag.evaluation.experiment_presets", "apply_evaluation_preset"),
    "resolve_condition_summary": ("polaris_rag.evaluation.experiment_presets", "resolve_condition_summary"),
    "RunInput": ("polaris_rag.evaluation.run_analysis", "RunInput"),
    "scaffold_annotation_rows": ("polaris_rag.evaluation.benchmark_annotations", "scaffold_annotation_rows"),
    "resolve_metric_specs": ("polaris_rag.evaluation.metrics", "resolve_metric_specs"),
    "stratified_split_raw_examples_by_categories": ("polaris_rag.evaluation.evaluation_dataset", "stratified_split_raw_examples_by_categories"),
    "stratified_split_raw_examples_by_annotation_labels": ("polaris_rag.evaluation.evaluation_dataset", "stratified_split_raw_examples_by_annotation_labels"),
    "split_raw_examples_by_ids": ("polaris_rag.evaluation.evaluation_dataset", "split_raw_examples_by_ids"),
    "to_evaluation_dataset": ("polaris_rag.evaluation.evaluation_dataset", "to_evaluation_dataset"),
    "validate_annotation_rows": ("polaris_rag.evaluation.benchmark_annotations", "validate_annotation_rows"),
    "write_analysis_outputs": ("polaris_rag.evaluation.benchmark_analysis", "write_analysis_outputs"),
    "write_run_comparison_outputs": ("polaris_rag.evaluation.run_analysis", "write_run_comparison_outputs"),
    "write_outputs": ("polaris_rag.evaluation.evaluator", "write_outputs"),
}


def __getattr__(name: str) -> Any:
    """Resolve lazily exposed attributes.
    
    Parameters
    ----------
    name : str
        Human-readable name for the resource or tracing span.
    
    Returns
    -------
    Any
        Result of the operation.
    
    Raises
    ------
    AttributeError
        If the requested attribute cannot be resolved.
    """
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return available attribute names for interactive discovery.
    
    Returns
    -------
    list[str]
        Available attribute names for the object or module.
    """
    return sorted(set(globals().keys()) | set(_EXPORTS.keys()))


__all__ = [
    "ANALYSIS_LABEL_COLUMNS",
    "ANNOTATION_COLUMNS",
    "ANNOTATION_METADATA_KEY",
    "AdaptiveConcurrencySettings",
    "ApiRequester",
    "AnnotationValidationError",
    "ConcurrencyTrial",
    "CORE_ANALYSIS_COLUMNS",
    "DEFAULT_VALIDITY_WEIGHTS_PATH",
    "DEFAULT_METRIC_ORDER",
    "DOCS_SCOPE_VALUES",
    "EvaluationRunResult",
    "Evaluator",
    "LABEL_VALUE_ORDERS",
    "METRIC_REGISTRY",
    "MetricSpec",
    "PrepProgressCallback",
    "PrepProgressEvent",
    "PrepRetryPolicy",
    "PresetContext",
    "RunInput",
    "apply_evaluation_preset",
    "build_analysis_rows",
    "build_condition_summary_rows",
    "build_manual_eval_outputs",
    "build_prepared_rows",
    "build_prepared_rows_from_api",
    "build_query_review_rows",
    "build_split_lookup",
    "build_source_distribution_rows",
    "build_subgroup_metric_rows",
    "executor_results_to_dataframe",
    "instantiate_metrics",
    "join_annotations_into_rows",
    "list_preset_names",
    "list_available_metric_names",
    "load_analysis_rows",
    "load_annotation_rows",
    "load_legacy_audit_labels",
    "load_prepared_rows",
    "load_raw_examples",
    "load_run_input",
    "load_sample_categories",
    "load_sample_ids",
    "persist_annotation_rows",
    "persist_analysis_rows",
    "persist_prepared_rows",
    "resolve_condition_summary",
    "scaffold_annotation_rows",
    "resolve_metric_specs",
    "stratified_split_raw_examples_by_categories",
    "stratified_split_raw_examples_by_annotation_labels",
    "split_raw_examples_by_ids",
    "to_evaluation_dataset",
    "validate_annotation_rows",
    "write_analysis_outputs",
    "write_run_comparison_outputs",
    "write_outputs",
]
