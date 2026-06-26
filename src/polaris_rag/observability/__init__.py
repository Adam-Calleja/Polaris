"""Observability helpers for Polaris.

This package groups the public helpers and types that belong to this subsystem of the
Polaris RAG codebase.

See Also
--------
mlflow_tracking
    Related module for mlflow Tracking.
"""

from .mlflow_tracking import (
    TRACE_CHILD_RUN_HEADER,
    TRACE_PARENT_RUN_HEADER,
    TRACE_STAGE_HEADER,
    EvaluationTrackingContext,
    EvaluationStageContext,
    MLflowRuntimeConfig,
    PromptRegistryRuntimeConfig,
    TraceRuntimeConfig,
    apply_mlflow_overrides,
    build_environment_snapshot,
    configure_mlflow_runtime,
    flatten_for_logging,
    load_mlflow_runtime_config,
    set_span_attributes,
    start_detached_span,
    set_span_inputs,
    set_span_outputs,
    start_span,
    update_current_trace,
)

__all__ = [
    "TRACE_CHILD_RUN_HEADER",
    "TRACE_PARENT_RUN_HEADER",
    "TRACE_STAGE_HEADER",
    "EvaluationTrackingContext",
    "EvaluationStageContext",
    "MLflowRuntimeConfig",
    "PromptRegistryRuntimeConfig",
    "TraceRuntimeConfig",
    "apply_mlflow_overrides",
    "build_environment_snapshot",
    "configure_mlflow_runtime",
    "flatten_for_logging",
    "load_mlflow_runtime_config",
    "set_span_attributes",
    "start_detached_span",
    "set_span_inputs",
    "set_span_outputs",
    "start_span",
    "update_current_trace",
]
