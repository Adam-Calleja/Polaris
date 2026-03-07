"""MLflow tracking and tracing helpers for Polaris."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterator, Mapping

logger = logging.getLogger(__name__)

TRACE_PARENT_RUN_HEADER = "X-Polaris-MLflow-Run-ID"
TRACE_CHILD_RUN_HEADER = "X-Polaris-MLflow-Child-Run-ID"
TRACE_STAGE_HEADER = "X-Polaris-MLflow-Stage"

_TRACING_ENABLED = False


@dataclass(frozen=True)
class TraceRuntimeConfig:
    """Tracing-specific MLflow runtime options."""

    enabled: bool = False
    destination_experiment: str | None = None


@dataclass(frozen=True)
class PromptRegistryRuntimeConfig:
    """Prompt-registry-specific MLflow runtime options."""

    enabled: bool = False
    name: str | None = None
    alias: str = "prod"


@dataclass(frozen=True)
class MLflowRuntimeConfig:
    """Resolved MLflow runtime configuration."""

    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str = "polaris-rag"
    tags: dict[str, str] | None = None
    tracing: TraceRuntimeConfig = TraceRuntimeConfig()
    prompt_registry: PromptRegistryRuntimeConfig = PromptRegistryRuntimeConfig()


class _NoopSpan:
    """Best-effort no-op span for environments without MLflow tracing."""

    def set_inputs(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_outputs(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_attributes(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def end(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _SpanGroup:
    """Fan-out span helper for mirrored tracing."""

    def __init__(self, spans: list[Any]):
        self._spans = [span for span in spans if span is not None]

    def set_inputs(self, inputs: Any) -> None:
        for span in self._spans:
            set_span_inputs(span, inputs)

    def set_outputs(self, outputs: Any) -> None:
        for span in self._spans:
            set_span_outputs(span, outputs)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for span in self._spans:
            set_span_attributes(span, attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        for span in self._spans:
            set_span_attributes(span, {key: value})


@dataclass(frozen=True)
class EvaluationStageContext:
    """Tracing and correlation helpers for one evaluation stage."""

    name: str
    parent_run_id: str | None
    child_run_id: str | None
    stage_root_span: Any | None = None
    aggregate_stage_span: Any | None = None

    def correlation_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.parent_run_id:
            headers[TRACE_PARENT_RUN_HEADER] = self.parent_run_id
        if self.child_run_id:
            headers[TRACE_CHILD_RUN_HEADER] = self.child_run_id
        headers[TRACE_STAGE_HEADER] = self.name
        return headers

    @contextlib.contextmanager
    def open_active_mirrored_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        inputs: Any | None = None,
        outputs: Any | None = None,
    ) -> Iterator[_SpanGroup]:
        """Open a child span in the active child trace and mirror it to the parent trace."""

        with contextlib.ExitStack() as stack:
            child_span = stack.enter_context(
                start_span(
                    name,
                    attributes=attributes,
                    inputs=inputs,
                    outputs=outputs,
                )
            )
            aggregate_span = stack.enter_context(
                start_detached_span(
                    name,
                    parent_span=self.aggregate_stage_span,
                    attributes=attributes,
                    inputs=inputs,
                    outputs=outputs,
                )
            )
            yield _SpanGroup([child_span, aggregate_span])

    @contextlib.contextmanager
    def open_detached_mirrored_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        inputs: Any | None = None,
        outputs: Any | None = None,
        include_child_trace: bool = True,
    ) -> Iterator[_SpanGroup]:
        """Open mirrored detached spans for worker-thread or cross-cutting events."""

        with contextlib.ExitStack() as stack:
            child_span = None
            if include_child_trace:
                child_span = stack.enter_context(
                    start_detached_span(
                        name,
                        parent_span=self.stage_root_span,
                        attributes=attributes,
                        inputs=inputs,
                        outputs=outputs,
                    )
                )
            aggregate_span = stack.enter_context(
                start_detached_span(
                    name,
                    parent_span=self.aggregate_stage_span,
                    attributes=attributes,
                    inputs=inputs,
                    outputs=outputs,
                )
            )
            yield _SpanGroup([child_span, aggregate_span])


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _import_mlflow() -> Any | None:
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


def _filter_supported_kwargs(func: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if v is not None}

    allowed = set(sig.parameters.keys())
    return {
        k: v
        for k, v in kwargs.items()
        if k in allowed and v is not None
    }


def _call_if_available(obj: Any, method_name: str, *args: Any, **kwargs: Any) -> Any | None:
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    filtered = _filter_supported_kwargs(method, kwargs)
    try:
        return method(*args, **filtered)
    except TypeError:
        return method(*args)


def load_mlflow_runtime_config(cfg: Any) -> MLflowRuntimeConfig:
    """Resolve runtime MLflow configuration from global config/raw mapping."""

    raw = _as_mapping(getattr(cfg, "raw", cfg))
    mlflow_cfg = _as_mapping(raw.get("mlflow", {}))

    tracing_cfg = _as_mapping(mlflow_cfg.get("tracing", {}))
    prompt_registry_cfg = _as_mapping(mlflow_cfg.get("prompt_registry", {}))

    tags_raw = _as_mapping(mlflow_cfg.get("tags", {}))
    tags = {str(k): str(v) for k, v in tags_raw.items()}

    tracking_uri_raw = mlflow_cfg.get("tracking_uri")
    tracking_uri = str(tracking_uri_raw).strip() if tracking_uri_raw else None
    if tracking_uri and "${" in tracking_uri:
        tracking_uri = None

    experiment_name_raw = mlflow_cfg.get("experiment_name")
    experiment_name = str(experiment_name_raw).strip() if experiment_name_raw else "polaris-rag"

    destination_raw = tracing_cfg.get("destination_experiment")
    destination_experiment = str(destination_raw).strip() if destination_raw else None

    prompt_name_raw = prompt_registry_cfg.get("name")
    prompt_name = str(prompt_name_raw).strip() if prompt_name_raw else None

    prompt_alias_raw = prompt_registry_cfg.get("alias")
    prompt_alias = str(prompt_alias_raw).strip() if prompt_alias_raw else "prod"

    return MLflowRuntimeConfig(
        enabled=_as_bool(mlflow_cfg.get("enabled"), False),
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        tags=tags,
        tracing=TraceRuntimeConfig(
            enabled=_as_bool(tracing_cfg.get("enabled"), False),
            destination_experiment=destination_experiment,
        ),
        prompt_registry=PromptRegistryRuntimeConfig(
            enabled=_as_bool(prompt_registry_cfg.get("enabled"), False),
            name=prompt_name,
            alias=prompt_alias,
        ),
    )


def apply_mlflow_overrides(
    config: MLflowRuntimeConfig,
    *,
    enabled: bool | None = None,
    experiment_name: str | None = None,
) -> MLflowRuntimeConfig:
    """Return a copy of MLflow runtime config with CLI overrides applied."""

    updated = config
    if enabled is not None:
        updated = replace(updated, enabled=bool(enabled))
    if experiment_name:
        updated = replace(updated, experiment_name=str(experiment_name).strip())
    return updated


def _resolve_experiment_id(mlflow: Any, experiment_name: str) -> str | None:
    get_experiment_by_name = getattr(mlflow, "get_experiment_by_name", None)
    if get_experiment_by_name is None:
        return None

    experiment = get_experiment_by_name(experiment_name)
    if experiment is not None:
        exp_id = getattr(experiment, "experiment_id", None)
        return str(exp_id) if exp_id is not None else None

    create_experiment = getattr(mlflow, "create_experiment", None)
    if create_experiment is None:
        return None

    exp_id = create_experiment(experiment_name)
    return str(exp_id) if exp_id is not None else None


def _configure_trace_destination(mlflow: Any, destination_experiment: str) -> None:
    tracing_api = getattr(mlflow, "tracing", None)
    set_destination = getattr(tracing_api, "set_destination", None)
    if set_destination is None:
        return

    experiment_id = _resolve_experiment_id(mlflow, destination_experiment)
    if experiment_id is None:
        return

    destination_obj: Any | None = None

    try:
        from mlflow.entities.trace_location import MlflowExperimentLocation  # type: ignore

        destination_obj = MlflowExperimentLocation(experiment_id=experiment_id)
    except Exception:
        destination_obj = None

    if destination_obj is None:
        try:
            from mlflow.tracing.destination import MlflowExperiment  # type: ignore

            destination_obj = MlflowExperiment(experiment_id=experiment_id)
        except Exception:
            destination_obj = None

    if destination_obj is None:
        return

    try:
        set_destination(destination_obj)
    except TypeError:
        _call_if_available(tracing_api, "set_destination", destination=destination_obj)


def configure_mlflow_runtime(
    config: MLflowRuntimeConfig,
    *,
    strict: bool = False,
) -> Any | None:
    """Configure MLflow tracking URI/experiment and optional tracing destination."""

    global _TRACING_ENABLED
    _TRACING_ENABLED = bool(config.enabled and config.tracing.enabled)

    if not config.enabled:
        return None

    mlflow = _import_mlflow()
    if mlflow is None:
        _TRACING_ENABLED = False
        message = (
            "MLflow is enabled in configuration but the 'mlflow' package is not available. "
            "Install with project tracking extras (e.g. '.[tracking]')."
        )
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
        return None

    if config.tracking_uri:
        _call_if_available(mlflow, "set_tracking_uri", uri=config.tracking_uri)

    _call_if_available(mlflow, "set_experiment", experiment_name=config.experiment_name)

    if config.tracing.enabled:
        destination_experiment = config.tracing.destination_experiment or config.experiment_name
        _configure_trace_destination(mlflow, destination_experiment=destination_experiment)

    return mlflow


@contextlib.contextmanager
def _start_run(
    mlflow: Any,
    *,
    run_name: str | None,
    tags: Mapping[str, str] | None,
    nested: bool,
    log_system_metrics: bool,
) -> Iterator[Any]:
    start_run = getattr(mlflow, "start_run", None)
    if start_run is None:
        raise RuntimeError("MLflow module does not provide start_run().")

    kwargs = {
        "run_name": run_name,
        "tags": dict(tags or {}),
        "nested": nested,
        "log_system_metrics": log_system_metrics,
    }
    filtered = _filter_supported_kwargs(start_run, kwargs)

    try:
        with start_run(**filtered) as run:
            yield run
        return
    except Exception as exc:
        message = str(exc).lower()
        if not log_system_metrics or "psutil" not in message:
            raise

    logger.warning(
        "MLflow system metrics requested but psutil is unavailable. "
        "Retrying run start with system metrics disabled."
    )
    fallback_kwargs = dict(filtered)
    fallback_kwargs["log_system_metrics"] = False
    with start_run(**fallback_kwargs) as run:
        yield run


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    return str(value)


def flatten_for_logging(
    data: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested config/manifest dictionaries for param logging."""

    flat: dict[str, Any] = {}

    def _walk(node: Any, base: str) -> None:
        if isinstance(node, Mapping):
            if not node:
                flat[base] = "{}"
                return
            for key, value in node.items():
                segment = str(key)
                next_base = f"{base}.{segment}" if base else segment
                _walk(value, next_base)
            return

        if isinstance(node, list):
            if not node:
                flat[base] = "[]"
                return
            for i, value in enumerate(node):
                next_base = f"{base}.{i}" if base else str(i)
                _walk(value, next_base)
            return

        flat[base] = _to_serializable(node)

    _walk(dict(data), prefix)
    return {k: v for k, v in flat.items() if k}


def _normalize_param_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        text = str(value)
    else:
        text = json.dumps(_to_serializable(value), ensure_ascii=False)

    if len(text) > 500:
        return text[:497] + "..."
    return text


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    sensitive_tokens = (
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
    )
    return any(token in lowered for token in sensitive_tokens)


def _metric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_environment_snapshot() -> dict[str, Any]:
    """Build a JSON-serializable environment snapshot artifact."""

    tracked_env = {
        key: os.environ.get(key)
        for key in (
            "POLARIS_CONFIG",
            "MLFLOW_TRACKING_URI",
            "EMBED_API_BASE",
            "RAG_API_BASE",
            "RAG_API_BASE_URL",
            "PYTHONPATH",
        )
        if os.environ.get(key) is not None
    }

    return {
        "captured_at": datetime.now(tz=timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "environment": tracked_env,
    }


@contextlib.contextmanager
def start_span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    inputs: Any | None = None,
    outputs: Any | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[Any]:
    """Start an MLflow trace span when tracing is enabled; otherwise no-op."""

    if not _TRACING_ENABLED:
        yield _NoopSpan()
        return

    mlflow = _import_mlflow()
    if mlflow is None:
        yield _NoopSpan()
        return

    start_span_fn = getattr(mlflow, "start_span", None)
    if start_span_fn is None:
        yield _NoopSpan()
        return

    kwargs = _filter_supported_kwargs(start_span_fn, {"name": name})

    try:
        span_cm = start_span_fn(**kwargs)
    except TypeError:
        span_cm = start_span_fn(name)

    with span_cm as span:
        if tags:
            update_current_trace(tags=tags)
        if attributes:
            set_span_attributes(span, attributes)
        if inputs is not None:
            set_span_inputs(span, inputs)
        if outputs is not None:
            set_span_outputs(span, outputs)
        yield span


def update_current_trace(*, tags: Mapping[str, str] | None = None) -> None:
    """Update current trace metadata/tags when supported by MLflow build."""

    if not _TRACING_ENABLED:
        return

    mlflow = _import_mlflow()
    if mlflow is None:
        return

    updater = getattr(mlflow, "update_current_trace", None)
    if updater is None:
        return

    if not tags:
        return

    kwargs = _filter_supported_kwargs(updater, {"tags": dict(tags)})
    if not kwargs:
        return

    try:
        updater(**kwargs)
    except Exception:
        logger.debug("Unable to update current trace tags", exc_info=True)


def set_span_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
    """Set span attributes in a version-compatible manner."""

    normalized = {str(k): _to_serializable(v) for k, v in attributes.items()}

    if hasattr(span, "set_attributes"):
        try:
            span.set_attributes(normalized)
            return
        except Exception:
            logger.debug("set_attributes failed on span", exc_info=True)

    setter = getattr(span, "set_attribute", None)
    if setter is None:
        return

    for key, value in normalized.items():
        try:
            setter(key, value)
        except Exception:
            logger.debug("set_attribute failed on span", exc_info=True)


def set_span_inputs(span: Any, inputs: Any) -> None:
    """Set span inputs when supported."""

    if hasattr(span, "set_inputs"):
        try:
            span.set_inputs(_to_serializable(inputs))
        except Exception:
            logger.debug("set_inputs failed on span", exc_info=True)


def set_span_outputs(span: Any, outputs: Any) -> None:
    """Set span outputs when supported."""

    if hasattr(span, "set_outputs"):
        try:
            span.set_outputs(_to_serializable(outputs))
        except Exception:
            logger.debug("set_outputs failed on span", exc_info=True)


def _trace_identifier(span: Any) -> str | None:
    for attr in ("request_id", "trace_id"):
        value = getattr(span, attr, None)
        if value:
            return str(value)
    return None


def _associate_trace_with_run(mlflow: Any, span: Any, run_id: str | None) -> None:
    trace_id = _trace_identifier(span)
    if not trace_id or not run_id:
        return

    internal_associator = getattr(mlflow, "_associate_trace_with_run", None)
    if callable(internal_associator):
        try:
            internal_associator(trace_id, run_id)
            return
        except Exception:
            logger.debug("Internal trace/run association hook failed", exc_info=True)

    try:
        from mlflow.tracing.constant import TraceMetadataKey  # type: ignore
        from mlflow.tracing.trace_manager import InMemoryTraceManager  # type: ignore

        manager = InMemoryTraceManager.get_instance()
        setter = getattr(manager, "set_trace_metadata", None)
        if callable(setter):
            setter(trace_id, TraceMetadataKey.SOURCE_RUN, run_id)
    except Exception:
        logger.debug("Unable to associate detached trace with run", exc_info=True)


@contextlib.contextmanager
def start_detached_span(
    name: str,
    *,
    parent_span: Any | None = None,
    attributes: Mapping[str, Any] | None = None,
    inputs: Any | None = None,
    outputs: Any | None = None,
    tags: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> Iterator[Any]:
    """Start a span without attaching it to the active tracing context."""

    if not _TRACING_ENABLED:
        yield _NoopSpan()
        return

    mlflow = _import_mlflow()
    if mlflow is None:
        yield _NoopSpan()
        return

    start_span_no_context = getattr(mlflow, "start_span_no_context", None)
    if start_span_no_context is None:
        if parent_span is None:
            with start_span(
                name,
                attributes=attributes,
                inputs=inputs,
                outputs=outputs,
                tags=tags,
            ) as span:
                yield span
            return
        yield _NoopSpan()
        return

    kwargs = {
        "name": name,
        "parent_span": parent_span,
        "tags": dict(tags or {}),
    }
    filtered = _filter_supported_kwargs(start_span_no_context, kwargs)
    try:
        span = start_span_no_context(**filtered)
    except TypeError:
        span = start_span_no_context(name, parent_span=parent_span)

    try:
        if parent_span is None and run_id:
            _associate_trace_with_run(mlflow, span, run_id)
        if attributes:
            set_span_attributes(span, attributes)
        if inputs is not None:
            set_span_inputs(span, inputs)
        if outputs is not None:
            set_span_outputs(span, outputs)
        yield span
    finally:
        end = getattr(span, "end", None)
        if callable(end):
            try:
                end()
            except Exception:
                logger.debug("Unable to end detached span", exc_info=True)


class EvaluationTrackingContext:
    """Context manager wrapper for MLflow evaluation tracking."""

    def __init__(
        self,
        runtime_config: MLflowRuntimeConfig,
        *,
        enabled_override: bool | None = None,
        experiment_override: str | None = None,
    ) -> None:
        updated_config = apply_mlflow_overrides(
            runtime_config,
            enabled=enabled_override,
            experiment_name=experiment_override,
        )
        if updated_config.tracing.enabled:
            updated_config = replace(
                updated_config,
                tracing=replace(
                    updated_config.tracing,
                    destination_experiment=updated_config.experiment_name,
                ),
            )
        self.runtime_config = updated_config
        self._mlflow: Any | None = None
        self._run_id: str | None = None
        self._active = False
        self._aggregate_root_span: Any | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.runtime_config.enabled)

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def trace_headers(self) -> dict[str, str]:
        if not self._run_id:
            return {}
        return {TRACE_PARENT_RUN_HEADER: self._run_id}

    @contextlib.contextmanager
    def open(
        self,
        *,
        run_name: str | None = None,
        extra_tags: Mapping[str, str] | None = None,
        strict: bool = True,
    ) -> Iterator["EvaluationTrackingContext"]:
        if not self.enabled:
            self._active = False
            yield self
            return

        self._mlflow = configure_mlflow_runtime(self.runtime_config, strict=strict)
        if self._mlflow is None:
            self._active = False
            yield self
            return

        tags = dict(self.runtime_config.tags or {})
        if extra_tags:
            tags.update({str(k): str(v) for k, v in extra_tags.items()})

        with _start_run(
            self._mlflow,
            run_name=run_name,
            tags=tags,
            nested=False,
            log_system_metrics=True,
        ) as run:
            self._active = True
            self._run_id = str(getattr(getattr(run, "info", None), "run_id", "") or "")
            trace_tags = {
                "polaris.source": "evaluation",
                "polaris.parent_run_id": self._run_id,
            }
            if extra_tags:
                trace_tags.update({f"run.tag.{k}": str(v) for k, v in extra_tags.items()})
            with start_detached_span(
                "polaris.eval.run",
                attributes={"component": "evaluation"},
                inputs={"run_name": run_name},
                tags=trace_tags,
                run_id=self._run_id,
            ) as aggregate_root_span:
                self._aggregate_root_span = aggregate_root_span
                try:
                    yield self
                finally:
                    self._aggregate_root_span = None
                    self._active = False
                    self._run_id = None
                    self._mlflow = None

    @contextlib.contextmanager
    def stage(
        self,
        name: str,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[EvaluationStageContext]:
        if not self._active or self._mlflow is None:
            yield EvaluationStageContext(
                name=name,
                parent_run_id=self._run_id,
                child_run_id=None,
            )
            return

        stage_tags = {"stage": name}
        if tags:
            stage_tags.update({str(k): str(v) for k, v in tags.items()})

        with _start_run(
            self._mlflow,
            run_name=f"stage:{name}",
            tags=stage_tags,
            nested=True,
            log_system_metrics=False,
        ) as run:
            child_run_id = str(getattr(getattr(run, "info", None), "run_id", "") or "")
            trace_tags = {
                "polaris.source": "evaluation",
                "polaris.parent_run_id": str(self._run_id or ""),
                "polaris.child_run_id": child_run_id,
                "polaris.stage": name,
            }
            if self._run_id:
                trace_tags["mlflow.parent_run_id"] = self._run_id

            with start_span(
                f"polaris.{name}",
                tags=trace_tags,
                attributes={"stage": name, "polaris.child_run_id": child_run_id},
            ) as stage_root_span:
                with start_detached_span(
                    f"polaris.{name}",
                    parent_span=self._aggregate_root_span,
                    attributes={
                        "stage": name,
                        "polaris.child_run_id": child_run_id,
                    },
                ) as aggregate_stage_span:
                    yield EvaluationStageContext(
                        name=name,
                        parent_run_id=self._run_id,
                        child_run_id=child_run_id,
                        stage_root_span=stage_root_span,
                        aggregate_stage_span=aggregate_stage_span,
                    )

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self._active or self._mlflow is None or not params:
            return

        normalized = {
            str(key): (
                "[REDACTED]"
                if _is_sensitive_key(str(key))
                else _normalize_param_value(value)
            )
            for key, value in params.items()
        }
        log_params = getattr(self._mlflow, "log_params", None)
        if log_params is None:
            return

        # Keep batches reasonably small for compatibility with stricter stores.
        items = list(normalized.items())
        for i in range(0, len(items), 100):
            batch = dict(items[i : i + 100])
            try:
                log_params(batch)
            except Exception:
                logger.warning("Failed to log MLflow params batch", exc_info=True)

    def log_flat_params(self, data: Mapping[str, Any], *, prefix: str = "") -> None:
        self.log_params(flatten_for_logging(data, prefix=prefix))

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        if not self._active or self._mlflow is None or not metrics:
            return

        log_metrics = getattr(self._mlflow, "log_metrics", None)
        if log_metrics is None:
            return

        normalized: dict[str, float] = {}
        for key, value in metrics.items():
            metric_value = _metric_value(value)
            if metric_value is None:
                continue
            normalized[str(key)] = metric_value

        if not normalized:
            return

        kwargs = _filter_supported_kwargs(log_metrics, {"metrics": normalized, "step": step})
        try:
            log_metrics(**kwargs)
        except Exception:
            logger.warning("Failed to log MLflow metrics", exc_info=True)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        if not self._active or self._mlflow is None:
            return

        p = Path(path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            return

        log_artifact = getattr(self._mlflow, "log_artifact", None)
        if log_artifact is None:
            return

        kwargs = _filter_supported_kwargs(
            log_artifact,
            {
                "local_path": str(p),
                "artifact_path": artifact_path,
            },
        )

        if not kwargs:
            kwargs = {"local_path": str(p)}

        try:
            log_artifact(**kwargs)
        except TypeError:
            try:
                log_artifact(str(p), artifact_path)
            except Exception:
                logger.warning("Failed to log MLflow artifact: %s", p, exc_info=True)
        except Exception:
            logger.warning("Failed to log MLflow artifact: %s", p, exc_info=True)

    def log_json_artifact(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        output_path: str | Path,
        artifact_path: str | None = None,
    ) -> Path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")
        self.log_artifact(path, artifact_path=artifact_path)
        return path


__all__ = [
    "TRACE_PARENT_RUN_HEADER",
    "TRACE_CHILD_RUN_HEADER",
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
