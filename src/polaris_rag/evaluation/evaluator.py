"""polaris_rag.evaluation.evaluator

Modern evaluation runner for Polaris on top of RAGAS 0.4 metrics collections.

This module intentionally avoids deprecated ``ragas.metrics`` legacy evaluation
surfaces and executes modern collection metrics directly with controlled
concurrency.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Protocol

import pandas as pd
from openai import AsyncOpenAI
from ragas import EvaluationDataset, RunConfig
from ragas.cache import DiskCacheBackend
from ragas.embeddings.base import embedding_factory
from ragas.executor import Executor
from ragas.llms import llm_factory

from polaris_rag.evaluation.metrics import (
    DEFAULT_METRIC_ORDER,
    METRIC_REGISTRY,
    MetricSpec,
    instantiate_metrics,
    resolve_metric_specs,
)
from polaris_rag.evaluation.run_analysis import build_analysis_rows, persist_analysis_rows
from polaris_rag.observability.mlflow_tracking import set_span_outputs, start_span

logger = logging.getLogger(__name__)


class EvaluatorTraceRecorder(Protocol):
    def set_outputs(self, outputs: Any) -> None: ...

    def set_attributes(self, attributes: Mapping[str, Any]) -> None: ...


EvaluatorTraceFactory = Callable[
    [str, Mapping[str, Any], Mapping[str, Any] | None],
    contextlib.AbstractContextManager[EvaluatorTraceRecorder],
]


@dataclass(frozen=True)
class AdaptiveConcurrencySettings:
    """Settings for adaptive max-worker tuning."""

    enabled: bool = True
    worker_candidates: tuple[int, ...] = (2, 4, 8, 12, 16)
    worker_cap: int = 16
    warmup_fraction: float = 0.15
    warmup_min_samples: int = 4
    warmup_max_samples: int = 16
    failure_threshold: float = 0.02


@dataclass(frozen=True)
class ConcurrencyTrial:
    """Measured result of a single max-worker candidate."""

    workers: int
    duration_seconds: float
    failure_rate: float
    throughput: float
    failures: int
    total_scores: int


@dataclass
class EvaluationRunResult:
    """Structured result for one evaluation run."""

    scores_df: pd.DataFrame
    selected_metrics: list[str]
    skipped_metrics: list[tuple[str, str]]
    selected_max_workers: int
    run_config: RunConfig
    batch_size: int
    tuning_trials: list[ConcurrencyTrial]
    duration_seconds: float
    failure_rate: float

    def summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""

        return {
            "rows": int(len(self.scores_df)),
            "selected_metrics": list(self.selected_metrics),
            "skipped_metrics": [
                {"metric": name, "reason": reason}
                for name, reason in self.skipped_metrics
            ],
            "selected_max_workers": int(self.selected_max_workers),
            "batch_size": int(self.batch_size),
            "run_config": {
                "timeout": int(self.run_config.timeout),
                "max_retries": int(self.run_config.max_retries),
                "max_wait": int(self.run_config.max_wait),
                "max_workers": int(self.run_config.max_workers),
                "log_tenacity": bool(self.run_config.log_tenacity),
                "seed": int(self.run_config.seed),
            },
            "duration_seconds": float(self.duration_seconds),
            "failure_rate": float(self.failure_rate),
            "tuning_trials": [asdict(t) for t in self.tuning_trials],
        }


@dataclass(frozen=True)
class EvaluatorMetricTraceContext:
    metric_name: str
    sample_id: str
    row_index: int
    required_columns: tuple[str, ...]


class _NoopTraceRecorder:
    def set_outputs(self, outputs: Any) -> None:
        return None

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        return None


_ACTIVE_EVALUATOR_TRACE_CONTEXT: ContextVar[EvaluatorMetricTraceContext | None] = ContextVar(
    "polaris_evaluator_trace_context",
    default=None,
)


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


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _select_best_trial(
    trials: list[ConcurrencyTrial],
    *,
    failure_threshold: float,
) -> int:
    """Pick worker count from measured trials.

    Priority:
    1) Highest throughput among trials within failure threshold.
    2) If none valid, lowest failure-rate trial (ties -> faster throughput).
    """

    valid = [t for t in trials if t.failure_rate <= failure_threshold]
    if valid:
        return max(valid, key=lambda t: t.throughput).workers

    if not trials:
        return 1

    return min(trials, key=lambda t: (t.failure_rate, -t.throughput)).workers


def _error_text_from_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _to_trace_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {str(k): _to_trace_payload(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_trace_payload(v) for v in value]

    if is_dataclass(value):
        return _to_trace_payload(asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_trace_payload(model_dump())
        except Exception:
            logger.debug("model_dump failed while serializing evaluator trace payload", exc_info=True)

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return _to_trace_payload(to_dict())
        except Exception:
            logger.debug("dict() failed while serializing evaluator trace payload", exc_info=True)

    return str(value)


def _trace_context_payload(context: EvaluatorMetricTraceContext | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "metric_name": context.metric_name,
        "sample_id": context.sample_id,
        "row_index": context.row_index,
        "required_columns": list(context.required_columns),
    }


@contextlib.contextmanager
def _open_evaluator_trace(
    trace_factory: EvaluatorTraceFactory | None,
    *,
    name: str,
    inputs: Mapping[str, Any],
    attributes: Mapping[str, Any] | None = None,
):
    if trace_factory is None:
        yield _NoopTraceRecorder()
        return

    with trace_factory(name, inputs, attributes) as recorder:
        yield recorder


def _install_traced_create(
    resource: Any,
    *,
    endpoint_name: str,
    trace_factory: EvaluatorTraceFactory | None,
) -> None:
    if resource is None or trace_factory is None:
        return

    create = getattr(resource, "create", None)
    if create is None or getattr(create, "_polaris_evaluator_traced", False):
        return

    async def _traced_create(*args: Any, **kwargs: Any) -> Any:
        context = _ACTIVE_EVALUATOR_TRACE_CONTEXT.get()
        trace_inputs = {
            "endpoint": endpoint_name,
            "model": kwargs.get("model"),
            "messages": _to_trace_payload(kwargs.get("messages")),
            "input": _to_trace_payload(kwargs.get("input")),
            "response_model": _to_trace_payload(kwargs.get("response_model")),
            "request_args": _to_trace_payload(list(args)),
            "request_kwargs": _to_trace_payload(dict(kwargs)),
            "metric_context": _trace_context_payload(context),
        }
        trace_attributes: dict[str, Any] = {
            "component": "evaluator_llm",
            "endpoint": endpoint_name,
        }
        if context is not None:
            trace_attributes.update(
                {
                    "metric_name": context.metric_name,
                    "sample_id": context.sample_id,
                    "row_index": context.row_index,
                }
            )

        with _open_evaluator_trace(
            trace_factory,
            name="polaris.ragas_evaluation.evaluator_llm",
            inputs=trace_inputs,
            attributes=trace_attributes,
        ) as recorder:
            try:
                response = await create(*args, **kwargs)
            except Exception as exc:
                recorder.set_attributes({"status": "error"})
                recorder.set_outputs({"error": _error_text_from_exception(exc)})
                raise

            recorder.set_attributes({"status": "success"})
            recorder.set_outputs({"response": _to_trace_payload(response)})
            return response

    setattr(_traced_create, "_polaris_evaluator_traced", True)
    setattr(resource, "create", _traced_create)


def _TracedAsyncOpenAIClient(
    client: AsyncOpenAI,
    *,
    trace_factory: EvaluatorTraceFactory | None,
) -> AsyncOpenAI:
    chat = getattr(client, "chat", None)
    _install_traced_create(
        getattr(chat, "completions", None),
        endpoint_name="chat.completions",
        trace_factory=trace_factory,
    )
    _install_traced_create(
        getattr(client, "responses", None),
        endpoint_name="responses",
        trace_factory=trace_factory,
    )
    return client


async def _score_metric(
    metric: Any,
    kwargs: dict[str, Any],
    trace_context: EvaluatorMetricTraceContext | None = None,
) -> Any:
    """Score one metric call and return raw numeric/string value."""

    token = None
    if trace_context is not None:
        token = _ACTIVE_EVALUATOR_TRACE_CONTEXT.set(trace_context)

    try:
        result = await metric.ascore(**kwargs)
        return result.value if hasattr(result, "value") else result
    finally:
        if token is not None:
            _ACTIVE_EVALUATOR_TRACE_CONTEXT.reset(token)


def _rows_from_executor_results(
    *,
    rows: list[dict[str, Any]],
    metric_names: list[str],
    results: list[Any],
) -> list[dict[str, Any]]:
    """Reconstruct row-wise metric outputs from flat executor results."""

    scored_rows: list[dict[str, Any]] = []
    n_metrics = len(metric_names)

    for i, row in enumerate(rows):
        out: dict[str, Any] = {
            "user_input": row.get("user_input"),
            "reference": row.get("reference"),
            "response": row.get("response"),
        }

        offset = i * n_metrics
        for j, metric_name in enumerate(metric_names):
            index = offset + j
            value = results[index] if index < len(results) else float("nan")
            if hasattr(value, "value"):
                value = value.value
            out[metric_name] = value

        scored_rows.append(out)

    return scored_rows


def executor_results_to_dataframe(
    *,
    rows: list[dict[str, Any]],
    metric_names: list[str],
    results: list[Any],
) -> pd.DataFrame:
    """Convert raw executor results into a score dataframe."""

    return pd.DataFrame(
        _rows_from_executor_results(
            rows=rows,
            metric_names=metric_names,
            results=results,
        )
    )


class Evaluator:
    """RAGAS metrics-collections evaluator with adaptive concurrency."""

    def __init__(
        self,
        *,
        llm_model: str,
        llm_api_base: str,
        llm_api_key: str | None,
        llm_kwargs: Mapping[str, Any] | None,
        embedding_model: str,
        embedding_api_base: str,
        embedding_api_key: str | None,
        embedding_kwargs: Mapping[str, Any] | None,
        requested_metrics: Iterable[str] | None = None,
        auto_gate_metrics: bool = True,
        run_config: RunConfig | None = None,
        batch_size: int | None = None,
        adaptive_concurrency: AdaptiveConcurrencySettings | None = None,
        raise_exceptions: bool = False,
        use_cache: bool = False,
        cache_dir: str = ".cache/ragas_eval",
        trace_evaluator_llm: bool = False,
        trace_factory: EvaluatorTraceFactory | None = None,
    ):
        self.requested_metrics = list(requested_metrics) if requested_metrics is not None else None
        self.auto_gate_metrics = auto_gate_metrics
        self.raise_exceptions = raise_exceptions
        self.trace_evaluator_llm = bool(trace_evaluator_llm)
        self.trace_factory = trace_factory

        self.run_config = run_config or RunConfig(
            timeout=120,
            max_retries=4,
            max_wait=30,
            max_workers=4,
            log_tenacity=True,
            seed=42,
        )

        self.batch_size = batch_size
        self.adaptive_concurrency = adaptive_concurrency or AdaptiveConcurrencySettings()

        self.cache = DiskCacheBackend(cache_dir=cache_dir) if use_cache else None
        if self.trace_evaluator_llm and self.trace_factory is None:
            logger.warning(
                "Evaluator LLM tracing was enabled without a trace_factory; evaluator requests will not be traced."
            )

        self.llm = self._build_llm(
            model=llm_model,
            api_base=llm_api_base,
            api_key=llm_api_key,
            llm_kwargs=llm_kwargs or {},
        )
        self.embeddings = self._build_embeddings(
            model=embedding_model,
            api_base=embedding_api_base,
            api_key=embedding_api_key,
            embedding_kwargs=embedding_kwargs or {},
        )

    def register_metric(self, metric_name: str) -> None:
        """Add a metric to the active requested metric list."""

        if metric_name not in METRIC_REGISTRY:
            available = ", ".join(sorted(METRIC_REGISTRY.keys()))
            raise KeyError(f"Unknown metric '{metric_name}'. Available metrics: {available}")

        if self.requested_metrics is None:
            self.requested_metrics = []
        if metric_name not in self.requested_metrics:
            self.requested_metrics.append(metric_name)

    def clear_registered_metrics(self) -> None:
        """Reset metric selection to the default metric order."""

        self.requested_metrics = None

    def list_metric_names(self, dataset_columns: set[str] | None = None) -> list[str]:
        """List requested/default metric names, optionally after auto-gating."""

        names = list(self.requested_metrics) if self.requested_metrics is not None else list(DEFAULT_METRIC_ORDER)
        if dataset_columns is None:
            return names

        specs, _ = resolve_metric_specs(
            dataset_columns=dataset_columns,
            requested_metrics=names,
            auto_gate=self.auto_gate_metrics,
        )
        return [spec.name for spec in specs]

    @classmethod
    def from_global_config(
        cls,
        cfg: Any,
        *,
        requested_metrics: Iterable[str] | None = None,
        trace_evaluator_llm: bool = False,
        trace_factory: EvaluatorTraceFactory | None = None,
    ) -> "Evaluator":
        """Build an evaluator from ``GlobalConfig``."""

        raw = _as_mapping(getattr(cfg, "raw", cfg))
        eval_cfg = _as_mapping(raw.get("evaluation", {}))

        llm_cfg = _as_mapping(raw.get("evaluator_llm", {}))
        embed_cfg = _as_mapping(raw.get("embedder", {}))

        run_cfg = _as_mapping(eval_cfg.get("run", {}))
        cache_cfg = _as_mapping(eval_cfg.get("cache", {}))
        tune_cfg = _as_mapping(eval_cfg.get("adaptive_concurrency", {}))
        metric_cfg = _as_mapping(eval_cfg.get("metrics", {}))

        req_metrics = list(requested_metrics) if requested_metrics is not None else metric_cfg.get("requested")
        if isinstance(req_metrics, str):
            req_metrics = [m.strip() for m in req_metrics.split(",") if m.strip()]

        run_config = RunConfig(
            timeout=_coerce_int(run_cfg.get("timeout"), 120),
            max_retries=_coerce_int(run_cfg.get("max_retries"), 4),
            max_wait=_coerce_int(run_cfg.get("max_wait"), 30),
            max_workers=_coerce_int(run_cfg.get("max_workers"), 4),
            log_tenacity=_as_bool(run_cfg.get("log_tenacity"), True),
            seed=_coerce_int(run_cfg.get("seed"), 42),
        )

        adaptive = AdaptiveConcurrencySettings(
            enabled=_as_bool(tune_cfg.get("enabled"), True),
            worker_candidates=tuple(int(x) for x in tune_cfg.get("worker_candidates", [2, 4, 8, 12, 16])),
            worker_cap=_coerce_int(tune_cfg.get("worker_cap"), 16),
            warmup_fraction=_coerce_float(tune_cfg.get("warmup_fraction"), 0.15),
            warmup_min_samples=_coerce_int(tune_cfg.get("warmup_min_samples"), 4),
            warmup_max_samples=_coerce_int(tune_cfg.get("warmup_max_samples"), 16),
            failure_threshold=_coerce_float(tune_cfg.get("failure_threshold"), 0.02),
        )

        return cls(
            llm_model=str(llm_cfg.get("model_name", "gpt-4o-mini")),
            llm_api_base=str(llm_cfg.get("api_base", "https://api.openai.com/v1")),
            llm_api_key=llm_cfg.get("api_key"),
            llm_kwargs=_as_mapping(llm_cfg.get("model_kwargs", {})),
            embedding_model=str(embed_cfg.get("model_name", "text-embedding-3-small")),
            embedding_api_base=str(embed_cfg.get("api_base", "https://api.openai.com/v1")),
            embedding_api_key=embed_cfg.get("api_key"),
            embedding_kwargs=_as_mapping(embed_cfg.get("model_kwargs", {})),
            requested_metrics=req_metrics,
            auto_gate_metrics=_as_bool(metric_cfg.get("auto_gate"), True),
            run_config=run_config,
            batch_size=_coerce_int(run_cfg.get("batch_size"), 0) or None,
            adaptive_concurrency=adaptive,
            raise_exceptions=_as_bool(run_cfg.get("raise_exceptions"), False),
            use_cache=_as_bool(cache_cfg.get("enabled"), False),
            cache_dir=str(cache_cfg.get("dir", ".cache/ragas_eval")),
            trace_evaluator_llm=trace_evaluator_llm,
            trace_factory=trace_factory,
        )

    def _build_llm(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str | None,
        llm_kwargs: Mapping[str, Any],
    ) -> Any:
        kwargs = dict(llm_kwargs)
        # Remove legacy stop-hack settings that hurt structured output paths.
        kwargs.pop("stop", None)
        kwargs.pop("stop_list", None)

        raw_client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key or "fake",
            timeout=float(self.run_config.timeout),
            max_retries=max(0, int(self.run_config.max_retries)),
        )
        client: Any = raw_client
        if self.trace_evaluator_llm and self.trace_factory is not None:
            client = _TracedAsyncOpenAIClient(raw_client, trace_factory=self.trace_factory)

        llm = llm_factory(
            model=model,
            provider="openai",
            client=client,
            adapter="auto",
            cache=self.cache,
            **kwargs,
        )

        if hasattr(llm, "set_run_config"):
            llm.set_run_config(self.run_config)

        return llm

    def _build_embeddings(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str | None,
        embedding_kwargs: Mapping[str, Any],
    ) -> Any:
        kwargs = dict(embedding_kwargs)

        client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key or "fake",
            timeout=float(self.run_config.timeout),
            max_retries=max(0, int(self.run_config.max_retries)),
        )

        embeddings = embedding_factory(
            provider="openai",
            model=model,
            client=client,
            interface="modern",
            cache=self.cache,
            **kwargs,
        )

        if hasattr(embeddings, "set_run_config"):
            embeddings.set_run_config(self.run_config)

        return embeddings

    def _resolve_runtime_metrics(
        self,
        *,
        dataset: EvaluationDataset,
    ) -> tuple[list[tuple[MetricSpec, Any]], list[tuple[str, str]]]:
        columns = set(dataset.features())
        specs, skipped = resolve_metric_specs(
            dataset_columns=columns,
            requested_metrics=self.requested_metrics,
            auto_gate=self.auto_gate_metrics,
        )
        instances = instantiate_metrics(specs, llm=self.llm, embeddings=self.embeddings)
        return list(zip(specs, instances)), skipped

    def _clone_run_config(self, *, max_workers: int) -> RunConfig:
        return RunConfig(
            timeout=int(self.run_config.timeout),
            max_retries=int(self.run_config.max_retries),
            max_wait=int(self.run_config.max_wait),
            max_workers=int(max_workers),
            exception_types=self.run_config.exception_types,
            log_tenacity=bool(self.run_config.log_tenacity),
            seed=int(self.run_config.seed),
        )

    @staticmethod
    def _batch_size_for_workers(workers: int, explicit: int | None) -> int:
        if explicit is not None:
            return int(explicit)
        return min(64, max(8, workers * 2))

    async def _score_dataset_async(
        self,
        *,
        dataset: EvaluationDataset,
        metrics: list[tuple[MetricSpec, Any]],
        run_config: RunConfig,
        batch_size: int,
        show_progress: bool,
        source_rows: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        executor, rows, metric_names = self._create_executor(
            dataset=dataset,
            metrics=metrics,
            run_config=run_config,
            batch_size=batch_size,
            show_progress=show_progress,
            source_rows=source_rows,
        )
        results = await executor.aresults()
        scored_rows = _rows_from_executor_results(
            rows=rows,
            metric_names=metric_names,
            results=results,
        )
        return pd.DataFrame(scored_rows)

    def _create_executor(
        self,
        *,
        dataset: EvaluationDataset,
        metrics: list[tuple[MetricSpec, Any]],
        run_config: RunConfig,
        batch_size: int,
        show_progress: bool,
        source_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[Executor, list[dict[str, Any]], list[str]]:
        """Create and populate a RAGAS executor for all metric jobs."""

        executor = Executor(
            desc="Evaluating",
            show_progress=show_progress,
            keep_progress_bar=True,
            raise_exceptions=self.raise_exceptions,
            run_config=run_config,
            batch_size=batch_size,
        )

        metric_names = [spec.name for spec, _ in metrics]
        rows = dataset.to_list()
        for row_index, row in enumerate(rows):
            source_row = source_rows[row_index] if source_rows is not None and row_index < len(source_rows) else row
            sample_id = str(
                source_row.get("id")
                or row.get("id")
                or f"row-{row_index}"
            )
            for spec, metric in metrics:
                kwargs = {col: row.get(col) for col in spec.required_columns}
                trace_context = EvaluatorMetricTraceContext(
                    metric_name=spec.name,
                    sample_id=sample_id,
                    row_index=row_index,
                    required_columns=tuple(sorted(spec.required_columns)),
                )
                executor.submit(_score_metric, metric, kwargs, trace_context)

        return executor, rows, metric_names

    def _score_dataset(
        self,
        *,
        dataset: EvaluationDataset,
        metrics: list[tuple[MetricSpec, Any]],
        run_config: RunConfig,
        batch_size: int,
        show_progress: bool,
        source_rows: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        return asyncio.run(
            self._score_dataset_async(
                dataset=dataset,
                metrics=metrics,
                run_config=run_config,
                batch_size=batch_size,
                show_progress=show_progress,
                source_rows=source_rows,
            )
        )

    def _tune_max_workers(
        self,
        *,
        dataset: EvaluationDataset,
        metrics: list[tuple[MetricSpec, Any]],
        source_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[int, list[ConcurrencyTrial]]:
        settings = self.adaptive_concurrency

        if not settings.enabled:
            return int(self.run_config.max_workers), []

        if len(dataset) <= 1:
            return max(1, int(self.run_config.max_workers)), []

        warmup_size = int(math.ceil(len(dataset) * settings.warmup_fraction))
        warmup_size = max(settings.warmup_min_samples, warmup_size)
        warmup_size = min(settings.warmup_max_samples, warmup_size, len(dataset))

        warmup_dataset = dataset[:warmup_size]
        warmup_source_rows = source_rows[:warmup_size] if source_rows is not None else None

        candidates = sorted(
            {
                max(1, int(w))
                for w in settings.worker_candidates
                if int(w) <= int(settings.worker_cap)
            }
        )
        if not candidates:
            candidates = [max(1, int(self.run_config.max_workers))]

        trials: list[ConcurrencyTrial] = []

        logger.info(
            "Running adaptive concurrency tuning on %s rows with candidates=%s",
            warmup_size,
            candidates,
        )

        for workers in candidates:
            batch_size = self._batch_size_for_workers(workers, self.batch_size)
            rc = self._clone_run_config(max_workers=workers)

            start = time.perf_counter()
            df = self._score_dataset(
                dataset=warmup_dataset,
                metrics=metrics,
                run_config=rc,
                batch_size=batch_size,
                show_progress=False,
                source_rows=warmup_source_rows,
            )
            duration = max(1e-9, time.perf_counter() - start)

            metric_columns = [spec.name for spec, _ in metrics]
            total_scores = len(df) * len(metric_columns)
            failures = int(df[metric_columns].isna().sum().sum()) if total_scores else 0
            failure_rate = (failures / total_scores) if total_scores else 0.0
            throughput = ((total_scores - failures) / duration) if duration > 0 else 0.0

            trials.append(
                ConcurrencyTrial(
                    workers=workers,
                    duration_seconds=duration,
                    failure_rate=failure_rate,
                    throughput=throughput,
                    failures=failures,
                    total_scores=total_scores,
                )
            )
            logger.info(
                "Tuning trial workers=%s duration=%.1fs throughput=%.2f failure_rate=%.4f failures=%s/%s",
                workers,
                duration,
                throughput,
                failure_rate,
                failures,
                total_scores,
            )

        selected = _select_best_trial(trials, failure_threshold=settings.failure_threshold)
        logger.info("Selected max_workers=%s after tuning.", selected)
        return selected, trials

    def evaluate(
        self,
        *,
        dataset: EvaluationDataset,
        source_rows: list[dict[str, Any]] | None = None,
        tune_concurrency: bool = True,
        show_progress: bool = True,
    ) -> EvaluationRunResult:
        """Run evaluation and return scores plus execution metadata."""

        runtime_metrics, skipped_metrics = self._resolve_runtime_metrics(dataset=dataset)
        logger.info(
            "Resolved evaluation runtime: rows=%s selected_metrics=%s skipped_metrics=%s",
            len(dataset),
            [spec.name for spec, _ in runtime_metrics],
            len(skipped_metrics),
        )

        selected_workers = int(self.run_config.max_workers)
        tuning_trials: list[ConcurrencyTrial] = []

        if tune_concurrency:
            with start_span(
                "polaris.ragas_evaluation.tune_concurrency",
                inputs={
                    "dataset_rows": len(dataset),
                    "requested_metrics": [spec.name for spec, _ in runtime_metrics],
                },
                attributes={"tune_concurrency": True},
            ) as tuning_span:
                selected_workers, tuning_trials = self._tune_max_workers(
                    dataset=dataset,
                    metrics=runtime_metrics,
                    source_rows=source_rows,
                )
                set_span_outputs(
                    tuning_span,
                    {
                        "selected_workers": selected_workers,
                        "trial_count": len(tuning_trials),
                        "trials": [asdict(t) for t in tuning_trials],
                    },
                )

        run_config = self._clone_run_config(max_workers=selected_workers)

        # Propagate selected run config to model wrappers.
        if hasattr(self.llm, "set_run_config"):
            self.llm.set_run_config(run_config)
        if hasattr(self.embeddings, "set_run_config"):
            self.embeddings.set_run_config(run_config)

        batch_size = self._batch_size_for_workers(selected_workers, self.batch_size)
        logger.info(
            "Starting score pass: rows=%s metrics=%s max_workers=%s batch_size=%s show_progress=%s",
            len(dataset),
            [spec.name for spec, _ in runtime_metrics],
            run_config.max_workers,
            batch_size,
            show_progress,
        )

        with start_span(
            "polaris.ragas_evaluation.score_dataset",
            inputs={
                "dataset_rows": len(dataset),
                "metrics": [spec.name for spec, _ in runtime_metrics],
                "batch_size": batch_size,
                "max_workers": run_config.max_workers,
            },
            attributes={"show_progress": show_progress},
        ) as score_span:
            start = time.perf_counter()
            scores_df = self._score_dataset(
                dataset=dataset,
                metrics=runtime_metrics,
                run_config=run_config,
                batch_size=batch_size,
                show_progress=show_progress,
                source_rows=source_rows,
            )
            duration = time.perf_counter() - start
            set_span_outputs(
                score_span,
                {
                    "rows": len(scores_df),
                    "duration_seconds": duration,
                },
            )
        logger.info(
            "Completed score pass in %.1fs for %s rows.",
            duration,
            len(scores_df),
        )

        if source_rows is not None and len(source_rows) == len(scores_df):
            ids = [str(row.get("id", f"row-{i}")) for i, row in enumerate(source_rows)]
            metas = [row.get("metadata", {}) for row in source_rows]
            scores_df.insert(0, "id", ids)
            scores_df["metadata"] = metas

        metric_names = [spec.name for spec, _ in runtime_metrics]
        total = len(scores_df) * len(metric_names)
        failures = int(scores_df[metric_names].isna().sum().sum()) if total else 0
        failure_rate = (failures / total) if total else 0.0
        logger.info(
            "Evaluation aggregation complete: rows=%s failure_rate=%.4f failures=%s/%s",
            len(scores_df),
            failure_rate,
            failures,
            total,
        )

        with start_span(
            "polaris.ragas_evaluation.aggregate_results",
            inputs={
                "rows": len(scores_df),
                "metrics": metric_names,
            },
        ) as aggregate_span:
            result = EvaluationRunResult(
                scores_df=scores_df,
                selected_metrics=metric_names,
                skipped_metrics=skipped_metrics,
                selected_max_workers=selected_workers,
                run_config=run_config,
                batch_size=batch_size,
                tuning_trials=tuning_trials,
                duration_seconds=duration,
                failure_rate=failure_rate,
            )
            set_span_outputs(
                aggregate_span,
                {
                    "selected_max_workers": selected_workers,
                    "failure_rate": failure_rate,
                    "skipped_metrics": [
                        {"metric": metric, "reason": reason}
                        for metric, reason in skipped_metrics
                    ],
                },
            )

        return result

    def build_executor(
        self,
        *,
        dataset: EvaluationDataset,
        tune_concurrency: bool = True,
        show_progress: bool = True,
    ) -> tuple[Executor, dict[str, Any]]:
        """Build an executor for cancellable metric execution.

        The caller can run ``executor.results()`` and call ``executor.cancel()``
        from another thread/context to request cancellation.
        """

        runtime_metrics, skipped_metrics = self._resolve_runtime_metrics(dataset=dataset)

        selected_workers = int(self.run_config.max_workers)
        tuning_trials: list[ConcurrencyTrial] = []
        if tune_concurrency:
            selected_workers, tuning_trials = self._tune_max_workers(
                dataset=dataset,
                metrics=runtime_metrics,
            )

        run_config = self._clone_run_config(max_workers=selected_workers)
        batch_size = self._batch_size_for_workers(selected_workers, self.batch_size)

        executor, rows, metric_names = self._create_executor(
            dataset=dataset,
            metrics=runtime_metrics,
            run_config=run_config,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        metadata = {
            "rows": rows,
            "metric_names": metric_names,
            "skipped_metrics": skipped_metrics,
            "selected_max_workers": selected_workers,
            "run_config": run_config,
            "batch_size": batch_size,
            "tuning_trials": tuning_trials,
        }
        return executor, metadata


def write_outputs(
    *,
    result: EvaluationRunResult,
    output_dir: str | Path,
    extra_manifest: Mapping[str, Any] | None = None,
    source_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Persist standard output artifacts for an evaluation run."""

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = out_dir / "scores.csv"
    scores_parquet = out_dir / "scores.parquet"
    summary_json = out_dir / "summary.json"
    manifest_json = out_dir / "run_manifest.json"
    analysis_rows_jsonl = out_dir / "analysis_rows.jsonl"

    result.scores_df.to_csv(scores_csv, index=False)

    try:
        result.scores_df.to_parquet(scores_parquet, index=False)
        parquet_written = True
    except Exception as exc:
        parquet_written = False
        logger.warning("Unable to write parquet output: %s", exc)

    summary_payload = result.summary_dict()
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    extra = dict(extra_manifest or {})
    condition_fields = {
        "preset_name": extra.get("preset_name"),
        "preset_description": extra.get("preset_description"),
        "condition_fingerprint": extra.get("condition_fingerprint"),
        "condition_summary": extra.get("condition_summary"),
    }
    analysis_rows_written = False
    if source_rows is not None:
        analysis_rows = build_analysis_rows(
            source_rows=source_rows,
            scores_df=result.scores_df,
            condition_fields=condition_fields,
        )
        persist_analysis_rows(analysis_rows, analysis_rows_jsonl)
        analysis_rows_written = True

    manifest_payload = {
        "summary": summary_payload,
        "preset_name": condition_fields["preset_name"],
        "preset_description": condition_fields["preset_description"],
        "condition_fingerprint": condition_fields["condition_fingerprint"],
        "condition_summary": condition_fields["condition_summary"],
        "dataset": extra.get("dataset"),
        "tune_concurrency": extra.get("tune_concurrency"),
        "trace_evaluator_llm": extra.get("trace_evaluator_llm"),
        "mlflow_parent_run_id": extra.get("mlflow_parent_run_id"),
        "config_file": extra.get("config_file"),
        "artifacts": {
            "scores_csv": str(scores_csv),
            "scores_parquet": str(scores_parquet) if parquet_written else None,
            "summary_json": str(summary_json),
            "analysis_rows_jsonl": str(analysis_rows_jsonl) if analysis_rows_written else None,
        },
        "extra": extra,
    }
    manifest_json.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    artifacts = {
        "scores_csv": scores_csv,
        "summary_json": summary_json,
        "manifest_json": manifest_json,
    }
    if parquet_written:
        artifacts["scores_parquet"] = scores_parquet
    if analysis_rows_written:
        artifacts["analysis_rows_jsonl"] = analysis_rows_jsonl

    return artifacts


__all__ = [
    "AdaptiveConcurrencySettings",
    "ConcurrencyTrial",
    "EvaluationRunResult",
    "Evaluator",
    "executor_results_to_dataframe",
    "write_outputs",
]
