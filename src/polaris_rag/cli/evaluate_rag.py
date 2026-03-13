"""Evaluation CLI entrypoint for Polaris RAG."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterator, Mapping

from polaris_rag.app.container import build_container
from polaris_rag.common.request_budget import (
    EVAL_POLICY_DIAGNOSTIC,
    EVAL_POLICY_OFFICIAL,
    INFRA_FAILURE_CLASSES,
    POLARIS_EVAL_POLICY_HEADER,
    POLARIS_TIMEOUT_HEADER,
    normalize_evaluation_policy,
    resolve_evaluation_deadlines,
)
from polaris_rag.config import GlobalConfig
from polaris_rag.evaluation.evaluation_dataset import (
    PrepProgressEvent,
    PrepRetryPolicy,
    build_prepared_rows,
    build_prepared_rows_from_api,
    load_prepared_rows,
    load_raw_examples,
    persist_prepared_rows,
    to_evaluation_dataset,
)
from polaris_rag.observability.mlflow_tracking import (
    EvaluationTrackingContext,
    EvaluationStageContext,
    build_environment_snapshot,
    load_mlflow_runtime_config,
)

logger = logging.getLogger(__name__)


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


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_metrics(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _write_prepared_rows_artifact(
    *,
    output_dir: Path,
    prepared_rows: list[dict[str, Any]],
    tracking: EvaluationTrackingContext,
) -> Path:
    path = output_dir / "prepared_rows.json"
    path.write_text(
        json.dumps(prepared_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tracking.log_artifact(path, artifact_path="outputs")
    return path


def _write_runtime_snapshots(
    *,
    output_dir: Path,
    config_payload: Mapping[str, Any],
    tracking: EvaluationTrackingContext,
) -> None:
    tracking.log_json_artifact(
        build_environment_snapshot(),
        output_path=output_dir / "env_snapshot.json",
        artifact_path="outputs",
    )
    tracking.log_json_artifact(
        config_payload,
        output_path=output_dir / "config_snapshot.json",
        artifact_path="inputs",
    )


def _compact_error_text(value: str | None, *, limit: int = 120) -> str | None:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


class _PrepProgressRenderer:
    """Render dataset-preparation progress as a single-line live bar."""

    def __init__(
        self,
        *,
        width: int = 24,
        interactive: bool = True,
        log_interval_seconds: float = 30.0,
    ):
        self.width = max(8, int(width))
        self.interactive = bool(interactive)
        self.log_interval_seconds = max(1.0, float(log_interval_seconds))
        self._active = False
        self._last_logged_elapsed = -1.0
        self._last_event: PrepProgressEvent | None = None

    def update(self, event: PrepProgressEvent) -> None:
        self._last_event = event
        total = event.total if event.total > 0 else 1
        fraction = event.completed / total
        pct = int(round(fraction * 100.0))
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = (event.completed / event.elapsed_seconds) if event.elapsed_seconds > 0 else 0.0
        line = (
            f"[prep:{event.mode}] [{bar}] {pct:3d}% "
            f"{event.completed}/{event.total} errors={event.failures} "
            f"elapsed={event.elapsed_seconds:5.1f}s rate={rate:5.2f}/s"
        )
        compact_error = _compact_error_text(event.last_error)
        if compact_error:
            line += f" last_error={compact_error}"
        if self.interactive:
            self._active = True
            print(f"\r{line}", end="", file=sys.stderr, flush=True)
            return

        should_log = (
            event.completed >= event.total
            or self._last_logged_elapsed < 0
            or (event.elapsed_seconds - self._last_logged_elapsed) >= self.log_interval_seconds
        )
        if should_log:
            self._last_logged_elapsed = event.elapsed_seconds
            logger.info("%s", line)

    def finish(self) -> None:
        if self.interactive and self._active:
            print(file=sys.stderr, flush=True)
            self._active = False
            return

        if not self.interactive and self._last_event is not None:
            event = self._last_event
            if event.completed < event.total and event.elapsed_seconds > self._last_logged_elapsed:
                self._last_logged_elapsed = event.elapsed_seconds
                self.update(event)


def _configure_logging(level_name: str = "INFO") -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    package_logger = logging.getLogger("polaris_rag")
    package_logger.setLevel(level)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


@contextmanager
def _phase_heartbeat(label: str, *, interval_seconds: float = 60.0) -> Iterator[None]:
    stop_event = threading.Event()
    started_at = time.perf_counter()

    def _worker() -> None:
        while not stop_event.wait(interval_seconds):
            elapsed = max(0.0, time.perf_counter() - started_at)
            logger.info("%s still running (elapsed=%.1fs).", label, elapsed)

    worker = threading.Thread(
        target=_worker,
        name=f"polaris-heartbeat-{label}",
        daemon=True,
    )
    logger.info("%s started.", label)
    worker.start()
    try:
        yield
    except Exception:
        elapsed = max(0.0, time.perf_counter() - started_at)
        logger.exception("%s failed after %.1fs.", label, elapsed)
        raise
    else:
        elapsed = max(0.0, time.perf_counter() - started_at)
        logger.info("%s finished in %.1fs.", label, elapsed)
    finally:
        stop_event.set()
        worker.join(timeout=max(0.1, interval_seconds))


def _source_error(row: Mapping[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("source_error")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _row_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _error_class(source_error: str) -> str:
    head = source_error.split(":", 1)[0].strip()
    return head or "UnknownError"


def _failure_class(row: Mapping[str, Any]) -> str | None:
    value = _row_metadata(row).get("failure_class")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _failure_stage(row: Mapping[str, Any]) -> str | None:
    value = _row_metadata(row).get("failure_stage")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _response_status(row: Mapping[str, Any]) -> str | None:
    value = _row_metadata(row).get("response_status")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _elapsed_ms(row: Mapping[str, Any]) -> int | None:
    value = _row_metadata(row).get("elapsed_ms")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _budget_ms(row: Mapping[str, Any]) -> int | None:
    value = _row_metadata(row).get("budget_ms")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = max(0.0, min(1.0, float(quantile))) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    fraction = rank - low
    return float(ordered[low] + ((ordered[high] - ordered[low]) * fraction))


def _latency_budget_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed_values: list[float] = []
    utilization_values: list[float] = []
    for row in rows:
        if _source_error(row):
            continue
        elapsed_ms = _elapsed_ms(row)
        if elapsed_ms is None:
            continue
        elapsed_values.append(float(elapsed_ms))
        budget_ms = _budget_ms(row)
        if budget_ms and budget_ms > 0:
            utilization_values.append(float(elapsed_ms) / float(budget_ms))

    if not elapsed_values:
        return {}

    report: dict[str, Any] = {
        "count": len(elapsed_values),
        "elapsed_ms": {
            "p50": _percentile(elapsed_values, 0.50),
            "p90": _percentile(elapsed_values, 0.90),
            "p95": _percentile(elapsed_values, 0.95),
            "max": float(max(elapsed_values)),
            "mean": float(sum(elapsed_values) / len(elapsed_values)),
        },
    }
    if utilization_values:
        report["budget_utilization"] = {
            "p50": _percentile(utilization_values, 0.50),
            "p90": _percentile(utilization_values, 0.90),
            "max": float(max(utilization_values)),
        }
    return report


def _max_infra_failure_rate(eval_cfg: Mapping[str, Any]) -> float:
    validity_cfg = _as_mapping(eval_cfg.get("validity", {}))
    return max(0.0, _as_float(validity_cfg.get("max_infra_failure_rate"), 0.01))


def _prep_stats(
    rows: list[dict[str, Any]],
    elapsed_seconds: float,
    *,
    evaluation_policy: str,
    max_infra_failure_rate: float,
) -> dict[str, Any]:
    total = len(rows)
    failures = 0
    error_classes: Counter[str] = Counter()
    failure_classes: Counter[str] = Counter()
    failure_stages: Counter[str] = Counter()
    response_statuses: Counter[str] = Counter()
    for row in rows:
        err = _source_error(row)
        if err:
            failures += 1
            error_classes[_error_class(err)] += 1
        failure_class = _failure_class(row)
        if failure_class:
            failure_classes[failure_class] += 1
        failure_stage = _failure_stage(row)
        if failure_stage:
            failure_stages[failure_stage] += 1
        response_status = _response_status(row)
        if response_status:
            response_statuses[response_status] += 1

    successes = total - failures
    rate = (total / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    coverage = (successes / total) if total else 0.0
    infra_failures = sum(failure_classes.get(name, 0) for name in INFRA_FAILURE_CLASSES)
    infra_failure_rate = (infra_failures / total) if total else 0.0
    failure_class_rates = {
        name: (float(count) / total) if total else 0.0
        for name, count in failure_classes.items()
    }
    failure_stage_rates = {
        name: (float(count) / total) if total else 0.0
        for name, count in failure_stages.items()
    }
    response_status_rates = {
        name: (float(count) / total) if total else 0.0
        for name, count in response_statuses.items()
    }
    normalized_policy = normalize_evaluation_policy(evaluation_policy, default=EVAL_POLICY_OFFICIAL)
    if normalized_policy == EVAL_POLICY_DIAGNOSTIC:
        run_validity = "DIAGNOSTIC"
    else:
        run_validity = "VALID" if infra_failure_rate <= max_infra_failure_rate else "INVALID"
    return {
        "prep_total_rows": total,
        "prep_success_rows": successes,
        "prep_failed_rows": failures,
        "prep_elapsed_seconds": elapsed_seconds,
        "prep_rate_rows_per_second": rate,
        "prep_error_classes": dict(error_classes),
        "prep_failure_classes": dict(failure_classes),
        "prep_failure_class_rates": failure_class_rates,
        "prep_failure_stages": dict(failure_stages),
        "prep_failure_stage_rates": failure_stage_rates,
        "prep_response_statuses": dict(response_statuses),
        "prep_response_status_rates": response_status_rates,
        "prep_coverage": coverage,
        "prep_empty_response_rate": (failure_classes.get("empty_response", 0) / total) if total else 0.0,
        "infra_failure_rate": infra_failure_rate,
        "run_validity": run_validity,
        "run_policy": normalized_policy,
        "max_infra_failure_rate": max_infra_failure_rate,
        "latency_budget_report": _latency_budget_report(rows),
    }


def _print_prep_summary(stats: Mapping[str, Any]) -> None:
    total = int(stats.get("prep_total_rows", 0))
    success = int(stats.get("prep_success_rows", 0))
    failed = int(stats.get("prep_failed_rows", 0))
    elapsed = float(stats.get("prep_elapsed_seconds", 0.0))
    rate = float(stats.get("prep_rate_rows_per_second", 0.0))
    print(
        f"Prepared {total} rows (success={success}, failed={failed}, elapsed={elapsed:.1f}s, rate={rate:.2f} rows/s).",
        file=sys.stderr,
    )
    run_validity = str(stats.get("run_validity", "") or "").strip()
    if run_validity:
        coverage = float(stats.get("prep_coverage", 0.0))
        infra_failure_rate = float(stats.get("infra_failure_rate", 0.0))
        print(
            f"Prep coverage={coverage:.3f} infra_failure_rate={infra_failure_rate:.3f} run_validity={run_validity}.",
            file=sys.stderr,
        )
    if failed <= 0:
        return

    classes = stats.get("prep_failure_classes") or stats.get("prep_error_classes", {})
    if not isinstance(classes, Mapping) or not classes:
        return

    top = sorted(classes.items(), key=lambda item: int(item[1]), reverse=True)[:5]
    summary = ", ".join(f"{name}={count}" for name, count in top)
    print(f"Prep failures by error type: {summary}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Polaris RAG evaluation with modern RAGAS metrics")

    parser.add_argument(
        "--config-file",
        "-c",
        required=True,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override raw evaluation dataset input path",
    )
    parser.add_argument(
        "--prepared-path",
        default=None,
        help="Override prepared dataset path (JSON/JSONL)",
    )
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="If set, use prepared dataset when available",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare dataset rows and exit before RAGAS scoring",
    )
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=None,
        help="Override response-generation worker count while preparing dataset",
    )
    parser.add_argument(
        "--generation-max-attempts",
        type=int,
        default=None,
        help="Maximum retry attempts per row during generation (1 disables retries)",
    )
    parser.add_argument(
        "--generation-retry-initial-backoff",
        type=float,
        default=None,
        help="Initial backoff delay (seconds) before generation retries",
    )
    parser.add_argument(
        "--generation-retry-max-backoff",
        type=float,
        default=None,
        help="Maximum backoff delay (seconds) for generation retries",
    )
    parser.add_argument(
        "--generation-retry-jitter",
        type=float,
        default=None,
        help="Random jitter (seconds) added to generation retry delays",
    )
    parser.add_argument(
        "--generation-retry-on-empty-response",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Retry generation when response is empty. "
            "Use --no-generation-retry-on-empty-response to disable."
        ),
    )
    parser.add_argument(
        "--generation-mode",
        choices=("pipeline", "api"),
        default=None,
        help="Preparation mode: in-process pipeline or HTTP API",
    )
    parser.add_argument(
        "--evaluation-policy",
        choices=(EVAL_POLICY_OFFICIAL, EVAL_POLICY_DIAGNOSTIC),
        default=None,
        help="Evaluation policy controlling deadlines and retry semantics",
    )
    parser.add_argument(
        "--replay-failures-from",
        default=None,
        help="Prepared rows file used to replay only failed rows in diagnostic mode",
    )
    parser.add_argument(
        "--query-api-url",
        default=None,
        help="Query API endpoint used when generation mode is 'api' (e.g., http://127.0.0.1:8000/v1/query)",
    )
    parser.add_argument(
        "--query-api-timeout",
        type=float,
        default=None,
        help="HTTP timeout (seconds) for query API calls in generation mode 'api'",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric names to request",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for scores/artifacts",
    )
    parser.add_argument(
        "--no-tune-concurrency",
        action="store_true",
        help="Disable adaptive worker tuning",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable MLflow tracking for this evaluation run. "
            "Overrides mlflow.enabled from config."
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=None,
        help="Override MLflow experiment name for this run.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name for the parent evaluation run.",
    )
    parser.add_argument(
        "--trace-evaluator-llm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Log raw evaluator LLM prompt/response traces during RAGAS evaluation. "
            "Requires MLflow tracing; use --no-trace-evaluator-llm to disable."
        ),
    )

    return parser.parse_args()


def _resolve_output_dir(eval_cfg: Mapping[str, Any], cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    cfg_value = eval_cfg.get("output_dir")
    if cfg_value:
        return Path(str(cfg_value)).expanduser().resolve()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (Path("data") / "eval_runs" / stamp).resolve()


def _resolve_generation_retry_policy(
    generation_cfg: Mapping[str, Any], args: argparse.Namespace
) -> PrepRetryPolicy:
    retry_cfg = _as_mapping(generation_cfg.get("retries", {}))

    max_attempts = (
        int(getattr(args, "generation_max_attempts"))
        if getattr(args, "generation_max_attempts", None) is not None
        else _as_int(retry_cfg.get("max_attempts"), 1)
    )
    initial_backoff_seconds = (
        float(getattr(args, "generation_retry_initial_backoff"))
        if getattr(args, "generation_retry_initial_backoff", None) is not None
        else _as_float(retry_cfg.get("initial_backoff_seconds"), 1.0)
    )
    max_backoff_seconds = (
        float(getattr(args, "generation_retry_max_backoff"))
        if getattr(args, "generation_retry_max_backoff", None) is not None
        else _as_float(retry_cfg.get("max_backoff_seconds"), 8.0)
    )
    jitter_seconds = (
        float(getattr(args, "generation_retry_jitter"))
        if getattr(args, "generation_retry_jitter", None) is not None
        else _as_float(retry_cfg.get("jitter_seconds"), 0.25)
    )
    retry_on_empty_response = (
        bool(getattr(args, "generation_retry_on_empty_response"))
        if getattr(args, "generation_retry_on_empty_response", None) is not None
        else _as_bool(retry_cfg.get("retry_on_empty_response"), True)
    )

    return PrepRetryPolicy(
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
        jitter_seconds=jitter_seconds,
        retry_on_empty_response=retry_on_empty_response,
    ).normalized()


def _resolve_evaluation_policy(
    generation_cfg: Mapping[str, Any],
    args: argparse.Namespace,
) -> str:
    configured = getattr(args, "evaluation_policy", None) or generation_cfg.get("policy")
    return normalize_evaluation_policy(configured, default=EVAL_POLICY_OFFICIAL)


def _resolve_evaluator_llm_tracing(
    eval_cfg: Mapping[str, Any],
    args: argparse.Namespace,
) -> bool:
    tracing_cfg = _as_mapping(eval_cfg.get("tracing", {}))
    cli_value = getattr(args, "trace_evaluator_llm", None)
    if cli_value is not None:
        return bool(cli_value)
    return _as_bool(tracing_cfg.get("evaluator_llm"), False)


def _build_evaluator_trace_factory(
    stage_context: EvaluationStageContext | None,
    *,
    enabled: bool,
):
    if not enabled or stage_context is None:
        return None

    return lambda name, inputs, attributes=None: stage_context.open_detached_mirrored_span(
        name,
        inputs=inputs,
        attributes=attributes,
        include_child_trace=True,
    )


def _resolve_generation_deadlines(
    generation_cfg: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    evaluation_policy: str,
):
    client_total_override = getattr(args, "query_api_timeout", None)
    return resolve_evaluation_deadlines(
        generation_cfg,
        policy=evaluation_policy,
        client_total_override=client_total_override,
    )


def _effective_retry_policy(
    retry_policy: PrepRetryPolicy,
    *,
    evaluation_policy: str,
    generation_mode: str,
) -> PrepRetryPolicy:
    if normalize_evaluation_policy(evaluation_policy, default=EVAL_POLICY_OFFICIAL) != EVAL_POLICY_OFFICIAL:
        return retry_policy
    if generation_mode != "api":
        return retry_policy
    return PrepRetryPolicy(
        max_attempts=1,
        initial_backoff_seconds=retry_policy.initial_backoff_seconds,
        max_backoff_seconds=retry_policy.max_backoff_seconds,
        jitter_seconds=retry_policy.jitter_seconds,
        retry_on_empty_response=False,
    ).normalized()


def _raw_examples_from_replay_rows(
    rows: list[dict[str, Any]],
    *,
    query_field: str,
    reference_field: str,
    id_field: str,
) -> list[dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    for row in rows:
        if not _source_error(row):
            continue
        metadata = _row_metadata(row)
        replay_rows.append(
            {
                id_field: str(row.get("id", "") or ""),
                query_field: str(row.get("user_input", "") or ""),
                reference_field: str(row.get("reference", "") or ""),
                "metadata": dict(_as_mapping(metadata.get("original_metadata", {}))),
            }
        )
    return replay_rows


def _resolve_prepared_rows(
    *,
    cfg: GlobalConfig,
    args: argparse.Namespace,
    eval_cfg: Mapping[str, Any],
    show_progress: bool,
    extra_api_headers: Mapping[str, str] | None = None,
    stage_context: EvaluationStageContext | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_cfg = _as_mapping(eval_cfg.get("dataset", {}))
    generation_cfg = _as_mapping(eval_cfg.get("generation", {}))
    replay_failures_from = getattr(args, "replay_failures_from", None)

    dataset_path = args.dataset_path or dataset_cfg.get("input_path")
    if not dataset_path and not replay_failures_from:
        raise ValueError("No evaluation dataset path configured. Set evaluation.dataset.input_path or pass --dataset-path.")

    prepared_path_raw = args.prepared_path or dataset_cfg.get("prepared_path")
    prepared_path = Path(str(prepared_path_raw)).expanduser().resolve() if prepared_path_raw else None

    reuse_prepared = args.reuse_prepared or _as_bool(dataset_cfg.get("reuse_prepared"), False)

    query_field = str(dataset_cfg.get("query_field", "query"))
    reference_field = str(dataset_cfg.get("reference_field", "expected_answer"))
    id_field = str(dataset_cfg.get("id_field", "id"))
    evaluation_policy = _resolve_evaluation_policy(generation_cfg, args)
    deadlines = _resolve_generation_deadlines(
        generation_cfg,
        args,
        evaluation_policy=evaluation_policy,
    )
    max_infra_failure_rate = _max_infra_failure_rate(eval_cfg)

    generation_workers = (
        int(args.generation_workers)
        if args.generation_workers is not None
        else _as_int(generation_cfg.get("workers"), 1)
    )
    generation_mode_arg = getattr(args, "generation_mode", None)
    generation_mode = str(generation_mode_arg or generation_cfg.get("mode", "pipeline")).strip().lower()
    if generation_mode not in {"pipeline", "api"}:
        raise ValueError(f"Unsupported evaluation.generation.mode={generation_mode!r}. Expected 'pipeline' or 'api'.")
    retry_policy = _effective_retry_policy(
        _resolve_generation_retry_policy(generation_cfg, args),
        evaluation_policy=evaluation_policy,
        generation_mode=generation_mode,
    )

    manifest: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path).expanduser().resolve()) if dataset_path else None,
        "prepared_path": str(prepared_path) if prepared_path else None,
        "reuse_prepared": bool(reuse_prepared),
        "query_field": query_field,
        "reference_field": reference_field,
        "id_field": id_field,
        "generation_workers": generation_workers,
        "generation_mode": generation_mode,
        "evaluation_policy": evaluation_policy,
        "generation_deadlines": deadlines.to_dict(),
        "max_infra_failure_rate": max_infra_failure_rate,
        "generation_retries": {
            "max_attempts": int(retry_policy.max_attempts),
            "initial_backoff_seconds": float(retry_policy.initial_backoff_seconds),
            "max_backoff_seconds": float(retry_policy.max_backoff_seconds),
            "jitter_seconds": float(retry_policy.jitter_seconds),
            "retry_on_empty_response": bool(retry_policy.retry_on_empty_response),
        },
    }

    request_trace_factory = None
    effective_extra_api_headers = dict(extra_api_headers or {})
    if stage_context is not None:
        effective_extra_api_headers.update(stage_context.correlation_headers())
        if generation_mode == "api":
            request_trace_factory = (
                lambda name, inputs, attributes=None: stage_context.open_detached_mirrored_span(
                    name,
                    inputs=inputs,
                    attributes=attributes,
                    include_child_trace=True,
                )
            )
        else:
            request_trace_factory = (
                lambda name, inputs, attributes=None: stage_context.open_detached_mirrored_span(
                    name,
                    inputs=inputs,
                    attributes=attributes,
                    include_child_trace=False,
                )
            )

    if reuse_prepared and prepared_path and prepared_path.exists():
        rows = load_prepared_rows(prepared_path)
        manifest.update(
            _prep_stats(
                rows,
                elapsed_seconds=0.0,
                evaluation_policy=evaluation_policy,
                max_infra_failure_rate=max_infra_failure_rate,
            )
        )
        manifest["prepared_source"] = "existing"
        return rows, manifest

    if replay_failures_from:
        if evaluation_policy != EVAL_POLICY_DIAGNOSTIC:
            raise ValueError("--replay-failures-from can only be used with --evaluation-policy diagnostic.")
        replay_source_path = Path(str(replay_failures_from)).expanduser().resolve()
        replay_source_rows = load_prepared_rows(replay_source_path)
        raw_examples = _raw_examples_from_replay_rows(
            replay_source_rows,
            query_field=query_field,
            reference_field=reference_field,
            id_field=id_field,
        )
        manifest["replay_failures_from"] = str(replay_source_path)
        manifest["replay_selected_rows"] = len(raw_examples)
    else:
        raw_examples = load_raw_examples(dataset_path)
    raise_exceptions = _as_bool(generation_cfg.get("raise_exceptions"), False)

    prep_renderer = (
        _PrepProgressRenderer(interactive=bool(show_progress and sys.stderr.isatty()))
        if show_progress
        else None
    )
    prep_started_at = time.perf_counter()

    try:
        if generation_mode == "api":
            query_api_url_arg = getattr(args, "query_api_url", None)
            api_url = str(query_api_url_arg or generation_cfg.get("api_url") or "").strip()
            if not api_url:
                raise ValueError(
                    "evaluation.generation.mode is 'api' but no API URL was provided. "
                    "Set evaluation.generation.api_url or pass --query-api-url."
                )
            timeout_seconds = deadlines.client_total_seconds
            api_headers = {
                str(k): str(v)
                for k, v in _as_mapping(generation_cfg.get("api_headers", {})).items()
            }
            if effective_extra_api_headers:
                api_headers.update({str(k): str(v) for k, v in effective_extra_api_headers.items()})
            api_headers[POLARIS_TIMEOUT_HEADER] = str(deadlines.server_total_ms)
            api_headers[POLARIS_EVAL_POLICY_HEADER] = evaluation_policy

            rows = build_prepared_rows_from_api(
                raw_examples=raw_examples,
                api_url=api_url,
                query_field=query_field,
                reference_field=reference_field,
                id_field=id_field,
                generation_workers=max(1, generation_workers),
                raise_exceptions=raise_exceptions,
                timeout_seconds=timeout_seconds,
                headers=api_headers,
                retry_policy=retry_policy,
                progress_callback=prep_renderer.update if prep_renderer else None,
                trace_factory=request_trace_factory,
                policy=evaluation_policy,
                budget_ms=deadlines.server_total_ms,
            )
            manifest["query_api_url"] = api_url
            manifest["query_api_timeout"] = timeout_seconds
            manifest["server_timeout_ms"] = deadlines.server_total_ms
            manifest["query_api_header_keys"] = sorted(api_headers.keys())
        else:
            container = build_container(cfg)
            rows = build_prepared_rows(
                raw_examples=raw_examples,
                pipeline=container.pipeline,
                query_field=query_field,
                reference_field=reference_field,
                id_field=id_field,
                generation_workers=max(1, generation_workers),
                llm_generate_overrides=_as_mapping(generation_cfg.get("llm_generate", {})),
                raise_exceptions=raise_exceptions,
                retry_policy=retry_policy,
                progress_callback=prep_renderer.update if prep_renderer else None,
                trace_factory=request_trace_factory,
                policy=evaluation_policy,
                budget_ms=deadlines.server_total_ms,
            )
    finally:
        if prep_renderer:
            prep_renderer.finish()

    prep_elapsed = max(0.0, time.perf_counter() - prep_started_at)
    prep_stats = _prep_stats(
        rows,
        elapsed_seconds=prep_elapsed,
        evaluation_policy=evaluation_policy,
        max_infra_failure_rate=max_infra_failure_rate,
    )
    manifest.update(prep_stats)
    if show_progress:
        _print_prep_summary(prep_stats)

    if prepared_path:
        persisted = persist_prepared_rows(rows, prepared_path)
        manifest["persisted_prepared_path"] = str(persisted)

    manifest["prepared_source"] = "diagnostic_replay" if replay_failures_from else "generated"
    return rows, manifest


def _collect_quality_metric_aggregates(result: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}

    scores_df = getattr(result, "scores_df", None)
    selected_metrics = list(getattr(result, "selected_metrics", []))
    if scores_df is None or not selected_metrics:
        return metrics

    for metric_name in selected_metrics:
        if metric_name not in scores_df.columns:
            continue

        numeric_values: list[float] = []
        for value in scores_df[metric_name].tolist():
            if isinstance(value, bool):
                numeric_values.append(float(int(value)))
                continue
            if isinstance(value, (int, float)):
                casted = float(value)
                if math.isfinite(casted):
                    numeric_values.append(casted)

        if not numeric_values:
            continue

        mean_value = sum(numeric_values) / len(numeric_values)
        variance = sum((x - mean_value) ** 2 for x in numeric_values) / len(numeric_values)

        metrics[f"quality.mean.{metric_name}"] = float(mean_value)
        metrics[f"quality.std.{metric_name}"] = float(math.sqrt(variance))
        metrics[f"quality.min.{metric_name}"] = float(min(numeric_values))
        metrics[f"quality.max.{metric_name}"] = float(max(numeric_values))

    return metrics


def _collect_system_metrics(
    *,
    result: Any,
    dataset_manifest: Mapping[str, Any],
    tune_concurrency: bool,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "system.eval.duration_seconds": float(getattr(result, "duration_seconds", 0.0)),
        "system.eval.failure_rate": float(getattr(result, "failure_rate", 0.0)),
        "system.eval.rows": float(len(getattr(result, "scores_df", []))),
        "system.eval.selected_max_workers": float(getattr(result, "selected_max_workers", 0)),
        "system.eval.batch_size": float(getattr(result, "batch_size", 0)),
        "system.eval.skipped_metrics_count": float(len(getattr(result, "skipped_metrics", []))),
        "system.eval.tuning_trial_count": float(len(getattr(result, "tuning_trials", []))),
        "system.eval.tune_concurrency_enabled": 1.0 if tune_concurrency else 0.0,
    }

    prep_metric_keys = (
        "prep_total_rows",
        "prep_success_rows",
        "prep_failed_rows",
        "prep_elapsed_seconds",
        "prep_rate_rows_per_second",
        "prep_coverage",
        "prep_empty_response_rate",
        "infra_failure_rate",
        "max_infra_failure_rate",
    )
    for key in prep_metric_keys:
        value = dataset_manifest.get(key)
        if isinstance(value, bool):
            metrics[f"system.prep.{key}"] = float(int(value))
        elif isinstance(value, (int, float)):
            metrics[f"system.prep.{key}"] = float(value)

    run_validity = str(dataset_manifest.get("run_validity", "") or "").strip().upper()
    if run_validity:
        metrics["system.prep.run_validity_valid"] = 1.0 if run_validity == "VALID" else 0.0
        metrics["system.prep.run_validity_invalid"] = 1.0 if run_validity == "INVALID" else 0.0

    return metrics


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "Logging MLflow dataset inputs requires pandas. Install the evaluation/tracking extras first."
        ) from exc
    return pd


def _build_mlflow_dataset(
    mlflow: Any,
    rows: list[dict[str, Any]],
    *,
    source: Path,
    dataset_name: str,
) -> Any:
    pd = _import_pandas()
    frame = pd.DataFrame.from_records(rows)
    return mlflow.data.from_pandas(
        frame,
        source=str(source),
        name=dataset_name,
    )


def _infer_mlflow_dataset_context(dataset_path: Path) -> tuple[str, str]:
    stem = dataset_path.stem.lower()
    validation_suffixes = (".dev", "_dev", "-dev", ".validation", "_validation", "-validation", ".val", "_val", "-val")
    testing_suffixes = (".test", "_test", "-test", ".testing", "_testing", "-testing")

    if stem.endswith(validation_suffixes):
        return "validation", "dev"
    if stem.endswith(testing_suffixes):
        return "testing", "test"
    return "evaluation", "dataset"


def _log_input_dataset_to_mlflow(
    tracking: EvaluationTrackingContext,
    dataset_manifest: Mapping[str, Any],
) -> None:
    if not tracking.enabled:
        return

    mlflow = getattr(tracking, "_mlflow", None)
    if mlflow is None:
        return

    dataset_path_raw = str(dataset_manifest.get("dataset_path") or "").strip()
    if not dataset_path_raw:
        return

    dataset_path = Path(dataset_path_raw).expanduser().resolve()
    if not dataset_path.exists() or not dataset_path.is_file():
        logger.warning("Skipping MLflow dataset input logging because dataset file is missing: %s", dataset_path)
        return

    try:
        raw_examples = load_raw_examples(dataset_path)
    except Exception:
        logger.warning("Failed to load raw dataset for MLflow input logging: %s", dataset_path, exc_info=True)
        return

    context, split = _infer_mlflow_dataset_context(dataset_path)

    try:
        dataset = _build_mlflow_dataset(
            mlflow,
            raw_examples,
            source=dataset_path,
            dataset_name=dataset_path.stem,
        )
    except Exception:
        logger.warning("Failed to construct MLflow dataset input for: %s", dataset_path, exc_info=True)
        return

    tracking.log_input(
        dataset,
        context=context,
        tags={
            "split": split,
            "rows": len(raw_examples),
            "dataset_path": str(dataset_path),
            "prepared_source": str(dataset_manifest.get("prepared_source", "")),
        },
    )


def main() -> None:
    _configure_logging()
    args = parse_args()
    show_progress = not args.no_progress
    interactive_progress = show_progress and sys.stderr.isatty()

    cfg = GlobalConfig.load(args.config_file)
    eval_cfg = _as_mapping(_as_mapping(cfg.raw).get("evaluation", {}))
    output_dir = _resolve_output_dir(eval_cfg, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if show_progress and not interactive_progress:
        logger.info("Non-interactive stderr detected; using line logs instead of live progress bars.")

    tracking_cfg = load_mlflow_runtime_config(cfg)
    tracking = EvaluationTrackingContext(
        tracking_cfg,
        enabled_override=getattr(args, "mlflow", None),
        experiment_override=getattr(args, "mlflow_experiment", None),
    )

    config_file = Path(args.config_file).expanduser().resolve()
    requested_metrics = _parse_metrics(args.metrics)
    tune_concurrency = not args.no_tune_concurrency
    prepare_only = bool(getattr(args, "prepare_only", False))
    trace_evaluator_llm_requested = False
    trace_evaluator_llm = False
    if not prepare_only:
        trace_evaluator_llm_requested = _resolve_evaluator_llm_tracing(eval_cfg, args)
        trace_evaluator_llm = bool(
            trace_evaluator_llm_requested
            and tracking.enabled
            and tracking.runtime_config.tracing.enabled
        )
    if trace_evaluator_llm_requested and not trace_evaluator_llm:
        logger.warning(
            "Evaluator LLM tracing was requested but MLflow tracing is disabled; continuing without evaluator traces."
        )

    with tracking.open(
        run_name=getattr(args, "mlflow_run_name", None),
        extra_tags={
            "entrypoint": "polaris-eval",
            "config_file": str(config_file),
        },
        strict=True,
    ):
        tracking.log_flat_params(_as_mapping(cfg.raw), prefix="config")
        tracking.log_params(
            {
                "input.config_file": str(config_file),
                "input.output_dir": str(output_dir),
                "runtime.show_progress": str(show_progress),
                "runtime.interactive_progress": str(interactive_progress),
                "runtime.prepare_only": str(prepare_only),
                "runtime.tune_concurrency": str(tune_concurrency),
                "runtime.trace_evaluator_llm": str(trace_evaluator_llm),
            }
        )
        tracking.log_artifact(config_file, artifact_path="inputs")

        with tracking.stage("dataset_preparation") as dataset_stage:
            with _phase_heartbeat("Dataset preparation", interval_seconds=60.0):
                prepared_rows, dataset_manifest = _resolve_prepared_rows(
                    cfg=cfg,
                    args=args,
                    eval_cfg=eval_cfg,
                    show_progress=show_progress,
                    extra_api_headers=tracking.trace_headers(),
                    stage_context=dataset_stage,
                )
            tracking.log_flat_params(dataset_manifest, prefix="dataset")
            _log_input_dataset_to_mlflow(tracking, dataset_manifest)
            tracking.log_metrics(
                {
                    "system.prep.prep_total_rows": dataset_manifest.get("prep_total_rows", 0),
                    "system.prep.prep_success_rows": dataset_manifest.get("prep_success_rows", 0),
                    "system.prep.prep_failed_rows": dataset_manifest.get("prep_failed_rows", 0),
                    "system.prep.prep_elapsed_seconds": dataset_manifest.get("prep_elapsed_seconds", 0.0),
                    "system.prep.prep_rate_rows_per_second": dataset_manifest.get("prep_rate_rows_per_second", 0.0),
                    "system.prep.prep_coverage": dataset_manifest.get("prep_coverage", 0.0),
                    "system.prep.infra_failure_rate": dataset_manifest.get("infra_failure_rate", 0.0),
                    "system.prep.prep_empty_response_rate": dataset_manifest.get("prep_empty_response_rate", 0.0),
                }
            )
            if dataset_manifest.get("run_validity") == "INVALID":
                logger.warning(
                    "Evaluation run is marked INVALID after dataset preparation (infra_failure_rate=%.3f > %.3f).",
                    float(dataset_manifest.get("infra_failure_rate", 0.0)),
                    float(dataset_manifest.get("max_infra_failure_rate", 0.0)),
                )
            tracking.log_json_artifact(
                dataset_manifest,
                output_path=output_dir / "dataset_manifest.json",
                artifact_path="dataset",
            )

        prepared_rows_path = _write_prepared_rows_artifact(
            output_dir=output_dir,
            prepared_rows=prepared_rows,
            tracking=tracking,
        )
        _write_runtime_snapshots(
            output_dir=output_dir,
            config_payload=_as_mapping(cfg.raw),
            tracking=tracking,
        )

        if prepare_only:
            print("Dataset preparation complete.")
            print(f"Rows: {len(prepared_rows)}")
            print(f"Prep run validity: {dataset_manifest.get('run_validity', 'UNKNOWN')}")
            print(f"Prepared rows: {prepared_rows_path}")
            return

        from polaris_rag.evaluation.evaluator import Evaluator, write_outputs

        dataset = to_evaluation_dataset(prepared_rows)

        with tracking.stage("ragas_evaluation") as evaluation_stage:
            evaluator = Evaluator.from_global_config(
                cfg,
                requested_metrics=requested_metrics,
                trace_evaluator_llm=trace_evaluator_llm,
                trace_factory=_build_evaluator_trace_factory(
                    evaluation_stage,
                    enabled=trace_evaluator_llm,
                ),
            )
            with evaluation_stage.open_active_mirrored_span(
                "polaris.ragas_evaluation.execute",
                attributes={"stage": "ragas_evaluation"},
                inputs={
                    "rows": len(prepared_rows),
                    "tune_concurrency": tune_concurrency,
                    "show_progress": interactive_progress,
                },
            ) as eval_trace:
                with _phase_heartbeat("RAGAS evaluation", interval_seconds=60.0):
                    result = evaluator.evaluate(
                        dataset=dataset,
                        source_rows=prepared_rows,
                        tune_concurrency=tune_concurrency,
                        show_progress=interactive_progress,
                    )
                eval_trace.set_outputs(
                    {
                        "rows": len(getattr(result, "scores_df", [])),
                        "selected_metrics": list(getattr(result, "selected_metrics", [])),
                        "selected_max_workers": int(getattr(result, "selected_max_workers", 0)),
                        "failure_rate": float(getattr(result, "failure_rate", 0.0)),
                    }
                )
            tracking.log_metrics(
                _collect_system_metrics(
                    result=result,
                    dataset_manifest=dataset_manifest,
                    tune_concurrency=tune_concurrency,
                )
            )
            tracking.log_metrics(_collect_quality_metric_aggregates(result))

        artifacts = write_outputs(
            result=result,
            output_dir=output_dir,
            extra_manifest={
                "config_file": str(config_file),
                "dataset": dataset_manifest,
                "tune_concurrency": tune_concurrency,
                "trace_evaluator_llm": trace_evaluator_llm,
                "mlflow_parent_run_id": tracking.run_id,
            },
        )

        for artifact_path in artifacts.values():
            tracking.log_artifact(artifact_path, artifact_path="outputs")

    print("Evaluation complete.")
    print(f"Rows: {len(result.scores_df)}")
    print(f"Selected metrics: {', '.join(result.selected_metrics)}")
    print(f"Skipped metrics: {len(result.skipped_metrics)}")
    print(f"Selected max_workers: {result.selected_max_workers}")
    print(f"Failure rate: {result.failure_rate:.4f}")
    print(f"Prep run validity: {dataset_manifest.get('run_validity', 'UNKNOWN')}")
    print("Artifacts:")
    for name, path in artifacts.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
