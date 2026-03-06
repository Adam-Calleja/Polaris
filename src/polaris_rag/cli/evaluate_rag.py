"""Evaluation CLI entrypoint for Polaris RAG."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

from polaris_rag.app.container import build_container
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
    build_environment_snapshot,
    load_mlflow_runtime_config,
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


class _PrepProgressRenderer:
    """Render dataset-preparation progress as a single-line live bar."""

    def __init__(self, *, width: int = 24):
        self.width = max(8, int(width))
        self._active = False

    def update(self, event: PrepProgressEvent) -> None:
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
        self._active = True
        print(f"\r{line}", end="", file=sys.stderr, flush=True)

    def finish(self) -> None:
        if self._active:
            print(file=sys.stderr, flush=True)
            self._active = False


def _source_error(row: Mapping[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("source_error")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _error_class(source_error: str) -> str:
    head = source_error.split(":", 1)[0].strip()
    return head or "UnknownError"


def _prep_stats(rows: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    total = len(rows)
    failures = 0
    error_classes: Counter[str] = Counter()
    for row in rows:
        err = _source_error(row)
        if err:
            failures += 1
            error_classes[_error_class(err)] += 1

    successes = total - failures
    rate = (total / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    return {
        "prep_total_rows": total,
        "prep_success_rows": successes,
        "prep_failed_rows": failures,
        "prep_elapsed_seconds": elapsed_seconds,
        "prep_rate_rows_per_second": rate,
        "prep_error_classes": dict(error_classes),
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
    if failed <= 0:
        return

    classes = stats.get("prep_error_classes", {})
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


def _resolve_prepared_rows(
    *,
    cfg: GlobalConfig,
    args: argparse.Namespace,
    eval_cfg: Mapping[str, Any],
    show_progress: bool,
    extra_api_headers: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_cfg = _as_mapping(eval_cfg.get("dataset", {}))
    generation_cfg = _as_mapping(eval_cfg.get("generation", {}))

    dataset_path = args.dataset_path or dataset_cfg.get("input_path")
    if not dataset_path:
        raise ValueError("No evaluation dataset path configured. Set evaluation.dataset.input_path or pass --dataset-path.")

    prepared_path_raw = args.prepared_path or dataset_cfg.get("prepared_path")
    prepared_path = Path(str(prepared_path_raw)).expanduser().resolve() if prepared_path_raw else None

    reuse_prepared = args.reuse_prepared or _as_bool(dataset_cfg.get("reuse_prepared"), False)

    query_field = str(dataset_cfg.get("query_field", "query"))
    reference_field = str(dataset_cfg.get("reference_field", "expected_answer"))
    id_field = str(dataset_cfg.get("id_field", "id"))
    retry_policy = _resolve_generation_retry_policy(generation_cfg, args)

    generation_workers = (
        int(args.generation_workers)
        if args.generation_workers is not None
        else _as_int(generation_cfg.get("workers"), 1)
    )
    generation_mode_arg = getattr(args, "generation_mode", None)
    generation_mode = str(generation_mode_arg or generation_cfg.get("mode", "pipeline")).strip().lower()
    if generation_mode not in {"pipeline", "api"}:
        raise ValueError(f"Unsupported evaluation.generation.mode={generation_mode!r}. Expected 'pipeline' or 'api'.")

    manifest: dict[str, Any] = {
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "prepared_path": str(prepared_path) if prepared_path else None,
        "reuse_prepared": bool(reuse_prepared),
        "query_field": query_field,
        "reference_field": reference_field,
        "id_field": id_field,
        "generation_workers": generation_workers,
        "generation_mode": generation_mode,
        "generation_retries": {
            "max_attempts": int(retry_policy.max_attempts),
            "initial_backoff_seconds": float(retry_policy.initial_backoff_seconds),
            "max_backoff_seconds": float(retry_policy.max_backoff_seconds),
            "jitter_seconds": float(retry_policy.jitter_seconds),
            "retry_on_empty_response": bool(retry_policy.retry_on_empty_response),
        },
    }

    if reuse_prepared and prepared_path and prepared_path.exists():
        rows = load_prepared_rows(prepared_path)
        manifest.update(_prep_stats(rows, elapsed_seconds=0.0))
        manifest["prepared_source"] = "existing"
        return rows, manifest

    raw_examples = load_raw_examples(dataset_path)
    raise_exceptions = _as_bool(generation_cfg.get("raise_exceptions"), False)

    prep_renderer = _PrepProgressRenderer() if show_progress else None
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
            timeout_seconds = (
                float(getattr(args, "query_api_timeout"))
                if getattr(args, "query_api_timeout", None) is not None
                else _as_float(generation_cfg.get("api_timeout"), 120.0)
            )
            api_headers = {
                str(k): str(v)
                for k, v in _as_mapping(generation_cfg.get("api_headers", {})).items()
            }
            if extra_api_headers:
                api_headers.update({str(k): str(v) for k, v in extra_api_headers.items()})

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
            )
            manifest["query_api_url"] = api_url
            manifest["query_api_timeout"] = timeout_seconds
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
            )
    finally:
        if prep_renderer:
            prep_renderer.finish()

    prep_elapsed = max(0.0, time.perf_counter() - prep_started_at)
    prep_stats = _prep_stats(rows, elapsed_seconds=prep_elapsed)
    manifest.update(prep_stats)
    if show_progress:
        _print_prep_summary(prep_stats)

    if prepared_path:
        persisted = persist_prepared_rows(rows, prepared_path)
        manifest["persisted_prepared_path"] = str(persisted)

    manifest["prepared_source"] = "generated"
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
    )
    for key in prep_metric_keys:
        value = dataset_manifest.get(key)
        if isinstance(value, bool):
            metrics[f"system.prep.{key}"] = float(int(value))
        elif isinstance(value, (int, float)):
            metrics[f"system.prep.{key}"] = float(value)

    return metrics


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    from polaris_rag.evaluation.evaluator import Evaluator, write_outputs

    cfg = GlobalConfig.load(args.config_file)
    eval_cfg = _as_mapping(_as_mapping(cfg.raw).get("evaluation", {}))
    output_dir = _resolve_output_dir(eval_cfg, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_cfg = load_mlflow_runtime_config(cfg)
    tracking = EvaluationTrackingContext(
        tracking_cfg,
        enabled_override=getattr(args, "mlflow", None),
        experiment_override=getattr(args, "mlflow_experiment", None),
    )

    config_file = Path(args.config_file).expanduser().resolve()
    requested_metrics = _parse_metrics(args.metrics)
    tune_concurrency = not args.no_tune_concurrency

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
                "runtime.tune_concurrency": str(tune_concurrency),
            }
        )
        tracking.log_artifact(config_file, artifact_path="inputs")

        with tracking.stage("dataset_preparation"):
            prepared_rows, dataset_manifest = _resolve_prepared_rows(
                cfg=cfg,
                args=args,
                eval_cfg=eval_cfg,
                show_progress=show_progress,
                extra_api_headers=tracking.trace_headers(),
            )
            tracking.log_flat_params(dataset_manifest, prefix="dataset")
            tracking.log_metrics(
                {
                    "system.prep.prep_total_rows": dataset_manifest.get("prep_total_rows", 0),
                    "system.prep.prep_success_rows": dataset_manifest.get("prep_success_rows", 0),
                    "system.prep.prep_failed_rows": dataset_manifest.get("prep_failed_rows", 0),
                    "system.prep.prep_elapsed_seconds": dataset_manifest.get("prep_elapsed_seconds", 0.0),
                    "system.prep.prep_rate_rows_per_second": dataset_manifest.get("prep_rate_rows_per_second", 0.0),
                }
            )
            tracking.log_json_artifact(
                dataset_manifest,
                output_path=output_dir / "dataset_manifest.json",
                artifact_path="dataset",
            )

        dataset = to_evaluation_dataset(prepared_rows)
        evaluator = Evaluator.from_global_config(
            cfg,
            requested_metrics=requested_metrics,
        )

        with tracking.stage("ragas_evaluation"):
            result = evaluator.evaluate(
                dataset=dataset,
                source_rows=prepared_rows,
                tune_concurrency=tune_concurrency,
                show_progress=show_progress,
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
                "mlflow_parent_run_id": tracking.run_id,
            },
        )

        for artifact_path in artifacts.values():
            tracking.log_artifact(artifact_path, artifact_path="outputs")

        prepared_rows_path = output_dir / "prepared_rows.json"
        prepared_rows_path.write_text(
            json.dumps(prepared_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tracking.log_artifact(prepared_rows_path, artifact_path="outputs")

        tracking.log_json_artifact(
            build_environment_snapshot(),
            output_path=output_dir / "env_snapshot.json",
            artifact_path="outputs",
        )
        tracking.log_json_artifact(
            _as_mapping(cfg.raw),
            output_path=output_dir / "config_snapshot.json",
            artifact_path="inputs",
        )

    print("Evaluation complete.")
    print(f"Rows: {len(result.scores_df)}")
    print(f"Selected metrics: {', '.join(result.selected_metrics)}")
    print(f"Skipped metrics: {len(result.skipped_metrics)}")
    print(f"Selected max_workers: {result.selected_max_workers}")
    print(f"Failure rate: {result.failure_rate:.4f}")
    print("Artifacts:")
    for name, path in artifacts.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
