"""Manifest-driven helpers for reproducible experiment automation.

This module provides a lightweight orchestration layer over the existing
Polaris CLI scripts. It is intentionally thin: the heavy lifting still happens
inside the established ingestion, evaluation, tuning, and analysis entrypoints.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIMARY_METRICS: tuple[str, ...] = (
    "factual_correctness",
    "faithfulness",
    "context_recall",
    "context_precision_without_reference",
)
EVALUATION_GRID_STAGE = "evaluation_grid"
SPLIT_STAGE = "split"
TUNE_VALIDITY_STAGE = "tune_validity_reranker"
BENCHMARK_ANALYSIS_STAGE = "benchmark_analysis"
SUPPORTED_STAGE_TYPES: frozenset[str] = frozenset(
    {
        EVALUATION_GRID_STAGE,
        SPLIT_STAGE,
        TUNE_VALIDITY_STAGE,
        BENCHMARK_ANALYSIS_STAGE,
    }
)
SUPPORTED_INGEST_KINDS: frozenset[str] = frozenset({"html", "jira", "external_docs"})


def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest YAML file and validate the top-level structure."""

    resolved = Path(path).expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"Experiment manifest {resolved} must contain a top-level mapping.")

    stages = payload.get("stages")
    if not isinstance(stages, Mapping) or not stages:
        raise ValueError("Experiment manifest must define a non-empty 'stages' mapping.")

    manifest = dict(payload)
    manifest["_manifest_path"] = str(resolved)
    manifest["_manifest_dir"] = str(resolved.parent)
    return manifest


def render_stage_condition_config(
    *,
    manifest_path: str | Path,
    stage_name: str,
    condition_name: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Render a generated config overlay for one stage or one stage condition."""

    manifest = load_experiment_manifest(manifest_path)
    stage_spec = _resolve_stage_spec(manifest, stage_name)
    condition_spec = _resolve_condition_spec(stage_spec, condition_name)

    base_config_path = _resolve_required_path(
        manifest,
        manifest.get("base_config"),
        field_name="base_config",
    )
    overrides = _merged_config_overrides(manifest, stage_spec, condition_spec)
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else _default_generated_config_path(stage_name, condition_spec.get("name") if condition_spec else None)
    )
    payload = {"extends": str(base_config_path)}
    payload = _deep_merge(payload, overrides)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return resolved_output


def run_experiment_stage(
    *,
    manifest_path: str | Path,
    stage_name: str,
    selected_conditions: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one manifest stage and persist an execution record."""

    manifest = load_experiment_manifest(manifest_path)
    stage_spec = _resolve_stage_spec(manifest, stage_name)
    stage_type = _stage_type(stage_spec)
    stage_dir = _stage_output_dir(manifest, stage_name)
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage_type == EVALUATION_GRID_STAGE:
        record = _run_evaluation_grid_stage(
            manifest=manifest,
            stage_name=stage_name,
            stage_spec=stage_spec,
            stage_dir=stage_dir,
            selected_conditions=selected_conditions,
            dry_run=dry_run,
        )
    elif stage_type == SPLIT_STAGE:
        record = _run_split_stage(
            manifest=manifest,
            stage_name=stage_name,
            stage_spec=stage_spec,
            stage_dir=stage_dir,
            dry_run=dry_run,
        )
    elif stage_type == TUNE_VALIDITY_STAGE:
        record = _run_tune_validity_stage(
            manifest=manifest,
            stage_name=stage_name,
            stage_spec=stage_spec,
            stage_dir=stage_dir,
            dry_run=dry_run,
        )
    elif stage_type == BENCHMARK_ANALYSIS_STAGE:
        record = _run_benchmark_analysis_stage(
            manifest=manifest,
            stage_name=stage_name,
            stage_spec=stage_spec,
            stage_dir=stage_dir,
            dry_run=dry_run,
        )
    else:  # pragma: no cover - protected by _stage_type validation
        raise ValueError(f"Unsupported stage type {stage_type!r}.")

    record_path = stage_dir / "stage_execution.json"
    record_path.write_text(json.dumps(_json_normalize(record), indent=2), encoding="utf-8")
    return record


def summarize_experiment_stage(
    *,
    manifest_path: str | Path,
    stage_name: str,
    output_dir: str | Path | None = None,
    run_comparison: bool = False,
    comparison_repeat: str = "latest",
    manual_eval_seed: int = 42,
) -> dict[str, Path]:
    """Summarize completed evaluation runs for one stage."""

    manifest = load_experiment_manifest(manifest_path)
    stage_spec = _resolve_stage_spec(manifest, stage_name)
    if _stage_type(stage_spec) != EVALUATION_GRID_STAGE:
        raise ValueError("Stage summarization currently supports only 'evaluation_grid' stages.")

    resolved_output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _stage_output_dir(manifest, stage_name) / "_summary"
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    run_rows = _collect_stage_run_rows(manifest, stage_name, stage_spec)
    if not run_rows:
        raise FileNotFoundError(
            f"No completed evaluation runs were found for stage {stage_name!r}. "
            "Run the stage before summarizing it."
        )

    leaderboard_rows = sorted(
        run_rows,
        key=lambda row: (
            -_sortable_metric(row.get("factual_correctness")),
            -_sortable_metric(row.get("faithfulness")),
            str(row.get("condition_name", "")),
            int(row.get("repeat_index", 0)),
        ),
    )
    metric_names = _metric_names_from_rows(leaderboard_rows)
    aggregate_rows = _aggregate_stage_rows(leaderboard_rows, metric_names)

    artifacts = {
        "leaderboard_csv": _write_csv_records(resolved_output / "leaderboard.csv", leaderboard_rows),
        "leaderboard_json": _write_json(resolved_output / "leaderboard.json", leaderboard_rows),
        "condition_aggregates_csv": _write_csv_records(
            resolved_output / "condition_aggregates.csv",
            aggregate_rows,
        ),
        "summary_manifest_json": _write_json(
            resolved_output / "summary_manifest.json",
            {
                "stage_name": stage_name,
                "rows": len(leaderboard_rows),
                "conditions": sorted({str(row["condition_name"]) for row in leaderboard_rows}),
                "metrics": metric_names,
                "comparison_repeat": comparison_repeat,
            },
        ),
    }

    if run_comparison:
        from polaris_rag.evaluation.run_analysis import load_run_input, write_run_comparison_outputs

        run_specs = _select_run_comparison_specs(leaderboard_rows, comparison_repeat=comparison_repeat)
        run_inputs = [
            load_run_input(condition_name=condition_name, run_dir=run_dir)
            for condition_name, run_dir in run_specs
        ]
        comparison_dir = resolved_output / "run_comparison"
        comparison_artifacts = write_run_comparison_outputs(
            runs=run_inputs,
            output_dir=comparison_dir,
            manual_eval_seed=int(manual_eval_seed),
        )
        artifacts["run_comparison_dir"] = comparison_dir
        for key, value in comparison_artifacts.items():
            artifacts[f"run_comparison_{key}"] = value

    return artifacts


def _run_evaluation_grid_stage(
    *,
    manifest: Mapping[str, Any],
    stage_name: str,
    stage_spec: Mapping[str, Any],
    stage_dir: Path,
    selected_conditions: Sequence[str] | None,
    dry_run: bool,
) -> dict[str, Any]:
    conditions = _stage_conditions(stage_spec, selected_conditions)
    dataset_path = _resolve_required_path(
        manifest,
        stage_spec.get("dataset_path"),
        field_name=f"stages.{stage_name}.dataset_path",
    )
    annotations_path = _resolve_optional_path(manifest, stage_spec.get("annotations_file"))
    repeats = _resolve_positive_int(stage_spec.get("repeats", 1), field_name=f"stages.{stage_name}.repeats")

    execution_conditions: list[dict[str, Any]] = []
    for condition_spec in conditions:
        condition_name = str(condition_spec["name"])
        condition_slug = _slugify(condition_name)
        config_path = render_stage_condition_config(
            manifest_path=str(manifest["_manifest_path"]),
            stage_name=stage_name,
            condition_name=condition_name,
        )
        run_options = _merged_run_options(manifest, stage_spec, condition_spec)

        condition_record = {
            "name": condition_name,
            "slug": condition_slug,
            "config_path": str(config_path),
            "preset": condition_spec.get("preset"),
            "ingestion_commands": [],
            "runs": [],
        }

        ingest_specs = _ingest_specs(condition_spec)
        for ingest_spec in ingest_specs:
            command = _build_ingest_command(
                kind=str(ingest_spec["kind"]),
                config_path=config_path,
                ingest_spec=ingest_spec,
            )
            condition_record["ingestion_commands"].append(command)
            _run_command(command, dry_run=dry_run)

        for repeat_index in range(1, repeats + 1):
            run_dir = stage_dir / condition_slug / f"run_{repeat_index:02d}"
            prepared_path = run_dir / "prepared_input.json"
            command = _build_evaluate_command(
                config_path=config_path,
                dataset_path=dataset_path,
                annotations_path=annotations_path,
                prepared_path=prepared_path,
                output_dir=run_dir,
                preset=condition_spec.get("preset"),
                run_options=run_options,
            )
            _run_command(command, dry_run=dry_run)
            condition_record["runs"].append(
                {
                    "repeat_index": repeat_index,
                    "output_dir": str(run_dir),
                    "prepared_path": str(prepared_path),
                    "command": command,
                }
            )
        execution_conditions.append(condition_record)

    return {
        "stage_name": stage_name,
        "stage_type": EVALUATION_GRID_STAGE,
        "dry_run": dry_run,
        "dataset_path": str(dataset_path),
        "annotations_file": str(annotations_path) if annotations_path else None,
        "repeats": repeats,
        "conditions": execution_conditions,
        "executed_at": _utc_timestamp(),
    }


def _run_split_stage(
    *,
    manifest: Mapping[str, Any],
    stage_name: str,
    stage_spec: Mapping[str, Any],
    stage_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    config_path = render_stage_condition_config(
        manifest_path=str(manifest["_manifest_path"]),
        stage_name=stage_name,
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "create_dev_test_sets.py"),
        "--config-file",
        str(config_path),
    ]
    command.extend(_append_cli_option("--dataset-file", _resolve_required_path(
        manifest,
        stage_spec.get("dataset_file"),
        field_name=f"stages.{stage_name}.dataset_file",
    )))
    command.extend(_append_cli_option("--test-samples-file", _resolve_optional_path(manifest, stage_spec.get("test_samples_file"))))
    for sample_id in stage_spec.get("test_sample_ids", []) or []:
        command.extend(["--test-sample-id", str(sample_id)])
    command.extend(_append_cli_option("--categories-file", _resolve_optional_path(manifest, stage_spec.get("categories_file"))))
    command.extend(_append_cli_option("--annotations-file", _resolve_optional_path(manifest, stage_spec.get("annotations_file"))))
    command.extend(_append_cli_option("--test-size", stage_spec.get("test_size")))
    command.extend(_append_cli_option("--test-fraction", stage_spec.get("test_fraction")))
    command.extend(_append_cli_option("--random-state", stage_spec.get("random_state")))
    command.extend(_append_cli_option("--dev-output-file", _resolve_optional_path(manifest, stage_spec.get("dev_output_file"))))
    command.extend(_append_cli_option("--test-output-file", _resolve_optional_path(manifest, stage_spec.get("test_output_file"))))
    command.extend(_append_cli_option("--id-field", stage_spec.get("id_field")))
    command.extend(_append_cli_option("--test-ids-output-file", _resolve_optional_path(manifest, stage_spec.get("test_ids_output_file"))))
    command.extend(_append_cli_option("--split-report-output-file", _resolve_optional_path(manifest, stage_spec.get("split_report_output_file"))))
    command.extend(_append_bool_cli_option("--mlflow", stage_spec.get("mlflow")))
    command.extend(_append_cli_option("--mlflow-experiment", stage_spec.get("mlflow_experiment")))
    command.extend(_append_cli_option("--mlflow-run-name", stage_spec.get("mlflow_run_name")))
    command.extend(_append_cli_option("--tracking-uri", stage_spec.get("tracking_uri")))
    command.extend(_append_cli_option("--mlflow-dataset-name", stage_spec.get("mlflow_dataset_name")))
    command.extend(_append_cli_option("--dev-context", stage_spec.get("dev_context")))
    command.extend(_append_cli_option("--test-context", stage_spec.get("test_context")))

    _run_command(command, dry_run=dry_run)
    return {
        "stage_name": stage_name,
        "stage_type": SPLIT_STAGE,
        "dry_run": dry_run,
        "config_path": str(config_path),
        "command": command,
        "output_dir": str(stage_dir),
        "executed_at": _utc_timestamp(),
    }


def _run_tune_validity_stage(
    *,
    manifest: Mapping[str, Any],
    stage_name: str,
    stage_spec: Mapping[str, Any],
    stage_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    config_path = render_stage_condition_config(
        manifest_path=str(manifest["_manifest_path"]),
        stage_name=stage_name,
    )
    output_path = _resolve_optional_path(manifest, stage_spec.get("output_path")) or (stage_dir / "validity_reranker.yaml")
    manifest_output_path = _resolve_optional_path(manifest, stage_spec.get("manifest_path")) or (
        stage_dir / "validity_reranker.manifest.json"
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "tune_validity_reranker.py"),
        "-c",
        str(config_path),
        "--dataset-path",
        str(_resolve_required_path(
            manifest,
            stage_spec.get("dataset_path"),
            field_name=f"stages.{stage_name}.dataset_path",
        )),
        "--output-path",
        str(output_path),
        "--manifest-path",
        str(manifest_output_path),
    ]
    command.extend(_append_cli_option("--generation-workers", stage_spec.get("generation_workers")))
    _run_command(command, dry_run=dry_run)
    return {
        "stage_name": stage_name,
        "stage_type": TUNE_VALIDITY_STAGE,
        "dry_run": dry_run,
        "config_path": str(config_path),
        "command": command,
        "output_path": str(output_path),
        "manifest_path": str(manifest_output_path),
        "executed_at": _utc_timestamp(),
    }


def _run_benchmark_analysis_stage(
    *,
    manifest: Mapping[str, Any],
    stage_name: str,
    stage_spec: Mapping[str, Any],
    stage_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir = _resolve_optional_path(manifest, stage_spec.get("output_dir")) or stage_dir
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_analysis.py"),
        "--dataset-file",
        str(_resolve_required_path(
            manifest,
            stage_spec.get("dataset_file"),
            field_name=f"stages.{stage_name}.dataset_file",
        )),
        "--dev-dataset-file",
        str(_resolve_required_path(
            manifest,
            stage_spec.get("dev_dataset_file"),
            field_name=f"stages.{stage_name}.dev_dataset_file",
        )),
        "--test-dataset-file",
        str(_resolve_required_path(
            manifest,
            stage_spec.get("test_dataset_file"),
            field_name=f"stages.{stage_name}.test_dataset_file",
        )),
        "--annotations-file",
        str(_resolve_required_path(
            manifest,
            stage_spec.get("annotations_file"),
            field_name=f"stages.{stage_name}.annotations_file",
        )),
        "--output-dir",
        str(output_dir),
    ]
    if bool(stage_spec.get("allow_unverified", False)):
        command.append("--allow-unverified")
    _run_command(command, dry_run=dry_run)
    return {
        "stage_name": stage_name,
        "stage_type": BENCHMARK_ANALYSIS_STAGE,
        "dry_run": dry_run,
        "command": command,
        "output_dir": str(output_dir),
        "executed_at": _utc_timestamp(),
    }


def _build_evaluate_command(
    *,
    config_path: Path,
    dataset_path: Path,
    annotations_path: Path | None,
    prepared_path: Path,
    output_dir: Path,
    preset: Any,
    run_options: Mapping[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_rag.py"),
        "-c",
        str(config_path),
        "--dataset-path",
        str(dataset_path),
        "--prepared-path",
        str(prepared_path),
        "--output-dir",
        str(output_dir),
    ]
    if annotations_path is not None:
        command.extend(["--annotations-file", str(annotations_path)])
    if preset is not None:
        command.extend(["--preset", str(preset)])
    command.extend(_append_cli_option("--metrics", _metrics_value(run_options.get("metrics"))))
    command.extend(_append_cli_option("--generation-mode", run_options.get("generation_mode")))
    command.extend(_append_cli_option("--generation-workers", run_options.get("generation_workers")))
    command.extend(_append_cli_option("--evaluation-policy", run_options.get("evaluation_policy")))
    command.extend(_append_cli_option("--query-api-url", run_options.get("query_api_url")))
    command.extend(_append_cli_option("--query-api-timeout", run_options.get("query_api_timeout")))
    command.extend(_append_cli_option("--mlflow-experiment", run_options.get("mlflow_experiment")))
    command.extend(_append_cli_option("--mlflow-run-name", run_options.get("mlflow_run_name")))
    command.extend(_append_bool_cli_option("--mlflow", run_options.get("mlflow")))
    command.extend(_append_bool_cli_option("--trace-evaluator-llm", run_options.get("trace_evaluator_llm")))
    if bool(run_options.get("reuse_prepared", False)):
        command.append("--reuse-prepared")
    if bool(run_options.get("prepare_only", False)):
        command.append("--prepare-only")
    if bool(run_options.get("no_tune_concurrency", False)):
        command.append("--no-tune-concurrency")
    if bool(run_options.get("no_progress", False)):
        command.append("--no-progress")
    return command


def _build_ingest_command(
    *,
    kind: str,
    config_path: Path,
    ingest_spec: Mapping[str, Any],
) -> list[str]:
    if kind not in SUPPORTED_INGEST_KINDS:
        supported = ", ".join(sorted(SUPPORTED_INGEST_KINDS))
        raise ValueError(f"Unsupported ingest kind {kind!r}. Supported values: {supported}")

    if kind == "html":
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "ingest_html_documents.py"),
            "-c",
            str(config_path),
            "-p",
            str(ingest_spec["homepage"]),
        ]
        if bool(ingest_spec.get("ingest_internal_links", False)):
            command.append("--ingest-internal-links")
    elif kind == "jira":
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "ingest_jira_tickets.py"),
            "-c",
            str(config_path),
        ]
        command.extend(_append_cli_option("--start-date", ingest_spec.get("start_date")))
        command.extend(_append_cli_option("--end-date", ingest_spec.get("end_date")))
        command.extend(_append_cli_option("--limit", ingest_spec.get("limit")))
        command.extend(_append_cli_option("--exclude-keys-file", ingest_spec.get("exclude_keys_file")))
        if bool(ingest_spec.get("dump_processed", False)):
            command.append("--dump-processed")
        command.extend(_append_cli_option("--dump-path", ingest_spec.get("dump_path")))
    else:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "ingest_external_docs.py"),
            "-c",
            str(config_path),
        ]
        command.extend(_append_cli_option("--source-register-file", ingest_spec.get("source_register_file")))

    command.extend(_append_cli_option("--persist-dir", ingest_spec.get("persist_dir")))
    command.extend(_append_cli_option("--qdrant-collection-name", ingest_spec.get("qdrant_collection_name")))
    command.extend(_append_cli_option("--source", ingest_spec.get("source")))
    command.extend(_append_cli_option("--vector-batch-size", ingest_spec.get("vector_batch_size")))
    command.extend(_append_cli_option("--embedding-workers", ingest_spec.get("embedding_workers")))
    command.extend(_append_cli_option("--conversion-engine", ingest_spec.get("conversion_engine")))
    command.extend(_append_cli_option("--chunking-strategy", ingest_spec.get("chunking_strategy")))
    command.extend(_append_cli_option("--chunk-size-tokens", ingest_spec.get("chunk_size_tokens")))
    command.extend(_append_cli_option("--chunk-overlap-tokens", ingest_spec.get("chunk_overlap_tokens")))
    return command


def _collect_stage_run_rows(
    manifest: Mapping[str, Any],
    stage_name: str,
    stage_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stage_dir = _stage_output_dir(manifest, stage_name)
    rows: list[dict[str, Any]] = []
    for condition_spec in _stage_conditions(stage_spec, None):
        condition_name = str(condition_spec["name"])
        condition_slug = _slugify(condition_name)
        condition_dir = stage_dir / condition_slug
        if not condition_dir.exists():
            continue
        for run_dir in sorted(condition_dir.glob("run_*")):
            scores_path = run_dir / "scores.csv"
            summary_path = run_dir / "summary.json"
            manifest_path = run_dir / "run_manifest.json"
            if not scores_path.exists() or not summary_path.exists() or not manifest_path.exists():
                continue

            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metric_means = _read_metric_means(scores_path)
            repeat_match = re.search(r"(\d+)$", run_dir.name)
            repeat_index = int(repeat_match.group(1)) if repeat_match else 0

            row = {
                "stage_name": stage_name,
                "condition_name": condition_name,
                "condition_slug": condition_slug,
                "repeat_index": repeat_index,
                "run_dir": str(run_dir.resolve()),
                "config_file": run_manifest.get("config_file"),
                "preset_name": run_manifest.get("preset_name") or condition_spec.get("preset"),
                "condition_fingerprint": run_manifest.get("condition_fingerprint"),
                "rows": _safe_int(summary_payload.get("rows")),
                "duration_seconds": _safe_float(summary_payload.get("duration_seconds")),
                "failure_rate": _safe_float(summary_payload.get("failure_rate")),
                "selected_max_workers": _safe_int(summary_payload.get("selected_max_workers")),
            }
            row.update(metric_means)
            rows.append(row)
    return rows


def _aggregate_stage_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition_name"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for condition_name, condition_rows in sorted(by_condition.items()):
        ordered_rows = sorted(condition_rows, key=lambda item: int(item.get("repeat_index", 0)))
        aggregate: dict[str, Any] = {
            "condition_name": condition_name,
            "runs": len(ordered_rows),
            "preset_name": ordered_rows[0].get("preset_name"),
        }
        for numeric_key in ("duration_seconds", "failure_rate"):
            values = _numeric_values(row.get(numeric_key) for row in ordered_rows)
            aggregate[f"{numeric_key}_mean"] = _mean_or_none(values)
            aggregate[f"{numeric_key}_min"] = min(values) if values else None
            aggregate[f"{numeric_key}_max"] = max(values) if values else None
            aggregate[f"{numeric_key}_std"] = _std_or_none(values)

        for metric_name in metric_names:
            values = _numeric_values(row.get(metric_name) for row in ordered_rows)
            aggregate[f"{metric_name}_mean"] = _mean_or_none(values)
            aggregate[f"{metric_name}_min"] = min(values) if values else None
            aggregate[f"{metric_name}_max"] = max(values) if values else None
            aggregate[f"{metric_name}_std"] = _std_or_none(values)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _select_run_comparison_specs(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparison_repeat: str,
) -> list[tuple[str, Path]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition_name"]), []).append(row)

    specs: list[tuple[str, Path]] = []
    for condition_name, condition_rows in sorted(grouped.items()):
        ordered = sorted(condition_rows, key=lambda item: int(item.get("repeat_index", 0)))
        if comparison_repeat == "latest":
            selected = ordered[-1]
        else:
            repeat_index = int(comparison_repeat)
            selected = next(
                (row for row in ordered if int(row.get("repeat_index", 0)) == repeat_index),
                None,
            )
            if selected is None:
                raise ValueError(
                    f"Condition {condition_name!r} does not have repeat index {repeat_index}."
                )
        specs.append((condition_name, Path(str(selected["run_dir"])).expanduser().resolve()))
    return specs


def _metrics_value(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return value or None
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, bytearray)):
        cleaned = [str(item).strip() for item in raw_value if str(item).strip()]
        return ",".join(cleaned) if cleaned else None
    return str(raw_value)


def _append_cli_option(flag: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Path):
        return [flag, str(value)]
    return [flag, str(value)]


def _append_bool_cli_option(flag: str, value: Any) -> list[str]:
    if value is None:
        return []
    suffix = flag[2:]
    return [flag if bool(value) else f"--no-{suffix}"]


def _run_command(command: Sequence[str], *, dry_run: bool) -> None:
    if dry_run:
        return
    subprocess.run(list(command), check=True, cwd=str(REPO_ROOT))


def _resolve_stage_spec(manifest: Mapping[str, Any], stage_name: str) -> dict[str, Any]:
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("Experiment manifest does not define a valid 'stages' mapping.")
    stage_spec = stages.get(stage_name)
    if not isinstance(stage_spec, Mapping):
        available = ", ".join(sorted(str(name) for name in stages.keys()))
        raise KeyError(f"Unknown stage {stage_name!r}. Available stages: {available}")
    return dict(stage_spec)


def _resolve_condition_spec(
    stage_spec: Mapping[str, Any],
    condition_name: str | None,
) -> dict[str, Any] | None:
    stage_type = _stage_type(stage_spec)
    if stage_type != EVALUATION_GRID_STAGE:
        if condition_name is not None:
            raise ValueError("Condition selection is only supported for 'evaluation_grid' stages.")
        return None

    conditions = _stage_conditions(stage_spec, None)
    if condition_name is None:
        if len(conditions) == 1:
            return dict(conditions[0])
        raise ValueError("This stage defines multiple conditions; pass --condition to select one.")

    matches = [condition for condition in conditions if str(condition["name"]) == condition_name]
    if not matches:
        available = ", ".join(str(condition["name"]) for condition in conditions)
        raise KeyError(f"Unknown condition {condition_name!r}. Available conditions: {available}")
    return dict(matches[0])


def _stage_conditions(
    stage_spec: Mapping[str, Any],
    selected_conditions: Sequence[str] | None,
) -> list[dict[str, Any]]:
    conditions_raw = stage_spec.get("conditions")
    if not isinstance(conditions_raw, Sequence) or isinstance(conditions_raw, (str, bytes, bytearray)):
        raise ValueError("Evaluation-grid stages must define a 'conditions' list.")

    conditions: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw_condition in conditions_raw:
        if not isinstance(raw_condition, Mapping):
            raise TypeError("Each condition must be a mapping.")
        name = str(raw_condition.get("name", "")).strip()
        if not name:
            raise ValueError("Each condition must define a non-empty 'name'.")
        if name in seen_names:
            raise ValueError(f"Duplicate condition name {name!r} in stage manifest.")
        seen_names.add(name)
        conditions.append(dict(raw_condition))

    if not selected_conditions:
        return conditions

    selected_set = {str(name) for name in selected_conditions}
    filtered = [condition for condition in conditions if str(condition["name"]) in selected_set]
    missing = sorted(selected_set - {str(condition["name"]) for condition in filtered})
    if missing:
        raise KeyError(f"Unknown selected condition(s): {', '.join(missing)}")
    return filtered


def _stage_type(stage_spec: Mapping[str, Any]) -> str:
    stage_type = str(stage_spec.get("type", "")).strip()
    if stage_type not in SUPPORTED_STAGE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_STAGE_TYPES))
        raise ValueError(f"Unsupported stage type {stage_type!r}. Supported values: {supported}")
    return stage_type


def _merged_config_overrides(
    manifest: Mapping[str, Any],
    stage_spec: Mapping[str, Any],
    condition_spec: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    defaults = _as_mapping(_as_mapping(manifest.get("defaults")).get("config_overrides"))
    merged = _deep_merge(merged, defaults)
    merged = _deep_merge(merged, _as_mapping(stage_spec.get("config_overrides")))
    if condition_spec is not None:
        merged = _deep_merge(merged, _as_mapping(condition_spec.get("config_overrides")))
    return merged


def _merged_run_options(
    manifest: Mapping[str, Any],
    stage_spec: Mapping[str, Any],
    condition_spec: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    defaults = _as_mapping(_as_mapping(manifest.get("defaults")).get("run_options"))
    merged = _deep_merge(merged, defaults)
    merged = _deep_merge(merged, _as_mapping(stage_spec.get("run_options")))
    merged = _deep_merge(merged, _as_mapping(condition_spec.get("run_options")))
    return merged


def _default_generated_config_path(stage_name: str, condition_name: str | None) -> Path:
    suffix = _slugify(condition_name or "stage")
    return REPO_ROOT / f".polaris_experiment_{_slugify(stage_name)}__{suffix}.yaml"


def _stage_output_dir(manifest: Mapping[str, Any], stage_name: str) -> Path:
    artifacts_root = _resolve_required_path(
        manifest,
        manifest.get("artifacts_root", "artifacts/experiments"),
        field_name="artifacts_root",
    )
    return artifacts_root / _slugify(stage_name)


def _resolve_required_path(
    manifest: Mapping[str, Any],
    raw_value: Any,
    *,
    field_name: str,
) -> Path:
    if raw_value is None or str(raw_value).strip() == "":
        raise ValueError(f"Missing required path value for {field_name}.")
    return _resolve_optional_path(manifest, raw_value)  # type: ignore[return-value]


def _resolve_optional_path(manifest: Mapping[str, Any], raw_value: Any) -> Path | None:
    if raw_value is None or str(raw_value).strip() == "":
        return None
    candidate = Path(str(raw_value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    manifest_dir = Path(str(manifest["_manifest_dir"]))
    return (manifest_dir / candidate).resolve()


def _ingest_specs(condition_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = condition_spec.get("ingest")
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        specs = [dict(raw)]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        specs = [dict(item) for item in raw if isinstance(item, Mapping)]
    else:
        raise TypeError("Condition 'ingest' must be a mapping or list of mappings.")

    for spec in specs:
        kind = str(spec.get("kind", "")).strip()
        if not kind:
            raise ValueError("Each ingest spec must define a non-empty 'kind'.")
        if kind not in SUPPORTED_INGEST_KINDS:
            supported = ", ".join(sorted(SUPPORTED_INGEST_KINDS))
            raise ValueError(f"Unsupported ingest kind {kind!r}. Supported values: {supported}")
        if kind == "html" and not str(spec.get("homepage", "")).strip():
            raise ValueError("HTML ingest specs must define 'homepage'.")
    return specs


def _read_metric_means(path: Path) -> dict[str, float | None]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in reader:
            for key, value in row.items():
                if value is None:
                    continue
                number = _safe_float(value)
                if number is None:
                    continue
                sums[key] = sums.get(key, 0.0) + number
                counts[key] = counts.get(key, 0) + 1
    means: dict[str, float | None] = {}
    for key in sorted(sums.keys()):
        count = counts.get(key, 0)
        means[key] = (sums[key] / count) if count else None
    return means


def _metric_names_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    discovered: set[str] = set()
    structural = {
        "stage_name",
        "condition_name",
        "condition_slug",
        "repeat_index",
        "run_dir",
        "config_file",
        "preset_name",
        "condition_fingerprint",
        "rows",
        "duration_seconds",
        "failure_rate",
        "selected_max_workers",
    }
    for row in rows:
        for key in row.keys():
            if key not in structural:
                discovered.add(str(key))
    ordered = [name for name in DEFAULT_PRIMARY_METRICS if name in discovered]
    ordered.extend(sorted(name for name in discovered if name not in DEFAULT_PRIMARY_METRICS))
    return ordered


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std_or_none(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.pstdev(values)


def _numeric_values(values: Iterable[Any]) -> list[float]:
    numeric: list[float] = []
    for value in values:
        number = _safe_float(value)
        if number is not None:
            numeric.append(number)
    return numeric


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _sortable_metric(value: Any) -> float:
    number = _safe_float(value)
    return number if number is not None else float("-inf")


def _resolve_positive_int(value: Any, *, field_name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if resolved <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return resolved


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_normalize(payload), indent=2), encoding="utf-8")
    return path


def _write_csv_records(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_text = str(key)
            if key_text in seen:
                continue
            seen.add(key_text)
            fieldnames.append(key_text)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(inner) for key, inner in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_normalize(item) for item in value]
    if isinstance(value, float):
        return value if not (math.isnan(value) or math.isinf(value)) else None
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(_as_mapping(merged[key]), _as_mapping(value))
        else:
            merged[key] = value
    return merged


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip().lower()).strip("._-")
    return cleaned or "condition"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "EVALUATION_GRID_STAGE",
    "BENCHMARK_ANALYSIS_STAGE",
    "SPLIT_STAGE",
    "SUPPORTED_STAGE_TYPES",
    "TUNE_VALIDITY_STAGE",
    "load_experiment_manifest",
    "render_stage_condition_config",
    "run_experiment_stage",
    "summarize_experiment_stage",
]
