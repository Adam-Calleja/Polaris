"""Evaluation CLI entrypoint for Polaris RAG."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from polaris_rag.app.container import build_container
from polaris_rag.config import GlobalConfig
from polaris_rag.evaluation.evaluation_dataset import (
    build_prepared_rows,
    build_prepared_rows_from_api,
    load_prepared_rows,
    load_raw_examples,
    persist_prepared_rows,
    to_evaluation_dataset,
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

    return parser.parse_args()


def _resolve_output_dir(eval_cfg: Mapping[str, Any], cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    cfg_value = eval_cfg.get("output_dir")
    if cfg_value:
        return Path(str(cfg_value)).expanduser().resolve()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (Path("data") / "eval_runs" / stamp).resolve()


def _resolve_prepared_rows(
    *,
    cfg: GlobalConfig,
    args: argparse.Namespace,
    eval_cfg: Mapping[str, Any],
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

    generation_workers = (
        int(args.generation_workers)
        if args.generation_workers is not None
        else _as_int(generation_cfg.get("workers"), 1)
    )
    generation_mode = str(args.generation_mode or generation_cfg.get("mode", "pipeline")).strip().lower()
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
    }

    if reuse_prepared and prepared_path and prepared_path.exists():
        rows = load_prepared_rows(prepared_path)
        manifest["prepared_source"] = "existing"
        return rows, manifest

    raw_examples = load_raw_examples(dataset_path)
    raise_exceptions = _as_bool(generation_cfg.get("raise_exceptions"), False)

    if generation_mode == "api":
        api_url = str(args.query_api_url or generation_cfg.get("api_url") or "").strip()
        if not api_url:
            raise ValueError(
                "evaluation.generation.mode is 'api' but no API URL was provided. "
                "Set evaluation.generation.api_url or pass --query-api-url."
            )
        timeout_seconds = (
            float(args.query_api_timeout)
            if args.query_api_timeout is not None
            else _as_float(generation_cfg.get("api_timeout"), 120.0)
        )

        rows = build_prepared_rows_from_api(
            raw_examples=raw_examples,
            api_url=api_url,
            query_field=query_field,
            reference_field=reference_field,
            id_field=id_field,
            generation_workers=max(1, generation_workers),
            raise_exceptions=raise_exceptions,
            timeout_seconds=timeout_seconds,
            headers=_as_mapping(generation_cfg.get("api_headers", {})),
        )
        manifest["query_api_url"] = api_url
        manifest["query_api_timeout"] = timeout_seconds
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
        )

    if prepared_path:
        persisted = persist_prepared_rows(rows, prepared_path)
        manifest["persisted_prepared_path"] = str(persisted)

    manifest["prepared_source"] = "generated"
    return rows, manifest


def main() -> None:
    args = parse_args()
    from polaris_rag.evaluation.evaluator import Evaluator, write_outputs

    cfg = GlobalConfig.load(args.config_file)
    eval_cfg = _as_mapping(_as_mapping(cfg.raw).get("evaluation", {}))

    prepared_rows, dataset_manifest = _resolve_prepared_rows(
        cfg=cfg,
        args=args,
        eval_cfg=eval_cfg,
    )

    dataset = to_evaluation_dataset(prepared_rows)

    requested_metrics = _parse_metrics(args.metrics)
    evaluator = Evaluator.from_global_config(
        cfg,
        requested_metrics=requested_metrics,
    )

    tune_concurrency = not args.no_tune_concurrency
    show_progress = not args.no_progress

    result = evaluator.evaluate(
        dataset=dataset,
        source_rows=prepared_rows,
        tune_concurrency=tune_concurrency,
        show_progress=show_progress,
    )

    output_dir = _resolve_output_dir(eval_cfg, args.output_dir)

    artifacts = write_outputs(
        result=result,
        output_dir=output_dir,
        extra_manifest={
            "config_file": str(Path(args.config_file).expanduser().resolve()),
            "dataset": dataset_manifest,
            "tune_concurrency": tune_concurrency,
        },
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
