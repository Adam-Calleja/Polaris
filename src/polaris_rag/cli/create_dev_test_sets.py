"""CLI entrypoint for splitting raw evaluation examples into dev and test sets."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from polaris_rag.config import GlobalConfig
from polaris_rag.evaluation import (
    load_raw_examples,
    load_sample_categories,
    load_sample_ids,
    persist_prepared_rows,
    stratified_split_raw_examples_by_categories,
    split_raw_examples_by_ids,
)
from polaris_rag.observability import EvaluationTrackingContext, load_mlflow_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dev and test datasets from a full raw dataset")
    parser.add_argument(
        "--config-file",
        "-c",
        default=None,
        help="Optional Polaris config YAML used to resolve MLflow runtime settings",
    )
    parser.add_argument(
        "--dataset-file",
        "-d",
        required=True,
        help="Path to the full raw evaluation dataset in JSON or JSONL format",
    )
    parser.add_argument(
        "--test-samples-file",
        "-t",
        default=None,
        help="Path to a file containing the chosen test sample IDs",
    )
    parser.add_argument(
        "--test-sample-id",
        action="append",
        default=[],
        help="Additional test sample ID (repeatable)",
    )
    parser.add_argument(
        "--categories-file",
        default=None,
        help="Optional JSON/YAML file mapping category names to sample IDs for stratified splitting",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=17,
        help="Number of stratified test samples to draw when using --categories-file (default: 17)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for stratified splitting when using --categories-file (default: 42)",
    )
    parser.add_argument(
        "--dev-output-file",
        default=None,
        help="Output path for the dev dataset (defaults next to dataset file)",
    )
    parser.add_argument(
        "--test-output-file",
        default=None,
        help="Output path for the test dataset (defaults next to dataset file)",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Dataset field containing the sample ID (default: id)",
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable MLflow dataset logging for this split operation",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=None,
        help="Override MLflow experiment name for this run",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name for this split operation",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override MLflow tracking URI",
    )
    parser.add_argument(
        "--mlflow-dataset-name",
        default=None,
        help="Base name to use for logged MLflow datasets (defaults to dataset file stem)",
    )
    parser.add_argument(
        "--dev-context",
        default="validation",
        help="MLflow dataset context to use for the dev split (default: validation)",
    )
    parser.add_argument(
        "--test-context",
        default="testing",
        help="MLflow dataset context to use for the test split (default: testing)",
    )
    return parser.parse_args()


def _default_output_path(dataset_path: Path, split_name: str) -> Path:
    suffix = dataset_path.suffix
    return dataset_path.with_name(f"{dataset_path.stem}.{split_name}{suffix}")


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "Creating MLflow dataset objects requires pandas. Install the evaluation/tracking extras first."
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


def _resolve_tracking(args: argparse.Namespace) -> EvaluationTrackingContext:
    cfg: Any = {}
    if args.config_file:
        cfg = GlobalConfig.load(args.config_file)

    runtime_cfg = load_mlflow_runtime_config(cfg)
    if args.tracking_uri:
        runtime_cfg = replace(runtime_cfg, tracking_uri=str(args.tracking_uri).strip())

    return EvaluationTrackingContext(
        runtime_cfg,
        enabled_override=getattr(args, "mlflow", None),
        experiment_override=getattr(args, "mlflow_experiment", None),
    )


def _log_split_datasets_to_mlflow(
    *,
    tracking: EvaluationTrackingContext,
    dataset_path: Path,
    dataset_name: str | None,
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    dev_output_path: Path,
    test_output_path: Path,
    dev_context: str,
    test_context: str,
    run_name: str | None,
) -> str | None:
    if not tracking.enabled:
        return None

    with tracking.open(
        run_name=run_name,
        extra_tags={
            "entrypoint": "polaris-create-dev-test-sets",
            "source_dataset": str(dataset_path),
        },
        strict=True,
    ):
        mlflow = tracking._mlflow
        if mlflow is None:
            return None

        dev_dataset_name = f"{dataset_name}-dev" if dataset_name else dev_output_path.stem
        test_dataset_name = f"{dataset_name}-test" if dataset_name else test_output_path.stem

        dev_dataset = _build_mlflow_dataset(
            mlflow,
            dev_rows,
            source=dev_output_path,
            dataset_name=dev_dataset_name,
        )
        test_dataset = _build_mlflow_dataset(
            mlflow,
            test_rows,
            source=test_output_path,
            dataset_name=test_dataset_name,
        )

        tracking.log_params(
            {
                "input.dataset_file": str(dataset_path),
                "input.mlflow_dataset_name": str(dataset_name or ""),
                "output.dev_file": str(dev_output_path),
                "output.test_file": str(test_output_path),
                "split.dev_rows": len(dev_rows),
                "split.test_rows": len(test_rows),
            }
        )
        tracking.log_artifact(dev_output_path, artifact_path="datasets")
        tracking.log_artifact(test_output_path, artifact_path="datasets")
        tracking.log_input(
            dev_dataset,
            context=dev_context,
            tags={
                "split": "dev",
                "parent_dataset": str(dataset_path),
                "output_file": str(dev_output_path),
                "rows": len(dev_rows),
            },
        )
        tracking.log_input(
            test_dataset,
            context=test_context,
            tags={
                "split": "test",
                "parent_dataset": str(dataset_path),
                "output_file": str(test_output_path),
                "rows": len(test_rows),
            },
        )
        return tracking.run_id


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_file).expanduser().resolve()
    raw_examples = load_raw_examples(dataset_path)

    using_categories = bool(args.categories_file)
    if using_categories and (args.test_samples_file or list(args.test_sample_id or [])):
        raise ValueError(
            "Use either --categories-file for stratified splitting or explicit test samples via "
            "--test-samples-file/--test-sample-id, not both."
        )

    split_stats: dict[str, Any] = {}
    if using_categories:
        categories = load_sample_categories(args.categories_file)
        dev_rows, test_rows, split_stats = stratified_split_raw_examples_by_categories(
            raw_examples,
            categories,
            test_size=int(args.test_size),
            random_state=int(args.random_state),
            id_field=str(args.id_field or "id"),
        )
    else:
        selected_ids: list[str] = []
        if args.test_samples_file:
            selected_ids.extend(load_sample_ids(args.test_samples_file))
        selected_ids.extend(str(value).strip() for value in list(args.test_sample_id or []) if str(value).strip())

        if not selected_ids:
            raise ValueError(
                "Provide either --categories-file or at least one explicit test sample via "
                "--test-samples-file/--test-sample-id."
            )

        dev_rows, test_rows = split_raw_examples_by_ids(
            raw_examples,
            selected_ids,
            id_field=str(args.id_field or "id"),
        )

    dev_output_path = (
        Path(args.dev_output_file).expanduser().resolve()
        if args.dev_output_file
        else _default_output_path(dataset_path, "dev")
    )
    test_output_path = (
        Path(args.test_output_file).expanduser().resolve()
        if args.test_output_file
        else _default_output_path(dataset_path, "test")
    )

    persist_prepared_rows(dev_rows, dev_output_path)
    persist_prepared_rows(test_rows, test_output_path)

    dataset_name = str(args.mlflow_dataset_name).strip() if args.mlflow_dataset_name else None
    tracking = _resolve_tracking(args)
    run_id = _log_split_datasets_to_mlflow(
        tracking=tracking,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dev_rows=dev_rows,
        test_rows=test_rows,
        dev_output_path=dev_output_path,
        test_output_path=test_output_path,
        dev_context=str(args.dev_context or "validation"),
        test_context=str(args.test_context or "testing"),
        run_name=str(args.mlflow_run_name).strip() if args.mlflow_run_name else None,
    )

    print("Dataset split complete.")
    print(f"Dataset file: {dataset_path}")
    print(f"Total rows: {len(raw_examples)}")
    print(f"Dev rows: {len(dev_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Dev output: {dev_output_path}")
    print(f"Test output: {test_output_path}")
    if split_stats:
        print(f"Test ids: {split_stats.get('test_ids', [])}")
        for category, count in dict(split_stats.get("category_test_counts", {})).items():
            print(f"{category}: {count} test")
    if run_id:
        print(f"MLflow run id: {run_id}")


if __name__ == "__main__":
    main()
