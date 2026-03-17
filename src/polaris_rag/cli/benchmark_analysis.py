"""CLI entrypoint for benchmark subgroup characterisation."""

from __future__ import annotations

import argparse
from pathlib import Path

from polaris_rag.evaluation import load_raw_examples
from polaris_rag.evaluation.benchmark_analysis import write_analysis_outputs
from polaris_rag.evaluation.benchmark_annotations import (
    build_split_lookup,
    join_annotations_into_rows,
    load_annotation_rows,
    validate_annotation_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterise benchmark annotation subgroups and emit experiment-1 artifacts"
    )
    parser.add_argument("--dataset-file", required=True, help="Path to the full raw benchmark dataset")
    parser.add_argument("--dev-dataset-file", required=True, help="Path to the dev split dataset")
    parser.add_argument("--test-dataset-file", required=True, help="Path to the test split dataset")
    parser.add_argument("--annotations-file", required=True, help="Path to the annotation CSV")
    parser.add_argument(
        "--output-dir",
        default="data/test/benchmark_analysis",
        help="Destination directory for experiment-1 outputs",
    )
    parser.add_argument(
        "--allow-unverified",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow seeded annotation rows instead of requiring review_status=verified",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_examples = load_raw_examples(args.dataset_file)
    dev_examples = load_raw_examples(args.dev_dataset_file)
    test_examples = load_raw_examples(args.test_dataset_file)
    split_lookup = build_split_lookup(dev_examples=dev_examples, test_examples=test_examples)

    annotation_rows = load_annotation_rows(args.annotations_file)
    validated = validate_annotation_rows(
        annotation_rows=annotation_rows,
        raw_examples=raw_examples,
        split_lookup=split_lookup,
        require_verified=not bool(args.allow_unverified),
    )
    annotated_rows = join_annotations_into_rows(raw_examples, validated)

    artifacts = write_analysis_outputs(rows=annotated_rows, output_dir=Path(args.output_dir))
    print(f"Benchmark analysis complete for {len(validated)} rows.")
    for label, path in artifacts.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
