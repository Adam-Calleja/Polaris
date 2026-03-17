"""CLI entrypoint for benchmark annotation scaffolding and validation."""

from __future__ import annotations

import argparse

from polaris_rag.evaluation import load_raw_examples
from polaris_rag.evaluation.benchmark_annotations import (
    ANNOTATION_COLUMNS,
    build_split_lookup,
    load_annotation_rows,
    load_legacy_audit_labels,
    persist_annotation_rows,
    scaffold_annotation_rows,
    validate_annotation_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold and validate benchmark annotation CSV files"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scaffold = subparsers.add_parser("scaffold", help="Generate a seeded annotation CSV")
    scaffold.add_argument("--dataset-file", required=True, help="Path to the full raw benchmark dataset")
    scaffold.add_argument("--dev-dataset-file", required=True, help="Path to the dev split dataset")
    scaffold.add_argument("--test-dataset-file", required=True, help="Path to the test split dataset")
    scaffold.add_argument(
        "--legacy-audit-file",
        default=None,
        help="Optional legacy audit CSV used to seed source_needed and validity_sensitive",
    )
    scaffold.add_argument("--output-file", required=True, help="Destination annotation CSV path")

    validate = subparsers.add_parser("validate", help="Validate an annotation CSV")
    validate.add_argument("--dataset-file", required=True, help="Path to the full raw benchmark dataset")
    validate.add_argument("--dev-dataset-file", required=True, help="Path to the dev split dataset")
    validate.add_argument("--test-dataset-file", required=True, help="Path to the test split dataset")
    validate.add_argument("--annotations-file", required=True, help="Path to the annotation CSV")
    validate.add_argument(
        "--output-file",
        default=None,
        help="Optional output path for normalized rows after validation",
    )
    validate.add_argument(
        "--require-verified",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require every annotation row to have review_status=verified",
    )
    validate.add_argument(
        "--regenerate-summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate summary cells from the benchmark when mismatches are detected",
    )
    validate.add_argument(
        "--regenerate-splits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate split cells from the provided dev/test datasets when mismatches are detected",
    )

    return parser.parse_args()


def _load_split_lookup(args: argparse.Namespace) -> tuple[list[dict], dict[str, str]]:
    raw_examples = load_raw_examples(args.dataset_file)
    dev_examples = load_raw_examples(args.dev_dataset_file)
    test_examples = load_raw_examples(args.test_dataset_file)
    split_lookup = build_split_lookup(dev_examples=dev_examples, test_examples=test_examples)
    return raw_examples, split_lookup


def _print_columns() -> None:
    print("Columns:")
    for name in ANNOTATION_COLUMNS:
        print(f"- {name}")


def main() -> None:
    args = parse_args()
    raw_examples, split_lookup = _load_split_lookup(args)

    if args.command == "scaffold":
        legacy_rows = (
            load_legacy_audit_labels(args.legacy_audit_file)
            if args.legacy_audit_file
            else None
        )
        rows = scaffold_annotation_rows(
            raw_examples=raw_examples,
            split_lookup=split_lookup,
            legacy_audit_labels=legacy_rows,
        )
        output_path = persist_annotation_rows(rows, args.output_file)
        print(f"Scaffolded {len(rows)} annotation rows to {output_path}")
        _print_columns()
        return

    rows = load_annotation_rows(args.annotations_file)
    validated = validate_annotation_rows(
        annotation_rows=rows,
        raw_examples=raw_examples,
        split_lookup=split_lookup,
        require_verified=bool(args.require_verified),
        regenerate_summary=bool(args.regenerate_summary),
        regenerate_splits=bool(args.regenerate_splits),
    )

    verified_count = sum(1 for row in validated if row["review_status"] == "verified")
    print(
        f"Validated {len(validated)} annotation rows "
        f"(verified={verified_count}, seeded={len(validated) - verified_count})."
    )

    if args.output_file:
        output_path = persist_annotation_rows(validated, args.output_file)
        print(f"Wrote normalized annotations to {output_path}")


if __name__ == "__main__":
    main()
