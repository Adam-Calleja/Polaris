import csv
import json
import sys

import pytest

from polaris_rag.cli import benchmark_annotations
from polaris_rag.evaluation.benchmark_annotations import (
    ANNOTATION_METADATA_KEY,
    AnnotationValidationError,
    join_annotations_into_rows,
    load_annotation_rows,
    scaffold_annotation_rows,
    validate_annotation_rows,
)


def _raw_examples():
    return [
        {
            "id": "ex-1",
            "summary": "Storage path question",
            "query": "Q1",
            "expected_answer": "A1",
            "metadata": {"k": 1},
        },
        {
            "id": "ex-2",
            "summary": "Compile package on system",
            "query": "Q2",
            "expected_answer": "A2",
            "metadata": {"k": 2},
        },
    ]


def _annotation_rows():
    return [
        {
            "id": "ex-1",
            "split": "dev",
            "summary": "Storage path question",
            "source_needed": "docs",
            "docs_scope_needed": "local_official",
            "validity_sensitive": "yes",
            "attachment_dependent": "no",
            "query_type": "local_operational",
            "version_sensitive": "no",
            "system_scope_required": "yes",
            "review_status": "verified",
            "notes": "",
        },
        {
            "id": "ex-2",
            "split": "test",
            "summary": "Compile package on system",
            "source_needed": "both",
            "docs_scope_needed": "local_and_external",
            "validity_sensitive": "yes",
            "attachment_dependent": "no",
            "query_type": "software_version",
            "version_sensitive": "yes",
            "system_scope_required": "yes",
            "review_status": "verified",
            "notes": "Needs both local stack and package semantics.",
        },
    ]


def test_scaffold_annotation_rows_seeds_legacy_values() -> None:
    split_lookup = {"ex-1": "dev", "ex-2": "test"}
    legacy = {
        "ex-1": {"evidence_need": "tickets", "temporal_sensitive": "no"},
        "ex-2": {"evidence_need": "docs", "temporal_sensitive": "yes"},
    }

    rows = scaffold_annotation_rows(
        raw_examples=_raw_examples(),
        split_lookup=split_lookup,
        legacy_audit_labels=legacy,
    )

    assert rows[0]["source_needed"] == "tickets"
    assert rows[0]["docs_scope_needed"] == "none"
    assert rows[0]["validity_sensitive"] == "no"
    assert rows[0]["review_status"] == "seeded"
    assert rows[1]["source_needed"] == "docs"
    assert rows[1]["docs_scope_needed"] == "local_official"


def test_validate_annotation_rows_rejects_ticket_row_with_docs_scope() -> None:
    with pytest.raises(AnnotationValidationError, match="source_needed='tickets'"):
        validate_annotation_rows(
            annotation_rows=[
                {
                    **_annotation_rows()[0],
                    "source_needed": "tickets",
                    "docs_scope_needed": "local_official",
                },
                _annotation_rows()[1],
            ],
            raw_examples=_raw_examples(),
            split_lookup={"ex-1": "dev", "ex-2": "test"},
        )


def test_validate_annotation_rows_requires_verified_when_requested() -> None:
    rows = _annotation_rows()
    rows[1]["review_status"] = "seeded"

    with pytest.raises(AnnotationValidationError, match="not verified"):
        validate_annotation_rows(
            annotation_rows=rows,
            raw_examples=_raw_examples(),
            split_lookup={"ex-1": "dev", "ex-2": "test"},
            require_verified=True,
        )


def test_validate_annotation_rows_can_regenerate_split_values() -> None:
    rows = _annotation_rows()
    rows[0]["split"] = ""

    validated = validate_annotation_rows(
        annotation_rows=rows,
        raw_examples=_raw_examples(),
        split_lookup={"ex-1": "dev", "ex-2": "test"},
        regenerate_splits=True,
    )

    assert validated[0]["split"] == "dev"


def test_join_annotations_into_rows_preserves_existing_metadata() -> None:
    merged = join_annotations_into_rows(_raw_examples(), _annotation_rows())

    assert merged[0]["metadata"]["k"] == 1
    assert merged[0]["metadata"][ANNOTATION_METADATA_KEY]["source_needed"] == "docs"
    assert merged[1]["metadata"][ANNOTATION_METADATA_KEY]["query_type"] == "software_version"


def test_benchmark_annotations_cli_scaffold_and_validate(tmp_path, monkeypatch, capsys) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(_raw_examples()), encoding="utf-8")

    dev_path = tmp_path / "dataset.dev.jsonl"
    dev_path.write_text(json.dumps([_raw_examples()[0]]), encoding="utf-8")
    test_path = tmp_path / "dataset.test.jsonl"
    test_path.write_text(json.dumps([_raw_examples()[1]]), encoding="utf-8")

    legacy_path = tmp_path / "legacy.csv"
    with legacy_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "split", "summary", "temporal_sensitive", "evidence_need"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "ex-1",
                "split": "dev",
                "summary": "Storage path question",
                "temporal_sensitive": "no",
                "evidence_need": "tickets",
            }
        )
        writer.writerow(
            {
                "id": "ex-2",
                "split": "test",
                "summary": "Compile package on system",
                "temporal_sensitive": "yes",
                "evidence_need": "docs",
            }
        )

    scaffold_path = tmp_path / "annotations.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_annotations.py",
            "scaffold",
            "--dataset-file",
            str(dataset_path),
            "--dev-dataset-file",
            str(dev_path),
            "--test-dataset-file",
            str(test_path),
            "--legacy-audit-file",
            str(legacy_path),
            "--output-file",
            str(scaffold_path),
        ],
    )
    benchmark_annotations.main()

    scaffolded = load_annotation_rows(scaffold_path)
    assert scaffolded[0]["review_status"] == "seeded"

    scaffolded[0]["docs_scope_needed"] = "none"
    scaffolded[0]["query_type"] = "general_how_to"
    scaffolded[0]["review_status"] = "verified"
    scaffolded[1]["docs_scope_needed"] = "external_official"
    scaffolded[1]["query_type"] = "software_version"
    scaffolded[1]["version_sensitive"] = "yes"
    scaffolded[1]["system_scope_required"] = "yes"
    scaffolded[1]["review_status"] = "verified"
    with scaffold_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scaffolded[0].keys()))
        writer.writeheader()
        writer.writerows(scaffolded)

    normalized_path = tmp_path / "normalized.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_annotations.py",
            "validate",
            "--dataset-file",
            str(dataset_path),
            "--dev-dataset-file",
            str(dev_path),
            "--test-dataset-file",
            str(test_path),
            "--annotations-file",
            str(scaffold_path),
            "--require-verified",
            "--regenerate-splits",
            "--output-file",
            str(normalized_path),
        ],
    )
    benchmark_annotations.main()

    captured = capsys.readouterr()
    assert "Validated 2 annotation rows" in captured.out
    assert normalized_path.exists()
