"""Benchmark annotation utilities for evaluation dataset characterisation.

This module combines public functions and classes used by the surrounding Polaris
subsystem.

Classes
-------
AnnotationValidationError
    Raised when benchmark annotation data is invalid.

Functions
---------
load_annotation_rows
    Load annotation rows from CSV.
persist_annotation_rows
    Persist annotation rows to CSV.
load_legacy_audit_labels
    Load legacy audit labels keyed by sample id.
build_split_lookup
    Build sample-id -> split mapping from dev/test datasets.
scaffold_annotation_rows
    Create scaffold annotation rows from the benchmark and legacy labels.
validate_annotation_rows
    Validate and normalize benchmark annotation rows.
annotation_rows_by_id
    Build annotation rows keyed by id.
join_annotations_into_rows
    Join validated annotation payloads onto benchmark rows under metadata.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from polaris_rag.evaluation.csv_utils import dict_reader as csv_dict_reader


ANNOTATION_COLUMNS: tuple[str, ...] = (
    "id",
    "split",
    "summary",
    "source_needed",
    "docs_scope_needed",
    "validity_sensitive",
    "attachment_dependent",
    "query_type",
    "version_sensitive",
    "system_scope_required",
    "review_status",
    "notes",
)

ANNOTATION_METADATA_KEY = "benchmark_annotation"

SOURCE_NEEDED_VALUES = ("docs", "tickets", "both")
DOCS_SCOPE_VALUES = ("local_official", "external_official", "local_and_external", "none")
YES_NO_VALUES = ("yes", "no")
QUERY_TYPE_VALUES = ("local_operational", "software_version", "general_how_to")
REVIEW_STATUS_VALUES = ("seeded", "verified")

LABEL_VALUE_ORDERS: dict[str, tuple[str, ...]] = {
    "source_needed": SOURCE_NEEDED_VALUES,
    "docs_scope_needed": DOCS_SCOPE_VALUES,
    "validity_sensitive": YES_NO_VALUES,
    "attachment_dependent": YES_NO_VALUES,
    "query_type": QUERY_TYPE_VALUES,
    "version_sensitive": YES_NO_VALUES,
    "system_scope_required": YES_NO_VALUES,
    "review_status": REVIEW_STATUS_VALUES,
}

CORE_ANALYSIS_COLUMNS: tuple[str, ...] = (
    "source_needed",
    "docs_scope_needed",
    "validity_sensitive",
    "attachment_dependent",
)

ANALYSIS_LABEL_COLUMNS: tuple[str, ...] = (
    "source_needed",
    "docs_scope_needed",
    "validity_sensitive",
    "attachment_dependent",
    "query_type",
    "version_sensitive",
    "system_scope_required",
)


class AnnotationValidationError(ValueError):
    """Raised when benchmark annotation data is invalid.
    
    Notes
    -----
    This type participates in the public interface of the surrounding Polaris
    subsystem.
    """


def _normalize_text(value: Any) -> str:
    """Normalize text.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    str
        Resulting string value.
    """
    return str(value or "").strip()


def _row_id(row: Mapping[str, Any], *, id_field: str = "id", index: int | None = None) -> str:
    """Row ID.
    
    Parameters
    ----------
    row : Mapping[str, Any]
        Value for row.
    id_field : str, optional
        Value for ID Field.
    index : int or None, optional
        Value for index.
    
    Returns
    -------
    str
        Resulting string value.
    
    Raises
    ------
    AnnotationValidationError
        If `AnnotationValidationError` is raised while executing the operation.
    """
    sample_id = _normalize_text(row.get(id_field))
    if sample_id:
        return sample_id
    if index is None:
        raise AnnotationValidationError(f"Missing '{id_field}' in row.")
    raise AnnotationValidationError(f"Missing '{id_field}' for row index={index}.")


def _metadata_dict(value: Any) -> dict[str, Any]:
    """Metadata Dict.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    
    Returns
    -------
    dict[str, Any]
        Structured result of the operation.
    """
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def load_annotation_rows(path: str | Path) -> list[dict[str, str]]:
    """Load annotation rows from CSV.
    
    Parameters
    ----------
    path : str or Path
        Filesystem path used by the operation.
    
    Returns
    -------
    list[dict[str, str]]
        Loaded annotation Rows.
    
    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    AnnotationValidationError
        If `AnnotationValidationError` is raised while executing the operation.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {resolved}")

    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv_dict_reader(handle)
        if reader.fieldnames is None:
            return []

        missing = [name for name in ANNOTATION_COLUMNS if name not in reader.fieldnames]
        if missing:
            missing_text = ", ".join(missing)
            raise AnnotationValidationError(
                f"Annotation CSV is missing required columns: {missing_text}"
            )

        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = {column: _normalize_text(row.get(column, "")) for column in ANNOTATION_COLUMNS}
            rows.append(normalized)
        return rows


def persist_annotation_rows(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Persist annotation rows to CSV.
    
    Parameters
    ----------
    rows : Iterable[Mapping[str, Any]]
        Value for rows.
    path : str or Path
        Filesystem path used by the operation.
    
    Returns
    -------
    Path
        Result of the operation.
    """

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _normalize_text(row.get(column, "")) for column in ANNOTATION_COLUMNS})

    return resolved


def load_legacy_audit_labels(path: str | Path) -> dict[str, dict[str, str]]:
    """Load legacy audit labels keyed by sample id.
    
    Parameters
    ----------
    path : str or Path
        Filesystem path used by the operation.
    
    Returns
    -------
    dict[str, dict[str, str]]
        Loaded legacy Audit Labels.
    
    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    AnnotationValidationError
        If `AnnotationValidationError` is raised while executing the operation.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Legacy audit label file not found: {resolved}")

    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv_dict_reader(handle)
        if reader.fieldnames is None:
            return {}
        if "id" not in reader.fieldnames:
            raise AnnotationValidationError("Legacy audit label CSV must contain an 'id' column.")

        rows: dict[str, dict[str, str]] = {}
        for index, row in enumerate(reader, start=1):
            sample_id = _normalize_text(row.get("id"))
            if not sample_id:
                raise AnnotationValidationError(
                    f"Legacy audit label row index={index} is missing 'id'."
                )
            if sample_id in rows:
                raise AnnotationValidationError(f"Duplicate legacy audit label id '{sample_id}'.")
            rows[sample_id] = {str(key): _normalize_text(value) for key, value in row.items()}
        return rows


def build_split_lookup(
    *,
    dev_examples: Iterable[Mapping[str, Any]],
    test_examples: Iterable[Mapping[str, Any]],
    id_field: str = "id",
) -> dict[str, str]:
    """Build sample-id -> split mapping from dev/test datasets.
    
    Parameters
    ----------
    dev_examples : Iterable[Mapping[str, Any]]
        Value for dev Examples.
    test_examples : Iterable[Mapping[str, Any]]
        Value for test Examples.
    id_field : str, optional
        Value for ID Field.
    
    Returns
    -------
    dict[str, str]
        Constructed split Lookup.
    
    Raises
    ------
    AnnotationValidationError
        If `AnnotationValidationError` is raised while executing the operation.
    """

    lookup: dict[str, str] = {}
    for split_name, examples in (("dev", dev_examples), ("test", test_examples)):
        for index, row in enumerate(examples):
            sample_id = _row_id(row, id_field=id_field, index=index)
            existing = lookup.get(sample_id)
            if existing is not None:
                raise AnnotationValidationError(
                    f"Sample id '{sample_id}' appears in both split datasets ({existing}, {split_name})."
                )
            lookup[sample_id] = split_name
    return lookup


def scaffold_annotation_rows(
    *,
    raw_examples: Iterable[Mapping[str, Any]],
    split_lookup: Mapping[str, str],
    legacy_audit_labels: Mapping[str, Mapping[str, Any]] | None = None,
    id_field: str = "id",
    summary_field: str = "summary",
) -> list[dict[str, str]]:
    """Create scaffold annotation rows from the benchmark and legacy labels.
    
    Parameters
    ----------
    raw_examples : Iterable[Mapping[str, Any]]
        Raw examples value to normalize.
    split_lookup : Mapping[str, str]
        Value for split Lookup.
    legacy_audit_labels : Mapping[str, Mapping[str, Any]] or None, optional
        Value for legacy Audit Labels.
    id_field : str, optional
        Value for ID Field.
    summary_field : str, optional
        Value for summary Field.
    
    Returns
    -------
    list[dict[str, str]]
        Collected results from the operation.
    
    Raises
    ------
    AnnotationValidationError
        If `AnnotationValidationError` is raised while executing the operation.
    """

    legacy = {str(key): dict(value) for key, value in (legacy_audit_labels or {}).items()}
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for index, raw_row in enumerate(raw_examples):
        sample_id = _row_id(raw_row, id_field=id_field, index=index)
        if sample_id in seen_ids:
            raise AnnotationValidationError(f"Duplicate benchmark sample id '{sample_id}'.")
        seen_ids.add(sample_id)

        split = _normalize_text(split_lookup.get(sample_id))
        if not split:
            raise AnnotationValidationError(f"Sample id '{sample_id}' is missing from split lookup.")

        summary = _normalize_text(raw_row.get(summary_field))
        if not summary:
            raise AnnotationValidationError(
                f"Benchmark sample '{sample_id}' is missing summary field '{summary_field}'."
            )

        legacy_row = legacy.get(sample_id, {})
        source_needed = _normalize_text(legacy_row.get("evidence_need"))
        validity_sensitive = _normalize_text(legacy_row.get("temporal_sensitive"))

        docs_scope_needed = "none" if source_needed == "tickets" else "local_official"
        notes = "Seeded from legacy audit labels; manual review required."
        if not legacy_row:
            source_needed = "docs"
            validity_sensitive = "no"
            notes = "No legacy audit labels found; scaffold defaulted conservatively."

        rows.append(
            {
                "id": sample_id,
                "split": split,
                "summary": summary,
                "source_needed": source_needed,
                "docs_scope_needed": docs_scope_needed,
                "validity_sensitive": validity_sensitive,
                "attachment_dependent": "no",
                "query_type": "local_operational",
                "version_sensitive": "no",
                "system_scope_required": "no",
                "review_status": "seeded",
                "notes": notes,
            }
        )

    return rows


def _validate_enum(row: Mapping[str, str], column: str, allowed_values: Iterable[str], *, sample_id: str) -> str:
    """Validate enum.
    
    Parameters
    ----------
    row : Mapping[str, str]
        Value for row.
    column : str
        Value for column.
    allowed_values : Iterable[str]
        Value for allowed Values.
    sample_id : str
        Stable identifier for sample.
    
    Returns
    -------
    str
        Resulting string value.
    
    Raises
    ------
    AnnotationValidationError
        If `AnnotationValidationError` is raised while executing the operation.
    """
    value = _normalize_text(row.get(column))
    allowed = tuple(str(item) for item in allowed_values)
    if value not in allowed:
        allowed_text = ", ".join(allowed)
        raise AnnotationValidationError(
            f"Sample '{sample_id}' has invalid {column}={value!r}. Expected one of: {allowed_text}"
        )
    return value


def validate_annotation_rows(
    *,
    annotation_rows: Iterable[Mapping[str, Any]],
    raw_examples: Iterable[Mapping[str, Any]],
    split_lookup: Mapping[str, str] | None = None,
    require_verified: bool = False,
    regenerate_summary: bool = False,
    regenerate_splits: bool = False,
    allow_extra_annotations: bool = False,
    id_field: str = "id",
    summary_field: str = "summary",
) -> list[dict[str, str]]:
    """Validate and normalize benchmark annotation rows.
    
    Parameters
    ----------
    annotation_rows : Iterable[Mapping[str, Any]]
        Value for annotation Rows.
    raw_examples : Iterable[Mapping[str, Any]]
        Raw examples value to normalize.
    split_lookup : Mapping[str, str] or None, optional
        Value for split Lookup.
    require_verified : bool, optional
        Value for require Verified.
    regenerate_summary : bool, optional
        Value for regenerate Summary.
    regenerate_splits : bool, optional
        Value for regenerate Splits.
    allow_extra_annotations : bool, optional
        Whether to allow extra Annotations.
    id_field : str, optional
        Value for ID Field.
    summary_field : str, optional
        Value for summary Field.
    
    Returns
    -------
    list[dict[str, str]]
        Collected results from the operation.
    
    Raises
    ------
    AnnotationValidationError
        If `AnnotationValidationError` is raised while executing the operation.
    """

    raw_by_id: dict[str, dict[str, Any]] = {}
    raw_order: list[str] = []
    for index, raw_row in enumerate(raw_examples):
        sample_id = _row_id(raw_row, id_field=id_field, index=index)
        if sample_id in raw_by_id:
            raise AnnotationValidationError(f"Duplicate benchmark sample id '{sample_id}'.")
        raw_by_id[sample_id] = dict(raw_row)
        raw_order.append(sample_id)

    candidate_rows: dict[str, dict[str, str]] = {}
    for index, raw_annotation in enumerate(annotation_rows, start=1):
        row = {column: _normalize_text(raw_annotation.get(column, "")) for column in ANNOTATION_COLUMNS}
        sample_id = row["id"]
        if not sample_id:
            raise AnnotationValidationError(f"Annotation row index={index} is missing 'id'.")
        if sample_id in candidate_rows:
            raise AnnotationValidationError(f"Duplicate annotation row for sample id '{sample_id}'.")
        if sample_id not in raw_by_id:
            if allow_extra_annotations:
                continue
            raise AnnotationValidationError(
                f"Annotation row references unknown sample id '{sample_id}'."
            )
        candidate_rows[sample_id] = row

    missing = [sample_id for sample_id in raw_order if sample_id not in candidate_rows]
    if missing:
        raise AnnotationValidationError(
            f"Annotation rows missing benchmark ids: {', '.join(missing)}"
        )

    normalized_rows: list[dict[str, str]] = []
    for sample_id in raw_order:
        row = dict(candidate_rows[sample_id])
        raw_row = raw_by_id[sample_id]

        expected_summary = _normalize_text(raw_row.get(summary_field))
        if not expected_summary:
            raise AnnotationValidationError(
                f"Benchmark sample '{sample_id}' is missing summary field '{summary_field}'."
            )
        if row["summary"] != expected_summary:
            if regenerate_summary:
                row["summary"] = expected_summary
            else:
                raise AnnotationValidationError(
                    f"Sample '{sample_id}' has summary mismatch between annotation CSV and benchmark."
                )

        if split_lookup is not None:
            expected_split = _normalize_text(split_lookup.get(sample_id))
            if not expected_split:
                raise AnnotationValidationError(
                    f"Split lookup is missing sample id '{sample_id}'."
                )
            if row["split"] != expected_split:
                if regenerate_splits:
                    row["split"] = expected_split
                else:
                    raise AnnotationValidationError(
                        f"Sample '{sample_id}' has split={row['split']!r}; expected {expected_split!r}."
                    )

        source_needed = _validate_enum(row, "source_needed", SOURCE_NEEDED_VALUES, sample_id=sample_id)
        docs_scope_needed = _validate_enum(
            row,
            "docs_scope_needed",
            DOCS_SCOPE_VALUES,
            sample_id=sample_id,
        )
        _validate_enum(row, "validity_sensitive", YES_NO_VALUES, sample_id=sample_id)
        _validate_enum(row, "attachment_dependent", YES_NO_VALUES, sample_id=sample_id)
        _validate_enum(row, "query_type", QUERY_TYPE_VALUES, sample_id=sample_id)
        _validate_enum(row, "version_sensitive", YES_NO_VALUES, sample_id=sample_id)
        _validate_enum(row, "system_scope_required", YES_NO_VALUES, sample_id=sample_id)
        review_status = _validate_enum(row, "review_status", REVIEW_STATUS_VALUES, sample_id=sample_id)

        if source_needed == "tickets" and docs_scope_needed != "none":
            raise AnnotationValidationError(
                f"Sample '{sample_id}' has source_needed='tickets' but docs_scope_needed={docs_scope_needed!r}."
            )
        if source_needed in {"docs", "both"} and docs_scope_needed == "none":
            raise AnnotationValidationError(
                f"Sample '{sample_id}' requires docs but docs_scope_needed='none'."
            )
        if require_verified and review_status != "verified":
            raise AnnotationValidationError(
                f"Sample '{sample_id}' is not verified (review_status={review_status!r})."
            )

        normalized_rows.append(row)

    return normalized_rows


def annotation_rows_by_id(
    annotation_rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Build annotation rows keyed by id.
    
    Parameters
    ----------
    annotation_rows : Iterable[Mapping[str, Any]]
        Value for annotation Rows.
    
    Returns
    -------
    dict[str, dict[str, str]]
        Structured result of the operation.
    
    Raises
    ------
    AnnotationValidationError
        If `AnnotationValidationError` is raised while executing the operation.
    """

    lookup: dict[str, dict[str, str]] = {}
    for index, raw_row in enumerate(annotation_rows, start=1):
        row = {column: _normalize_text(raw_row.get(column, "")) for column in ANNOTATION_COLUMNS}
        sample_id = row["id"]
        if not sample_id:
            raise AnnotationValidationError(f"Annotation row index={index} is missing 'id'.")
        if sample_id in lookup:
            raise AnnotationValidationError(f"Duplicate annotation row for sample id '{sample_id}'.")
        lookup[sample_id] = row
    return lookup


def join_annotations_into_rows(
    raw_examples: Iterable[Mapping[str, Any]],
    annotation_rows: Iterable[Mapping[str, Any]],
    *,
    metadata_key: str = ANNOTATION_METADATA_KEY,
    id_field: str = "id",
) -> list[dict[str, Any]]:
    """Join validated annotation payloads onto benchmark rows under metadata.
    
    Parameters
    ----------
    raw_examples : Iterable[Mapping[str, Any]]
        Raw examples value to normalize.
    annotation_rows : Iterable[Mapping[str, Any]]
        Value for annotation Rows.
    metadata_key : str, optional
        Value for metadata Key.
    id_field : str, optional
        Value for ID Field.
    
    Returns
    -------
    list[dict[str, Any]]
        Collected results from the operation.
    
    Raises
    ------
    AnnotationValidationError
        If `AnnotationValidationError` is raised while executing the operation.
    """

    annotations = annotation_rows_by_id(annotation_rows)
    merged_rows: list[dict[str, Any]] = []

    for index, raw_row in enumerate(raw_examples):
        sample_id = _row_id(raw_row, id_field=id_field, index=index)
        annotation = annotations.get(sample_id)
        if annotation is None:
            raise AnnotationValidationError(
                f"Cannot join annotations: sample id '{sample_id}' is missing."
            )

        merged = dict(raw_row)
        metadata = _metadata_dict(raw_row.get("metadata"))
        metadata[metadata_key] = {
            column: annotation[column]
            for column in ANNOTATION_COLUMNS
            if column != "id"
        }
        merged["metadata"] = metadata
        merged_rows.append(merged)

    return merged_rows


__all__ = [
    "ANALYSIS_LABEL_COLUMNS",
    "ANNOTATION_COLUMNS",
    "ANNOTATION_METADATA_KEY",
    "AnnotationValidationError",
    "CORE_ANALYSIS_COLUMNS",
    "DOCS_SCOPE_VALUES",
    "LABEL_VALUE_ORDERS",
    "QUERY_TYPE_VALUES",
    "REVIEW_STATUS_VALUES",
    "SOURCE_NEEDED_VALUES",
    "YES_NO_VALUES",
    "annotation_rows_by_id",
    "build_split_lookup",
    "join_annotations_into_rows",
    "load_annotation_rows",
    "load_legacy_audit_labels",
    "persist_annotation_rows",
    "scaffold_annotation_rows",
    "validate_annotation_rows",
]
