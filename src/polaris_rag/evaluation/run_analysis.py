"""Stage 5 analysis artifacts and run-comparison exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from polaris_rag.evaluation.benchmark_annotations import ANALYSIS_LABEL_COLUMNS, ANNOTATION_METADATA_KEY
from polaris_rag.evaluation.evaluation_dataset import load_prepared_rows


RESERVED_SCORE_COLUMNS = {"id", "metadata"}
MANUAL_RATING_COLUMNS: tuple[str, ...] = (
    "correctness",
    "actionability",
    "safety_policy",
    "source_appropriateness",
)


@dataclass(frozen=True)
class RunInput:
    """Resolved run input used by the analysis CLI."""

    condition_name: str
    run_dir: Path
    analysis_rows: list[dict[str, Any]]


def build_analysis_rows(
    *,
    source_rows: list[dict[str, Any]],
    scores_df: Any,
    condition_fields: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Join prepared rows, automatic scores, and condition metadata into a stable artifact."""

    score_rows = _score_records(scores_df)
    if len(score_rows) != len(source_rows):
        raise ValueError(
            "analysis rows require one score row per prepared row. "
            f"Got {len(score_rows)} score rows for {len(source_rows)} prepared rows."
        )

    condition = {
        "preset_name": condition_fields.get("preset_name") if isinstance(condition_fields, Mapping) else None,
        "preset_description": condition_fields.get("preset_description") if isinstance(condition_fields, Mapping) else None,
        "condition_fingerprint": condition_fields.get("condition_fingerprint") if isinstance(condition_fields, Mapping) else None,
        "condition_summary": _normalized_value(condition_fields.get("condition_summary")) if isinstance(condition_fields, Mapping) else None,
    }

    rows: list[dict[str, Any]] = []
    for prepared_row, score_row in zip(source_rows, score_rows, strict=True):
        metadata = _metadata(prepared_row)
        rows.append(
            {
                "id": str(prepared_row.get("id", "") or ""),
                "query": str(prepared_row.get("user_input", "") or ""),
                "reference": str(prepared_row.get("reference", "") or ""),
                "response": str(prepared_row.get("response", "") or ""),
                "retrieved_context_ids": list(prepared_row.get("retrieved_context_ids", []) or []),
                "retrieved_contexts": list(prepared_row.get("retrieved_contexts", []) or []),
                "metrics": {
                    key: _normalized_scalar(value)
                    for key, value in score_row.items()
                    if key not in RESERVED_SCORE_COLUMNS
                },
                "benchmark_annotation": _normalized_value(metadata.get(ANNOTATION_METADATA_KEY)),
                "query_constraints": _normalized_value(metadata.get("query_constraints")),
                "retrieval_sources": list(metadata.get("retrieval_sources", []) or []),
                "retrieval_source_types": list(metadata.get("retrieval_source_types", []) or []),
                "retrieval_features": _normalized_value(metadata.get("retrieval_features")) or [],
                "ranked_context_metadata": _normalized_value(metadata.get("ranked_context_metadata")) or [],
                "retrieval_trace": _normalized_value(metadata.get("retrieval_trace")) or [],
                "reranker_profile": _normalized_value(metadata.get("reranker_profile")),
                "reranker_fingerprint": _optional_text(metadata.get("reranker_fingerprint")),
                "response_status": _optional_text(metadata.get("response_status")),
                "failure_class": _optional_text(metadata.get("failure_class")),
                "failure_stage": _optional_text(metadata.get("failure_stage")),
                "source_error": _optional_text(metadata.get("source_error")),
                "condition": condition,
            }
        )
    return rows


def persist_analysis_rows(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Persist analysis rows as JSONL."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_normalized_value(dict(row)), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return resolved


def load_analysis_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load persisted analysis rows from JSONL."""

    resolved = Path(path).expanduser().resolve()
    text = resolved.read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected object row in {resolved}:{line_no}")
        rows.append(dict(payload))
    return rows


def load_run_input(condition_name: str, run_dir: str | Path) -> RunInput:
    """Load analysis rows for one saved run directory."""

    resolved_dir = Path(run_dir).expanduser().resolve()
    analysis_path = resolved_dir / "analysis_rows.jsonl"
    if analysis_path.exists():
        rows = load_analysis_rows(analysis_path)
        return RunInput(condition_name=condition_name, run_dir=resolved_dir, analysis_rows=rows)

    prepared_path = resolved_dir / "prepared_rows.json"
    scores_path = resolved_dir / "scores.csv"
    manifest_path = resolved_dir / "run_manifest.json"
    if not prepared_path.exists() or not scores_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Run directory {resolved_dir} does not contain analysis_rows.jsonl "
            "or the fallback prepared_rows.json + scores.csv + run_manifest.json bundle."
        )

    prepared_rows = load_prepared_rows(prepared_path)
    score_rows = _load_csv_records(scores_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    condition_fields = {
        "preset_name": manifest.get("preset_name"),
        "preset_description": manifest.get("preset_description"),
        "condition_fingerprint": manifest.get("condition_fingerprint"),
        "condition_summary": manifest.get("condition_summary"),
    }
    rows = build_analysis_rows(
        source_rows=prepared_rows,
        scores_df=score_rows,
        condition_fields=condition_fields,
    )
    return RunInput(condition_name=condition_name, run_dir=resolved_dir, analysis_rows=rows)


def write_run_comparison_outputs(
    *,
    runs: list[RunInput],
    output_dir: str | Path,
    manual_eval_seed: int = 42,
) -> dict[str, Path]:
    """Write cross-run comparison artifacts for dissertation analysis."""

    if not runs:
        raise ValueError("At least one run is required to write comparison outputs.")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    by_condition = {run.condition_name: _rows_with_condition_name(run) for run in runs}
    combined_rows = [row for rows in by_condition.values() for row in rows]

    condition_summary_rows = build_condition_summary_rows(by_condition)
    subgroup_rows = build_subgroup_metric_rows(by_condition)
    source_distribution_rows = build_source_distribution_rows(by_condition)
    review_rows = build_query_review_rows(by_condition)
    manual_sheet_rows, manual_key_rows, manual_manifest = build_manual_eval_outputs(
        by_condition=by_condition,
        seed=manual_eval_seed,
        runs=runs,
    )

    artifacts = {
        "condition_summary_csv": _write_csv_records(output / "condition_summary.csv", condition_summary_rows),
        "condition_summary_json": _write_json(output / "condition_summary.json", condition_summary_rows),
        "subgroup_metrics_csv": _write_csv_records(output / "subgroup_metrics.csv", subgroup_rows),
        "source_distribution_csv": _write_csv_records(output / "source_distribution.csv", source_distribution_rows),
        "query_review_sheet_csv": _write_csv_records(output / "query_review_sheet.csv", review_rows),
        "manual_eval_sheet_csv": _write_csv_records(output / "manual_eval_sheet.csv", manual_sheet_rows),
        "manual_eval_key_csv": _write_csv_records(output / "manual_eval_key.csv", manual_key_rows),
        "manual_eval_manifest_json": _write_json(output / "manual_eval_manifest.json", manual_manifest),
    }
    if combined_rows:
        artifacts["combined_analysis_rows_jsonl"] = persist_analysis_rows(
            combined_rows,
            output / "combined_analysis_rows.jsonl",
        )
    return artifacts


def build_condition_summary_rows(by_condition: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    metric_names = _metric_names(by_condition)
    rows: list[dict[str, Any]] = []
    for condition_name, records in by_condition.items():
        condition_meta = _condition_meta(records)
        row = {
            "condition_name": condition_name,
            "preset_name": condition_meta.get("preset_name"),
            "condition_fingerprint": condition_meta.get("condition_fingerprint"),
            "rows": len(records),
            "success_rows": sum(1 for record in records if not record.get("failure_class")),
            "failed_rows": sum(1 for record in records if record.get("failure_class")),
        }
        for metric_name in metric_names:
            row[f"mean_{metric_name}"] = _mean(
                _coerce_numeric(record.get("metrics", {}).get(metric_name))
                for record in records
            )
        rows.append(row)
    return rows


def build_subgroup_metric_rows(by_condition: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    metric_names = _metric_names(by_condition)
    rows: list[dict[str, Any]] = []
    for condition_name, records in by_condition.items():
        rows.extend(
            _subgroup_rows_for_records(
                condition_name=condition_name,
                records=records,
                metric_names=metric_names,
            )
        )
    return rows


def build_source_distribution_rows(by_condition: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_name, records in by_condition.items():
        counts: dict[tuple[int, str, str], int] = {}
        totals: dict[int, int] = {}
        for record in records:
            contexts = record.get("ranked_context_metadata")
            if not isinstance(contexts, list):
                continue
            for item in contexts:
                if not isinstance(item, Mapping):
                    continue
                rank = int(item.get("rank", 0) or 0)
                if rank <= 0:
                    continue
                source = str(item.get("source", "") or "unknown")
                source_authority = str(item.get("source_authority", "") or "unknown")
                counts[(rank, source, source_authority)] = counts.get((rank, source, source_authority), 0) + 1
                totals[rank] = totals.get(rank, 0) + 1

        for (rank, source, source_authority), count in sorted(counts.items()):
            total = totals.get(rank, 0)
            rows.append(
                {
                    "condition_name": condition_name,
                    "rank": rank,
                    "source": source,
                    "source_authority": source_authority,
                    "count": count,
                    "total": total,
                    "proportion": (count / total) if total else 0.0,
                }
            )
    return rows


def build_query_review_rows(by_condition: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    metric_names = _metric_names(by_condition)
    rows: list[dict[str, Any]] = []
    for condition_name, records in by_condition.items():
        for record in records:
            annotation = _annotation(record)
            row = {
                "condition_name": condition_name,
                "id": record.get("id"),
                "query": record.get("query"),
                "summary": annotation.get("summary"),
                "reference": record.get("reference"),
                "response": record.get("response"),
                "response_status": record.get("response_status"),
                "failure_class": record.get("failure_class"),
                "failure_stage": record.get("failure_stage"),
                "source_error": record.get("source_error"),
                "retrieval_sources": "|".join(record.get("retrieval_sources", []) or []),
                "retrieval_source_types": "|".join(record.get("retrieval_source_types", []) or []),
                "query_constraints_json": json.dumps(record.get("query_constraints"), ensure_ascii=False, sort_keys=True),
                "ranked_source_summary": _ranked_source_summary(record),
                "retrieval_feature_summary": _retrieval_feature_summary(record),
            }
            for label in ANALYSIS_LABEL_COLUMNS:
                row[label] = annotation.get(label)
            for metric_name in metric_names:
                row[metric_name] = _coerce_numeric(record.get("metrics", {}).get(metric_name))
            rows.append(row)
    return rows


def build_manual_eval_outputs(
    *,
    by_condition: Mapping[str, list[dict[str, Any]]],
    seed: int,
    runs: list[RunInput],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    condition_names = list(by_condition.keys())
    labels = [f"system_{chr(ord('A') + index)}" for index in range(len(condition_names))]
    records_by_condition_and_id = {
        condition_name: {str(record.get("id", "") or ""): record for record in records}
        for condition_name, records in by_condition.items()
    }
    query_ids = _ordered_query_ids(by_condition)

    sheet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        reference_record = records_by_condition_and_id[condition_names[0]].get(query_id)
        if reference_record is None:
            continue
        permutation = list(condition_names)
        _per_query_rng(seed=seed, query_id=query_id).shuffle(permutation)
        row = {
            "id": query_id,
            "query": reference_record.get("query"),
            "summary": _annotation(reference_record).get("summary"),
        }
        for label, condition_name in zip(labels, permutation, strict=True):
            record = records_by_condition_and_id[condition_name].get(query_id)
            if record is None:
                raise ValueError(
                    f"Manual-evaluation export requires every condition to contain query id {query_id!r}."
                )
            row[f"{label}_answer"] = record.get("response")
            row[f"{label}_source_summary"] = _ranked_source_summary(record)
            for rating_name in MANUAL_RATING_COLUMNS:
                row[f"{label}_{rating_name}"] = ""
            key_rows.append(
                {
                    "id": query_id,
                    "label": label,
                    "condition_name": condition_name,
                    "preset_name": _condition_meta([record]).get("preset_name"),
                    "condition_fingerprint": _condition_meta([record]).get("condition_fingerprint"),
                    "run_dir": str(next(run.run_dir for run in runs if run.condition_name == condition_name)),
                }
            )
        sheet_rows.append(row)

    manifest = {
        "seed": int(seed),
        "labels": labels,
        "conditions": condition_names,
        "query_count": len(sheet_rows),
        "runs": [
            {
                "condition_name": run.condition_name,
                "run_dir": str(run.run_dir),
            }
            for run in runs
        ],
    }
    return sheet_rows, key_rows, manifest


def _rows_with_condition_name(run: RunInput) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in run.analysis_rows:
        updated = dict(record)
        updated["condition_name"] = run.condition_name
        rows.append(updated)
    return rows


def _ordered_query_ids(by_condition: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    first_condition = next(iter(by_condition.values()))
    return [str(record.get("id", "") or "") for record in first_condition]


def _per_query_rng(*, seed: int, query_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _subgroup_rows_for_records(
    *,
    condition_name: str,
    records: list[dict[str, Any]],
    metric_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        _subgroup_metric_row(
            condition_name=condition_name,
            label_name="__all__",
            label_value="all",
            records=records,
            metric_names=metric_names,
        )
    )
    for label_name in ANALYSIS_LABEL_COLUMNS:
        values = sorted(
            {
                str(_annotation(record).get(label_name, "") or "")
                for record in records
                if str(_annotation(record).get(label_name, "") or "")
            }
        )
        for label_value in values:
            subset = [record for record in records if str(_annotation(record).get(label_name, "") or "") == label_value]
            rows.append(
                _subgroup_metric_row(
                    condition_name=condition_name,
                    label_name=label_name,
                    label_value=label_value,
                    records=subset,
                    metric_names=metric_names,
                )
            )
    return rows


def _subgroup_metric_row(
    *,
    condition_name: str,
    label_name: str,
    label_value: str,
    records: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, Any]:
    row = {
        "condition_name": condition_name,
        "label_name": label_name,
        "label_value": label_value,
        "row_count": len(records),
    }
    for metric_name in metric_names:
        row[f"mean_{metric_name}"] = _mean(
            _coerce_numeric(record.get("metrics", {}).get(metric_name))
            for record in records
        )
    return row


def _condition_meta(records: list[dict[str, Any]]) -> Mapping[str, Any]:
    if not records:
        return {}
    value = records[0].get("condition")
    return value if isinstance(value, Mapping) else {}


def _annotation(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("benchmark_annotation")
    return value if isinstance(value, Mapping) else {}


def _ranked_source_summary(record: Mapping[str, Any]) -> str:
    contexts = record.get("ranked_context_metadata")
    if not isinstance(contexts, list):
        return ""
    parts: list[str] = []
    for item in contexts[:3]:
        if not isinstance(item, Mapping):
            continue
        title = str(item.get("doc_title") or item.get("title") or item.get("doc_id") or "").strip()
        source = str(item.get("source", "") or "unknown").strip()
        source_authority = str(item.get("source_authority", "") or "unknown").strip()
        validity_status = str(item.get("validity_status", "") or "unknown").strip()
        parts.append(f"r{item.get('rank', '?')}:{source}/{source_authority}/{validity_status}:{title}")
    return " | ".join(parts)


def _retrieval_feature_summary(record: Mapping[str, Any]) -> str:
    features = record.get("retrieval_features")
    if not isinstance(features, list):
        return ""
    parts: list[str] = []
    for item in features[:3]:
        if not isinstance(item, Mapping):
            continue
        parts.append(
            "r{rank}:{source_authority}:final={final_score:.3f}:auth={authority_feature:.3f}:scope={scope_feature:.3f}:ver={version_feature:.3f}:status={status_feature:.3f}:fresh={freshness_feature:.3f}".format(
                rank=int(item.get("rank", 0) or 0),
                source_authority=str(item.get("source_authority", "") or "unknown"),
                final_score=float(item.get("final_score", 0.0) or 0.0),
                authority_feature=float(item.get("authority_feature", 0.0) or 0.0),
                scope_feature=float(item.get("scope_feature", 0.0) or 0.0),
                version_feature=float(item.get("version_feature", 0.0) or 0.0),
                status_feature=float(item.get("status_feature", 0.0) or 0.0),
                freshness_feature=float(item.get("freshness_feature", 0.0) or 0.0),
            )
        )
    return " | ".join(parts)


def _metric_names(by_condition: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    names: set[str] = set()
    for records in by_condition.values():
        for record in records:
            metrics = record.get("metrics")
            if isinstance(metrics, Mapping):
                names.update(str(name) for name in metrics.keys())
    return sorted(names)


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def _score_records(scores_df: Any) -> list[dict[str, Any]]:
    if isinstance(scores_df, list):
        return [dict(row) for row in scores_df if isinstance(row, Mapping)]
    to_dict = getattr(scores_df, "to_dict", None)
    if callable(to_dict):
        try:
            records = to_dict(orient="records")
            if isinstance(records, list):
                return [dict(record) for record in records if isinstance(record, Mapping)]
        except TypeError:
            pass
    columns = list(getattr(scores_df, "columns", []))
    if not columns:
        return []
    length = len(scores_df)
    records: list[dict[str, Any]] = []
    for index in range(length):
        record: dict[str, Any] = {}
        for column in columns:
            series = scores_df[column]
            values = series.tolist() if hasattr(series, "tolist") else list(series)
            record[str(column)] = values[index]
        records.append(record)
    return records


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _normalized_scalar(value: Any) -> Any:
    numeric = _coerce_numeric(value)
    if numeric is not None:
        return numeric
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    return str(value)


def _coerce_numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalized_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_normalized_value(item) for item in value]
    return str(value)


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_csv_records(path: Path, rows: list[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in rows[0].keys()})
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalized_value(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


__all__ = [
    "RunInput",
    "build_analysis_rows",
    "build_condition_summary_rows",
    "build_manual_eval_outputs",
    "build_query_review_rows",
    "build_source_distribution_rows",
    "build_subgroup_metric_rows",
    "load_analysis_rows",
    "load_run_input",
    "persist_analysis_rows",
    "write_run_comparison_outputs",
]
