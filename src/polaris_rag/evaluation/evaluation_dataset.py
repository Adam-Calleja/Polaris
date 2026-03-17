"""polaris_rag.evaluation.evaluation_dataset

Utilities for preparing Polaris datasets for RAGAS evaluation.

The expected raw format contains at least:
- query
- expected_answer

Preparation executes the Polaris RAG pipeline to generate:
- response
- retrieved_contexts
- retrieved_context_ids
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import random
import socket
import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from polaris_rag.common.request_budget import (
    EVAL_POLICY_INTERACTIVE,
    EmptyResponseError,
    FAILURE_CLASS_API_INTERNAL_ERROR,
    FAILURE_CLASS_INVALID_INPUT,
    FAILURE_CLASS_TRANSPORT_ERROR,
    FAILURE_STAGE_API,
    FAILURE_STAGE_DATASET,
    FAILURE_STAGE_GENERATION,
    build_failure_detail,
    is_timeout_exception,
    normalize_evaluation_policy,
)

if TYPE_CHECKING:
    from ragas import EvaluationDataset


@dataclass(frozen=True)
class PrepProgressEvent:
    """Progress snapshot emitted while preparing rows."""

    completed: int
    total: int
    successes: int
    failures: int
    elapsed_seconds: float
    mode: str
    last_error: str | None = None


PrepProgressCallback = Callable[[PrepProgressEvent], None]
ApiRequester = Callable[[str, str, float, Mapping[str, str] | None], dict[str, Any]]


class PrepTraceRecorder(Protocol):
    def set_outputs(self, outputs: Any) -> None: ...

    def set_attributes(self, attributes: Mapping[str, Any]) -> None: ...


PrepAttemptTraceFactory = Callable[
    [str, Mapping[str, Any], Mapping[str, Any] | None],
    contextlib.AbstractContextManager[PrepTraceRecorder],
]


class _NoopTraceRecorder:
    def set_outputs(self, outputs: Any) -> None:
        return None

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        return None


@dataclass(frozen=True)
class PrepRetryPolicy:
    """Retry policy for per-row generation during dataset preparation."""

    max_attempts: int = 1
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 8.0
    jitter_seconds: float = 0.25
    retry_on_empty_response: bool = True

    @classmethod
    def from_value(cls, value: "PrepRetryPolicy | Mapping[str, Any] | None") -> "PrepRetryPolicy":
        if isinstance(value, PrepRetryPolicy):
            return cls(
                max_attempts=value.max_attempts,
                initial_backoff_seconds=value.initial_backoff_seconds,
                max_backoff_seconds=value.max_backoff_seconds,
                jitter_seconds=value.jitter_seconds,
                retry_on_empty_response=value.retry_on_empty_response,
            ).normalized()

        if isinstance(value, Mapping):
            return cls(
                max_attempts=_to_int(value.get("max_attempts"), 1),
                initial_backoff_seconds=_to_float(value.get("initial_backoff_seconds"), 1.0),
                max_backoff_seconds=_to_float(value.get("max_backoff_seconds"), 8.0),
                jitter_seconds=_to_float(value.get("jitter_seconds"), 0.25),
                retry_on_empty_response=_to_bool(value.get("retry_on_empty_response"), True),
            ).normalized()

        return cls().normalized()

    def normalized(self) -> "PrepRetryPolicy":
        initial_backoff = max(0.0, float(self.initial_backoff_seconds))
        max_backoff = max(0.0, float(self.max_backoff_seconds))
        if max_backoff < initial_backoff:
            max_backoff = initial_backoff

        return PrepRetryPolicy(
            max_attempts=max(1, int(self.max_attempts)),
            initial_backoff_seconds=initial_backoff,
            max_backoff_seconds=max_backoff,
            jitter_seconds=max(0.0, float(self.jitter_seconds)),
            retry_on_empty_response=bool(self.retry_on_empty_response),
        )


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
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


class QueryAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        detail_map = dict(detail or {})
        self.status_code = status_code
        self.detail = detail_map
        self.failure_class = str(
            detail_map.get(
                "failure_class",
                FAILURE_CLASS_API_INTERNAL_ERROR if status_code and status_code >= 500 else FAILURE_CLASS_TRANSPORT_ERROR,
            )
        )
        self.failure_stage = str(detail_map.get("failure_stage", FAILURE_STAGE_API))
        self.response_status = str(
            detail_map.get(
                "response_status",
                "timeout" if status_code == 504 else "error",
            )
        )


def _as_metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _build_row_metadata(
    base_metadata: Any,
    *,
    source_error: str | None = None,
    failure_class: str | None = None,
    failure_stage: str | None = None,
    http_status: int | None = None,
    elapsed_ms: int | None = None,
    policy: str | None = None,
    budget_ms: int | None = None,
    response_status: str | None = None,
    original_metadata: Any | None = None,
) -> dict[str, Any]:
    metadata = _as_metadata_dict(base_metadata)
    if source_error is not None:
        metadata["source_error"] = source_error
    if failure_class is not None:
        metadata["failure_class"] = failure_class
    if failure_stage is not None:
        metadata["failure_stage"] = failure_stage
    if http_status is not None:
        metadata["http_status"] = int(http_status)
    if elapsed_ms is not None:
        metadata["elapsed_ms"] = max(0, int(elapsed_ms))
    if policy is not None:
        metadata["policy"] = normalize_evaluation_policy(policy, default=EVAL_POLICY_INTERACTIVE)
    if budget_ms is not None:
        metadata["budget_ms"] = max(0, int(budget_ms))
    if response_status is not None:
        metadata["response_status"] = str(response_status)
    if original_metadata is not None:
        metadata["original_metadata"] = _as_metadata_dict(original_metadata)
    return metadata


def _failure_metadata_from_exception(
    exc: BaseException,
    *,
    original_metadata: Any,
    elapsed_ms: int | None = None,
    http_status: int | None = None,
    policy: str | None = None,
    budget_ms: int | None = None,
    source_error: str | None = None,
) -> dict[str, Any]:
    detail = build_failure_detail(
        exc,
        elapsed_ms=elapsed_ms,
        http_status=http_status,
        response_status=getattr(exc, "response_status", None),
    )
    return _build_row_metadata(
        {},
        source_error=source_error or str(detail.get("error", f"{type(exc).__name__}: {exc}")),
        failure_class=str(detail.get("failure_class")),
        failure_stage=str(detail.get("failure_stage")),
        http_status=int(detail["http_status"]) if detail.get("http_status") is not None else http_status,
        elapsed_ms=int(detail["elapsed_ms"]) if detail.get("elapsed_ms") is not None else elapsed_ms,
        policy=policy,
        budget_ms=budget_ms,
        response_status=str(detail.get("response_status", "error")),
        original_metadata=original_metadata,
    )


def _parse_error_detail(body: str) -> dict[str, Any]:
    text = str(body or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"error": text}
    if isinstance(parsed, Mapping):
        detail = parsed.get("detail", parsed)
        if isinstance(detail, Mapping):
            return dict(detail)
    return {"error": text}


def _extract_doc_id(node: Any) -> str:
    for attr in ("id_", "node_id", "id"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value:
            return value
    return "<unknown-doc-id>"


def _extract_text(node: Any) -> str:
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text

    if hasattr(node, "get_content"):
        try:
            content = node.get_content()
            return content if isinstance(content, str) else str(content)
        except Exception:
            return ""

    return ""


def _normalise_source_nodes(source_nodes: list[Any]) -> tuple[list[str], list[str]]:
    context_texts: list[str] = []
    context_ids: list[str] = []

    for source in source_nodes:
        node = getattr(source, "node", source)
        context_texts.append(_extract_text(node))
        context_ids.append(_extract_doc_id(node))

    return context_texts, context_ids


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Support files ending in .jsonl that actually contain a JSON array.
    if text[0] == "[":
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(value)!r}")
        return [row for row in value if isinstance(row, dict)]

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(
                f"Expected object row in JSONL at {path}:{line_no}, got {type(obj)!r}"
            )
        rows.append(obj)

    return rows


def load_raw_examples(path: str | Path) -> list[dict[str, Any]]:
    """Load raw evaluation examples from JSON or JSONL."""

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Raw dataset not found: {p}")
    return _read_json_or_jsonl(p)


def load_prepared_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load prepared rows from JSON or JSONL."""

    return load_raw_examples(path)


def load_sample_ids(path: str | Path) -> list[str]:
    """Load sample IDs from a plain-text file or JSON/JSONL payload."""

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Sample ID file not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    suffix = p.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        if text[0] == "[":
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError(f"Expected JSON list in {p}, got {type(payload)!r}")
            return _coerce_sample_ids(payload, source=str(p))

        ids: list[str] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {p}:{line_no}: {exc}") from exc
            ids.extend(_coerce_sample_ids([payload], source=f"{p}:{line_no}"))
        return ids

    ids: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        ids.extend(_coerce_sample_ids([value], source=f"{p}:{line_no}"))
    return ids


def load_sample_categories(path: str | Path) -> dict[str, list[str]]:
    """Load category -> sample-id mappings from JSON or YAML."""

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Sample category file not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)

    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping of category -> sample ids in {p}, got {type(payload)!r}")

    categories: dict[str, list[str]] = {}
    for raw_category, raw_ids in payload.items():
        category = str(raw_category).strip()
        if not category:
            raise ValueError(f"Encountered empty category name in {p}")
        if not isinstance(raw_ids, list):
            raise ValueError(f"Expected list of sample ids for category '{category}' in {p}")
        categories[category] = _coerce_sample_ids(raw_ids, source=f"{p}:{category}")
    return categories


def split_raw_examples_by_ids(
    raw_examples: Iterable[Mapping[str, Any]],
    selected_ids: Iterable[str],
    *,
    id_field: str = "id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split raw examples into dev and test rows using the provided test IDs."""

    dataset_by_id: dict[str, dict[str, Any]] = {}
    dataset_order: list[str] = []

    for index, example in enumerate(raw_examples):
        row = dict(example)
        sample_id = str(row.get(id_field, "") or "").strip()
        if not sample_id:
            raise ValueError(f"Missing '{id_field}' for dataset row index={index}")
        if sample_id in dataset_by_id:
            raise ValueError(f"Duplicate dataset sample id '{sample_id}' at index={index}")
        dataset_by_id[sample_id] = row
        dataset_order.append(sample_id)

    normalized_selected_ids = _coerce_sample_ids(selected_ids, source="selected_ids")
    selected_counts: dict[str, int] = {}
    for sample_id in normalized_selected_ids:
        selected_counts[sample_id] = selected_counts.get(sample_id, 0) + 1
    duplicate_ids = sorted(sample_id for sample_id, count in selected_counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate test sample ids provided: {', '.join(duplicate_ids)}")

    missing_ids = [sample_id for sample_id in normalized_selected_ids if sample_id not in dataset_by_id]
    if missing_ids:
        raise ValueError(f"Unknown test sample ids: {', '.join(missing_ids)}")

    selected_id_set = set(normalized_selected_ids)
    test_rows = [dataset_by_id[sample_id] for sample_id in normalized_selected_ids]
    dev_rows = [dataset_by_id[sample_id] for sample_id in dataset_order if sample_id not in selected_id_set]
    return dev_rows, test_rows


def stratified_split_raw_examples_by_categories(
    raw_examples: Iterable[Mapping[str, Any]],
    categories: Mapping[str, list[str]],
    *,
    test_size: int,
    random_state: int = 42,
    id_field: str = "id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split raw examples using stratified category labels for non-singleton categories."""

    dataset_by_id: dict[str, dict[str, Any]] = {}
    dataset_order: list[str] = []

    for index, example in enumerate(raw_examples):
        row = dict(example)
        sample_id = str(row.get(id_field, "") or "").strip()
        if not sample_id:
            raise ValueError(f"Missing '{id_field}' for dataset row index={index}")
        if sample_id in dataset_by_id:
            raise ValueError(f"Duplicate dataset sample id '{sample_id}' at index={index}")
        dataset_by_id[sample_id] = row
        dataset_order.append(sample_id)

    category_lookup: dict[str, str] = {}
    normalized_categories: dict[str, list[str]] = {}
    for raw_category, raw_ids in categories.items():
        category = str(raw_category).strip()
        sample_ids = _coerce_sample_ids(raw_ids, source=f"categories:{category}")
        normalized_categories[category] = sample_ids
        for sample_id in sample_ids:
            existing_category = category_lookup.get(sample_id)
            if existing_category is not None:
                raise ValueError(
                    f"Sample id '{sample_id}' is assigned to multiple categories: "
                    f"'{existing_category}' and '{category}'"
                )
            category_lookup[sample_id] = category

    missing_ids = sorted(sample_id for sample_id in category_lookup if sample_id not in dataset_by_id)
    if missing_ids:
        raise ValueError(f"Category mapping contains unknown sample ids: {', '.join(missing_ids)}")

    eligible_rows = [
        (sample_id, category)
        for category, sample_ids in normalized_categories.items()
        if len(sample_ids) > 1
        for sample_id in sample_ids
    ]
    if not eligible_rows:
        raise ValueError("No eligible categories with more than one sample were provided.")

    train_test_split = _import_train_test_split()
    ticket_ids = [sample_id for sample_id, _ in eligible_rows]
    labels = [category for _, category in eligible_rows]

    dev_ids, test_ids = train_test_split(
        ticket_ids,
        test_size=int(test_size),
        random_state=int(random_state),
        stratify=labels,
    )

    eligible_id_set = set(ticket_ids)
    remaining_dev_ids = [
        sample_id
        for sample_id in dataset_order
        if sample_id not in eligible_id_set
    ]

    dev_rows = [dataset_by_id[sample_id] for sample_id in dev_ids]
    dev_rows.extend(dataset_by_id[sample_id] for sample_id in remaining_dev_ids)
    test_rows = [dataset_by_id[sample_id] for sample_id in test_ids]

    category_test_counts = {
        category: len(set(test_ids) & set(sample_ids))
        for category, sample_ids in normalized_categories.items()
    }
    stats = {
        "test_ids": list(test_ids),
        "dev_ids": list(dev_ids),
        "category_test_counts": category_test_counts,
        "excluded_singleton_categories": sorted(
            category for category, sample_ids in normalized_categories.items() if len(sample_ids) <= 1
        ),
        "uncategorized_ids": [
            sample_id for sample_id in dataset_order if sample_id not in category_lookup
        ],
    }
    return dev_rows, test_rows, stats


def _resolve_requested_test_size(
    *,
    total_rows: int,
    test_size: int | None,
    test_fraction: float | None,
) -> int:
    if total_rows <= 1:
        raise ValueError("Need at least two rows to create a dev/test split.")

    if test_size is not None and test_fraction is not None:
        raise ValueError("Provide either test_size or test_fraction, not both.")

    if test_fraction is not None:
        fraction = float(test_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"test_fraction must be between 0 and 1, got {fraction!r}")
        resolved = int(round(total_rows * fraction))
    elif test_size is not None:
        resolved = int(test_size)
    else:
        raise ValueError("Provide either test_size or test_fraction.")

    if resolved <= 0 or resolved >= total_rows:
        raise ValueError(
            f"Resolved test size must be between 1 and {total_rows - 1}, got {resolved}."
        )
    return resolved


def _annotation_docs_scope_bucket(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized in {"local_and_external", "external_official"}:
        return "external_involved"
    if normalized == "local_official":
        return "local_official"
    return "none"


def _annotation_split_fields(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source_needed": str(row.get("source_needed", "") or "").strip(),
        "docs_scope_bucket": _annotation_docs_scope_bucket(row.get("docs_scope_needed", "")),
        "validity_sensitive": str(row.get("validity_sensitive", "") or "").strip(),
        "attachment_dependent": str(row.get("attachment_dependent", "") or "").strip(),
        "query_type": str(row.get("query_type", "") or "").strip(),
        "version_sensitive": str(row.get("version_sensitive", "") or "").strip(),
        "system_scope_required": str(row.get("system_scope_required", "") or "").strip(),
    }


def _annotation_feature_names(split_fields: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(f"{field}={value}" for field, value in split_fields.items())


def _count_selected_features(
    selected_ids: Iterable[str],
    *,
    features_by_id: Mapping[str, tuple[str, ...]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sample_id in selected_ids:
        counts.update(features_by_id[sample_id])
    return counts


def _multilabel_selection_score(
    counts: Mapping[str, int],
    *,
    target_counts: Mapping[str, int],
    total_counts: Mapping[str, int],
) -> float:
    score = 0.0
    for feature_name, total_count in total_counts.items():
        observed = int(counts.get(feature_name, 0))
        target = int(target_counts.get(feature_name, 0))
        diff = observed - target
        weight = 1.0 / max(int(total_count), 1)
        score += weight * float(diff * diff)
        if target > 0 and observed == 0:
            score += 100.0
    return score


def _greedy_initial_multilabel_selection(
    *,
    sample_ids: list[str],
    features_by_id: Mapping[str, tuple[str, ...]],
    target_counts: Mapping[str, int],
    total_counts: Mapping[str, int],
    test_size: int,
    random_state: int,
) -> list[str]:
    rng = random.Random(int(random_state))
    remaining_ids = list(sample_ids)
    rng.shuffle(remaining_ids)

    selected_ids: list[str] = []
    selected_counts: Counter[str] = Counter()

    while remaining_ids and len(selected_ids) < test_size:
        deficits = {
            feature_name: max(0, int(target_counts.get(feature_name, 0)) - int(selected_counts.get(feature_name, 0)))
            for feature_name in total_counts
        }
        best_id: str | None = None
        best_score: tuple[float, float] | None = None

        for sample_id in remaining_ids:
            gain = 0.0
            overfill = 0.0
            for feature_name in features_by_id[sample_id]:
                weight = 1.0 / max(int(total_counts.get(feature_name, 1)), 1)
                deficit = deficits.get(feature_name, 0)
                if deficit > 0:
                    gain += float(deficit) + weight
                else:
                    overfill += weight
            candidate_score = (gain, -overfill)
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_id = sample_id

        if best_id is None or (best_score is not None and best_score[0] <= 0.0):
            break

        remaining_ids.remove(best_id)
        selected_ids.append(best_id)
        selected_counts.update(features_by_id[best_id])

    if len(selected_ids) < test_size:
        selected_ids.extend(remaining_ids[: test_size - len(selected_ids)])

    return selected_ids


def _optimise_multilabel_selection(
    *,
    sample_ids: list[str],
    features_by_id: Mapping[str, tuple[str, ...]],
    target_counts: Mapping[str, int],
    total_counts: Mapping[str, int],
    test_size: int,
    random_state: int,
    restarts: int = 8,
) -> tuple[list[str], float, Counter[str]]:
    sample_order = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    best_selected_ids: list[str] = []
    best_counts: Counter[str] = Counter()
    best_score = float("inf")

    for restart in range(max(int(restarts), 1)):
        selected_ids = _greedy_initial_multilabel_selection(
            sample_ids=sample_ids,
            features_by_id=features_by_id,
            target_counts=target_counts,
            total_counts=total_counts,
            test_size=test_size,
            random_state=int(random_state) + restart,
        )
        selected_ids = list(dict.fromkeys(selected_ids))[:test_size]
        selected_set = set(selected_ids)
        unselected_ids = [sample_id for sample_id in sample_ids if sample_id not in selected_set]
        current_counts = _count_selected_features(selected_ids, features_by_id=features_by_id)
        current_score = _multilabel_selection_score(
            current_counts,
            target_counts=target_counts,
            total_counts=total_counts,
        )

        improved = True
        while improved:
            improved = False
            swap_choice: tuple[float, int, int, str, str] | None = None
            next_counts: Counter[str] | None = None

            for out_id in selected_ids:
                for in_id in unselected_ids:
                    candidate_counts = Counter(current_counts)
                    candidate_counts.subtract(features_by_id[out_id])
                    candidate_counts.update(features_by_id[in_id])
                    candidate_score = _multilabel_selection_score(
                        candidate_counts,
                        target_counts=target_counts,
                        total_counts=total_counts,
                    )
                    candidate_key = (
                        candidate_score,
                        sample_order[in_id],
                        sample_order[out_id],
                        in_id,
                        out_id,
                    )
                    if swap_choice is None or candidate_key < swap_choice:
                        swap_choice = candidate_key
                        next_counts = candidate_counts

            if swap_choice is None or next_counts is None:
                break

            candidate_score, _, _, in_id, out_id = swap_choice
            if candidate_score + 1e-12 >= current_score:
                break

            selected_ids = [in_id if sample_id == out_id else sample_id for sample_id in selected_ids]
            selected_ids.sort(key=sample_order.get)
            selected_set = set(selected_ids)
            unselected_ids = [sample_id for sample_id in sample_ids if sample_id not in selected_set]
            current_counts = next_counts
            current_score = candidate_score
            improved = True

        candidate_selected_ids = sorted(selected_ids, key=sample_order.get)
        candidate_key = (current_score, tuple(candidate_selected_ids))
        best_key = (best_score, tuple(best_selected_ids))
        if not best_selected_ids or candidate_key < best_key:
            best_selected_ids = candidate_selected_ids
            best_counts = Counter(current_counts)
            best_score = current_score

    return best_selected_ids, best_score, best_counts


def _count_field_values(rows: Iterable[Mapping[str, str]], *, field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field, "") or "").strip() for row in rows)
    return {key: counts[key] for key in sorted(counts)}


def stratified_split_raw_examples_by_annotation_labels(
    raw_examples: Iterable[Mapping[str, Any]],
    annotation_rows: Iterable[Mapping[str, Any]],
    *,
    test_size: int | None = None,
    test_fraction: float | None = None,
    random_state: int = 42,
    id_field: str = "id",
    require_verified: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split raw examples using multilabel benchmark annotations."""

    from polaris_rag.evaluation.benchmark_annotations import validate_annotation_rows

    raw_rows = [dict(row) for row in raw_examples]
    validated_annotations = validate_annotation_rows(
        annotation_rows=annotation_rows,
        raw_examples=raw_rows,
        split_lookup=None,
        require_verified=bool(require_verified),
    )

    dataset_by_id: dict[str, dict[str, Any]] = {}
    dataset_order: list[str] = []
    for index, row in enumerate(raw_rows):
        sample_id = str(row.get(id_field, "") or "").strip()
        if not sample_id:
            raise ValueError(f"Missing '{id_field}' for dataset row index={index}")
        if sample_id in dataset_by_id:
            raise ValueError(f"Duplicate dataset sample id '{sample_id}' at index={index}")
        dataset_by_id[sample_id] = row
        dataset_order.append(sample_id)

    resolved_test_size = _resolve_requested_test_size(
        total_rows=len(dataset_order),
        test_size=test_size,
        test_fraction=test_fraction,
    )

    annotations_by_id = {str(row["id"]): dict(row) for row in validated_annotations}
    split_fields_by_id = {
        sample_id: _annotation_split_fields(annotations_by_id[sample_id])
        for sample_id in dataset_order
    }
    features_by_id = {
        sample_id: _annotation_feature_names(split_fields_by_id[sample_id])
        for sample_id in dataset_order
    }
    feature_names = sorted({feature for features in features_by_id.values() for feature in features})
    feature_totals = {
        feature_name: sum(1 for features in features_by_id.values() if feature_name in features)
        for feature_name in feature_names
    }
    feature_target_counts = {
        feature_name: int(round(feature_totals[feature_name] * resolved_test_size / len(dataset_order)))
        for feature_name in feature_names
    }

    test_ids, selection_score, feature_test_counts = _optimise_multilabel_selection(
        sample_ids=dataset_order,
        features_by_id=features_by_id,
        target_counts=feature_target_counts,
        total_counts=feature_totals,
        test_size=resolved_test_size,
        random_state=random_state,
    )
    test_id_set = set(test_ids)
    dev_ids = [sample_id for sample_id in dataset_order if sample_id not in test_id_set]

    dev_rows = [dataset_by_id[sample_id] for sample_id in dev_ids]
    test_rows = [dataset_by_id[sample_id] for sample_id in test_ids]

    annotation_dev_rows = [annotations_by_id[sample_id] for sample_id in dev_ids]
    annotation_test_rows = [annotations_by_id[sample_id] for sample_id in test_ids]

    field_names = (
        "source_needed",
        "docs_scope_bucket",
        "validity_sensitive",
        "attachment_dependent",
        "query_type",
        "version_sensitive",
        "system_scope_required",
    )
    field_distributions = {
        "all": {
            field: _count_field_values(split_fields_by_id.values(), field=field)
            for field in field_names
        },
        "dev": {
            field: _count_field_values(
                (_annotation_split_fields(row) for row in annotation_dev_rows),
                field=field,
            )
            for field in field_names
        },
        "test": {
            field: _count_field_values(
                (_annotation_split_fields(row) for row in annotation_test_rows),
                field=field,
            )
            for field in field_names
        },
    }

    stats = {
        "strategy": "annotation_multilabel",
        "random_state": int(random_state),
        "test_size": int(resolved_test_size),
        "test_fraction": float(resolved_test_size / len(dataset_order)),
        "selection_score": float(selection_score),
        "test_ids": list(test_ids),
        "dev_ids": list(dev_ids),
        "feature_totals": dict(feature_totals),
        "feature_target_counts": dict(feature_target_counts),
        "feature_test_counts": {feature_name: int(feature_test_counts.get(feature_name, 0)) for feature_name in feature_names},
        "field_distributions": field_distributions,
    }
    return dev_rows, test_rows, stats


def persist_prepared_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Persist prepared rows to JSON or JSONL based on file extension."""

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    row_list = list(rows)
    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        with p.open("w", encoding="utf-8") as f:
            for row in row_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        p.write_text(json.dumps(row_list, ensure_ascii=False, indent=2), encoding="utf-8")

    return p


def _coerce_sample_ids(values: Iterable[Any], *, source: str) -> list[str]:
    ids: list[str] = []
    for value in values:
        sample_id = ""
        if isinstance(value, str):
            sample_id = value.strip()
        elif isinstance(value, Mapping):
            sample_id = str(value.get("id", "") or "").strip()
        else:
            raise ValueError(f"Unsupported sample id payload in {source}: {type(value)!r}")

        if not sample_id:
            raise ValueError(f"Missing sample id in {source}")
        ids.append(sample_id)
    return ids


def _import_train_test_split() -> Any:
    try:
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise RuntimeError(
            "Stratified category splitting requires scikit-learn. Install the evaluation extras first."
        ) from exc
    return train_test_split


def _prepare_one(
    index: int,
    example: dict[str, Any],
    *,
    pipeline: Any,
    query_field: str,
    reference_field: str,
    id_field: str,
    llm_generate_overrides: dict[str, Any] | None,
    raise_exceptions: bool,
    policy: str | None = None,
    budget_ms: int | None = None,
) -> tuple[int, dict[str, Any]]:
    query = str(example.get(query_field, "") or "").strip()
    reference = str(example.get(reference_field, "") or "").strip()
    sample_id = str(example.get(id_field, f"row-{index}"))
    base_metadata = example.get("metadata", {})

    if not query:
        if raise_exceptions:
            raise ValueError(f"Missing query for example index={index} id={sample_id}")
        return index, {
            "id": sample_id,
            "user_input": "",
            "reference": reference,
            "response": "",
            "retrieved_contexts": [],
            "retrieved_context_ids": [],
            "metadata": _build_row_metadata(
                {},
                source_error="missing query",
                failure_class=FAILURE_CLASS_INVALID_INPUT,
                failure_stage=FAILURE_STAGE_DATASET,
                policy=policy,
                budget_ms=budget_ms,
                response_status="invalid_input",
            ),
        }

    started_at = time.perf_counter()
    try:
        run_kwargs: dict[str, Any] = {}
        if llm_generate_overrides:
            run_kwargs["llm_generate"] = dict(llm_generate_overrides)

        result = pipeline.run(query, **run_kwargs)
        response = str(result.get("response", "") or "")
        source_nodes = result.get("source_nodes", []) or []
        retrieved_contexts, retrieved_context_ids = _normalise_source_nodes(source_nodes)
        timings = result.get("timings", {})
        elapsed_ms = _to_int(
            _as_metadata_dict(timings).get("retrieval_elapsed_ms"),
            -1,
        )
        generation_elapsed_ms = _to_int(_as_metadata_dict(timings).get("generation_elapsed_ms"), -1)
        if generation_elapsed_ms >= 0:
            elapsed_ms = max(elapsed_ms, 0) + generation_elapsed_ms if elapsed_ms >= 0 else generation_elapsed_ms
        if elapsed_ms < 0:
            elapsed_ms = max(0, int(round((time.perf_counter() - started_at) * 1000.0)))

        row = {
            "id": sample_id,
            "user_input": query,
            "reference": reference,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_context_ids": retrieved_context_ids,
            "metadata": _build_row_metadata(
                base_metadata,
                elapsed_ms=elapsed_ms,
                policy=policy,
                budget_ms=budget_ms,
                response_status="ok" if response.strip() else "empty_response",
            ),
        }
        return index, row

    except Exception as exc:
        if raise_exceptions:
            raise

        elapsed_ms = max(0, int(round((time.perf_counter() - started_at) * 1000.0)))

        row = {
            "id": sample_id,
            "user_input": query,
            "reference": reference,
            "response": "",
            "retrieved_contexts": [],
            "retrieved_context_ids": [],
            "metadata": _failure_metadata_from_exception(
                exc,
                original_metadata=base_metadata,
                elapsed_ms=elapsed_ms,
                policy=policy,
                budget_ms=budget_ms,
                source_error=f"{type(exc).__name__}: {exc} (example index={index} id={sample_id})",
            ),
        }
        return index, row


def _post_query_api(
    api_url: str,
    query: str,
    timeout_seconds: float,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update({str(k): str(v) for k, v in headers.items()})

    payload = json.dumps({"query": query}).encode("utf-8")
    request = Request(api_url, data=payload, headers=request_headers, method="POST")

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        detail_payload = _parse_error_detail(detail)
        message = str(detail_payload.get("error", f"query API HTTP {exc.code}: {detail}"))
        raise QueryAPIError(
            message,
            status_code=int(exc.code),
            detail={
                **detail_payload,
                "http_status": int(exc.code),
            },
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise QueryAPIError(
            f"query API transport timeout after {float(timeout_seconds):.3f}s",
            detail={
                "failure_class": FAILURE_CLASS_TRANSPORT_ERROR,
                "failure_stage": FAILURE_STAGE_API,
                "response_status": "client_timeout",
            },
        ) from exc
    except URLError as exc:
        raise QueryAPIError(
            f"query API connection error: {exc}",
            detail={
                "failure_class": FAILURE_CLASS_TRANSPORT_ERROR,
                "failure_stage": FAILURE_STAGE_API,
                "response_status": "connection_error",
            },
        ) from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise QueryAPIError(
            f"query API returned invalid JSON: {exc}",
            detail={
                "failure_class": FAILURE_CLASS_API_INTERNAL_ERROR,
                "failure_stage": FAILURE_STAGE_API,
                "response_status": "invalid_json",
            },
        ) from exc

    if not isinstance(parsed, dict):
        raise QueryAPIError(
            f"query API returned non-object response: {type(parsed)!r}",
            detail={
                "failure_class": FAILURE_CLASS_API_INTERNAL_ERROR,
                "failure_stage": FAILURE_STAGE_API,
                "response_status": "invalid_json",
            },
        )

    return parsed


def _extract_api_context_chunks(value: Any) -> tuple[list[str], list[str]]:
    if not isinstance(value, list):
        return [], []

    texts: list[str] = []
    ids: list[str] = []
    for idx, chunk in enumerate(value, start=1):
        if isinstance(chunk, Mapping):
            text = str(chunk.get("text", "") or "")
            doc_id = str(chunk.get("doc_id", f"context-{idx}") or f"context-{idx}")
        else:
            text = str(chunk or "")
            doc_id = f"context-{idx}"
        texts.append(text)
        ids.append(doc_id)

    return texts, ids


def _prepare_one_via_api(
    index: int,
    example: dict[str, Any],
    *,
    api_url: str,
    query_field: str,
    reference_field: str,
    id_field: str,
    raise_exceptions: bool,
    timeout_seconds: float,
    headers: Mapping[str, str] | None,
    requester: ApiRequester,
    policy: str | None = None,
    budget_ms: int | None = None,
) -> tuple[int, dict[str, Any]]:
    query = str(example.get(query_field, "") or "").strip()
    reference = str(example.get(reference_field, "") or "").strip()
    sample_id = str(example.get(id_field, f"row-{index}"))
    base_metadata = example.get("metadata", {})

    if not query:
        if raise_exceptions:
            raise ValueError(f"Missing query for example index={index} id={sample_id}")
        return index, {
            "id": sample_id,
            "user_input": "",
            "reference": reference,
            "response": "",
            "retrieved_contexts": [],
            "retrieved_context_ids": [],
            "metadata": _build_row_metadata(
                {},
                source_error="missing query",
                failure_class=FAILURE_CLASS_INVALID_INPUT,
                failure_stage=FAILURE_STAGE_DATASET,
                policy=policy,
                budget_ms=budget_ms,
                response_status="invalid_input",
            ),
        }

    started_at = time.perf_counter()
    try:
        result = requester(api_url, query, timeout_seconds, headers)
        response = str(result.get("answer", result.get("response", "")) or "")
        retrieved_contexts, retrieved_context_ids = _extract_api_context_chunks(
            result.get("context", [])
        )
        elapsed_ms = max(0, int(round((time.perf_counter() - started_at) * 1000.0)))

        row = {
            "id": sample_id,
            "user_input": query,
            "reference": reference,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_context_ids": retrieved_context_ids,
            "metadata": _build_row_metadata(
                base_metadata,
                http_status=200,
                elapsed_ms=elapsed_ms,
                policy=policy,
                budget_ms=budget_ms,
                response_status="ok" if response.strip() else "empty_response",
            ),
        }
        return index, row

    except Exception as exc:
        if raise_exceptions:
            raise

        elapsed_ms = max(0, int(round((time.perf_counter() - started_at) * 1000.0)))
        status_code = getattr(exc, "status_code", None)
        metadata = _failure_metadata_from_exception(
            exc,
            original_metadata=base_metadata,
            elapsed_ms=elapsed_ms,
            http_status=status_code,
            policy=policy,
            budget_ms=budget_ms,
            source_error=f"{type(exc).__name__}: {exc} (example index={index} id={sample_id})",
        )

        row = {
            "id": sample_id,
            "user_input": query,
            "reference": reference,
            "response": "",
            "retrieved_contexts": [],
            "retrieved_context_ids": [],
            "metadata": metadata,
        }
        return index, row


def _error_text_from_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _row_trace_outputs(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    source_error = None
    failure_class = None
    failure_stage = None
    response_status = None
    if isinstance(metadata, Mapping):
        value = metadata.get("source_error")
        if value is not None:
            source_error = str(value)
        if metadata.get("failure_class") is not None:
            failure_class = str(metadata.get("failure_class"))
        if metadata.get("failure_stage") is not None:
            failure_stage = str(metadata.get("failure_stage"))
        if metadata.get("response_status") is not None:
            response_status = str(metadata.get("response_status"))

    return {
        "response": str(row.get("response", "") or ""),
        "retrieved_context_ids": list(row.get("retrieved_context_ids", []) or []),
        "retrieved_contexts": list(row.get("retrieved_contexts", []) or []),
        "source_error": source_error,
        "failure_class": failure_class,
        "failure_stage": failure_stage,
        "response_status": response_status,
    }


@contextlib.contextmanager
def _open_attempt_trace(
    trace_factory: PrepAttemptTraceFactory | None,
    *,
    name: str,
    inputs: Mapping[str, Any],
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[PrepTraceRecorder]:
    if trace_factory is None:
        yield _NoopTraceRecorder()
        return

    with trace_factory(name, inputs, attributes) as recorder:
        yield recorder


def _row_has_source_error(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    return isinstance(metadata, Mapping) and bool(metadata.get("source_error"))


def _source_error_text(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("source_error")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_missing_query_source_error(source_error: str | None) -> bool:
    return (source_error or "").strip().lower() == "missing query"


def _is_missing_query_exception(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and str(exc).startswith("Missing query for example ")


def _row_has_empty_response(row: dict[str, Any]) -> bool:
    response = row.get("response")
    return str(response or "").strip() == ""


def _compute_retry_delay_seconds(policy: PrepRetryPolicy, attempt_number: int) -> float:
    # Exponential backoff where attempt_number is 1-indexed.
    expo = policy.initial_backoff_seconds * (2 ** max(0, attempt_number - 1))
    base_delay = min(policy.max_backoff_seconds, expo)
    jitter = random.uniform(0.0, policy.jitter_seconds) if policy.jitter_seconds > 0 else 0.0
    return max(0.0, base_delay + jitter)


def _sleep_before_retry(policy: PrepRetryPolicy, attempt_number: int) -> None:
    delay_seconds = _compute_retry_delay_seconds(policy, attempt_number=attempt_number)
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def _fail_soft_empty_response_row(
    *,
    row: dict[str, Any],
    message: str,
    original_metadata: Any,
    policy: str | None = None,
    budget_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "id": str(row.get("id", "") or ""),
        "user_input": str(row.get("user_input", "") or ""),
        "reference": str(row.get("reference", "") or ""),
        "response": "",
        "retrieved_contexts": list(row.get("retrieved_contexts", []) or []),
        "retrieved_context_ids": list(row.get("retrieved_context_ids", []) or []),
        "metadata": _build_row_metadata(
            row.get("metadata", {}),
            source_error=message,
            failure_class="empty_response",
            failure_stage=FAILURE_STAGE_GENERATION,
            http_status=_to_int(_as_metadata_dict(row.get("metadata", {})).get("http_status"), 200),
            elapsed_ms=_to_int(_as_metadata_dict(row.get("metadata", {})).get("elapsed_ms"), 0),
            policy=policy,
            budget_ms=budget_ms,
            response_status="empty_response",
            original_metadata=original_metadata,
        ),
    }


def _prepare_one_with_retries(
    index: int,
    example: dict[str, Any],
    *,
    pipeline: Any,
    query_field: str,
    reference_field: str,
    id_field: str,
    llm_generate_overrides: dict[str, Any] | None,
    raise_exceptions: bool,
    retry_policy: PrepRetryPolicy,
    trace_factory: PrepAttemptTraceFactory | None = None,
    evaluation_policy: str | None = None,
    budget_ms: int | None = None,
) -> tuple[int, dict[str, Any]]:
    policy = PrepRetryPolicy.from_value(retry_policy)
    sample_id = str(example.get(id_field, f"row-{index}"))
    query = str(example.get(query_field, "") or "").strip()

    for attempt in range(1, policy.max_attempts + 1):
        with _open_attempt_trace(
            trace_factory,
            name="polaris.dataset_preparation.pipeline_request",
            inputs={
                "sample_id": sample_id,
                "query": query,
                "attempt": attempt,
            },
            attributes={
                "stage": "dataset_preparation",
                "mode": "pipeline",
                "sample_id": sample_id,
                "attempt": attempt,
            },
        ) as trace_recorder:
            try:
                row_index, row = _prepare_one(
                    index,
                    example,
                    pipeline=pipeline,
                    query_field=query_field,
                    reference_field=reference_field,
                    id_field=id_field,
                    llm_generate_overrides=llm_generate_overrides,
                    raise_exceptions=raise_exceptions,
                    policy=evaluation_policy,
                    budget_ms=budget_ms,
                )
            except Exception as exc:
                trace_recorder.set_outputs({"error": _error_text_from_exception(exc)})
                trace_recorder.set_attributes(
                    {
                        "status": "error",
                        "failure_class": getattr(exc, "failure_class", None),
                        "failure_stage": getattr(exc, "failure_stage", None),
                        "response_status": getattr(exc, "response_status", "error"),
                    }
                )
                if attempt >= policy.max_attempts:
                    raise
                # Missing query is deterministic and should never be retried.
                if _is_missing_query_exception(exc):
                    raise
                _sleep_before_retry(policy, attempt_number=attempt)
                continue

            source_error = _source_error_text(row)
            trace_recorder.set_outputs(_row_trace_outputs(row))
            if source_error:
                trace_recorder.set_attributes(
                    {
                        "status": "source_error",
                        "failure_class": _as_metadata_dict(row.get("metadata", {})).get("failure_class"),
                        "failure_stage": _as_metadata_dict(row.get("metadata", {})).get("failure_stage"),
                        "response_status": _as_metadata_dict(row.get("metadata", {})).get("response_status"),
                    }
                )
                if _is_missing_query_source_error(source_error) or attempt >= policy.max_attempts:
                    return row_index, row
                _sleep_before_retry(policy, attempt_number=attempt)
                continue

            if _row_has_empty_response(row):
                error_message = (
                    f"ValueError: response is empty after {attempt} attempt(s) "
                    f"for example index={index} id={sample_id}"
                )
                trace_recorder.set_attributes({"status": "empty_response"})
                should_retry_empty = policy.retry_on_empty_response and attempt < policy.max_attempts
                if not should_retry_empty:
                    if raise_exceptions:
                        raise EmptyResponseError(error_message)
                    return row_index, _fail_soft_empty_response_row(
                        row=row,
                        message=error_message,
                        original_metadata=example.get("metadata", {}),
                        policy=evaluation_policy,
                        budget_ms=budget_ms,
                    )
                _sleep_before_retry(policy, attempt_number=attempt)
                continue

            trace_recorder.set_attributes({"status": "success"})
            return row_index, row

    # Defensive fallback; loop always returns/raises.
    raise RuntimeError("unreachable retry state in _prepare_one_with_retries")


def _prepare_one_via_api_with_retries(
    index: int,
    example: dict[str, Any],
    *,
    api_url: str,
    query_field: str,
    reference_field: str,
    id_field: str,
    raise_exceptions: bool,
    timeout_seconds: float,
    headers: Mapping[str, str] | None,
    requester: ApiRequester,
    retry_policy: PrepRetryPolicy,
    trace_factory: PrepAttemptTraceFactory | None = None,
    evaluation_policy: str | None = None,
    budget_ms: int | None = None,
) -> tuple[int, dict[str, Any]]:
    policy = PrepRetryPolicy.from_value(retry_policy)
    sample_id = str(example.get(id_field, f"row-{index}"))
    query = str(example.get(query_field, "") or "").strip()

    for attempt in range(1, policy.max_attempts + 1):
        with _open_attempt_trace(
            trace_factory,
            name="polaris.dataset_preparation.api_request",
            inputs={
                "sample_id": sample_id,
                "query": query,
                "attempt": attempt,
                "api_url": api_url,
            },
            attributes={
                "stage": "dataset_preparation",
                "mode": "api",
                "sample_id": sample_id,
                "attempt": attempt,
                "api_url": api_url,
                "timeout_seconds": float(timeout_seconds),
            },
        ) as trace_recorder:
            try:
                row_index, row = _prepare_one_via_api(
                    index,
                    example,
                    api_url=api_url,
                    query_field=query_field,
                    reference_field=reference_field,
                    id_field=id_field,
                    raise_exceptions=raise_exceptions,
                    timeout_seconds=timeout_seconds,
                    headers=headers,
                    requester=requester,
                    policy=evaluation_policy,
                    budget_ms=budget_ms,
                )
            except Exception as exc:
                trace_recorder.set_outputs({"error": _error_text_from_exception(exc)})
                trace_recorder.set_attributes(
                    {
                        "status": "error",
                        "failure_class": getattr(exc, "failure_class", None),
                        "failure_stage": getattr(exc, "failure_stage", None),
                        "response_status": getattr(exc, "response_status", "error"),
                    }
                )
                if _is_missing_query_exception(exc) or attempt >= policy.max_attempts:
                    raise
                _sleep_before_retry(policy, attempt_number=attempt)
                continue

            source_error = _source_error_text(row)
            trace_recorder.set_outputs(_row_trace_outputs(row))
            if source_error:
                trace_recorder.set_attributes(
                    {
                        "status": "source_error",
                        "failure_class": _as_metadata_dict(row.get("metadata", {})).get("failure_class"),
                        "failure_stage": _as_metadata_dict(row.get("metadata", {})).get("failure_stage"),
                        "response_status": _as_metadata_dict(row.get("metadata", {})).get("response_status"),
                    }
                )
                if _is_missing_query_source_error(source_error) or attempt >= policy.max_attempts:
                    return row_index, row
                _sleep_before_retry(policy, attempt_number=attempt)
                continue

            if _row_has_empty_response(row):
                error_message = (
                    f"ValueError: response is empty after {attempt} attempt(s) "
                    f"for example index={index} id={sample_id}"
                )
                trace_recorder.set_attributes({"status": "empty_response"})
                should_retry_empty = policy.retry_on_empty_response and attempt < policy.max_attempts
                if not should_retry_empty:
                    if raise_exceptions:
                        raise EmptyResponseError(error_message)
                    return row_index, _fail_soft_empty_response_row(
                        row=row,
                        message=error_message,
                        original_metadata=example.get("metadata", {}),
                        policy=evaluation_policy,
                        budget_ms=budget_ms,
                    )
                _sleep_before_retry(policy, attempt_number=attempt)
                continue

            trace_recorder.set_attributes({"status": "success"})
            return row_index, row

    # Defensive fallback; loop always returns/raises.
    raise RuntimeError("unreachable retry state in _prepare_one_via_api_with_retries")


def _emit_progress(
    *,
    callback: PrepProgressCallback | None,
    mode: str,
    completed: int,
    total: int,
    successes: int,
    failures: int,
    started_at: float,
    last_error: str | None = None,
) -> None:
    if callback is None:
        return

    callback(
        PrepProgressEvent(
            completed=completed,
            total=total,
            successes=successes,
            failures=failures,
            elapsed_seconds=max(0.0, time.perf_counter() - started_at),
            mode=mode,
            last_error=last_error,
        )
    )


def build_prepared_rows(
    *,
    raw_examples: list[dict[str, Any]],
    pipeline: Any,
    query_field: str = "query",
    reference_field: str = "expected_answer",
    id_field: str = "id",
    generation_workers: int = 1,
    llm_generate_overrides: dict[str, Any] | None = None,
    raise_exceptions: bool = False,
    retry_policy: PrepRetryPolicy | Mapping[str, Any] | None = None,
    progress_callback: PrepProgressCallback | None = None,
    trace_factory: PrepAttemptTraceFactory | None = None,
    policy: str | None = None,
    budget_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Convert raw examples into rows compatible with ``EvaluationDataset``.

    Parameters
    ----------
    raw_examples : list[dict[str, Any]]
        Raw dataset rows.
    pipeline : Any
        Polaris pipeline with ``run(query, **kwargs)``.
    generation_workers : int, default 1
        Number of worker threads used while generating responses/contexts.
    """

    workers = max(1, int(generation_workers))
    effective_retry_policy = PrepRetryPolicy.from_value(retry_policy)
    total = len(raw_examples)
    completed = 0
    successes = 0
    failures = 0
    started_at = time.perf_counter()
    indexed_rows: list[tuple[int, dict[str, Any]]] = []

    def _record_progress(row: dict[str, Any], *, last_error: str | None = None) -> None:
        nonlocal completed, successes, failures
        completed += 1
        if _row_has_source_error(row) or last_error:
            failures += 1
        else:
            successes += 1
        _emit_progress(
            callback=progress_callback,
            mode="pipeline",
            completed=completed,
            total=total,
            successes=successes,
            failures=failures,
            started_at=started_at,
            last_error=last_error,
        )

    if total == 0:
        _emit_progress(
            callback=progress_callback,
            mode="pipeline",
            completed=0,
            total=0,
            successes=0,
            failures=0,
            started_at=started_at,
        )
        return []

    if workers == 1:
        for i, example in enumerate(raw_examples):
            try:
                row = _prepare_one_with_retries(
                    i,
                    example,
                    pipeline=pipeline,
                    query_field=query_field,
                    reference_field=reference_field,
                    id_field=id_field,
                    llm_generate_overrides=llm_generate_overrides,
                    raise_exceptions=raise_exceptions,
                    retry_policy=effective_retry_policy,
                    trace_factory=trace_factory,
                    evaluation_policy=policy,
                    budget_ms=budget_ms,
                )
            except Exception as exc:
                _record_progress({}, last_error=_error_text_from_exception(exc))
                raise
            indexed_rows.append(row)
            _record_progress(row[1], last_error=_source_error_text(row[1]))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _prepare_one_with_retries,
                    i,
                    example,
                    pipeline=pipeline,
                    query_field=query_field,
                    reference_field=reference_field,
                    id_field=id_field,
                    llm_generate_overrides=llm_generate_overrides,
                    raise_exceptions=raise_exceptions,
                    retry_policy=effective_retry_policy,
                    trace_factory=trace_factory,
                    evaluation_policy=policy,
                    budget_ms=budget_ms,
                )
                for i, example in enumerate(raw_examples)
            ]
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    _record_progress({}, last_error=_error_text_from_exception(exc))
                    raise
                indexed_rows.append(row)
                _record_progress(row[1], last_error=_source_error_text(row[1]))

    indexed_rows.sort(key=lambda x: x[0])
    return [row for _, row in indexed_rows]


def build_prepared_rows_from_api(
    *,
    raw_examples: list[dict[str, Any]],
    api_url: str,
    query_field: str = "query",
    reference_field: str = "expected_answer",
    id_field: str = "id",
    generation_workers: int = 1,
    raise_exceptions: bool = False,
    timeout_seconds: float = 120.0,
    headers: Mapping[str, str] | None = None,
    requester: ApiRequester | None = None,
    retry_policy: PrepRetryPolicy | Mapping[str, Any] | None = None,
    progress_callback: PrepProgressCallback | None = None,
    trace_factory: PrepAttemptTraceFactory | None = None,
    policy: str | None = None,
    budget_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Convert raw examples into prepared rows by calling Polaris query API."""

    workers = max(1, int(generation_workers))
    effective_retry_policy = PrepRetryPolicy.from_value(retry_policy)
    total = len(raw_examples)
    completed = 0
    successes = 0
    failures = 0
    started_at = time.perf_counter()
    indexed_rows: list[tuple[int, dict[str, Any]]] = []
    effective_requester = requester or _post_query_api

    def _record_progress(row: dict[str, Any], *, last_error: str | None = None) -> None:
        nonlocal completed, successes, failures
        completed += 1
        if _row_has_source_error(row) or last_error:
            failures += 1
        else:
            successes += 1
        _emit_progress(
            callback=progress_callback,
            mode="api",
            completed=completed,
            total=total,
            successes=successes,
            failures=failures,
            started_at=started_at,
            last_error=last_error,
        )

    if total == 0:
        _emit_progress(
            callback=progress_callback,
            mode="api",
            completed=0,
            total=0,
            successes=0,
            failures=0,
            started_at=started_at,
        )
        return []

    if workers == 1:
        for i, example in enumerate(raw_examples):
            try:
                row = _prepare_one_via_api_with_retries(
                    i,
                    example,
                    api_url=api_url,
                    query_field=query_field,
                    reference_field=reference_field,
                    id_field=id_field,
                    raise_exceptions=raise_exceptions,
                    timeout_seconds=float(timeout_seconds),
                    headers=headers,
                    requester=effective_requester,
                    retry_policy=effective_retry_policy,
                    trace_factory=trace_factory,
                    evaluation_policy=policy,
                    budget_ms=budget_ms,
                )
            except Exception as exc:
                _record_progress({}, last_error=_error_text_from_exception(exc))
                raise
            indexed_rows.append(row)
            _record_progress(row[1], last_error=_source_error_text(row[1]))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _prepare_one_via_api_with_retries,
                    i,
                    example,
                    api_url=api_url,
                    query_field=query_field,
                    reference_field=reference_field,
                    id_field=id_field,
                    raise_exceptions=raise_exceptions,
                    timeout_seconds=float(timeout_seconds),
                    headers=headers,
                    requester=effective_requester,
                    retry_policy=effective_retry_policy,
                    trace_factory=trace_factory,
                    evaluation_policy=policy,
                    budget_ms=budget_ms,
                )
                for i, example in enumerate(raw_examples)
            ]
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    _record_progress({}, last_error=_error_text_from_exception(exc))
                    raise
                indexed_rows.append(row)
                _record_progress(row[1], last_error=_source_error_text(row[1]))

    indexed_rows.sort(key=lambda x: x[0])
    return [row for _, row in indexed_rows]


def to_evaluation_dataset(rows: list[dict[str, Any]]) -> "EvaluationDataset":
    """Create a RAGAS ``EvaluationDataset`` from prepared rows."""

    from ragas import EvaluationDataset

    return EvaluationDataset.from_list(rows)


__all__ = [
    "ApiRequester",
    "PrepProgressCallback",
    "PrepProgressEvent",
    "PrepRetryPolicy",
    "build_prepared_rows",
    "build_prepared_rows_from_api",
    "load_prepared_rows",
    "load_raw_examples",
    "load_sample_categories",
    "load_sample_ids",
    "persist_prepared_rows",
    "stratified_split_raw_examples_by_annotation_labels",
    "stratified_split_raw_examples_by_categories",
    "split_raw_examples_by_ids",
    "to_evaluation_dataset",
]
