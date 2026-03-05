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

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
) -> tuple[int, dict[str, Any]]:
    query = str(example.get(query_field, "") or "").strip()
    reference = str(example.get(reference_field, "") or "").strip()
    sample_id = str(example.get(id_field, f"row-{index}"))

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
            "metadata": {"source_error": "missing query"},
        }

    try:
        run_kwargs: dict[str, Any] = {}
        if llm_generate_overrides:
            run_kwargs["llm_generate"] = dict(llm_generate_overrides)

        result = pipeline.run(query, **run_kwargs)
        response = str(result.get("response", "") or "")
        source_nodes = result.get("source_nodes", []) or []
        retrieved_contexts, retrieved_context_ids = _normalise_source_nodes(source_nodes)

        row = {
            "id": sample_id,
            "user_input": query,
            "reference": reference,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_context_ids": retrieved_context_ids,
            "metadata": example.get("metadata", {}),
        }
        return index, row

    except Exception as exc:
        if raise_exceptions:
            raise

        row = {
            "id": sample_id,
            "user_input": query,
            "reference": reference,
            "response": "",
            "retrieved_contexts": [],
            "retrieved_context_ids": [],
            "metadata": {
                "source_error": f"{type(exc).__name__}: {exc}",
                "original_metadata": example.get("metadata", {}),
            },
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
        raise RuntimeError(f"query API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"query API connection error: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"query API returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"query API returned non-object response: {type(parsed)!r}")

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
) -> tuple[int, dict[str, Any]]:
    query = str(example.get(query_field, "") or "").strip()
    reference = str(example.get(reference_field, "") or "").strip()
    sample_id = str(example.get(id_field, f"row-{index}"))

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
            "metadata": {"source_error": "missing query"},
        }

    try:
        result = requester(api_url, query, timeout_seconds, headers)
        response = str(result.get("answer", result.get("response", "")) or "")
        retrieved_contexts, retrieved_context_ids = _extract_api_context_chunks(
            result.get("context", [])
        )

        row = {
            "id": sample_id,
            "user_input": query,
            "reference": reference,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_context_ids": retrieved_context_ids,
            "metadata": example.get("metadata", {}),
        }
        return index, row

    except Exception as exc:
        if raise_exceptions:
            raise

        row = {
            "id": sample_id,
            "user_input": query,
            "reference": reference,
            "response": "",
            "retrieved_contexts": [],
            "retrieved_context_ids": [],
            "metadata": {
                "source_error": f"{type(exc).__name__}: {exc}",
                "original_metadata": example.get("metadata", {}),
            },
        }
        return index, row


def _error_text_from_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _row_has_source_error(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    return isinstance(metadata, Mapping) and bool(metadata.get("source_error"))


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
    progress_callback: PrepProgressCallback | None = None,
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
                row = _prepare_one(
                    i,
                    example,
                    pipeline=pipeline,
                    query_field=query_field,
                    reference_field=reference_field,
                    id_field=id_field,
                    llm_generate_overrides=llm_generate_overrides,
                    raise_exceptions=raise_exceptions,
                )
            except Exception as exc:
                _record_progress({}, last_error=_error_text_from_exception(exc))
                raise
            indexed_rows.append(row)
            _record_progress(row[1])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _prepare_one,
                    i,
                    example,
                    pipeline=pipeline,
                    query_field=query_field,
                    reference_field=reference_field,
                    id_field=id_field,
                    llm_generate_overrides=llm_generate_overrides,
                    raise_exceptions=raise_exceptions,
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
                _record_progress(row[1])

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
    progress_callback: PrepProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Convert raw examples into prepared rows by calling Polaris query API."""

    workers = max(1, int(generation_workers))
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
                row = _prepare_one_via_api(
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
                )
            except Exception as exc:
                _record_progress({}, last_error=_error_text_from_exception(exc))
                raise
            indexed_rows.append(row)
            _record_progress(row[1])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _prepare_one_via_api,
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
                _record_progress(row[1])

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
    "build_prepared_rows",
    "build_prepared_rows_from_api",
    "load_prepared_rows",
    "load_raw_examples",
    "persist_prepared_rows",
    "to_evaluation_dataset",
]
