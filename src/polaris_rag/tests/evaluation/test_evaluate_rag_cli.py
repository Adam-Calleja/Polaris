from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
import csv
import logging
import sys
from types import SimpleNamespace

import pytest

from polaris_rag.cli import evaluate_rag
from polaris_rag.common import POLARIS_EVAL_INCLUDE_METADATA_HEADER
from polaris_rag.evaluation.experiment_presets import PresetContext
from polaris_rag.evaluation.evaluation_dataset import PrepProgressEvent


@dataclass
class _DummyContainer:
    pipeline: object


class _DummyTraceRecorder:
    def __init__(self) -> None:
        self.outputs = None
        self.attributes = None

    def set_outputs(self, outputs):  # noqa: ANN001
        self.outputs = outputs

    def set_attributes(self, attributes):  # noqa: ANN001
        self.attributes = dict(attributes)


class _DummyStageContext:
    def __init__(self, name: str = "dataset_preparation") -> None:
        self.name = name
        self.open_calls: list[dict[str, object]] = []
        self.active_open_calls: list[dict[str, object]] = []

    def correlation_headers(self) -> dict[str, str]:
        return {
            "X-Polaris-MLflow-Run-ID": "run-parent",
            "X-Polaris-MLflow-Child-Run-ID": "run-child",
            "X-Polaris-MLflow-Stage": self.name,
        }

    @contextmanager
    def open_detached_mirrored_span(self, name: str, *, inputs=None, attributes=None, include_child_trace=True):  # noqa: ANN001
        self.open_calls.append(
            {
                "name": name,
                "inputs": inputs,
                "attributes": attributes,
                "include_child_trace": include_child_trace,
            }
        )
        yield _DummyTraceRecorder()

    @contextmanager
    def open_active_mirrored_span(self, name: str, *, inputs=None, attributes=None, outputs=None):  # noqa: ANN001
        self.active_open_calls.append(
            {
                "name": name,
                "inputs": inputs,
                "attributes": attributes,
                "outputs": outputs,
            }
        )
        yield _DummyTraceRecorder()


class _DummyTracking:
    enabled = True

    def __init__(self) -> None:
        self._mlflow = object()
        self.logged_inputs: list[dict[str, object]] = []

    def log_input(self, dataset, *, context=None, tags=None):  # noqa: ANN001
        self.logged_inputs.append(
            {
                "dataset": dataset,
                "context": context,
                "tags": dict(tags or {}),
            }
        )


class _DummyMainTracking:
    def __init__(self, *, enabled: bool = True, tracing_enabled: bool = True) -> None:
        self.enabled = enabled
        self.runtime_config = SimpleNamespace(tracing=SimpleNamespace(enabled=tracing_enabled))
        self._run_id = "run-parent"
        self.logged_param_batches: list[dict[str, object]] = []
        self.logged_flat_params: list[dict[str, object]] = []
        self.logged_metrics: list[dict[str, object]] = []
        self.logged_artifacts: list[tuple[str, str | None]] = []
        self.logged_json_artifacts: list[dict[str, object]] = []
        self.dataset_stage = _DummyStageContext("dataset_preparation")
        self.eval_stage = _DummyStageContext("ragas_evaluation")

    @property
    def run_id(self) -> str:
        return self._run_id

    def trace_headers(self) -> dict[str, str]:
        return {"X-Polaris-MLflow-Run-ID": self._run_id}

    @contextmanager
    def open(self, *, run_name=None, extra_tags=None, strict=True):  # noqa: ANN001
        yield self

    @contextmanager
    def stage(self, name: str, *, tags=None):  # noqa: ANN001
        if name == "dataset_preparation":
            yield self.dataset_stage
            return
        if name == "ragas_evaluation":
            yield self.eval_stage
            return
        raise AssertionError(f"Unexpected stage: {name}")

    def log_flat_params(self, data, *, prefix=""):  # noqa: ANN001
        self.logged_flat_params.append({"prefix": prefix, "data": dict(data)})

    def log_params(self, params):  # noqa: ANN001
        self.logged_param_batches.append(dict(params))

    def log_artifact(self, path, *, artifact_path=None):  # noqa: ANN001
        self.logged_artifacts.append((str(path), artifact_path))

    def log_json_artifact(self, data, *, output_path, artifact_path=None):  # noqa: ANN001
        self.logged_json_artifacts.append(
            {
                "data": data,
                "output_path": str(output_path),
                "artifact_path": artifact_path,
            }
        )

    def log_metrics(self, metrics):  # noqa: ANN001
        self.logged_metrics.append(dict(metrics))

    def log_input(self, dataset, *, context=None, tags=None):  # noqa: ANN001
        return None


class _DummySeries:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def tolist(self) -> list[float]:
        return list(self._values)


class _DummyScoresFrame:
    def __init__(self, rows: list[dict[str, object]], metric_values: dict[str, list[float]]) -> None:
        self._rows = list(rows)
        self.columns = list(metric_values.keys())
        self._metric_values = {key: list(values) for key, values in metric_values.items()}

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: str) -> _DummySeries:
        return _DummySeries(self._metric_values[key])


def _fake_build_prepared_rows(**kwargs):  # noqa: ANN003
    callback = kwargs.get("progress_callback")
    if callback:
        callback(
            PrepProgressEvent(
                completed=1,
                total=1,
                successes=1,
                failures=0,
                elapsed_seconds=0.1,
                mode="pipeline",
                last_error=None,
            )
        )
    return [
        {
            "id": "row-1",
            "user_input": "Q1",
            "reference": "A1",
            "response": "R1",
            "retrieved_contexts": ["ctx-1"],
            "retrieved_context_ids": ["doc-1"],
            "metadata": {},
        }
    ]


def _cfg_with_reranker(tmp_path, rerank_cfg: dict[str, object]):  # noqa: ANN001
    config_path = tmp_path / "config.yaml"
    config_path.write_text("retriever: {}\n", encoding="utf-8")
    return SimpleNamespace(
        retriever={
            "type": "multi_collection",
            "rerank": dict(rerank_cfg),
        },
        config_path=config_path,
    )


def test_log_input_dataset_to_mlflow_uses_test_context(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "tickets.test.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")

    tracking = _DummyTracking()

    monkeypatch.setattr(
        evaluate_rag,
        "_build_mlflow_dataset",
        lambda mlflow, rows, *, source, dataset_name: {
            "rows": list(rows),
            "source": str(source),
            "dataset_name": dataset_name,
        },
    )

    evaluate_rag._log_input_dataset_to_mlflow(
        tracking,
        {
            "dataset_path": str(dataset_path),
            "prepared_source": "generated",
        },
    )

    assert tracking.logged_inputs == [
        {
            "dataset": {
                "rows": [{"id": "1", "query": "Q1", "expected_answer": "A1"}],
                "source": str(dataset_path.resolve()),
                "dataset_name": "tickets.test",
            },
            "context": "testing",
            "tags": {
                "split": "test",
                "rows": 1,
                "dataset_path": str(dataset_path.resolve()),
                "prepared_source": "generated",
            },
        }
    ]


def test_resolve_prepared_rows_adds_manifest_stats(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    prepared_path = tmp_path / "prepared.json"

    monkeypatch.setattr(evaluate_rag, "build_container", lambda cfg: _DummyContainer(pipeline=object()))
    monkeypatch.setattr(
        evaluate_rag,
        "build_prepared_rows",
        _fake_build_prepared_rows,
    )

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=str(prepared_path),
        reuse_prepared=False,
        generation_workers=1,
    )

    rows, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1}},
        show_progress=False,
    )

    assert len(rows) == 1
    assert manifest["prepared_source"] == "generated"
    assert manifest["prep_total_rows"] == 1
    assert manifest["prep_success_rows"] == 1
    assert manifest["prep_failed_rows"] == 0
    assert "prep_elapsed_seconds" in manifest
    assert "prep_rate_rows_per_second" in manifest


def test_resolve_prepared_rows_passes_reranker_metadata_to_preparation(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    seen_reranker_profiles: list[object] = []
    seen_reranker_fingerprints: list[object] = []

    def _capture_build_prepared_rows(**kwargs):  # noqa: ANN003
        seen_reranker_profiles.append(kwargs.get("reranker_profile"))
        seen_reranker_fingerprints.append(kwargs.get("reranker_fingerprint"))
        return _fake_build_prepared_rows(**kwargs)

    monkeypatch.setattr(
        evaluate_rag,
        "build_container",
        lambda cfg: SimpleNamespace(
            pipeline=object(),
            retriever_source_settings={"docs": {"weight": 1.0}, "tickets": {"weight": 1.0}},
        ),
    )
    monkeypatch.setattr(evaluate_rag, "build_prepared_rows", _capture_build_prepared_rows)

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
        generation_mode="pipeline",
        generation_max_attempts=None,
        generation_retry_initial_backoff=None,
        generation_retry_max_backoff=None,
        generation_retry_jitter=None,
        generation_retry_on_empty_response=None,
        replay_failures_from=None,
    )

    cfg = _cfg_with_reranker(tmp_path, {"type": "rrf", "rrf_k": 60})

    _, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=cfg,
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1, "mode": "pipeline"}},
        show_progress=False,
    )

    assert seen_reranker_profiles
    assert seen_reranker_profiles[-1]["type"] == "rrf"
    assert seen_reranker_fingerprints[-1] == manifest["reranker_fingerprint"]
    assert manifest["reranker_profile"]["rrf_k"] == 60


def test_resolve_prepared_rows_rejects_reuse_when_reranker_fingerprint_differs(monkeypatch, tmp_path) -> None:
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(
        '[{"id":"row-1","user_input":"Q1","reference":"A1","response":"R1","retrieved_contexts":["ctx-1"],'
        '"retrieved_context_ids":["doc-1"],"metadata":{"reranker_fingerprint":"old-fingerprint"}}]',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_rag,
        "build_container",
        lambda cfg: SimpleNamespace(
            pipeline=object(),
            retriever_source_settings={"docs": {"weight": 1.0}, "tickets": {"weight": 1.0}},
        ),
    )

    args = Namespace(
        dataset_path=None,
        prepared_path=str(prepared_path),
        reuse_prepared=True,
        generation_workers=1,
        generation_mode="pipeline",
        generation_max_attempts=None,
        generation_retry_initial_backoff=None,
        generation_retry_max_backoff=None,
        generation_retry_jitter=None,
        generation_retry_on_empty_response=None,
        replay_failures_from=None,
    )

    cfg = _cfg_with_reranker(tmp_path, {"type": "rrf", "rrf_k": 60})

    with pytest.raises(ValueError, match="different reranker configuration"):
        evaluate_rag._resolve_prepared_rows(
            cfg=cfg,
            args=args,
            eval_cfg={"dataset": {"prepared_path": str(prepared_path)}, "generation": {"workers": 1, "mode": "pipeline"}},
            show_progress=False,
        )


def test_resolve_prepared_rows_joins_verified_annotations(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '[{"id":"1","summary":"Storage path question","query":"Q1","expected_answer":"A1","metadata":{"k":1}}]',
        encoding="utf-8",
    )
    annotations_path = tmp_path / "annotations.csv"
    with annotations_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
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
            }
        )

    captured_raw_examples: list[object] = []

    def _capture_build_prepared_rows(**kwargs):  # noqa: ANN003
        raw_examples = kwargs.get("raw_examples")
        captured_raw_examples.append(raw_examples)
        return [
            {
                "id": "row-1",
                "user_input": "Q1",
                "reference": "A1",
                "response": "R1",
                "retrieved_contexts": ["ctx-1"],
                "retrieved_context_ids": ["doc-1"],
                "metadata": dict(raw_examples[0]["metadata"]),
            }
        ]

    monkeypatch.setattr(evaluate_rag, "build_container", lambda cfg: _DummyContainer(pipeline=object()))
    monkeypatch.setattr(evaluate_rag, "build_prepared_rows", _capture_build_prepared_rows)

    args = Namespace(
        dataset_path=str(dataset_path),
        annotations_file=str(annotations_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
    )

    rows, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1}},
        show_progress=False,
    )

    assert captured_raw_examples
    assert captured_raw_examples[-1][0]["metadata"]["benchmark_annotation"]["source_needed"] == "docs"
    assert rows[0]["metadata"]["benchmark_annotation"]["query_type"] == "local_operational"
    assert manifest["annotations_path"] == str(annotations_path.resolve())
    assert manifest["annotation_rows"] == 1


def test_resolve_prepared_rows_passes_progress_callback_when_enabled(
    monkeypatch, tmp_path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    seen_callbacks: list[object] = []

    def _capture_build_prepared_rows(**kwargs):  # noqa: ANN003
        seen_callbacks.append(kwargs.get("progress_callback"))
        return _fake_build_prepared_rows(**kwargs)

    monkeypatch.setattr(evaluate_rag, "build_container", lambda cfg: _DummyContainer(pipeline=object()))
    monkeypatch.setattr(evaluate_rag, "build_prepared_rows", _capture_build_prepared_rows)

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
    )

    evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1}},
        show_progress=True,
    )

    assert seen_callbacks
    assert seen_callbacks[-1] is not None


def test_resolve_generation_retry_policy_prefers_cli_overrides() -> None:
    args = Namespace(
        generation_max_attempts=4,
        generation_retry_initial_backoff=0.5,
        generation_retry_max_backoff=3.0,
        generation_retry_jitter=0.1,
        generation_retry_on_empty_response=False,
    )

    policy = evaluate_rag._resolve_generation_retry_policy(
        {"retries": {"max_attempts": 2, "retry_on_empty_response": True}},
        args,
    )

    assert policy.max_attempts == 4
    assert policy.initial_backoff_seconds == 0.5
    assert policy.max_backoff_seconds == 3.0
    assert policy.jitter_seconds == 0.1
    assert policy.retry_on_empty_response is False


def test_resolve_prepared_rows_passes_retry_policy(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    seen_retry_policies: list[object] = []

    def _capture_build_prepared_rows(**kwargs):  # noqa: ANN003
        seen_retry_policies.append(kwargs.get("retry_policy"))
        return _fake_build_prepared_rows(**kwargs)

    monkeypatch.setattr(evaluate_rag, "build_container", lambda cfg: _DummyContainer(pipeline=object()))
    monkeypatch.setattr(evaluate_rag, "build_prepared_rows", _capture_build_prepared_rows)

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
        generation_max_attempts=3,
        generation_retry_initial_backoff=0.0,
        generation_retry_max_backoff=0.0,
        generation_retry_jitter=0.0,
        generation_retry_on_empty_response=True,
        generation_mode=None,
        query_api_url=None,
        query_api_timeout=None,
    )

    _, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1}},
        show_progress=False,
    )

    assert seen_retry_policies
    policy = seen_retry_policies[-1]
    assert policy is not None
    assert policy.max_attempts == 3
    assert manifest["generation_retries"]["max_attempts"] == 3


def test_resolve_prepared_rows_forces_official_api_retry_policy(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    seen_retry_policies: list[object] = []

    def _capture_build_prepared_rows_from_api(**kwargs):  # noqa: ANN003
        seen_retry_policies.append(kwargs.get("retry_policy"))
        return _fake_build_prepared_rows(**kwargs)

    monkeypatch.setattr(
        evaluate_rag,
        "build_prepared_rows_from_api",
        _capture_build_prepared_rows_from_api,
    )

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
        generation_max_attempts=4,
        generation_retry_initial_backoff=0.0,
        generation_retry_max_backoff=0.0,
        generation_retry_jitter=0.0,
        generation_retry_on_empty_response=True,
        generation_mode="api",
        query_api_url="http://127.0.0.1:8000/v1/query",
        query_api_timeout=30.0,
        evaluation_policy="official",
        replay_failures_from=None,
    )

    _, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1, "mode": "api"}},
        show_progress=False,
    )

    assert seen_retry_policies
    policy = seen_retry_policies[-1]
    assert policy.max_attempts == 1
    assert policy.retry_on_empty_response is False
    assert manifest["generation_retries"]["max_attempts"] == 1
    assert manifest["generation_retries"]["retry_on_empty_response"] is False


def test_resolve_prepared_rows_diagnostic_replays_only_failed_rows(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    replay_path = tmp_path / "prepared_rows.json"
    replay_path.write_text(
        '[{"id":"ok-1","user_input":"ok","reference":"A1","response":"R1","retrieved_contexts":[],"retrieved_context_ids":[],"metadata":{}},'
        '{"id":"bad-1","user_input":"bad","reference":"A2","response":"","retrieved_contexts":[],"retrieved_context_ids":[],"metadata":{"source_error":"TimeoutError: boom","original_metadata":{"ticket":"123"}}}]',
        encoding="utf-8",
    )
    seen_raw_examples: list[object] = []

    def _capture_build_prepared_rows_from_api(**kwargs):  # noqa: ANN003
        seen_raw_examples.append(kwargs.get("raw_examples"))
        return _fake_build_prepared_rows(**kwargs)

    monkeypatch.setattr(
        evaluate_rag,
        "build_prepared_rows_from_api",
        _capture_build_prepared_rows_from_api,
    )

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
        generation_max_attempts=None,
        generation_retry_initial_backoff=None,
        generation_retry_max_backoff=None,
        generation_retry_jitter=None,
        generation_retry_on_empty_response=None,
        generation_mode="api",
        query_api_url="http://127.0.0.1:8000/v1/query",
        query_api_timeout=30.0,
        evaluation_policy="diagnostic",
        replay_failures_from=str(replay_path),
    )

    _, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1, "mode": "api"}},
        show_progress=False,
    )

    assert seen_raw_examples
    assert seen_raw_examples[-1] == [
        {
            "id": "bad-1",
            "query": "bad",
            "expected_answer": "A2",
            "metadata": {"ticket": "123"},
        }
    ]
    assert manifest["prepared_source"] == "diagnostic_replay"
    assert manifest["replay_selected_rows"] == 1


def test_resolve_prepared_rows_diagnostic_replay_does_not_require_dataset_path(monkeypatch, tmp_path) -> None:
    replay_path = tmp_path / "prepared_rows.json"
    replay_path.write_text(
        '[{"id":"bad-1","user_input":"bad","reference":"A2","response":"","retrieved_contexts":[],"retrieved_context_ids":[],"metadata":{"source_error":"TimeoutError: boom"}}]',
        encoding="utf-8",
    )
    seen_raw_examples: list[object] = []

    def _capture_build_prepared_rows_from_api(**kwargs):  # noqa: ANN003
        seen_raw_examples.append(kwargs.get("raw_examples"))
        return _fake_build_prepared_rows(**kwargs)

    monkeypatch.setattr(
        evaluate_rag,
        "build_prepared_rows_from_api",
        _capture_build_prepared_rows_from_api,
    )

    args = Namespace(
        dataset_path=None,
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
        generation_max_attempts=None,
        generation_retry_initial_backoff=None,
        generation_retry_max_backoff=None,
        generation_retry_jitter=None,
        generation_retry_on_empty_response=None,
        generation_mode="api",
        query_api_url="http://127.0.0.1:8000/v1/query",
        query_api_timeout=30.0,
        evaluation_policy="diagnostic",
        replay_failures_from=str(replay_path),
    )

    _, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1, "mode": "api"}},
        show_progress=False,
    )

    assert seen_raw_examples == [[{"id": "bad-1", "query": "bad", "expected_answer": "A2", "metadata": {}}]]
    assert manifest["dataset_path"] is None
    assert manifest["prepared_source"] == "diagnostic_replay"


def test_resolve_prepared_rows_merges_extra_api_headers(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    seen_headers: list[object] = []

    def _capture_build_prepared_rows_from_api(**kwargs):  # noqa: ANN003
        seen_headers.append(kwargs.get("headers"))
        return [
            {
                "id": "row-1",
                "user_input": "Q1",
                "reference": "A1",
                "response": "R1",
                "retrieved_contexts": ["ctx-1"],
                "retrieved_context_ids": ["doc-1"],
                "metadata": {},
            }
        ]

    monkeypatch.setattr(
        evaluate_rag,
        "build_prepared_rows_from_api",
        _capture_build_prepared_rows_from_api,
    )

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
        generation_max_attempts=None,
        generation_retry_initial_backoff=None,
        generation_retry_max_backoff=None,
        generation_retry_jitter=None,
        generation_retry_on_empty_response=None,
        generation_mode="api",
        query_api_url="http://127.0.0.1:8000/v1/query",
        query_api_timeout=30.0,
    )

    _, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={
            "dataset": {},
            "generation": {"workers": 1, "api_headers": {"X-Base": "1"}},
        },
        show_progress=False,
        extra_api_headers={"X-Polaris-MLflow-Run-ID": "run-123"},
    )

    assert seen_headers
    assert seen_headers[-1] == {
        "X-Base": "1",
        "X-Polaris-MLflow-Run-ID": "run-123",
        "X-Polaris-Timeout-Ms": "29999",
        "X-Polaris-Eval-Policy": "official",
        POLARIS_EVAL_INCLUDE_METADATA_HEADER: "true",
    }
    assert "query_api_header_keys" in manifest
    assert manifest["query_api_timeout"] == 30.0
    assert manifest["server_timeout_ms"] == 29999


def test_resolve_prepared_rows_uses_stage_context_for_api_mode(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    seen_headers: list[object] = []
    seen_trace_factories: list[object] = []
    stage_context = _DummyStageContext()

    def _capture_build_prepared_rows_from_api(**kwargs):  # noqa: ANN003
        seen_headers.append(kwargs.get("headers"))
        seen_trace_factories.append(kwargs.get("trace_factory"))
        trace_factory = kwargs.get("trace_factory")
        assert trace_factory is not None
        with trace_factory("polaris.dataset_preparation.api_request", {"sample_id": "row-1"}, {"mode": "api"}):
            pass
        return [
            {
                "id": "row-1",
                "user_input": "Q1",
                "reference": "A1",
                "response": "R1",
                "retrieved_contexts": ["ctx-1"],
                "retrieved_context_ids": ["doc-1"],
                "metadata": {},
            }
        ]

    monkeypatch.setattr(
        evaluate_rag,
        "build_prepared_rows_from_api",
        _capture_build_prepared_rows_from_api,
    )

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
        generation_max_attempts=None,
        generation_retry_initial_backoff=None,
        generation_retry_max_backoff=None,
        generation_retry_jitter=None,
        generation_retry_on_empty_response=None,
        generation_mode="api",
        query_api_url="http://127.0.0.1:8000/v1/query",
        query_api_timeout=30.0,
    )

    evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={
            "dataset": {},
            "generation": {"workers": 1, "api_headers": {"X-Base": "1"}},
        },
        show_progress=False,
        stage_context=stage_context,
    )

    assert seen_headers[-1] == {
        "X-Base": "1",
        "X-Polaris-MLflow-Run-ID": "run-parent",
        "X-Polaris-MLflow-Child-Run-ID": "run-child",
        "X-Polaris-MLflow-Stage": "dataset_preparation",
        "X-Polaris-Timeout-Ms": "29999",
        "X-Polaris-Eval-Policy": "official",
        POLARIS_EVAL_INCLUDE_METADATA_HEADER: "true",
    }
    assert seen_trace_factories[-1] is not None
    assert stage_context.open_calls[-1]["include_child_trace"] is True


def test_resolve_prepared_rows_stamps_condition_metadata(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")

    monkeypatch.setattr(evaluate_rag, "build_container", lambda cfg: _DummyContainer(pipeline=object()))
    monkeypatch.setattr(evaluate_rag, "build_prepared_rows", _fake_build_prepared_rows)

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
        generation_mode="pipeline",
        generation_max_attempts=None,
        generation_retry_initial_backoff=None,
        generation_retry_max_backoff=None,
        generation_retry_jitter=None,
        generation_retry_on_empty_response=None,
        replay_failures_from=None,
    )

    preset_context = PresetContext(
        preset_name="docs_only",
        preset_description="Docs-only baseline with RRF reranking.",
        condition_summary={"sources": [{"name": "docs"}]},
        condition_fingerprint="condition-fp",
    )

    rows, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1, "mode": "pipeline"}},
        show_progress=False,
        preset_context=preset_context,
    )

    assert rows[0]["metadata"]["condition_fingerprint"] == "condition-fp"
    assert rows[0]["metadata"]["preset_name"] == "docs_only"
    assert manifest["condition_fingerprint"] == "condition-fp"
    assert manifest["preset_name"] == "docs_only"


def test_resolve_prepared_rows_rejects_reuse_when_condition_fingerprint_differs(monkeypatch, tmp_path) -> None:
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(
        '[{"id":"row-1","user_input":"Q1","reference":"A1","response":"R1","retrieved_contexts":["ctx-1"],'
        '"retrieved_context_ids":["doc-1"],"metadata":{"reranker_fingerprint":"fingerprint-123","condition_fingerprint":"old-condition"}}]',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_rag,
        "build_container",
        lambda cfg: SimpleNamespace(
            pipeline=object(),
            retriever_source_settings={"docs": {"weight": 1.0}, "tickets": {"weight": 1.0}},
        ),
    )
    monkeypatch.setattr(evaluate_rag, "_resolve_reranker_metadata", lambda cfg: ({"type": "rrf"}, "fingerprint-123"))

    args = Namespace(
        dataset_path=None,
        prepared_path=str(prepared_path),
        reuse_prepared=True,
        generation_workers=1,
        generation_mode="pipeline",
        generation_max_attempts=None,
        generation_retry_initial_backoff=None,
        generation_retry_max_backoff=None,
        generation_retry_jitter=None,
        generation_retry_on_empty_response=None,
        replay_failures_from=None,
    )

    cfg = _cfg_with_reranker(tmp_path, {"type": "rrf", "rrf_k": 60})
    preset_context = PresetContext(
        preset_name="docs_only",
        preset_description="Docs-only baseline with RRF reranking.",
        condition_summary={"sources": [{"name": "docs"}]},
        condition_fingerprint="new-condition",
    )

    with pytest.raises(ValueError, match="different evaluation condition"):
        evaluate_rag._resolve_prepared_rows(
            cfg=cfg,
            args=args,
            eval_cfg={"dataset": {"prepared_path": str(prepared_path)}, "generation": {"workers": 1, "mode": "pipeline"}},
            show_progress=False,
            preset_context=preset_context,
        )


def test_resolve_prepared_rows_uses_parent_only_trace_factory_for_pipeline_mode(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    seen_trace_factories: list[object] = []
    stage_context = _DummyStageContext()

    def _capture_build_prepared_rows(**kwargs):  # noqa: ANN003
        seen_trace_factories.append(kwargs.get("trace_factory"))
        trace_factory = kwargs.get("trace_factory")
        assert trace_factory is not None
        with trace_factory("polaris.dataset_preparation.pipeline_request", {"sample_id": "row-1"}, {"mode": "pipeline"}):
            pass
        return _fake_build_prepared_rows(**kwargs)

    monkeypatch.setattr(evaluate_rag, "build_container", lambda cfg: _DummyContainer(pipeline=object()))
    monkeypatch.setattr(evaluate_rag, "build_prepared_rows", _capture_build_prepared_rows)

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
        generation_max_attempts=None,
        generation_retry_initial_backoff=None,
        generation_retry_max_backoff=None,
        generation_retry_jitter=None,
        generation_retry_on_empty_response=None,
        generation_mode="pipeline",
        query_api_url=None,
        query_api_timeout=None,
    )

    evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1}},
        show_progress=False,
        stage_context=stage_context,
    )

    assert seen_trace_factories[-1] is not None
    assert stage_context.open_calls[-1]["include_child_trace"] is False


def test_prep_progress_renderer_logs_compact_last_error(caplog) -> None:
    renderer = evaluate_rag._PrepProgressRenderer(
        interactive=False,
        log_interval_seconds=1.0,
    )

    event = PrepProgressEvent(
        completed=1,
        total=2,
        successes=0,
        failures=1,
        elapsed_seconds=1.0,
        mode="api",
        last_error="TimeoutError: query API timed out after waiting far too long for the downstream generator",
    )

    with caplog.at_level(logging.INFO):
        renderer.update(event)

    assert "last_error=TimeoutError: query API timed out" in caplog.text


def test_resolve_evaluator_llm_tracing_prefers_cli_override() -> None:
    args = Namespace(trace_evaluator_llm=False)

    resolved = evaluate_rag._resolve_evaluator_llm_tracing(
        {"tracing": {"evaluator_llm": True}},
        args,
    )

    assert resolved is False


def test_main_passes_evaluator_trace_factory_when_enabled(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "eval-output"
    tracking = _DummyMainTracking(enabled=True, tracing_enabled=True)
    captured_from_global_config: dict[str, object] = {}
    captured_manifest: dict[str, object] = {}

    class _FakeEvaluator:
        def evaluate(self, **kwargs):  # noqa: ANN003
            trace_factory = captured_from_global_config.get("trace_factory")
            assert trace_factory is not None
            with trace_factory(  # type: ignore[misc]
                "polaris.ragas_evaluation.evaluator_llm",
                {"messages": [{"role": "user", "content": "judge this"}]},
                {"component": "evaluator_llm"},
            ):
                pass
            return SimpleNamespace(
                scores_df=_DummyScoresFrame(
                    [{"id": "row-1"}],
                    {"faithfulness": [0.5]},
                ),
                selected_metrics=["faithfulness"],
                skipped_metrics=[],
                selected_max_workers=4,
                failure_rate=0.0,
                duration_seconds=1.0,
            )

    class _FakeEvaluatorType:
        @staticmethod
        def from_global_config(cfg, **kwargs):  # noqa: ANN001, ANN003
            captured_from_global_config.update(kwargs)
            return _FakeEvaluator()

    def _fake_write_outputs(**kwargs):  # noqa: ANN003
        captured_manifest.update(kwargs.get("extra_manifest", {}))
        return {"summary_json": output_dir / "summary.json"}

    monkeypatch.setattr(
        evaluate_rag,
        "parse_args",
        lambda: Namespace(
            config_file=str(tmp_path / "config.yaml"),
            dataset_path=None,
            prepared_path=None,
            reuse_prepared=False,
            generation_workers=None,
            generation_max_attempts=None,
            generation_retry_initial_backoff=None,
            generation_retry_max_backoff=None,
            generation_retry_jitter=None,
            generation_retry_on_empty_response=None,
            generation_mode=None,
            evaluation_policy=None,
            replay_failures_from=None,
            query_api_url=None,
            query_api_timeout=None,
            metrics=None,
            output_dir=str(output_dir),
            no_tune_concurrency=False,
            no_progress=True,
            mlflow=None,
            mlflow_experiment=None,
            mlflow_run_name=None,
            trace_evaluator_llm=None,
        ),
    )
    monkeypatch.setattr(
        evaluate_rag.GlobalConfig,
        "load",
        lambda path: SimpleNamespace(
            raw={
                "evaluation": {
                    "output_dir": str(output_dir),
                    "tracing": {"evaluator_llm": True},
                },
                "mlflow": {"enabled": True, "tracing": {"enabled": True}},
            }
        ),
    )
    monkeypatch.setattr(evaluate_rag, "load_mlflow_runtime_config", lambda cfg: object())
    monkeypatch.setattr(evaluate_rag, "EvaluationTrackingContext", lambda *args, **kwargs: tracking)
    monkeypatch.setattr(
        evaluate_rag,
        "_resolve_prepared_rows",
        lambda **kwargs: (
            [
                {
                    "id": "row-1",
                    "user_input": "Q1",
                    "reference": "A1",
                    "response": "R1",
                    "retrieved_contexts": ["ctx-1"],
                    "retrieved_context_ids": ["doc-1"],
                    "metadata": {},
                }
            ],
            {"run_validity": "VALID", "prepared_source": "generated"},
        ),
    )
    monkeypatch.setattr(evaluate_rag, "to_evaluation_dataset", lambda rows: rows)
    monkeypatch.setattr(evaluate_rag, "build_environment_snapshot", lambda: {"python": "test"})
    monkeypatch.setitem(
        sys.modules,
        "polaris_rag.evaluation.evaluator",
        SimpleNamespace(
            Evaluator=_FakeEvaluatorType,
            write_outputs=_fake_write_outputs,
        ),
    )

    evaluate_rag.main()

    assert captured_from_global_config["trace_evaluator_llm"] is True
    assert captured_from_global_config["trace_factory"] is not None
    assert tracking.eval_stage.open_calls[-1]["include_child_trace"] is True
    assert any(
        batch.get("runtime.trace_evaluator_llm") == "True"
        for batch in tracking.logged_param_batches
    )
    assert captured_manifest["trace_evaluator_llm"] is True


def test_main_disables_evaluator_tracing_when_mlflow_tracing_is_off(monkeypatch, tmp_path, caplog) -> None:
    output_dir = tmp_path / "eval-output"
    tracking = _DummyMainTracking(enabled=True, tracing_enabled=False)
    captured_from_global_config: dict[str, object] = {}
    captured_manifest: dict[str, object] = {}

    class _FakeEvaluator:
        def evaluate(self, **kwargs):  # noqa: ANN003
            return SimpleNamespace(
                scores_df=_DummyScoresFrame(
                    [{"id": "row-1"}],
                    {"faithfulness": [0.5]},
                ),
                selected_metrics=["faithfulness"],
                skipped_metrics=[],
                selected_max_workers=4,
                failure_rate=0.0,
                duration_seconds=1.0,
            )

    class _FakeEvaluatorType:
        @staticmethod
        def from_global_config(cfg, **kwargs):  # noqa: ANN001, ANN003
            captured_from_global_config.update(kwargs)
            return _FakeEvaluator()

    def _fake_write_outputs(**kwargs):  # noqa: ANN003
        captured_manifest.update(kwargs.get("extra_manifest", {}))
        return {"summary_json": output_dir / "summary.json"}

    monkeypatch.setattr(
        evaluate_rag,
        "parse_args",
        lambda: Namespace(
            config_file=str(tmp_path / "config.yaml"),
            dataset_path=None,
            prepared_path=None,
            reuse_prepared=False,
            generation_workers=None,
            generation_max_attempts=None,
            generation_retry_initial_backoff=None,
            generation_retry_max_backoff=None,
            generation_retry_jitter=None,
            generation_retry_on_empty_response=None,
            generation_mode=None,
            evaluation_policy=None,
            replay_failures_from=None,
            query_api_url=None,
            query_api_timeout=None,
            metrics=None,
            output_dir=str(output_dir),
            no_tune_concurrency=False,
            no_progress=True,
            mlflow=None,
            mlflow_experiment=None,
            mlflow_run_name=None,
            trace_evaluator_llm=True,
        ),
    )
    monkeypatch.setattr(
        evaluate_rag.GlobalConfig,
        "load",
        lambda path: SimpleNamespace(
            raw={
                "evaluation": {
                    "output_dir": str(output_dir),
                    "tracing": {"evaluator_llm": True},
                },
                "mlflow": {"enabled": True, "tracing": {"enabled": False}},
            }
        ),
    )
    monkeypatch.setattr(evaluate_rag, "load_mlflow_runtime_config", lambda cfg: object())
    monkeypatch.setattr(evaluate_rag, "EvaluationTrackingContext", lambda *args, **kwargs: tracking)
    monkeypatch.setattr(
        evaluate_rag,
        "_resolve_prepared_rows",
        lambda **kwargs: (
            [
                {
                    "id": "row-1",
                    "user_input": "Q1",
                    "reference": "A1",
                    "response": "R1",
                    "retrieved_contexts": ["ctx-1"],
                    "retrieved_context_ids": ["doc-1"],
                    "metadata": {},
                }
            ],
            {"run_validity": "VALID", "prepared_source": "generated"},
        ),
    )
    monkeypatch.setattr(evaluate_rag, "to_evaluation_dataset", lambda rows: rows)
    monkeypatch.setattr(evaluate_rag, "build_environment_snapshot", lambda: {"python": "test"})
    monkeypatch.setitem(
        sys.modules,
        "polaris_rag.evaluation.evaluator",
        SimpleNamespace(
            Evaluator=_FakeEvaluatorType,
            write_outputs=_fake_write_outputs,
        ),
    )

    with caplog.at_level(logging.WARNING):
        evaluate_rag.main()

    assert captured_from_global_config["trace_evaluator_llm"] is False
    assert captured_from_global_config["trace_factory"] is None
    assert any(
        batch.get("runtime.trace_evaluator_llm") == "False"
        for batch in tracking.logged_param_batches
    )
    assert captured_manifest["trace_evaluator_llm"] is False
    assert "Evaluator LLM tracing was requested but MLflow tracing is disabled" in caplog.text


def test_main_prepare_only_skips_ragas_evaluation(monkeypatch, tmp_path, capsys) -> None:
    output_dir = tmp_path / "prep-output"
    tracking = _DummyMainTracking(enabled=True, tracing_enabled=True)

    monkeypatch.setattr(
        evaluate_rag,
        "parse_args",
        lambda: Namespace(
            config_file=str(tmp_path / "config.yaml"),
            dataset_path=None,
            prepared_path=None,
            reuse_prepared=False,
            prepare_only=True,
            generation_workers=None,
            generation_max_attempts=None,
            generation_retry_initial_backoff=None,
            generation_retry_max_backoff=None,
            generation_retry_jitter=None,
            generation_retry_on_empty_response=None,
            generation_mode=None,
            evaluation_policy=None,
            replay_failures_from=None,
            query_api_url=None,
            query_api_timeout=None,
            metrics=None,
            output_dir=str(output_dir),
            no_tune_concurrency=False,
            no_progress=True,
            mlflow=None,
            mlflow_experiment=None,
            mlflow_run_name=None,
            trace_evaluator_llm=True,
        ),
    )
    monkeypatch.setattr(
        evaluate_rag.GlobalConfig,
        "load",
        lambda path: SimpleNamespace(
            raw={
                "evaluation": {
                    "output_dir": str(output_dir),
                    "tracing": {"evaluator_llm": True},
                },
                "mlflow": {"enabled": True, "tracing": {"enabled": True}},
            }
        ),
    )
    monkeypatch.setattr(evaluate_rag, "load_mlflow_runtime_config", lambda cfg: object())
    monkeypatch.setattr(evaluate_rag, "EvaluationTrackingContext", lambda *args, **kwargs: tracking)
    monkeypatch.setattr(
        evaluate_rag,
        "_resolve_prepared_rows",
        lambda **kwargs: (
            [
                {
                    "id": "row-1",
                    "user_input": "Q1",
                    "reference": "A1",
                    "response": "R1",
                    "retrieved_contexts": ["ctx-1"],
                    "retrieved_context_ids": ["doc-1"],
                    "metadata": {},
                }
            ],
            {"run_validity": "VALID", "prepared_source": "generated"},
        ),
    )
    monkeypatch.setattr(
        evaluate_rag,
        "to_evaluation_dataset",
        lambda rows: (_ for _ in ()).throw(AssertionError("prepare-only should not score rows")),
    )
    monkeypatch.setattr(evaluate_rag, "build_environment_snapshot", lambda: {"python": "test"})

    evaluate_rag.main()

    captured = capsys.readouterr()
    prepared_rows_path = output_dir / "prepared_rows.json"

    assert "Dataset preparation complete." in captured.out
    assert "Prep run validity: VALID" in captured.out
    assert prepared_rows_path.exists()
    assert "\"id\": \"row-1\"" in prepared_rows_path.read_text(encoding="utf-8")
    assert tracking.eval_stage.open_calls == []
    assert tracking.eval_stage.active_open_calls == []
    assert any(
        batch.get("runtime.prepare_only") == "True"
        for batch in tracking.logged_param_batches
    )
    assert any(
        artifact["output_path"].endswith("dataset_manifest.json")
        for artifact in tracking.logged_json_artifacts
    )
    assert any(
        artifact["output_path"].endswith("env_snapshot.json")
        for artifact in tracking.logged_json_artifacts
    )
    assert any(
        artifact["output_path"].endswith("config_snapshot.json")
        for artifact in tracking.logged_json_artifacts
    )
