from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass

from polaris_rag.cli import evaluate_rag
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
    def __init__(self) -> None:
        self.open_calls: list[dict[str, object]] = []

    def correlation_headers(self) -> dict[str, str]:
        return {
            "X-Polaris-MLflow-Run-ID": "run-parent",
            "X-Polaris-MLflow-Child-Run-ID": "run-child",
            "X-Polaris-MLflow-Stage": "dataset_preparation",
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
    }
    assert "query_api_header_keys" in manifest


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
    }
    assert seen_trace_factories[-1] is not None
    assert stage_context.open_calls[-1]["include_child_trace"] is True


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
