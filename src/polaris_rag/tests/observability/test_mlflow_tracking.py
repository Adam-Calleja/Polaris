from __future__ import annotations

from contextlib import contextmanager

from polaris_rag.observability import mlflow_tracking


class _FakeRunInfo:
    def __init__(self, run_id: str):
        self.run_id = run_id


class _FakeRun:
    def __init__(self, run_id: str):
        self.info = _FakeRunInfo(run_id)


class _FakeSpan:
    def __init__(
        self,
        name: str,
        *,
        parent_span: "_FakeSpan | None" = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.parent_span = parent_span
        self.tags = dict(tags or {})
        self.inputs = None
        self.outputs = None
        self.attributes: dict[str, object] = {}
        self.trace_id = ""
        self.ended = False

    def set_inputs(self, inputs):  # noqa: ANN001
        self.inputs = inputs

    def set_outputs(self, outputs):  # noqa: ANN001
        self.outputs = outputs

    def set_attributes(self, attributes):  # noqa: ANN001
        self.attributes.update(dict(attributes))

    def set_attribute(self, key, value):  # noqa: ANN001
        self.attributes[key] = value

    def end(self) -> None:
        self.ended = True


class _FakeMLflow:
    def __init__(self) -> None:
        self.set_tracking_uri_calls: list[str] = []
        self.set_experiment_calls: list[str] = []
        self.start_run_calls: list[dict[str, object]] = []
        self.log_params_calls: list[dict[str, str]] = []
        self.log_metrics_calls: list[dict[str, float]] = []
        self.log_artifact_calls: list[tuple[str, str | None]] = []
        self.log_input_calls: list[dict[str, object]] = []
        self.start_span_calls: list[dict[str, object]] = []
        self.start_span_no_context_calls: list[dict[str, object]] = []
        self.update_current_trace_calls: list[dict[str, str]] = []
        self.associate_trace_calls: list[tuple[str, str]] = []
        self._next_run_id = 1
        self._next_trace_id = 1

    def set_tracking_uri(self, uri: str) -> None:
        self.set_tracking_uri_calls.append(uri)

    def set_experiment(self, experiment_name: str):
        self.set_experiment_calls.append(experiment_name)
        return type("_Exp", (), {"experiment_id": "1"})

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
        log_system_metrics: bool = False,
    ):
        self.start_run_calls.append(
            {
                "run_name": run_name,
                "tags": dict(tags or {}),
                "nested": nested,
                "log_system_metrics": log_system_metrics,
            }
        )
        run = _FakeRun(run_id=f"run-{self._next_run_id}")
        self._next_run_id += 1
        yield run

    def log_params(self, params: dict[str, str]) -> None:
        self.log_params_calls.append(dict(params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.log_metrics_calls.append(dict(metrics))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.log_artifact_calls.append((local_path, artifact_path))

    def log_input(self, dataset, context: str | None = None, tags: dict[str, str] | None = None) -> None:  # noqa: ANN001
        self.log_input_calls.append(
            {
                "dataset": dataset,
                "context": context,
                "tags": dict(tags or {}),
            }
        )

    @contextmanager
    def start_span(self, name: str):
        span = _FakeSpan(name)
        span.trace_id = f"trace-{self._next_trace_id}"
        self._next_trace_id += 1
        self.start_span_calls.append({"name": name})
        yield span
        span.end()

    def start_span_no_context(
        self,
        name: str,
        parent_span: _FakeSpan | None = None,
        tags: dict[str, str] | None = None,
    ) -> _FakeSpan:
        span = _FakeSpan(name, parent_span=parent_span, tags=tags)
        span.trace_id = getattr(parent_span, "trace_id", "") or f"trace-{self._next_trace_id}"
        if not getattr(parent_span, "trace_id", None):
            self._next_trace_id += 1
        self.start_span_no_context_calls.append(
            {
                "name": name,
                "parent": getattr(parent_span, "name", None),
                "tags": dict(tags or {}),
                "trace_id": span.trace_id,
            }
        )
        return span

    def update_current_trace(self, tags: dict[str, str] | None = None) -> None:
        self.update_current_trace_calls.append(dict(tags or {}))

    def _associate_trace_with_run(self, trace_id: str, run_id: str) -> None:
        self.associate_trace_calls.append((trace_id, run_id))


def test_load_mlflow_runtime_config_defaults() -> None:
    cfg = {"foo": "bar"}
    runtime_cfg = mlflow_tracking.load_mlflow_runtime_config(cfg)

    assert runtime_cfg.enabled is False
    assert runtime_cfg.tracking_uri is None
    assert runtime_cfg.experiment_name == "polaris-rag"
    assert runtime_cfg.tracing.enabled is False
    assert runtime_cfg.prompt_registry.enabled is False
    assert runtime_cfg.prompt_registry.alias == "prod"


def test_load_mlflow_runtime_config_reads_nested_sections() -> None:
    cfg = {
        "mlflow": {
            "enabled": True,
            "tracking_uri": "http://mlflow:5000",
            "experiment_name": "evals",
            "tags": {"project": "polaris"},
            "tracing": {
                "enabled": True,
                "destination_experiment": "traces",
            },
            "prompt_registry": {
                "enabled": True,
                "name": "hpc_prompt",
                "alias": "prod",
            },
        }
    }

    runtime_cfg = mlflow_tracking.load_mlflow_runtime_config(cfg)

    assert runtime_cfg.enabled is True
    assert runtime_cfg.tracking_uri == "http://mlflow:5000"
    assert runtime_cfg.experiment_name == "evals"
    assert runtime_cfg.tags == {"project": "polaris"}
    assert runtime_cfg.tracing.enabled is True
    assert runtime_cfg.tracing.destination_experiment == "traces"
    assert runtime_cfg.prompt_registry.enabled is True
    assert runtime_cfg.prompt_registry.name == "hpc_prompt"


def test_evaluation_tracking_context_logs_nested_runs(monkeypatch, tmp_path) -> None:
    fake_mlflow = _FakeMLflow()
    monkeypatch.setattr(mlflow_tracking, "_import_mlflow", lambda: fake_mlflow)

    runtime_cfg = mlflow_tracking.MLflowRuntimeConfig(
        enabled=True,
        tracking_uri="http://mlflow:5000",
        experiment_name="polaris-evals",
        tags={"project": "polaris"},
        tracing=mlflow_tracking.TraceRuntimeConfig(enabled=False),
    )
    tracker = mlflow_tracking.EvaluationTrackingContext(runtime_cfg)

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")

    with tracker.open(run_name="eval-run"):
        assert tracker.run_id == "run-1"
        assert tracker.trace_headers()[mlflow_tracking.TRACE_PARENT_RUN_HEADER] == "run-1"

        tracker.log_params({"a": 1, "b": True})
        tracker.log_metrics({"m1": 1.0, "m2": 2})
        tracker.log_artifact(artifact, artifact_path="outputs")

        with tracker.stage("dataset_preparation"):
            pass

    assert fake_mlflow.set_tracking_uri_calls == ["http://mlflow:5000"]
    assert fake_mlflow.set_experiment_calls == ["polaris-evals"]
    assert len(fake_mlflow.start_run_calls) == 2
    assert fake_mlflow.start_run_calls[0]["run_name"] == "eval-run"
    assert fake_mlflow.start_run_calls[0]["nested"] is False
    assert fake_mlflow.start_run_calls[1]["run_name"] == "stage:dataset_preparation"
    assert fake_mlflow.start_run_calls[1]["nested"] is True
    assert fake_mlflow.log_params_calls
    assert fake_mlflow.log_metrics_calls
    assert fake_mlflow.log_artifact_calls


def test_evaluation_tracking_context_logs_dataset_inputs(monkeypatch) -> None:
    fake_mlflow = _FakeMLflow()
    monkeypatch.setattr(mlflow_tracking, "_import_mlflow", lambda: fake_mlflow)

    runtime_cfg = mlflow_tracking.MLflowRuntimeConfig(
        enabled=True,
        tracking_uri="http://mlflow:5000",
        experiment_name="polaris-evals",
        tracing=mlflow_tracking.TraceRuntimeConfig(enabled=False),
    )
    tracker = mlflow_tracking.EvaluationTrackingContext(runtime_cfg)

    with tracker.open(run_name="dataset-run"):
        tracker.log_input(
            {"name": "dev-dataset"},
            context="validation",
            tags={"split": "dev"},
        )

    assert fake_mlflow.log_input_calls == [
        {
            "dataset": {"name": "dev-dataset"},
            "context": "validation",
            "tags": {"split": "dev"},
        }
    ]


def test_evaluation_tracking_context_emits_parent_and_stage_trace_context(monkeypatch) -> None:
    fake_mlflow = _FakeMLflow()
    monkeypatch.setattr(mlflow_tracking, "_import_mlflow", lambda: fake_mlflow)

    runtime_cfg = mlflow_tracking.MLflowRuntimeConfig(
        enabled=True,
        tracking_uri="http://mlflow:5000",
        experiment_name="polaris-evals",
        tags={"project": "polaris"},
        tracing=mlflow_tracking.TraceRuntimeConfig(enabled=True, destination_experiment="polaris-rag-traces"),
    )
    tracker = mlflow_tracking.EvaluationTrackingContext(runtime_cfg)

    with tracker.open(run_name="eval-run"):
        with tracker.stage("dataset_preparation") as stage_ctx:
            assert stage_ctx.parent_run_id == "run-1"
            assert stage_ctx.child_run_id == "run-2"
            assert stage_ctx.correlation_headers() == {
                mlflow_tracking.TRACE_PARENT_RUN_HEADER: "run-1",
                mlflow_tracking.TRACE_CHILD_RUN_HEADER: "run-2",
                mlflow_tracking.TRACE_STAGE_HEADER: "dataset_preparation",
            }

    assert fake_mlflow.associate_trace_calls == [("trace-1", "run-1")]
    assert fake_mlflow.set_experiment_calls == ["polaris-evals"]
    assert fake_mlflow.start_span_calls[0]["name"] == "polaris.dataset_preparation"
    assert fake_mlflow.start_span_no_context_calls[0]["name"] == "polaris.eval.run"
    assert fake_mlflow.start_span_no_context_calls[1]["name"] == "polaris.dataset_preparation"
