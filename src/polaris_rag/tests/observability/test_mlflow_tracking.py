from __future__ import annotations

from contextlib import contextmanager

from polaris_rag.observability import mlflow_tracking


class _FakeRunInfo:
    def __init__(self, run_id: str):
        self.run_id = run_id


class _FakeRun:
    def __init__(self, run_id: str):
        self.info = _FakeRunInfo(run_id)


class _FakeMLflow:
    def __init__(self) -> None:
        self.set_tracking_uri_calls: list[str] = []
        self.set_experiment_calls: list[str] = []
        self.start_run_calls: list[dict[str, object]] = []
        self.log_params_calls: list[dict[str, str]] = []
        self.log_metrics_calls: list[dict[str, float]] = []
        self.log_artifact_calls: list[tuple[str, str | None]] = []
        self._next_run_id = 1

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
