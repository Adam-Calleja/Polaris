import json
import sys
from contextlib import contextmanager

from polaris_rag.cli import create_dev_test_sets
from polaris_rag.evaluation.evaluation_dataset import load_raw_examples
from polaris_rag.observability import mlflow_tracking


class _FakeRunInfo:
    def __init__(self, run_id: str):
        self.run_id = run_id


class _FakeRun:
    def __init__(self, run_id: str):
        self.info = _FakeRunInfo(run_id)


class _FakeDatasetAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def from_pandas(self, frame, source: str, name: str):  # noqa: ANN001
        dataset = {"frame": frame, "source": source, "name": name}
        self.calls.append(dataset)
        return dataset


class _FakeMLflow:
    def __init__(self) -> None:
        self.data = _FakeDatasetAPI()
        self.set_tracking_uri_calls: list[str] = []
        self.set_experiment_calls: list[str] = []
        self.start_run_calls: list[dict[str, object]] = []
        self.log_params_calls: list[dict[str, str]] = []
        self.log_artifact_calls: list[tuple[str, str | None]] = []
        self.log_input_calls: list[dict[str, object]] = []

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
        yield _FakeRun("run-1")

    def log_params(self, params: dict[str, str]) -> None:
        self.log_params_calls.append(dict(params))

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


class _FakeSpan:
    trace_id = "trace-1"

    def set_inputs(self, inputs):  # noqa: ANN001
        return None

    def set_outputs(self, outputs):  # noqa: ANN001
        return None

    def set_attributes(self, attributes):  # noqa: ANN001
        return None

    def set_attribute(self, key, value):  # noqa: ANN001
        return None

    def end(self) -> None:
        return None


@contextmanager
def _fake_span_context(*args, **kwargs):  # noqa: ANN002, ANN003
    yield _FakeSpan()


class _FakePandasDataFrame:
    @staticmethod
    def from_records(rows):  # noqa: ANN001, ANN205
        return list(rows)


class _FakePandas:
    DataFrame = _FakePandasDataFrame


def test_main_splits_dataset_from_test_sample_file(tmp_path, monkeypatch, capsys) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "ex-1", "query": "Q1", "expected_answer": "A1"},
                {"id": "ex-2", "query": "Q2", "expected_answer": "A2"},
                {"id": "ex-3", "query": "Q3", "expected_answer": "A3"},
            ]
        ),
        encoding="utf-8",
    )
    sample_ids_path = tmp_path / "test_ids.txt"
    sample_ids_path.write_text("ex-3\nex-1\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_dev_test_sets.py",
            "--dataset-file",
            str(dataset_path),
            "--test-samples-file",
            str(sample_ids_path),
        ],
    )

    create_dev_test_sets.main()

    dev_output_path = tmp_path / "dataset.dev.jsonl"
    test_output_path = tmp_path / "dataset.test.jsonl"

    dev_rows = load_raw_examples(dev_output_path)
    test_rows = load_raw_examples(test_output_path)

    assert [row["id"] for row in dev_rows] == ["ex-2"]
    assert [row["id"] for row in test_rows] == ["ex-3", "ex-1"]

    captured = capsys.readouterr()
    assert "Dataset split complete." in captured.out


def test_main_logs_split_datasets_to_mlflow(tmp_path, monkeypatch, capsys) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "ex-1", "query": "Q1", "expected_answer": "A1"},
                {"id": "ex-2", "query": "Q2", "expected_answer": "A2"},
                {"id": "ex-3", "query": "Q3", "expected_answer": "A3"},
            ]
        ),
        encoding="utf-8",
    )
    sample_ids_path = tmp_path / "test_ids.txt"
    sample_ids_path.write_text("ex-2\n", encoding="utf-8")

    fake_mlflow = _FakeMLflow()
    monkeypatch.setattr(mlflow_tracking, "_import_mlflow", lambda: fake_mlflow)
    monkeypatch.setattr(mlflow_tracking, "start_detached_span", _fake_span_context)
    monkeypatch.setattr(mlflow_tracking, "start_span", _fake_span_context)
    monkeypatch.setattr(create_dev_test_sets, "_import_pandas", lambda: _FakePandas)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_dev_test_sets.py",
            "--dataset-file",
            str(dataset_path),
            "--test-samples-file",
            str(sample_ids_path),
            "--mlflow",
            "--tracking-uri",
            "http://mlflow:5000",
            "--mlflow-experiment",
            "dataset-splits",
            "--mlflow-run-name",
            "split-run",
        ],
    )

    create_dev_test_sets.main()

    assert fake_mlflow.set_tracking_uri_calls == ["http://mlflow:5000"]
    assert fake_mlflow.set_experiment_calls == ["dataset-splits"]
    assert fake_mlflow.start_run_calls[0]["run_name"] == "split-run"
    assert [call["context"] for call in fake_mlflow.log_input_calls] == ["validation", "testing"]
    assert fake_mlflow.log_input_calls[0]["tags"]["split"] == "dev"
    assert fake_mlflow.log_input_calls[1]["tags"]["split"] == "test"

    captured = capsys.readouterr()
    assert "MLflow run id: run-1" in captured.out


def test_main_supports_stratified_category_split(tmp_path, monkeypatch, capsys) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": "a-1", "query": "Q1", "expected_answer": "A1"},
                {"id": "a-2", "query": "Q2", "expected_answer": "A2"},
                {"id": "b-1", "query": "Q3", "expected_answer": "A3"},
                {"id": "b-2", "query": "Q4", "expected_answer": "A4"},
                {"id": "solo-1", "query": "Q5", "expected_answer": "A5"},
            ]
        ),
        encoding="utf-8",
    )
    categories_path = tmp_path / "categories.json"
    categories_path.write_text(
        json.dumps(
            {
                "cat-a": ["a-1", "a-2"],
                "cat-b": ["b-1", "b-2"],
                "cat-solo": ["solo-1"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        create_dev_test_sets,
        "stratified_split_raw_examples_by_categories",
        lambda raw_examples, categories, *, test_size, random_state, id_field: (
            [raw_examples[1], raw_examples[3], raw_examples[4]],
            [raw_examples[0], raw_examples[2]],
            {
                "test_ids": ["a-1", "b-1"],
                "category_test_counts": {"cat-a": 1, "cat-b": 1, "cat-solo": 0},
            },
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_dev_test_sets.py",
            "--dataset-file",
            str(dataset_path),
            "--categories-file",
            str(categories_path),
            "--test-size",
            "2",
        ],
    )

    create_dev_test_sets.main()

    dev_output_path = tmp_path / "dataset.dev.jsonl"
    test_output_path = tmp_path / "dataset.test.jsonl"

    dev_rows = load_raw_examples(dev_output_path)
    test_rows = load_raw_examples(test_output_path)

    assert [row["id"] for row in dev_rows] == ["a-2", "b-2", "solo-1"]
    assert [row["id"] for row in test_rows] == ["a-1", "b-1"]

    captured = capsys.readouterr()
    assert "Test ids: ['a-1', 'b-1']" in captured.out
    assert "cat-solo: 0 test" in captured.out
