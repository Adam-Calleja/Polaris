import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

from openai import AsyncOpenAI
import pytest

pytest.importorskip("ragas")

from polaris_rag.evaluation import evaluator


class _RecordedTrace:
    def __init__(self, name, inputs, attributes=None):  # noqa: ANN001
        self.name = name
        self.inputs = dict(inputs)
        self.attributes = dict(attributes or {})
        self.outputs = None

    def set_outputs(self, outputs):  # noqa: ANN001
        self.outputs = outputs

    def set_attributes(self, attributes):  # noqa: ANN001
        self.attributes.update(dict(attributes))


class _TraceFactory:
    def __init__(self) -> None:
        self.calls: list[_RecordedTrace] = []

    @contextmanager
    def __call__(self, name, inputs, attributes=None):  # noqa: ANN001
        trace = _RecordedTrace(name, inputs, attributes)
        self.calls.append(trace)
        yield trace


def test_traced_client_preserves_async_openai_instance() -> None:
    trace_factory = _TraceFactory()
    client = AsyncOpenAI(
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        max_retries=0,
    )

    traced_client = evaluator._TracedAsyncOpenAIClient(client, trace_factory=trace_factory)

    assert traced_client is client
    assert isinstance(traced_client, AsyncOpenAI)


def test_traced_client_logs_prompt_response_and_metric_context() -> None:
    trace_factory = _TraceFactory()

    class _FakeResponse:
        def model_dump(self) -> dict[str, str]:
            return {"id": "resp-1", "output_text": "supported"}

    class _FakeCompletions:
        async def create(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return _FakeResponse()

    traced_client = evaluator._TracedAsyncOpenAIClient(
        SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions())),
        trace_factory=trace_factory,
    )

    class _FakeMetric:
        async def ascore(self, **kwargs):  # noqa: ANN003
            await traced_client.chat.completions.create(
                model="judge-model",
                messages=[{"role": "user", "content": kwargs["response"]}],
                response_model="JudgeResponse",
                temperature=0,
            )
            return SimpleNamespace(value=0.75)

    assert trace_factory.calls == []

    result = asyncio.run(
        evaluator._score_metric(
            _FakeMetric(),
            {"response": "Check this answer"},
            evaluator.EvaluatorMetricTraceContext(
                metric_name="faithfulness",
                sample_id="sample-1",
                row_index=0,
                required_columns=("response",),
            ),
        )
    )

    assert result == 0.75
    assert len(trace_factory.calls) == 1
    trace = trace_factory.calls[0]
    assert trace.name == "polaris.ragas_evaluation.evaluator_llm"
    assert trace.inputs["model"] == "judge-model"
    assert trace.inputs["messages"] == [{"role": "user", "content": "Check this answer"}]
    assert trace.inputs["response_model"] == "JudgeResponse"
    assert trace.inputs["metric_context"] == {
        "metric_name": "faithfulness",
        "sample_id": "sample-1",
        "row_index": 0,
        "required_columns": ["response"],
    }
    assert trace.attributes["component"] == "evaluator_llm"
    assert trace.attributes["metric_name"] == "faithfulness"
    assert trace.attributes["sample_id"] == "sample-1"
    assert trace.attributes["row_index"] == 0
    assert trace.attributes["status"] == "success"
    assert trace.outputs == {"response": {"id": "resp-1", "output_text": "supported"}}


def test_traced_client_logs_errors_and_reraises() -> None:
    trace_factory = _TraceFactory()

    class _FailingCompletions:
        async def create(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("judge exploded")

    traced_client = evaluator._TracedAsyncOpenAIClient(
        SimpleNamespace(chat=SimpleNamespace(completions=_FailingCompletions())),
        trace_factory=trace_factory,
    )

    class _FakeMetric:
        async def ascore(self, **kwargs):  # noqa: ANN003
            await traced_client.chat.completions.create(
                model="judge-model",
                messages=[{"role": "user", "content": kwargs["response"]}],
            )
            return SimpleNamespace(value=1.0)

    with pytest.raises(RuntimeError, match="judge exploded"):
        asyncio.run(
            evaluator._score_metric(
                _FakeMetric(),
                {"response": "Check this answer"},
                evaluator.EvaluatorMetricTraceContext(
                    metric_name="answer_relevancy",
                    sample_id="sample-err",
                    row_index=2,
                    required_columns=("response",),
                ),
            )
        )

    assert len(trace_factory.calls) == 1
    trace = trace_factory.calls[0]
    assert trace.attributes["status"] == "error"
    assert trace.outputs == {"error": "RuntimeError: judge exploded"}


def test_resolve_runtime_metrics_skips_embeddings_for_llm_only_metrics(monkeypatch) -> None:
    built: list[str] = []
    captured: dict[str, object] = {}

    def _fake_build_llm(self, **kwargs):  # noqa: ANN001, ANN003
        built.append("llm")
        return "fake-llm"

    def _fake_build_embeddings(self, **kwargs):  # noqa: ANN001, ANN003
        built.append("embeddings")
        return "fake-embeddings"

    def _fake_instantiate_metrics(specs, *, llm, embeddings, metric_config=None):  # noqa: ANN001, ANN003
        captured["metric_names"] = [spec.name for spec in specs]
        captured["llm"] = llm
        captured["embeddings"] = embeddings
        return [object() for _ in specs]

    class _FakeDataset:
        def features(self) -> set[str]:
            return {"user_input", "response", "retrieved_contexts"}

    monkeypatch.setattr(evaluator.Evaluator, "_build_llm", _fake_build_llm)
    monkeypatch.setattr(evaluator.Evaluator, "_build_embeddings", _fake_build_embeddings)
    monkeypatch.setattr(evaluator, "instantiate_metrics", _fake_instantiate_metrics)

    instance = evaluator.Evaluator(
        llm_model="judge-model",
        llm_api_base="http://localhost:8080/v1",
        llm_api_key=None,
        llm_kwargs={},
        embedding_model="embed-model",
        embedding_api_base="http://localhost:8081/v1",
        embedding_api_key=None,
        embedding_kwargs={},
        requested_metrics=["faithfulness"],
    )

    metrics, skipped = instance._resolve_runtime_metrics(dataset=_FakeDataset())

    assert [spec.name for spec, _ in metrics] == ["faithfulness"]
    assert skipped == []
    assert built == ["llm"]
    assert captured["metric_names"] == ["faithfulness"]
    assert captured["llm"] == "fake-llm"
    assert captured["embeddings"] is None
    assert instance.llm == "fake-llm"
    assert instance.embeddings is None


def test_resolve_runtime_metrics_skips_llm_and_embeddings_for_retrieval_metrics(monkeypatch) -> None:
    built: list[str] = []
    captured: dict[str, object] = {}

    def _fake_build_llm(self, **kwargs):  # noqa: ANN001, ANN003
        built.append("llm")
        return "fake-llm"

    def _fake_build_embeddings(self, **kwargs):  # noqa: ANN001, ANN003
        built.append("embeddings")
        return "fake-embeddings"

    def _fake_instantiate_metrics(specs, *, llm, embeddings, metric_config=None):  # noqa: ANN001, ANN003
        captured["metric_names"] = [spec.name for spec in specs]
        captured["llm"] = llm
        captured["embeddings"] = embeddings
        return [object() for _ in specs]

    class _FakeDataset:
        def features(self) -> set[str]:
            return {"retrieved_context_ids", "reference_context_ids"}

    monkeypatch.setattr(evaluator.Evaluator, "_build_llm", _fake_build_llm)
    monkeypatch.setattr(evaluator.Evaluator, "_build_embeddings", _fake_build_embeddings)
    monkeypatch.setattr(evaluator, "instantiate_metrics", _fake_instantiate_metrics)

    instance = evaluator.Evaluator(
        llm_model="judge-model",
        llm_api_base="http://localhost:8080/v1",
        llm_api_key=None,
        llm_kwargs={},
        embedding_model="embed-model",
        embedding_api_base="http://localhost:8081/v1",
        embedding_api_key=None,
        embedding_kwargs={},
        requested_metrics=["retrieval_recall_at_k"],
    )

    metrics, skipped = instance._resolve_runtime_metrics(dataset=_FakeDataset())

    assert [spec.name for spec, _ in metrics] == ["retrieval_recall_at_k"]
    assert skipped == []
    assert built == []
    assert captured["metric_names"] == ["retrieval_recall_at_k"]
    assert captured["llm"] is None
    assert captured["embeddings"] is None
    assert instance.llm is None
    assert instance.embeddings is None


def test_create_executor_uses_source_row_ids_for_trace_context(monkeypatch) -> None:
    class _FakeExecutor:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs
            self.submit_calls: list[tuple[object, tuple[object, ...]]] = []

        def submit(self, fn, *args):  # noqa: ANN001, ANN002
            self.submit_calls.append((fn, args))

    class _FakeDataset:
        def to_list(self) -> list[dict[str, str]]:
            return [{"user_input": "Q1", "reference": "A1", "response": "R1"}]

    monkeypatch.setattr(evaluator, "Executor", _FakeExecutor)

    instance = object.__new__(evaluator.Evaluator)
    instance.raise_exceptions = False

    spec = evaluator.MetricSpec(
        name="faithfulness",
        required_columns=frozenset({"response", "reference"}),
        builder=lambda llm, embeddings: None,
    )
    executor_obj, rows, metric_names = evaluator.Evaluator._create_executor(
        instance,
        dataset=_FakeDataset(),
        metrics=[(spec, object())],
        run_config=SimpleNamespace(),
        batch_size=8,
        show_progress=False,
        source_rows=[{"id": "source-1"}],
    )

    assert rows == [{"user_input": "Q1", "reference": "A1", "response": "R1"}]
    assert metric_names == ["faithfulness"]
    assert len(executor_obj.submit_calls) == 1
    submitted_fn, submitted_args = executor_obj.submit_calls[0]
    assert submitted_fn is evaluator._score_metric
    assert submitted_args[1] == {"reference": "A1", "response": "R1"}
    trace_context = submitted_args[2]
    assert trace_context.metric_name == "faithfulness"
    assert trace_context.sample_id == "source-1"
    assert trace_context.row_index == 0
    assert trace_context.required_columns == ("reference", "response")


def test_create_executor_falls_back_to_row_index_when_source_row_id_missing(monkeypatch) -> None:
    class _FakeExecutor:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.submit_calls: list[tuple[object, tuple[object, ...]]] = []

        def submit(self, fn, *args):  # noqa: ANN001, ANN002
            self.submit_calls.append((fn, args))

    class _FakeDataset:
        def to_list(self) -> list[dict[str, str]]:
            return [{"user_input": "Q1", "reference": "A1", "response": "R1"}]

    monkeypatch.setattr(evaluator, "Executor", _FakeExecutor)

    instance = object.__new__(evaluator.Evaluator)
    instance.raise_exceptions = False

    spec = evaluator.MetricSpec(
        name="faithfulness",
        required_columns=frozenset({"response"}),
        builder=lambda llm, embeddings: None,
    )
    executor_obj, _, _ = evaluator.Evaluator._create_executor(
        instance,
        dataset=_FakeDataset(),
        metrics=[(spec, object())],
        run_config=SimpleNamespace(),
        batch_size=8,
        show_progress=False,
        source_rows=[{}],
    )

    trace_context = executor_obj.submit_calls[0][1][2]
    assert trace_context.sample_id == "row-0"
