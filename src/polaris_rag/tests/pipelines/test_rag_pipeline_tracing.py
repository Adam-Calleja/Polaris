from __future__ import annotations

from contextlib import contextmanager

from polaris_rag.common.request_budget import RequestBudget
from polaris_rag.pipelines.rag_pipeline import RAGPipeline


class _FakeNode:
    def __init__(self, node_id: str, text: str):
        self.id_ = node_id
        self.text = text


class _FakeSource:
    def __init__(self, node: _FakeNode):
        self.node = node
        self.score = 0.9


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query: str, **kwargs):
        self.calls.append({"query": query, "kwargs": dict(kwargs)})
        return [_FakeSource(_FakeNode("doc-1", "ctx-1"))]


class _FakePromptBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def build(self, name: str, **kwargs):
        self.calls.append({"name": name, "kwargs": dict(kwargs)})
        return f"PROMPT::{name}::{kwargs.get('question')}"


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return "ANSWER"


class _Span:
    def __init__(self, name: str):
        self.name = name
        self.outputs = None



def test_pipeline_emits_expected_trace_spans(monkeypatch) -> None:
    span_calls: list[str] = []

    @contextmanager
    def _fake_start_span(name: str, **kwargs):  # noqa: ANN001
        span_calls.append(name)
        yield _Span(name)

    def _fake_set_span_outputs(span: _Span, outputs):  # noqa: ANN001
        span.outputs = outputs

    monkeypatch.setattr("polaris_rag.pipelines.rag_pipeline.start_span", _fake_start_span)
    monkeypatch.setattr("polaris_rag.pipelines.rag_pipeline.set_span_outputs", _fake_set_span_outputs)

    retriever = _FakeRetriever()
    prompt_builder = _FakePromptBuilder()
    llm = _FakeLLM()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        prompt_name="hpc_prompt",
        llm=llm,
    )

    result = pipeline.run("How do I submit a job?", llm_generate={"temperature": 0.1})

    assert result["response"] == "ANSWER"
    assert result["prompt"].startswith("PROMPT::hpc_prompt")
    assert retriever.calls[0]["kwargs"] == {}

    assert span_calls == [
        "rag.pipeline.run",
        "rag.pipeline.retrieve",
        "rag.pipeline.prompt_render",
        "rag.pipeline.generate",
    ]


def test_pipeline_propagates_budget_to_retrieval_and_generation() -> None:
    retriever = _FakeRetriever()
    prompt_builder = _FakePromptBuilder()
    llm = _FakeLLM()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        prompt_name="hpc_prompt",
        llm=llm,
    )

    budget = RequestBudget.from_timeout_ms(
        timeout_ms=110000,
        policy="official",
        retrieval_cap_ms=10000,
        cleanup_reserve_ms=5000,
    )

    pipeline.run("How do I submit a job?", request_budget=budget)

    assert retriever.calls[0]["kwargs"]["timeout_seconds"] == 10.0
    assert llm.calls[0]["kwargs"]["timeout_seconds"] > 90.0
