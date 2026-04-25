from __future__ import annotations

from contextlib import contextmanager

from polaris_rag.common.request_budget import RequestBudget
from polaris_rag.pipelines.rag_pipeline import RAGPipeline
from polaris_rag.retrieval.query_constraints import QueryConstraints


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


class _MultiNodeRetriever:
    def __init__(self, nodes: list[_FakeSource]) -> None:
        self.nodes = list(nodes)
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query: str, **kwargs):
        self.calls.append({"query": query, "kwargs": dict(kwargs)})
        return list(self.nodes)


class _FakePromptBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def build(self, name: str, **kwargs):
        self.calls.append({"name": name, "kwargs": dict(kwargs)})
        return f"PROMPT::{name}::{kwargs.get('question')}"


class _FakeChatPromptBuilder(_FakePromptBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.message_calls: list[dict[str, object]] = []

    def build_messages(self, name: str, **kwargs):
        self.message_calls.append({"name": name, "kwargs": dict(kwargs)})
        return [
            {"role": "system", "content": f"SYSTEM::{name}"},
            {"role": "user", "content": f"USER::{kwargs.get('question')}"},
        ]


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return "ANSWER"


class _FakeChatLLM(_FakeLLM):
    supports_chat_messages = True


class _FakeResolver:
    def __init__(self) -> None:
        self.calls: list[list[object]] = []

    def resolve(self, source_nodes):  # noqa: ANN001
        self.calls.append(list(source_nodes))
        return [_FakeSource(_FakeNode("ticket-1", "FULL-TICKET"))]


class _FakeQueryConstraintParser:
    def parse(self, query: str) -> QueryConstraints:
        return QueryConstraints(
            query_type="software_version",
            software_names=["GROMACS"],
            software_versions=["2024.4"],
            scope_family_names=["cclake"],
            version_sensitive_guess=True,
        )


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


def test_pipeline_uses_resolved_contexts_for_prompt_and_output() -> None:
    retriever = _FakeRetriever()
    prompt_builder = _FakePromptBuilder()
    llm = _FakeLLM()
    resolver = _FakeResolver()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        prompt_name="hpc_prompt",
        llm=llm,
        context_resolver=resolver,
    )

    result = pipeline.run("How do I submit a job?")

    docs = prompt_builder.calls[0]["kwargs"]["docs"]
    assert resolver.calls
    assert docs[0].node.id_ == "ticket-1"
    assert docs[0].node.text == "FULL-TICKET"
    assert result["source_nodes"][0].node.id_ == "ticket-1"
    assert result["raw_source_nodes"][0].node.id_ == "doc-1"


def test_pipeline_uses_structured_messages_for_chat_models() -> None:
    retriever = _FakeRetriever()
    prompt_builder = _FakeChatPromptBuilder()
    llm = _FakeChatLLM()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        prompt_name="hpc_prompt",
        llm=llm,
    )

    result = pipeline.run("How do I submit a job?")

    assert result["prompt"] == "PROMPT::hpc_prompt::How do I submit a job?"
    assert result["prompt_messages"] == [
        {"role": "system", "content": "SYSTEM::hpc_prompt"},
        {"role": "user", "content": "USER::How do I submit a job?"},
    ]
    assert llm.calls[0]["prompt"] == result["prompt_messages"]
    assert prompt_builder.message_calls[0]["kwargs"]["docs"][0].node.id_ == "doc-1"


def test_pipeline_includes_and_forwards_query_constraints() -> None:
    retriever = _FakeRetriever()
    prompt_builder = _FakePromptBuilder()
    llm = _FakeLLM()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        prompt_name="hpc_prompt",
        llm=llm,
        query_constraint_parser=_FakeQueryConstraintParser(),
    )

    result = pipeline.run("How do I run GROMACS 2024.4?")

    forwarded = retriever.calls[0]["kwargs"]["query_constraints"]
    assert isinstance(forwarded, QueryConstraints)
    assert forwarded.software_names == ["GROMACS"]
    assert result["query_constraints"] == {
        "query_type": "software_version",
        "system_names": [],
        "partition_names": [],
        "service_names": [],
        "scope_family_names": ["cclake"],
        "software_names": ["GROMACS"],
        "software_versions": ["2024.4"],
        "module_names": [],
        "toolchain_names": [],
        "toolchain_versions": [],
        "scope_required": None,
        "version_sensitive_guess": True,
    }


def test_pipeline_sanitizes_reference_key_and_trailing_repetition() -> None:
    retriever = _MultiNodeRetriever(
        [
            _FakeSource(_FakeNode("HPCSSUP-81211", "ctx-1")),
            _FakeSource(_FakeNode("a03b4ce7-a07d-47fd-b5d2-ad1cb1c84b47", "ctx-2")),
            _FakeSource(_FakeNode("9ecc5a67-40b6-42f7-8dba-beb304414ea6", "ctx-3")),
            _FakeSource(_FakeNode("HPCSSUP-87120", "ctx-4")),
            _FakeSource(_FakeNode("767259c7-b7f8-4f0f-a71f-27c965f69a3c", "ctx-5")),
        ]
    )
    prompt_builder = _FakePromptBuilder()

    class _RepeatingLLM:
        def generate(self, prompt: str, **kwargs):  # noqa: ANN001
            return (
                "CLASSIFICATION\n"
                "Category: Software / GROMACS Version\n"
                "Technical Action Required: No\n"
                "Portal Action Required: No\n"
                "Urgency: Low\n\n"
                "QUICK ASSESSMENT\n"
                "The latest version of GROMACS available on CCLake is 2025.1 [4].\n\n"
                "REFERENCE KEY\n"
                "[4] : HPCSSUP-87120\n"
                "[1] : HPCSSUP-81211\n"
                "[2] : a03b4ce7-a07d-47fd-b5d2-ad1cb1c84b47\n"
                "The correct response is provided above.\n"
                "The correct response is provided above.\n"
            )

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        prompt_name="hpc_prompt",
        llm=_RepeatingLLM(),
    )

    result = pipeline.run("What is the latest GROMACS version available on CCLake?")

    assert result["raw_response"].endswith("The correct response is provided above.\n")
    assert result["response"] == (
        "CLASSIFICATION\n"
        "Category: Software / GROMACS Version\n"
        "Technical Action Required: No\n"
        "Portal Action Required: No\n"
        "Urgency: Low\n\n"
        "QUICK ASSESSMENT\n"
        "The latest version of GROMACS available on CCLake is 2025.1 [4].\n\n"
        "REFERENCE KEY\n"
        "[4] : HPCSSUP-87120"
    )


def test_pipeline_keeps_empty_reference_key_when_no_citations_are_used() -> None:
    retriever = _FakeRetriever()
    prompt_builder = _FakePromptBuilder()

    class _ReferenceKeyOnlyLLM:
        def generate(self, prompt: str, **kwargs):  # noqa: ANN001
            return (
                "QUICK ASSESSMENT\n"
                "There is not enough retrieved evidence to confirm the path safely.\n\n"
                "REFERENCE KEY\n"
                "The final answer is provided above.\n"
            )

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        prompt_name="hpc_prompt",
        llm=_ReferenceKeyOnlyLLM(),
    )

    result = pipeline.run("Can you confirm the exact new path for my project data?")

    assert result["response"] == (
        "QUICK ASSESSMENT\n"
        "There is not enough retrieved evidence to confirm the path safely.\n\n"
        "REFERENCE KEY"
    )
