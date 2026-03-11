"""polaris_rag.pipelines.rag_pipeline

End-to-end Retrieval-Augmented Generation (RAG) pipeline orchestration.

This module defines the :class:`RAGPipeline`, which coordinates query-time
retrieval, prompt construction, LLM invocation, and (optionally) downstream
post-processing.

Classes
-------
RAGPipeline
    Orchestrates retrieval → prompt building → generation → post-processing.

Notes
-----
Post-processing hooks are not yet implemented; the pipeline currently
returns raw model output alongside the final resolved source nodes used
for prompt rendering.
"""

import time
from typing import TYPE_CHECKING, Any

from polaris_rag.common.request_budget import (
    GenerationTimeoutError,
    RequestBudget,
    RequestBudgetExceededError,
    RetrievalTimeoutError,
)
from polaris_rag.retrieval.types import Retriever
from polaris_rag.generation.prompt_builder import PromptBuilder
from polaris_rag.observability.mlflow_tracking import (
    set_span_attributes,
    set_span_outputs,
    start_span,
)

if TYPE_CHECKING:
    from polaris_rag.generation.llm_interface import BaseLLM

class RAGPipeline:
    """Retrieval-Augmented Generation (RAG) orchestrator.

    This class wires together:
    - a retriever to fetch relevant document chunks
    - a prompt builder to assemble an LLM prompt
    - an LLM interface for text generation

    The pipeline is intentionally lightweight and stateless beyond its
    configured components, making it safe to reuse across requests.

    Parameters
    ----------
    retriever : Retriever
        Component that embeds the query and fetches top-k document chunks.
    prompt_builder : PromptBuilder
        Component that builds a textual prompt from the query and retrieved chunks.
    prompt_name : str
        Name of the prompt template to use when rendering prompts.
    llm : BaseLLM
        Language model interface used for text generation.
    llm_generate_defaults : dict or None, optional
        Default keyword arguments forwarded to ``llm.generate``. If ``None``,
        a conservative default configuration is used.
    """

    def __init__(self, 
                 retriever: Retriever, 
                 prompt_builder: PromptBuilder, 
                 prompt_name: str,
                 llm: "BaseLLM",
                 context_resolver: Any | None = None,
                 llm_generate_defaults: dict | None = None,
        ):
        """Initialise the RAG pipeline.

        Parameters
        ----------
        retriever : Retriever
            Component that embeds the query and fetches top-k document chunks.
        prompt_builder : PromptBuilder
            Component that builds a textual prompt from the query and retrieved chunks.
        prompt_name : str
            Name of the prompt template to use.
        llm : BaseLLM
            Language model interface for text generation.
        llm_generate_defaults : dict or None, optional
            Default keyword arguments forwarded to ``llm.generate``. If ``None``,
            sensible defaults for stop sequences, temperature, and max tokens
            are applied.
        """
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.prompt_name = prompt_name
        self.llm = llm
        self.context_resolver = context_resolver
        self.llm_generate_defaults = llm_generate_defaults or {
            'stop': ['\nUser:', '\n\nUser:'],
            'temperature': 0.2,
            'max_tokens': 1024,
        }

    def run(self, query: str, **kwargs) -> Any:
        """Execute the RAG pipeline for a single query.

        The execution order is:
        1. Retrieve relevant document chunks for the query.
        2. Build a prompt using the configured template.
        3. Generate a response from the language model.

        Parameters
        ----------
        query : str
            User’s natural language question.
        **kwargs : Any
            Additional keyword arguments forwarded to the retriever. The special
            key ``llm_generate`` may be used to override generation parameters
            for this call only.

        Returns
        -------
        dict
            Dictionary containing:
            - ``"prompt"``: the rendered prompt string
            - ``"response"``: the raw model output
            - ``"source_nodes"``: final resolved context nodes passed to the model
            - ``"raw_source_nodes"``: raw retrieved nodes prior to context resolution
        """
        request_budget = kwargs.pop("request_budget", None)
        if request_budget is not None and not isinstance(request_budget, RequestBudget):
            raise TypeError(f"request_budget must be a RequestBudget or None, got {type(request_budget)!r}")

        call_overrides = kwargs.pop("llm_generate", None) or {}
        retriever_kwargs = dict(kwargs)

        with start_span(
            "rag.pipeline.run",
            inputs={"query": query},
            attributes={"prompt_name": self.prompt_name},
        ) as pipeline_span:
            if request_budget is not None:
                set_span_attributes(pipeline_span, request_budget.to_attributes())
            with start_span(
                "rag.pipeline.retrieve",
                inputs={"query": query, "kwargs": retriever_kwargs},
            ) as retrieval_span:
                if request_budget is not None:
                    set_span_attributes(retrieval_span, request_budget.to_attributes())
                retrieval_started_at = time.perf_counter()
                retrieval_timeout_seconds = None
                if request_budget is not None:
                    retrieval_timeout_seconds = request_budget.child_timeout_seconds(
                        stage="retrieval",
                        reserve_ms=request_budget.cleanup_reserve_ms,
                    )
                    retriever_kwargs.setdefault("timeout_seconds", retrieval_timeout_seconds)
                try:
                    retrieved_chunks = self.retriever.retrieve(query, **retriever_kwargs)
                except RequestBudgetExceededError:
                    set_span_attributes(
                        retrieval_span,
                        {
                            "failure_class": "retrieval_timeout",
                            "response_status": "timeout",
                            "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                        },
                    )
                    set_span_outputs(
                        retrieval_span,
                        {
                            "error": "RequestBudgetExceededError: retrieval cannot start because the request budget is exhausted",
                        },
                    )
                    raise
                except RetrievalTimeoutError:
                    set_span_attributes(
                        retrieval_span,
                        {
                            "failure_class": "retrieval_timeout",
                            "response_status": "timeout",
                            "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                        },
                    )
                    set_span_outputs(
                        retrieval_span,
                        {
                            "error": "RetrievalTimeoutError: retrieval exceeded request deadline",
                        },
                    )
                    raise
                except TimeoutError as exc:
                    set_span_attributes(
                        retrieval_span,
                        {
                            "failure_class": "retrieval_timeout",
                            "response_status": "timeout",
                            "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                        },
                    )
                    set_span_outputs(
                        retrieval_span,
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    raise RetrievalTimeoutError("retrieval exceeded request deadline") from exc

                retrieval_elapsed_ms = max(0, int(round((time.perf_counter() - retrieval_started_at) * 1000.0)))
                set_span_attributes(
                    retrieval_span,
                    {
                        "retrieval_elapsed_ms": retrieval_elapsed_ms,
                        "response_status": "ok",
                    },
                )
                set_span_outputs(
                    retrieval_span,
                    {
                        "retrieved_count": len(retrieved_chunks),
                        "retrieved_contexts": self._serialize_nodes(retrieved_chunks),
                        "retrieval_elapsed_ms": retrieval_elapsed_ms,
                    },
                )

            resolved_contexts = self._resolve_contexts(retrieved_chunks)
            with start_span(
                "rag.pipeline.prompt_render",
                inputs={"query": query, "retrieved_contexts": self._serialize_nodes(resolved_contexts)},
            ) as prompt_span:
                prompt = self.prompt_builder.build(
                    name=self.prompt_name,
                    question=query,
                    docs=resolved_contexts,
                )
                set_span_outputs(
                    prompt_span,
                    {
                        "prompt": prompt,
                        "resolved_contexts": self._serialize_nodes(resolved_contexts),
                    },
                )

            gen_kwargs = {**self.llm_generate_defaults, **call_overrides}
            with start_span(
                "rag.pipeline.generate",
                inputs={"prompt": prompt, "llm_generate": gen_kwargs},
            ) as generation_span:
                if request_budget is not None:
                    set_span_attributes(generation_span, request_budget.to_attributes())
                generation_started_at = time.perf_counter()
                generation_timeout_seconds = None
                if request_budget is not None:
                    generation_timeout_seconds = request_budget.child_timeout_seconds(
                        stage="generation",
                        reserve_ms=request_budget.cleanup_reserve_ms,
                    )
                    gen_kwargs["timeout_seconds"] = generation_timeout_seconds
                try:
                    raw_output = self.llm.generate(prompt, **gen_kwargs)
                except RequestBudgetExceededError:
                    set_span_attributes(
                        generation_span,
                        {
                            "failure_class": "generation_timeout",
                            "response_status": "timeout",
                            "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                        },
                    )
                    set_span_outputs(
                        generation_span,
                        {
                            "error": "RequestBudgetExceededError: generation cannot start because the request budget is exhausted",
                        },
                    )
                    raise
                except GenerationTimeoutError:
                    set_span_attributes(
                        generation_span,
                        {
                            "failure_class": "generation_timeout",
                            "response_status": "timeout",
                            "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                        },
                    )
                    set_span_outputs(
                        generation_span,
                        {
                            "error": "GenerationTimeoutError: generation exceeded request deadline",
                        },
                    )
                    raise
                except TimeoutError as exc:
                    set_span_attributes(
                        generation_span,
                        {
                            "failure_class": "generation_timeout",
                            "response_status": "timeout",
                            "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                        },
                    )
                    set_span_outputs(
                        generation_span,
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    raise GenerationTimeoutError("generation exceeded request deadline") from exc

                generation_elapsed_ms = max(
                    0,
                    int(round((time.perf_counter() - generation_started_at) * 1000.0)),
                )
                response_status = "ok" if str(raw_output or "").strip() else "empty_response"
                set_span_attributes(
                    generation_span,
                    {
                        "generation_elapsed_ms": generation_elapsed_ms,
                        "response_status": response_status,
                    },
                )
                set_span_outputs(
                    generation_span,
                    {
                        "response": raw_output,
                        "generation_elapsed_ms": generation_elapsed_ms,
                    },
                )

            set_span_outputs(
                pipeline_span,
                {
                    "prompt": prompt,
                    "response": raw_output,
                    "retrieved_contexts": self._serialize_nodes(resolved_contexts),
                    "raw_retrieved_contexts": self._serialize_nodes(retrieved_chunks),
                    "timings": {
                        "retrieval_elapsed_ms": retrieval_elapsed_ms,
                        "generation_elapsed_ms": generation_elapsed_ms,
                    },
                },
            )
            set_span_attributes(
                pipeline_span,
                {
                    "retrieval_elapsed_ms": retrieval_elapsed_ms,
                    "generation_elapsed_ms": generation_elapsed_ms,
                    "response_status": "ok" if str(raw_output or "").strip() else "empty_response",
                    "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                },
            )

        return {
            "prompt": prompt,
            "response": raw_output,
            "source_nodes": resolved_contexts,
            "raw_source_nodes": retrieved_chunks,
            "timings": {
                "retrieval_elapsed_ms": retrieval_elapsed_ms,
                "generation_elapsed_ms": generation_elapsed_ms,
            },
            "budget": request_budget.to_attributes() if request_budget is not None else {},
        }

    def __call__(self, query: str, **kwargs) -> Any:
        """Execute the pipeline as a callable.

        This is a convenience wrapper around :meth:`run`.

        Parameters
        ----------
        query : str
            User’s natural language question.
        **kwargs : Any
            Additional keyword arguments forwarded to :meth:`run`.

        Returns
        -------
        Any
            Result of :meth:`run(query, **kwargs)`.
        """
        return self.run(query, **kwargs)

    def _resolve_contexts(self, source_nodes: list[Any]) -> list[Any]:
        if self.context_resolver is None:
            return source_nodes
        return list(self.context_resolver.resolve(source_nodes))

    @staticmethod
    def _serialize_nodes(source_nodes: list[Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for idx, source in enumerate(source_nodes, start=1):
            node = getattr(source, "node", source)

            doc_id = None
            for attr in ("id_", "node_id", "id"):
                value = getattr(node, attr, None)
                if isinstance(value, str) and value:
                    doc_id = value
                    break

            text = getattr(node, "text", None)
            if not isinstance(text, str) and hasattr(node, "get_content"):
                try:
                    content = node.get_content()
                    text = content if isinstance(content, str) else str(content)
                except Exception:
                    text = ""
            if not isinstance(text, str):
                text = str(text or "")

            score_raw = getattr(source, "score", None)
            score = float(score_raw) if isinstance(score_raw, (int, float)) else None

            items.append(
                {
                    "rank": idx,
                    "doc_id": doc_id or "<unknown-doc-id>",
                    "text": text,
                    "score": score,
                }
            )

        return items

__all__ = ['RAGPipeline']
