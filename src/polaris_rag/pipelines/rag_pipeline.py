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
The pipeline applies a small post-processing pass to trim trailing
generation drift from ticket-assistant responses and to rebuild the
reference key from citations actually used in the answer body.
"""

import re
import time
from typing import TYPE_CHECKING, Any, Mapping

from polaris_rag.common.request_budget import (
    GenerationTimeoutError,
    RequestBudget,
    RequestBudgetExceededError,
    RetrievalTimeoutError,
)
from polaris_rag.retrieval.node_utils import extract_doc_id, serialize_source_nodes
from polaris_rag.retrieval.query_constraints import QueryConstraints, serialize_query_constraints
from polaris_rag.retrieval.types import Retriever
from polaris_rag.generation.prompt_builder import PromptBuilder
from polaris_rag.observability.mlflow_tracking import (
    set_span_attributes,
    set_span_outputs,
    start_span,
)

if TYPE_CHECKING:
    from polaris_rag.generation.llm_interface import BaseLLM


_REFERENCE_KEY_HEADING = "REFERENCE KEY"
_REFERENCE_KEY_PATTERN = re.compile(r"^\s*REFERENCE KEY\s*:?\s*$", re.IGNORECASE)
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _sanitize_generated_response(raw_output: Any, source_nodes: list[Any]) -> str:
    """Trim post-answer drift and rebuild the reference key from cited documents."""
    text = str(raw_output or "").strip()
    if not text:
        return text

    lines = text.splitlines()
    reference_key_index = next(
        (index for index, line in enumerate(lines) if _REFERENCE_KEY_PATTERN.match(line)),
        None,
    )
    if reference_key_index is None:
        return text

    body_lines = lines[:reference_key_index]
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    cited_numbers = sorted({int(match) for match in _CITATION_PATTERN.findall("\n".join(body_lines))})
    doc_id_map = {
        index: extract_doc_id(getattr(source, "node", source))
        for index, source in enumerate(source_nodes, start=1)
    }

    reference_key_lines = [_REFERENCE_KEY_HEADING]
    for citation_number in cited_numbers:
        doc_id = doc_id_map.get(citation_number)
        if doc_id:
            reference_key_lines.append(f"[{citation_number}] : {doc_id}")

    rebuilt = "\n".join(body_lines)
    if rebuilt:
        rebuilt = f"{rebuilt}\n\n" + "\n".join(reference_key_lines)
    else:
        rebuilt = "\n".join(reference_key_lines)
    return rebuilt.strip()


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
                 query_constraint_parser: Any | None = None,
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
        self.query_constraint_parser = query_constraint_parser
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
            - ``"raw_response"``: the raw model output
            - ``"response"``: the sanitized model output
            - ``"source_nodes"``: final resolved context nodes passed to the model
            - ``"raw_source_nodes"``: raw retrieved nodes prior to context resolution
        """
        request_budget = kwargs.pop("request_budget", None)
        if request_budget is not None and not isinstance(request_budget, RequestBudget):
            raise TypeError(f"request_budget must be a RequestBudget or None, got {type(request_budget)!r}")

        call_overrides = kwargs.pop("llm_generate", None) or {}
        retriever_kwargs = dict(kwargs)
        query_constraints = self._resolve_query_constraints(
            query=query,
            provided=retriever_kwargs.pop("query_constraints", None),
        )
        serialized_query_constraints = serialize_query_constraints(query_constraints)
        if query_constraints is not None:
            retriever_kwargs["query_constraints"] = query_constraints

        with start_span(
            "rag.pipeline.run",
            inputs={"query": query},
            attributes={"prompt_name": self.prompt_name},
        ) as pipeline_span:
            if request_budget is not None:
                set_span_attributes(pipeline_span, request_budget.to_attributes())
            with start_span(
                "rag.pipeline.retrieve",
                inputs={
                    "query": query,
                    "kwargs": {
                        key: value
                        for key, value in retriever_kwargs.items()
                        if key != "query_constraints"
                    },
                    "query_constraints": serialized_query_constraints,
                },
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
                        "query_constraints": serialized_query_constraints,
                        "reranker_profile": self._reranker_profile(),
                        "reranker_fingerprint": self._reranker_fingerprint(),
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

                response = _sanitize_generated_response(raw_output, resolved_contexts)
                generation_elapsed_ms = max(
                    0,
                    int(round((time.perf_counter() - generation_started_at) * 1000.0)),
                )
                response_status = "ok" if response.strip() else "empty_response"
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
                        "raw_response": raw_output,
                        "response": response,
                        "generation_elapsed_ms": generation_elapsed_ms,
                    },
                )

            set_span_outputs(
                pipeline_span,
                {
                    "prompt": prompt,
                    "raw_response": raw_output,
                    "response": response,
                    "retrieved_contexts": self._serialize_nodes(resolved_contexts),
                    "raw_retrieved_contexts": self._serialize_nodes(retrieved_chunks),
                    "query_constraints": serialized_query_constraints,
                    "reranker_profile": self._reranker_profile(),
                    "reranker_fingerprint": self._reranker_fingerprint(),
                    "retrieval_trace": self._serialize_nodes(retrieved_chunks),
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
                    "response_status": response_status,
                    "budget_remaining_ms": request_budget.remaining_ms() if request_budget is not None else None,
                },
            )

        return {
            "prompt": prompt,
            "raw_response": raw_output,
            "response": response,
            "source_nodes": resolved_contexts,
            "raw_source_nodes": retrieved_chunks,
            "query_constraints": serialized_query_constraints,
            "reranker_profile": self._reranker_profile(),
            "reranker_fingerprint": self._reranker_fingerprint(),
            "retrieval_trace": self._serialize_nodes(retrieved_chunks),
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
        """Resolve contexts.
        
        Parameters
        ----------
        source_nodes : list[Any]
            Retrieved nodes or node wrappers to serialize.
        
        Returns
        -------
        list[Any]
            Collected results from the operation.
        """
        if self.context_resolver is None:
            return source_nodes
        return list(self.context_resolver.resolve(source_nodes))

    def _resolve_query_constraints(
        self,
        *,
        query: str,
        provided: Any,
    ) -> QueryConstraints | None:
        """Resolve query Constraints.
        
        Parameters
        ----------
        query : str
            User query text.
        provided : Any
            Value for provided.
        
        Returns
        -------
        QueryConstraints or None
            Result of the operation.
        """
        normalized = QueryConstraints.from_value(provided)
        if normalized is not None:
            return normalized
        if self.query_constraint_parser is None:
            return None
        return QueryConstraints.from_value(self.query_constraint_parser.parse(query))

    def _reranker_profile(self) -> Mapping[str, Any] | None:
        """Reranker Profile.
        
        Returns
        -------
        Mapping[str, Any] or None
            Result of the operation.
        """
        getter = getattr(self.retriever, "reranker_profile", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return None
        return None

    def _reranker_fingerprint(self) -> str | None:
        """Reranker Fingerprint.
        
        Returns
        -------
        str or None
            Result of the operation.
        """
        getter = getattr(self.retriever, "reranker_fingerprint", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return None
        return None

    @staticmethod
    def _serialize_nodes(source_nodes: list[Any]) -> list[dict[str, Any]]:
        """Serialize nodes.
        
        Parameters
        ----------
        source_nodes : list[Any]
            Retrieved nodes or node wrappers to serialize.
        
        Returns
        -------
        list[dict[str, Any]]
            Collected results from the operation.
        """
        return serialize_source_nodes(source_nodes, include_text=True)

__all__ = ['RAGPipeline']
