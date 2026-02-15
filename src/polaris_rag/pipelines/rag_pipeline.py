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
returns raw model output alongside retrieved source nodes.
"""

from typing import Any

from polaris_rag.retrieval.retriever import VectorIndexRetriever as Retriever
from polaris_rag.generation.prompt_builder import PromptBuilder
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
                 llm: BaseLLM,
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
            - ``"source_nodes"``: retrieved document chunks
        """
        retrieved_chunks = self.retriever.retrieve(query, **kwargs)

        prompt = self.prompt_builder.build(name=self.prompt_name, question=query, docs=retrieved_chunks)

        call_overrides = kwargs.pop('llm_generate', None) or {}
        gen_kwargs = {**self.llm_generate_defaults, **call_overrides}

        raw_output = self.llm.generate(prompt, **gen_kwargs)

        return {'prompt': prompt, 'response': raw_output, 'source_nodes': retrieved_chunks}

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

__all__ = ['RAGPipeline']