"""polaris_rag.app.container

Composition root for the Polaris RAG system.

This module is the single place where concrete implementations are wired
together from configuration (LLM clients, embedder, stores, retriever, and the
end-to-end RAG pipeline). Components are constructed lazily and cached on first
access to avoid repeated expensive initialisation.

Notes
-----
- Keep this module importable with minimal side effects:
  - do not perform network calls at import time
  - do not read files at import time
  - construct expensive objects lazily (cached on first access)

- Most components are created via existing factories (embedder, stores,
  retriever, docstore, LLM interface). This module centralises those calls to
  prevent accidental duplication of clients/models per request.

Examples
--------
>>> from polaris_rag.config import GlobalConfig
>>> from polaris_rag.app.container import build_container
>>> cfg = GlobalConfig.load("config.yaml")
>>> c = build_container(cfg)
>>> answer = c.pipeline.run("my question")
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping

from polaris_rag.common.tokenisation import HeuristicTokenCounter, HuggingFaceTokenCounter, TiktokenTokenCounter

@dataclass(frozen=True)
class PolarisContainer:
    """Holds the configured, cached runtime components for the application.

    This dataclass acts as the composition root container: it wires together
    concrete implementations of LLM clients, token counters, embedders, stores,
    retrievers, and the end-to-end pipeline. Properties are cached to avoid
    repeated expensive construction.

    Parameters
    ----------
    config : Any
        Loaded global configuration object (typically :class:`polaris_rag.config.GlobalConfig`).
    """

    config: Any

    @cached_property
    def generator_llm(self) -> Any:
        """Return the LLM used to generate final answers.

        Returns
        -------
        Any
            Configured generator LLM client instance.
        """
        from polaris_rag.generation.llm_interface import create_llm

        section = _as_mapping(self.config.generator_llm)
        return create_llm(dict(section))

    @cached_property
    def evaluator_llm(self) -> Any:
        """Return the LLM used for evaluation/metrics.

        Returns
        -------
        Any
            Configured evaluator LLM client instance.
        """
        from polaris_rag.generation.llm_interface import create_llm

        section = _as_mapping(self.config.evaluator_llm)
        return create_llm(dict(section))
    
    @cached_property
    def llamaindex_llm(self):
        """Return a LlamaIndex-compatible LLM wrapper.

        Returns
        -------
        Any
            A LlamaIndex LLM instance constructed from the generator LLM config.

        Raises
        ------
        ValueError
            If the configured generator LLM provider is not supported for the
            LlamaIndex hybrid-fusion path.
        """
        cfg = _as_mapping(self.config.generator_llm)
        kind = (cfg.get("type") or cfg.get("kind") or cfg.get("provider") or "").lower().replace("-", "_")

        if kind in {"openai_like", "openailike", "open_ailike", "open_ai_like"}:
            from llama_index.llms.openai_like import OpenAILike

            model = cfg.get("model_name") or cfg.get("model")
            api_base = cfg.get("api_base")
            api_key = cfg.get("api_key")
            return OpenAILike(model=model, api_base=api_base, api_key=api_key, is_chat_model=True)

        raise ValueError(f"Hybrid fusion requires a LlamaIndex LLM; only OpenAI-like is wired. Got {kind!r}")

    @cached_property
    def prompt_builder(self) -> Any:
        """Return the prompt builder.

        The builder is initialised from ``config.prompts`` if configured.

        Notes
        -----
        To be packaging- and Docker-friendly, prompt sources are resolved
        relative to the loaded config file directory (when available), not the
        current working directory.

        Returns
        -------
        Any
            A :class:`polaris_rag.generation.prompt_builder.PromptBuilder` instance.
        """
        from polaris_rag.generation.prompt_builder import PromptBuilder

        prompts = getattr(self.config, "prompts", None)
        builder = PromptBuilder()

        if prompts is None:
            return builder

        cfg_path = (
            getattr(self.config, "config_path", None)
            or getattr(self.config, "_config_path", None)
            or getattr(self.config, "path", None)
        )
        base_dir = None
        if cfg_path:
            try:
                base_dir = Path(cfg_path).expanduser().resolve().parent
            except Exception:
                base_dir = None

        sources: list[str]
        if isinstance(prompts, str):
            sources = [prompts]
        elif isinstance(prompts, (list, tuple)):
            sources = [str(p) for p in prompts]
        else:
            raise TypeError(f"config.prompts must be a str or list[str], got {type(prompts)!r}")

        for src in sources:
            builder.register_from_source(src, base_dir=base_dir)

        return builder
    
    @cached_property
    def prompt_name(self) -> str:
        """Return the configured prompt name.

        Returns
        -------
        str
            Name of the prompt to use from the prompt builder.

        Raises
        ------
        ValueError
            If no ``prompt_name`` is configured, or if the configured prompt
            name is not registered.
        """
        prompt_name = getattr(self.config, "prompt_name", None)
        if prompt_name is None:
            raise ValueError("No prompt_name configured in config; cannot proceed.")

        if getattr(self.config, "prompts", None) is not None:
            if not self.prompt_builder.has_prompt(prompt_name):
                available = ", ".join(self.prompt_builder.list_prompts())
                raise ValueError(
                    f"Configured prompt_name {prompt_name!r} was not found in loaded prompts. "
                    f"Available: [{available}]"
                )

        return str(prompt_name)

    @cached_property
    def token_counter(self):
        """Return the token counter used for chunk sizing.

        Configuration is read from ``config.tokenization``. If the section is
        missing, a :class:`polaris_rag.common.tokenisation.HeuristicTokenCounter`
        is used by default.

        Returns
        -------
        Any
            A token counter implementation (heuristic, ``tiktoken``, or Hugging Face).

        Raises
        ------
        ValueError
            If an unknown tokenization type is configured, or if required
            configuration keys are missing for the chosen type.
        """
        cfg = _as_mapping(getattr(self.config, "tokenization", {}))
        kind = (cfg.get("type") or "heuristic").lower().replace("-", "_")

        if kind in {"heuristic", "char", "chars"}:
            cpt = cfg.get("chars_per_token", 4)
            try:
                cpt = int(cpt)
            except Exception:
                cpt = 4
            return HeuristicTokenCounter(chars_per_token=cpt)

        if kind in {"tiktoken", "openai", "openai_like"}:
            enc = cfg.get("encoding") or "cl100k_base"
            return TiktokenTokenCounter.from_encoding_name(str(enc))

        if kind in {"huggingface", "hf", "transformers"}:
            model_name = cfg.get("model_name")
            if not model_name:
                raise ValueError("tokenization.model_name is required for huggingface tokenization")

            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(str(model_name))
            return HuggingFaceTokenCounter(tokenizer=tok)

        raise ValueError(f"Unknown tokenization type: {kind!r}")

    @cached_property
    def embedder(self) -> Any:
        """Return the embedding model/client.

        Returns
        -------
        Any
            Configured embedder instance used to embed documents and queries.
        """
        from polaris_rag.retrieval.embedder import create_embedder

        section = _as_mapping(self.config.embedder)
        return create_embedder(section)

    @cached_property
    def vector_store(self) -> Any:
        """Return the vector store client/wrapper.

        Returns
        -------
        Any
            Configured vector store instance (e.g., Qdrant-backed store).
        """
        section = _as_mapping(self.config.vector_store)

        try:
            from polaris_rag.retrieval.vector_store import QdrantIndexStore
            return QdrantIndexStore.from_config_dict(
                section,
                llm=self.generator_llm,
                embedder=self.embedder,
            )
        except ImportError:
            from polaris_rag.retrieval.vector_store import create_vector_store

            return create_vector_store(section)

    @cached_property
    def vector_stores(self) -> dict[str, Any]:
        """Return vector store instances keyed by source name.

        Returns
        -------
        dict[str, Any]
            Mapping of source name to configured vector-store instance.
        """
        sections = _as_mapping(self.config.vector_stores)

        stores: dict[str, Any] = {}
        try:
            from polaris_rag.retrieval.vector_store import QdrantIndexStore

            for source_name, source_cfg in sections.items():
                stores[str(source_name)] = QdrantIndexStore.from_config_dict(
                    dict(_as_mapping(source_cfg)),
                    llm=self.generator_llm,
                    embedder=self.embedder,
                )
        except ImportError:
            from polaris_rag.retrieval.vector_store import create_vector_store

            for source_name, source_cfg in sections.items():
                stores[str(source_name)] = create_vector_store(
                    dict(_as_mapping(source_cfg)),
                    llm=self.generator_llm,
                    embedder=self.embedder,
                )

        return stores

    @cached_property
    def doc_store(self) -> Any:
        """Return the document/index store.

        Returns
        -------
        Any
            Configured document store instance.
        """
        section = _as_mapping(self.config.doc_store)
        kind = section.get("type")

        from polaris_rag.retrieval.document_store_factory import create_docstore

        return create_docstore(kind)

    @cached_property
    def retriever_source_settings(self) -> dict[str, dict[str, Any]]:
        """Return validated per-source retriever settings.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping keyed by source name with ``top_k``, ``filters`` and ``weight``.

        Raises
        ------
        KeyError
            If ``retriever.sources`` is missing.
        TypeError
            If ``retriever.sources`` is not a list or source names are invalid.
        ValueError
            If ``retriever.sources`` is empty.
        """
        section = _as_mapping(self.config.retriever)
        raw_sources = section.get("sources")
        if raw_sources is None:
            raise KeyError("Missing 'retriever.sources' in configuration.")
        if not isinstance(raw_sources, list):
            raise TypeError("'retriever.sources' must be a list of mappings.")
        if not raw_sources:
            raise ValueError("'retriever.sources' must define at least one source.")

        default_top_k = section.get("top_k")
        default_filters = section.get("filters")
        settings: dict[str, dict[str, Any]] = {}

        for idx, item in enumerate(raw_sources):
            item_map = _as_mapping(item)
            source_name = item_map.get("name")
            if not isinstance(source_name, str) or not source_name.strip():
                raise TypeError(f"'retriever.sources[{idx}].name' must be a non-empty string.")

            settings[source_name] = {
                "top_k": item_map.get("top_k", default_top_k),
                "filters": item_map.get("filters", default_filters),
                "weight": item_map.get("weight", 1.0),
            }

        return settings

    @cached_property
    def source_storage_contexts(self) -> dict[str, Any]:
        """Return source-scoped storage contexts for retrieval.

        Returns
        -------
        dict[str, Any]
            Mapping from source name to storage context.
        """
        from polaris_rag.retrieval.document_store_factory import build_storage_context

        stores = self.vector_stores
        source_settings = self.retriever_source_settings
        contexts: dict[str, Any] = {}

        for source_name in source_settings.keys():
            if source_name not in stores:
                available = sorted(stores.keys())
                raise ValueError(
                    f"Retriever source {source_name!r} is not configured in vector_stores. "
                    f"Available: {available}"
                )

            contexts[source_name] = build_storage_context(
                vector_store=stores[source_name],
                docstore=self.doc_store,
                persist_dir=None,
            )

        return contexts

    @cached_property
    def source_retrievers(self) -> dict[str, Any]:
        """Return retriever instances keyed by configured source name.

        Returns
        -------
        dict[str, Any]
            Mapping from source name to retriever instance.
        """
        section = _as_mapping(self.config.retriever)
        source_kind = (
            section.get("source_type")
            or section.get("type")
            or "vector"
        )
        if source_kind not in {"vector", "hybrid"}:
            source_kind = "vector"

        from polaris_rag.retrieval.retriever_factory import create

        source_settings = self.retriever_source_settings
        contexts = self.source_storage_contexts
        retrievers: dict[str, Any] = {}

        for source_name, cfg in source_settings.items():
            create_kwargs: dict[str, Any] = {
                "kind": source_kind,
                "storage_context": contexts[source_name],
                "top_k": cfg.get("top_k"),
                "filters": cfg.get("filters"),
            }
            if source_kind == "hybrid":
                create_kwargs["llm"] = self.llamaindex_llm

            retrievers[source_name] = create(**create_kwargs)

        return retrievers

    @cached_property
    def storage_context(self) -> Any:
        """Return the storage context for indexing/retrieval.

        This accessor returns ``None`` if the storage-context helpers are not
        available in the current environment.

        Returns
        -------
        Any or None
            Storage context object or ``None`` if unavailable.
        """
        try:
            from polaris_rag.retrieval.document_store_factory import build_storage_context
        except ImportError:
            return None

        vector_store = self.vector_store
        doc_store = self.doc_store
        persist_dir = None
        sc_cfg = getattr(self.config, "storage_context", {})
        
        if isinstance(sc_cfg, dict):
            persist_dir = sc_cfg.get("persist_dir")
            if persist_dir:
                cfg_path = (
                    getattr(self.config, "config_path", None)
                    or getattr(self.config, "_config_path", None)
                    or getattr(self.config, "path", None)
                )
                if cfg_path:
                    base_dir = Path(cfg_path).expanduser().resolve().parent
                    p = Path(str(persist_dir))
                    if not p.is_absolute():
                        persist_dir = str((base_dir / p).resolve())

        return build_storage_context(
            vector_store=vector_store,
            docstore=doc_store,
            persist_dir=persist_dir,
        )

    @cached_property
    def retriever(self) -> Any:
        """Return the configured retriever.

        Returns
        -------
        Any
            Retriever instance configured over the stores and embeddings.
        """
        section = _as_mapping(self.config.retriever)
        kind = section.get("type")

        from polaris_rag.retrieval.retriever_factory import create

        if kind == "multi_collection":
            return create(
                kind=kind,
                source_retrievers=self.source_retrievers,
                source_settings=self.retriever_source_settings,
                final_top_k=section.get("final_top_k"),
                rerank=section.get("rerank"),
            )

        if kind == "hybrid":
            return create(
                kind=kind,
                storage_context=self.storage_context,
                top_k=section.get("top_k"),
                filters=section.get("filters"),
                llm=self.llamaindex_llm,
            )
        else:
            return create(
                kind=kind,
                storage_context=self.storage_context,
                top_k=section.get("top_k"),
                filters=section.get("filters"),
            )

    @cached_property
    def pipeline(self) -> Any:
        """Return the fully wired RAG pipeline.

        Returns
        -------
        Any
            A :class:`polaris_rag.pipelines.rag_pipeline.RAGPipeline` instance.
        """
        from polaris_rag.pipelines.rag_pipeline import RAGPipeline

        try:
            return RAGPipeline(
                retriever=self.retriever,
                generator_llm=self.generator_llm,
                prompt_builder=self.prompt_builder,
            )
        except TypeError:
            return RAGPipeline(
                retriever=self.retriever,
                prompt_builder=self.prompt_builder,
                prompt_name=self.prompt_name,
                llm=self.generator_llm,
            )


def build_container(config: Any) -> PolarisContainer:
    """Create a :class:`~polaris_rag.app.container.PolarisContainer`.

    This function is intentionally small so it can serve as a single entry point
    for Streamlit apps, FastAPI lifespan hooks, CLI scripts, and tests.

    Parameters
    ----------
    config : Any
        Loaded global configuration object (typically :class:`polaris_rag.config.GlobalConfig`).

    Returns
    -------
    PolarisContainer
        Container instance with cached component accessors.
    """

    return PolarisContainer(config=config)

def _as_mapping(obj: Any) -> Mapping[str, Any]:
    """Coerce an object into a mapping.

    Parameters
    ----------
    obj : Any
        Object to interpret as a mapping. If ``obj`` is already a mapping it is
        returned as-is. If it has a ``__dict__``, that dictionary is returned.

    Returns
    -------
    Mapping[str, Any]
        A dictionary-like view of ``obj``.

    Raises
    ------
    TypeError
        If ``obj`` cannot be interpreted as a mapping.
    """
    if isinstance(obj, Mapping):
        return obj

    if hasattr(obj, "__dict__"):
        return dict(vars(obj))

    raise TypeError(f"Expected mapping type but got {type(obj)}")


__all__ = ["PolarisContainer", "build_container"]
