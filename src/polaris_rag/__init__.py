"""polaris_rag

Polaris RAG system package.

This package contains the core building blocks for a production-grade
Retrieval-Augmented Generation (RAG) assistant, including configuration,
retrieval components, prompt/generation utilities, and end-to-end pipeline
orchestration.

Attributes
----------
__version__ : str
    Package version string. Defaults to ``"0.0.0-dev"`` when package metadata is
    unavailable.

Modules
-------
config
    Global configuration loader and cached accessors.
app
    Application container and composition root for wiring components.
pipelines
    High-level pipeline orchestration (retrieval → prompting → generation).
retrieval
    Document loading, preprocessing, chunking, embedding, stores, and retrievers.
generation
    LLM and prompt-building interfaces and factories.
common
    Shared schemas and utilities (e.g., documents and chunks).

Exports
-------
GlobalConfig
    Global configuration loader and accessor.
PolarisContainer
    Cached runtime component container for applications.
build_container
    Factory function to construct a configured :class:`~polaris_rag.app.container.PolarisContainer`.
RAGPipeline
    End-to-end Retrieval-Augmented Generation pipeline.
Document
    Canonical document container schema.
DocumentChunk
    Chunk schema derived from a parent document.
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("polaris-rag")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

_EXPORTS: dict[str, tuple[str, str]] = {
    "GlobalConfig": ("polaris_rag.config", "GlobalConfig"),
    "PolarisContainer": ("polaris_rag.app.container", "PolarisContainer"),
    "build_container": ("polaris_rag.app.container", "build_container"),
    "RAGPipeline": ("polaris_rag.pipelines.rag_pipeline", "RAGPipeline"),
    "Document": ("polaris_rag.common", "Document"),
    "DocumentChunk": ("polaris_rag.common", "DocumentChunk"),
}


def __getattr__(name: str):
    """Resolve lazily exposed attributes.
    
    Parameters
    ----------
    name : str
        Human-readable name for the resource or tracing span.
    
    Raises
    ------
    AttributeError
        If the requested attribute cannot be resolved.
    """
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return available attribute names for interactive discovery.
    
    Returns
    -------
    list[str]
        Available attribute names for the object or module.
    """
    return sorted(set(globals().keys()) | set(_EXPORTS.keys()))

__all__ = [
    "__version__",
    "GlobalConfig",
    "PolarisContainer",
    "build_container",
    "RAGPipeline",
    "Document",
    "DocumentChunk",
]
