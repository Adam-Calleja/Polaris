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

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("polaris-rag")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .config import GlobalConfig
from .app.container import PolarisContainer, build_container
from .pipelines.rag_pipeline import RAGPipeline
from .common import Document, DocumentChunk

__all__ = [
    "__version__",
    "GlobalConfig",
    "PolarisContainer",
    "build_container",
    "RAGPipeline",
    "Document",
    "DocumentChunk",
]
