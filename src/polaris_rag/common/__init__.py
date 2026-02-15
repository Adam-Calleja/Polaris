"""
Common building blocks shared across the RAG stack.

This package provides small, widely-used primitives (e.g., document schemas and
ID aliases) intended to be imported by multiple layers of the system.

Classes
-------
Document
    Canonical document container.
DocumentChunk
    Chunk of a document with associated metadata.

Attributes
----------
DocId : TypeAlias
    Type alias for document identifiers.
ChunkId : TypeAlias
    Type alias for chunk identifiers.
__version__ : str
    Package version string. Defaults to "0.0.0-dev" when package metadata is
    unavailable.

See Also
--------
polaris_rag.common.schemas
    Defines :class:`~polaris_rag.common.schemas.Document` and
    :class:`~polaris_rag.common.schemas.DocumentChunk`.

Notes
-----
- ``metadata`` fields are intentionally left untyped; prefer defensive access
  (e.g., ``dict.get`` with defaults) when reading from metadata.
"""
from __future__ import annotations
from typing import TypeAlias

from .schemas import (
    Document,
    DocumentChunk,
)

DocId: TypeAlias = str
ChunkId: TypeAlias = str

__all__ = [
    "Document",
    "DocumentChunk",
    "DocId",
    "ChunkId",
    "__version__",
]

try:
    try:
        from importlib.metadata import version, PackageNotFoundError
        __version__ = version("my_rag_project")
    except PackageNotFoundError:
        __version__ = "0.0.0-dev"
except Exception:
    __version__ = "0.0.0-dev"