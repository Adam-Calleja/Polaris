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
    MarkdownDocument,
)
from .request_budget import (
    DEFAULT_EVALUATION_DEADLINES,
    EVAL_POLICY_DIAGNOSTIC,
    EVAL_POLICY_INTERACTIVE,
    EVAL_POLICY_OFFICIAL,
    EmptyResponseError,
    EvaluationDeadlines,
    FAILURE_CLASS_API_INTERNAL_ERROR,
    FAILURE_CLASS_EMPTY_RESPONSE,
    FAILURE_CLASS_GENERATION_TIMEOUT,
    FAILURE_CLASS_INVALID_INPUT,
    FAILURE_CLASS_RETRIEVAL_TIMEOUT,
    FAILURE_CLASS_TRANSPORT_ERROR,
    FAILURE_STAGE_API,
    FAILURE_STAGE_DATASET,
    FAILURE_STAGE_GENERATION,
    FAILURE_STAGE_RETRIEVAL,
    GenerationTimeoutError,
    INFRA_FAILURE_CLASSES,
    POLARIS_EVAL_POLICY_HEADER,
    POLARIS_TIMEOUT_HEADER,
    PolarisRuntimeError,
    PolarisTimeoutError,
    RequestBudget,
    RequestBudgetExceededError,
    RetrievalTimeoutError,
    VALID_EVAL_POLICIES,
    build_failure_detail,
    is_timeout_exception,
    normalize_evaluation_policy,
    resolve_evaluation_deadlines,
)
from .evaluation_api import POLARIS_EVAL_INCLUDE_METADATA_HEADER

DocId: TypeAlias = str
ChunkId: TypeAlias = str

__all__ = [
    "Document",
    "DocumentChunk",
    "MarkdownDocument",
    "DocId",
    "ChunkId",
    "DEFAULT_EVALUATION_DEADLINES",
    "EVAL_POLICY_DIAGNOSTIC",
    "EVAL_POLICY_INTERACTIVE",
    "EVAL_POLICY_OFFICIAL",
    "EmptyResponseError",
    "EvaluationDeadlines",
    "FAILURE_CLASS_API_INTERNAL_ERROR",
    "FAILURE_CLASS_EMPTY_RESPONSE",
    "FAILURE_CLASS_GENERATION_TIMEOUT",
    "FAILURE_CLASS_INVALID_INPUT",
    "FAILURE_CLASS_RETRIEVAL_TIMEOUT",
    "FAILURE_CLASS_TRANSPORT_ERROR",
    "FAILURE_STAGE_API",
    "FAILURE_STAGE_DATASET",
    "FAILURE_STAGE_GENERATION",
    "FAILURE_STAGE_RETRIEVAL",
    "GenerationTimeoutError",
    "INFRA_FAILURE_CLASSES",
    "POLARIS_EVAL_POLICY_HEADER",
    "POLARIS_EVAL_INCLUDE_METADATA_HEADER",
    "POLARIS_TIMEOUT_HEADER",
    "PolarisRuntimeError",
    "PolarisTimeoutError",
    "RequestBudget",
    "RequestBudgetExceededError",
    "RetrievalTimeoutError",
    "VALID_EVAL_POLICIES",
    "__version__",
    "build_failure_detail",
    "is_timeout_exception",
    "normalize_evaluation_policy",
    "resolve_evaluation_deadlines",
]

try:
    try:
        from importlib.metadata import version, PackageNotFoundError
        __version__ = version("my_rag_project")
    except PackageNotFoundError:
        __version__ = "0.0.0-dev"
except Exception:
    __version__ = "0.0.0-dev"
