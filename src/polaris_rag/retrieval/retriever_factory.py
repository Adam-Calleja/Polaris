"""polaris_rag.retrieval.retriever_factory

Factory and registry for retriever implementations.

This module provides a lightweight plugin-style registry for retriever
implementations used by the retrieval layer. Retriever constructors are
registered under a string key and instantiated via a single factory
function.

Functions
---------
register
    Decorator used to register a retriever builder under a name.
create
    Construct a retriever instance by kind.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.core import StorageContext

from polaris_rag.retrieval.retriever import (
    HybridRetriever,
    MultiCollectionRetriever,
    VectorIndexRetriever,
)
from polaris_rag.retrieval.types import Retriever

_BUILDERS: Dict[str, Callable[..., Retriever]] = {}

def register(name: str):
    """Register a retriever builder under a name.

    This decorator registers a callable that constructs a retriever instance.
    The callable must accept keyword arguments compatible with the
    :class:`~polaris_rag.retrieval.types.Retriever` interface.

    Parameters
    ----------
    name : str
        Name under which the retriever builder should be registered.

    Returns
    -------
    Callable
        Decorator that registers the wrapped builder function.
    """
    def _wrap(fn: Callable[..., Retriever]):
        """Wrap and register a retriever builder function.

        Parameters
        ----------
        fn : Callable[..., Retriever]
            Builder function that returns a retriever instance.

        Returns
        -------
        Callable[..., Retriever]
            The original builder function, unmodified.
        """
        _BUILDERS[name] = fn
        return fn
    return _wrap

def create(
    *,
    kind: str,
    storage_context: Optional[StorageContext] = None,
    top_k: Optional[int] = None,
    filters: Optional[MetadataFilters] = None,
    **kwargs,
) -> Retriever:
    """Create a retriever instance by kind.

    Parameters
    ----------
    kind : str
        Registered retriever kind to instantiate (e.g., ``"vector"``,
        ``"hybrid"``, ``"multi_collection"``).
    storage_context : StorageContext or None, optional
        Storage context providing access to vector and document stores. Required
        for retriever kinds that depend on a single store context.
    top_k : int or None, optional
        Number of top results to retrieve. Forwarded when provided.
    filters : MetadataFilters or None, optional
        Optional metadata filters applied during retrieval. Forwarded when provided.
    **kwargs : Any
        Additional keyword arguments forwarded to the retriever builder.

    Returns
    -------
    Retriever
        Instantiated retriever.

    Raises
    ------
    ValueError
        If ``kind`` does not correspond to a registered retriever.
    """
    if kind not in _BUILDERS:
        raise ValueError(f"Unknown retriever kind: {kind}. Available: {list(_BUILDERS)}")

    call_kwargs = dict(kwargs)
    if storage_context is not None:
        call_kwargs["storage_context"] = storage_context
    if top_k is not None:
        call_kwargs["top_k"] = top_k
    if filters is not None:
        call_kwargs["filters"] = filters

    return _BUILDERS[kind](**call_kwargs)

@register("vector")
def _build_vector(**kw) -> Retriever:
    """Build a vector-only retriever.

    Returns
    -------
    Retriever
        Vector-based retriever instance.
    """
    return VectorIndexRetriever(**kw)

@register("hybrid")
def _build_hybrid(**kw) -> Retriever:
    """Build a hybrid retriever.

    Returns
    -------
    Retriever
        Hybrid retriever instance combining vector and keyword search.
    """
    return HybridRetriever(**kw)


@register("multi_collection")
def _build_multi_collection(**kw) -> Retriever:
    """Build a multi-collection retriever.

    Returns
    -------
    Retriever
        Retriever that merges and reranks candidates from multiple sources.
    """
    return MultiCollectionRetriever(**kw)
