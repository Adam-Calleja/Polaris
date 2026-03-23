"""polaris_rag.retrieval.types

Shared type definitions for the retrieval layer.

This module defines lightweight protocol and type abstractions used to
decouple retriever implementations from concrete backend classes.

Classes
-------
Retriever
    Protocol defining the minimal retriever interface.
"""

from typing import Any, Protocol, List
from llama_index.core.schema import NodeWithScore

class Retriever(Protocol):
    """Protocol defining the retriever interface.

    A retriever is responsible for taking a natural-language query string and
    returning a ranked list of document nodes with associated scores. Concrete
    implementations may use vector similarity search, keyword search, or a
    hybrid of multiple strategies.

    Methods
    -------
    retrieve
        Retrieve nodes relevant to a query.
    """
    def retrieve(
        self,
        query: str,
        *,
        timeout_seconds: float | None = None,
        query_constraints: Any | None = None,
        **kwargs,
    ) -> List[NodeWithScore]:
        """Retrieve nodes for a query.

        Parameters
        ----------
        query : str
            Natural-language query string.
        timeout_seconds : float or None, optional
            Per-request retrieval timeout when supported by the implementation.
        query_constraints : Any or None, optional
            Optional parsed query-constraint payload. Current retrievers accept
            this for forward compatibility even when they do not yet use it.

        Returns
        -------
        list[NodeWithScore]
            Ranked list of retrieved nodes with associated scores.
        """
        ...
