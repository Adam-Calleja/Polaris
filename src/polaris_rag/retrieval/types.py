"""polaris_rag.retrieval.types

Shared type definitions for the retrieval layer.

This module defines lightweight protocol and type abstractions used to
decouple retriever implementations from concrete backend classes.

Classes
-------
Retriever
    Protocol defining the minimal retriever interface.
"""

from typing import Protocol, List
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
    def retrieve(self, query: str, *, timeout_seconds: float | None = None, **kwargs) -> List[NodeWithScore]:
        """Retrieve nodes for a query.

        Parameters
        ----------
        query : str
            Natural-language query string.
        timeout_seconds : float or None, optional
            Per-request retrieval timeout when supported by the implementation.

        Returns
        -------
        list[NodeWithScore]
            Ranked list of retrieved nodes with associated scores.
        """
        ...
