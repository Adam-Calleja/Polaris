"""polaris_rag.retrieval.embedder

Embedding interfaces and factories for the retrieval layer.

This module defines a small provider-agnostic interface for producing vector
embeddings from text, along with concrete implementations backed by
LlamaIndex embedding wrappers. A factory function is provided to construct
an embedder implementation from configuration.

Classes
-------
BaseEmbedder
    Abstract interface specifying the API used by the retrieval pipeline.
HuggingFaceEmbedder
    Embedder backed by a Hugging Face SentenceTransformer via LlamaIndex.
OpenAILikeEmbedder
    Embedder backed by an OpenAI-compatible HTTP API via LlamaIndex.

Functions
---------
create_embedder
    Create an embedder implementation from a configuration mapping.
"""

from abc import ABC, abstractmethod
import time
import httpx
from langchain_core.callbacks import BaseCallbackHandler
from llama_index.core.base.embeddings.base import BaseEmbedding as LlamaIndexBaseEmbedding
from typing import Any, Dict, Mapping, Optional
import yaml
import asyncio
from openai import OpenAI as OpenAIClient
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError

from polaris_rag.common.request_budget import RetrievalTimeoutError

OPENAI_RETRYABLE_STATUS_CODES = frozenset({408, 424, 429, 500, 502, 503, 504})


def _as_bool(value: Any, default: bool) -> bool:
    """As Bool.
    
    Parameters
    ----------
    value : Any
        Input value to normalize, coerce, or inspect.
    default : bool
        Fallback value to use when normalization fails.
    
    Returns
    -------
    bool
        `True` if as Bool; otherwise `False`.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _normalize_api_base_sequence(
    *,
    primary_api_base: str,
    api_bases: Any = None,
    failover_api_bases: Any = None,
) -> tuple[str, ...]:
    ordered: list[str] = []

    def _extend(raw: Any) -> None:
        if raw is None:
            return
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        for value in values:
            normalized = str(value or "").strip()
            if normalized and normalized not in ordered:
                ordered.append(normalized)

    _extend(primary_api_base)
    _extend(api_bases)
    _extend(failover_api_bases)

    if not ordered:
        raise ValueError("At least one embedder api_base must be configured.")

    return tuple(ordered)


def _is_retryable_openai_embedding_error(exc: Exception) -> bool:
    if isinstance(exc, (OpenAIAPITimeoutError, OpenAIAPIConnectionError, httpx.TimeoutException)):
        return True

    if isinstance(exc, OpenAIAPIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code in OPENAI_RETRYABLE_STATUS_CODES

    return False


class BaseEmbedder(ABC):
    """Abstract interface for text embedding.
    
    Concrete implementations wrap a provider-specific embedder and expose a small,
    consistent API used by the Polaris retrieval pipeline.
    
    Methods
    -------
    get_embedder
        Return the LlamaIndex embedding instance.
    from_config
        Create an embedder from a YAML configuration file.
    from_config_dict
        Create an embedder from a configuration mapping.
    embed_query
        Embed a single query string.
    embed_documents
        Embed multiple documents.
    aembed_documents
        Asynchronously embed a batch of documents.
    """

    @abstractmethod
    def get_embedder(self) -> LlamaIndexBaseEmbedding:
        """
        Return the LlamaIndex embedding instance.

        Returns
        -------
        LlamaIndexBaseEmbedding
            The underlying LlamaIndex embedding.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(
            cls,
            config_path: str,
            callback_manager: BaseCallbackHandler = None
        ) -> "BaseEmbedder":
        """Create an embedder from a YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        BaseEmbedder
            An initialised embedder implementation.

        Raises
        ------
        FileNotFoundError
            If ``config_path`` does not exist.
        ValueError
            If the configuration is invalid for the concrete implementation.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config_dict(
            cls,
            config: Dict[str, Any],
            callback_manager: BaseCallbackHandler = None
        ) -> "BaseEmbedder":
        """Create an embedder from a configuration mapping.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration mapping.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        BaseEmbedder
            An initialised embedder implementation.

        Raises
        ------
        ValueError
            If required configuration keys are missing or invalid.
        """
        pass

    def embed_query(self, query: str, timeout_seconds: float | None = None) -> list[float]:
        """Embed a single query string.

        This method attempts to call common embedding method names on the
        underlying embedder in the following order:
        ``embed_query``, ``get_text_embedding``, ``embed_documents``, ``embed``.

        Parameters
        ----------
        query : str
            Query string to embed.
        timeout_seconds : float or None, optional
            Per-request timeout for the embedding call when supported.

        Returns
        -------
        list[float]
            Embedding vector for the query.

        Raises
        ------
        AttributeError
            If no compatible embedding method is available on the underlying embedder.
        """
        embedder = self.get_embedder()
        for method in ("embed_query", "get_text_embedding", "embed_documents", "embed"):
            if hasattr(embedder, method):
                fn = getattr(embedder, method)
                if method == "embed_documents":
                    return fn([query])[0]
                return fn(query)
            
        raise AttributeError(f"No embedding method found on {embedder!r}")

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed multiple documents.

        If the underlying embedder does not provide a native ``embed_documents``
        method, embeddings are computed by calling :meth:`embed_query` for each
        document.

        Parameters
        ----------
        documents : list[str]
            Documents to embed.

        Returns
        -------
        list[list[float]]
            Embedding vectors for each document.
        """
        embedder = self.get_embedder()

        if hasattr(embedder, "embed_documents"):
            return embedder.embed_documents(documents)
        
        return [self.embed_query(doc) for doc in documents]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed a batch of documents.

        If the underlying embedder does not provide an async API, embeddings are
        computed in a thread pool via ``run_in_executor``.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list[list[float]]
            Embedding vectors for each text.
        """
        loop = asyncio.get_running_loop()
        
        return await loop.run_in_executor(None, self.embed_documents, texts)

class HuggingFaceEmbedder(BaseEmbedder):
    """Embedder backed by a Hugging Face SentenceTransformer via LlamaIndex.

    This implementation wraps :class:`llama_index.embeddings.huggingface.HuggingFaceEmbedding`.

    Parameters
    ----------
    model_name : str
        Name or path of the embedding model.
    device : str
        Device identifier (e.g., ``"cuda"``, ``"cpu"``, ``"mps"``).
    trust_remote_code : bool, optional
        Whether to allow custom model code from the Hugging Face Hub.
    callback_manager : BaseCallbackHandler, optional
        Optional callback handler for logging/telemetry/streaming.
    model_kwargs : dict[str, Any] or None, optional
        Additional keyword arguments forwarded to the underlying embedder.
    """

    def __init__(
            self,
            model_name: str,
            *,
            device: str, 
            trust_remote_code: bool = False, 
            callback_manager: BaseCallbackHandler = None,
            model_kwargs: dict[str, Any] = None,
        ):
        """Initialise a Hugging Face embedder.

        Parameters
        ----------
        model_name : str
            Name or path of the embedding model.
        device : str
            Device identifier (e.g., ``"cuda"``, ``"cpu"``, ``"mps"``).
        trust_remote_code : bool, optional
            Whether to allow custom model code from the Hugging Face Hub.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.
        model_kwargs : dict[str, Any] or None, optional
            Additional keyword arguments forwarded to the underlying embedder.
        """
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self.embedder = HuggingFaceEmbedding(
            model_name = model_name,
            trust_remote_code=trust_remote_code,
            device=device,
            callback_manager=callback_manager,
            model_kwargs=model_kwargs or {},
        )

    def get_embedder(self) -> LlamaIndexBaseEmbedding:
        """Return the underlying LlamaIndex embedding object.

        Returns
        -------
        LlamaIndexBaseEmbedding
            Wrapped LlamaIndex embedding instance.
        """
        return self.embedder

    @classmethod
    def from_config(
            cls,
            config_path: str,
            callback_manager: BaseCallbackHandler = None
        ) -> "HuggingFaceEmbedder":
        """Create a Hugging Face embedder from YAML configuration.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        HuggingFaceEmbedder
            An initialised embedder instance.

        Notes
        -----
        The YAML is expected to contain ``model_name`` and ``device`` keys, plus
        optional ``trust_remote_code`` and ``model_kwargs``.
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls.from_config_dict(cfg, callback_manager)

    @classmethod
    def from_config_dict(
            cls,
            config: Dict[str, Any],
            callback_manager: BaseCallbackHandler = None
        ) -> "HuggingFaceEmbedder":
        """Create a Hugging Face embedder from a configuration mapping.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration mapping.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        HuggingFaceEmbedder
            An initialised embedder instance.

        Raises
        ------
        KeyError
            If required keys (e.g., ``model_name`` or ``device``) are missing.
        """
        return cls(
            model_name=config["model_name"],
            device=config["device"],
            trust_remote_code=config.get("trust_remote_code", False),
            callback_manager=callback_manager,
            model_kwargs=config.get("model_kwargs", {}),
        )
    
class OpenAILikeEmbedder(BaseEmbedder):
    """Embedder backed by an OpenAI-compatible embedding API via LlamaIndex.

    This implementation wraps :class:`llama_index.embeddings.openai_like.OpenAILikeEmbedding`.

    Parameters
    ----------
    model_name : str
        Model identifier for the embedding endpoint.
    api_base : str
        Base URL for the OpenAI-compatible embedding API endpoint.
    callback_manager : BaseCallbackHandler, optional
        Optional callback handler for logging/telemetry/streaming.
    model_kwargs : dict[str, Any] or None, optional
        Additional keyword arguments forwarded to the underlying embedder.
    """

    def __init__(
            self,
            model_name: str, 
            *,
            api_base: str,
            api_bases: list[str] | tuple[str, ...] | None = None,
            failover_api_bases: list[str] | tuple[str, ...] | None = None,
            api_key: str = None,
            callback_manager: BaseCallbackHandler = None,
            model_kwargs: dict[str, Any] = None,
            timeout: float = 60.0,
            max_retries: int = 10,
            embed_batch_size: int = 10,
            num_workers: Optional[int] = None,
            reuse_client: bool = True,
            request_max_attempts: int = 3,
            request_base_backoff_seconds: float = 1.0,
        ):
        """Initialise an OpenAI-compatible embedder.

        Parameters
        ----------
        model_name : str
            Model identifier for the embedding endpoint.
        api_base : str
            Base URL for the OpenAI-compatible embedding API endpoint.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.
        model_kwargs : dict[str, Any] or None, optional
            Additional keyword arguments forwarded to the underlying embedder.
        """
        self.model_name = model_name
        self.api_bases = _normalize_api_base_sequence(
            primary_api_base=api_base,
            api_bases=api_bases,
            failover_api_bases=failover_api_bases,
        )
        self.api_base = self.api_bases[0]
        self._active_api_base_index = 0
        self.api_key = api_key or "fake"
        self.additional_kwargs = dict(model_kwargs or {})
        self.default_timeout_seconds = float(timeout)
        self.default_max_retries = int(max_retries)
        self.embed_batch_size = max(1, int(embed_batch_size))
        self.request_max_attempts = max(1, int(request_max_attempts))
        self.request_base_backoff_seconds = max(0.0, float(request_base_backoff_seconds))
        self.callback_manager = callback_manager
        self.num_workers = num_workers
        self.reuse_client = reuse_client
        self.embedder = self._build_llamaindex_embedder(api_base=self.api_base)

    def _build_llamaindex_embedder(self, *, api_base: str):
        from llama_index.embeddings.openai_like import OpenAILikeEmbedding

        return OpenAILikeEmbedding(
            model_name=self.model_name,
            api_base=api_base,
            callback_manager=self.callback_manager,
            additional_kwargs=self.additional_kwargs,
            api_key=self.api_key,
            timeout=self.default_timeout_seconds,
            max_retries=self.default_max_retries,
            embed_batch_size=self.embed_batch_size,
            num_workers=self.num_workers,
            reuse_client=self.reuse_client,
        )

    def _set_active_api_base(self, api_base: str) -> None:
        normalized = str(api_base or "").strip()
        if not normalized:
            raise ValueError("api_base must be non-empty.")
        if normalized == self.api_base:
            return
        if normalized not in self.api_bases:
            raise ValueError(f"api_base {normalized!r} is not in configured failover pool.")

        self.api_base = normalized
        self._active_api_base_index = self.api_bases.index(normalized)
        self.embedder = self._build_llamaindex_embedder(api_base=normalized)

    def _build_openai_client(self, *, api_base: str, timeout_seconds: float | None) -> OpenAIClient:
        client_timeout = self.default_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        return OpenAIClient(
            api_key=self.api_key,
            base_url=api_base,
            timeout=client_timeout,
            max_retries=max(0, int(self.default_max_retries)),
        )

    def _request_embeddings_once(
        self,
        *,
        api_base: str,
        inputs: list[str],
        timeout_seconds: float | None,
    ) -> list[list[float]]:
        client = self._build_openai_client(api_base=api_base, timeout_seconds=timeout_seconds)
        request_kwargs: dict[str, Any] = {}
        if self.additional_kwargs:
            request_kwargs["extra_body"] = dict(self.additional_kwargs)

        response = client.embeddings.create(
            model=self.model_name,
            input=inputs,
            timeout=self.default_timeout_seconds if timeout_seconds is None else float(timeout_seconds),
            **request_kwargs,
        )
        data = sorted(
            list(getattr(response, "data", None) or []),
            key=lambda item: int(getattr(item, "index", 0)),
        )
        vectors: list[list[float]] = []
        for item in data:
            embedding = getattr(item, "embedding", None)
            if isinstance(embedding, list):
                vectors.append([float(x) for x in embedding])
            else:
                vectors.append(list(embedding or []))
        return vectors

    def _embed_batch_with_failover(
        self,
        *,
        inputs: list[str],
        timeout_seconds: float | None,
    ) -> list[list[float]]:
        if not inputs:
            return []

        last_exc: Exception | None = None
        endpoint_count = len(self.api_bases)
        starting_index = self._active_api_base_index

        for endpoint_offset in range(endpoint_count):
            endpoint_index = (starting_index + endpoint_offset) % endpoint_count
            api_base = self.api_bases[endpoint_index]
            self._set_active_api_base(api_base)

            for attempt in range(1, self.request_max_attempts + 1):
                try:
                    return self._request_embeddings_once(
                        api_base=api_base,
                        inputs=inputs,
                        timeout_seconds=timeout_seconds,
                    )
                except (OpenAIAPITimeoutError, httpx.TimeoutException, TimeoutError) as exc:
                    last_exc = RetrievalTimeoutError(
                        f"embedding request timed out after "
                        f"{float(self.default_timeout_seconds if timeout_seconds is None else timeout_seconds):.3f}s"
                    )
                    retryable = True
                    wrapped_exc = exc
                except Exception as exc:  # pragma: no cover - exercised via helper in tests
                    last_exc = exc
                    retryable = _is_retryable_openai_embedding_error(exc)
                    wrapped_exc = exc

                if not retryable:
                    raise last_exc

                if attempt < self.request_max_attempts:
                    retry_delay = self.request_base_backoff_seconds * (2 ** (attempt - 1))
                    print(
                        f"Embedding request failed against {api_base} "
                        f"(attempt {attempt}/{self.request_max_attempts}): {wrapped_exc}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                    continue

                if endpoint_offset + 1 < endpoint_count:
                    next_api_base = self.api_bases[(endpoint_index + 1) % endpoint_count]
                    print(
                        f"Embedding endpoint {api_base} exhausted after {self.request_max_attempts} attempts. "
                        f"Failing over to {next_api_base}."
                    )

        if last_exc is None:
            raise RuntimeError("Embedding failover loop exited without a result or exception.")
        raise last_exc

    def get_embedder(self) -> LlamaIndexBaseEmbedding:
        """Return the underlying LlamaIndex embedding object.

        Returns
        -------
        LlamaIndexBaseEmbedding
            Wrapped LlamaIndex embedding instance.
        """
        return self.embedder

    def embed_query(self, query: str, timeout_seconds: float | None = None) -> list[float]:
        """Embed Query.
        
        Parameters
        ----------
        query : str
            User query text.
        timeout_seconds : float or None, optional
            timeout Seconds expressed in seconds.
        
        Returns
        -------
        list[float]
            Collected results from the operation.
        
        Raises
        ------
        RetrievalTimeoutError
            If `RetrievalTimeoutError` is raised while executing the operation.
        """
        vectors = self._embed_batch_with_failover(inputs=[query], timeout_seconds=timeout_seconds)
        return vectors[0] if vectors else []

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        if not documents:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(documents), self.embed_batch_size):
            batch = documents[start:start + self.embed_batch_size]
            vectors.extend(self._embed_batch_with_failover(inputs=batch, timeout_seconds=None))
        return vectors

    @classmethod
    def from_config(
            cls,
            config_path: str,
            callback_manager: BaseCallbackHandler = None
        ) -> "OpenAILikeEmbedder":
        """Create an OpenAI-compatible embedder from YAML configuration.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        OpenAILikeEmbedder
            An initialised embedder instance.

        Notes
        -----
        The YAML is expected to contain ``model_name`` and ``api_base`` keys, plus
        optional ``model_kwargs``.
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls.from_config_dict(cfg, callback_manager)

    @classmethod
    def from_config_dict(
            cls,
            config: Dict[str, Any],
            callback_manager: BaseCallbackHandler = None
        ) -> "OpenAILikeEmbedder":
        """Create an OpenAI-compatible embedder from a configuration mapping.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration mapping.
        callback_manager : BaseCallbackHandler, optional
            Optional callback handler for logging/telemetry/streaming.

        Returns
        -------
        OpenAILikeEmbedder
            An initialised embedder instance.

        Raises
        ------
        KeyError
            If required keys (e.g., ``model_name`` or ``api_base``) are missing.
        """
        return cls(
            model_name=config["model_name"],
            api_base=config["api_base"],
            api_bases=config.get("api_bases"),
            failover_api_bases=config.get("failover_api_bases") or config.get("backup_api_bases"),
            api_key=config.get("api_key"),
            callback_manager=callback_manager,
            model_kwargs=config.get("model_kwargs", {}),
            timeout=float(config.get("timeout", config.get("request_timeout", 60.0))),
            max_retries=int(config.get("max_retries", 10)),
            embed_batch_size=int(config.get("embed_batch_size", 10)),
            num_workers=config.get("num_workers"),
            reuse_client=_as_bool(config.get("reuse_client"), True),
            request_max_attempts=int(config.get("request_max_attempts", 3)),
            request_base_backoff_seconds=float(config.get("request_base_backoff_seconds", 1.0)),
        )
    

# ----------------- Factory helpers -----------------

def _get_embedder_kind(cfg: Mapping[str, Any]) -> str:
    """Extract the embedder kind/type/provider discriminator from a config mapping.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Configuration mapping.

    Returns
    -------
    str
        The first non-empty discriminator value found, or an empty string if none
        is present.
    """
    for key in ("kind", "type", "provider", "backend", "impl"):
        val = cfg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _normalize_embedder_kind(kind: str) -> str:
    """Normalise an embedder kind/type string to a stable registry key.

    Parameters
    ----------
    kind : str
        Provider/type discriminator value.

    Returns
    -------
    str
        Normalised registry key (e.g., ``"OpenAILike"`` -> ``"openai_like"``).

    Notes
    -----
    The normalisation process:
    - converts CamelCase to snake_case
    - replaces whitespace and hyphens with underscores
    - collapses repeated underscores
    - applies a small set of provider-specific aliases
    """
    k = kind.strip()
    if not k:
        return ""

    # Insert underscores between camel-case boundaries.
    out: list[str] = []
    prev = ""
    for ch in k:
        if prev and prev.islower() and ch.isupper():
            out.append("_")
        out.append(ch)
        prev = ch

    k2 = "".join(out)
    k2 = k2.replace("-", "_").replace(" ", "_")

    while "__" in k2:
        k2 = k2.replace("__", "_")

    k2 = k2.lower()

    k2 = k2.replace("openailike", "openai_like")
    k2 = k2.replace("open_ailike", "openai_like")
    k2 = k2.replace("open_ai_like", "openai_like")

    k2 = k2.replace("huggingfacetgi", "huggingface_tgi")
    k2 = k2.replace("huggingface_tgi", "huggingface_tgi")

    return k2


def create_embedder(
    config: dict,
    callback_manager: Optional[BaseCallbackHandler] = None,
) -> BaseEmbedder:
    """Create an embedder implementation from a configuration mapping.

    This is the preferred entry point for wiring embedders (used by the
    application container). The concrete implementation is selected by a
    discriminator field in the configuration (one of: ``kind``, ``type``,
    ``provider``, ``backend``, or ``impl``).

    Parameters
    ----------
    config : dict
        Configuration mapping used to construct the embedder.
    callback_manager : BaseCallbackHandler, optional
        Optional callback handler for logging/telemetry/streaming.

    Returns
    -------
    BaseEmbedder
        An initialised embedder implementation.

    Raises
    ------
    TypeError
        If ``config`` is not a mapping.
    ValueError
        If the discriminator selects an unsupported implementation.

    Notes
    -----
    If no discriminator is provided, the default implementation is
    :class:`~polaris_rag.retrieval.embedder.HuggingFaceEmbedder`.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"create_embedder expected a mapping/dict, got {type(config)}")

    kind_raw = _get_embedder_kind(config)
    kind = _normalize_embedder_kind(kind_raw)

    registry = {
        "huggingface": HuggingFaceEmbedder,
        "hf": HuggingFaceEmbedder,
        "openai_like": OpenAILikeEmbedder,
        "openai": OpenAILikeEmbedder,
        "openailike": OpenAILikeEmbedder,
    }

    cls = registry.get(kind) if kind else HuggingFaceEmbedder

    if cls is None:
        raise ValueError(
            f"Unknown embedder kind '{kind_raw}' (normalized to '{kind}'). "
            f"Supported kinds: {sorted(registry.keys())}."
        )

    # Prefer dict-based constructor
    if hasattr(cls, "from_config_dict"):
        return cls.from_config_dict(dict(config), callback_manager=callback_manager)

    # Very unlikely fallback
    raise TypeError(
        f"Embedder class {cls.__name__} does not implement from_config_dict"
    )

    raise TypeError(
        f"Embedder class {cls.__name__} does not implement from_config_dict/from_config"
    )


__all__ = [
    "BaseEmbedder",
    "HuggingFaceEmbedder",
    "OpenAILikeEmbedder",
    "create_embedder",
]
