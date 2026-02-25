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
from langchain_core.callbacks import BaseCallbackHandler
from llama_index.core.base.embeddings.base import BaseEmbedding as LlamaIndexBaseEmbedding
from typing import Any, Dict, Mapping, Optional
import yaml
import asyncio


def _as_bool(value: Any, default: bool) -> bool:
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


class BaseEmbedder(ABC):
    """Abstract interface for text embedding.

    Concrete implementations wrap a provider-specific embedder and expose a small,
    consistent API used by the Polaris retrieval pipeline.
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

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        This method attempts to call common embedding method names on the
        underlying embedder in the following order:
        ``embed_query``, ``get_text_embedding``, ``embed_documents``, ``embed``.

        Parameters
        ----------
        query : str
            Query string to embed.

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
            api_key: str = None,
            callback_manager: BaseCallbackHandler = None,
            model_kwargs: dict[str, Any] = None,
            timeout: float = 60.0,
            max_retries: int = 10,
            embed_batch_size: int = 10,
            num_workers: Optional[int] = None,
            reuse_client: bool = True,
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
        from llama_index.embeddings.openai_like import OpenAILikeEmbedding

        self.embedder = OpenAILikeEmbedding(
            model_name=model_name,
            api_base=api_base,
            callback_manager=callback_manager,
            additional_kwargs=model_kwargs or {},
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            embed_batch_size=embed_batch_size,
            num_workers=num_workers,
            reuse_client=reuse_client,
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
            api_key=config.get("api_key"),
            callback_manager=callback_manager,
            model_kwargs=config.get("model_kwargs", {}),
            timeout=float(config.get("timeout", config.get("request_timeout", 60.0))),
            max_retries=int(config.get("max_retries", 10)),
            embed_batch_size=int(config.get("embed_batch_size", 10)),
            num_workers=config.get("num_workers"),
            reuse_client=_as_bool(config.get("reuse_client"), True),
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
