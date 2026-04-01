"""Sparse encoder abstractions for hybrid dense+sparse retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SparseEmbedding:
    """Sparse embedding payload compatible with Qdrant sparse vectors."""

    indices: list[int]
    values: list[float]

    def is_empty(self) -> bool:
        """Return True when there are no non-zero coordinates."""
        return not self.indices or not self.values


class BaseSparseEncoder(ABC):
    """Abstract interface for sparse text encoders."""

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        """Encode a batch of texts into sparse vectors."""
        raise NotImplementedError

    def encode_query(self, text: str) -> SparseEmbedding:
        """Encode a single query into a sparse vector."""
        embeddings = self.encode_documents([text])
        return embeddings[0] if embeddings else SparseEmbedding(indices=[], values=[])

    @abstractmethod
    def profile(self) -> dict[str, Any]:
        """Return a stable JSON-serializable profile."""
        raise NotImplementedError


def _coerce_sparse_embedding(value: Any) -> SparseEmbedding:
    """Coerce provider output into a :class:`SparseEmbedding`."""
    indices = getattr(value, "indices", None)
    values = getattr(value, "values", None)
    if indices is None and isinstance(value, Mapping):
        indices = value.get("indices")
    if values is None and isinstance(value, Mapping):
        values = value.get("values")
    if indices is None or values is None:
        raise TypeError(
            "Sparse encoder outputs must expose 'indices' and 'values'. "
            f"Received {type(value)!r}."
        )
    return SparseEmbedding(
        indices=[int(item) for item in list(indices)],
        values=[float(item) for item in list(values)],
    )


class FastEmbedSparseEncoder(BaseSparseEncoder):
    """Sparse encoder backed by FastEmbed SPLADE models."""

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int = 32,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Iterable[str] | None = None,
    ) -> None:
        init_kwargs: dict[str, Any] = {"model_name": model_name}
        if cache_dir:
            init_kwargs["cache_dir"] = cache_dir
        if threads is not None:
            init_kwargs["threads"] = int(threads)
        if providers:
            init_kwargs["providers"] = list(providers)

        self.model_name = str(model_name)
        self.batch_size = max(1, int(batch_size))
        self._init_kwargs = init_kwargs
        self._encoder = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            try:
                from fastembed import SparseTextEmbedding  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "FastEmbedSparseEncoder requires the optional 'fastembed' dependency. "
                    "Install it before enabling sparse retrieval."
                ) from exc
            self._encoder = SparseTextEmbedding(**self._init_kwargs)
        return self._encoder

    def encode_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        if not texts:
            return []
        raw = self._get_encoder().embed(texts, batch_size=self.batch_size)
        return [_coerce_sparse_embedding(item) for item in raw]

    def profile(self) -> dict[str, Any]:
        return {
            "type": "fastembed",
            "model_name": self.model_name,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_config_dict(cls, config: Mapping[str, Any]) -> "FastEmbedSparseEncoder":
        providers = config.get("providers")
        return cls(
            model_name=str(config["model_name"]),
            batch_size=int(config.get("batch_size", 32)),
            cache_dir=str(config["cache_dir"]) if config.get("cache_dir") else None,
            threads=int(config["threads"]) if config.get("threads") is not None else None,
            providers=list(providers) if isinstance(providers, list) else None,
        )


def _get_sparse_encoder_kind(cfg: Mapping[str, Any]) -> str:
    for key in ("kind", "type", "provider", "backend", "impl"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_sparse_encoder_kind(kind: str) -> str:
    normalized = kind.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.replace("fast_embed", "fastembed")


def create_sparse_encoder(config: Mapping[str, Any] | None) -> BaseSparseEncoder | None:
    """Create a sparse encoder or return ``None`` when disabled."""
    if not isinstance(config, Mapping):
        return None
    if config.get("enabled") is False:
        return None

    kind_raw = _get_sparse_encoder_kind(config) or "fastembed"
    kind = _normalize_sparse_encoder_kind(kind_raw)
    registry = {
        "fastembed": FastEmbedSparseEncoder,
        "splade": FastEmbedSparseEncoder,
    }
    cls = registry.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown sparse encoder kind '{kind_raw}' (normalized to '{kind}'). "
            f"Supported kinds: {sorted(registry.keys())}."
        )
    return cls.from_config_dict(config)


__all__ = [
    "BaseSparseEncoder",
    "FastEmbedSparseEncoder",
    "SparseEmbedding",
    "create_sparse_encoder",
]
