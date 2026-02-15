"""polaris_rag.common.tokenisation

Token counting utilities.

This module provides a small abstraction used by chunkers and splitters to
size chunks by *token count* without coupling those components to any
particular LLM provider or tokenizer library.

The core idea is to depend only on a minimal token-counting interface,
allowing the concrete implementation to be selected via configuration.

Classes
-------
TokenCounter
    Minimal protocol defining the token-counting interface.
HeuristicTokenCounter
    Lightweight, dependency-free approximate token counter.
TiktokenTokenCounter
    Exact token counter backed by the ``tiktoken`` library.
HuggingFaceTokenCounter
    Token counter backed by a Hugging Face tokenizer instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class TokenCounter(Protocol):
    """A minimal interface for token-based sizing.

    Implementations provide a consistent way to estimate or compute the number
    of tokens in a string, and to extract the trailing portion of text by token
    count.
    """

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""

    def tail(self, text: str, n_tokens: int) -> str:
        """Return the last ``n_tokens`` tokens of ``text`` as text."""


@dataclass(frozen=True)
class HeuristicTokenCounter:
    """Dependency-free, approximate token counter.

    This implementation estimates token counts using a fixed
    characters-per-token ratio. While not exact, it is often sufficient for
    chunk sizing and avoids heavyweight tokenizer dependencies.

    Attributes
    ----------
    chars_per_token : int
        Approximate number of characters per token. Defaults to ``4``.
    """

    chars_per_token: int = 4

    def count(self, text: str) -> int:
        if not text:
            return 0
        cpt = max(1, int(self.chars_per_token))
        return max(1, len(text) // cpt)

    def tail(self, text: str, n_tokens: int) -> str:
        if not text or n_tokens <= 0:
            return ""
        cpt = max(1, int(self.chars_per_token))
        return text[-(n_tokens * cpt) :]


@dataclass(frozen=True)
class TiktokenTokenCounter:
    """Token counter backed by the ``tiktoken`` library.

    This implementation provides exact token counts for encodings supported
    by ``tiktoken``.

    Attributes
    ----------
    encoding_name : str
        Name of the ``tiktoken`` encoding.
    _enc : Any
        Internal ``tiktoken`` encoding object.
    """

    encoding_name: str
    _enc: Any

    @classmethod
    def from_encoding_name(cls, encoding_name: str) -> "TiktokenTokenCounter":
        """Construct a token counter from an encoding name.

        Parameters
        ----------
        encoding_name : str
            Name of the ``tiktoken`` encoding to load.

        Returns
        -------
        TiktokenTokenCounter
            A token counter initialised with the requested encoding.
        """
        import tiktoken # type: ignore

        enc = tiktoken.get_encoding(encoding_name)
        return cls(encoding_name=encoding_name, _enc=enc)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    def tail(self, text: str, n_tokens: int) -> str:
        if not text or n_tokens <= 0:
            return ""
        toks = self._enc.encode(text)
        if not toks:
            return ""
        return self._enc.decode(toks[-n_tokens:])


@dataclass(frozen=True)
class HuggingFaceTokenCounter:
    """Token counter backed by a Hugging Face tokenizer.

    This implementation delegates tokenisation to an externally constructed
    tokenizer instance (e.g., from ``transformers.AutoTokenizer``). The
    ``transformers`` library is not imported at module import time.

    Attributes
    ----------
    tokenizer : Any
        Hugging Face tokenizer instance used for encoding and decoding.
    """

    tokenizer: Any

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def tail(self, text: str, n_tokens: int) -> str:
        if not text or n_tokens <= 0:
            return ""
        toks = self.tokenizer.encode(text)
        if not toks:
            return ""
        return self.tokenizer.decode(toks[-n_tokens:])


__all__ = [
    "TokenCounter",
    "HeuristicTokenCounter",
    "TiktokenTokenCounter",
    "HuggingFaceTokenCounter",
]