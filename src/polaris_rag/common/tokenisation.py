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

    def split(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
        """Split ``text`` into overlapping windows of at most ``max_tokens``."""


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

    def split(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
        if not text:
            return []

        max_tokens = int(max_tokens)
        overlap_tokens = max(0, int(overlap_tokens))
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")

        cpt = max(1, int(self.chars_per_token))
        window_chars = max_tokens * cpt
        step_chars = max(1, (max_tokens - overlap_tokens) * cpt)

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(text_length, start + window_chars)
            if end < text_length:
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            next_start = max(0, end - (overlap_tokens * cpt))
            if next_start <= start:
                next_start = start + step_chars
            while next_start < text_length and text[next_start].isspace():
                next_start += 1
            start = next_start

        return chunks


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

    def split(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
        if not text:
            return []

        max_tokens = int(max_tokens)
        overlap_tokens = max(0, int(overlap_tokens))
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")

        toks = self._enc.encode(text)
        if not toks:
            return []

        step = max_tokens - overlap_tokens
        return [
            self._enc.decode(toks[start : start + max_tokens]).strip()
            for start in range(0, len(toks), step)
            if self._enc.decode(toks[start : start + max_tokens]).strip()
        ]


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

    def split(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
        if not text:
            return []

        max_tokens = int(max_tokens)
        overlap_tokens = max(0, int(overlap_tokens))
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")

        toks = self.tokenizer.encode(text)
        if not toks:
            return []

        step = max_tokens - overlap_tokens
        chunks: list[str] = []
        for start in range(0, len(toks), step):
            chunk = self.tokenizer.decode(toks[start : start + max_tokens]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


__all__ = [
    "TokenCounter",
    "HeuristicTokenCounter",
    "TiktokenTokenCounter",
    "HuggingFaceTokenCounter",
]
