"""Markdown-first token chunking utilities."""

from __future__ import annotations

from typing import Iterable

from polaris_rag.common import DocumentChunk, MarkdownDocument
from polaris_rag.common.tokenisation import TokenCounter
from polaris_rag.retrieval.ingestion_settings import MARKDOWN_TOKEN_CHUNKING_STRATEGY


def _markdown_chunk_id(parent_id: str, chunk_index: int) -> str:
    return f"{parent_id}::chunk::{int(chunk_index):04d}"


class MarkdownTokenChunker:
    """Naive token-window chunker for markdown-normalized documents."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        chunk_size_tokens: int = 800,
        overlap_tokens: int = 80,
    ) -> None:
        self.token_counter = token_counter
        self.chunk_size_tokens = int(chunk_size_tokens)
        self.overlap_tokens = max(0, int(overlap_tokens))
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be > 0")
        if self.overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    def chunk(self, document: MarkdownDocument) -> list[DocumentChunk]:
        windows = self.token_counter.split(
            document.text,
            max_tokens=self.chunk_size_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        if not windows:
            return []

        chunks: list[DocumentChunk] = []
        for chunk_index, text in enumerate(windows):
            metadata = dict(document.metadata or {})
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunking_strategy": MARKDOWN_TOKEN_CHUNKING_STRATEGY,
                    "token_count": int(self.token_counter.count(text)),
                    "overlap_tokens": self.overlap_tokens if chunk_index > 0 else 0,
                }
            )
            chunk_id = _markdown_chunk_id(document.id, chunk_index)
            chunks.append(
                DocumentChunk(
                    parent_id=document.id,
                    prev_id=chunks[-1].id if chunks else None,
                    next_id=None,
                    text=text,
                    document_type=document.document_type,
                    id=chunk_id,
                    metadata=metadata,
                    source_node=document,
                )
            )
            if len(chunks) > 1:
                chunks[-2].next_id = chunk_id

        return chunks


def get_chunks_from_markdown_documents(
    documents: Iterable[MarkdownDocument],
    *,
    token_counter: TokenCounter,
    chunk_size: int = 800,
    overlap: int = 80,
) -> list[DocumentChunk]:
    chunker = MarkdownTokenChunker(
        token_counter=token_counter,
        chunk_size_tokens=chunk_size,
        overlap_tokens=overlap,
    )
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(document))
    return chunks


__all__ = [
    "MarkdownTokenChunker",
    "get_chunks_from_markdown_documents",
]
