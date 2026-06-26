from polaris_rag.common import MarkdownDocument
from polaris_rag.retrieval.markdown_chunker import MarkdownTokenChunker


class DummyTokenCounter:
    def count(self, text: str) -> int:
        return len((text or "").split())

    def tail(self, text: str, n_tokens: int) -> str:
        tokens = (text or "").split()
        return " ".join(tokens[-n_tokens:])

    def split(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
        tokens = (text or "").split()
        if not tokens:
            return []
        step = max_tokens - overlap_tokens
        return [
            " ".join(tokens[start : start + max_tokens])
            for start in range(0, len(tokens), step)
            if tokens[start : start + max_tokens]
        ]


def test_markdown_token_chunker_builds_deterministic_linked_chunks():
    document = MarkdownDocument(
        id="https://docs.example.org/guide",
        document_type="html",
        text="one two three four five six seven eight nine ten",
        metadata={"source": "https://docs.example.org/guide"},
    )

    chunker = MarkdownTokenChunker(
        token_counter=DummyTokenCounter(),
        chunk_size_tokens=4,
        overlap_tokens=1,
    )

    chunks = chunker.chunk(document)

    assert [chunk.id for chunk in chunks] == [
        "https://docs.example.org/guide::chunk::0000",
        "https://docs.example.org/guide::chunk::0001",
        "https://docs.example.org/guide::chunk::0002",
        "https://docs.example.org/guide::chunk::0003",
    ]
    assert chunks[0].prev_id is None
    assert chunks[0].next_id == chunks[1].id
    assert chunks[1].prev_id == chunks[0].id
    assert chunks[1].next_id == chunks[2].id
    assert chunks[-1].next_id is None
    assert chunks[0].metadata["chunking_strategy"] == "markdown_token"
    assert chunks[0].metadata["token_count"] == 4
    assert chunks[1].metadata["overlap_tokens"] == 1


def test_markdown_token_chunker_skips_empty_documents():
    document = MarkdownDocument(
        id="ticket-1",
        document_type="helpdesk_ticket",
        text="",
        metadata={"ticket_key": "ticket-1"},
    )

    chunker = MarkdownTokenChunker(
        token_counter=DummyTokenCounter(),
        chunk_size_tokens=10,
        overlap_tokens=2,
    )

    assert chunker.chunk(document) == []
