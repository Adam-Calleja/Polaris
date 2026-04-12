"""Shared Jira ingestion helpers.

This module centralises the common batch-processing steps used by Jira ticket
ingestion flows so both the standalone CLI and experiment-stage fanout logic
can reuse the same filtering, preprocessing, chunking, dump, and target-reset
behaviour.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

from polaris_rag.retrieval.ingestion_settings import (
    JIRA_TURNS_TOKEN_CHUNKING_STRATEGY,
    MARKDOWN_TOKEN_CHUNKING_STRATEGY,
)
from polaris_rag.retrieval.markdown_chunker import get_chunks_from_markdown_documents
from polaris_rag.retrieval.markdown_converter import convert_tickets_to_markdown
from polaris_rag.retrieval.metadata_enricher import enrich_documents_with_authority_metadata


def clear_persist_dir(persist_dir: str) -> None:
    """Remove the resolved persist directory and any previously persisted state."""
    target = Path(persist_dir).expanduser().resolve()
    if not target.exists():
        return
    if target == Path(target.anchor):
        raise RuntimeError(f"Refusing to remove filesystem root as persist dir: {target}")
    if target.is_dir():
        shutil.rmtree(target)
        return
    target.unlink()


def filter_jira_tickets(
    tickets: Sequence[dict[str, Any]],
    *,
    exclude_keys: Sequence[str],
    unwanted_summaries: Sequence[str],
) -> list[dict[str, Any]]:
    """Apply CLI/config exclusion rules to a fetched Jira batch."""
    filtered = list(tickets)
    if exclude_keys:
        excluded_key_set = {str(key).upper() for key in exclude_keys}
        filtered = [
            ticket for ticket in filtered if str(ticket.get("key", "")).upper() not in excluded_key_set
        ]

    if unwanted_summaries:
        filtered = [
            ticket
            for ticket in filtered
            if not any(unwanted in ticket.get("fields", {}).get("summary", "") for unwanted in unwanted_summaries)
        ]

    return filtered


def prepare_jira_tickets_for_chunking(
    tickets: Sequence[dict[str, Any]],
    *,
    chunking_strategy: str,
    conversion_engine: str | None,
    conversion_options: dict[str, Any] | None,
    registry_artifact_path: str | None,
    source_name: str,
) -> list[Any]:
    """Convert raw Jira API payloads into chunk-ready documents."""
    if chunking_strategy == MARKDOWN_TOKEN_CHUNKING_STRATEGY:
        processed_tickets = convert_tickets_to_markdown(
            list(tickets),
            engine=conversion_engine,
            options=conversion_options,
        )
    elif chunking_strategy == JIRA_TURNS_TOKEN_CHUNKING_STRATEGY:
        from polaris_rag.retrieval.document_preprocessor import preprocess_jira_tickets

        processed_tickets = preprocess_jira_tickets(list(tickets))
    else:
        raise ValueError(f"Unsupported ticket chunking strategy: {chunking_strategy!r}")

    return enrich_documents_with_authority_metadata(
        processed_tickets,
        registry_artifact_path=registry_artifact_path,
        source_name=source_name,
    )


def chunk_processed_jira_tickets(
    processed_tickets: Sequence[Any],
    *,
    chunking_strategy: str,
    token_counter: Any,
    chunk_size: int,
    overlap: int,
) -> list[Any]:
    """Generate chunks from preprocessed Jira ticket documents."""
    if chunking_strategy == MARKDOWN_TOKEN_CHUNKING_STRATEGY:
        return get_chunks_from_markdown_documents(
            list(processed_tickets),
            token_counter=token_counter,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    if chunking_strategy == JIRA_TURNS_TOKEN_CHUNKING_STRATEGY:
        from polaris_rag.retrieval.text_splitter import get_chunks_from_jira_tickets

        return get_chunks_from_jira_tickets(
            tickets=list(processed_tickets),
            token_counter=token_counter,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    raise ValueError(f"Unsupported ticket chunking strategy: {chunking_strategy!r}")


def dump_processed_tickets(processed_tickets: Sequence[Any], dump_path: Path) -> None:
    """Append processed Jira ticket text to a debug dump file."""
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\n\n" + ("-" * 10) + "\n\n"
    with dump_path.open("a", encoding="utf-8") as handle:
        for ticket in processed_tickets:
            handle.write(getattr(ticket, "text", ""))
            handle.write(sep)


__all__ = [
    "chunk_processed_jira_tickets",
    "clear_persist_dir",
    "dump_processed_tickets",
    "filter_jira_tickets",
    "prepare_jira_tickets_for_chunking",
]
