"""Jira ticket ingestion entrypoint."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return Path.cwd()


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.app.container import build_container
from polaris_rag.config import GlobalConfig
from polaris_rag.retrieval.document_store_factory import (
    add_chunks_to_docstore,
    add_documents_to_docstore,
    build_storage_context,
    delete_ref_docs_from_docstore,
    load_or_create_source_document_store,
    persist_docstore,
    persist_storage,
    source_document_store_path,
)
from polaris_rag.retrieval.ingestion_settings import (
    JIRA_TURNS_TOKEN_CHUNKING_STRATEGY,
    MARKDOWN_TOKEN_CHUNKING_STRATEGY,
    resolve_chunking_settings,
    resolve_conversion_settings,
)
from polaris_rag.retrieval.metadata_enricher import (
    enrich_documents_with_authority_metadata,
    resolve_authority_registry_artifact_path,
)
from polaris_rag.retrieval.markdown_chunker import get_chunks_from_markdown_documents
from polaris_rag.retrieval.markdown_converter import convert_tickets_to_markdown


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Jira tickets into the RAG stores")

    parser.add_argument(
        "--config-file",
        "-c",
        required=True,
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--persist-dir",
        "-d",
        required=False,
        type=str,
        default=None,
        help="Override persist dir from config (optional)",
    )
    parser.add_argument(
        "--start-date",
        "-s",
        required=False,
        type=str,
        default=None,
        help="Start date for fetching Jira tickets (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        "-e",
        required=False,
        type=str,
        default=None,
        help="End date for fetching Jira tickets (YYYY-MM-DD, exclusive upper bound). Defaults to today when unset.",
    )
    parser.add_argument(
        "--limit",
        "-l",
        required=False,
        type=int,
        default=None,
        help="Maximum number of tickets to fetch (optional; defaults to no limit).",
    )
    parser.add_argument(
        "--exclude-keys-file",
        required=False,
        type=str,
        default=None,
        help="Path to newline-delimited Jira keys to exclude (optional; no default exclusion file is applied).",
    )
    parser.add_argument(
        "--qdrant-collection-name",
        required=False,
        type=str,
        default=None,
        help="Override selected source collection name from config (optional).",
    )
    parser.add_argument(
        "--source",
        required=False,
        type=str,
        default="tickets",
        help="Named source in config.vector_stores to ingest into (default: tickets).",
    )
    parser.add_argument(
        "--vector-batch-size",
        "-b",
        required=False,
        type=int,
        default=16,
        help="Batch size for vector-store inserts (default: 16).",
    )
    parser.add_argument(
        "--embedding-workers",
        required=False,
        type=int,
        default=None,
        help=(
            "Enable concurrent embedding requests by setting embedder.num_workers. "
            "Values > 1 use async insertion with parallel embedding batches."
        ),
    )
    parser.add_argument(
        "--dump-processed",
        action="store_true",
        help="If set, dump processed ticket text to --dump-path",
    )
    parser.add_argument(
        "--dump-path",
        type=str,
        default=None,
        help="Path to write processed ticket dump (optional; defaults to data/debug/jira_processed_tickets.txt)",
    )
    parser.add_argument(
        "--conversion-engine",
        required=False,
        type=str,
        default=None,
        help="Override the markdown conversion engine for this source (optional).",
    )
    parser.add_argument(
        "--chunking-strategy",
        required=False,
        type=str,
        default=None,
        help="Override the chunking strategy for this source (optional).",
    )
    parser.add_argument(
        "--chunk-size-tokens",
        required=False,
        type=int,
        default=None,
        help="Override the markdown token chunk size (optional).",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        required=False,
        type=int,
        default=None,
        help="Override the markdown token overlap (optional).",
    )

    return parser.parse_args()


def _get_ingestion_cfg(cfg: GlobalConfig) -> Mapping[str, Any]:
    ingestion = getattr(cfg, "ingestion", None)
    return _as_mapping(ingestion)


def _get_jira_ingestion_cfg(cfg: GlobalConfig) -> Mapping[str, Any]:
    ingestion = _get_ingestion_cfg(cfg)
    jira = ingestion.get("jira") if isinstance(ingestion, Mapping) else None
    return jira if isinstance(jira, Mapping) else {}


def _resolve_persist_dir(cfg: GlobalConfig, cli_value: str | None) -> str:
    if cli_value:
        return cli_value

    sc = getattr(cfg, "storage_context", None)
    if sc is not None:
        if hasattr(sc, "persist_dir") and getattr(sc, "persist_dir"):
            return str(getattr(sc, "persist_dir"))
        sc_map = _as_mapping(sc)
        if sc_map.get("persist_dir"):
            return str(sc_map["persist_dir"])

    return str(REPO_ROOT / "data" / "storage" / "local")


def _resolve_dates(cfg: GlobalConfig, start_cli: str | None, end_cli: str | None) -> tuple[str, str]:
    jira_cfg = _get_jira_ingestion_cfg(cfg)
    today = datetime.now().strftime("%Y-%m-%d")

    start = start_cli or jira_cfg.get("start_date") or "2024-01-01"
    end = end_cli or jira_cfg.get("end_date") or today
    return str(start), str(end)


def _resolve_limit(cfg: GlobalConfig, limit_cli: int | None) -> int | None:
    if limit_cli is not None:
        return limit_cli
    jira_cfg = _get_jira_ingestion_cfg(cfg)
    lim = jira_cfg.get("limit")
    return int(lim) if lim is not None else None


def _resolve_unwanted_summaries(cfg: GlobalConfig) -> list[str]:
    jira_cfg = _get_jira_ingestion_cfg(cfg)
    summaries = jira_cfg.get("exclude_summaries")
    if isinstance(summaries, list) and all(isinstance(x, str) for x in summaries):
        return summaries

    return [
        "Internal HPC Application Form",
        "Self-Service Gateway",
        "SSH public key",
    ]


def _read_exclude_keys_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    keys: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            candidate = line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            keys.append(candidate.upper())
    return list(dict.fromkeys(keys))


def _resolve_exclude_keys(cfg: GlobalConfig, cli_value: str | None) -> list[str]:
    jira_cfg = _get_jira_ingestion_cfg(cfg)
    cfg_inline_keys = jira_cfg.get("exclude_keys")
    cfg_file = jira_cfg.get("exclude_keys_file")

    keys: list[str] = []
    if isinstance(cfg_inline_keys, list):
        keys.extend([key.upper() for key in cfg_inline_keys if isinstance(key, str) and key.strip()])

    exclude_file = cli_value or cfg_file
    if exclude_file:
        candidate = Path(exclude_file)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        keys.extend(_read_exclude_keys_file(candidate))

    return list(dict.fromkeys(keys))


def _resolve_embedding_workers(cfg: GlobalConfig, cli_value: int | None) -> int | None:
    if cli_value is not None:
        return int(cli_value)

    embedder_cfg = _as_mapping(getattr(cfg, "embedder", None))
    workers = embedder_cfg.get("num_workers")
    if workers is None:
        return None
    try:
        return int(workers)
    except (TypeError, ValueError):
        return None


def _dump_processed_tickets(processed_tickets: list[Any], dump_path: Path) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\n\n" + ("-" * 10) + "\n\n"
    with dump_path.open("a", encoding="utf-8") as handle:
        for ticket in processed_tickets:
            handle.write(getattr(ticket, "text", ""))
            handle.write(sep)


def _override_qdrant_collection_name(cfg: GlobalConfig, source: str, cli_value: str | None) -> None:
    if not cli_value:
        return

    vector_stores = cfg.raw.get("vector_stores")
    if not isinstance(vector_stores, dict):
        raise TypeError("'vector_stores' config must be a mapping.")

    selected_store = vector_stores.get(source)
    if not isinstance(selected_store, dict):
        raise KeyError(f"Missing 'vector_stores.{source}' config; cannot override collection name.")

    selected_store["collection_name"] = cli_value


def _build_source_storage_context(container: Any, source: str) -> Any:
    stores = container.vector_stores
    if source not in stores:
        raise KeyError(f"Unknown source {source!r}. Available sources: {sorted(stores.keys())}")

    return build_storage_context(
        vector_store=stores[source],
        docstore=container.doc_store,
        persist_dir=None,
    )


def main() -> None:
    args = parse_args()
    from polaris_rag.retrieval.document_loader import load_support_tickets
    from polaris_rag.retrieval.document_preprocessor import preprocess_jira_tickets
    from polaris_rag.retrieval.text_splitter import get_chunks_from_jira_tickets

    cfg = GlobalConfig.load(args.config_file)
    if args.embedding_workers is not None:
        if args.embedding_workers < 1:
            raise ValueError("--embedding-workers must be >= 1 when provided.")
        embedder_cfg = cfg.raw.get("embedder")
        if not isinstance(embedder_cfg, dict):
            embedder_cfg = {}
            cfg.raw["embedder"] = embedder_cfg
        embedder_cfg["num_workers"] = int(args.embedding_workers)

    _override_qdrant_collection_name(cfg, args.source, args.qdrant_collection_name)

    container = build_container(cfg)

    persist_dir = _resolve_persist_dir(cfg, args.persist_dir)
    start_date, end_date = _resolve_dates(cfg, args.start_date, args.end_date)
    limit = _resolve_limit(cfg, args.limit)
    unwanted_summaries = _resolve_unwanted_summaries(cfg)
    exclude_keys = _resolve_exclude_keys(cfg, args.exclude_keys_file)

    dump_path = Path(args.dump_path) if args.dump_path else (REPO_ROOT / "data" / "debug" / "jira_processed_tickets.txt")
    storage_context = _build_source_storage_context(container, args.source)

    print("Loading Jira tickets...")
    tickets = load_support_tickets(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        cfg=cfg,
        exclude_keys=exclude_keys,
    )
    print(f"Fetched {len(tickets)} tickets from Jira.")

    if exclude_keys:
        excluded_key_set = set(exclude_keys)
        tickets = [ticket for ticket in tickets if str(ticket.get("key", "")).upper() not in excluded_key_set]

    if unwanted_summaries:
        tickets = [
            ticket
            for ticket in tickets
            if not any(unwanted in ticket.get("fields", {}).get("summary", "") for unwanted in unwanted_summaries)
        ]

    chunking_settings = resolve_chunking_settings(
        cfg,
        source=args.source,
        strategy_override=args.chunking_strategy,
        chunk_size_override=args.chunk_size_tokens,
        overlap_override=args.chunk_overlap_tokens,
    )
    conversion_settings = resolve_conversion_settings(
        cfg,
        source=args.source,
        engine_override=args.conversion_engine,
    )
    registry_artifact_path = resolve_authority_registry_artifact_path(cfg)

    print(f"Loaded {len(tickets)} tickets. Preprocessing...")
    if chunking_settings.strategy == MARKDOWN_TOKEN_CHUNKING_STRATEGY:
        processed_tickets = convert_tickets_to_markdown(
            tickets,
            engine=conversion_settings.engine,
            options=conversion_settings.options,
        )
        processed_tickets = enrich_documents_with_authority_metadata(
            processed_tickets,
            registry_artifact_path=registry_artifact_path,
            source_name=args.source,
        )
        chunks = get_chunks_from_markdown_documents(
            processed_tickets,
            token_counter=container.token_counter,
            chunk_size=chunking_settings.chunk_size_tokens,
            overlap=chunking_settings.overlap_tokens,
        )
    elif chunking_settings.strategy == JIRA_TURNS_TOKEN_CHUNKING_STRATEGY:
        processed_tickets = preprocess_jira_tickets(tickets)
        processed_tickets = enrich_documents_with_authority_metadata(
            processed_tickets,
            registry_artifact_path=registry_artifact_path,
            source_name=args.source,
        )
        chunks = get_chunks_from_jira_tickets(
            tickets=processed_tickets,
            token_counter=container.token_counter,
            chunk_size=chunking_settings.chunk_size_tokens,
            overlap=chunking_settings.overlap_tokens,
        )
    else:
        raise ValueError(f"Unsupported ticket chunking strategy: {chunking_settings.strategy!r}")

    ticket_ids = list(dict.fromkeys(str(ticket.id) for ticket in processed_tickets if getattr(ticket, "id", None)))

    if args.dump_processed:
        print(f"Dumping processed tickets to: {dump_path}")
        _dump_processed_tickets(processed_tickets, dump_path)
    print("Generating chunks from tickets...")
    source_document_store = load_or_create_source_document_store(persist_dir=persist_dir)

    if ticket_ids:
        print(f"Removing existing chunks for {len(ticket_ids)} tickets...")
        if hasattr(storage_context.vector_store, "delete_ref_docs"):
            storage_context.vector_store.delete_ref_docs(ticket_ids)
        elif hasattr(storage_context.vector_store, "delete_ref_doc"):
            for ticket_id in ticket_ids:
                storage_context.vector_store.delete_ref_doc(ticket_id)
        delete_ref_docs_from_docstore(storage_context.docstore, ticket_ids)

    print("Adding chunks to vector store...")
    embedding_workers = _resolve_embedding_workers(cfg, args.embedding_workers)
    use_async_embeddings = embedding_workers is not None and embedding_workers > 1
    vector_batch_size = max(1, int(args.vector_batch_size))
    total_chunks = len(chunks)
    mode = "async/concurrent" if use_async_embeddings else "sync/sequential"
    print(
        f"Embedding/indexing {total_chunks} chunks "
        f"(insert mode: {mode}, workers: {embedding_workers or 1}, batch size: {vector_batch_size})..."
    )
    storage_context.vector_store.insert_chunks(
        chunks,
        batch_size=vector_batch_size,
        use_async=use_async_embeddings,
    )

    print("Adding chunks to document store...")
    add_chunks_to_docstore(storage=storage_context, chunks=chunks)

    print("Persisting full tickets to source document store...")
    add_documents_to_docstore(source_document_store, processed_tickets)

    print("Persisting storage context...")
    persist_storage(storage=storage_context, persist_dir=persist_dir)
    persist_docstore(
        source_document_store,
        persist_path=source_document_store_path(persist_dir),
    )

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
