"""Jira ticket ingestion entrypoint.

This module exposes public helper functions used by the surrounding Polaris subsystem.

Functions
---------
parse_args
    Parse args.
main
    Run the command-line entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def _find_repo_root(start: Path) -> Path:
    """Find Repo Root.
    
    Parameters
    ----------
    start : Path
        Value for start.
    
    Returns
    -------
    Path
        Result of the operation.
    """
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
    chunk_document_store_path,
    delete_ref_docs_from_docstore,
    load_or_create_chunk_document_store,
    load_or_create_source_document_store,
    persist_docstore,
    source_document_store_path,
)
from polaris_rag.retrieval.ingestion_settings import (
    resolve_chunking_settings,
    resolve_conversion_settings,
)
from polaris_rag.retrieval.jira_ingestion import (
    chunk_processed_jira_tickets,
    clear_persist_dir,
    dump_processed_tickets,
    filter_jira_tickets,
    prepare_jira_tickets_for_chunking,
)
from polaris_rag.retrieval.metadata_enricher import (
    resolve_authority_registry_artifact_path,
)


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    """As Mapping.
    
    Parameters
    ----------
    obj : Any
        Value for obj.
    
    Returns
    -------
    Mapping[str, Any]
        Result of the operation.
    """
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def parse_args() -> argparse.Namespace:
    """Parse args.
    
    Returns
    -------
    argparse.Namespace
        Parsed args.
    """
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
        "--fetch-batch-size",
        required=False,
        type=int,
        default=None,
        help="Maximum number of Jira tickets to process per ingestion batch (optional).",
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
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Create the target Qdrant collection and exit without loading or indexing tickets.",
    )
    parser.add_argument(
        "--clear-collection",
        action="store_true",
        help=(
            "Delete and recreate the target Qdrant collection and remove the target persist dir "
            "before ingestion begins."
        ),
    )

    return parser.parse_args()


def _get_ingestion_cfg(cfg: GlobalConfig) -> Mapping[str, Any]:
    """Return ingestion Cfg.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    
    Returns
    -------
    Mapping[str, Any]
        Result of the operation.
    """
    ingestion = getattr(cfg, "ingestion", None)
    return _as_mapping(ingestion)


def _get_jira_ingestion_cfg(cfg: GlobalConfig) -> Mapping[str, Any]:
    """Return jira Ingestion Cfg.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    
    Returns
    -------
    Mapping[str, Any]
        Result of the operation.
    """
    ingestion = _get_ingestion_cfg(cfg)
    jira = ingestion.get("jira") if isinstance(ingestion, Mapping) else None
    return jira if isinstance(jira, Mapping) else {}


def _resolve_persist_dir(cfg: GlobalConfig, cli_value: str | None) -> str:
    """Resolve persist Dir.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    cli_value : str or None, optional
        Optional value provided via the command line.
    
    Returns
    -------
    str
        Resulting string value.
    """
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
    """Resolve dates.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    start_cli : str or None, optional
        Value for start CLI.
    end_cli : str or None, optional
        Value for end CLI.
    
    Returns
    -------
    tuple[str, str]
        Collected results from the operation.
    """
    jira_cfg = _get_jira_ingestion_cfg(cfg)
    today = datetime.now().strftime("%Y-%m-%d")

    start = start_cli or jira_cfg.get("start_date") or "2024-01-01"
    end = end_cli or jira_cfg.get("end_date") or today
    return str(start), str(end)


def _resolve_limit(cfg: GlobalConfig, limit_cli: int | None) -> int | None:
    """Resolve limit.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    limit_cli : int or None, optional
        Value for limit CLI.
    
    Returns
    -------
    int or None
        Result of the operation.
    """
    if limit_cli is not None:
        return limit_cli
    jira_cfg = _get_jira_ingestion_cfg(cfg)
    lim = jira_cfg.get("limit")
    return int(lim) if lim is not None else None


def _resolve_unwanted_summaries(cfg: GlobalConfig) -> list[str]:
    """Resolve unwanted Summaries.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
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
    """Read exclude Keys File.
    
    Parameters
    ----------
    path : Path
        Filesystem path used by the operation.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
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
    """Resolve exclude Keys.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    cli_value : str or None, optional
        Optional value provided via the command line.
    
    Returns
    -------
    list[str]
        Collected results from the operation.
    """
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
    """Resolve embedding Workers.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    cli_value : int or None, optional
        Optional value provided via the command line.
    
    Returns
    -------
    int or None
        Result of the operation.
    """
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


def _resolve_fetch_batch_size(cfg: GlobalConfig, cli_value: int | None) -> int | None:
    """Resolve Jira fetch/process batch size."""
    if cli_value is not None:
        if cli_value < 1:
            raise ValueError("--fetch-batch-size must be >= 1 when provided.")
        return int(cli_value)

    jira_cfg = _get_jira_ingestion_cfg(cfg)
    value = jira_cfg.get("fetch_batch_size")
    if value is None:
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ingestion.jira.fetch_batch_size must be an integer when provided.") from exc
    if resolved < 1:
        raise ValueError("ingestion.jira.fetch_batch_size must be >= 1 when provided.")
    return resolved


def _filter_tickets(
    tickets: list[dict[str, Any]],
    *,
    exclude_keys: list[str],
    unwanted_summaries: list[str],
) -> list[dict[str, Any]]:
    """Apply CLI/config exclusion rules to a fetched Jira batch."""
    return filter_jira_tickets(
        tickets,
        exclude_keys=exclude_keys,
        unwanted_summaries=unwanted_summaries,
    )


def _dump_processed_tickets(processed_tickets: list[Any], dump_path: Path) -> None:
    """Dump Processed Tickets.
    
    Parameters
    ----------
    processed_tickets : list[Any]
        Value for processed Tickets.
    dump_path : Path
        Filesystem path used by the operation.
    """
    dump_processed_tickets(processed_tickets, dump_path)


def _override_qdrant_collection_name(cfg: GlobalConfig, source: str, cli_value: str | None) -> None:
    """Override Qdrant Collection Name.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    source : str
        Source definition, source name, or source identifier to process.
    cli_value : str or None, optional
        Optional value provided via the command line.
    
    Raises
    ------
    TypeError
        If the provided value has an unexpected type.
    KeyError
        If a required mapping entry is missing.
    """
    if not cli_value:
        return

    vector_stores = cfg.raw.get("vector_stores")
    if not isinstance(vector_stores, dict):
        raise TypeError("'vector_stores' config must be a mapping.")

    selected_store = vector_stores.get(source)
    if not isinstance(selected_store, dict):
        raise KeyError(f"Missing 'vector_stores.{source}' config; cannot override collection name.")

    selected_store["collection_name"] = cli_value


def _build_source_storage_context(container: Any, source: str, *, persist_dir: str | None = None) -> Any:
    """Build source Storage Context.
    
    Parameters
    ----------
    container : Any
        Value for container.
    source : str
        Source definition, source name, or source identifier to process.
    
    Returns
    -------
    Any
        Result of the operation.
    
    Raises
    ------
    KeyError
        If a required mapping entry is missing.
    """
    stores = container.vector_stores
    if source not in stores:
        raise KeyError(f"Unknown source {source!r}. Available sources: {sorted(stores.keys())}")

    docstore = load_or_create_chunk_document_store(
        persist_dir=persist_dir,
        source=source,
    )
    return build_storage_context(
        vector_store=stores[source],
        docstore=docstore,
        persist_dir=None,
    )


def _ensure_index_only_collection(storage_context: Any, source: str) -> None:
    """Create an empty collection for the selected source and exit."""
    vector_store = getattr(storage_context, "vector_store", None)
    if vector_store is None or not hasattr(vector_store, "ensure_collection_exists"):
        raise RuntimeError(
            f"Vector store for source {source!r} does not support index-only collection creation."
        )

    print(f"Ensuring empty Qdrant collection for source {source!r}...")
    vector_store.ensure_collection_exists()
    print("Collection ready.")


def _clear_persist_dir(persist_dir: str) -> None:
    """Remove the resolved persist directory and all persisted ingestion state."""
    clear_persist_dir(persist_dir)


def _clear_ingestion_target(storage_context: Any, *, persist_dir: str, source: str) -> None:
    """Reset the backing collection and local persisted state once before ingest."""
    vector_store = getattr(storage_context, "vector_store", None)
    if vector_store is None or not hasattr(vector_store, "recreate_collection"):
        raise RuntimeError(
            f"Vector store for source {source!r} does not support collection recreation."
        )

    print(f"Recreating Qdrant collection for source {source!r}...")
    vector_store.recreate_collection()
    print(f"Removing persisted ingestion state under: {Path(persist_dir).expanduser().resolve()}")
    _clear_persist_dir(persist_dir)
    print("Fresh ingestion target ready.")


def main() -> None:
    """Run the command-line entrypoint.
    
    Raises
    ------
    ValueError
        If the provided value is invalid for the operation.
    """
    args = parse_args()

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

    persist_dir = _resolve_persist_dir(cfg, args.persist_dir)
    container = build_container(cfg)
    storage_context = _build_source_storage_context(container, args.source, persist_dir=persist_dir)
    if args.clear_collection:
        _clear_ingestion_target(storage_context, persist_dir=persist_dir, source=args.source)
    if args.index_only:
        _ensure_index_only_collection(storage_context, args.source)
        return

    from polaris_rag.retrieval.document_loader import iter_support_ticket_batches

    start_date, end_date = _resolve_dates(cfg, args.start_date, args.end_date)
    limit = _resolve_limit(cfg, args.limit)
    fetch_batch_size = _resolve_fetch_batch_size(cfg, args.fetch_batch_size)
    unwanted_summaries = _resolve_unwanted_summaries(cfg)
    exclude_keys = _resolve_exclude_keys(cfg, args.exclude_keys_file)

    dump_path = Path(args.dump_path) if args.dump_path else (REPO_ROOT / "data" / "debug" / "jira_processed_tickets.txt")
    source_document_store = load_or_create_source_document_store(persist_dir=persist_dir)
    embedding_workers = _resolve_embedding_workers(cfg, args.embedding_workers)
    use_async_embeddings = embedding_workers is not None and embedding_workers > 1
    vector_batch_size = max(1, int(args.vector_batch_size))

    print("Loading Jira tickets...")
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
    processed_batches = 0
    total_fetched = 0
    total_after_filters = 0
    total_chunks = 0
    mode = "async/concurrent" if use_async_embeddings else "sync/sequential"
    should_delete_existing = not args.clear_collection

    if not should_delete_existing:
        print("Fresh-target mode enabled: skipping per-ticket delete requests during batch ingest.")

    for batch_index, fetched_tickets in enumerate(
        iter_support_ticket_batches(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            cfg=cfg,
            exclude_keys=exclude_keys,
            batch_size=fetch_batch_size,
        ),
        start=1,
    ):
        total_fetched += len(fetched_tickets)
        print(f"Fetched Jira batch {batch_index}: {len(fetched_tickets)} tickets.")

        tickets = _filter_tickets(
            fetched_tickets,
            exclude_keys=exclude_keys,
            unwanted_summaries=unwanted_summaries,
        )
        if not tickets:
            print(f"Skipping Jira batch {batch_index}: no tickets remain after filtering.")
            continue

        total_after_filters += len(tickets)
        print(f"Processing Jira batch {batch_index}: {len(tickets)} tickets after filtering.")

        processed_tickets = prepare_jira_tickets_for_chunking(
            tickets,
            chunking_strategy=chunking_settings.strategy,
            conversion_engine=conversion_settings.engine,
            conversion_options=conversion_settings.options,
            registry_artifact_path=registry_artifact_path,
            source_name=args.source,
        )
        chunks = chunk_processed_jira_tickets(
            processed_tickets,
            chunking_strategy=chunking_settings.strategy,
            token_counter=container.token_counter,
            chunk_size=chunking_settings.chunk_size_tokens,
            overlap=chunking_settings.overlap_tokens,
        )

        ticket_ids = list(dict.fromkeys(str(ticket.id) for ticket in processed_tickets if getattr(ticket, "id", None)))

        if args.dump_processed:
            print(f"Dumping processed tickets from batch {batch_index} to: {dump_path}")
            _dump_processed_tickets(processed_tickets, dump_path)
        print(f"Generated {len(chunks)} chunks from Jira batch {batch_index}.")

        if ticket_ids and should_delete_existing:
            print(f"Removing existing chunks for {len(ticket_ids)} tickets in batch {batch_index}...")
            if hasattr(storage_context.vector_store, "delete_ref_docs"):
                storage_context.vector_store.delete_ref_docs(ticket_ids)
            elif hasattr(storage_context.vector_store, "delete_ref_doc"):
                for ticket_id in ticket_ids:
                    storage_context.vector_store.delete_ref_doc(ticket_id)
            delete_ref_docs_from_docstore(storage_context.docstore, ticket_ids)

        print(
            f"Embedding/indexing {len(chunks)} chunks from Jira batch {batch_index} "
            f"(insert mode: {mode}, workers: {embedding_workers or 1}, batch size: {vector_batch_size})..."
        )
        storage_context.vector_store.insert_chunks(
            chunks,
            batch_size=vector_batch_size,
            use_async=use_async_embeddings,
        )

        print(f"Adding batch {batch_index} chunks to document store...")
        add_chunks_to_docstore(storage=storage_context, chunks=chunks)

        print(f"Persisting full tickets from batch {batch_index} to source document store...")
        add_documents_to_docstore(source_document_store, processed_tickets)

        print(f"Persisting storage after Jira batch {batch_index}...")
        persist_docstore(
            storage_context.docstore,
            persist_path=chunk_document_store_path(persist_dir, args.source),
        )
        persist_docstore(
            source_document_store,
            persist_path=source_document_store_path(persist_dir),
        )

        processed_batches += 1
        total_chunks += len(chunks)

    print(
        "Ingestion complete! "
        f"Fetched {total_fetched} tickets, processed {total_after_filters} tickets, "
        f"indexed {total_chunks} chunks across {processed_batches} batch(es)."
    )


if __name__ == "__main__":
    main()
