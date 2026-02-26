"""Jira ticket ingestion entrypoint.

This script is intentionally *thin*: it loads configuration, fetches tickets,
preprocesses + chunks them, and writes them to the configured stores.

Design goals
------------
- No hard-coded absolute paths.
- Config-first: most parameters come from YAML, with CLI overrides.
- Works when running from a repo checkout (adds `<repo>/src` to sys.path).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

# Make `src/polaris_rag` importable when running from a repo checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.config import GlobalConfig
from polaris_rag.retrieval.document_loader import load_support_tickets
from polaris_rag.retrieval.document_preprocessor import preprocess_jira_tickets
from polaris_rag.retrieval.text_splitter import get_chunks_from_jira_tickets
from polaris_rag.app.container import build_container
from polaris_rag.retrieval.document_store_factory import (
    add_chunks_to_docstore,
    build_storage_context,
    persist_storage,
)


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

    # Prefer config.storage_context.persist_dir; allow override for one-off runs.
    parser.add_argument(
        "--persist-dir",
        "-d",
        required=False,
        type=str,
        default=None,
        help="Override persist dir from config (optional)",
    )

    # Prefer config.ingestion.jira.start_date/end_date (if present); allow CLI override.
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

    # Prefer config.ingestion.jira.limit (if present); allow override.
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
        "--vector-batch-retries",
        required=False,
        type=int,
        default=5,
        help="Retries per vector batch before splitting into smaller batches (default: 5).",
    )
    parser.add_argument(
        "--vector-batch-retry-delay",
        required=False,
        type=float,
        default=5.0,
        help="Base retry delay in seconds for failed vector batches (default: 5.0).",
    )

    # Optional debug dump of processed ticket text.
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

    return parser.parse_args()


def _get_ingestion_cfg(cfg: GlobalConfig) -> Mapping[str, Any]:
    # Supports either a dict-style config or a pydantic-style object.
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
    # Common patterns: storage_context.persist_dir, or storage_context["persist_dir"].
    if sc is not None:
        if hasattr(sc, "persist_dir") and getattr(sc, "persist_dir"):
            return str(getattr(sc, "persist_dir"))
        sc_map = _as_mapping(sc)
        if sc_map.get("persist_dir"):
            return str(sc_map["persist_dir"])

    # Safe repo-relative default.
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

    # Backwards-compatible defaults.
    return [
        "Internal HPC Application Form",
        "Self-Service Gateway",
        "SSH public key",
    ]


def _read_exclude_keys_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    keys: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
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
        keys.extend([k.upper() for k in cfg_inline_keys if isinstance(k, str) and k.strip()])

    exclude_file = cli_value or cfg_file
    if exclude_file:
        candidate = Path(exclude_file)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        keys.extend(_read_exclude_keys_file(candidate))

    return list(dict.fromkeys(keys))


def _dump_processed_tickets(processed_tickets: list[Any], dump_path: Path) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\n\n" + ("-" * 10) + "\n\n"
    with dump_path.open("a", encoding="utf-8") as f:
        for t in processed_tickets:
            f.write(getattr(t, "text", ""))
            f.write(sep)

def _append_failed_chunk_log(
    log_path: Path,
    chunk: Any,
    exc: Exception,
    *,
    retries: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_id = getattr(chunk, "id", None)
    parent_id = getattr(chunk, "parent_id", None)
    text = getattr(chunk, "text", "") or ""
    text_len = len(text)
    metadata = getattr(chunk, "metadata", {}) or {}
    turn_range = metadata.get("turn_range")
    chunk_type = metadata.get("chunk_type")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"chunk_id={chunk_id}\tparent_id={parent_id}\tchunk_type={chunk_type}\t"
            f"turn_range={turn_range}\ttext_len={text_len}\tretries={retries}\t"
            f"error={type(exc).__name__}: {exc}\n"
        )

def _insert_vector_batch_with_retries(
    storage_context: Any,
    batch: list[Any],
    *,
    batch_start_idx: int,
    total_chunks: int,
    retries: int,
    retry_delay: float,
    failed_chunks_log_path: Path,
) -> int:
    attempt = 0
    while True:
        try:
            storage_context.vector_store.insert_chunks(batch, batch_size=0)
            return 0
        except Exception as exc:
            attempt += 1
            if attempt <= retries:
                wait_seconds = max(0.0, retry_delay) * attempt
                print(
                    f"Batch {batch_start_idx + 1}-{batch_start_idx + len(batch)}/{total_chunks} failed "
                    f"({type(exc).__name__}: {exc}). Retry {attempt}/{retries} in {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)
                continue

            if len(batch) > 1:
                split = len(batch) // 2
                print(
                    f"Batch {batch_start_idx + 1}-{batch_start_idx + len(batch)}/{total_chunks} failed after "
                    f"{retries} retries. Splitting into {split} and {len(batch) - split} chunks."
                )
                left_skipped = _insert_vector_batch_with_retries(
                    storage_context,
                    batch[:split],
                    batch_start_idx=batch_start_idx,
                    total_chunks=total_chunks,
                    retries=retries,
                    retry_delay=retry_delay,
                    failed_chunks_log_path=failed_chunks_log_path,
                )
                right_skipped = _insert_vector_batch_with_retries(
                    storage_context,
                    batch[split:],
                    batch_start_idx=batch_start_idx + split,
                    total_chunks=total_chunks,
                    retries=retries,
                    retry_delay=retry_delay,
                    failed_chunks_log_path=failed_chunks_log_path,
                )
                return left_skipped + right_skipped

            # Single-chunk failure after all retries: skip and log.
            chunk = batch[0]
            _append_failed_chunk_log(
                failed_chunks_log_path,
                chunk,
                exc,
                retries=retries,
            )
            print(
                f"Skipping chunk {batch_start_idx + 1}/{total_chunks} after {retries} retries "
                f"({type(exc).__name__}: {exc}). Logged to {failed_chunks_log_path}"
            )
            return 1


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


def _build_source_storage_context(container, source: str):
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

    cfg = GlobalConfig.load(args.config_file)
    _override_qdrant_collection_name(cfg, args.source, args.qdrant_collection_name)

    # Build runtime objects (vector store, docstore, token counter, etc.).
    container = build_container(cfg)

    # Resolve run parameters (config-first, CLI overrides).
    persist_dir = _resolve_persist_dir(cfg, args.persist_dir)
    start_date, end_date = _resolve_dates(cfg, args.start_date, args.end_date)
    limit = _resolve_limit(cfg, args.limit)
    unwanted_summaries = _resolve_unwanted_summaries(cfg)
    exclude_keys = _resolve_exclude_keys(cfg, args.exclude_keys_file)

    # Where to write debug dumps (optional).
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

    # Defensive client-side filter for robustness against case mismatches.
    if exclude_keys:
        excluded_key_set = set(exclude_keys)
        tickets = [t for t in tickets if str(t.get("key", "")).upper() not in excluded_key_set]

    # Filter tickets by summary substrings.
    if unwanted_summaries:
        tickets = [
            t
            for t in tickets
            if not any(u in t.get("fields", {}).get("summary", "") for u in unwanted_summaries)
        ]

    print(f"Loaded {len(tickets)} tickets. Preprocessing...")
    processed_tickets = preprocess_jira_tickets(tickets)

    if args.dump_processed:
        print(f"Dumping processed tickets to: {dump_path}")
        _dump_processed_tickets(processed_tickets, dump_path)

    print("Generating chunks from tickets...")
    chunks = get_chunks_from_jira_tickets(tickets=processed_tickets, token_counter=container.token_counter)

    print("Adding chunks to vector store...")
    vector_batch_size = max(1, int(args.vector_batch_size))
    vector_batch_retries = max(0, int(args.vector_batch_retries))
    vector_batch_retry_delay = max(0.0, float(args.vector_batch_retry_delay))
    failed_chunks_log_path = Path(persist_dir) / "failed_vector_chunks.log"
    total_chunks = len(chunks)
    skipped_chunks = 0
    print(f"Embedding/indexing {total_chunks} chunks (batch size: {vector_batch_size})...")
    for start in range(0, total_chunks, vector_batch_size):
        batch = chunks[start:start + vector_batch_size]
        skipped_chunks += _insert_vector_batch_with_retries(
            storage_context,
            batch,
            batch_start_idx=start,
            total_chunks=total_chunks,
            retries=vector_batch_retries,
            retry_delay=vector_batch_retry_delay,
            failed_chunks_log_path=failed_chunks_log_path,
        )
        print(f"Inserted {min(start + vector_batch_size, total_chunks)}/{total_chunks} chunks")
    if skipped_chunks:
        print(f"Skipped {skipped_chunks} chunk(s). See: {failed_chunks_log_path}")

    print("Adding chunks to document store...")
    add_chunks_to_docstore(storage=storage_context, chunks=chunks)

    print("Persisting storage context...")
    persist_storage(storage=storage_context, persist_dir=persist_dir)

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
