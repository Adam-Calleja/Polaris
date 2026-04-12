"""Shared Jira fanout indexing for experiment stages."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from polaris_rag.app.container import build_container
from polaris_rag.cli.ingest_jira_tickets import (
    _override_qdrant_collection_name,
    _resolve_dates,
    _resolve_embedding_workers,
    _resolve_exclude_keys,
    _resolve_fetch_batch_size,
    _resolve_limit,
    _resolve_persist_dir,
    _resolve_unwanted_summaries,
)
from polaris_rag.config import GlobalConfig
from polaris_rag.retrieval.document_loader import iter_support_ticket_batches
from polaris_rag.retrieval.document_store_factory import (
    add_chunks_to_docstore,
    add_documents_to_docstore,
    build_storage_context,
    load_or_create_source_document_store,
    persist_docstore,
    persist_storage,
    source_document_store_path,
)
from polaris_rag.retrieval.ingestion_settings import (
    JIRA_TURNS_TOKEN_CHUNKING_STRATEGY,
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
from polaris_rag.retrieval.metadata_enricher import resolve_authority_registry_artifact_path

SHARED_JIRA_FANOUT_INDEX_TYPE = "shared_jira_fanout"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class SharedJiraConditionEntry:
    """Rendered condition inputs required for shared stage indexing."""

    name: str
    slug: str
    preset: str | None
    config_path: Path
    condition_spec: Mapping[str, Any]
    ingest_spec: Mapping[str, Any]


@dataclass
class SharedJiraConditionPlan:
    """Resolved immutable plan for one condition's shared Jira indexing lane."""

    entry: SharedJiraConditionEntry
    cfg: Any
    source: str
    persist_dir: str
    collection_name: str
    start_date: str
    end_date: str
    limit: int | None
    fetch_batch_size: int | None
    exclude_keys: tuple[str, ...]
    unwanted_summaries: tuple[str, ...]
    chunking_strategy: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    conversion_engine: str | None
    conversion_options: dict[str, Any]
    conversion_profile: str
    registry_artifact_path: str | None
    embedding_workers: int | None
    vector_batch_size: int
    clear_target: bool
    dump_processed: bool
    dump_path: Path


@dataclass
class SharedJiraConditionRuntime:
    """Mutable runtime state for one condition's indexing lane."""

    plan: SharedJiraConditionPlan
    container: Any
    storage_context: Any
    source_document_store: Any
    total_chunks: int = 0
    total_tickets: int = 0
    processed_batches: int = 0
    persist_count: int = 0


def _condition_collection_name(cfg: Any, source: str) -> str:
    vector_stores = getattr(cfg, "raw", {}).get("vector_stores", {})
    if not isinstance(vector_stores, Mapping):
        raise TypeError("'vector_stores' config must be a mapping.")
    selected_store = vector_stores.get(source)
    if not isinstance(selected_store, Mapping):
        raise KeyError(f"Missing 'vector_stores.{source}' config.")
    collection_name = selected_store.get("collection_name")
    if not str(collection_name or "").strip():
        raise ValueError(f"Missing 'vector_stores.{source}.collection_name' config.")
    return str(collection_name)


def _build_condition_plan(
    entry: SharedJiraConditionEntry,
    *,
    clear_targets: bool,
) -> SharedJiraConditionPlan:
    ingest_spec = dict(entry.ingest_spec)
    cfg = GlobalConfig.load(str(entry.config_path))
    source = str(ingest_spec.get("source") or "tickets")
    _override_qdrant_collection_name(cfg, source, ingest_spec.get("qdrant_collection_name"))

    persist_dir = _resolve_persist_dir(cfg, ingest_spec.get("persist_dir"))
    start_date, end_date = _resolve_dates(cfg, ingest_spec.get("start_date"), ingest_spec.get("end_date"))
    limit = _resolve_limit(cfg, ingest_spec.get("limit"))
    fetch_batch_size = _resolve_fetch_batch_size(cfg, ingest_spec.get("fetch_batch_size"))
    unwanted_summaries = tuple(_resolve_unwanted_summaries(cfg))
    exclude_keys = tuple(_resolve_exclude_keys(cfg, ingest_spec.get("exclude_keys_file")))
    chunking_settings = resolve_chunking_settings(
        cfg,
        source=source,
        strategy_override=ingest_spec.get("chunking_strategy"),
        chunk_size_override=ingest_spec.get("chunk_size_tokens"),
        overlap_override=ingest_spec.get("chunk_overlap_tokens"),
    )
    conversion_settings = resolve_conversion_settings(
        cfg,
        source=source,
        engine_override=ingest_spec.get("conversion_engine"),
    )
    registry_artifact_path = resolve_authority_registry_artifact_path(cfg)
    embedding_workers = _resolve_embedding_workers(cfg, ingest_spec.get("embedding_workers"))
    vector_batch_size = max(1, int(ingest_spec.get("vector_batch_size", 16)))
    dump_path = (
        Path(str(ingest_spec["dump_path"])).expanduser()
        if ingest_spec.get("dump_path")
        else (Path(__file__).resolve().parents[3] / "data" / "debug" / "jira_processed_tickets.txt")
    )

    return SharedJiraConditionPlan(
        entry=entry,
        cfg=cfg,
        source=source,
        persist_dir=persist_dir,
        collection_name=_condition_collection_name(cfg, source),
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        fetch_batch_size=fetch_batch_size,
        exclude_keys=exclude_keys,
        unwanted_summaries=unwanted_summaries,
        chunking_strategy=str(chunking_settings.strategy),
        chunk_size_tokens=int(chunking_settings.chunk_size_tokens),
        chunk_overlap_tokens=int(chunking_settings.overlap_tokens),
        conversion_engine=conversion_settings.engine,
        conversion_options=dict(conversion_settings.options or {}),
        conversion_profile=_stable_json(
            {
                "engine": conversion_settings.engine,
                "options": dict(conversion_settings.options or {}),
            }
        ),
        registry_artifact_path=str(registry_artifact_path) if registry_artifact_path else None,
        embedding_workers=embedding_workers,
        vector_batch_size=vector_batch_size,
        clear_target=bool(clear_targets or ingest_spec.get("clear_collection", False)),
        dump_processed=bool(ingest_spec.get("dump_processed", False)),
        dump_path=dump_path,
    )


def _validate_condition_plans(plans: Sequence[SharedJiraConditionPlan]) -> None:
    if not plans:
        raise ValueError("Shared Jira fanout requires at least one condition.")

    anchor = plans[0]
    if anchor.chunking_strategy != JIRA_TURNS_TOKEN_CHUNKING_STRATEGY:
        raise ValueError(
            "Shared Jira fanout currently supports only "
            f"{JIRA_TURNS_TOKEN_CHUNKING_STRATEGY!r} chunking."
        )

    seen_collections: set[str] = set()
    seen_persist_dirs: set[str] = set()
    anchor_profile = {
        "source": anchor.source,
        "start_date": anchor.start_date,
        "end_date": anchor.end_date,
        "limit": anchor.limit,
        "fetch_batch_size": anchor.fetch_batch_size,
        "exclude_keys": anchor.exclude_keys,
        "unwanted_summaries": anchor.unwanted_summaries,
        "chunking_strategy": anchor.chunking_strategy,
        "conversion_profile": anchor.conversion_profile,
        "registry_artifact_path": anchor.registry_artifact_path,
    }

    for plan in plans:
        if str(plan.entry.ingest_spec.get("kind", "")).strip() != "jira":
            raise ValueError("Shared Jira fanout supports only Jira ingest specs.")
        if plan.chunking_strategy != JIRA_TURNS_TOKEN_CHUNKING_STRATEGY:
            raise ValueError(
                f"Condition {plan.entry.name!r} uses unsupported chunking strategy "
                f"{plan.chunking_strategy!r}; expected {JIRA_TURNS_TOKEN_CHUNKING_STRATEGY!r}."
            )
        if plan.embedding_workers not in (None, 1):
            raise ValueError(
                f"Condition {plan.entry.name!r} must not set embedding workers above 1 "
                "when shared Jira fanout indexing is enabled."
            )
        if not plan.clear_target:
            raise ValueError(
                f"Condition {plan.entry.name!r} must enable target clearing via either "
                "'index_strategy.clear_targets: true' or 'ingest.clear_collection: true'."
            )

        profile = {
            "source": plan.source,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "limit": plan.limit,
            "fetch_batch_size": plan.fetch_batch_size,
            "exclude_keys": plan.exclude_keys,
            "unwanted_summaries": plan.unwanted_summaries,
            "chunking_strategy": plan.chunking_strategy,
            "conversion_profile": plan.conversion_profile,
            "registry_artifact_path": plan.registry_artifact_path,
        }
        if profile != anchor_profile:
            raise ValueError(
                f"Condition {plan.entry.name!r} is not compatible with shared Jira fanout. "
                "All selected conditions must share the same Jira fetch/filter/preprocess profile."
            )

        if plan.collection_name in seen_collections:
            raise ValueError(
                f"Duplicate collection name {plan.collection_name!r} among shared Jira conditions."
            )
        seen_collections.add(plan.collection_name)

        if plan.persist_dir in seen_persist_dirs:
            raise ValueError(
                f"Duplicate persist dir {plan.persist_dir!r} among shared Jira conditions."
            )
        seen_persist_dirs.add(plan.persist_dir)


def _initialize_condition_runtime(plan: SharedJiraConditionPlan) -> SharedJiraConditionRuntime:
    container = build_container(plan.cfg)
    stores = container.vector_stores
    if plan.source not in stores:
        raise KeyError(f"Unknown source {plan.source!r}. Available sources: {sorted(stores.keys())}")
    storage_context = build_storage_context(
        vector_store=stores[plan.source],
        docstore=container.doc_store,
        persist_dir=None,
    )
    vector_store = getattr(storage_context, "vector_store", None)
    if vector_store is None or not hasattr(vector_store, "recreate_collection"):
        raise RuntimeError(
            f"Vector store for source {plan.source!r} does not support collection recreation."
        )

    print(f"[{plan.entry.name}] Recreating collection {plan.collection_name!r}...")
    vector_store.recreate_collection()
    print(f"[{plan.entry.name}] Clearing persisted state under {Path(plan.persist_dir).expanduser().resolve()}...")
    clear_persist_dir(plan.persist_dir)

    source_document_store = load_or_create_source_document_store(persist_dir=plan.persist_dir)
    return SharedJiraConditionRuntime(
        plan=plan,
        container=container,
        storage_context=storage_context,
        source_document_store=source_document_store,
    )


def _process_condition_batch(
    runtime: SharedJiraConditionRuntime,
    *,
    processed_tickets: Sequence[Any],
    batch_index: int,
) -> dict[str, Any]:
    plan = runtime.plan
    chunks = chunk_processed_jira_tickets(
        processed_tickets,
        chunking_strategy=plan.chunking_strategy,
        token_counter=runtime.container.token_counter,
        chunk_size=plan.chunk_size_tokens,
        overlap=plan.chunk_overlap_tokens,
    )
    print(
        f"[{plan.entry.name}] Embedding/indexing {len(chunks)} chunks from shared Jira batch {batch_index} "
        f"(batch size: {plan.vector_batch_size})..."
    )
    runtime.storage_context.vector_store.insert_chunks(
        chunks,
        batch_size=plan.vector_batch_size,
        use_async=False,
    )
    add_chunks_to_docstore(storage=runtime.storage_context, chunks=chunks)
    add_documents_to_docstore(runtime.source_document_store, processed_tickets)
    runtime.total_chunks += len(chunks)
    runtime.total_tickets += len(processed_tickets)
    runtime.processed_batches += 1
    return {
        "condition_name": plan.entry.name,
        "batch_index": batch_index,
        "chunk_count": len(chunks),
        "ticket_count": len(processed_tickets),
    }


def _persist_condition_runtime(runtime: SharedJiraConditionRuntime) -> None:
    plan = runtime.plan
    print(f"[{plan.entry.name}] Persisting storage context to {plan.persist_dir}...")
    persist_storage(storage=runtime.storage_context, persist_dir=plan.persist_dir)
    persist_docstore(
        runtime.source_document_store,
        persist_path=source_document_store_path(plan.persist_dir),
    )
    runtime.persist_count += 1


def _strategy_parallelism(index_strategy: Mapping[str, Any], *, condition_count: int) -> int:
    raw = index_strategy.get("condition_parallelism", condition_count)
    parallelism = int(raw)
    if parallelism < 1:
        raise ValueError("index_strategy.condition_parallelism must be >= 1.")
    return min(parallelism, max(1, condition_count))


def _strategy_persist_every_batches(index_strategy: Mapping[str, Any]) -> int:
    raw = int(index_strategy.get("persist_every_batches", 0))
    if raw < 0:
        raise ValueError("index_strategy.persist_every_batches must be >= 0.")
    return raw


def run_shared_jira_fanout_index(
    *,
    condition_entries: Sequence[SharedJiraConditionEntry],
    index_strategy: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """Execute a shared Jira fetch/preprocess pass with per-condition fanout indexing."""
    strategy_type = str(index_strategy.get("type", "")).strip()
    if strategy_type != SHARED_JIRA_FANOUT_INDEX_TYPE:
        raise ValueError(f"Unsupported shared Jira index strategy type {strategy_type!r}.")

    clear_targets = bool(index_strategy.get("clear_targets", False))
    plans = [_build_condition_plan(entry, clear_targets=clear_targets) for entry in condition_entries]
    _validate_condition_plans(plans)

    parallelism = _strategy_parallelism(index_strategy, condition_count=len(plans))
    persist_every_batches = _strategy_persist_every_batches(index_strategy)
    summary: dict[str, Any] = {
        "type": SHARED_JIRA_FANOUT_INDEX_TYPE,
        "clear_targets": clear_targets,
        "condition_parallelism": parallelism,
        "persist_every_batches": persist_every_batches,
        "selected_conditions": [plan.entry.name for plan in plans],
        "fetched_tickets": 0,
        "processed_tickets": 0,
        "processed_batches": 0,
        "elapsed_seconds": 0.0,
    }

    if dry_run:
        return {
            "conditions": [
                {
                    "name": plan.entry.name,
                    "slug": plan.entry.slug,
                    "config_path": str(plan.entry.config_path),
                    "preset": plan.entry.preset,
                    "execution_phase": "index",
                    "ingestion_commands": [],
                    "runs": [],
                    "ingestion_skipped": False,
                    "shared_ingestion": True,
                    "collection_name": plan.collection_name,
                    "persist_dir": plan.persist_dir,
                    "indexed_chunks": 0,
                    "processed_batches": 0,
                    "persist_count": 0,
                }
                for plan in plans
            ],
            "summary": summary,
        }

    runtimes = [_initialize_condition_runtime(plan) for plan in plans]
    anchor = plans[0]
    total_fetched = 0
    total_after_filters = 0
    processed_batches = 0
    started_at = time.perf_counter()

    print(
        "Starting shared Jira fanout indexing for "
        f"{len(runtimes)} condition(s) with parallelism={parallelism}."
    )

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        for batch_index, fetched_tickets in enumerate(
            iter_support_ticket_batches(
                start_date=anchor.start_date,
                end_date=anchor.end_date,
                limit=anchor.limit,
                cfg=anchor.cfg,
                exclude_keys=list(anchor.exclude_keys),
                batch_size=anchor.fetch_batch_size,
            ),
            start=1,
        ):
            total_fetched += len(fetched_tickets)
            print(f"Fetched shared Jira batch {batch_index}: {len(fetched_tickets)} tickets.")
            tickets = filter_jira_tickets(
                fetched_tickets,
                exclude_keys=anchor.exclude_keys,
                unwanted_summaries=anchor.unwanted_summaries,
            )
            if not tickets:
                print(f"Skipping shared Jira batch {batch_index}: no tickets remain after filtering.")
                continue

            processed_batches += 1
            total_after_filters += len(tickets)
            print(
                f"Processing shared Jira batch {batch_index}: "
                f"{len(tickets)} tickets after filtering."
            )
            processed_tickets = prepare_jira_tickets_for_chunking(
                tickets,
                chunking_strategy=anchor.chunking_strategy,
                conversion_engine=anchor.conversion_engine,
                conversion_options=anchor.conversion_options,
                registry_artifact_path=anchor.registry_artifact_path,
                source_name=anchor.source,
            )

            for runtime in runtimes:
                if runtime.plan.dump_processed:
                    print(f"[{runtime.plan.entry.name}] Dumping processed tickets to {runtime.plan.dump_path}...")
                    dump_processed_tickets(processed_tickets, runtime.plan.dump_path)

            futures = [
                (
                    runtime,
                    executor.submit(
                        _process_condition_batch,
                        runtime,
                        processed_tickets=processed_tickets,
                        batch_index=batch_index,
                    ),
                )
                for runtime in runtimes
            ]
            for runtime, future in futures:
                future.result()

            if persist_every_batches and processed_batches % persist_every_batches == 0:
                for runtime in runtimes:
                    _persist_condition_runtime(runtime)

    if not persist_every_batches or processed_batches % persist_every_batches != 0:
        for runtime in runtimes:
            _persist_condition_runtime(runtime)

    elapsed_seconds = time.perf_counter() - started_at
    summary.update(
        {
            "fetched_tickets": total_fetched,
            "processed_tickets": total_after_filters,
            "processed_batches": processed_batches,
            "elapsed_seconds": elapsed_seconds,
        }
    )
    print(
        "Shared Jira fanout indexing complete! "
        f"Fetched {total_fetched} tickets, processed {total_after_filters} tickets "
        f"across {processed_batches} batch(es) in {elapsed_seconds:.1f}s."
    )

    return {
        "conditions": [
            {
                "name": runtime.plan.entry.name,
                "slug": runtime.plan.entry.slug,
                "config_path": str(runtime.plan.entry.config_path),
                "preset": runtime.plan.entry.preset,
                "execution_phase": "index",
                "ingestion_commands": [],
                "runs": [],
                "ingestion_skipped": False,
                "shared_ingestion": True,
                "collection_name": runtime.plan.collection_name,
                "persist_dir": runtime.plan.persist_dir,
                "indexed_chunks": runtime.total_chunks,
                "processed_batches": runtime.processed_batches,
                "indexed_tickets": runtime.total_tickets,
                "persist_count": runtime.persist_count,
            }
            for runtime in runtimes
        ],
        "summary": summary,
    }


__all__ = [
    "SHARED_JIRA_FANOUT_INDEX_TYPE",
    "SharedJiraConditionEntry",
    "SharedJiraConditionPlan",
    "SharedJiraConditionRuntime",
    "run_shared_jira_fanout_index",
]
