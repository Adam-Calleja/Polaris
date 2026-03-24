"""HTML ingestion entrypoint."""

from __future__ import annotations

import argparse
import sys
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
from polaris_rag.retrieval.document_preprocessor import preprocess_html_documents
from polaris_rag.retrieval.document_store_factory import (
    add_chunks_to_docstore,
    build_storage_context,
    delete_ref_docs_from_docstore,
    persist_storage,
)
from polaris_rag.retrieval.ingestion_settings import (
    HTML_HIERARCHICAL_CHUNKING_STRATEGY,
    MARKDOWN_TOKEN_CHUNKING_STRATEGY,
    resolve_chunking_settings,
    resolve_conversion_settings,
)
from polaris_rag.retrieval.metadata_enricher import (
    enrich_documents_with_authority_metadata,
    localize_doc_chunk_scope_family_metadata,
    resolve_authority_registry_artifact_path,
)
from polaris_rag.retrieval.markdown_chunker import get_chunks_from_markdown_documents
from polaris_rag.retrieval.markdown_converter import convert_documents_to_markdown
from polaris_rag.retrieval.text_splitter import get_chunks_from_documents


def get_internal_links(homepage: str) -> list[str]:
    from polaris_rag.retrieval.document_loader import get_internal_links as _get_internal_links

    return _get_internal_links(homepage)


def load_website_docs(links: list[str]) -> list[Any]:
    from polaris_rag.retrieval.document_loader import load_website_docs as _load_website_docs

    return _load_website_docs(links)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest HTML documents into the RAG stores")

    parser.add_argument(
        "--homepage",
        "-p",
        required=True,
        type=str,
        help="Homepage URL to ingest.",
    )
    parser.add_argument(
        "--ingest-internal-links",
        "-i",
        action="store_true",
        help="If set, ingest internal links discovered from the homepage.",
    )
    parser.add_argument(
        "--config-file",
        "-c",
        required=True,
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--persist-dir",
        "-d",
        required=False,
        type=str,
        default=None,
        help="Override persist dir from config (optional).",
    )
    parser.add_argument(
        "--qdrant-collection-name",
        "--collection-name",
        required=False,
        type=str,
        default=None,
        help="Override selected source collection name from config (optional).",
    )
    parser.add_argument(
        "--source",
        required=False,
        type=str,
        default="docs",
        help="Named source in config.vector_stores to ingest into (default: docs).",
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


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


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


def main() -> None:
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
    container = build_container(cfg)

    persist_dir = _resolve_persist_dir(cfg, args.persist_dir)
    conditions = cfg.document_preprocess_html_conditions
    tags = cfg.document_preprocess_html_tags
    link_classes = cfg.document_preprocess_html_link_classes

    if args.ingest_internal_links:
        links = get_internal_links(args.homepage)
    else:
        links = [args.homepage]

    print(f"Loading HTML pages: {len(links)} URL(s)")
    documents = load_website_docs(links)
    processed_documents = preprocess_html_documents(
        documents,
        tags=tags,
        conditions=conditions,
    )
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

    if chunking_settings.strategy == MARKDOWN_TOKEN_CHUNKING_STRATEGY:
        markdown_documents = convert_documents_to_markdown(
            processed_documents,
            engine=conversion_settings.engine,
            options=conversion_settings.options,
        )
        markdown_documents = enrich_documents_with_authority_metadata(
            markdown_documents,
            registry_artifact_path=registry_artifact_path,
            source_name=args.source,
        )
        chunks = get_chunks_from_markdown_documents(
            markdown_documents,
            token_counter=container.token_counter,
            chunk_size=chunking_settings.chunk_size_tokens,
            overlap=chunking_settings.overlap_tokens,
        )
        document_ids = [str(document.id) for document in markdown_documents if getattr(document, "id", None)]
    elif chunking_settings.strategy == HTML_HIERARCHICAL_CHUNKING_STRATEGY:
        processed_documents = enrich_documents_with_authority_metadata(
            processed_documents,
            registry_artifact_path=registry_artifact_path,
            source_name=args.source,
        )
        chunks = get_chunks_from_documents(
            processed_documents,
            link_classes=link_classes,
        )
        document_ids = [str(document.id) for document in processed_documents if getattr(document, "id", None)]
    else:
        raise ValueError(f"Unsupported docs chunking strategy: {chunking_settings.strategy!r}")

    chunks = localize_doc_chunk_scope_family_metadata(
        chunks,
        registry_artifact_path=registry_artifact_path,
    )

    storage_context = _build_source_storage_context(container, args.source)
    if storage_context is None:
        raise RuntimeError("Storage context is not available; cannot persist HTML ingestion.")

    if document_ids:
        print(f"Removing existing chunks for {len(document_ids)} documents...")
        if hasattr(storage_context.vector_store, "delete_ref_docs"):
            storage_context.vector_store.delete_ref_docs(document_ids)
        elif hasattr(storage_context.vector_store, "delete_ref_doc"):
            for document_id in document_ids:
                storage_context.vector_store.delete_ref_doc(document_id)
        delete_ref_docs_from_docstore(storage_context.docstore, document_ids)

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
    add_chunks_to_docstore(
        storage=storage_context,
        chunks=chunks,
    )

    print("Persisting storage context...")
    persist_storage(
        storage=storage_context,
        persist_dir=persist_dir,
    )

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
