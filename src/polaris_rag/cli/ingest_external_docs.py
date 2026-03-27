"""Register-driven ingestion entrypoint for external official docs.

This module exposes public helper functions used by the surrounding Polaris subsystem.

Functions
---------
get_internal_links
    Return internal Links.
load_website_docs
    Load website Docs.
parse_args
    Parse args.
main
    Run the command-line entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
from polaris_rag.authority.source_register import (
    attach_source_register_metadata,
    discover_all_external_source_urls,
    load_external_source_register,
)
from polaris_rag.cli.ingest_html_documents import (
    _build_source_storage_context,
    _override_qdrant_collection_name,
    _resolve_embedding_workers,
    _resolve_persist_dir,
)
from polaris_rag.config import GlobalConfig
from polaris_rag.retrieval.document_preprocessor import preprocess_html_documents
from polaris_rag.retrieval.document_store_factory import (
    add_chunks_to_docstore,
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
    """Return internal Links.
    
    Parameters
    ----------
    homepage : str
        Value for homepage.
    
    Returns
    -------
    list[str]
        Requested internal Links.
    """
    from polaris_rag.retrieval.document_loader import get_internal_links as _get_internal_links

    return _get_internal_links(homepage)


def load_website_docs(links: list[str]):
    """Load website Docs.
    
    Parameters
    ----------
    links : list[str]
        Value for links.
    """
    from polaris_rag.retrieval.document_loader import load_website_docs as _load_website_docs

    return _load_website_docs(links)


def parse_args() -> argparse.Namespace:
    """Parse args.
    
    Returns
    -------
    argparse.Namespace
        Parsed args.
    """
    parser = argparse.ArgumentParser(description="Ingest registered external official docs into the RAG stores")
    parser.add_argument(
        "--config-file",
        "-c",
        required=True,
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--source-register-file",
        required=False,
        type=str,
        default="data/authority/source_register.external_v1.yaml",
        help="Path to the external official source register YAML.",
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
        default="external_docs",
        help="Named source in config.vector_stores to ingest into (default: external_docs).",
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
        help="Enable concurrent embedding requests by setting embedder.num_workers.",
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


def _load_external_documents(register_path: str | Path):
    """Load external Documents.
    
    Parameters
    ----------
    register_path : str or Path
        Filesystem path used by the operation.
    """
    register = load_external_source_register(register_path)
    discovered_urls = discover_all_external_source_urls(register, get_internal_links=get_internal_links)
    documents = []
    for source in register.sources:
        links = discovered_urls[source.source_id]
        if not links:
            continue
        batch = load_website_docs(links)
        documents.extend(attach_source_register_metadata(batch, source=source))
    return register, documents


def main() -> None:
    """Run the command-line entrypoint.
    
    Raises
    ------
    RuntimeError
        If `RuntimeError` is raised while executing the operation.
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
    container = build_container(cfg)

    persist_dir = _resolve_persist_dir(cfg, args.persist_dir)
    conditions = cfg.document_preprocess_html_conditions
    tags = cfg.document_preprocess_html_tags
    link_classes = cfg.document_preprocess_html_link_classes

    register, documents = _load_external_documents(args.source_register_file)
    if not documents:
        raise RuntimeError("No external HTML pages were loaded from the source register.")

    print(f"Loading registered external HTML pages: {len(documents)} page(s) across {len(register.sources)} sources")
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
        raise RuntimeError("Storage context is not available; cannot persist external HTML ingestion.")

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
        storage_context,
        persist_dir,
    )

    print(
        "External docs ingestion complete: "
        f"{len(document_ids)} documents, {len(chunks)} chunks, source={args.source!r}, persist_dir={persist_dir}"
    )


if __name__ == "__main__":
    main()
