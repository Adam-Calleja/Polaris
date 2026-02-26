"""HTML ingestion entrypoint.

This script loads a website homepage (and optionally internal links),
preprocesses HTML content, chunks it, and writes the chunks to the configured
vector/doc stores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/polaris_rag` importable when running from a repo checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.app.container import build_container
from polaris_rag.config import GlobalConfig
from polaris_rag.retrieval.document_loader import get_internal_links, load_website_docs
from polaris_rag.retrieval.document_preprocessor import preprocess_html_documents
from polaris_rag.retrieval.document_store_factory import add_chunks_to_docstore, persist_storage
from polaris_rag.retrieval.text_splitter import get_chunks_from_documents


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
        help="Override Qdrant collection name from config (optional).",
    )

    parser.add_argument(
        "--vector-batch-size",
        "-b",
        required=False,
        type=int,
        default=16,
        help="Batch size for vector-store inserts (default: 16).",
    )

    return parser.parse_args()


def _as_mapping(obj):
    if isinstance(obj, dict):
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


def _override_qdrant_collection_name(cfg: GlobalConfig, cli_value: str | None) -> None:
    if not cli_value:
        return

    vector_stores = cfg.raw.get("vector_stores")
    if not isinstance(vector_stores, dict):
        raise TypeError("'vector_stores' config must be a mapping.")

    docs_store = vector_stores.get("docs")
    if not isinstance(docs_store, dict):
        raise KeyError("Missing 'vector_stores.docs' config; cannot override docs collection name.")

    docs_store["collection_name"] = cli_value


def main() -> None:
    args = parse_args()

    cfg = GlobalConfig.load(args.config_file)
    _override_qdrant_collection_name(cfg, args.qdrant_collection_name)
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

    chunks = get_chunks_from_documents(
        processed_documents,
        link_classes=link_classes,
    )

    storage_context = container.storage_context
    if storage_context is None:
        raise RuntimeError("Storage context is not available; cannot persist HTML ingestion.")

    print("Adding chunks to vector store...")
    vector_batch_size = max(1, int(args.vector_batch_size))
    total_chunks = len(chunks)
    print(f"Embedding/indexing {total_chunks} chunks (batch size: {vector_batch_size})...")
    for start in range(0, total_chunks, vector_batch_size):
        batch = chunks[start:start + vector_batch_size]
        storage_context.vector_store.insert_chunks(batch, batch_size=0)
        print(f"Inserted {min(start + vector_batch_size, total_chunks)}/{total_chunks} chunks")

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
