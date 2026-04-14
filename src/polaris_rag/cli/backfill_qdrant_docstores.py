"""Backfill per-source chunk docstores from existing Qdrant payloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable


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
    chunk_document_store_path,
    create_docstore,
    persist_docstore,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rebuild per-source chunk docstores by scrolling existing Qdrant payloads."
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
        "--source",
        action="append",
        default=None,
        help="Backfill only the named source. Repeat to select multiple sources. Defaults to all configured sources.",
    )
    parser.add_argument(
        "--scroll-batch-size",
        type=int,
        default=512,
        help="Batch size for Qdrant scroll requests (default: 512).",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=512,
        help="Batch size for docstore add_documents calls (default: 512).",
    )
    return parser.parse_args()


def _resolve_persist_dir(cfg: GlobalConfig, cli_value: str | None) -> str:
    if cli_value:
        return str(Path(cli_value).expanduser().resolve())

    storage_cfg = getattr(cfg, "raw", {}).get("storage_context", {})
    raw_persist_dir = storage_cfg.get("persist_dir") if isinstance(storage_cfg, dict) else None
    if not raw_persist_dir:
        raise ValueError("No storage_context.persist_dir configured and no --persist-dir override was provided.")

    persist_path = Path(str(raw_persist_dir))
    cfg_path = getattr(cfg, "config_path", None)
    if cfg_path and not persist_path.is_absolute():
        persist_path = Path(cfg_path).expanduser().resolve().parent / persist_path
    return str(persist_path.expanduser().resolve())


def _selected_sources(requested_sources: Iterable[str] | None, available_sources: Iterable[str]) -> list[str]:
    available = {str(source): str(source) for source in available_sources}
    if requested_sources is None:
        return sorted(available.values())

    selected: list[str] = []
    for source in requested_sources:
        normalized = str(source or "").strip()
        if not normalized:
            continue
        if normalized not in available:
            raise KeyError(
                f"Unknown source {normalized!r}. Available sources: {sorted(available.values())}"
            )
        if normalized not in selected:
            selected.append(normalized)
    return selected


def _flush_nodes(docstore: Any, nodes: list[Any]) -> int:
    if not nodes:
        return 0
    docstore.add_documents(nodes, allow_update=True)
    return len(nodes)


def main() -> None:
    """Run the command-line entrypoint."""
    args = parse_args()
    if args.scroll_batch_size < 1:
        raise ValueError("--scroll-batch-size must be >= 1.")
    if args.write_batch_size < 1:
        raise ValueError("--write-batch-size must be >= 1.")

    cfg = GlobalConfig.load(args.config_file)
    persist_dir = _resolve_persist_dir(cfg, args.persist_dir)
    container = build_container(cfg)
    stores = container.vector_stores
    selected_sources = _selected_sources(args.source, stores.keys())

    if not selected_sources:
        raise ValueError("No sources selected for backfill.")

    print(f"Backfilling chunk docstores into {persist_dir} for sources: {', '.join(selected_sources)}")
    for source_name in selected_sources:
        vector_store = stores[source_name]
        iterator = getattr(vector_store, "iter_payload_nodes", None)
        if not callable(iterator):
            raise RuntimeError(
                f"Vector store for source {source_name!r} does not support payload-node iteration."
            )

        docstore = create_docstore("simple")
        buffered_nodes: list[Any] = []
        total_nodes = 0

        for node in iterator(batch_size=args.scroll_batch_size):
            buffered_nodes.append(node)
            if len(buffered_nodes) >= args.write_batch_size:
                total_nodes += _flush_nodes(docstore, buffered_nodes)
                buffered_nodes = []

        total_nodes += _flush_nodes(docstore, buffered_nodes)
        persist_path = chunk_document_store_path(persist_dir, source_name)
        persist_docstore(docstore, persist_path=persist_path)
        print(
            f"[{source_name}] Rebuilt chunk docstore with {total_nodes} node(s) -> {persist_path}"
        )

    print("Qdrant-to-docstore backfill complete.")


if __name__ == "__main__":
    main()
