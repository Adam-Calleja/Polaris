"""Count support tickets stored in a persisted docstore."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from llama_index.core.storage.docstore import SimpleDocumentStore


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return Path.cwd()


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.config import GlobalConfig
from polaris_rag.retrieval.document_store_factory import (
    chunk_document_store_path,
    source_document_store_path,
)


_TICKET_DOCUMENT_TYPES = {"helpdesk_ticket", "jira_ticket"}
_CHUNK_ID_PATTERN = re.compile(r"::chunk::\d+$")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Count the number of support tickets stored in a persisted docstore."
    )
    parser.add_argument(
        "--docstore-path",
        type=str,
        default=None,
        help="Direct path to a persisted docstore JSON file.",
    )
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        default=None,
        help="Path to the YAML configuration file used to resolve storage_context.persist_dir.",
    )
    parser.add_argument(
        "--persist-dir",
        "-d",
        type=str,
        default=None,
        help="Override the configured persist dir when --docstore-path is not provided.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="tickets",
        help="Source name used when resolving a chunk docstore path (default: tickets).",
    )
    parser.add_argument(
        "--docstore-kind",
        choices=("auto", "source", "chunk"),
        default="auto",
        help=(
            "How to interpret the stored nodes. "
            "'source' counts one stored ticket per node, "
            "'chunk' counts unique ticket ids across ticket chunks, "
            "'auto' infers the mode from the stored node shape."
        ),
    )
    return parser.parse_args()


def _resolve_persist_dir(cfg: GlobalConfig | None, cli_value: str | None) -> str:
    if cli_value:
        return str(Path(cli_value).expanduser().resolve())

    if cfg is None:
        raise ValueError("Either --docstore-path or one of --persist-dir/--config-file must be provided.")

    storage_cfg = getattr(cfg, "raw", {}).get("storage_context", {})
    raw_persist_dir = storage_cfg.get("persist_dir") if isinstance(storage_cfg, dict) else None
    if not raw_persist_dir:
        raise ValueError("No storage_context.persist_dir configured and no --persist-dir override was provided.")

    persist_path = Path(str(raw_persist_dir))
    cfg_path = getattr(cfg, "config_path", None)
    if cfg_path and not persist_path.is_absolute():
        persist_path = Path(cfg_path).expanduser().resolve().parent / persist_path
    return str(persist_path.expanduser().resolve())


def _resolve_docstore_path(args: argparse.Namespace) -> Path:
    if args.docstore_path:
        return Path(args.docstore_path).expanduser().resolve()

    cfg = GlobalConfig.load(args.config_file) if args.config_file else None
    persist_dir = _resolve_persist_dir(cfg, args.persist_dir)
    source_path = Path(source_document_store_path(persist_dir))
    chunk_path = Path(chunk_document_store_path(persist_dir, args.source))

    if args.docstore_kind == "source":
        return source_path
    if args.docstore_kind == "chunk":
        return chunk_path
    if source_path.exists():
        return source_path
    if chunk_path.exists():
        return chunk_path
    return source_path


def _docstore_nodes(docstore: Any) -> list[Any]:
    docs = getattr(docstore, "docs", None)
    if not isinstance(docs, Mapping):
        return []
    return [node for _, node in sorted(docs.items(), key=lambda item: str(item[0])) if node is not None]


def _node_metadata(node: Any) -> dict[str, Any]:
    metadata = getattr(node, "metadata", None)
    return dict(metadata or {}) if isinstance(metadata, Mapping) else {}


def _node_id(node: Any) -> str:
    for attr in ("id_", "node_id", "id"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _is_ticket_node(node: Any) -> bool:
    metadata = _node_metadata(node)
    document_type = str(metadata.get("document_type") or "").strip().lower()
    if document_type in _TICKET_DOCUMENT_TYPES:
        return True

    text = str(getattr(node, "text", "") or "")
    return text.startswith("<BEGIN_TICKET>") or "[TICKET_SUMMARY]" in text


def _ticket_parent_id(node: Any) -> str:
    metadata = _node_metadata(node)
    parent_id = str(
        metadata.get("parent_id")
        or metadata.get("ticket_key")
        or getattr(node, "ref_doc_id", "")
        or ""
    ).strip()
    if parent_id:
        return parent_id

    node_id = _node_id(node)
    if _CHUNK_ID_PATTERN.search(node_id):
        return _CHUNK_ID_PATTERN.sub("", node_id)
    return node_id


def _looks_like_chunk_docstore(nodes: list[Any]) -> bool:
    for node in nodes:
        if not _is_ticket_node(node):
            continue
        metadata = _node_metadata(node)
        if any(key in metadata for key in ("parent_id", "chunk_index", "chunk_type")):
            return True
        node_id = _node_id(node)
        if _CHUNK_ID_PATTERN.search(node_id):
            return True
    return False


def count_tickets(docstore: Any, *, docstore_kind: str = "auto") -> int:
    """Count tickets stored in a source or chunk docstore."""
    nodes = [node for node in _docstore_nodes(docstore) if _is_ticket_node(node)]
    if docstore_kind == "auto":
        count_mode = "chunk" if _looks_like_chunk_docstore(nodes) else "source"
    else:
        count_mode = docstore_kind

    if count_mode == "source":
        return len(nodes)

    ticket_ids: set[str] = set()
    for node in nodes:
        ticket_id = _ticket_parent_id(node)
        if ticket_id:
            ticket_ids.add(ticket_id)
    return len(ticket_ids)


def main() -> None:
    """Run the command-line entrypoint."""
    args = parse_args()
    docstore_path = _resolve_docstore_path(args)
    if not docstore_path.exists():
        raise FileNotFoundError(f"Docstore file not found: {docstore_path}")

    docstore = SimpleDocumentStore.from_persist_path(str(docstore_path))
    print(count_tickets(docstore, docstore_kind=args.docstore_kind))


if __name__ == "__main__":
    main()
