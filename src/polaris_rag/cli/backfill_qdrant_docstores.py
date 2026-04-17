"""Backfill per-source chunk docstores from existing Qdrant payloads."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from llama_index.core.schema import TextNode

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
    source_document_store_path,
)

_TICKET_DOCUMENT_TYPES = {"helpdesk_ticket", "jira_ticket"}
_CHUNK_ID_PATTERN = re.compile(r"::chunk::(\d+)$")
_INITIAL_DESCRIPTION_PATTERN = re.compile(r"\[INITIAL_DESCRIPTION\]\s*(.*?)\s*$", re.DOTALL)
_MESSAGE_BLOCK_PATTERN = re.compile(
    r"<MESSAGE\s+id=([^\s>]+)\s+role=([^\s>]+)"
    r"(?:\s+part=(\d+)/(\d+)\s+original_id=([^\s>]+))?>\s*(.*?)\s*</MESSAGE>",
    re.DOTALL,
)
_CHUNK_ONLY_METADATA_KEYS = {
    "chunk_index",
    "chunk_type",
    "turn_range",
    "speakers",
    "overlap",
    "parent_id",
    "document_type",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild per-source chunk docstores from existing Qdrant payloads and "
            "reconstruct ticket source documents where possible."
        )
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


def _node_metadata(node: Any) -> dict[str, Any]:
    metadata = getattr(node, "metadata", None)
    return dict(metadata or {}) if isinstance(metadata, Mapping) else {}


def _ticket_chunk_index(node: Any) -> int:
    metadata = _node_metadata(node)
    raw_index = metadata.get("chunk_index")
    if isinstance(raw_index, int):
        return raw_index
    if isinstance(raw_index, str) and raw_index.isdigit():
        return int(raw_index)

    node_id = str(getattr(node, "id_", "") or getattr(node, "node_id", "") or "")
    match = _CHUNK_ID_PATTERN.search(node_id)
    if match is not None:
        return int(match.group(1))
    return 0


def _ticket_source_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in dict(metadata).items()
        if key not in _CHUNK_ONLY_METADATA_KEYS
    }
    result.pop("parent_id", None)
    result["document_type"] = str(metadata.get("document_type") or "helpdesk_ticket")
    return result


def _extract_initial_description(chunk_text: str) -> str:
    match = _INITIAL_DESCRIPTION_PATTERN.search(str(chunk_text or ""))
    if match is None:
        return ""
    return str(match.group(1) or "").strip()


def _message_parts(chunk_text: str) -> list[tuple[str, str, int, int, str]]:
    parts: list[tuple[str, str, int, int, str]] = []
    for match in _MESSAGE_BLOCK_PATTERN.finditer(str(chunk_text or "")):
        message_id = str(match.group(1) or "").strip()
        role = str(match.group(2) or "").strip()
        part_index_raw = match.group(3)
        part_total_raw = match.group(4)
        original_id_raw = match.group(5)
        body = str(match.group(6) or "").strip()

        if part_index_raw is None or part_total_raw is None or original_id_raw is None:
            parts.append((message_id, role, 1, 1, body))
            continue

        parts.append(
            (
                str(original_id_raw).strip() or message_id,
                role,
                int(part_index_raw),
                int(part_total_raw),
                body,
            )
        )
    return parts


def _message_sort_key(message_id: str) -> tuple[int, str]:
    normalized = str(message_id or "").strip()
    digits = "".join(ch for ch in normalized if ch.isdigit())
    if digits:
        return int(digits), normalized
    return 10**9, normalized


def _reconstruct_ticket_node(parent_id: str, chunks: Iterable[TextNode]) -> TextNode | None:
    ordered_chunks = sorted(list(chunks), key=lambda node: (_ticket_chunk_index(node), getattr(node, "id_", "")))
    if not ordered_chunks:
        return None

    base_metadata = _ticket_source_metadata(_node_metadata(ordered_chunks[0]))
    summary = str(base_metadata.get("summary") or "").strip()
    ticket_key = str(base_metadata.get("ticket_key") or parent_id).strip()
    status = str(base_metadata.get("status") or "").strip()
    created_at = str(base_metadata.get("created_at") or "").strip()

    initial_description = ""
    messages: dict[str, dict[str, Any]] = {}
    message_order: list[str] = []

    for chunk in ordered_chunks:
        metadata = _node_metadata(chunk)
        chunk_type = str(metadata.get("chunk_type") or "").strip().lower()
        chunk_text = str(getattr(chunk, "text", "") or "")

        if chunk_type == "initial" or "[INITIAL_DESCRIPTION]" in chunk_text:
            extracted = _extract_initial_description(chunk_text)
            if extracted and not initial_description:
                initial_description = extracted

        for original_id, role, part_index, part_total, body in _message_parts(chunk_text):
            if original_id not in messages:
                messages[original_id] = {
                    "role": role,
                    "part_total": part_total,
                    "parts": {},
                }
                message_order.append(original_id)
            record = messages[original_id]
            record["role"] = record.get("role") or role
            record["part_total"] = max(int(record.get("part_total") or 1), int(part_total))
            record_parts = record.setdefault("parts", {})
            if part_index not in record_parts and body:
                record_parts[part_index] = body

    if not initial_description and not messages:
        return None

    if not message_order:
        message_order = sorted(messages.keys(), key=_message_sort_key)

    lines = [
        "<BEGIN_TICKET>",
        f"[TICKET_KEY] {ticket_key}",
        f"[STATUS] {status}",
        f"[CREATED] {created_at}",
        f"[SUMMARY] {summary}",
        "",
        "[INITIAL_DESCRIPTION]",
        initial_description,
        "",
        "[CONVERSATION]",
    ]

    for original_id in message_order:
        record = messages[original_id]
        role = str(record.get("role") or "OTHER")
        record_parts = record.get("parts", {})
        ordered_parts = [
            str(record_parts[index]).strip()
            for index in sorted(record_parts.keys())
            if str(record_parts[index]).strip()
        ]
        body = "\n\n".join(ordered_parts).strip()
        lines.append(f"<MESSAGE id={original_id} role={role}>")
        lines.append(body)
        lines.append("</MESSAGE>")
        lines.append("")

    lines.append("<END_TICKET>")
    text = "\n".join(lines).replace("\n\n\n", "\n\n").strip()
    return TextNode(id_=str(parent_id), text=text, metadata=base_metadata)


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
    source_docstore = create_docstore("simple")
    ticket_chunks_by_parent: dict[str, list[TextNode]] = {}
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
            metadata = _node_metadata(node)
            document_type = str(metadata.get("document_type") or "").strip().lower()
            parent_id = str(metadata.get("parent_id") or "").strip()
            if document_type in _TICKET_DOCUMENT_TYPES and parent_id:
                ticket_chunks_by_parent.setdefault(parent_id, []).append(node)
            if len(buffered_nodes) >= args.write_batch_size:
                total_nodes += _flush_nodes(docstore, buffered_nodes)
                buffered_nodes = []

        total_nodes += _flush_nodes(docstore, buffered_nodes)
        persist_path = chunk_document_store_path(persist_dir, source_name)
        persist_docstore(docstore, persist_path=persist_path)
        print(
            f"[{source_name}] Rebuilt chunk docstore with {total_nodes} node(s) -> {persist_path}"
        )

    rebuilt_ticket_count = 0
    for parent_id, chunks in sorted(ticket_chunks_by_parent.items()):
        node = _reconstruct_ticket_node(parent_id, chunks)
        if node is None:
            continue
        source_docstore.add_documents([node], allow_update=True)
        rebuilt_ticket_count += 1

    source_persist_path = source_document_store_path(persist_dir)
    persist_docstore(source_docstore, persist_path=source_persist_path)
    print(
        f"Rebuilt source docstore with {rebuilt_ticket_count} ticket document(s) -> {source_persist_path}"
    )
    print("Qdrant-to-docstore backfill complete.")


if __name__ == "__main__":
    main()
