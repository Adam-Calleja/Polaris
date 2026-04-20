from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.cli import count_docstore_tickets


def _persist_docstore(path: Path, nodes: list[TextNode]) -> None:
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes, allow_update=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    docstore.persist(persist_path=str(path))


def test_main_counts_source_docstore_tickets_from_direct_path(capsys, monkeypatch, tmp_path: Path) -> None:
    docstore_path = tmp_path / "source_docstore.json"
    _persist_docstore(
        docstore_path,
        [
            TextNode(
                id_="HPCSSUP-1",
                text="<BEGIN_TICKET>\n[TICKET_KEY] HPCSSUP-1\n<END_TICKET>",
                metadata={"document_type": "helpdesk_ticket"},
            ),
            TextNode(
                id_="HPCSSUP-2",
                text="<BEGIN_TICKET>\n[TICKET_KEY] HPCSSUP-2\n<END_TICKET>",
                metadata={"document_type": "helpdesk_ticket"},
            ),
            TextNode(
                id_="doc-1",
                text="HTML document",
                metadata={"document_type": "html"},
            ),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "count_docstore_tickets.py",
            "--docstore-path",
            str(docstore_path),
        ],
    )

    count_docstore_tickets.main()

    assert capsys.readouterr().out.strip() == "2"


def test_main_counts_unique_tickets_from_chunk_docstore_in_auto_mode(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    docstore_path = tmp_path / "chunk_docstore.tickets.json"
    _persist_docstore(
        docstore_path,
        [
            TextNode(
                id_="HPCSSUP-1::chunk::0000",
                text="[TICKET_SUMMARY] Ticket one",
                metadata={
                    "document_type": "helpdesk_ticket",
                    "parent_id": "HPCSSUP-1",
                    "chunk_index": 0,
                    "chunk_type": "initial",
                },
            ),
            TextNode(
                id_="HPCSSUP-1::chunk::0001",
                text="[CONVERSATION]\n<MESSAGE id=0001 role=REPORTER>\nHello\n</MESSAGE>",
                metadata={
                    "document_type": "helpdesk_ticket",
                    "parent_id": "HPCSSUP-1",
                    "chunk_index": 1,
                    "chunk_type": "conversation",
                },
            ),
            TextNode(
                id_="HPCSSUP-2::chunk::0000",
                text="[TICKET_SUMMARY] Ticket two",
                metadata={
                    "document_type": "helpdesk_ticket",
                    "parent_id": "HPCSSUP-2",
                    "chunk_index": 0,
                    "chunk_type": "initial",
                },
            ),
            TextNode(
                id_="docs::chunk::0000",
                text="Non-ticket document chunk",
                metadata={
                    "document_type": "html",
                    "parent_id": "docs",
                    "chunk_index": 0,
                },
            ),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "count_docstore_tickets.py",
            "--docstore-path",
            str(docstore_path),
        ],
    )

    count_docstore_tickets.main()

    assert capsys.readouterr().out.strip() == "2"


def test_main_uses_ticket_key_when_chunk_parent_id_is_missing(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    docstore_path = tmp_path / "legacy_chunk_docstore.tickets.json"
    _persist_docstore(
        docstore_path,
        [
            TextNode(
                id_="uuid-1",
                text="[TICKET_SUMMARY] Ticket one",
                metadata={
                    "document_type": "helpdesk_ticket",
                    "ticket_key": "HPCSSUP-1",
                    "chunk_index": 0,
                    "chunk_type": "initial",
                },
            ),
            TextNode(
                id_="uuid-2",
                text="[CONVERSATION]\n<MESSAGE id=0001 role=REPORTER>\nHello\n</MESSAGE>",
                metadata={
                    "document_type": "helpdesk_ticket",
                    "ticket_key": "HPCSSUP-1",
                    "chunk_index": 1,
                    "chunk_type": "conversation",
                },
            ),
            TextNode(
                id_="uuid-3",
                text="[TICKET_SUMMARY] Ticket two",
                metadata={
                    "document_type": "helpdesk_ticket",
                    "ticket_key": "HPCSSUP-2",
                    "chunk_index": 0,
                    "chunk_type": "initial",
                },
            ),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "count_docstore_tickets.py",
            "--docstore-path",
            str(docstore_path),
        ],
    )

    count_docstore_tickets.main()

    assert capsys.readouterr().out.strip() == "2"


def test_main_resolves_persist_dir_from_config(capsys, monkeypatch, tmp_path: Path) -> None:
    persist_dir = tmp_path / "persist"
    docstore_path = persist_dir / "source_docstore.json"
    _persist_docstore(
        docstore_path,
        [
            TextNode(
                id_="HPCSSUP-1",
                text="<BEGIN_TICKET>\n[TICKET_KEY] HPCSSUP-1\n<END_TICKET>",
                metadata={"document_type": "helpdesk_ticket"},
            ),
        ],
    )

    fake_cfg = SimpleNamespace(
        raw={"storage_context": {"persist_dir": "persist"}},
        config_path=tmp_path / "config.yaml",
    )
    monkeypatch.setattr(count_docstore_tickets.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "count_docstore_tickets.py",
            "-c",
            str(tmp_path / "config.yaml"),
        ],
    )

    count_docstore_tickets.main()

    assert capsys.readouterr().out.strip() == "1"
