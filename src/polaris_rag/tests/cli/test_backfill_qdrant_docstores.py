from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from llama_index.core.schema import NodeRelationship, ObjectType, RelatedNodeInfo, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from polaris_rag.cli import backfill_qdrant_docstores
from polaris_rag.retrieval.document_store_factory import (
    chunk_document_store_path,
    source_document_store_path,
)


def _payload_node(
    node_id: str,
    *,
    parent_id: str,
    text: str,
    document_type: str = "html",
    metadata: dict[str, object] | None = None,
) -> TextNode:
    node_metadata = {"parent_id": parent_id, "document_type": document_type}
    if metadata:
        node_metadata.update(metadata)
    return TextNode(
        id_=node_id,
        text=text,
        metadata=node_metadata,
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(
                node_id=parent_id,
                node_type=ObjectType.DOCUMENT,
                metadata={},
                hash=None,
            )
        },
    )


class _FakeVectorStore:
    def __init__(self, nodes: list[TextNode]) -> None:
        self._nodes = list(nodes)
        self.scroll_batch_sizes: list[int] = []

    def iter_payload_nodes(self, *, batch_size: int = 512, **_: object):
        self.scroll_batch_sizes.append(batch_size)
        for node in self._nodes:
            yield node


def test_main_backfills_per_source_chunk_docstores(tmp_path: Path, monkeypatch) -> None:
    persist_dir = tmp_path / "persist"
    fake_cfg = SimpleNamespace(
        raw={"storage_context": {"persist_dir": str(persist_dir)}},
        config_path=tmp_path / "config.yaml",
    )
    docs_store = _FakeVectorStore(
        [
            _payload_node(
                "docs::chunk::0000",
                parent_id="docs",
                text="Install GROMACS from the docs collection.",
            )
        ]
    )
    tickets_store = _FakeVectorStore(
        [
            _payload_node(
                "ticket-1::chunk::0000",
                parent_id="ticket-1",
                text="Ticket discussion about module loading.",
            )
        ]
    )
    fake_container = SimpleNamespace(
        vector_stores={
            "docs": docs_store,
            "tickets": tickets_store,
        }
    )

    monkeypatch.setattr(backfill_qdrant_docstores.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(backfill_qdrant_docstores, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_qdrant_docstores.py",
            "-c",
            "config/config.yaml",
            "--scroll-batch-size",
            "7",
            "--write-batch-size",
            "2",
        ],
    )

    backfill_qdrant_docstores.main()

    docs_path = chunk_document_store_path(persist_dir, "docs")
    tickets_path = chunk_document_store_path(persist_dir, "tickets")
    assert Path(docs_path).exists()
    assert Path(tickets_path).exists()
    assert docs_store.scroll_batch_sizes == [7]
    assert tickets_store.scroll_batch_sizes == [7]

    docs_docstore = SimpleDocumentStore.from_persist_path(docs_path)
    tickets_docstore = SimpleDocumentStore.from_persist_path(tickets_path)
    assert docs_docstore.get_document("docs::chunk::0000").text == "Install GROMACS from the docs collection."
    assert tickets_docstore.get_document("ticket-1::chunk::0000").ref_doc_id == "ticket-1"


def test_reconstruct_ticket_node_reassembles_ticket_text() -> None:
    reconstructed = backfill_qdrant_docstores._reconstruct_ticket_node(
        "HPCSSUP-1",
        [
            _payload_node(
                "HPCSSUP-1::chunk::0002",
                parent_id="HPCSSUP-1",
                document_type="helpdesk_ticket",
                text=(
                    "[TICKET_SUMMARY] Example summary\n\n"
                    "[CONTEXT]\nignored overlap\n\n"
                    "[CONVERSATION]\n"
                    "<MESSAGE id=0002b role=HELPDESK_ASSIGNEE part=2/2 original_id=0002>\n"
                    "Second paragraph\n"
                    "</MESSAGE>"
                ),
                metadata={
                    "chunk_type": "conversation",
                    "chunk_index": 2,
                    "ticket_key": "HPCSSUP-1",
                    "status": "RESOLVED",
                    "created_at": "2025-01-01T12:00:00.000+0000",
                    "summary": "Example summary",
                },
            ),
            _payload_node(
                "HPCSSUP-1::chunk::0000",
                parent_id="HPCSSUP-1",
                document_type="helpdesk_ticket",
                text="[TICKET_SUMMARY] Example summary\n\n[INITIAL_DESCRIPTION]\nOriginal description",
                metadata={
                    "chunk_type": "initial",
                    "chunk_index": 0,
                    "ticket_key": "HPCSSUP-1",
                    "status": "RESOLVED",
                    "created_at": "2025-01-01T12:00:00.000+0000",
                    "summary": "Example summary",
                },
            ),
            _payload_node(
                "HPCSSUP-1::chunk::0001",
                parent_id="HPCSSUP-1",
                document_type="helpdesk_ticket",
                text=(
                    "[TICKET_SUMMARY] Example summary\n\n"
                    "[CONVERSATION]\n"
                    "<MESSAGE id=0001 role=REPORTER>\n"
                    "Initial message\n"
                    "</MESSAGE>\n"
                    "<MESSAGE id=0002a role=HELPDESK_ASSIGNEE part=1/2 original_id=0002>\n"
                    "First paragraph\n"
                    "</MESSAGE>"
                ),
                metadata={
                    "chunk_type": "conversation",
                    "chunk_index": 1,
                    "ticket_key": "HPCSSUP-1",
                    "status": "RESOLVED",
                    "created_at": "2025-01-01T12:00:00.000+0000",
                    "summary": "Example summary",
                },
            ),
        ],
    )

    assert reconstructed is not None
    assert reconstructed.id_ == "HPCSSUP-1"
    assert reconstructed.metadata["summary"] == "Example summary"
    assert "[INITIAL_DESCRIPTION]\nOriginal description" in reconstructed.text
    assert "<MESSAGE id=0001 role=REPORTER>\nInitial message\n</MESSAGE>" in reconstructed.text
    assert (
        "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>\nFirst paragraph\n\nSecond paragraph\n</MESSAGE>"
        in reconstructed.text
    )
    assert reconstructed.text.endswith("<END_TICKET>")


def test_main_backfills_ticket_source_docstore(tmp_path: Path, monkeypatch) -> None:
    persist_dir = tmp_path / "persist"
    fake_cfg = SimpleNamespace(
        raw={"storage_context": {"persist_dir": str(persist_dir)}},
        config_path=tmp_path / "config.yaml",
    )
    tickets_store = _FakeVectorStore(
        [
            _payload_node(
                "HPCSSUP-1::chunk::0000",
                parent_id="HPCSSUP-1",
                document_type="helpdesk_ticket",
                text="[TICKET_SUMMARY] Example summary\n\n[INITIAL_DESCRIPTION]\nOriginal description",
                metadata={
                    "chunk_type": "initial",
                    "chunk_index": 0,
                    "ticket_key": "HPCSSUP-1",
                    "status": "RESOLVED",
                    "created_at": "2025-01-01T12:00:00.000+0000",
                    "summary": "Example summary",
                },
            ),
            _payload_node(
                "HPCSSUP-1::chunk::0001",
                parent_id="HPCSSUP-1",
                document_type="helpdesk_ticket",
                text=(
                    "[TICKET_SUMMARY] Example summary\n\n"
                    "[CONVERSATION]\n"
                    "<MESSAGE id=0001 role=REPORTER>\n"
                    "Initial message\n"
                    "</MESSAGE>"
                ),
                metadata={
                    "chunk_type": "conversation",
                    "chunk_index": 1,
                    "ticket_key": "HPCSSUP-1",
                    "status": "RESOLVED",
                    "created_at": "2025-01-01T12:00:00.000+0000",
                    "summary": "Example summary",
                },
            ),
        ]
    )
    fake_container = SimpleNamespace(vector_stores={"tickets": tickets_store})

    monkeypatch.setattr(backfill_qdrant_docstores.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(backfill_qdrant_docstores, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_qdrant_docstores.py",
            "-c",
            "config/config.yaml",
            "--source",
            "tickets",
        ],
    )

    backfill_qdrant_docstores.main()

    source_path = source_document_store_path(persist_dir)
    assert Path(source_path).exists()

    source_docstore = SimpleDocumentStore.from_persist_path(source_path)
    stored = source_docstore.get_document("HPCSSUP-1")
    assert stored is not None
    assert stored.metadata["document_type"] == "helpdesk_ticket"
    assert stored.metadata["summary"] == "Example summary"
    assert stored.text.startswith("<BEGIN_TICKET>\n[TICKET_KEY] HPCSSUP-1")
    assert "[INITIAL_DESCRIPTION]\nOriginal description" in stored.text
    assert "<MESSAGE id=0001 role=REPORTER>\nInitial message\n</MESSAGE>" in stored.text
