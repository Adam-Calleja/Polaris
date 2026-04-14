from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from llama_index.core.schema import NodeRelationship, ObjectType, RelatedNodeInfo, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from polaris_rag.cli import backfill_qdrant_docstores
from polaris_rag.retrieval.document_store_factory import chunk_document_store_path


def _payload_node(node_id: str, *, parent_id: str, text: str) -> TextNode:
    return TextNode(
        id_=node_id,
        text=text,
        metadata={"parent_id": parent_id, "document_type": "html"},
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
