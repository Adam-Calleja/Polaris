from types import SimpleNamespace

from llama_index.core import StorageContext
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from polaris_rag.common.schemas import Document
from polaris_rag.retrieval.context_resolver import SupportTicketContextResolver
from polaris_rag.retrieval.document_store_factory import (
    add_chunks_to_docstore,
    add_documents_to_docstore,
    delete_ref_docs_from_docstore,
)


def test_add_chunks_to_docstore_sets_ref_doc_id_and_supports_ticket_cleanup() -> None:
    docstore = SimpleDocumentStore()
    storage = StorageContext.from_defaults(docstore=docstore)
    chunks = [
        SimpleNamespace(
            id="ticket-1::chunk::0000",
            parent_id="ticket-1",
            text="chunk-0",
            document_type="helpdesk_ticket",
            metadata={"chunk_index": 0},
        ),
        SimpleNamespace(
            id="ticket-1::chunk::0001",
            parent_id="ticket-1",
            text="chunk-1",
            document_type="helpdesk_ticket",
            metadata={"chunk_index": 1},
        ),
    ]

    add_chunks_to_docstore(storage=storage, chunks=chunks)

    assert docstore.get_document("ticket-1::chunk::0000").ref_doc_id == "ticket-1"
    assert docstore.get_document("ticket-1::chunk::0001").ref_doc_id == "ticket-1"

    delete_ref_docs_from_docstore(docstore, ["ticket-1"])

    assert docstore.docs == {}


def test_add_documents_to_docstore_stores_full_ticket_by_ticket_id() -> None:
    docstore = SimpleDocumentStore()
    documents = [
        Document(
            text="FULL TICKET",
            document_type="helpdesk_ticket",
            id="ticket-1",
            metadata={
                "resolved_at": "2025-01-01T11:00:00.000+0000",
                "time_to_resolution": 3600.0,
            },
        )
    ]

    add_documents_to_docstore(docstore, documents)

    stored = docstore.get_document("ticket-1")
    assert stored is not None
    assert stored.text == "FULL TICKET"
    assert stored.metadata["resolved_at"] == "2025-01-01T11:00:00.000+0000"
    assert stored.metadata["time_to_resolution"] == 3600.0


def test_context_resolver_expands_and_deduplicates_ticket_chunks() -> None:
    source_document_store = SimpleDocumentStore()
    add_documents_to_docstore(
        source_document_store,
        [
            Document(
                text="FULL TICKET",
                document_type="helpdesk_ticket",
                id="ticket-1",
                metadata={"resolved_at": "2025-01-01T11:00:00.000+0000"},
            )
        ],
    )
    resolver = SupportTicketContextResolver(source_document_store=source_document_store)

    raw_nodes = [
        NodeWithScore(
            node=TextNode(
                text="chunk-1",
                id_="ticket-1::chunk::0001",
                metadata={"document_type": "helpdesk_ticket", "parent_id": "ticket-1"},
            ),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(
                text="chunk-2",
                id_="ticket-1::chunk::0002",
                metadata={"document_type": "helpdesk_ticket", "parent_id": "ticket-1"},
            ),
            score=0.8,
        ),
        NodeWithScore(
            node=TextNode(
                text="HTML CHUNK",
                id_="doc-html::chunk::0001",
                metadata={"document_type": "html"},
            ),
            score=0.7,
        ),
    ]

    resolved = resolver.resolve(raw_nodes)

    assert len(resolved) == 2
    assert resolved[0].node.id_ == "ticket-1"
    assert resolved[0].node.text == "FULL TICKET"
    assert resolved[1].node.id_ == "doc-html::chunk::0001"
    assert resolved[1].node.text == "HTML CHUNK"


def test_context_resolver_falls_back_to_first_raw_chunk_when_ticket_lookup_missing() -> None:
    resolver = SupportTicketContextResolver(source_document_store=SimpleDocumentStore())
    raw_nodes = [
        NodeWithScore(
            node=TextNode(
                text="chunk-1",
                id_="ticket-1::chunk::0001",
                metadata={"document_type": "helpdesk_ticket", "parent_id": "ticket-1"},
            ),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(
                text="chunk-2",
                id_="ticket-1::chunk::0002",
                metadata={"document_type": "helpdesk_ticket", "parent_id": "ticket-1"},
            ),
            score=0.8,
        ),
    ]

    resolved = resolver.resolve(raw_nodes)

    assert len(resolved) == 1
    assert resolved[0].node.id_ == "ticket-1::chunk::0001"
    assert resolved[0].node.text == "chunk-1"
