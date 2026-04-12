from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from llama_index.vector_stores.qdrant import QdrantVectorStore

from polaris_rag.common import DocumentChunk
from polaris_rag.retrieval.sparse_encoder import SparseEmbedding
from polaris_rag.retrieval.vector_store import (
    QdrantIndexStore,
    PolarisQdrantVectorStore,
    qdrant_point_id_from_node_id,
)


def test_qdrant_point_id_from_node_id_preserves_uuid_values():
    node_id = "bd0f3fc2-8f7e-4f69-8b7c-0b5f92dce4b4"

    assert qdrant_point_id_from_node_id(node_id) == node_id


def test_qdrant_point_id_from_node_id_maps_readable_chunk_ids_to_deterministic_uuid():
    node_id = "HPCSSUP-96994::chunk::0000"

    point_id = qdrant_point_id_from_node_id(node_id)

    assert point_id == qdrant_point_id_from_node_id(node_id)
    assert str(UUID(str(point_id))) == point_id


def test_polaris_qdrant_vector_store_translates_point_ids_but_preserves_logical_ids(monkeypatch):
    logical_ids = [
        "HPCSSUP-96994::chunk::0000",
        "bd0f3fc2-8f7e-4f69-8b7c-0b5f92dce4b4",
    ]
    points = [
        SimpleNamespace(id=logical_ids[0]),
        SimpleNamespace(id=logical_ids[1]),
    ]

    def fake_build_points(self, nodes, sparse_vector_name):
        return points, logical_ids

    monkeypatch.setattr(QdrantVectorStore, "_build_points", fake_build_points)

    store = object.__new__(PolarisQdrantVectorStore)

    built_points, returned_ids = PolarisQdrantVectorStore._build_points(store, [], "")

    assert returned_ids == logical_ids
    assert str(UUID(str(built_points[0].id))) == built_points[0].id
    assert built_points[1].id == logical_ids[1]


def test_delete_ref_doc_is_noop_when_collection_does_not_exist():
    vector_store = SimpleNamespace(
        collection_name="support_tickets_turn_based_qwen_3_8b",
        delete_called_with=None,
    )

    def _delete(ref_doc_id):
        vector_store.delete_called_with = ref_doc_id

    vector_store.delete = _delete

    delete_calls: list[str] = []
    client = SimpleNamespace(
        collection_exists=lambda collection_name: False,
        delete=lambda **kwargs: delete_calls.append(kwargs["collection_name"]),
    )

    store = object.__new__(QdrantIndexStore)
    store.collection_name = "support_tickets_turn_based_qwen_3_8b"
    store.client = client
    store.vector_store = vector_store

    store.delete_ref_doc("HPCSSUP-96994")

    assert vector_store.delete_called_with is None
    assert delete_calls == []


def test_insert_chunks_embeds_and_upserts_in_batches():
    events: list[tuple[str, list[str]]] = []

    class FakeEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            events.append(("embed", list(texts)))
            return [[float(len(text)), 0.0] for text in texts]

    store = object.__new__(QdrantIndexStore)
    store.embedder = FakeEmbedder()
    store.sparse_encoder = None

    def fake_upsert_chunks(*, chunks, dense_vectors, sparse_vectors, batch_size):
        events.append(("upsert", [chunk.id for chunk in chunks]))
        assert len(chunks) == len(dense_vectors) == len(sparse_vectors)
        assert batch_size == 2

    store._upsert_chunks = fake_upsert_chunks

    chunks = [
        DocumentChunk(
            parent_id="ticket-1",
            prev_id="",
            next_id="",
            text=f"chunk {idx}",
            document_type="helpdesk_ticket",
            id=f"chunk-{idx}",
        )
        for idx in range(5)
    ]

    store.insert_chunks(chunks, batch_size=2, use_async=False)

    assert events == [
        ("embed", ["chunk 0", "chunk 1"]),
        ("upsert", ["chunk-0", "chunk-1"]),
        ("embed", ["chunk 2", "chunk 3"]),
        ("upsert", ["chunk-2", "chunk-3"]),
        ("embed", ["chunk 4"]),
        ("upsert", ["chunk-4"]),
    ]


def test_ensure_collection_exists_bootstraps_empty_collection_schema():
    ensure_calls: list[dict[str, object]] = []

    class FakeEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["polaris collection bootstrap"]
            return [[1.0, 2.0, 3.0]]

    store = object.__new__(QdrantIndexStore)
    store.collection_name = "support_tickets"
    store.embedder = FakeEmbedder()
    store.sparse_encoder = object()
    store.client = SimpleNamespace(collection_exists=lambda collection_name: False)
    store._ensure_collection = lambda *, dense_dim, enable_sparse: ensure_calls.append(
        {
            "dense_dim": dense_dim,
            "enable_sparse": enable_sparse,
        }
    )

    store.ensure_collection_exists()

    assert ensure_calls == [{"dense_dim": 3, "enable_sparse": True}]


def test_recreate_collection_resets_then_bootstraps_schema():
    events: list[tuple[str, object]] = []

    store = object.__new__(QdrantIndexStore)
    store.collection_name = "support_tickets"
    store.client = SimpleNamespace(
        collection_exists=lambda collection_name: True,
        delete_collection=lambda *, collection_name: events.append(("delete", collection_name)),
    )
    store.ensure_collection_exists = lambda *, sample_text="polaris collection bootstrap": events.append(
        ("ensure", sample_text)
    )

    store.recreate_collection()

    assert events == [
        ("delete", "support_tickets"),
        ("ensure", "polaris collection bootstrap"),
    ]


def test_upsert_chunks_enables_sparse_schema_when_sparse_encoder_is_configured():
    ensure_calls: list[dict[str, object]] = []
    upsert_calls: list[str] = []

    store = object.__new__(QdrantIndexStore)
    store.collection_name = "support_tickets"
    store.dense_vector_name = "dense"
    store.sparse_vector_name = "sparse"
    store.sparse_encoder = object()
    store.client = SimpleNamespace(
        upsert=lambda **kwargs: upsert_calls.append(kwargs["collection_name"]),
    )

    def fake_ensure_collection(*, dense_dim: int, enable_sparse: bool) -> None:
        ensure_calls.append(
            {
                "dense_dim": dense_dim,
                "enable_sparse": enable_sparse,
            }
        )

    store._ensure_collection = fake_ensure_collection

    store._upsert_chunks(
        chunks=[
            DocumentChunk(
                parent_id="ticket-1",
                prev_id="",
                next_id="",
                text="hello",
                document_type="helpdesk_ticket",
                id="chunk-1",
            )
        ],
        dense_vectors=[[1.0, 2.0]],
        sparse_vectors=[SparseEmbedding(indices=[], values=[])],
        batch_size=16,
    )

    assert ensure_calls == [{"dense_dim": 2, "enable_sparse": True}]
    assert upsert_calls == ["support_tickets"]
