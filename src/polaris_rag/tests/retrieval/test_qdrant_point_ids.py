from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from llama_index.vector_stores.qdrant import QdrantVectorStore

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
    store.client = client
    store.vector_store = vector_store

    store.delete_ref_doc("HPCSSUP-96994")

    assert vector_store.delete_called_with is None
    assert delete_calls == []
