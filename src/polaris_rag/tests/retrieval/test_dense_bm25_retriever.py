from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

from llama_index.core import StorageContext
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.retrieval.document_store_factory import add_chunks_to_docstore
from polaris_rag.retrieval.retriever import BM25IndexRetriever, DenseBM25HybridRetriever


def _chunk(chunk_id: str, *, parent_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        parent_id=parent_id,
        text=text,
        document_type="html",
        metadata={},
    )


class _FakeVectorStore:
    def __init__(self, results):  # noqa: ANN001
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

    def profile(self) -> dict[str, object]:
        return {
            "backend": "fake",
            "collection_name": "docs",
            "dense_model": "dense-model",
            "sparse_model": None,
        }

    def query_dense_nodes(self, query: str, **kwargs):  # noqa: ANN003
        self.calls.append({"query": query, **kwargs})
        return list(self._results)


def test_bm25_retriever_returns_keyword_hits_and_stamps_trace() -> None:
    docstore = SimpleDocumentStore()
    storage = StorageContext.from_defaults(docstore=docstore)
    add_chunks_to_docstore(
        storage=storage,
        chunks=[
            _chunk(
                "gromacs::chunk::0000",
                parent_id="gromacs",
                text="Installing GROMACS with module load gromacs and a fresh environment.",
            ),
            _chunk(
                "slurm::chunk::0000",
                parent_id="slurm",
                text="Submitting jobs with sbatch and checking queue state.",
            ),
        ],
    )

    retriever = BM25IndexRetriever(
        storage_context=storage,
        retrieval_profile={"bm25_top_k": 2},
    )

    results = retriever.retrieve("install gromacs module")

    assert [item.node.id_ for item in results] == ["gromacs::chunk::0000"]
    trace = results[0].node.metadata["retrieval_signal_trace"]
    assert trace["signal_type"] == "bm25"
    assert trace["bm25_rank"] == 1
    assert trace["bm25_score"] > 0.0


def test_dense_bm25_hybrid_retriever_fuses_dense_and_keyword_results() -> None:
    docstore = SimpleDocumentStore()
    storage = StorageContext.from_defaults(docstore=docstore)
    add_chunks_to_docstore(
        storage=storage,
        chunks=[
            _chunk(
                "shared::chunk::0000",
                parent_id="shared",
                text="Install GROMACS, then install GROMACS from the module tree and verify the environment.",
            ),
            _chunk(
                "bm25-only::chunk::0000",
                parent_id="bm25-only",
                text="GROMACS notes and a checklist.",
            ),
            _chunk(
                "dense-only::chunk::0000",
                parent_id="dense-only",
                text="Environment troubleshooting guidance.",
            ),
        ],
    )
    vector_store = _FakeVectorStore(
        [
            NodeWithScore(node=TextNode(id_="shared::chunk::0000", text="shared", metadata={}), score=0.9),
            NodeWithScore(node=TextNode(id_="dense-only::chunk::0000", text="dense", metadata={}), score=0.4),
        ]
    )

    retriever = DenseBM25HybridRetriever(
        storage_context=storage,
        vector_store=vector_store,
        retrieval_profile={
            "dense_top_k": 2,
            "bm25_top_k": 2,
            "top_k": 3,
            "fusion": {
                "type": "rrf",
                "rrf_k": 60,
                "signal_weights": {"dense": 2.0, "bm25": 1.0},
            },
        },
    )

    results = retriever.retrieve("install gromacs")

    assert [item.node.id_ for item in results] == [
        "shared::chunk::0000",
        "dense-only::chunk::0000",
        "bm25-only::chunk::0000",
    ]
    trace = results[0].node.metadata["retrieval_signal_trace"]
    assert trace["hybrid_kind"] == "dense_bm25"
    assert trace["dense_rank"] == 1
    assert trace["bm25_rank"] == 1
    assert trace["fusion_score"] > results[1].score
