from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import types


_MODULE_PATH = Path(__file__).resolve().parents[4] / "scripts" / "ingest_html_documents.py"
_SPEC = spec_from_file_location("test_ingest_html_documents_module", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
sys.modules.setdefault("atlassian", types.SimpleNamespace(Jira=object))
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ingest_html_documents = _MODULE


def test_build_source_storage_context_uses_named_store(monkeypatch):
    called: dict[str, object] = {}

    def fake_build_storage_context(*, vector_store, docstore, persist_dir):
        called["vector_store"] = vector_store
        called["docstore"] = docstore
        called["persist_dir"] = persist_dir
        return "storage-context"

    monkeypatch.setattr(ingest_html_documents, "build_storage_context", fake_build_storage_context)

    container = SimpleNamespace(
        vector_stores={"docs": "docs-store"},
        doc_store="chunk-docstore",
    )

    result = ingest_html_documents._build_source_storage_context(container, "docs")

    assert result == "storage-context"
    assert called == {
        "vector_store": "docs-store",
        "docstore": "chunk-docstore",
        "persist_dir": None,
    }


def test_override_qdrant_collection_name_updates_selected_source():
    cfg = SimpleNamespace(raw={"vector_stores": {"docs": {"collection_name": "old"}}})

    ingest_html_documents._override_qdrant_collection_name(cfg, "docs", "new-collection")

    assert cfg.raw["vector_stores"]["docs"]["collection_name"] == "new-collection"


def test_parse_args_supports_source_and_batching_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_html_documents.py",
            "-c",
            "config/config.yaml",
            "-p",
            "https://docs.example.org",
            "--source",
            "docs",
            "--vector-batch-size",
            "8",
            "--embedding-workers",
            "3",
        ],
    )

    args = ingest_html_documents.parse_args()

    assert args.source == "docs"
    assert args.vector_batch_size == 8
    assert args.embedding_workers == 3
