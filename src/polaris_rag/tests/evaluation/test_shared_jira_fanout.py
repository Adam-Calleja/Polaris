from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from polaris_rag.evaluation import shared_jira_fanout as fanout


def _entry(name: str, tmp_path: Path) -> fanout.SharedJiraConditionEntry:
    return fanout.SharedJiraConditionEntry(
        name=name,
        slug=name,
        preset="tickets_only",
        config_path=tmp_path / f"{name}.yaml",
        condition_spec={},
        ingest_spec={"kind": "jira"},
    )


def _plan(
    name: str,
    tmp_path: Path,
    **overrides,
) -> fanout.SharedJiraConditionPlan:
    entry = _entry(name, tmp_path)
    defaults = {
        "entry": entry,
        "cfg": SimpleNamespace(),
        "source": "tickets",
        "persist_dir": str(tmp_path / f"persist_{name}"),
        "collection_name": f"collection_{name}",
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "limit": None,
        "fetch_batch_size": 200,
        "exclude_keys": tuple(),
        "unwanted_summaries": tuple(),
        "chunking_strategy": "jira_turns_token",
        "chunk_size_tokens": 600,
        "chunk_overlap_tokens": 0,
        "conversion_engine": "native_jira",
        "conversion_options": {},
        "conversion_profile": '{"engine":"native_jira","options":{}}',
        "registry_artifact_path": str(tmp_path / "registry.json"),
        "embedding_workers": 1,
        "vector_batch_size": 16,
        "clear_target": True,
        "dump_processed": False,
        "dump_path": tmp_path / f"{name}.dump.txt",
    }
    defaults.update(overrides)
    return fanout.SharedJiraConditionPlan(**defaults)


def test_validate_condition_plans_rejects_mixed_fetch_profiles(tmp_path: Path) -> None:
    plans = [
        _plan("tickets_cs600_ov0", tmp_path),
        _plan("tickets_cs800_ov0", tmp_path, fetch_batch_size=100),
    ]

    with pytest.raises(ValueError, match="same Jira fetch/filter/preprocess profile"):
        fanout._validate_condition_plans(plans)


def test_validate_condition_plans_rejects_invalid_embedding_workers(tmp_path: Path) -> None:
    plans = [
        _plan("tickets_cs600_ov0", tmp_path),
        _plan("tickets_cs800_ov0", tmp_path, embedding_workers=2),
    ]

    with pytest.raises(ValueError, match="embedding workers above 1"):
        fanout._validate_condition_plans(plans)


def test_validate_condition_plans_rejects_mixed_chunking_strategies(tmp_path: Path) -> None:
    plans = [
        _plan("tickets_cs600_ov0", tmp_path),
        _plan("tickets_cs800_ov0", tmp_path, chunking_strategy="markdown_token"),
    ]

    with pytest.raises(ValueError, match="unsupported chunking strategy"):
        fanout._validate_condition_plans(plans)


def test_initialize_condition_runtime_recreates_collection_and_clears_persist_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recreate_calls: list[str] = []
    cleared_dirs: list[str] = []
    loaded_chunk_docstores: list[tuple[str | None, str]] = []
    loaded_docstores: list[str] = []

    class FakeVectorStore:
        def recreate_collection(self) -> None:
            recreate_calls.append("tickets")

    fake_container = SimpleNamespace(
        vector_stores={"tickets": FakeVectorStore()},
        doc_store="chunk-docstore",
    )

    monkeypatch.setattr(fanout, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(
        fanout,
        "build_storage_context",
        lambda *, vector_store, docstore, persist_dir: SimpleNamespace(
            vector_store=vector_store,
            docstore=docstore,
        ),
    )
    monkeypatch.setattr(fanout, "clear_persist_dir", lambda persist_dir: cleared_dirs.append(persist_dir))
    monkeypatch.setattr(
        fanout,
        "load_or_create_chunk_document_store",
        lambda *, persist_dir, source: loaded_chunk_docstores.append((persist_dir, source)) or "chunk-docstore",
    )
    monkeypatch.setattr(
        fanout,
        "load_or_create_source_document_store",
        lambda *, persist_dir: loaded_docstores.append(str(persist_dir)) or "source-docstore",
    )

    runtime = fanout._initialize_condition_runtime(_plan("tickets_cs600_ov0", tmp_path))

    assert recreate_calls == ["tickets"]
    assert cleared_dirs == [str(tmp_path / "persist_tickets_cs600_ov0")]
    assert loaded_chunk_docstores == [(None, "tickets")]
    assert loaded_docstores == [str(tmp_path / "persist_tickets_cs600_ov0")]
    assert runtime.source_document_store == "source-docstore"


def test_initialize_condition_runtime_resume_uses_existing_persisted_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recreate_calls: list[str] = []
    cleared_dirs: list[str] = []
    loaded_chunk_docstores: list[tuple[str | None, str]] = []
    loaded_docstores: list[str] = []
    storage_persist_dirs: list[str | None] = []
    persist_dir = tmp_path / "persist_tickets_cs600_ov0"
    persist_dir.mkdir()

    class FakeVectorStore:
        def recreate_collection(self) -> None:
            recreate_calls.append("tickets")

    fake_container = SimpleNamespace(
        vector_stores={"tickets": FakeVectorStore()},
        doc_store="chunk-docstore",
    )

    monkeypatch.setattr(fanout, "build_container", lambda cfg: fake_container)
    monkeypatch.setattr(
        fanout,
        "build_storage_context",
        lambda *, vector_store, docstore, persist_dir: storage_persist_dirs.append(persist_dir) or SimpleNamespace(
            vector_store=vector_store,
            docstore=docstore,
        ),
    )
    monkeypatch.setattr(fanout, "clear_persist_dir", lambda persist_dir: cleared_dirs.append(persist_dir))
    monkeypatch.setattr(
        fanout,
        "load_or_create_chunk_document_store",
        lambda *, persist_dir, source: loaded_chunk_docstores.append((persist_dir, source)) or "chunk-docstore",
    )
    monkeypatch.setattr(
        fanout,
        "load_or_create_source_document_store",
        lambda *, persist_dir: loaded_docstores.append(str(persist_dir)) or "source-docstore",
    )

    runtime = fanout._initialize_condition_runtime(
        _plan("tickets_cs600_ov0", tmp_path, persist_dir=str(persist_dir)),
        resume=True,
        checkpoint_state={
            "total_chunks": 5,
            "total_tickets": 3,
            "processed_batches": 2,
            "persist_count": 1,
        },
    )

    assert recreate_calls == []
    assert cleared_dirs == []
    assert storage_persist_dirs == [None]
    assert loaded_chunk_docstores == [(str(persist_dir), "tickets")]
    assert loaded_docstores == [str(persist_dir)]
    assert runtime.total_chunks == 5
    assert runtime.total_tickets == 3
    assert runtime.processed_batches == 2
    assert runtime.persist_count == 1


def test_run_shared_jira_fanout_index_fetches_once_processes_all_conditions_and_persists_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    entry_a = _entry("tickets_cs600_ov0", tmp_path)
    entry_b = _entry("tickets_cs800_ov0", tmp_path)
    plan_a = _plan("tickets_cs600_ov0", tmp_path, entry=entry_a, chunk_size_tokens=600)
    plan_b = _plan("tickets_cs800_ov0", tmp_path, entry=entry_b, chunk_size_tokens=800)
    plans = {
        entry_a.name: plan_a,
        entry_b.name: plan_b,
    }

    events: list[tuple[str, object]] = []
    delete_calls: list[tuple[str, tuple[str, ...]]] = []
    added_chunk_calls: list[tuple[str, list[str]]] = []
    added_document_calls: list[tuple[str, list[str]]] = []
    persist_docstore_calls: list[tuple[str, str]] = []

    class FakeVectorStore:
        def __init__(self, name: str) -> None:
            self._name = name

        def recreate_collection(self) -> None:
            events.append(("recreate", self._name))

        def insert_chunks(self, chunks, batch_size, use_async) -> None:
            events.append(("insert", (self._name, [chunk.parent_id for chunk in chunks], batch_size, use_async)))

        def delete_ref_docs(self, ids) -> None:
            delete_calls.append((self._name, tuple(ids)))

    def _make_runtime(plan: fanout.SharedJiraConditionPlan) -> fanout.SharedJiraConditionRuntime:
        return fanout.SharedJiraConditionRuntime(
            plan=plan,
            container=SimpleNamespace(token_counter="token-counter"),
            storage_context=SimpleNamespace(
                vector_store=FakeVectorStore(plan.entry.name),
                docstore=f"docstore:{plan.entry.name}",
            ),
            source_document_store=f"source-docstore:{plan.entry.name}",
        )

    monkeypatch.setattr(
        fanout,
        "_build_condition_plan",
        lambda entry, clear_targets, default_vector_batch_size=None: plans[entry.name],
    )
    monkeypatch.setattr(
        fanout,
        "_initialize_condition_runtime",
        lambda plan, resume=False, checkpoint_state=None: _make_runtime(plan),
    )

    def _iter_batches(*, start_date, end_date, limit, cfg, exclude_keys, batch_size):
        events.append(("iter", (start_date, end_date, limit, tuple(exclude_keys), batch_size)))
        events.append(("fetch", 1))
        yield [
            {"key": "HPCSSUP-1", "fields": {"summary": "Summary 1"}},
            {"key": "HPCSSUP-2", "fields": {"summary": "Summary 2"}},
        ]
        events.append(("fetch", 2))
        yield [
            {"key": "HPCSSUP-3", "fields": {"summary": "Summary 3"}},
        ]

    monkeypatch.setattr(fanout, "iter_support_ticket_batches", _iter_batches)
    monkeypatch.setattr(
        fanout,
        "filter_jira_tickets",
        lambda tickets, exclude_keys, unwanted_summaries: list(tickets),
    )
    monkeypatch.setattr(
        fanout,
        "prepare_jira_tickets_for_chunking",
        lambda tickets, **kwargs: [SimpleNamespace(id=ticket["key"]) for ticket in tickets],
    )
    monkeypatch.setattr(
        fanout,
        "chunk_processed_jira_tickets",
        lambda processed_tickets, **kwargs: [
            SimpleNamespace(
                id=f"{ticket.id}::chunk::0000",
                parent_id=ticket.id,
                text=ticket.id,
                document_type="helpdesk_ticket",
                metadata={},
            )
            for ticket in processed_tickets
        ],
    )
    monkeypatch.setattr(
        fanout,
        "add_chunks_to_docstore",
        lambda storage, chunks: added_chunk_calls.append(
            (str(storage.docstore), [chunk.parent_id for chunk in chunks])
        ) or len(chunks),
    )
    monkeypatch.setattr(
        fanout,
        "add_documents_to_docstore",
        lambda docstore, documents: added_document_calls.append(
            (str(docstore), [document.id for document in documents])
        ) or len(documents),
    )
    monkeypatch.setattr(
        fanout,
        "persist_docstore",
        lambda docstore, persist_path: persist_docstore_calls.append((str(docstore), str(persist_path))),
    )
    monkeypatch.setattr(
        fanout,
        "chunk_document_store_path",
        lambda persist_dir, source: str(Path(persist_dir) / f"chunk_docstore.{source}.json"),
    )
    monkeypatch.setattr(
        fanout,
        "source_document_store_path",
        lambda persist_dir: str(Path(persist_dir) / "source_docstore.json"),
    )

    class _ImmediateFuture:
        def __init__(self, fn, *args, **kwargs) -> None:
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        def result(self):
            return self._fn(*self._args, **self._kwargs)

    class _ImmediateExecutor:
        def __init__(self, *, max_workers: int) -> None:
            events.append(("executor", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            runtime = args[0]
            events.append(("submit", runtime.plan.entry.name))
            return _ImmediateFuture(fn, *args, **kwargs)

    monkeypatch.setattr(fanout, "ThreadPoolExecutor", _ImmediateExecutor)

    result = fanout.run_shared_jira_fanout_index(
        condition_entries=[entry_a, entry_b],
        index_strategy={
            "type": "shared_jira_fanout",
            "clear_targets": True,
            "condition_parallelism": 6,
            "persist_every_batches": 0,
        },
        dry_run=False,
    )

    assert [event for event in events if event[0] == "iter"] == [
        ("iter", ("2024-01-01", "2024-02-01", None, tuple(), 200))
    ]
    assert [event for event in events if event[0] == "submit"] == [
        ("submit", "tickets_cs600_ov0"),
        ("submit", "tickets_cs800_ov0"),
        ("submit", "tickets_cs600_ov0"),
        ("submit", "tickets_cs800_ov0"),
    ]
    fetch_two_index = events.index(("fetch", 2))
    first_batch_insert_indexes = [
        index
        for index, event in enumerate(events)
        if event[0] == "insert" and event[1][1] == ["HPCSSUP-1", "HPCSSUP-2"]
    ]
    assert first_batch_insert_indexes
    assert fetch_two_index > max(first_batch_insert_indexes)
    assert delete_calls == []
    assert len(persist_docstore_calls) == 4
    assert result["summary"]["condition_parallelism"] == 2
    assert result["summary"]["fetched_tickets"] == 3
    assert result["summary"]["processed_tickets"] == 3
    assert result["summary"]["processed_batches"] == 2
    assert {record["name"]: record["indexed_chunks"] for record in result["conditions"]} == {
        "tickets_cs600_ov0": 3,
        "tickets_cs800_ov0": 3,
    }
    assert {record["name"]: record["persist_count"] for record in result["conditions"]} == {
        "tickets_cs600_ov0": 1,
        "tickets_cs800_ov0": 1,
    }
    assert added_chunk_calls == [
        ("docstore:tickets_cs600_ov0", ["HPCSSUP-1", "HPCSSUP-2"]),
        ("docstore:tickets_cs800_ov0", ["HPCSSUP-1", "HPCSSUP-2"]),
        ("docstore:tickets_cs600_ov0", ["HPCSSUP-3"]),
        ("docstore:tickets_cs800_ov0", ["HPCSSUP-3"]),
    ]
    assert added_document_calls == [
        ("source-docstore:tickets_cs600_ov0", ["HPCSSUP-1", "HPCSSUP-2"]),
        ("source-docstore:tickets_cs800_ov0", ["HPCSSUP-1", "HPCSSUP-2"]),
        ("source-docstore:tickets_cs600_ov0", ["HPCSSUP-3"]),
        ("source-docstore:tickets_cs800_ov0", ["HPCSSUP-3"]),
    ]


def test_run_shared_jira_fanout_index_resumes_from_checkpoint_and_skips_completed_batches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    entry = _entry("tickets_cs600_ov0", tmp_path)
    plan = _plan("tickets_cs600_ov0", tmp_path, entry=entry)
    init_calls: list[tuple[bool, dict[str, int] | None]] = []
    persisted: list[str] = []
    cleared_checkpoint_paths: list[str] = []
    written_checkpoints: list[dict[str, object]] = []
    inserted_batches: list[list[str]] = []

    checkpoint_payload = {
        "version": fanout.SHARED_JIRA_FANOUT_CHECKPOINT_VERSION,
        "selected_conditions": [entry.name],
        "plan_signatures": {
            entry.name: fanout._condition_checkpoint_signature(plan),
        },
        "last_completed_fetched_batch_index": 1,
        "summary": {
            "fetched_tickets": 2,
            "processed_tickets": 2,
            "processed_batches": 1,
        },
        "conditions": {
            entry.name: {
                "total_chunks": 2,
                "total_tickets": 2,
                "processed_batches": 1,
                "persist_count": 1,
            }
        },
    }

    monkeypatch.setattr(
        fanout,
        "_build_condition_plan",
        lambda entry, clear_targets, default_vector_batch_size=None: plan,
    )
    monkeypatch.setattr(fanout, "_load_checkpoint", lambda checkpoint_path: checkpoint_payload)
    monkeypatch.setattr(
        fanout,
        "_write_checkpoint",
        lambda checkpoint_path, payload: written_checkpoints.append(dict(payload)),
    )
    monkeypatch.setattr(
        fanout,
        "_clear_checkpoint",
        lambda checkpoint_path: cleared_checkpoint_paths.append(str(checkpoint_path)),
    )

    def _initialize_runtime(plan, resume=False, checkpoint_state=None):
        init_calls.append((resume, dict(checkpoint_state or {})))
        return fanout.SharedJiraConditionRuntime(
            plan=plan,
            container=SimpleNamespace(token_counter="token-counter"),
            storage_context=SimpleNamespace(
                vector_store=SimpleNamespace(
                    insert_chunks=lambda chunks, batch_size, use_async: inserted_batches.append(
                        [chunk.parent_id for chunk in chunks]
                    )
                ),
                docstore="docstore:tickets_cs600_ov0",
            ),
            source_document_store="source-docstore:tickets_cs600_ov0",
            total_chunks=int((checkpoint_state or {}).get("total_chunks", 0)),
            total_tickets=int((checkpoint_state or {}).get("total_tickets", 0)),
            processed_batches=int((checkpoint_state or {}).get("processed_batches", 0)),
            persist_count=int((checkpoint_state or {}).get("persist_count", 0)),
        )

    monkeypatch.setattr(fanout, "_initialize_condition_runtime", _initialize_runtime)
    monkeypatch.setattr(
        fanout,
        "iter_support_ticket_batches",
        lambda **kwargs: iter(
            [
                [{"key": "HPCSSUP-1", "fields": {"summary": "Summary 1"}}],
                [{"key": "HPCSSUP-2", "fields": {"summary": "Summary 2"}}],
            ]
        ),
    )
    monkeypatch.setattr(
        fanout,
        "filter_jira_tickets",
        lambda tickets, exclude_keys, unwanted_summaries: list(tickets),
    )
    monkeypatch.setattr(
        fanout,
        "prepare_jira_tickets_for_chunking",
        lambda tickets, **kwargs: [SimpleNamespace(id=ticket["key"]) for ticket in tickets],
    )
    monkeypatch.setattr(
        fanout,
        "chunk_processed_jira_tickets",
        lambda processed_tickets, **kwargs: [
            SimpleNamespace(
                id=f"{ticket.id}::chunk::0000",
                parent_id=ticket.id,
                text=ticket.id,
                document_type="helpdesk_ticket",
                metadata={},
            )
            for ticket in processed_tickets
        ],
    )
    monkeypatch.setattr(fanout, "add_chunks_to_docstore", lambda storage, chunks: len(chunks))
    monkeypatch.setattr(fanout, "add_documents_to_docstore", lambda docstore, documents: len(documents))
    monkeypatch.setattr(
        fanout,
        "persist_docstore",
        lambda docstore, persist_path: persisted.append(f"docstore:{persist_path}"),
    )
    monkeypatch.setattr(
        fanout,
        "chunk_document_store_path",
        lambda persist_dir, source: str(Path(persist_dir) / f"chunk_docstore.{source}.json"),
    )
    monkeypatch.setattr(
        fanout,
        "source_document_store_path",
        lambda persist_dir: str(Path(persist_dir) / "source_docstore.json"),
    )

    class _ImmediateFuture:
        def __init__(self, fn, *args, **kwargs) -> None:
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        def result(self):
            return self._fn(*self._args, **self._kwargs)

    class _ImmediateExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _ImmediateFuture(fn, *args, **kwargs)

    monkeypatch.setattr(fanout, "ThreadPoolExecutor", _ImmediateExecutor)

    result = fanout.run_shared_jira_fanout_index(
        condition_entries=[entry],
        index_strategy={
            "type": "shared_jira_fanout",
            "clear_targets": True,
            "condition_parallelism": 1,
            "persist_every_batches": 1,
            "resume_from_checkpoint": True,
        },
        dry_run=False,
        stage_dir=tmp_path,
    )

    assert init_calls == [
        (
            True,
            {
                "total_chunks": 2,
                "total_tickets": 2,
                "processed_batches": 1,
                "persist_count": 1,
            },
        )
    ]
    assert inserted_batches == [["HPCSSUP-2"]]
    assert result["summary"]["fetched_tickets"] == 3
    assert result["summary"]["processed_tickets"] == 3
    assert result["summary"]["processed_batches"] == 2
    assert result["conditions"][0]["indexed_chunks"] == 3
    assert result["conditions"][0]["persist_count"] == 2
    assert len(written_checkpoints) == 1
    assert written_checkpoints[0]["last_completed_fetched_batch_index"] == 2
    assert cleared_checkpoint_paths == [
        str(tmp_path.resolve() / fanout.SHARED_JIRA_FANOUT_CHECKPOINT_FILENAME)
    ]
