from argparse import Namespace
from dataclasses import dataclass

from polaris_rag.cli import evaluate_rag
from polaris_rag.evaluation.evaluation_dataset import PrepProgressEvent


@dataclass
class _DummyContainer:
    pipeline: object


def _fake_build_prepared_rows(**kwargs):  # noqa: ANN003
    callback = kwargs.get("progress_callback")
    if callback:
        callback(
            PrepProgressEvent(
                completed=1,
                total=1,
                successes=1,
                failures=0,
                elapsed_seconds=0.1,
                mode="pipeline",
                last_error=None,
            )
        )
    return [
        {
            "id": "row-1",
            "user_input": "Q1",
            "reference": "A1",
            "response": "R1",
            "retrieved_contexts": ["ctx-1"],
            "retrieved_context_ids": ["doc-1"],
            "metadata": {},
        }
    ]


def test_resolve_prepared_rows_adds_manifest_stats(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    prepared_path = tmp_path / "prepared.json"

    monkeypatch.setattr(evaluate_rag, "build_container", lambda cfg: _DummyContainer(pipeline=object()))
    monkeypatch.setattr(
        evaluate_rag,
        "build_prepared_rows",
        _fake_build_prepared_rows,
    )

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=str(prepared_path),
        reuse_prepared=False,
        generation_workers=1,
    )

    rows, manifest = evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1}},
        show_progress=False,
    )

    assert len(rows) == 1
    assert manifest["prepared_source"] == "generated"
    assert manifest["prep_total_rows"] == 1
    assert manifest["prep_success_rows"] == 1
    assert manifest["prep_failed_rows"] == 0
    assert "prep_elapsed_seconds" in manifest
    assert "prep_rate_rows_per_second" in manifest


def test_resolve_prepared_rows_passes_progress_callback_when_enabled(
    monkeypatch, tmp_path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")
    seen_callbacks: list[object] = []

    def _capture_build_prepared_rows(**kwargs):  # noqa: ANN003
        seen_callbacks.append(kwargs.get("progress_callback"))
        return _fake_build_prepared_rows(**kwargs)

    monkeypatch.setattr(evaluate_rag, "build_container", lambda cfg: _DummyContainer(pipeline=object()))
    monkeypatch.setattr(evaluate_rag, "build_prepared_rows", _capture_build_prepared_rows)

    args = Namespace(
        dataset_path=str(dataset_path),
        prepared_path=None,
        reuse_prepared=False,
        generation_workers=1,
    )

    evaluate_rag._resolve_prepared_rows(
        cfg=object(),  # type: ignore[arg-type]
        args=args,
        eval_cfg={"dataset": {}, "generation": {"workers": 1}},
        show_progress=True,
    )

    assert seen_callbacks
    assert seen_callbacks[-1] is not None

