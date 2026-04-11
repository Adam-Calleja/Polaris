import asyncio

import pytest

pytest.importorskip("ragas.metrics.collections")

import polaris_rag.evaluation.metrics as metrics_module
from polaris_rag.evaluation.metrics import instantiate_metrics, resolve_metric_specs


def test_auto_gates_summary_score_when_reference_contexts_missing() -> None:
    dataset_columns = {"user_input", "response", "reference", "retrieved_contexts"}

    specs, skipped = resolve_metric_specs(
        dataset_columns=dataset_columns,
        requested_metrics=None,
        auto_gate=True,
    )

    metric_names = [spec.name for spec in specs]
    skipped_map = {name: reason for name, reason in skipped}

    assert "summary_score" not in metric_names
    assert "summary_score" in skipped_map
    assert "reference_contexts" in skipped_map["summary_score"]


def test_unknown_metric_raises_key_error() -> None:
    dataset_columns = {"user_input", "response", "reference", "retrieved_contexts"}

    try:
        resolve_metric_specs(
            dataset_columns=dataset_columns,
            requested_metrics=["not_a_metric"],
            auto_gate=True,
        )
    except KeyError as exc:
        assert "Unknown metric" in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown metric")


def test_retrieval_metrics_are_available_when_reference_context_ids_exist() -> None:
    dataset_columns = {"retrieved_context_ids", "reference_context_ids"}

    specs, skipped = resolve_metric_specs(
        dataset_columns=dataset_columns,
        requested_metrics=["retrieval_recall_at_k", "retrieval_ndcg_at_10"],
        auto_gate=True,
    )

    assert [spec.name for spec in specs] == ["retrieval_recall_at_k", "retrieval_ndcg_at_10"]
    assert skipped == []


def test_retrieval_metrics_compute_expected_scores() -> None:
    dataset_columns = {"retrieved_context_ids", "reference_context_ids"}
    specs, _ = resolve_metric_specs(
        dataset_columns=dataset_columns,
        requested_metrics=["retrieval_recall_at_k", "retrieval_ndcg_at_10"],
        auto_gate=True,
    )
    metrics = instantiate_metrics(specs, llm=None, embeddings=None)

    async def _score():  # noqa: ANN202
        recall = await metrics[0].ascore(
            retrieved_context_ids=["doc-1", "doc-3", "doc-2"],
            reference_context_ids=["doc-1", "doc-2"],
        )
        ndcg = await metrics[1].ascore(
            retrieved_context_ids=["doc-1", "doc-3", "doc-2"],
            reference_context_ids=["doc-1", "doc-2"],
        )
        return recall, ndcg

    recall, ndcg = asyncio.run(_score())

    assert recall == 1.0
    assert 0.9 < ndcg < 1.0


def test_factual_correctness_accepts_metric_config_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeFactualCorrectness:
        def __init__(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)

    monkeypatch.setattr(metrics_module, "FactualCorrectness", _FakeFactualCorrectness)

    specs, _ = resolve_metric_specs(
        dataset_columns={"response", "reference"},
        requested_metrics=["factual_correctness"],
        auto_gate=True,
    )

    metrics = instantiate_metrics(
        specs,
        llm="fake-llm",
        embeddings=None,
        metric_config={
            "factual_correctness": {
                "mode": "precision",
                "atomicity": "low",
                "coverage": "high",
            }
        },
    )

    assert len(metrics) == 1
    assert captured["llm"] == "fake-llm"
    assert captured["mode"] == "precision"
    assert captured["atomicity"] == "low"
    assert captured["coverage"] == "high"
