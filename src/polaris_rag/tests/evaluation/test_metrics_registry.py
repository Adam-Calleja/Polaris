from polaris_rag.evaluation.metrics import resolve_metric_specs


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
