from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from polaris_rag.cli import tune_validity_reranker


def test_weight_trials_build_expected_default_grid_size() -> None:
    trials = tune_validity_reranker._weight_trials()

    assert len(trials) == 486
    assert trials[0]["authority"] == 0.0
    assert trials[0]["scope_family"] == 0.0
    assert trials[-1]["freshness"] == 0.01


def test_select_best_trial_uses_objective_then_tie_breaks() -> None:
    best = tune_validity_reranker._select_best_trial(
        [
            tune_validity_reranker.TrialResult(
                weights={
                    "authority": 0.08,
                    "scope": 0.04,
                    "scope_family": 0.02,
                    "version": 0.04,
                    "status": 0.04,
                    "freshness": 0.01,
                },
                objective=0.75,
                metric_means={
                    "factual_correctness": 0.8,
                    "faithfulness": 0.7,
                    "context_precision_without_reference": 0.75,
                },
                reranker_fingerprint="a",
                prepared_rows=10,
            ),
            tune_validity_reranker.TrialResult(
                weights={
                    "authority": 0.04,
                    "scope": 0.04,
                    "scope_family": 0.02,
                    "version": 0.04,
                    "status": 0.04,
                    "freshness": 0.01,
                },
                objective=0.75,
                metric_means={
                    "factual_correctness": 0.8,
                    "faithfulness": 0.71,
                    "context_precision_without_reference": 0.75,
                },
                reranker_fingerprint="b",
                prepared_rows=10,
            ),
        ]
    )

    assert best.reranker_fingerprint == "b"


def test_main_writes_weights_and_manifest(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "validity_reranker.dev_v2.yaml"
    manifest_path = tmp_path / "validity_reranker.dev_v2.manifest.json"
    dataset_path = tmp_path / "dev.jsonl"
    dataset_path.write_text('{"id":"1","query":"Q1","expected_answer":"A1"}\n', encoding="utf-8")

    monkeypatch.setattr(
        tune_validity_reranker,
        "_parse_args",
        lambda: SimpleNamespace(
            config_file=str(tmp_path / "config.yaml"),
            dataset_path=str(dataset_path),
            output_path=str(output_path),
            manifest_path=str(manifest_path),
            generation_workers=None,
        ),
    )
    monkeypatch.setattr(
        tune_validity_reranker.GlobalConfig,
        "load",
        lambda path: SimpleNamespace(config_path=Path(path), raw={}),
    )
    monkeypatch.setattr(
        tune_validity_reranker,
        "load_raw_examples",
        lambda path: [{"id": "1", "query": "Q1", "expected_answer": "A1"}],
    )
    monkeypatch.setattr(
        tune_validity_reranker,
        "_weight_trials",
        lambda grid=None: [
            {"authority": 0.0, "scope": 0.0, "scope_family": 0.0, "version": 0.0, "status": 0.0, "freshness": 0.0},
            {"authority": 0.08, "scope": 0.04, "scope_family": 0.02, "version": 0.04, "status": 0.04, "freshness": 0.01},
        ],
    )

    def _fake_run_trial(*, cfg, raw_examples, weights, generation_workers):  # noqa: ANN001
        _ = cfg, raw_examples, generation_workers
        objective = 0.5 if weights["authority"] == 0.0 else 0.8
        return tune_validity_reranker.TrialResult(
            weights=dict(weights),
            objective=objective,
            metric_means={
                "factual_correctness": objective,
                "faithfulness": objective,
                "context_precision_without_reference": objective,
            },
            reranker_fingerprint="fp-best" if objective > 0.5 else "fp-base",
            prepared_rows=1,
        )

    monkeypatch.setattr(tune_validity_reranker, "_run_trial", _fake_run_trial)

    tune_validity_reranker.main()

    assert output_path.exists()
    assert manifest_path.exists()
    assert "authority: 0.08" in output_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_trial"]["reranker_fingerprint"] == "fp-best"
