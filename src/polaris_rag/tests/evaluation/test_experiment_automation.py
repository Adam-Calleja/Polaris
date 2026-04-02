from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from polaris_rag.evaluation import experiment_automation as automation


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def test_render_stage_condition_config_merges_overrides(tmp_path: Path) -> None:
    base_config = tmp_path / "config.yaml"
    base_config.write_text("extends: 'config.base.yaml'\n", encoding="utf-8")
    output_path = tmp_path / "generated.yaml"

    manifest_path = _write_manifest(
        tmp_path,
        {
            "artifacts_root": "artifacts/experiments",
            "base_config": "config.yaml",
            "defaults": {
                "config_overrides": {
                    "evaluation": {
                        "generation": {
                            "mode": "pipeline",
                        }
                    }
                }
            },
            "stages": {
                "stage4_source_ablation": {
                    "type": "evaluation_grid",
                    "dataset_path": "dataset.jsonl",
                    "config_overrides": {
                        "retriever": {
                            "top_k": 5,
                        }
                    },
                    "conditions": [
                        {
                            "name": "naive_combined",
                            "preset": "naive_combined",
                            "config_overrides": {
                                "retriever": {
                                    "final_top_k": 6,
                                }
                            },
                        }
                    ],
                }
            },
        },
    )

    rendered = automation.render_stage_condition_config(
        manifest_path=manifest_path,
        stage_name="stage4_source_ablation",
        condition_name="naive_combined",
        output_path=output_path,
    )

    assert rendered == output_path.resolve()
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["extends"] == str(base_config.resolve())
    assert payload["evaluation"]["generation"]["mode"] == "pipeline"
    assert payload["retriever"]["top_k"] == 5
    assert payload["retriever"]["final_top_k"] == 6


def test_run_experiment_stage_builds_ingest_and_eval_commands(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    base_config = tmp_path / "config.yaml"
    base_config.write_text("prompt_name: test\n", encoding="utf-8")

    manifest_path = _write_manifest(
        tmp_path,
        {
            "artifacts_root": "artifacts/experiments",
            "base_config": "config.yaml",
            "defaults": {
                "run_options": {
                    "generation_mode": "pipeline",
                    "generation_workers": 1,
                    "no_progress": True,
                }
            },
            "stages": {
                "stage0b_docs_chunking": {
                    "type": "evaluation_grid",
                    "dataset_path": "dev.jsonl",
                    "annotations_file": "annotations.csv",
                    "repeats": 2,
                    "conditions": [
                        {
                            "name": "docs_cs400_ov0",
                            "preset": "docs_only",
                            "ingest": {
                                "kind": "html",
                                "homepage": "https://docs.example.org",
                                "ingest_internal_links": True,
                                "qdrant_collection_name": "exp_docs_cs400_ov0",
                                "chunking_strategy": "markdown_token",
                                "chunk_size_tokens": 400,
                                "chunk_overlap_tokens": 0,
                            },
                        }
                    ],
                }
            },
        },
    )

    calls: list[list[str]] = []

    def _fake_run(command, check, cwd):
        calls.append(list(command))
        assert check is True
        assert Path(cwd) == repo_root
        return None

    monkeypatch.setattr(automation, "REPO_ROOT", repo_root)
    monkeypatch.setattr(automation.subprocess, "run", _fake_run)

    record = automation.run_experiment_stage(
        manifest_path=manifest_path,
        stage_name="stage0b_docs_chunking",
        dry_run=False,
    )

    assert record["stage_type"] == "evaluation_grid"
    assert len(calls) == 3
    assert "ingest_html_documents.py" in calls[0][1]
    assert calls[0][-1] == "0"
    assert "evaluate_rag.py" in calls[1][1]
    assert "--preset" in calls[1]
    assert "--generation-mode" in calls[1]
    assert "--generation-workers" in calls[1]
    assert "--no-progress" in calls[1]
    assert calls[1][calls[1].index("--preset") + 1] == "docs_only"
    assert calls[2][calls[2].index("--output-dir") + 1].endswith("run_02")


def test_run_experiment_stage_prepare_phase_builds_prepare_only_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    base_config = tmp_path / "config.yaml"
    base_config.write_text("prompt_name: test\n", encoding="utf-8")

    manifest_path = _write_manifest(
        tmp_path,
        {
            "artifacts_root": "artifacts/experiments",
            "base_config": "config.yaml",
            "defaults": {
                "run_options": {
                    "generation_mode": "pipeline",
                    "generation_workers": 1,
                }
            },
            "stages": {
                "stage0a_generator_selection": {
                    "type": "evaluation_grid",
                    "dataset_path": "dev.jsonl",
                    "repeats": 1,
                    "conditions": [
                        {
                            "name": "dense_baseline",
                            "preset": "dense_only",
                            "run_options": {
                                "reuse_prepared": True,
                            },
                            "ingest": {
                                "kind": "html",
                                "homepage": "https://docs.example.org",
                            },
                        }
                    ],
                }
            },
        },
    )

    calls: list[list[str]] = []

    def _fake_run(command, check, cwd):
        calls.append(list(command))
        assert check is True
        assert Path(cwd) == repo_root
        return None

    monkeypatch.setattr(automation, "REPO_ROOT", repo_root)
    monkeypatch.setattr(automation.subprocess, "run", _fake_run)

    record = automation.run_experiment_stage(
        manifest_path=manifest_path,
        stage_name="stage0a_generator_selection",
        execution_phase="prepare",
        dry_run=False,
    )

    assert record["execution_phase"] == "prepare"
    assert len(calls) == 2
    assert "ingest_html_documents.py" in calls[0][1]
    assert "--prepare-only" in calls[1]
    assert "--reuse-prepared" not in calls[1]
    assert record["conditions"][0]["ingestion_skipped"] is False
    assert record["conditions"][0]["runs"][0]["execution_phase"] == "prepare"


def test_run_experiment_stage_evaluate_phase_reuses_prepared_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    base_config = tmp_path / "config.yaml"
    base_config.write_text("prompt_name: test\n", encoding="utf-8")

    manifest_path = _write_manifest(
        tmp_path,
        {
            "artifacts_root": "artifacts/experiments",
            "base_config": "config.yaml",
            "stages": {
                "stage4_source_ablation": {
                    "type": "evaluation_grid",
                    "dataset_path": "test.jsonl",
                    "repeats": 1,
                    "conditions": [
                        {
                            "name": "docs_only",
                            "preset": "docs_only",
                            "ingest": {
                                "kind": "html",
                                "homepage": "https://docs.example.org",
                            },
                        }
                    ],
                }
            },
        },
    )

    stage_root = tmp_path / "artifacts" / "experiments" / "stage4_source_ablation"
    prepared_path = stage_root / "docs_only" / "run_01" / "prepared_input.json"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.write_text("[]", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(command, check, cwd):
        calls.append(list(command))
        assert check is True
        assert Path(cwd) == repo_root
        return None

    monkeypatch.setattr(automation, "REPO_ROOT", repo_root)
    monkeypatch.setattr(automation.subprocess, "run", _fake_run)

    record = automation.run_experiment_stage(
        manifest_path=manifest_path,
        stage_name="stage4_source_ablation",
        execution_phase="evaluate",
        dry_run=False,
    )

    assert record["execution_phase"] == "evaluate"
    assert len(calls) == 1
    assert "evaluate_rag.py" in calls[0][1]
    assert "--reuse-prepared" in calls[0]
    assert "--prepare-only" not in calls[0]
    assert record["conditions"][0]["ingestion_skipped"] is True


def test_run_experiment_stage_evaluate_phase_requires_existing_prepared_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_config = tmp_path / "config.yaml"
    base_config.write_text("prompt_name: test\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "artifacts_root": "artifacts/experiments",
            "base_config": "config.yaml",
            "stages": {
                "stage4_source_ablation": {
                    "type": "evaluation_grid",
                    "dataset_path": "test.jsonl",
                    "conditions": [
                        {"name": "docs_only", "preset": "docs_only"},
                    ],
                }
            },
        },
    )

    monkeypatch.setattr(automation, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="Prepared rows are missing"):
        automation.run_experiment_stage(
            manifest_path=manifest_path,
            stage_name="stage4_source_ablation",
            execution_phase="evaluate",
            dry_run=False,
        )


def test_summarize_experiment_stage_writes_leaderboard_and_aggregates(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        {
            "artifacts_root": "artifacts/experiments",
            "base_config": "config.yaml",
            "stages": {
                "stage4_source_ablation": {
                    "type": "evaluation_grid",
                    "dataset_path": "test.jsonl",
                    "conditions": [
                        {"name": "docs_only", "preset": "docs_only"},
                        {"name": "naive_combined", "preset": "naive_combined"},
                    ],
                }
            },
        },
    )

    stage_root = tmp_path / "artifacts" / "experiments" / "stage4_source_ablation"
    docs_run = stage_root / "docs_only" / "run_01"
    naive_run = stage_root / "naive_combined" / "run_01"
    docs_run.mkdir(parents=True)
    naive_run.mkdir(parents=True)

    for run_dir, preset_name, factual_values in (
        (docs_run, "docs_only", [0.4, 0.6]),
        (naive_run, "naive_combined", [0.8, 0.9]),
    ):
        with (run_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["factual_correctness", "faithfulness", "context_recall"],
            )
            writer.writeheader()
            for factual_value in factual_values:
                writer.writerow(
                    {
                        "factual_correctness": factual_value,
                        "faithfulness": 0.5,
                        "context_recall": 0.25,
                    }
                )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "rows": 2,
                    "duration_seconds": 12.0 if preset_name == "docs_only" else 10.0,
                    "failure_rate": 0.0,
                    "selected_max_workers": 1,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "preset_name": preset_name,
                    "config_file": "/tmp/generated.yaml",
                    "condition_fingerprint": f"{preset_name}-fp",
                }
            ),
            encoding="utf-8",
        )

    artifacts = automation.summarize_experiment_stage(
        manifest_path=manifest_path,
        stage_name="stage4_source_ablation",
    )

    leaderboard_path = artifacts["leaderboard_csv"]
    aggregate_path = artifacts["condition_aggregates_csv"]
    assert leaderboard_path.exists()
    assert aggregate_path.exists()

    leaderboard_rows = list(csv.DictReader(leaderboard_path.open("r", encoding="utf-8")))
    assert leaderboard_rows[0]["condition_name"] == "naive_combined"
    assert leaderboard_rows[0]["factual_correctness"] == "0.850000"

    aggregate_rows = list(csv.DictReader(aggregate_path.open("r", encoding="utf-8")))
    by_condition = {row["condition_name"]: row for row in aggregate_rows}
    assert by_condition["docs_only"]["factual_correctness_mean"] == "0.500000"
    assert by_condition["naive_combined"]["factual_correctness_mean"] == "0.850000"
