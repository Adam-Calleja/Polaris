from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from polaris_rag.evaluation.run_analysis import (
    RunInput,
    build_analysis_rows,
    load_analysis_rows,
    load_run_input,
    write_run_comparison_outputs,
)


def _prepared_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "ex-1",
            "user_input": "How do I load GROMACS 2024.4?",
            "reference": "Use the local docs answer.",
            "response": "module load GROMACS/2024.4",
            "retrieved_context_ids": ["doc-1"],
            "retrieved_contexts": ["ctx-1"],
            "metadata": {
                "query_constraints": {"software_names": ["GROMACS"], "software_versions": ["2024.4"]},
                "retrieval_sources": ["docs"],
                "retrieval_source_types": ["local_official"],
                "retrieval_features": [
                    {
                        "rank": 1,
                        "doc_id": "doc-1",
                        "source": "docs",
                        "source_authority": "local_official",
                        "authority_feature": 0.4,
                        "scope_feature": 0.1,
                        "version_feature": 0.3,
                        "status_feature": 0.2,
                        "freshness_feature": 0.05,
                        "final_score": 1.1,
                    }
                ],
                "ranked_context_metadata": [
                    {
                        "rank": 1,
                        "doc_id": "doc-1",
                        "source": "docs",
                        "source_authority": "local_official",
                        "validity_status": "current",
                        "doc_title": "Official Docs",
                    }
                ],
                "retrieval_trace": [
                    {
                        "rank": 1,
                        "doc_id": "doc-1",
                        "source": "docs",
                        "source_authority": "local_official",
                        "rerank_trace": {"final_score": 1.1},
                    }
                ],
                "benchmark_annotation": {
                    "summary": "Versioned software lookup",
                    "source_needed": "docs",
                    "docs_scope_needed": "local_official",
                },
                "response_status": "ok",
            },
        }
    ]


def test_build_analysis_rows_and_load_round_trip(tmp_path: Path) -> None:
    rows = build_analysis_rows(
        source_rows=_prepared_rows(),
        scores_df=[{"faithfulness": 0.9, "factual_correctness": 0.8}],
        condition_fields={
            "preset_name": "validity_aware",
            "preset_description": "Frozen validity-aware reranker over docs+tickets.",
            "condition_fingerprint": "abc123",
            "condition_summary": {"sources": [{"name": "docs"}]},
        },
    )

    path = tmp_path / "analysis_rows.jsonl"
    from polaris_rag.evaluation.run_analysis import persist_analysis_rows

    persist_analysis_rows(rows, path)
    loaded = load_analysis_rows(path)

    assert loaded == rows
    assert loaded[0]["condition"]["preset_name"] == "validity_aware"
    assert loaded[0]["retrieval_sources"] == ["docs"]


def test_write_outputs_writes_analysis_rows_and_manifest_fields(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("ragas")
    from ragas import RunConfig

    from polaris_rag.evaluation.evaluator import EvaluationRunResult, write_outputs

    result = EvaluationRunResult(
        scores_df=pd.DataFrame([{"faithfulness": 0.9, "factual_correctness": 0.8}]),
        selected_metrics=["faithfulness", "factual_correctness"],
        skipped_metrics=[],
        selected_max_workers=4,
        run_config=RunConfig(timeout=30, max_retries=1, max_wait=60, max_workers=4, log_tenacity=False, seed=7),
        batch_size=1,
        tuning_trials=[],
        duration_seconds=1.2,
        failure_rate=0.0,
    )

    artifacts = write_outputs(
        result=result,
        output_dir=tmp_path,
        extra_manifest={
            "preset_name": "validity_aware",
            "preset_description": "Frozen validity-aware reranker over docs+tickets.",
            "condition_fingerprint": "abc123",
            "condition_summary": {"sources": [{"name": "docs"}]},
            "dataset": {"run_validity": "VALID"},
            "tune_concurrency": True,
            "trace_evaluator_llm": False,
            "mlflow_parent_run_id": "run-parent",
            "config_file": "/tmp/config.yaml",
        },
        source_rows=_prepared_rows(),
    )

    assert (tmp_path / "scores.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "analysis_rows.jsonl").exists()
    assert artifacts["analysis_rows_jsonl"] == tmp_path / "analysis_rows.jsonl"

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["preset_name"] == "validity_aware"
    assert manifest["condition_fingerprint"] == "abc123"
    assert manifest["artifacts"]["analysis_rows_jsonl"].endswith("analysis_rows.jsonl")


def test_load_run_input_falls_back_to_prepared_rows_scores_and_manifest(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared_rows.json"
    prepared_path.write_text(json.dumps(_prepared_rows()), encoding="utf-8")

    with (tmp_path / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["faithfulness", "factual_correctness"])
        writer.writeheader()
        writer.writerow({"faithfulness": "0.9", "factual_correctness": "0.8"})

    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "preset_name": "validity_aware",
                "preset_description": "Frozen validity-aware reranker over docs+tickets.",
                "condition_fingerprint": "abc123",
                "condition_summary": {"sources": [{"name": "docs"}]},
            }
        ),
        encoding="utf-8",
    )

    run = load_run_input("validity_aware", tmp_path)

    assert run.condition_name == "validity_aware"
    assert run.analysis_rows[0]["metrics"]["faithfulness"] == 0.9
    assert run.analysis_rows[0]["condition"]["condition_fingerprint"] == "abc123"


def test_write_run_comparison_outputs_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    rows_a = build_analysis_rows(
        source_rows=_prepared_rows(),
        scores_df=[{"faithfulness": 0.9}],
        condition_fields={"preset_name": "docs_only", "condition_fingerprint": "docs-fp", "condition_summary": {}},
    )
    rows_b = build_analysis_rows(
        source_rows=[
            {
                **_prepared_rows()[0],
                "response": "ticket answer",
                "metadata": {
                    **_prepared_rows()[0]["metadata"],
                    "retrieval_sources": ["tickets"],
                    "retrieval_source_types": ["ticket_memory"],
                    "ranked_context_metadata": [
                        {
                            "rank": 1,
                            "doc_id": "ticket-1",
                            "source": "tickets",
                            "source_authority": "ticket_memory",
                            "validity_status": "unknown",
                            "doc_title": "Ticket Memory",
                        }
                    ],
                },
            }
        ],
        scores_df=[{"faithfulness": 0.7}],
        condition_fields={"preset_name": "tickets_only", "condition_fingerprint": "tickets-fp", "condition_summary": {}},
    )
    runs = [
        RunInput("docs_only", tmp_path / "docs", rows_a),
        RunInput("tickets_only", tmp_path / "tickets", rows_b),
    ]

    output_a = tmp_path / "comparison_a"
    output_b = tmp_path / "comparison_b"
    write_run_comparison_outputs(runs=runs, output_dir=output_a, manual_eval_seed=11)
    write_run_comparison_outputs(runs=runs, output_dir=output_b, manual_eval_seed=11)

    assert (output_a / "condition_summary.csv").read_text(encoding="utf-8") == (
        output_b / "condition_summary.csv"
    ).read_text(encoding="utf-8")
    assert (output_a / "manual_eval_sheet.csv").read_text(encoding="utf-8") == (
        output_b / "manual_eval_sheet.csv"
    ).read_text(encoding="utf-8")
