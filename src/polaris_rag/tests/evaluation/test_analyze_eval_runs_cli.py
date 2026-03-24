from __future__ import annotations

import json
import sys

from polaris_rag.cli import analyze_eval_runs


def _write_analysis_row(path, *, response: str, source: str, source_authority: str) -> None:  # noqa: ANN001
    payload = {
        "id": "ex-1",
        "query": "How do I load GROMACS 2024.4?",
        "reference": "Use the official docs answer.",
        "response": response,
        "retrieved_context_ids": ["doc-1"],
        "retrieved_contexts": ["ctx-1"],
        "metrics": {"faithfulness": 0.9 if source == "docs" else 0.7},
        "benchmark_annotation": {"summary": "Versioned software lookup", "source_needed": "docs"},
        "query_constraints": {"software_names": ["GROMACS"], "software_versions": ["2024.4"]},
        "retrieval_sources": [source],
        "retrieval_source_types": [source_authority],
        "retrieval_features": [],
        "ranked_context_metadata": [
            {
                "rank": 1,
                "doc_id": "doc-1",
                "source": source,
                "source_authority": source_authority,
                "validity_status": "current" if source == "docs" else "unknown",
                "doc_title": "Official Docs" if source == "docs" else "Ticket Memory",
            }
        ],
        "retrieval_trace": [],
        "reranker_profile": {"type": "rrf"},
        "reranker_fingerprint": "fp",
        "response_status": "ok",
        "condition": {
            "preset_name": "docs_only" if source == "docs" else "tickets_only",
            "condition_fingerprint": "docs-fp" if source == "docs" else "tickets-fp",
            "condition_summary": {},
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_analyze_eval_runs_cli_writes_comparison_artifacts(tmp_path, monkeypatch, capsys) -> None:
    docs_dir = tmp_path / "docs_run"
    tickets_dir = tmp_path / "tickets_run"
    docs_dir.mkdir()
    tickets_dir.mkdir()
    _write_analysis_row(docs_dir / "analysis_rows.jsonl", response="module load GROMACS/2024.4", source="docs", source_authority="local_official")
    _write_analysis_row(tickets_dir / "analysis_rows.jsonl", response="try the ticket answer", source="tickets", source_authority="ticket_memory")

    output_dir = tmp_path / "comparison"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_eval_runs.py",
            "--run",
            f"docs_only={docs_dir}",
            "--run",
            f"tickets_only={tickets_dir}",
            "--output-dir",
            str(output_dir),
            "--manual-eval-seed",
            "7",
        ],
    )

    analyze_eval_runs.main()

    assert (output_dir / "condition_summary.csv").exists()
    assert (output_dir / "subgroup_metrics.csv").exists()
    assert (output_dir / "source_distribution.csv").exists()
    assert (output_dir / "query_review_sheet.csv").exists()
    assert (output_dir / "manual_eval_sheet.csv").exists()
    assert (output_dir / "manual_eval_key.csv").exists()
    assert (output_dir / "manual_eval_manifest.json").exists()
    assert (output_dir / "combined_analysis_rows.jsonl").exists()

    captured = capsys.readouterr()
    assert "Run analysis complete for 2 conditions." in captured.out
