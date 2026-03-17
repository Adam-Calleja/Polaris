import csv
import json
import sys

from polaris_rag.cli import benchmark_analysis


def _write_annotation_csv(path, rows) -> None:  # noqa: ANN001
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_benchmark_analysis_cli_writes_experiment_one_artifacts(tmp_path, monkeypatch, capsys) -> None:
    dataset_rows = [
        {
            "id": "ex-1",
            "summary": "Storage path question",
            "query": "Q1",
            "expected_answer": "A1",
            "metadata": {},
        },
        {
            "id": "ex-2",
            "summary": "Compile package on system",
            "query": "Q2",
            "expected_answer": "A2",
            "metadata": {},
        },
        {
            "id": "ex-3",
            "summary": "Attachment-heavy finance request",
            "query": "Q3",
            "expected_answer": "A3",
            "metadata": {},
        },
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(dataset_rows), encoding="utf-8")

    dev_path = tmp_path / "dataset.dev.jsonl"
    dev_path.write_text(json.dumps(dataset_rows[:2]), encoding="utf-8")
    test_path = tmp_path / "dataset.test.jsonl"
    test_path.write_text(json.dumps(dataset_rows[2:]), encoding="utf-8")

    annotation_rows = [
        {
            "id": "ex-1",
            "split": "dev",
            "summary": "Storage path question",
            "source_needed": "docs",
            "docs_scope_needed": "local_official",
            "validity_sensitive": "yes",
            "attachment_dependent": "no",
            "query_type": "local_operational",
            "version_sensitive": "no",
            "system_scope_required": "yes",
            "review_status": "verified",
            "notes": "",
        },
        {
            "id": "ex-2",
            "split": "dev",
            "summary": "Compile package on system",
            "source_needed": "both",
            "docs_scope_needed": "local_and_external",
            "validity_sensitive": "yes",
            "attachment_dependent": "no",
            "query_type": "software_version",
            "version_sensitive": "yes",
            "system_scope_required": "yes",
            "review_status": "verified",
            "notes": "",
        },
        {
            "id": "ex-3",
            "split": "test",
            "summary": "Attachment-heavy finance request",
            "source_needed": "tickets",
            "docs_scope_needed": "none",
            "validity_sensitive": "no",
            "attachment_dependent": "yes",
            "query_type": "general_how_to",
            "version_sensitive": "no",
            "system_scope_required": "no",
            "review_status": "verified",
            "notes": "",
        },
    ]
    annotations_path = tmp_path / "annotations.csv"
    _write_annotation_csv(annotations_path, annotation_rows)

    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_analysis.py",
            "--dataset-file",
            str(dataset_path),
            "--dev-dataset-file",
            str(dev_path),
            "--test-dataset-file",
            str(test_path),
            "--annotations-file",
            str(annotations_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    benchmark_analysis.main()

    assert (output_dir / "composition_counts.csv").exists()
    assert (output_dir / "composition_combinations.csv").exists()
    assert (output_dir / "composition_summary.json").exists()
    assert (output_dir / "composition_summary.md").exists()
    assert (output_dir / "composition_figure.png").exists()
    assert (output_dir / "composition_figure.svg").exists()

    summary = json.loads((output_dir / "composition_summary.json").read_text(encoding="utf-8"))
    assert summary["totals"] == {"all": 3, "dev": 2, "test": 1}
    assert summary["targeted_subsets"]["all"]["attachment_dependent"] == 1

    counts_csv = (output_dir / "composition_counts.csv").read_text(encoding="utf-8")
    assert "source_needed" in counts_csv
    assert "docs_scope_needed" in counts_csv

    captured = capsys.readouterr()
    assert "Benchmark analysis complete for 3 rows." in captured.out
