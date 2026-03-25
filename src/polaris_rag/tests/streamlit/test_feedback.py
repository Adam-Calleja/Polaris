from __future__ import annotations

from polaris_rag.streamlit.feedback import FeedbackRecord, append_feedback_record, compute_response_fingerprint, feedback_summary


def test_feedback_logger_appends_jsonl_and_summarizes(tmp_path) -> None:
    log_path = tmp_path / "ui_feedback" / "feedback.jsonl"

    append_feedback_record(
        log_path,
        FeedbackRecord(
            created_at="2026-03-25T10:00:00+00:00",
            response_fingerprint=compute_response_fingerprint("q1", "a1", context_doc_ids=["doc-1"], scenario_id="strong_answer"),
            query="q1",
            scenario_id="strong_answer",
            answer_status_code="grounded",
            evidence_count=2,
            helpful="yes",
            grounded="yes",
            citation_quality="strong",
            failure_type="none",
            notes="clear answer",
        ),
    )
    append_feedback_record(
        log_path,
        FeedbackRecord(
            created_at="2026-03-25T10:05:00+00:00",
            response_fingerprint=compute_response_fingerprint("q2", "a2", context_doc_ids=[], scenario_id="timeout_case"),
            query="q2",
            scenario_id="timeout_case",
            answer_status_code="no_evidence",
            evidence_count=0,
            helpful="no",
            grounded="no",
            citation_quality="weak",
            failure_type="timeout",
            notes="deadline hit",
        ),
    )

    summary = feedback_summary(log_path)

    assert summary["total"] == 2
    assert summary["helpful_yes"] == 1
    assert summary["grounded_yes"] == 1
    assert summary["by_scenario"] == [
        {"scenario_id": "strong_answer", "count": 1},
        {"scenario_id": "timeout_case", "count": 1},
    ]
    assert summary["failure_types"] == [
        {"failure_type": "none", "count": 1},
        {"failure_type": "timeout", "count": 1},
    ]
