"""File-backed feedback logging helpers for Polaris UI integrations.

This module persists UI feedback records to disk and computes aggregate summaries
consumed by the API and frontend.

Classes
-------
FeedbackRecord
    Structured record for feedback.

Functions
---------
append_feedback_record
    Append feedback Record.
load_feedback_records
    Load feedback Records.
feedback_summary
    Feedback Summary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeedbackRecord:
    """Structured record for feedback.
    
    Attributes
    ----------
    created_at : str
        Value for created At.
    response_fingerprint : str
        Value for response Fingerprint.
    query : str
        User query text.
    scenario_id : str or None
        Stable identifier for scenario.
    answer_status_code : str
        Value for answer Status Code.
    evidence_count : int
        Value for evidence Count.
    helpful : str
        Value for helpful.
    grounded : str
        Value for grounded.
    citation_quality : str
        Value for citation Quality.
    failure_type : str
        Value for failure Type.
    notes : str
        Value for notes.
    """
    created_at: str
    response_fingerprint: str
    query: str
    scenario_id: str | None
    answer_status_code: str
    evidence_count: int
    helpful: str
    grounded: str
    citation_quality: str
    failure_type: str
    notes: str = ""


def append_feedback_record(log_path: str | Path, record: FeedbackRecord) -> None:
    """Append feedback Record.
    
    Parameters
    ----------
    log_path : str or Path
        Filesystem path used by the operation.
    record : FeedbackRecord
        Feedback or artifact record to persist.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def load_feedback_records(log_path: str | Path) -> list[dict[str, Any]]:
    """Load feedback Records.
    
    Parameters
    ----------
    log_path : str or Path
        Filesystem path used by the operation.
    
    Returns
    -------
    list[dict[str, Any]]
        Loaded feedback Records.
    """
    path = Path(log_path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def feedback_summary(log_path: str | Path) -> dict[str, Any]:
    """Feedback Summary.
    
    Parameters
    ----------
    log_path : str or Path
        Filesystem path used by the operation.
    
    Returns
    -------
    dict[str, Any]
        Structured result of the operation.
    """
    records = load_feedback_records(log_path)
    if not records:
        return {
            "total": 0,
            "helpful_yes": 0,
            "grounded_yes": 0,
            "by_scenario": [],
            "failure_types": [],
        }

    helpful_yes = sum(1 for record in records if str(record.get("helpful")) == "yes")
    grounded_yes = sum(1 for record in records if str(record.get("grounded")) == "yes")

    scenario_counter: Counter[str] = Counter()
    failure_counter: Counter[str] = Counter()
    for record in records:
        scenario = str(record.get("scenario_id") or "ad_hoc")
        scenario_counter[scenario] += 1
        failure_type = str(record.get("failure_type") or "none")
        failure_counter[failure_type] += 1

    return {
        "total": len(records),
        "helpful_yes": helpful_yes,
        "grounded_yes": grounded_yes,
        "by_scenario": [
            {"scenario_id": scenario_id, "count": count}
            for scenario_id, count in sorted(scenario_counter.items())
        ],
        "failure_types": [
            {"failure_type": failure_type, "count": count}
            for failure_type, count in sorted(failure_counter.items())
        ],
    }
