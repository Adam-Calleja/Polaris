from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FeedbackRecord:
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


def compute_response_fingerprint(
    query: str,
    answer: str,
    *,
    context_doc_ids: list[str] | None = None,
    scenario_id: str | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(query).strip().encode("utf-8"))
    digest.update(b"\n")
    digest.update(str(answer).strip().encode("utf-8"))
    digest.update(b"\n")
    digest.update(",".join(context_doc_ids or []).encode("utf-8"))
    digest.update(b"\n")
    digest.update(str(scenario_id or "").encode("utf-8"))
    return digest.hexdigest()


def append_feedback_record(log_path: str | Path, record: FeedbackRecord) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def load_feedback_records(log_path: str | Path) -> list[dict[str, Any]]:
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


def coerce_feedback_record(value: Mapping[str, Any]) -> FeedbackRecord:
    return FeedbackRecord(
        created_at=str(value.get("created_at") or ""),
        response_fingerprint=str(value.get("response_fingerprint") or ""),
        query=str(value.get("query") or ""),
        scenario_id=(str(value.get("scenario_id")) if value.get("scenario_id") is not None else None),
        answer_status_code=str(value.get("answer_status_code") or "no_evidence"),
        evidence_count=int(value.get("evidence_count") or 0),
        helpful=str(value.get("helpful") or "unknown"),
        grounded=str(value.get("grounded") or "unknown"),
        citation_quality=str(value.get("citation_quality") or "unknown"),
        failure_type=str(value.get("failure_type") or "none"),
        notes=str(value.get("notes") or ""),
    )
