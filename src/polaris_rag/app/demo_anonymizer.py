"""Demo-mode response anonymization helpers.

This module provides a lightweight post-generation anonymization pass for
frontend demo sessions. The anonymizer combines deterministic pattern matches
with an optional generator-LLM entity extraction pass, then applies a stable
alias map across the final answer and returned evidence snippets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from jinja2 import Template


logger = logging.getLogger("polaris_rag.demo_anonymizer")

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d().\-\s]{6,}\d)(?!\w)")
_TICKET_KEY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,15}-\d+\b")
_IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HOME_DIR_USERNAME_PATTERN = re.compile(
    r"(?i)(?:/home|/users|/user|/rds/user|/rds/project)/([A-Za-z][A-Za-z0-9._-]{1,31})(?=/)"
)
_LABELLED_USERNAME_PATTERN = re.compile(
    r"(?i)\b(?:crsid|username|login|account)\b\s*[:=]?\s*([A-Za-z](?:[A-Za-z0-9._-]{0,30}[A-Za-z0-9])?)\b"
)
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
_ALIAS_VISIBLE_PATTERN = re.compile(
    r"\b(?:TICKET|DOC|PERSON|EMAIL|PHONE|USER|PROJECT|ACCOUNT|ORG|LOCATION|IDENTIFIER|SENSITIVE_INFO)_\d+\b"
)

_PROMPT_FALLBACK = """You are a careful and precise assistant that extracts personal data from support text.
Return only a JSON object with these keys:
{
  "names": [],
  "email_addresses": [],
  "phone_numbers": [],
  "usernames": [],
  "project_codes": [],
  "account_numbers": [],
  "institutions": [],
  "locations": [],
  "other_identifiable_info": [],
  "special_category_info": []
}

Only extract explicit identifiers that appear verbatim in the text.

TEXT
{{ticket}}
"""

_ENTITY_KEYS: tuple[str, ...] = (
    "ticket_keys",
    "names",
    "email_addresses",
    "phone_numbers",
    "usernames",
    "project_codes",
    "account_numbers",
    "institutions",
    "locations",
    "other_identifiable_info",
    "special_category_info",
)

_ENTITY_ALIAS_PREFIX: dict[str, str] = {
    "ticket_keys": "TICKET",
    "names": "PERSON",
    "email_addresses": "EMAIL",
    "phone_numbers": "PHONE",
    "usernames": "USER",
    "project_codes": "PROJECT",
    "account_numbers": "ACCOUNT",
    "institutions": "ORG",
    "locations": "LOCATION",
    "other_identifiable_info": "IDENTIFIER",
    "special_category_info": "SENSITIVE_INFO",
}


@dataclass(frozen=True)
class AnonymizedQueryPayload:
    """Redacted response payload returned to the API layer."""

    answer: str
    context: list[dict[str, Any]]
    aliases: dict[str, str] = field(default_factory=dict)


def anonymize_query_payload(
    *,
    answer: str,
    context: Sequence[Mapping[str, Any] | Any],
    llm: Any | None = None,
    timeout_seconds: float | None = None,
) -> AnonymizedQueryPayload:
    """Apply demo-mode anonymization to an answer and evidence payload."""
    normalized_context = [_normalize_context_row(item, index=index) for index, item in enumerate(context, start=1)]
    combined_text = _build_detection_text(answer=answer, context=normalized_context)
    detected_entities = _merge_entities(
        _detect_entities_with_rules(combined_text),
        _detect_entities_with_llm(combined_text, llm=llm, timeout_seconds=timeout_seconds),
    )
    aliases = _build_alias_map(answer=answer, context=normalized_context, detected_entities=detected_entities)

    redacted_answer = _apply_aliases(str(answer or ""), aliases)
    redacted_context: list[dict[str, Any]] = []
    for row in normalized_context:
        doc_id = str(row.get("doc_id", "") or "")
        redacted_context.append(
            {
                "rank": row["rank"],
                "doc_id": aliases.get(doc_id, _apply_aliases(doc_id, aliases)),
                "text": _apply_aliases(str(row.get("text", "") or ""), aliases),
                "score": row.get("score"),
                "source": row.get("source"),
            }
        )

    return AnonymizedQueryPayload(
        answer=redacted_answer,
        context=redacted_context,
        aliases=aliases,
    )


def _normalize_context_row(item: Mapping[str, Any] | Any, *, index: int) -> dict[str, Any]:
    """Coerce a context row into a predictable mapping shape."""
    if isinstance(item, Mapping):
        payload = dict(item)
    else:
        payload = {
            "rank": getattr(item, "rank", index),
            "doc_id": getattr(item, "doc_id", ""),
            "text": getattr(item, "text", ""),
            "score": getattr(item, "score", None),
            "source": getattr(item, "source", None),
        }

    score_raw = payload.get("score")
    score = float(score_raw) if isinstance(score_raw, (int, float)) else None
    source_raw = payload.get("source")
    source = str(source_raw).strip() if isinstance(source_raw, str) and source_raw.strip() else None

    return {
        "rank": int(payload.get("rank") or index),
        "doc_id": str(payload.get("doc_id", "") or ""),
        "text": str(payload.get("text", "") or ""),
        "score": score,
        "source": source,
    }


def _build_detection_text(*, answer: str, context: Sequence[Mapping[str, Any]]) -> str:
    """Build a compact combined text block for identifier detection."""
    parts = ["ANSWER", str(answer or "").strip()]
    for row in context:
        parts.extend(
            [
                "",
                f"CONTEXT ITEM {row['rank']}",
                f"DOC_ID: {row.get('doc_id', '')}",
                f"SOURCE: {row.get('source') or ''}",
                str(row.get("text", "") or "").strip(),
            ]
        )
    return "\n".join(parts).strip()


def _detect_entities_with_rules(text: str) -> dict[str, list[str]]:
    """Detect straightforward identifiers without an LLM call."""
    entities = _empty_entities()
    entities["ticket_keys"] = _dedupe_preserve_order(_TICKET_KEY_PATTERN.findall(text))
    entities["email_addresses"] = _dedupe_preserve_order(_EMAIL_PATTERN.findall(text))
    entities["phone_numbers"] = _dedupe_preserve_order(match.group(0).strip() for match in _PHONE_PATTERN.finditer(text))
    entities["usernames"] = _dedupe_preserve_order(
        [
            *[match.group(1).strip() for match in _HOME_DIR_USERNAME_PATTERN.finditer(text)],
            *[match.group(1).strip() for match in _LABELLED_USERNAME_PATTERN.finditer(text)],
        ]
    )
    entities["other_identifiable_info"] = _dedupe_preserve_order(_IPV4_PATTERN.findall(text))
    return entities


def _detect_entities_with_llm(
    text: str,
    *,
    llm: Any | None,
    timeout_seconds: float | None,
) -> dict[str, list[str]]:
    """Ask the generator model to extract identifiers as structured JSON."""
    if llm is None:
        return _empty_entities()

    try:
        prompt = Template(_load_prompt_template()).render(ticket=text)
    except Exception:
        logger.warning("Falling back to built-in anonymization prompt template.")
        prompt = Template(_PROMPT_FALLBACK).render(ticket=text)

    try:
        raw_output = llm.generate(
            prompt,
            temperature=0,
            max_tokens=600,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("LLM anonymization pass failed; falling back to rule-based redaction: %s", type(exc).__name__)
        return _empty_entities()

    payload = _parse_json_object(str(raw_output or ""))
    return _normalize_llm_entities(payload)


def _merge_entities(*entity_groups: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Merge entity dictionaries while preserving discovery order."""
    merged = _empty_entities()
    for entity_group in entity_groups:
        for key in _ENTITY_KEYS:
            merged[key].extend(_normalize_entity_values(key, entity_group.get(key, [])))
            merged[key] = _dedupe_preserve_order(merged[key])
    return merged


def _build_alias_map(
    *,
    answer: str,
    context: Sequence[Mapping[str, Any]],
    detected_entities: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Create a stable placeholder map applied across the response payload."""
    aliases: dict[str, str] = {}
    counters: defaultdict[str, int] = defaultdict(int)
    combined_text = _build_detection_text(answer=answer, context=context)

    for row in context:
        doc_id = str(row.get("doc_id", "") or "").strip()
        if not doc_id or doc_id in aliases:
            continue
        prefix = "TICKET" if _is_ticket_doc_id(doc_id, source=row.get("source")) else "DOC"
        aliases[doc_id] = _next_alias(prefix, counters)

    for key in _ENTITY_KEYS:
        ordered_values = _sort_by_first_occurrence(
            _normalize_entity_values(key, detected_entities.get(key, [])),
            haystack=combined_text,
        )
        for raw_value in ordered_values:
            if not raw_value or raw_value in aliases or _ALIAS_VISIBLE_PATTERN.search(raw_value):
                continue
            aliases[raw_value] = _next_alias(_ENTITY_ALIAS_PREFIX[key], counters)

    return aliases


def _is_ticket_doc_id(doc_id: str, *, source: Any) -> bool:
    """Return True when a doc identifier should use a ticket-style alias."""
    if isinstance(source, str) and source.strip().lower() == "tickets":
        return True
    return bool(_TICKET_KEY_PATTERN.fullmatch(doc_id.strip()))


def _next_alias(prefix: str, counters: defaultdict[str, int]) -> str:
    """Return the next numbered placeholder for a category prefix."""
    counters[prefix] += 1
    return f"{prefix}_{counters[prefix]:03d}"


def _apply_aliases(text: str, aliases: Mapping[str, str]) -> str:
    """Apply aliases in longest-first order to minimize nested substitutions."""
    output = str(text or "")
    if not output or not aliases:
        return output

    for raw_value, alias in sorted(aliases.items(), key=lambda item: (-len(item[0]), item[0])):
        if not raw_value or raw_value == alias:
            continue
        output = _pattern_for_value(raw_value).sub(alias, output)
    return output


def _pattern_for_value(value: str) -> re.Pattern[str]:
    """Build a safe literal replacement pattern for a detected identifier."""
    escaped = re.escape(value)
    if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
    if re.fullmatch(r"[A-Za-z0-9._-]+", value):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
    return re.compile(escaped)


def _parse_json_object(raw_output: str) -> Mapping[str, Any]:
    """Extract and parse the first JSON object returned by the LLM."""
    cleaned = _CODE_FENCE_PATTERN.sub("", str(raw_output or "")).strip()
    match = _JSON_OBJECT_PATTERN.search(cleaned)
    if match is None:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _normalize_llm_entities(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    """Coerce the LLM detector payload to the internal entity structure."""
    normalized = _empty_entities()
    for key in _ENTITY_KEYS:
        normalized[key] = _normalize_entity_values(key, payload.get(key, []))
    return normalized


def _normalize_entity_values(key: str, values: Any) -> list[str]:
    """Normalize one detector field into a flat string list."""
    if key == "ticket_keys":
        if isinstance(values, str):
            return [values.strip()] if values.strip() else []
        if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
            return []
        return [str(value).strip() for value in values if str(value).strip()]

    if key == "names":
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            return []
        flattened: list[str] = []
        for item in values:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                for nested in item:
                    candidate = str(nested or "").strip()
                    if candidate:
                        flattened.append(candidate)
                continue
            candidate = str(item or "").strip()
            if candidate:
                flattened.append(candidate)
        return _dedupe_preserve_order(flattened)

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        candidate = str(values or "").strip()
        return [candidate] if candidate else []

    return _dedupe_preserve_order(str(value or "").strip() for value in values if str(value or "").strip())


def _sort_by_first_occurrence(values: Sequence[str], *, haystack: str) -> list[str]:
    """Sort values by first appearance, then by input order when absent."""
    positions = {value: haystack.find(value) for value in values}
    return sorted(
        values,
        key=lambda value: (
            positions[value] if positions[value] >= 0 else len(haystack) + list(values).index(value),
            list(values).index(value),
        ),
    )


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    """Return the first occurrence of each non-empty string."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        candidate = str(value or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        output.append(candidate)
    return output


def _empty_entities() -> dict[str, list[str]]:
    """Return the normalized empty detector shape."""
    return {key: [] for key in _ENTITY_KEYS}


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    """Load the checked-in anonymization prompt template when available."""
    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "anonymisation_prompt.jinja2"
    if not prompt_path.exists():
        return _PROMPT_FALLBACK
    return prompt_path.read_text(encoding="utf-8")


__all__ = ["AnonymizedQueryPayload", "anonymize_query_payload"]
