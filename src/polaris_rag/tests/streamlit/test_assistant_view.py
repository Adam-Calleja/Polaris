from __future__ import annotations

from polaris_rag.streamlit.views.assistant import parse_answer_sections


def test_parse_answer_sections_splits_known_headings() -> None:
    answer = """
CLASSIFICATION
Category: Storage

QUICK ASSESSMENT
This looks like a self-service task. [1]

ACTION
Use the storage portal. [1]

REFERENCE KEY
[1] : storage-portal-doc
""".strip()

    sections = parse_answer_sections(answer)

    assert [section.key for section in sections] == [
        "CLASSIFICATION",
        "QUICK ASSESSMENT",
        "ACTION",
        "REFERENCE KEY",
    ]
    assert sections[1].body == "This looks like a self-service task. [1]"


def test_parse_answer_sections_falls_back_to_single_response() -> None:
    sections = parse_answer_sections("Plain answer without headings.")

    assert len(sections) == 1
    assert sections[0].key == "RESPONSE"
    assert sections[0].body == "Plain answer without headings."
