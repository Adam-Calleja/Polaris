from __future__ import annotations

from polaris_rag.streamlit.views.assistant import (
    assistant_view_state,
    latest_exchange,
    parse_answer_sections,
    quick_prompt_cards,
)


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


def test_latest_exchange_returns_latest_pair_and_older_messages() -> None:
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
    ]

    latest_user, latest_assistant, older_messages = latest_exchange(messages)

    assert latest_user == {"role": "user", "content": "new question"}
    assert latest_assistant == {"role": "assistant", "content": "new answer"}
    assert older_messages == [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]


def test_assistant_view_state_is_landing_without_assistant_message() -> None:
    assert assistant_view_state([]) == "landing"
    assert assistant_view_state([{"role": "user", "content": "hello"}]) == "landing"


def test_assistant_view_state_is_active_with_assistant_message() -> None:
    assert assistant_view_state(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    ) == "active"


def test_quick_prompt_cards_return_first_two_prompts() -> None:
    prompts = quick_prompt_cards()

    assert len(prompts) == 2
    assert prompts[0] == "We need to renew RDS and transfer ownership. How should this be handled?"
    assert prompts[1] == "Can you confirm the exact new path for my project data?"
