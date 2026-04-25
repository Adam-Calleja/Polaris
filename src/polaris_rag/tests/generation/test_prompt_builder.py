from __future__ import annotations

from polaris_rag.generation.prompt_builder import PromptBuilder


def test_prompt_builder_build_messages_splits_examples_and_strips_assistant_suffix() -> None:
    builder = PromptBuilder()
    builder.register_from_dict(
        {
            "name": "chat_prompt",
            "system": "Stay grounded in retrieved evidence.",
            "few_shot": [
                {
                    "role": "example",
                    "content": (
                        "User Query:\nHow should this be handled?\n\n"
                        "Retrieved documents (numbered):\n[1] ID: ticket-1\nctx\n\n"
                        "Assistant:\nACTION\nUse the portal. [1]"
                    ),
                }
            ],
            "user": "User Query:\n{{ question }}\n\nAssistant:",
        }
    )

    messages = builder.build_messages(name="chat_prompt", question="What now?")

    assert messages == [
        {"role": "system", "content": "Stay grounded in retrieved evidence."},
        {
            "role": "user",
            "content": (
                "User Query:\nHow should this be handled?\n\n"
                "Retrieved documents (numbered):\n[1] ID: ticket-1\nctx"
            ),
        },
        {"role": "assistant", "content": "ACTION\nUse the portal. [1]"},
        {"role": "user", "content": "User Query:\nWhat now?"},
    ]
