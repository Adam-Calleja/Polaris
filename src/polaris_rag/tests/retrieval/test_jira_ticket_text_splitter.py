from types import SimpleNamespace

import pytest

from polaris_rag.retrieval.text_splitter import get_chunks_from_jira_ticket


class DummyTokenizer:
    """
    Very simple tokenizer for tests:
    - encode: splits on whitespace, returns list of "tokens"
    - decode: joins token list with a single space
    """

    def encode(self, text: str):
        """
        Split text into tokens by whitespace.
        """
        text = text or ""
        tokens = text.strip().split()
        return tokens

    def decode(self, tokens):
        """
        Join list of tokens into a single string with spaces.
        """
        if isinstance(tokens, list):
            return " ".join(tokens)
        return str(tokens)


class DummyConfig:
    """Minimal stub to satisfy GlobalConfig.load usage in get_chunks_from_jira_ticket."""

    def __init__(self, model_name: str = "dummy-model"):
        embedder = SimpleNamespace(embedder=SimpleNamespace(model_name=model_name))
        self.embedder = embedder


class DummyAutoTokenizer:
    """Stub AutoTokenizer with a from_pretrained constructor returning DummyTokenizer."""

    @classmethod
    def from_pretrained(cls, model_name: str):
        return DummyTokenizer()


@pytest.fixture(autouse=True)
def patch_tokenizer_and_config(monkeypatch):
    """
    Patch GlobalConfig.load and AutoTokenizer.from_pretrained so tests do not
    depend on real config files or HF models.

    We patch by string path to guarantee we hit the exact symbols that
    JIRATicketChunker uses inside src.retrieval.text_splitter.
    """
    monkeypatch.setattr(
        "src.retrieval.text_splitter.GlobalConfig.load",
        lambda path: DummyConfig(),
        raising=True,
    )

    monkeypatch.setattr(
        "src.retrieval.text_splitter.AutoTokenizer",
        DummyAutoTokenizer,
        raising=True,
    )

    yield


def _make_ticket(summary: str, initial_desc: str, conversation_body: str) -> SimpleNamespace:
    """
    Helper to build a minimal ticket object compatible with get_chunks_from_jira_ticket.

    The ticket.text is a normalised Jira ticket string with header, initial description
    and a CONVERSATION section containing raw <MESSAGE> blocks.
    """
    text = (
        "<BEGIN_TICKET>\n"
        "[TICKET_KEY] HPCSSUP-1\n"
        "[STATUS] RESOLVED\n"
        "[CREATED] 2024-01-01T00:00:00.000+0000\n"
        f"[SUMMARY] {summary}\n\n"
        "[INITIAL_DESCRIPTION]\n"
        f"{initial_desc}\n\n"
        "[CONVERSATION]\n"
        f"{conversation_body}\n"
        "<END_TICKET>\n"
    )

    return SimpleNamespace(
        id="ticket-1",
        text=text,
        metadata={"summary": summary, "document_type": "jira_ticket"},
    )


def test_single_conversation_chunk_no_overlap():
    """
    A ticket with two short messages should produce:
    - 1 initial chunk
    - 1 conversation chunk
    The conversation chunk should not contain a [CONTEXT] section and should
    report no overlap in metadata. All chunks should be non-empty and
    conversation chunks should respect the chunk_size token budget.
    """
    conversation_body = (
        "<MESSAGE id=0001 role=USER>\n"
        "First message from the user.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>\n"
        "Reply from support.\n"
        "</MESSAGE>"
    )

    ticket = _make_ticket(
        summary="Internal HPC Application Form",
        initial_desc="User reports a problem with the internal HPC application form.",
        conversation_body=conversation_body,
    )

    chunk_size = 800
    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=chunk_size, overlap=5)

    assert len(chunks) == 2

    initial_chunk, conversation_chunk = chunks

    assert initial_chunk.metadata["chunk_type"] == "initial"
    assert initial_chunk.metadata["chunk_index"] == 0
    assert "[INITIAL_DESCRIPTION]" in initial_chunk.text
    assert "User reports a problem" in initial_chunk.text

    assert conversation_chunk.metadata["chunk_type"] == "conversation"
    assert conversation_chunk.metadata["chunk_index"] == 1
    assert conversation_chunk.metadata["overlap"]["has_overlap"] is False

    assert "[CONVERSATION]" in conversation_chunk.text
    assert "<MESSAGE id=0001 role=USER>" in conversation_chunk.text
    assert "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>" in conversation_chunk.text
    assert "[CONTEXT]" not in conversation_chunk.text

    tokenizer = DummyTokenizer()
    for chunk in chunks:
        num_tokens = len(tokenizer.encode(chunk.text))
        assert num_tokens > 0 
        assert num_tokens <= chunk_size


def test_multiple_conversation_chunks_with_overlap():
    """
    A ticket with four short messages should produce:
    - 1 initial chunk
    - 2 conversation chunks
    The second conversation chunk should contain a [CONTEXT] section derived
    from the last message in the previous chunk, and its metadata should
    indicate overlap. Conversation chunks should respect the chunk_size token
    budget.
    """
    msg3_text = "This is the third support message with a distinctive overlap tail"
    conversation_body = (
        "<MESSAGE id=0001 role=USER>\n"
        "First message from the user.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>\n"
        "First reply from support.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0003 role=HELPDESK_ASSIGNEE>\n"
        f"{msg3_text}\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0004 role=USER>\n"
        "Final short message from the user.\n"
        "</MESSAGE>"
    )

    ticket = _make_ticket(
        summary="Internal HPC Application Form",
        initial_desc="Initial description is here.",
        conversation_body=conversation_body,
    )

    chunk_size = 800
    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=chunk_size, overlap=5)

    assert len(chunks) == 3
    initial_chunk, conversation_chunk_1, conversation_chunk_2 = chunks

    assert initial_chunk.metadata["chunk_type"] == "initial"
    assert conversation_chunk_1.metadata["chunk_type"] == "conversation"
    assert conversation_chunk_2.metadata["chunk_type"] == "conversation"

    assert conversation_chunk_1.metadata["overlap"]["has_overlap"] is False
    assert "[CONTEXT]" not in conversation_chunk_1.text

    assert conversation_chunk_2.metadata["overlap"]["has_overlap"] is True
    assert "[CONTEXT]" in conversation_chunk_2.text

    expected_tail = " ".join(msg3_text.split()[-5:])
    assert expected_tail in conversation_chunk_2.text

    assert conversation_chunk_1.metadata["turn_range"] == "0001-0003"
    assert conversation_chunk_2.metadata["turn_range"] == "0004-0004"

    tokenizer = DummyTokenizer()
    for chunk in (conversation_chunk_1, conversation_chunk_2):
        num_tokens = len(tokenizer.encode(chunk.text))
        assert num_tokens > 0
        assert num_tokens <= chunk_size


def test_long_message_is_split_into_parts():
    """
    A single long message with multiple paragraphs that exceeds the per-chunk
    token budget should be split into multiple TicketMessagePart instances, and
    these parts should be distributed across multiple conversation chunks, each
    respecting the chunk_size limit.

    This verifies that:
    - the long message is split into part=1/2, part=2/2, etc. with original_id
    - no unsplit <MESSAGE id=0001 ...> block appears
    - each conversation chunk produced stays within the chunk_size token budget
      under the DummyTokenizer
    - the first part appears in the earlier conversation chunk, and the second
      part in the later one.
    """
    para1 = "Paragraph one has several words in it."
    para2 = "Paragraph two also has several words in it."
    long_message_text = f"{para1}\n\n{para2}"

    conversation_body = (
        "<MESSAGE id=0001 role=HELPDESK_ASSIGNEE>\n"
        f"{long_message_text}\n"
        "</MESSAGE>"
    )

    ticket = _make_ticket(
        summary="Internal HPC Application Form",
        initial_desc="Some initial description.",
        conversation_body=conversation_body,
    )

    chunk_size = 20
    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=chunk_size, overlap=0)

    assert len(chunks) >= 3

    conversation_chunks = [
        chunk for chunk in chunks if chunk.metadata["chunk_type"] == "conversation"
    ]
    assert len(conversation_chunks) == 2

    conversation_chunks.sort(key=lambda chunk: chunk.metadata["chunk_index"])
    conversation_chunk_1, conversation_chunk_2 = conversation_chunks

    convo_text_all = "\n".join(chunk.text for chunk in conversation_chunks)

    assert "part=1/2 original_id=0001" in convo_text_all
    assert "part=2/2 original_id=0001" in convo_text_all
    assert "<MESSAGE id=0001 role=HELPDESK_ASSIGNEE>" not in convo_text_all

    assert "part=1/2 original_id=0001" in conversation_chunk_1.text
    assert "part=1/2 original_id=0001" not in conversation_chunk_2.text
    assert "part=2/2 original_id=0001" not in conversation_chunk_1.text
    assert "part=2/2 original_id=0001" in conversation_chunk_2.text

    tokenizer = DummyTokenizer()
    for chunk in conversation_chunks:
        num_tokens = len(tokenizer.encode(chunk.text))
        assert num_tokens > 0
        assert num_tokens <= chunk_size


def test_very_long_message_can_span_three_conversation_chunks():
    """
    A very long single message that cannot fit into one or two chunks under the
    configured token budget should be split into at least three parts, each
    placed into separate conversation chunks.

    This verifies that:
    - the message is split into part=1/3, part=2/3, part=3/3 with original_id
    - no unsplit <MESSAGE id=0001 ...> block appears
    - at least three conversation chunks are produced
    - each part appears in at least one conversation chunk
    - all conversation chunks respect the chunk_size token budget under
      DummyTokenizer.
    """
    para1 = (
        "Paragraph one has many words in it and is intended to take up a "
        "significant portion of the available token budget for the first chunk."
    )
    para2 = (
        "Paragraph two continues with additional explanatory text, further "
        "increasing the total token count so that the message can no longer "
        "fit into just one or two chunks without exceeding the budget."
    )
    para3 = (
        "Paragraph three adds yet more detail and examples, ensuring that the "
        "overall message length is large enough to force a three-way split "
        "across multiple conversation chunks."
    )
    long_message_text = f"{para1}\n\n{para2}\n\n{para3}"

    conversation_body = (
        "<MESSAGE id=0001 role=HELPDESK_ASSIGNEE>\n"
        f"{long_message_text}\n"
        "</MESSAGE>"
    )

    ticket = _make_ticket(
        summary="Very long single message requiring three chunks.",
        initial_desc="Some initial description.",
        conversation_body=conversation_body,
    )

    chunk_size = 60
    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=chunk_size, overlap=0)

    conversation_chunks = [
        chunk for chunk in chunks if chunk.metadata["chunk_type"] == "conversation"
    ]
    assert len(conversation_chunks) >= 3

    convo_text_all = "\n".join(chunk.text for chunk in conversation_chunks)

    assert "part=1/3 original_id=0001" in convo_text_all
    assert "part=2/3 original_id=0001" in convo_text_all
    assert "part=3/3 original_id=0001" in convo_text_all
    assert "<MESSAGE id=0001 role=HELPDESK_ASSIGNEE>" not in convo_text_all

    part_labels = [
        "part=1/3 original_id=0001",
        "part=2/3 original_id=0001",
        "part=3/3 original_id=0001",
    ]
    for label in part_labels:
        assert any(label in chunk.text for chunk in conversation_chunks)

    tokenizer = DummyTokenizer()
    for chunk in conversation_chunks:
        num_tokens = len(tokenizer.encode(chunk.text))
        assert num_tokens > 0
        assert num_tokens <= chunk_size


def test_prev_and_next_links_are_wired_correctly():
    """
    Chunks returned by get_chunks_from_jira_ticket should have prev_id/next_id
    wired to form a simple forward chain, and chunk_index should be consistent
    with list order.
    """
    conversation_body = (
        "<MESSAGE id=0001 role=USER>\n"
        "First message.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>\n"
        "Second message.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0003 role=USER>\n"
        "Third message.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0004 role=HELPDESK_ASSIGNEE>\n"
        "Fourth message.\n"
        "</MESSAGE>"
    )

    ticket = _make_ticket(
        summary="Internal HPC Application Form",
        initial_desc="Description.",
        conversation_body=conversation_body,
    )

    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=20, overlap=3)

    assert len(chunks) >= 2

    assert chunks[0].prev_id is None
    for i in range(1, len(chunks)):
        assert chunks[i].prev_id == chunks[i - 1].id

    for i in range(len(chunks) - 1):
        assert chunks[i].next_id == chunks[i + 1].id

    assert chunks[-1].next_id is None

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_ticket_with_no_conversation_messages_produces_only_initial_chunk():
    """
    A ticket with no <MESSAGE> blocks in the conversation section should produce
    only a single 'initial' chunk and no conversation chunks.
    """
    ticket = _make_ticket(
        summary="Internal HPC Application Form",
        initial_desc="User reports a problem.",
        conversation_body="",
    )

    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=50, overlap=5)

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.metadata["chunk_type"] == "initial"
    assert "[INITIAL_DESCRIPTION]" in chunk.text
   
    assert all("conversation" != chunk.metadata["chunk_type"] for chunk in chunks)


def test_empty_conversation_message_does_not_create_extra_chunks():
    """
    A ticket containing an empty <MESSAGE> followed by a non-empty one should
    still behave like a simple short conversation:
    - 1 initial chunk
    - 1 conversation chunk
    The empty message must not cause extra chunks or failures.
    """
    conversation_body = (
        "<MESSAGE id=0001 role=USER>\n"
        "\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>\n"
        "Non-empty reply from support.\n"
        "</MESSAGE>"
    )

    ticket = _make_ticket(
        summary="Internal HPC Application Form",
        initial_desc="Some description.",
        conversation_body=conversation_body,
    )

    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=50, overlap=5)

    assert len(chunks) == 2

    initial_chunk, conversation_chunk = chunks
    assert initial_chunk.metadata["chunk_type"] == "initial"
    assert conversation_chunk.metadata["chunk_type"] == "conversation"

    assert "Non-empty reply from support." in conversation_chunk.text

    tokenizer = DummyTokenizer()
    num_tokens = len(tokenizer.encode(conversation_chunk.text))
    assert num_tokens > 0
    assert num_tokens <= 50


def test_two_short_messages_then_split_long_third_message_across_two_chunks():
    """
    A conversation where two initial messages are short and the third message is long
    enough to exceed the per-chunk token budget should behave as follows:

    - The first conversation chunk contains:
      * both short messages (ids 0001 and 0002) in full, and
      * the first part of the long third message (part=1/2 original_id=0003).
    - The second conversation chunk contains:
      * the remaining part of the long third message (part=2/2 original_id=0003).
    - No unsplit <MESSAGE id=0003 ...> block appears in the final chunks.
    - Both conversation chunks respect the chunk_size token budget.
    """
    short_msg_1 = "Short message one."
    short_msg_2 = "Short message two."

    para1 = "Paragraph one has several words in it."
    para2 = "Paragraph two also has several words in it."
    long_message_text = f"{para1}\n\n{para2}"

    conversation_body = (
        "<MESSAGE id=0001 role=USER>\n"
        f"{short_msg_1}\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>\n"
        f"{short_msg_2}\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0003 role=HELPDESK_ASSIGNEE>\n"
        f"{long_message_text}\n"
        "</MESSAGE>"
    )

    ticket = _make_ticket(
        summary="Two short then one long message.",
        initial_desc="Initial description.",
        conversation_body=conversation_body,
    )

    chunk_size = 22
    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=chunk_size, overlap=0)

    conversation_chunks = [
        chunk for chunk in chunks if chunk.metadata["chunk_type"] == "conversation"
    ]
    assert len(conversation_chunks) >= 2

    conversation_chunks.sort(key=lambda chunk: chunk.metadata["chunk_index"])
    convo_text_all = "\n".join(chunk.text for chunk in conversation_chunks)

    assert "<MESSAGE id=0001 role=USER>" in convo_text_all
    assert short_msg_1 in convo_text_all
    assert "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>" in convo_text_all
    assert short_msg_2 in convo_text_all

    assert "part=1/2 original_id=0003" in convo_text_all
    assert "part=2/2 original_id=0003" in convo_text_all
    assert "<MESSAGE id=0003 role=HELPDESK_ASSIGNEE>" not in convo_text_all

    first_part_index = next(
        i for i, c in enumerate(conversation_chunks)
        if "part=1/2 original_id=0003" in c.text
    )
    second_part_index = next(
        i for i, c in enumerate(conversation_chunks)
        if "part=2/2 original_id=0003" in c.text
    )
    assert first_part_index < second_part_index

    tokenizer = DummyTokenizer()
    for chunk in conversation_chunks:
        num_tokens = len(tokenizer.encode(chunk.text))
        assert num_tokens > 0
        assert num_tokens <= chunk_size


def test_overlap_can_create_additional_chunk_when_exceeding_budget():
    """
    When overlap is enabled and a second conversation chunk would normally fit within
    the chunk_size limit without any context, but prepending the full overlap tail
    would cause that chunk to overflow, the splitter should:

    - Preserve the full overlap for the next chunk, even if that means creating an
      additional conversation chunk.
    - Potentially create a third conversation chunk as a result of overlap pushing
      the second chunk over the budget.
    - Ensure that all conversation chunks respect the chunk_size token budget.
    - Ensure that the second conversation chunk includes a [CONTEXT] section with
      tokens from the tail of the previous chunk's last message.
    """
    msg1 = "Short message one."
    msg2 = "Short message two."
    msg3 = "Another short support reply."
    tail_token = "TAILTOKEN"
    msg4 = (
        "This is a longer message that will nearly fill the token budget "
        f"before adding any overlap tail {tail_token}."
    )

    conversation_body = (
        "<MESSAGE id=0001 role=USER>\n"
        f"{msg1}\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>\n"
        f"{msg2}\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0003 role=HELPDESK_ASSIGNEE>\n"
        f"{msg3}\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0004 role=USER>\n"
        f"{msg4}\n"
        "</MESSAGE>"
    )

    ticket = _make_ticket(
        summary="Overlap creates extra chunk scenario.",
        initial_desc="Initial description.",
        conversation_body=conversation_body,
    )

    chunk_size = 40
    overlap = 5
    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=chunk_size, overlap=overlap)

    conversation_chunks = [
        chunk for chunk in chunks if chunk.metadata["chunk_type"] == "conversation"
    ]
    assert len(conversation_chunks) >= 2

    conversation_chunks.sort(key=lambda chunk: chunk.metadata["chunk_index"])
    conversation_1, conversation_2 = conversation_chunks[0], conversation_chunks[1]

    assert conversation_2.metadata["overlap"]["has_overlap"] is True
    assert "[CONTEXT]" in conversation_2.text

    assert tail_token in conversation_2.text

    tokenizer = DummyTokenizer()
    for chunk in conversation_chunks:
        num_tokens = len(tokenizer.encode(chunk.text))
        assert num_tokens > 0
        assert num_tokens <= chunk_size


def test_conversation_chunks_respect_max_message_count():
    """
    Conversation chunks should not contain more than a small, fixed number of
    messages (e.g., 3). This test assumes a max logical message count of 3 per
    conversation chunk and verifies that, for a ticket with 5 short messages:

    - We get 1 initial chunk and 2 conversation chunks.
    - The first conversation chunk contains messages 0001–0003.
    - The second conversation chunk contains messages 0004–0005.
    - No conversation chunk contains more than 3 <MESSAGE> blocks.
    Token budget is deliberately large so that the split is driven by message
    count rather than token size.
    """
    conversation_body = (
        "<MESSAGE id=0001 role=USER>\n"
        "Short message 1.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0002 role=HELPDESK_ASSIGNEE>\n"
        "Short message 2.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0003 role=USER>\n"
        "Short message 3.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0004 role=HELPDESK_ASSIGNEE>\n"
        "Short message 4.\n"
        "</MESSAGE>\n\n"
        "<MESSAGE id=0005 role=USER>\n"
        "Short message 5.\n"
        "</MESSAGE>"
    )

    ticket = _make_ticket(
        summary="Max message count per chunk scenario.",
        initial_desc="Initial description.",
        conversation_body=conversation_body,
    )

    chunk_size = 800
    chunks = get_chunks_from_jira_ticket(ticket, chunk_size=chunk_size, overlap=0)

    conversation_chunks = [
        chunk for chunk in chunks if chunk.metadata["chunk_type"] == "conversation"
    ]
    assert len(conversation_chunks) == 2

    conversation_chunks.sort(key=lambda chunk: chunk.metadata["chunk_index"])
    conversation_1, conversation_2 = conversation_chunks

    def count_messages(text: str) -> int:
        return text.count("<MESSAGE id=")

    assert count_messages(conversation_1.text) <= 3
    assert count_messages(conversation_2.text) <= 3

    assert count_messages(conversation_1.text) == 3
    assert count_messages(conversation_2.text) == 2

    if "turn_range" in conversation_1.metadata and "turn_range" in conversation_2.metadata:
        assert conversation_1.metadata["turn_range"] == "0001-0003"
        assert conversation_2.metadata["turn_range"] == "0004-0005"
