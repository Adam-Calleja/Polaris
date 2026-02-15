import pytest
import copy
from src.retrieval.document_preprocessor import (
    adf_to_text,
    build_jira_ticket_text,
    preprocess_jira_ticket,
    preprocess_jira_tickets,
)

import src.retrieval.document_preprocessor as document_preprocessor

def _make_base_ticket(
    key: str = "TEST-1",
    summary: str = "Test summary",
    description_text: str = "Ticket description",
    status_name: str = "Open",
    created: str = "2025-01-01T12:00:00.000+0000",
) -> dict:
    """
    Create a minimal Jira ticket payload suitable for build_jira_ticket_text tests.
    """
    description_adf = {
        "type": "paragraph",
        "content": [{"type": "text", "text": description_text}],
    }

    return {
        "key": key,
        "fields": {
            "summary": summary,
            "description": description_adf,
            "status": {"name": status_name},
            "created": created,
            "comment": {"comments": []},
            "assignee": {"emailAddress": "assignee@example.com"},
            "creator": {"emailAddress": "creator@example.com"},
            "customfield_10042": [{"emailAddress": "pi@example.com"}],
            "reporter": {"emailAddress": "reporter@example.com"},
        },
    }

def _make_preprocess_ticket(
    key: str = "TEST-1",
    summary: str = "Test summary",
    status_name: str = "Open",
    created: str = "2025-01-01T09:00:00.000+0000",
    updated: str = "2025-01-01T10:00:00.000+0000",
    resolution_date: str | None = "2025-01-01T11:00:00.000+0000",
) -> dict:
    """
    Create a Jira ticket shaped for preprocess_jira_ticket tests.

    Starts from _make_base_ticket and adds/adjusts the fields that
    preprocess_jira_ticket expects, including fields.key, updated, and
    resolutionDate.
    """
    ticket = _make_base_ticket(
        key=key,
        summary=summary,
        status_name=status_name,
        created=created,
    )

    ticket["fields"]["key"] = ticket["key"]

    ticket["fields"]["updated"] = updated
    ticket["fields"]["resolutionDate"] = resolution_date

    return ticket

def test_adf_to_text_none_returns_empty():
    """
    Test that ``adf_to_text`` returns an empty string when given ``None`` input.
    """
    assert adf_to_text(None) == ""

def test_adf_to_text_empty_list_returns_empty():
    """
    Test that ``adf_to_text`` returns an empty string when given an empty list of nodes.
    """
    assert adf_to_text([]) == ""

def test_adf_to_text_blockquote_single_line():
    """
    Test that a single-line blockquote node is rendered with a ``>`` prefix and trailing newline.
    """
    node = {
            "type": "blockquote",
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Hello world"
                    }
                ]
                }
            ]
        }
    
    assert adf_to_text([node]) == "> Hello world\n"

def test_adf_to_text_blockquote_multi_line():
    """
    Test that a multi-paragraph blockquote is rendered with ``>`` prefixes on every line.
    """
    node = {
            "type": "blockquote",
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Hello world"
                    }
                ]
                },
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Goodbye world"
                    }
                ]
                }
            ]
        }
    
    assert adf_to_text([node]) == "> Hello world\n> Goodbye world\n"

def test_adf_to_text_bullet_list():
    """
    Test that a bullet list made of paragraph children is rendered as ``- `` prefixed lines.
    """
    node = {
            "type": "bulletList",
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Hello world"
                    }
                ]
                },
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "- Goodbye world"
                    }
                ]
                }
            ]
        }
    
    assert adf_to_text([node]) == "- Hello world\n- - Goodbye world\n"

def test_adf_to_text_bullet_list_with_list_items():
    """
    Test that a bullet list composed of ``listItem`` children is rendered as ``- `` prefixed lines.
    """
    node = {
        "type": "bulletList",
        "content": [
            {
                "type": "listItem",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Hello world"}],
                    }
                ],
            },
            {
                "type": "listItem",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Goodbye world"}],
                    }
                ],
            },
        ],
    }

    assert adf_to_text([node]) == "- Hello world\n- Goodbye world\n"

def test_adf_to_text_codeblock():
    """
    Test that a ``codeBlock`` node is rendered as a fenced code block including the language. 
    """
    node = {
            "type": "codeBlock",
            "attrs": {
                "language": "javascript"
            },
            "content": [
                {
                "type": "text",
                "text": "var foo = {};\nvar bar = [];"
                }
            ]
        }
    
    assert adf_to_text([node]) == "```javascript\nvar foo = {};\nvar bar = [];\n```\n"

def test_adf_to_text_expand():
    """
    Test that an ``expand`` node is rendered as its title followed by the expanded body text.
    """
    node = {
            "type": "expand",
            "attrs": {
                "title": "Hello world"
            },
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Hello world"
                    }
                ]
                }
            ]
        }
    
    assert adf_to_text([node]) == "Hello world\n\nHello world\n"

def test_adf_to_text_heading():
    """
    Test that a heading node is rendered with the correct number of leading ``#`` characters.
    """
    node = {
            "type": "heading",
            "attrs": {
                "level": 3
            },
            "content": [
                {
                "type": "text",
                "text": "Heading 1"
                }
            ]
        }

    assert adf_to_text([node]) == "### Heading 1\n"

def test_adf_to_text_mediagroup():
    """
    Test that a ``mediaGroup`` with multiple media items is rendered as repeated attachment markers.
    """
    node = {
            "type": "mediaGroup",
            "content": [
                {
                "type": "media",
                "attrs": {
                    "type": "file",
                    "id": "6e7c7f2c-dd7a-499c-bceb-6f32bfbf30b5",
                    "collection": "ae730abd-a389-46a7-90eb-c03e75a45bf6"
                }
                },
                {
                "type": "media",
                "attrs": {
                    "type": "file",
                    "id": "6e7c7f2c-dd7a-499c-bceb-6f32bfbf30b5",
                    "collection": "ae730abd-a389-46a7-90eb-c03e75a45bf6"
                }
                }
            ]
        }
    
    assert adf_to_text([node]) == "[FILE ATTACHMENT]\n[FILE ATTACHMENT]\n"

def test_adf_to_text_mediasingle():
    """
    Test that a ``mediaSingle`` node is rendered as a single attachment marker.
    """
    node = {
            "type": "mediaSingle",
            "attrs": {
                "layout": "center"
            },
            "content": [
                {
                "type": "media",
                "attrs": {
                    "id": "4478e39c-cf9b-41d1-ba92-68589487cd75",
                    "type": "file",
                    "collection": "MediaServicesSample",
                    "alt": "moon.jpeg",
                    "width": 225,
                    "height": 225
                }
                }
            ]
        }
    
    assert adf_to_text([node]) == "[FILE ATTACHMENT]\n"

def test_adf_to_text_ordered_list():
    """
    Test that an ordered list with an explicit starting order renders numbered items from that start.
    """
    node = {
            "type": "orderedList",
            "attrs": {
                "order": 3
            },
            "content": [
                {
                "type": "listItem",
                "content": [
                    {
                    "type": "paragraph",
                    "content": [
                        {
                        "type": "text",
                        "text": "Hello world"
                        }
                    ]
                    }
                ]
                },
                {
                "type": "listItem",
                "content": [
                    {
                    "type": "paragraph",
                    "content": [
                        {
                        "type": "text",
                        "text": "Goodbye world"
                        }
                    ]
                    }
                ]
                }
            ]
        }


    assert adf_to_text([node]) == "3. Hello world\n4. Goodbye world\n"

def test_adf_to_text_ordered_list_no_order():
    """
    Test that an ordered list without an explicit start order begins numbering at 1.
    """
    node = {
            "type": "orderedList",
            "attrs": {
            },
            "content": [
                {
                "type": "listItem",
                "content": [
                    {
                    "type": "paragraph",
                    "content": [
                        {
                        "type": "text",
                        "text": "Hello world"
                        }
                    ]
                    }
                ]
                },
                {
                "type": "listItem",
                "content": [
                    {
                    "type": "paragraph",
                    "content": [
                        {
                        "type": "text",
                        "text": "Goodbye world"
                        }
                    ]
                    }
                ]
                }
            ]
        }

    assert adf_to_text([node]) == "1. Hello world\n2. Goodbye world\n"

def test_adf_to_text_panel():
    """
    Test that a panel node is rendered with its panel type prefix followed by the body text.
    """
    node = {
            "type": "panel",
            "attrs": {
                "panelType": "info"
            },
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Hello world"
                    }
                ]
                }
            ]
        }
    
    assert adf_to_text([node]) == "INFO: Hello world\n"

def test_adf_to_text_paragraph():
    """
    Test that a paragraph with multiple text nodes is concatenated into a single line with newline.
    """
    node = {
            "type": "paragraph",
            "content": [
                {
                "type": "text",
                "text": "Hello world. "
                },
                {
                "type": "text",
                "text": "How are you?"
                }
            ]
        }
    
    assert adf_to_text([node]) == "Hello world. How are you?\n"

def test_adf_to_text_rule():
    """
    Test that a document containing a paragraph and a rule renders a horizontal rule separator.
    """
    node = {
            "version": 1,
            "type": "doc",
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Hello world"
                    }
                ]
                },
                {
                "type": "rule"
                }
            ]
        }
    
    assert adf_to_text([node]) == "Hello world\n\n---\n"

def test_adf_to_text_table():
    """
    Test that a table node is rendered as pipe-separated rows with a trailing newline.
    """
    node = {
            "type": "table",
            "attrs": {
                "isNumberColumnEnabled": False,
                "layout": "center",
                "width": 900,
                "displayMode": "default"
            },
            "content": [
                {
                "type": "tableRow",
                "content": [
                    {
                    "type": "tableCell",
                    "attrs": {},
                    "content": [
                        {
                        "type": "paragraph",
                        "content": [
                            {
                            "type": "text",
                            "text": " Row one, cell one"
                            }
                        ]
                        }
                    ]
                    },
                    {
                    "type": "tableCell",
                    "attrs": {},
                    "content": [
                        {
                        "type": "paragraph",
                        "content": [
                            {
                            "type": "text",
                            "text": "Row one, cell two"
                            }
                        ]
                        }
                    ]
                    }
                ]
                },
                {
                "type": "tableRow",
                "content": [
                    {
                    "type": "tableCell",
                    "attrs": {},
                    "content": [
                        {
                        "type": "paragraph",
                        "content": [
                            {
                            "type": "text",
                            "text": " Row two, cell one"
                            }
                        ]
                        }
                    ]
                    },
                    {
                    "type": "tableCell",
                    "attrs": {},
                    "content": [
                        {
                        "type": "paragraph",
                        "content": [
                            {
                            "type": "text",
                            "text": "Row two, cell two"
                            }
                        ]
                        }
                    ]
                    }
                ]
                }
            ]
        }
    
    assert adf_to_text([node]) == "| Row one, cell one|Row one, cell two|\n| Row two, cell one|Row two, cell two|\n"

def test_adf_to_text_list_item():
    """
    Test that a standalone ``listItem`` node is rendered as a single ``- `` prefixed line.
    """
    node = {
        "type": "listItem",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello world",
                    }
                ],
            }
        ],
    }

    assert adf_to_text([node]) == "- Hello world\n"

def test_adf_to_text_media_block():
    """
    Test that a ``media`` block with marks is rendered as a file attachment placeholder.
    """
    node = {
            "type": "media",
            "attrs": {
                "id": "4478e39c-cf9b-41d1-ba92-68589487cd75",
                "type": "file",
                "collection": "MediaServicesSample",
                "alt": "moon.jpeg",
                "width": 225,
                "height": 225
            },
            "marks": [
                {
                "type": "link",
                "attrs": {
                    "href": "https://developer.atlassian.com/platform/atlassian-document-format/concepts/document-structure/nodes/media/#media"
                }
                },
                {
                "type": "border",
                "attrs": {
                    "color": "#091e4224",
                    "size": 2
                }
                },
                {
                "type": "annotation",
                "attrs": {
                    "id": "c4cbe18e-9902-4734-bf9b-1426a81ef785",
                    "annotationType": "inlineComment"
                }
                }
            ]
        }

    assert adf_to_text([node]) == "[FILE ATTACHMENT]\n"

def test_adf_to_text_nested_expand():
    """
    Test that a ``nestedExpand`` node is rendered equivalently to an ``expand`` node.
    """
    node = {
            "type": "nestedExpand",
            "attrs": {
                "title": "Hello world"
            },
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Hello world"
                    }
                ]
                }
            ]
        }

    assert adf_to_text([node]) == "Hello world\n\nHello world\n"

def test_adf_to_text_table_cell_alone():
    """
    Test that a standalone table cell is rendered as the text content of its paragraph.
    """
    node = {
            "type": "tableCell",
            "attrs": {},
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Hello world"
                    }
                ]
                }
            ]
        }

    assert adf_to_text([node]) == "Hello world"

def test_adf_to_text_table_header_alone():
    """
    Test that a standalone table header is rendered as the text content of its paragraph.
    """
    node = {
            "type": "tableHeader",
            "attrs": {},
            "content": [
                {
                "type": "paragraph",
                "content": [
                    {
                    "type": "text",
                    "text": "Hello world header"
                    }
                ]
                }
            ]
        }

    assert adf_to_text([node]) == "Hello world header"

def test_adf_to_text_table_row():
    """
    Test that a table row of headers is rendered as a single pipe-separated header row.
    """
    node = {
            "type": "tableRow",
            "content": [
                {
                "type": "tableHeader",
                "attrs": {},
                "content": [
                    {
                    "type": "paragraph",
                    "content": [
                        {
                        "type": "text",
                        "text": "Heading one",
                        "marks": [
                            {
                            "type": "strong"
                            }
                        ]
                        }
                    ]
                    }
                ]
                },
                {
                "type": "tableHeader",
                "attrs": {},
                "content": [
                    {
                    "type": "paragraph",
                    "content": [
                        {
                        "type": "text",
                        "text": "Heading two",
                        "marks": [
                            {
                            "type": "strong"
                            }
                        ]
                        }
                    ]
                    }
                ]
                }
            ]
        }
    
    assert adf_to_text([node]) == "|Heading one|Heading two|\n"

def test_adf_to_text_date():
    """
    Test that a date node with a timestamp is rendered as the raw timestamp string.
    """
    node = {
        "type": "date",
        "attrs": {
            "timestamp": "1582152559",
        },
    }

    assert adf_to_text([node]) == "1582152559"

def test_adf_to_text_date_no_timestamp():
    """
    Test that a date node without a timestamp attribute is rendered as an empty string.
    """
    node = {
        "type": "date",
        "attrs": {
        },
    }

    assert adf_to_text([node]) == ""

def test_adf_to_text_emoji():
    """
    Test that an emoji node with a ``shortName`` is rendered as that short name.
    """
    node = {
        "type": "emoji",
        "attrs": {
            "shortName": ":grinning:",
            "text": "",
        },
    }

    assert adf_to_text([node]) == ":grinning:"

def test_adf_to_text_emoji_missing_shortname():
    """
    Test that an emoji node without a ``shortName`` attribute is rendered as an empty string.
    """
    node = {"type": "emoji", "attrs": {}}

    assert adf_to_text([node]) == ""

def test_adf_to_text_hard_break_in_paragraph():
    """
    Test that a ``hardBreak`` inside a paragraph is rendered as a line break in the output.
    """
    node = {
        "type": "paragraph",
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "hardBreak"},
            {"type": "text", "text": "world"},
        ],
    }

    assert adf_to_text([node]) == "Hello\nworld\n"

def test_adf_to_text_inline_card():
    """
    Test that an inline card with a URL attribute is rendered as that URL.
    """
    node = {
        "type": "inlineCard",
        "attrs": {
            "url": "https://atlassian.com",
        },
    }

    assert adf_to_text([node]) == "https://atlassian.com"

def test_adf_to_text_inline_card_empty_attrs():
    """
    Test that an inline card without a URL attribute is rendered as an empty string.
    """
    node = {
        "type": "inlineCard",
        "attrs": {
        },
    }

    assert adf_to_text([node]) == ""

def test_adf_to_text_mention():
    """
    Test that a mention node with explicit text is rendered as that text.
    """
    node = {
        "type": "mention",
        "attrs": {
            "id": "ABCDE-ABCDE-ABCDE-ABCDE",
            "text": "@Bradley Ayers",
            "userType": "APP",
        },
    }

    assert adf_to_text([node]) == "@Bradley Ayers"

def test_adf_to_text_mention_no_text():
    """
    Test that a mention node without text falls back to rendering the ``id`` attribute.
    """
    node = {
        "type": "mention",
        "attrs": {
            "id": "ABCDE-ABCDE-ABCDE-ABCDE",
            "userType": "APP",
        },
    }

    assert adf_to_text([node]) == "@ABCDE-ABCDE-ABCDE-ABCDE"

def test_adf_to_text_status():
    """
    Test that a status node with text is rendered as its text label.
    """
    node = {
        "type": "status",
        "attrs": {
            "localId": "abcdef12-abcd-abcd-abcd-abcdef123456",
            "text": "In Progress",
            "color": "yellow",
        },
    }

    assert adf_to_text([node]) == "In Progress"

def test_adf_to_text_status_no_text():
    """
    Test that a status node without text is rendered as an empty string.
    """
    node = {
        "type": "status",
        "attrs": {
            "localId": "abcdef12-abcd-abcd-abcd-abcdef123456",
            "color": "yellow",
        },
    }

    assert adf_to_text([node]) == ""

def test_adf_to_text_text_node():
    """
    Test that a plain text node is rendered as its raw text.
    """
    node = {
        "type": "text",
        "text": "Hello world",
    }

    assert adf_to_text([node]) == "Hello world"

def test_adf_to_text_code_mark():
    """
    Test that a text node with a ``code`` mark is wrapped in backticks.
    """
    node = {
        "type": "text",
        "text": "Hello world",
        "marks": [
            {
            "type": "code"
            }
        ]
    }

    assert adf_to_text([node]) == "`Hello world`"

def test_adf_to_text_link_mark():
    """
    Test that a text node with a link mark is rendered in Markdown link form.
    """
    node = {
            "type": "text",
            "text": "Hello world",
            "marks": [
                {
                "type": "link",
                "attrs": {
                    "href": "http://atlassian.com",
                    "title": "Atlassian"
                }
                }
            ]
        }

    assert adf_to_text([node]) == "[Hello world](http://atlassian.com)"

def test_adf_to_text_code_and_link_marks():
    """
    Test that a text node with both code and link marks is rendered as a code span inside a link.
    """
    node = {
        "type": "text",
        "text": "sbatch",
        "marks": [
            {"type": "code"},
            {"type": "link", "attrs": {"href": "http://example.com"}},
        ],
    }

    assert adf_to_text([node]) == "[`sbatch`](http://example.com)"

def test_adf_to_text_unknown_node_type_recurse_content():
    """
    Test that an unknown node type with content is rendered by recursively processing its children.
    """
    node = {
        "type": "unknownNode",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Hello world"},
                ],
            }
        ],
    }

    assert adf_to_text([node]) == "Hello world\n"

def test_adf_to_text_unknown_node_type_no_content():
    """
    Test that an unknown node type without content is rendered as an empty string.
    """
    node = {
        "type": "unknownNode",
        "content": [],
    }

    assert adf_to_text([node]) == ""

def test_build_jira_ticket_text_basic_structure_and_assignee_role():
    """
    Test that ``build_jira_ticket_text`` builds the expected header, initial description,
    and a single assignee message with the correct role and message id.
    """
    ticket = _make_base_ticket()
    ticket["fields"]["comment"]["comments"] = [
        {
            "author": {
                "emailAddress": "assignee@example.com",
                "displayName": "Helpdesk Assignee",
            },
            "body": {
                "type": "paragraph",
                "content": [{"type": "text", "text": "First comment"}],
            },
        }
    ]

    text = build_jira_ticket_text(ticket)

    assert text.startswith("<BEGIN_TICKET>\n")
    assert "[TICKET_KEY] TEST-1\n" in text
    assert "[STATUS] OPEN\n" in text
    assert "[CREATED] 2025-01-01T12:00:00.000+0000\n" in text
    assert "[SUMMARY] Test summary\n" in text
    assert "[INITIAL_DESCRIPTION]\nTicket description\n\n" in text

    expected_message = (
        "<MESSAGE id=0001 role=HELPDESK_ASSIGNEE>\n"
        "First comment\n"
        "</MESSAGE>\n"
    )
    assert expected_message in text
    assert text.rstrip().endswith("<END_TICKET>")

def test_build_jira_ticket_text_no_comments():
    """
    Test that a ticket with no comments still includes the conversation header and no messages.
    """
    ticket = _make_base_ticket()
    ticket["fields"]["comment"]["comments"] = []
    text = build_jira_ticket_text(ticket)
    assert "[CONVERSATION]\n" in text
    assert "<MESSAGE id=" not in text

def test_build_jira_ticket_text_multiple_comments_numbering():
    """
    Test that multiple comments are rendered as sequentially numbered ``MESSAGE`` blocks.
    """
    ticket = _make_base_ticket()
    ticket["fields"]["comment"]["comments"] = [
        {
            "author": {"emailAddress": "assignee@example.com", "displayName": "Helpdesk Assignee"},
            "body": {"type": "paragraph", "content": [{"type": "text", "text": "First comment"}]},
        },
        {
            "author": {"emailAddress": "assignee@example.com", "displayName": "Helpdesk Assignee"},
            "body": {"type": "paragraph", "content": [{"type": "text", "text": "Second comment"}]},
        },
        {
            "author": {"emailAddress": "assignee@example.com", "displayName": "Helpdesk Assignee"},
            "body": {"type": "paragraph", "content": [{"type": "text", "text": "Third comment"}]},
        },
    ]

    text = build_jira_ticket_text(ticket)
    idx1 = text.index("MESSAGE id=0001")
    idx2 = text.index("MESSAGE id=0002")
    idx3 = text.index("MESSAGE id=0003")
    assert idx1 < idx2 < idx3

def test_build_jira_ticket_text_uses_adf_to_text_for_description_and_comments():
    """
    Test that both the initial description and comments are rendered using ``adf_to_text``.
    """
    ticket = _make_base_ticket(description_text="Line one.\nLine two.")
    ticket["fields"]["comment"]["comments"] = [
        {
            "author": {"emailAddress": "assignee@example.com", "displayName": "Helpdesk Assignee"},
            "body": {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Comment line one."},
                    {"type": "hardBreak"},
                    {"type": "text", "text": "Comment line two."},
                ],
            },
        }
    ]

    text = build_jira_ticket_text(ticket)
    assert "[INITIAL_DESCRIPTION]\nLine one.\nLine two.\n\n" in text
    assert (
        "<MESSAGE id=0001 role=HELPDESK_ASSIGNEE>\n"
        "Comment line one.\nComment line two.\n"
        "</MESSAGE>\n"
    ) in text

def test_build_jira_ticket_text_creator_role():
    """
    Test that a comment authored by the ticket creator is tagged with the ``TICKET_CREATOR`` role.
    """
    ticket = _make_base_ticket()
    ticket["fields"]["comment"]["comments"] = [
        {
            "author": {"emailAddress": "creator@example.com", "displayName": "Ticket Creator"},
            "body": {"type": "paragraph", "content": [{"type": "text", "text": "Creator comment"}]},
        }
    ]
    text = build_jira_ticket_text(ticket)
    assert "role=TICKET_CREATOR" in text

def test_build_jira_ticket_text_principal_investigator_role():
    """
    Test that a comment authored by the principal investigator is tagged with the corresponding role.
    """
    ticket = _make_base_ticket()
    ticket["fields"]["comment"]["comments"] = [
        {
            "author": {"emailAddress": "pi@example.com", "displayName": "Principal Investigator"},
            "body": {"type": "paragraph", "content": [{"type": "text", "text": "PI comment"}]},
        }
    ]
    text = build_jira_ticket_text(ticket)
    assert "role=HELPDESK_PRINCIPAL_INVESTIGATOR" in text

def test_build_jira_ticket_text_reporter_role_when_different_from_creator():
    """
    Test that a comment authored by the reporter, when distinct from the creator, uses the ``REPORTER`` role.
    """
    ticket = _make_base_ticket()
    ticket["fields"]["comment"]["comments"] = [
        {
            "author": {"emailAddress": "reporter@example.com", "displayName": "Ticket Reporter"},
            "body": {"type": "paragraph", "content": [{"type": "text", "text": "Reporter comment"}]},
        }
    ]
    text = build_jira_ticket_text(ticket)
    assert "role=REPORTER" in text

def test_build_jira_ticket_text_automation_pseudouser_role():
    """
    Test that a comment authored by an automation pseudo-user is tagged with the automation role.
    """
    ticket = _make_base_ticket()
    ticket["fields"]["comment"]["comments"] = [
        {
            "author": {"emailAddress": "", "displayName": "Automation Pseudo-User"},
            "body": {"type": "paragraph", "content": [{"type": "text", "text": "Automated comment"}]},
        }
    ]
    text = build_jira_ticket_text(ticket)
    assert "role=AUTOMATED_PSEUDOUSER" in text

def test_build_jira_ticket_text_other_role_fallback():
    """
    Test that comments by users that do not match any special role mapping fall back to ``OTHER``.
    """
    ticket = _make_base_ticket()
    ticket["fields"]["comment"]["comments"] = [
        {
            "author": {"emailAddress": "someone_else@example.com", "displayName": "Random User"},
            "body": {"type": "paragraph", "content": [{"type": "text", "text": "Other user comment"}]},
        }
    ]
    text = build_jira_ticket_text(ticket)
    assert "role=OTHER" in text

@pytest.mark.parametrize("field_name", ["assignee", "creator", "reporter"])
@pytest.mark.parametrize("mode", ["none", "missing"])
def test_build_jira_ticket_text_handles_missing_or_none_user_fields(field_name, mode):
    """
    ``build_jira_ticket_text`` must handle ``assignee``, ``creator`` and
    ``reporter`` being ``None`` or entirely missing from ``ticket["fields"]``
    without raising and still render the conversation.
    """
    ticket = _make_base_ticket()
    fields = ticket["fields"]

    if mode == "none":
        fields[field_name] = None
    elif mode == "missing":
        fields.pop(field_name, None)

    comment = copy.deepcopy(fields["comment"]["comments"][0]) if fields["comment"]["comments"] else {
        "author": {"emailAddress": "creator@example.com", "displayName": "Ticket Creator"},
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Test comment"}]}
            ],
        },
    }
    fields["comment"]["comments"] = [comment]

    text = build_jira_ticket_text(ticket)

    assert "Test comment" in text
    assert "<MESSAGE id=0001" in text

@pytest.mark.parametrize(
    "pi_value",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty_list"),
        pytest.param([None], id="list_with_none"),
    ],
)
def test_build_jira_ticket_text_handles_missing_or_empty_pi_field(pi_value):
    """
    ``build_jira_ticket_text`` must tolerate ``customfield_10042`` being absent,
    ``None``, an empty list, or containing ``None`` as the first element
    without raising.
    """
    ticket = _make_base_ticket()
    fields = ticket["fields"]

    if pi_value is None:
        fields.pop("customfield_10042", None)
    else:
        fields["customfield_10042"] = pi_value

    text = build_jira_ticket_text(ticket)

    assert "[TICKET_KEY]" in text
    assert "[CONVERSATION]" in text
    assert "HELPDESK_PRINCIPAL_INVESTIGATOR" not in text

def test_build_jira_ticket_text_missing_required_field_raises_keyerror():
    """
    Test that omitting a required ticket field (such as ``summary``) raises ``KeyError``.
    """
    ticket = _make_base_ticket()
    del ticket["fields"]["summary"]
    with pytest.raises(KeyError):
        build_jira_ticket_text(ticket)

def test_preprocess_jira_ticket_resolved_happy_path(monkeypatch):
    """
    Resolved ticket: basic metadata, status uppercasing, non-null time_to_resolution,
    index_version, and delegation to build_jira_ticket_text.
    """
    ticket = _make_preprocess_ticket(
        key="HPCSSUP-12345",
        summary="Internal HPC Application Form",
        status_name="Resolved",
        created="2025-01-01T09:00:00.000+0000",
        updated="2025-01-01T10:00:00.000+0000",
        resolution_date="2025-01-01T11:00:00.000+0000",
    )

    def fake_builder(t):
        assert t is ticket
        return "TRANSCRIPT"

    monkeypatch.setattr(
        document_preprocessor,
        "build_jira_ticket_text",
        fake_builder,
    )

    doc = preprocess_jira_ticket(ticket)

    assert doc.text == "TRANSCRIPT"

    assert doc.id == "HPCSSUP-12345"
    assert doc.metadata["ticket_key"] == "HPCSSUP-12345"
    assert doc.metadata["summary"] == "Internal HPC Application Form"
    assert doc.metadata["created_at"] == "2025-01-01T09:00:00.000+0000"
    assert doc.metadata["updated_at"] == "2025-01-01T10:00:00.000+0000"
    assert doc.metadata["resolved_at"] == "2025-01-01T11:00:00.000+0000"

    assert doc.metadata["status"] == "RESOLVED"

    assert doc.metadata["time_to_resolution"] is not None
    assert isinstance(doc.metadata["time_to_resolution"], float)

    assert "ingestion_date" in doc.metadata
    assert isinstance(doc.metadata["ingestion_date"], str)
    assert doc.metadata["index_version"] == "test"

    assert doc.metadata["document_type"] == "helpdesk_ticket"
    assert doc.document_type == "helpdesk_ticket"


def test_preprocess_jira_ticket_unresolved_ticket_sets_time_to_resolution_none(monkeypatch):
    """
    Unresolved ticket: resolutionDate is None, so resolved_at and time_to_resolution
    reflect that.
    """
    ticket = _make_preprocess_ticket(
        key="HPCSSUP-12346",
        status_name="Open",
        resolution_date=None,
    )

    monkeypatch.setattr(
        document_preprocessor,
        "build_jira_ticket_text",
        lambda t: "TRANSCRIPT",
    )

    doc = preprocess_jira_ticket(ticket)

    assert doc.metadata["resolved_at"] is None
    assert doc.metadata["time_to_resolution"] is None
    assert doc.metadata["status"] == "OPEN"


def test_preprocess_jira_ticket_missing_required_field_raises_keyerror(monkeypatch):
    """
    If a mandatory field (e.g. summary) is missing, we expect a KeyError.
    """
    ticket = _make_preprocess_ticket()
    del ticket["fields"]["summary"]

    monkeypatch.setattr(
        document_preprocessor,
        "build_jira_ticket_text",
        lambda t: "TRANSCRIPT",
    )

    with pytest.raises(KeyError):
        preprocess_jira_ticket(ticket)


def test_preprocess_jira_ticket_invalid_date_raises_valueerror(monkeypatch):
    """
    Invalid date format should propagate as ValueError when parsing dates.
    """
    ticket = _make_preprocess_ticket(
        created="not-a-date",
        resolution_date="2025-01-01T11:00:00.000+0000",
    )

    monkeypatch.setattr(
        document_preprocessor,
        "build_jira_ticket_text",
        lambda t: "TRANSCRIPT",
    )

    with pytest.raises(ValueError):
        preprocess_jira_ticket(ticket)


def test_preprocess_jira_ticket_uses_build_jira_ticket_text(monkeypatch):
    """
    Ensure preprocess_jira_ticket delegates transcript construction to
    build_jira_ticket_text.
    """
    ticket = _make_preprocess_ticket()

    calls = {"count": 0}

    def fake_builder(t):
        calls["count"] += 1
        return "FROM_BUILDER"

    monkeypatch.setattr(
        document_preprocessor,
        "build_jira_ticket_text",
        fake_builder,
    )

    doc = preprocess_jira_ticket(ticket)

    assert calls["count"] == 1
    assert doc.text == "FROM_BUILDER"


def test_preprocess_jira_ticket_no_anonymizer_leaves_text_unchanged(monkeypatch):
    """
    When no anonymizer is provided (or anonymizer is None), the transcript
    from build_jira_ticket_text should be used as-is.
    """
    ticket = _make_preprocess_ticket()

    original_text = "User email: user@example.com"

    monkeypatch.setattr(
        document_preprocessor,
        "build_jira_ticket_text",
        lambda t: original_text,
    )

    doc = preprocess_jira_ticket(ticket)

    assert doc.text == original_text


def test_preprocess_jira_ticket_applies_anonymizer(monkeypatch):
    """
    When an anonymizer is provided, it should be called once with the
    raw transcript and its output used as Document.text.
    """
    ticket = _make_preprocess_ticket()

    original_text = "User email: user@example.com"
    monkeypatch.setattr(
        document_preprocessor,
        "build_jira_ticket_text",
        lambda t: original_text,
    )

    calls = {}

    def fake_anonymizer(s: str) -> str:
        calls["arg"] = s
        return s.replace("user@example.com", "<EMAIL>")

    doc = preprocess_jira_ticket(ticket, anonymizer=fake_anonymizer)

    assert calls["arg"] == original_text
    assert doc.text == "User email: <EMAIL>"
    assert doc.metadata["summary"] == "Test summary"
    assert doc.metadata["ticket_key"] == "TEST-1"


def test_preprocess_jira_ticket_anonymizer_identity(monkeypatch):
    """
    Identity anonymizer should not change the transcript, but should still be
    compatible with the preprocess flow.
    """
    ticket = _make_preprocess_ticket()

    monkeypatch.setattr(
        document_preprocessor,
        "build_jira_ticket_text",
        lambda t: "TRANSCRIPT",
    )

    identity = lambda s: s 

    doc = preprocess_jira_ticket(ticket, anonymizer=identity)

    assert doc.text == "TRANSCRIPT"

def test_preprocess_jira_tickets_empty_list_returns_empty_list():
    """
    An empty input list should produce an empty output list.
    """
    docs = preprocess_jira_tickets([])
    assert docs == []


def test_preprocess_jira_tickets_delegates_and_preserves_order(monkeypatch):
    """
    Multiple tickets: ensure preprocess_jira_ticket is called once per ticket,
    in order, and that the returned list preserves that order.
    """
    t1 = _make_preprocess_ticket(key="T-1")
    t2 = _make_preprocess_ticket(key="T-2")
    t3 = _make_preprocess_ticket(key="T-3")
    tickets = [t1, t2, t3]

    calls: list[str] = []

    def fake_preprocess(ticket: dict):
        key = ticket["fields"]["key"]
        calls.append(key)
        return f"DOC-{key}"

    monkeypatch.setattr(
        document_preprocessor,
        "preprocess_jira_ticket",
        fake_preprocess,
    )

    docs = preprocess_jira_tickets(tickets)

    assert calls == ["T-1", "T-2", "T-3"]

    assert docs == ["DOC-T-1", "DOC-T-2", "DOC-T-3"]

    assert isinstance(docs, list)


def test_preprocess_jira_tickets_single_ticket(monkeypatch):
    """
    Single-ticket convenience case: still returns a list with one element.
    """
    ticket = _make_preprocess_ticket(key="ONLY-1")

    def fake_preprocess(ticket: dict):
        return "DOC-ONLY-1"

    monkeypatch.setattr(
        document_preprocessor,
        "preprocess_jira_ticket",
        fake_preprocess,
    )

    docs = preprocess_jira_tickets([ticket])

    assert isinstance(docs, list)
    assert docs == ["DOC-ONLY-1"]


def test_preprocess_jira_tickets_propagates_exceptions(monkeypatch):
    """
    If preprocess_jira_ticket raises, the exception should propagate and not
    be swallowed or wrapped by preprocess_jira_tickets.
    """
    t1 = _make_preprocess_ticket(key="T-1")
    t2 = _make_preprocess_ticket(key="T-2")
    tickets = [t1, t2]

    calls: list[str] = []

    def fake_preprocess(ticket: dict):
        key = ticket["fields"]["key"]
        calls.append(key)
        if key == "T-2":
            raise KeyError("boom")
        return f"DOC-{key}"

    monkeypatch.setattr(
        document_preprocessor,
        "preprocess_jira_ticket",
        fake_preprocess,
    )

    with pytest.raises(KeyError, match="boom"):
        preprocess_jira_tickets(tickets)

    assert calls == ["T-1", "T-2"]