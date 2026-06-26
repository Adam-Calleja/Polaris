"""polaris_rag.retrieval.document_preprocessor

Document preprocessing utilities for multiple source formats.

This module contains helpers for:
- sanitising and normalising HTML documents prior to indexing
- converting Jira Atlassian Document Format (ADF) JSON to plain text
- producing enriched :class:`~polaris_rag.common.schemas.Document` objects for Jira tickets

Functions
---------
preprocess_html
    Remove specified HTML tags and conditionally remove elements.
preprocess_html_documents
    Apply :func:`preprocess_html` to a list of HTML :class:`~polaris_rag.common.schemas.Document` objects.
apply_marks_to_text
    Apply ADF text marks (e.g., code, link) to a string.
adf_to_text
    Convert Atlassian Document Format (ADF) JSON into plain text.
build_jira_ticket_text
    Build a plain-text transcript for a Jira ticket, including comments.
preprocess_jira_ticket
    Convert a Jira issue dictionary into a :class:`~polaris_rag.common.schemas.Document`.
preprocess_jira_tickets
    Batch wrapper for :func:`preprocess_jira_ticket`.
"""

from bs4 import BeautifulSoup, UnicodeDammit
from datetime import datetime 

from polaris_rag.common import Document, MarkdownDocument


def _jira_comment_role(ticket_fields: dict, comment: dict) -> str:
    """Infer a stable role label for a Jira comment author."""
    assignee = ticket_fields.get("assignee") or {}
    creator = ticket_fields.get("creator") or {}
    reporter = ticket_fields.get("reporter") or {}
    pi_field = ticket_fields.get("customfield_10042")

    if isinstance(pi_field, list) and pi_field:
        principal_investigator = pi_field[0] or {}
    else:
        principal_investigator = {}

    author = comment.get("author", {})
    author_email = author.get("emailAddress", "")
    author_displayname = author.get("displayName", "")

    if author_displayname == "Automation Pseudo-User":
        return "AUTOMATED_PSEUDOUSER"
    if author_email == assignee.get("emailAddress", ""):
        return "HELPDESK_ASSIGNEE"
    if author_email == creator.get("emailAddress", ""):
        return "TICKET_CREATOR"
    if author_email == principal_investigator.get("emailAddress", ""):
        return "HELPDESK_PRINCIPAL_INVESTIGATOR"
    if creator.get("emailAddress", "") != reporter.get("emailAddress", "") and author_email == reporter.get("emailAddress", ""):
        return "REPORTER"
    return "OTHER"


def _build_jira_ticket_metadata(ticket: dict) -> dict[str, object]:
    """Build jira Ticket Metadata.
    
    Parameters
    ----------
    ticket : dict
        Value for ticket.
    
    Returns
    -------
    dict[str, object]
        Structured result of the operation.
    """
    created_date = ticket["fields"].get("created", None)
    updated_date = ticket["fields"].get("updated", None)
    resolution_date = ticket["fields"].get("resolutionDate", None)
    summary = ticket["fields"].get("summary", None)
    key = ticket.get("key", None)
    status = ticket["fields"]["status"]["name"].upper()

    date_format = "%Y-%m-%dT%H:%M:%S.%f%z"

    if resolution_date:
        start = datetime.strptime(created_date, date_format)
        end = datetime.strptime(resolution_date, date_format)
        delta = end - start
        time_to_resolution = delta.total_seconds()
    else:
        time_to_resolution = None

    ingestion_date = datetime.strftime(datetime.now(), date_format)
    index_version = "test"

    return {
        "created_at": created_date,
        "updated_at": updated_date,
        "resolved_at": resolution_date,
        "time_to_resolution": time_to_resolution,
        "summary": summary,
        "ticket_key": key,
        "document_type": "helpdesk_ticket",
        "status": status,
        "ingestion_date": ingestion_date,
        "index_version": index_version,
    }


def _normalise_markdown(text: str) -> str:
    """Normalize markdown.
    
    Parameters
    ----------
    text : str
        Text value to inspect, tokenize, or encode.
    
    Returns
    -------
    str
        Resulting string value.
    """
    lines = [(line or "").rstrip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    output = "\n".join(lines)
    while "\n\n\n" in output:
        output = output.replace("\n\n\n", "\n\n")
    return output.strip()

def preprocess_html(
        html: str,
        tags: list[str],
        conditions: list[dict[str, str]]) -> str:
    """Preprocess an HTML document by removing specified tags.

    This function:
    - removes all tags named in ``tags`` unconditionally
    - removes tags specified by ``conditions`` only when their condition evaluates
      to ``True`` for a given element

    Parameters
    ----------
    html : str
        HTML content to preprocess.
    tags : list[str]
        Tag names to remove entirely (e.g., ``["script", "style"]``).
    conditions : list[dict[str, str]]
        Conditional removal rules. Each rule must contain:
        - ``"tag"``: the tag name to search for
        - ``"condition"``: a Python boolean expression (string) evaluated with
          ``element`` bound to the current BeautifulSoup element.

    Returns
    -------
    str
        Sanitised HTML as a string.

    Raises
    ------
    KeyError
        If a condition rule is missing required keys (``"tag"`` or ``"condition"``).
    SyntaxError
        If a condition expression cannot be parsed by :func:`eval`.
    Exception
        Any exception raised while evaluating a condition expression.

    Notes
    -----
    Conditional removal uses :func:`eval` with ``element`` in scope. Only use
    trusted condition strings.
    """

    dammit = UnicodeDammit(html, smart_quotes_to="unicode")
    fixed_html = dammit.unicode_markup

    soup = BeautifulSoup(fixed_html, 'html.parser')

    for tag in soup(tags):
        tag.decompose()

    for condition_dict in conditions:
        tag = condition_dict["tag"]
        condition = condition_dict["condition"]
        for element in soup.find_all(tag):
            # BeautifulSoup tags that were already removed as descendants of an
            # earlier decompose() call can remain in the precomputed find_all()
            # result list with attrs set to None. Skip those detached nodes
            # before evaluating the configured condition expression.
            if getattr(element, "parent", None) is None or getattr(element, "attrs", None) is None:
                continue
            if eval(condition):
                element.decompose()

    return str(soup)

def preprocess_html_documents(
        documents: list[Document],
        tags: list[str],
        conditions: list[dict[str, str]]) -> list[Document]:
    """Preprocess a list of HTML documents by removing specified tags.

    For each document in ``documents``, this function calls
    :func:`preprocess_html` on ``document.text`` and returns a new
    :class:`~polaris_rag.common.schemas.Document` with the processed text while
    preserving ``id``, ``document_type``, and ``metadata``.

    Parameters
    ----------
    documents : list[Document]
        HTML documents to preprocess.
    tags : list[str]
        Tag names to remove entirely.
    conditions : list[dict[str, str]]
        Conditional removal rules passed through to :func:`preprocess_html`.

    Returns
    -------
    list[Document]
        Processed documents (same length/order as the input).
    """

    processed_documents = []

    for document in documents:
        id = document.id 
        text = document.text
        document_type = document.document_type
        metadata = document.metadata

        processed_text = preprocess_html(html=text, tags=tags, conditions=conditions)

        processed_documents.append(Document(
            text=processed_text,
            id=id,
            document_type=document_type,
            metadata=metadata
        ))

    return processed_documents

def apply_marks_to_text(text: str, mark_list: list[dict]) -> str:
    """Apply ADF marks to a text string.

    Parameters
    ----------
    text : str
        Text to which marks should be applied.
    mark_list : list[dict]
        List of ADF mark dictionaries. Each dictionary should have a ``"type"``
        key and optionally an ``"attrs"`` key.

    Returns
    -------
    str
        Text with marks applied. Code marks are wrapped in backticks, and link
        marks are converted to Markdown-style links.
    """
    output = text

    has_code = any(mark.get("type") == "code" for mark in mark_list)
    if has_code:
        output = f"`{output}`"

    link_mark = next((mark for mark in mark_list if mark.get("type") == "link"), None)
    if link_mark:
        href = link_mark.get("attrs", {}).get("href", "")
        output = f"[{output}]({href})" if href else output

    return output

def adf_to_text(node) -> str:
    """
    Convert Atlassian Document Format (ADF) JSON to plain text.

    Supports common ADF node types and produces a readable text layout with
    newlines, list markers, blockquotes, and simple placeholders for media.

    Parameters
    ----------
    node : dict | list | None
        An ADF node (``dict``), a list of nodes, or ``None``. The function
        expects the ADF schema used by Jira descriptions and comments, where
        nodes have keys such as ``"type"``, ``"content"``, ``"text"``, and
        optional ``"attrs"``.

    Returns
    -------
    str
        Plain-text representation of the input ADF.

    Notes
    -----
    The following node types are handled:

    - ``"text"``: emits the node's ``text``.
    - ``"hardBreak"``: emits ``"\\n"``.
    - ``"paragraph"``, ``"heading"``, ``"panel"``: concatenates child text and appends ``"\\n"``.
    - ``"bulletList"``: concatenates child items as lines prefixed by ``"- "`` (via ``listItem``).
    - ``"orderedList"``: emits numbered items (``"1. ..."``).
    - ``"listItem"``: emits a line prefixed by ``"- "`` followed by child text.
    - ``"blockquote"``: prefixes each line with ``"> "``.
    - ``"codeBlock"``: emits child text followed by ``"\\n"``.
    - ``"rule"``: emits a horizontal rule marker ``"\\n---\\n"``.
    - ``"media"``/``"mediaSingle"``: emits ``"[attachment]\\n"`` placeholder.
    - ``"mention"``: emits ``"@{text or id}"``.
    - ``"emoji"``: emits ``attrs["shortName"]`` if present.

    Unrecognised types are traversed recursively via their ``content`` if any.

    Examples
    --------
    >>> adf_to_text({"type": "paragraph", "content": [{"type": "text", "text": "Hello"}]})
    'Hello\\n'
    >>> adf_to_text({"type": "orderedList", "content": [{"type": "listItem",
    ...   "content": [{"type": "paragraph", "content": [{"type": "text", "text": "First"}]}]}]})
    '1. First\\n'
    """
    if not node:
        return ""
    if isinstance(node, list):
        return "".join(adf_to_text(n) for n in node)

    node_type = node.get("type")
    content = node.get("content", [])
    text = node.get("text", "")
    marks = node.get("marks", [])

    if node_type == 'blockquote':
        body = "".join(adf_to_text(node) for node in content)
        return "\n".join((f"> {line}" if line else "") for line in body.splitlines()) + "\n"
    if node_type == "bulletList":
        items = []
        for node in content:
            item_text = adf_to_text(node).strip()
            if item_text.startswith("- ") and node.get("type") == "listItem":
                item_text = item_text[2:]

            items.append(f"- {item_text}\n")

        return "".join(items)
    if node_type == "codeBlock":
        language = node.get("attrs", {}).get("language", "")
        return f"```{language}\n" + "".join(adf_to_text(node) for node in content) + "\n```\n"
    if node_type == "expand":
        title = node.get("attrs", {}).get("title", "")
        body = "".join(adf_to_text(node) for node in content).strip()
        return f"{title}\n\n{body}\n"
    if node_type == "heading":
        level = node.get("attrs", {}).get("level", 1)
        heading_text = "".join(adf_to_text(node) for node in content).strip()
        return f"{'#' * level} {heading_text}\n"
    if node_type == "mediaGroup" or node_type == "mediaSingle":
        return "".join(adf_to_text(node) for node in content)
    if node_type == "orderedList":
        order = node.get("attrs", {}).get("order", 1)
        output = []
        for i, node in enumerate(content, order):
            item_text = adf_to_text(node).strip()
            if item_text.startswith("- ") and node.get("type") == "listItem":
                item_text = item_text[2:]
            
            output.append(f"{i}. {item_text}\n")
        return "".join(output)
    if node_type == "panel":
        panel_type = node.get("attrs", {}).get("panelType", "info")
        body = "".join(adf_to_text(node) for node in content).strip()
        return f"{panel_type.upper()}: {body}\n"
    if node_type == "paragraph":
        return "".join(adf_to_text(node) for node in content) + "\n"
    if node_type == "rule":
         return "\n---\n"
    if node_type == "table":
        rows = []
        for row in content:
            row_text = "| " + "|".join(adf_to_text(cell).strip() for cell in row.get("content", [])) + "|\n"
            rows.append(row_text)
        return "".join(rows)
    if node_type == "listItem":
        return "- " + "".join(adf_to_text(node) for node in content).strip() + "\n"
    if node_type == "media":
        media_type = node.get("attrs", {}).get("type", "unknown")
        return f"[{media_type.upper()} ATTACHMENT]\n"
    if node_type == "nestedExpand":
        title = node.get("attrs", {}).get("title", "")
        body = "".join(adf_to_text(node) for node in content).strip()
        return f"{title}\n\n{body}\n"
    if node_type == "tableCell" or node_type == "tableHeader":
        return "".join(adf_to_text(node) for node in content).strip()
    if node_type == "tableRow":
        return "|" + "|".join(adf_to_text(node) for node in content) + "|\n"
    if node_type == "date":
        timestamp = node.get("attrs", {}).get("timestamp", "")
        return str(timestamp) if timestamp else ""
    if node_type == "emoji":
        return node.get("attrs", {}).get("shortName", "")
    if node_type == "hardBreak":
        return "\n"
    if node_type == "inlineCard":
        url = node.get("attrs", {}).get("url", "")
        return url if url else ""
    if node_type == "mention":
        attrs = node.get("attrs", {})
        return attrs.get("text") or "@" + attrs.get("id", "mention")
    if node_type == "status":
        return node.get("attrs", {}).get("text", "")
    if node_type == "text":
        if marks:
            return apply_marks_to_text(text, marks)
        return text
    return "".join(adf_to_text(node) for node in content)

def build_jira_ticket_text(ticket: dict) -> str:
    """
    Build a plain-text transcript for a Jira ticket and its conversation.

    The output contains a header block with key metadata, the initial description
    (converted from ADF), and a ``CONVERSATION`` section where each comment is
    wrapped in a ``<MESSAGE ...>`` tag with a zero-padded incremental id and a
    derived ``role``.

    Parameters
    ----------
    ticket : dict
        Jira issue object as returned by the REST API. The function expects at
        least the following keys/paths:
          - ``ticket["key"]`` : str
          - ``ticket["fields"]["summary"]`` : str
          - ``ticket["fields"]["description"]`` : ADF JSON
          - ``ticket["fields"]["status"]["name"]`` : str
          - ``ticket["fields"]["created"]`` : str (ISO-like datetime)
          - ``ticket["fields"]["comment"]["comments"]`` : list[dict]
          - Optional user fields used for role inference:
            ``assignee``, ``creator``, ``customfield_10042`` (principal investigator),
            ``reporter`` (each with ``emailAddress``).

    Returns
    -------
    str
        A ticket transcript bounded by ``<BEGIN_TICKET>`` and ``<END_TICKET>``.
        Each comment appears as:
        ``<MESSAGE id=0001 role=SOME_ROLE>...\\n</MESSAGE>``

    Raises
    ------
    KeyError
        If required fields are absent from ``ticket``.
    TypeError
        If values are of unexpected types (e.g., description not ADF-like).

    Notes
    -----
    Role assignment is derived by matching the comment author's email against
    ``assignee``, ``creator``, principal investigator (``customfield_10042``),
    and ``reporter``. If no rule matches, role defaults to ``"OTHER"``.
    The initial description and each comment body are converted via
    :func:`adf_to_text`.

    See Also
    --------
    adf_to_text : ADF to text conversion used for descriptions and comments.
    """
    summary = ticket['fields']['summary']
    description = ticket['fields']['description']
    comments = ticket['fields']['comment']['comments']
    key = ticket['key']
    status = ticket['fields']['status']['name'].upper()
    created_date = ticket['fields']['created']

    text = (
        "<BEGIN_TICKET>\n"
        f"[TICKET_KEY] {key}\n"
        f"[STATUS] {status}\n"
        f"[CREATED] {created_date}\n"
        f"[SUMMARY] {summary}\n"
        f"\n[INITIAL_DESCRIPTION]\n{adf_to_text(description)}\n"
        f"\n[CONVERSATION]\n"
    )

    for message_counter in range(len(comments)):
        comment = comments[message_counter]
        user = _jira_comment_role(ticket["fields"], comment)

        message_count = str(message_counter + 1).zfill(4)
        text += f"<MESSAGE id={message_count} role={user}>\n"
        text += adf_to_text(comment['body']).strip()
        text += f"\n</MESSAGE>\n\n"

    text += "<END_TICKET>"

    return text


def build_jira_ticket_markdown(ticket: dict) -> str:
    """Render a Jira ticket into readable Markdown while preserving turns.
    
    Parameters
    ----------
    ticket : dict
        Value for ticket.
    
    Returns
    -------
    str
        Constructed jira Ticket Markdown.
    """
    fields = ticket["fields"]
    summary = fields.get("summary", "")
    description = fields.get("description")
    comments = fields.get("comment", {}).get("comments", [])
    key = ticket["key"]
    status = fields["status"]["name"].upper()
    created_date = fields.get("created", "")
    updated_date = fields.get("updated", "")
    resolved_date = fields.get("resolutionDate", "")

    parts = [
        f"# Ticket {key}",
        "",
        f"- Summary: {summary}",
        f"- Status: {status}",
        f"- Created: {created_date}",
    ]
    if updated_date:
        parts.append(f"- Updated: {updated_date}")
    if resolved_date:
        parts.append(f"- Resolved: {resolved_date}")

    initial_description = _normalise_markdown(adf_to_text(description))
    parts.extend([
        "",
        "## Initial Description",
        "",
        initial_description or "_No description provided._",
        "",
        "## Conversation",
    ])

    if not comments:
        parts.extend(["", "_No comments._"])
        return _normalise_markdown("\n".join(parts))

    for idx, comment in enumerate(comments, start=1):
        role = _jira_comment_role(fields, comment)
        body = _normalise_markdown(adf_to_text(comment.get("body")))
        parts.extend(
            [
                "",
                f"### Message {str(idx).zfill(4)} ({role})",
                "",
                body or "_Empty message._",
            ]
        )

    return _normalise_markdown("\n".join(parts))


def convert_jira_ticket_to_markdown(ticket: dict) -> MarkdownDocument:
    """Convert a Jira issue payload into a markdown-normalized document.
    
    Parameters
    ----------
    ticket : dict
        Value for ticket.
    
    Returns
    -------
    MarkdownDocument
        Result of the operation.
    """
    metadata = _build_jira_ticket_metadata(ticket)
    metadata["conversion_engine"] = "native_jira"
    metadata["source_format"] = "jira_adf"
    return MarkdownDocument(
        text=build_jira_ticket_markdown(ticket),
        document_type="helpdesk_ticket",
        id=str(ticket.get("key", "")),
        metadata=metadata,
        source_node=ticket,
    )


def convert_jira_tickets_to_markdown(tickets: list[dict]) -> list[MarkdownDocument]:
    """Batch-convert Jira issues into markdown-normalized documents.
    
    Parameters
    ----------
    tickets : list[dict]
        Value for tickets.
    
    Returns
    -------
    list[MarkdownDocument]
        Collected results from the operation.
    """
    return [convert_jira_ticket_to_markdown(ticket) for ticket in tickets]

def preprocess_jira_ticket(ticket: dict, anonymizer=None) -> Document:
    """
    Convert a Jira issue to a :class:`~polaris_rag.common.schemas.Document` with enriched metadata.

    This function extracts key fields from a Jira issue, computes timing
    metadata if the ticket is resolved, generates a human-readable ticket
    transcript via :func:`build_jira_ticket_text`, and returns a ``Document``
    object suitable for indexing.

    Parameters
    ----------
    ticket : dict
        Jira issue object (see :func:`build_jira_ticket_text` for required keys).

    Returns
    -------
    Document
        A document with:
          - ``text`` : str
                Output of :func:`build_jira_ticket_text`.
          - ``id`` : str
                The ticket key.
          - ``document_type`` : ``"helpdesk_ticket"``.
          - ``metadata`` : dict
                Contains:
                ``created_at``, ``updated_at``, ``resolved_at``,
                ``time_to_resolution`` (float seconds, or ``None`` if unresolved),
                ``summary``, ``ticket_key``, ``status``,
                ``ingestion_date``, and ``index_version``.

    Raises
    ------
    KeyError
        If mandatory fields (e.g., ``fields.created`` or ``fields.status``) 
        are missing.
    ValueError
        If date strings cannot be parsed according to the expected format.
    TypeError
        If fields are of unexpected types.

    Notes
    -----
    - Date parsing uses the format ``"%Y-%m-%dT%H:%M:%S.%f%z"``.
    - If the issue has a ``resolutionDate``, ``time_to_resolution`` is computed
      as ``resolutionDate - created`` and stored as the total number of seconds
      (float). Otherwise, it is set to ``None``.
    - Unresolved tickets therefore omit resolution-dependent metadata.
    - ``index_version`` is currently hard-coded to ``"test"``.
    - ``ingestion_date`` represents the current system time at preprocessing.

    See Also
    --------
    build_jira_ticket_text : Generates the ticket transcript.
    preprocess_jira_tickets : Batch wrapper for multiple issues.
    """
    fields = ticket["fields"]
    fields["summary"]
    fields["status"]
    fields["created"]

    metadata = _build_jira_ticket_metadata(ticket)
    key = ticket.get('key', None)

    text = build_jira_ticket_text(ticket)
    if anonymizer is not None:
        text = anonymizer(text)

    document = Document(
        text=text,
        document_type="helpdesk_ticket",
        id=key,
        metadata=metadata, 
    )

    return document

def preprocess_jira_tickets(tickets: list[dict], anonymizer=None) -> list[Document]:
    """
    Batch-convert Jira issues into ``Document`` objects.

    Parameters
    ----------
    tickets : list[dict]
        Iterable of Jira issue objects compatible with
        :func:`preprocess_jira_ticket`.

    Returns
    -------
    list[Document]
        One processed ``Document`` per input ticket.

    Raises
    ------
    Exception
        Any exception propagated from :func:`preprocess_jira_ticket`.

    See Also
    --------
    preprocess_jira_ticket : Single-issue preprocessing.
    """
    if anonymizer is None:
        preprocessed_tickets = [preprocess_jira_ticket(ticket) for ticket in tickets]
    else:
        preprocessed_tickets = [preprocess_jira_ticket(ticket, anonymizer=anonymizer) for ticket in tickets]

    return preprocessed_tickets
