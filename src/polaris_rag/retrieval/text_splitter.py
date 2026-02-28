"""polaris_rag.retrieval.text_splitter

Text splitting and chunking utilities for the retrieval layer.

This module provides functionality for converting raw
:class:`~polaris_rag.common.schemas.Document` objects into
:class:`~polaris_rag.common.schemas.DocumentChunk` objects suitable for
embedding and retrieval. It includes:
- HTML chunking via a custom hierarchical splitter based on headings
- Jira ticket chunking into token-bounded, semantically coherent pieces

Classes
-------
HTMLHeirarchicalSplitter
    Custom :class:`llama_index.core.node_parser.HTMLNodeParser` that groups
    text under headings and emits linked chunks with preserved metadata.
TicketMessagePart
    Represents a (possibly split) Jira message turn for chunking.
JIRATicketChunker
    Chunk Jira tickets into an initial description chunk and conversation chunks.

Functions
---------
get_chunks_from_documents
    Convert a list of :class:`~polaris_rag.common.schemas.Document` objects to
    :class:`~polaris_rag.common.schemas.DocumentChunk` objects using the
    appropriate splitter for each document type.
get_chunks_from_jira_ticket
    Chunk a single Jira ticket using a provided :class:`~polaris_rag.common.tokenisation.TokenCounter`.
get_chunks_from_jira_tickets
    Chunk multiple Jira tickets using a provided :class:`~polaris_rag.common.tokenisation.TokenCounter`.
"""

from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field
from llama_index.core.node_parser import HTMLNodeParser
from llama_index.core.schema import BaseNode, MetadataMode, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.schema import Document as LlamaIndexDocument
from uuid import uuid4
import re
from dataclasses import dataclass

from polaris_rag.common import Document, DocumentChunk
from polaris_rag.common.tokenisation import TokenCounter

DEFAULT_HEADERS = ["h1", "h2", "h3"]
DEFAULT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "b", "i", "u", "pre"]
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
DEFAULT_UNWANTED_CHARS_PATTERN = r"[¶]"

class HTMLHeirarchicalSplitter(HTMLNodeParser):
    """HTML splitter that groups text under the same heading.

    This splitter parses HTML documents, groups content under headings, and
    creates linked chunks with metadata including heading text/level and any
    extracted external links.

    If a header-derived chunk exceeds the configured character budget, it is
    split across multiple chunks with ``PREVIOUS``/``NEXT`` relationships.

    Attributes
    ----------
    headers : list[str]
        HTML heading tags (e.g., ``["h1", "h2"]``) used to group content.
    tags : list[str]
        HTML tags to extract text from.
    ignore : list[str]
        HTML tags to ignore during parsing.
    overlap : int
        Maximum number of characters to overlap between adjacent chunks.
    chunk_size : int
        Maximum size (in characters) for each chunk.
    unwanted_chars_pattern : str
        Regex pattern specifying characters to remove from extracted text.
    """

    headers: List[str] = Field(
        default=DEFAULT_HEADERS, description="HTML headers to group nodes by."
    )

    tags: List[str] = Field(
        default=DEFAULT_TAGS, description="HTML tags to extract text from."
    )

    ignore: List[str] = Field(
        default=[], description="HTML tags to ignore"
    )

    overlap: int = Field(
        default=DEFAULT_OVERLAP, description="Maximum overlap"
    )

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE, description="Maximum chunk size"
    )

    unwanted_chars_pattern: str = Field(
        default=DEFAULT_UNWANTED_CHARS_PATTERN, description="Unwanted characters in document"
    )

    def get_nodes_from_node(
            self, 
            node: BaseNode,
            link_classes: list[str]) -> List[TextNode]:
        """Extract structured text chunks from a single HTML node.

        Parameters
        ----------
        node : BaseNode
            LlamaIndex node containing raw HTML content and metadata.
        link_classes : list[str]
            CSS classes that identify external links (e.g., ``["reference", "external"]``).

        Returns
        -------
        list[TextNode]
            List of structured nodes with heading metadata and linked relationships.

        Raises
        ------
        ImportError
            If BeautifulSoup (``bs4``) is not installed.
        """
        try:
            from bs4 import BeautifulSoup, Tag
        except ImportError:
            raise ImportError("bs4 is required to read HTML files.")

        text = node.get_content(metadata_mode=MetadataMode.NONE)
        soup = BeautifulSoup(text, "html.parser")
        last_tag = None
        last_header = ""
        last_header_level = ""
        external_links = []
        atomic_nodes = []

        tags = soup.find_all(self.tags)

        for tag in tags:
            if tag.name in self.ignore:
                continue

            tag_links = tag.find_all("a", recursive=False)
            tag_text = self._extract_text_from_tag(tag)
            if isinstance(tag, Tag) and (tag.name == last_tag or last_tag is None):
                if tag.name in self.headers:
                    last_header = tag.text
                    last_header_level = tag.name
                for link in tag_links:
                    if 'class' in link.attrs:
                        if link['class'] == link_classes:
                            external_links.append(link['href'])
            else:
                if isinstance(tag, Tag):
                    last_tag = tag.name
                    if tag.name in self.headers:
                        last_header = tag.text
                        last_header_level = tag.name
                if tag.name == "a" and link['class'] == ['reference', 'external']:
                    external_links.append(tag['href'])

            atomic_nodes.append(
                    self._build_node_from_split(
                        f"{tag_text}\n".strip(), node, {"header_level": last_header_level, "header": last_header, "external_links": external_links}
                    )
                )
            last_tag = tag.name
            
            external_links = []

        first_node = TextNode(id_=str(uuid4()), metadata=(node.metadata | {'header_level': "", "header": "", "external_links": []}))
        first_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=node.id_
        )

        html_chunks = [first_node]
        previous_header = ""
        previous_header_level = ""

        for atom in atomic_nodes:
            chunk = html_chunks[-1]
            chunk_text = chunk.text
            chunk_id = chunk.id_
            chunk_external_links = chunk.metadata['external_links']

            atom_header = atom.metadata['header']
            atom_header_level = atom.metadata['header_level']
            atom_external_links = atom.metadata['external_links']
            atom_text = atom.text
            new_text = chunk_text + atom_text
            parent_id = node.id_

            if len(atom_text) == 0:
                continue

            text_too_long = (len(new_text) > (self.chunk_size - self.overlap) and len(html_chunks) > 1) or (len(new_text) > (self.chunk_size) and len(html_chunks) == 1)
    
            if (previous_header != "" and (atom_header != previous_header or atom_header_level != previous_header_level)) or text_too_long:
                previous_header = atom_header
                previous_header_level = atom_header_level

                external_links = chunk_external_links + atom_external_links
                new_node = TextNode(
                    text = atom_text,
                    id_ = str(uuid4()),
                    metadata=node.metadata | {'header_level': atom_header_level, "header": atom_header, "external_links": external_links}
                )

                new_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=parent_id
                )
                new_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=chunk_id
                )

                chunk.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=new_node.id_
                )

                html_chunks[-1] = chunk
                html_chunks.append(new_node)
            else:
                chunk.text = new_text
                
                external_links = chunk_external_links + atom_external_links
                chunk.metadata['external_links'] = external_links

                html_chunks[-1] = chunk
        
        return html_chunks
    
    def get_nodes_from_documents(self, documents, link_classes, show_progress = False, **kwargs):
        """Convert a list of HTML nodes into text chunks.

        Parameters
        ----------
        documents : list[BaseNode]
            HTML documents in LlamaIndex node format.
        link_classes : list[str]
            CSS classes that identify external links.
        show_progress : bool, optional
            Whether to show a progress bar. Defaults to ``False``.
        **kwargs : Any
            Additional keyword arguments (currently ignored).

        Returns
        -------
        list[TextNode]
            Parsed and structured nodes across all input documents.
        """
        all_chunks = []

        for document in documents:
            chunks = self.get_nodes_from_node(document, link_classes=link_classes)
            all_chunks.extend(chunks)

        return all_chunks
    
    def _extract_text_from_tag(
        self, tag: Any
    ) -> str:
        """Extract visible text from a BeautifulSoup tag.

        Parameters
        ----------
        tag : Any
            BeautifulSoup element representing an HTML tag or text node.

        Returns
        -------
        str
            Extracted and cleaned text from the element.
        """
        from bs4 import NavigableString, Tag

        compiled_unwanted_chars = re.compile(self.unwanted_chars_pattern)

        texts = []
        if isinstance(tag, Tag):
            for elem in tag.children:
                if isinstance(elem, NavigableString):
                    if elem.strip():
                        texts.append(elem.strip())
                # elif isinstance(elem, Tag):
                #     if elem.name in self.tags or elem.name == 'a':
                #         texts.append(elem.get_text().strip())
                # elif isinstance(elem, PageElement):
                else:
                    texts.append(elem.get_text().strip())
        else:
            texts.append(tag.get_text().strip())

        if tag.name in ["p", "li"]:
            return compiled_unwanted_chars.sub("", " ".join(texts))
        return compiled_unwanted_chars.sub("", "\n".join(texts))
    

def get_chunks_from_documents(
        documents: list[Document],
        *,
        chunk_size: Optional[int] = None, 
        overlap: Optional[int] = None, 
        ignore: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        headers: Optional[list[str]] = None,
        link_classes: Optional[list[str]] = None
) -> list[DocumentChunk]:
    """Split documents into chunks suitable for retrieval.

    The splitter used depends on ``document.document_type``. Currently, HTML
    documents are chunked using :class:`HTMLHeirarchicalSplitter`.

    Parameters
    ----------
    documents : list[Document]
        Documents to split.
    chunk_size : int, optional
        Maximum size (in characters) of each HTML chunk.
    overlap : int, optional
        Overlap (in characters) between adjacent HTML chunks.
    ignore : list[str], optional
        HTML tags to ignore when extracting content.
    tags : list[str], optional
        HTML tags to extract text from.
    headers : list[str], optional
        Heading tags to group content under.
    link_classes : list[str] or None, optional
        CSS classes identifying external links. If ``None`` (default), no external
        link classes are matched.

    Returns
    -------
    list[DocumentChunk]
        Structured document chunks with navigation metadata.

    Raises
    ------
    ValueError
        If a document type is not supported.
    """
    if link_classes is None:
        link_classes = []
    splitter_kwargs = {
        k: v for k, v in {
            "chunk_size": chunk_size,
            "overlap":    overlap,
            "ignore":     ignore,
            "tags":       tags,
            "headers":    headers,
        }.items() if v is not None
    }

    chunks = []

    for document in documents:
        document_type = document.document_type

        llama_document = LlamaIndexDocument(
                text = document.text,
                id_ = document.id,
                metadata = document.metadata
            )

        if document_type == 'html':
            chunker = HTMLHeirarchicalSplitter(**splitter_kwargs)
        else:
            raise ValueError(f"Unsupported document_type: {document_type!r}")
        
        llama_chunks = chunker.get_nodes_from_node(llama_document, link_classes=link_classes)

        for llama_chunk in llama_chunks:
            parent_id = llama_chunk.source_node.node_id
            if llama_chunk.next_node:
                next_id = llama_chunk.next_node.node_id
            else:
                next_id = ""
            if llama_chunk.prev_node:
                prev_id = llama_chunk.prev_node.node_id
            else:
                prev_id = ""

            chunk = DocumentChunk(
                parent_id=parent_id,
                prev_id=prev_id,
                next_id=next_id,
                text=llama_chunk.text,
                document_type=document_type,
                metadata=llama_chunk.metadata,
                source_node=document
            )

            chunks.append(chunk)

    return chunks

@dataclass
class TicketMessagePart:
    """Represents a (possibly split) Jira message turn for chunking.

    Attributes
    ----------
    original_id : str
        Original message identifier from the ticket transcript.
    role : str
        Role label for the message (e.g., ``"USER"``, ``"HELPDESK_ASSIGNEE"``).
    text : str
        Message body text for this part.
    part_index : int
        1-based index of this part within the original message.
    part_total : int
        Total number of parts for the original message.
    """

    original_id: str
    role: str
    text: str
    part_index: int
    part_total: int

    def format_for_chunk(self) -> str:
        """Render this message part as a ``<MESSAGE>...</MESSAGE>`` block.

        Returns
        -------
        str
            Formatted message block including attributes needed for traceability.
        """
        if self.part_total == 1:
            header = f"<MESSAGE id={self.original_id} role={self.role}>\n"
            footer = "\n</MESSAGE>"
            return f"{header}{self.text}\n{footer}"

        suffix = chr(ord("a") + self.part_index - 1)
        part_id = f"{self.original_id}{suffix}"
        header = (
            f"<MESSAGE id={part_id} role={self.role} "
            f"part={self.part_index}/{self.part_total} original_id={self.original_id}>\n"
        )
        footer = "\n</MESSAGE>"
        return f"{header}{self.text}\n{footer}"

class JIRATicketChunker:
    """Chunk Jira tickets into token-bounded, semantically coherent pieces.

    The chunker produces:
    - an initial chunk containing the ticket summary (optional) and initial description
    - one or more conversation chunks containing ``<MESSAGE>`` blocks, optionally
      with overlap context from the previous chunk

    Oversized single messages are split into paragraph-bounded parts.

    Parameters
    ----------
    token_counter : TokenCounter
        Token counter used to measure and bound chunk size.
    chunk_size_tokens : int, optional
        Maximum token budget per chunk. Defaults to ``800``.
    overlap_tokens : int, optional
        Size of overlap context (in tokens) from the previous chunk. Defaults to ``80``.
    max_turns_per_chunk : int or None, optional
        Maximum distinct message turns per conversation chunk. If ``None``, no limit
        is applied. Defaults to ``3``.
    include_summary : bool, optional
        Whether to include the ticket summary in chunks. Defaults to ``True``.
    include_context : bool, optional
        Whether to include overlap context in conversation chunks. Defaults to ``True``.
    """

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        chunk_size_tokens: int = 800,
        overlap_tokens: int = 80,
        max_turns_per_chunk: Optional[int] = 3,
        include_summary: bool = True,
        include_context: bool = True,
    ) -> None:
        """Initialise a JIRATicketChunker.

        Parameters
        ----------
        token_counter : TokenCounter
            Token counter used to measure and bound chunk size.
        chunk_size_tokens : int, optional
            Maximum token budget per chunk. Defaults to ``800``.
        overlap_tokens : int, optional
            Overlap context size in tokens. Defaults to ``80``.
        max_turns_per_chunk : int or None, optional
            Maximum distinct turns per chunk. Defaults to ``3``.
        include_summary : bool, optional
            Whether to include the ticket summary. Defaults to ``True``.
        include_context : bool, optional
            Whether to include overlap context. Defaults to ``True``.
        """
        self.token_counter = token_counter
        self.chunk_size_tokens = int(chunk_size_tokens)
        self.overlap_tokens = max(0, int(overlap_tokens))
        self.max_turns_per_chunk = max_turns_per_chunk
        self.include_summary = include_summary
        self.include_context = include_context

    def _num_tokens(self, text: str) -> int:
        """Return the token count for ``text``."""
        return int(self.token_counter.count(text))

    def _last_n_tokens_text(self, text: str, n: int) -> str:
        """Return the last ``n`` tokens of ``text`` as text."""
        return self.token_counter.tail(text, n)

    def _extract_initial_description(self, ticket_text: str) -> str:
        """Extract the initial description block from a ticket transcript."""
        pattern = r"\[INITIAL_DESCRIPTION\]\s*(.*?)\s*\[CONVERSATION\]"
        match = re.search(pattern, ticket_text, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    def _parse_messages(self, ticket_text: str) -> list[tuple[str, str, str]]:
        """Parse ``<MESSAGE>`` blocks from a ticket transcript."""
        pattern = r"<MESSAGE\s+id=(\d+)\s+role=([A-Z_]+)>\s*(.*?)\s*</MESSAGE>"
        return re.findall(pattern, ticket_text, flags=re.DOTALL)

    def _build_initial_chunk(self, ticket: Document) -> DocumentChunk:
        """Build the initial summary/description chunk for a ticket."""
        initial_description_text = self._extract_initial_description(ticket.text)
        summary_line = f"[TICKET_SUMMARY] {ticket.metadata['summary']}\n\n" if self.include_summary else ""
        initial_chunk_text = (
            f"{summary_line}"
            f"[INITIAL_DESCRIPTION]\n"
            f"{initial_description_text}"
        )
        metadata = {
            "chunk_type": "initial",
            "chunk_index": 0,
            "turn_range": None,
            "speakers": ["TICKET_CREATOR"],
            "overlap": {"has_overlap": False, "size": 0},
        }
        return DocumentChunk(
            parent_id=ticket.id,
            prev_id=None,
            next_id=None,
            text=initial_chunk_text,
            document_type=ticket.metadata['document_type'],
            id=str(uuid4()),
            metadata=metadata,
            source_node=ticket,
        )

    def _build_prefix(self, ticket: Document, prev_conv_body: str) -> str:
        """Build the conversation prefix, including optional summary/context."""
        parts: List[str] = []
        if self.include_summary:
            parts.append(f"[TICKET_SUMMARY] {ticket.metadata['summary']}\n\n")

        if self.include_context and prev_conv_body and self.overlap_tokens > 0:
            matches = re.findall(
                r"<MESSAGE\b[^>]*>(.*?)</MESSAGE>",
                prev_conv_body,
                flags=re.DOTALL,
            )
            last_text = matches[-1].strip() if matches else prev_conv_body
            context = self._last_n_tokens_text(last_text, self.overlap_tokens)
            parts.append("[CONTEXT]\n")
            parts.append(context)
            parts.append("\n\n")

        parts.append("[CONVERSATION]\n")
        return "".join(parts)
    
    def _split_message_into_parts(
        self,
        turn: str,
        role: str,
        msg_text: str,
        base_prefix: str,
    ) -> list["TicketMessagePart"]:
        """Split a Jira message into parts that fit under the token budget.

        Parameters
        ----------
        turn : str
            Original message identifier.
        role : str
            Role label for the message.
        msg_text : str
            Message body text.
        base_prefix : str
            Prefix text included before the message in a conversation chunk.

        Returns
        -------
        list[TicketMessagePart]
            One or more message parts whose formatted representation can be packed
            into chunks under ``chunk_size_tokens``.

        Raises
        ------
        ValueError
            If ``chunk_size_tokens`` is too small to accommodate required wrappers.
        """
        msg_text = msg_text.strip()
        if not msg_text:
            return [
                TicketMessagePart(
                    original_id=turn,
                    role=role,
                    text="",
                    part_index=1,
                    part_total=1,
                )
            ]
        
        full_msg = f"<MESSAGE id={turn} role={role}>\n{msg_text}\n</MESSAGE>"
        if self._num_tokens(base_prefix + full_msg) <= self.chunk_size_tokens:
            return [
                TicketMessagePart(
                    original_id=turn,
                    role=role,
                    text=msg_text,
                    part_index=1,
                    part_total=1,
                )
            ]

        max_header = (
            "<MESSAGE id=000000 role=HELPDESK_ASSIGNEE "
            "part=100/100 original_id=000000>\n"
        )
        minimal_footer = "\n</MESSAGE>"
        overhead_tokens = self._num_tokens(base_prefix + max_header + minimal_footer)
        body_budget = self.chunk_size_tokens - overhead_tokens
        if body_budget <= 0:
            raise ValueError(
                "chunk_size_tokens is too small to accommodate even an empty "
                "message body once prefix and headers are included."
            )

        paragraphs = [p.strip() for p in msg_text.split("\n\n") if p.strip()]
        parts_text: list[str] = []
        current: str = ""

        def _fits_body_budget(text: str) -> bool:
            return self._num_tokens(text) <= body_budget

        for para in paragraphs:
            # if not _fits_body_budget(para):
            #     raise ValueError(
            #         "A single paragraph in message id={turn} is too large to fit "
            #         "into a chunk under the current chunk_size_tokens. "
            #         "Increase chunk_size_tokens or pre-truncate the input."
            #     )

            if not current:
                current = para
            else:
                candidate = current + "\n\n" + para
                if _fits_body_budget(candidate):
                    current = candidate
                else:
                    parts_text.append(current.strip())
                    current = para

        if current:
            parts_text.append(current.strip())

        if not parts_text:
            parts_text = [msg_text]

        total = len(parts_text)
        result: list[TicketMessagePart] = []
        for idx, text in enumerate(parts_text, start=1):
            result.append(
                TicketMessagePart(
                    original_id=turn,
                    role=role,
                    text=text,
                    part_index=idx,
                    part_total=total,
                )
            )
        return result

    def chunk(self, ticket: Document) -> list[DocumentChunk]:
        """Chunk a Jira ticket into an initial chunk and conversation chunks.

        Parameters
        ----------
        ticket : Document
            Jira ticket document whose ``text`` contains an ``[INITIAL_DESCRIPTION]``
            section and ``<MESSAGE>`` blocks.

        Returns
        -------
        list[DocumentChunk]
            Chunks in traversal order. Each chunk includes metadata describing its
            type (initial vs conversation), turn ranges, speakers, and overlap.

        Raises
        ------
        ValueError
            If the configured token budget cannot accommodate the required prefixes
            and wrappers for conversation chunks.
        """
        chunks: list[DocumentChunk] = []

        initial_chunk = self._build_initial_chunk(ticket)
        chunks.append(initial_chunk)

        base_prefix = self._build_prefix(ticket, prev_conv_body="")
        minimal_header = "<MESSAGE id=0 role=USER>\n"
        minimal_footer = "\n</MESSAGE>"
        overhead_tokens = self._num_tokens(base_prefix + minimal_header + minimal_footer)
        if overhead_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "chunk_size_tokens={cs} is too small for the conversation prefix and "
                "minimal message wrapper (requires at least {needed} tokens).".format(
                    cs=self.chunk_size_tokens, needed=overhead_tokens + 1
                )
            )

        messages = self._parse_messages(ticket.text)

        parts: list[TicketMessagePart] = []
        for turn, role, msg_text in messages:
            parts.extend(self._split_message_into_parts(turn, role, msg_text, base_prefix))

        if not parts:
            return chunks

        prev_conv_body = ""
        current_body = ""
        current_turns: list[str] = []
        current_speakers: list[str] = []
        current_prefix: Optional[str] = None

        def finalize_current_chunk() -> None:
            nonlocal prev_conv_body, current_body, current_turns, current_speakers, current_prefix
            if not current_body:
                return
            prefix = current_prefix if current_prefix is not None else self._build_prefix(ticket, prev_conv_body)
            chunk_text = prefix + current_body
            has_prev = bool(prev_conv_body)
            metadata = {
                "chunk_type": "conversation",
                "chunk_index": len(chunks),
                "turn_range": f"{current_turns[0]}-{current_turns[-1]}" if current_turns else None,
                "speakers": current_speakers[:],
                "overlap": {
                    "has_overlap": has_prev and self.include_context and self.overlap_tokens > 0,
                    "size": self.overlap_tokens if has_prev else 0,
                },
            }
            chunks.append(
                DocumentChunk(
                    parent_id=ticket.id,
                    prev_id=chunks[-1].id,
                    next_id=None,
                    text=chunk_text,
                    document_type=ticket.metadata["document_type"],
                    id=str(uuid4()),
                    metadata=metadata,
                    source_node=ticket,
                )
            )
            prev_conv_body = current_body
            current_body = ""
            current_turns = []
            current_speakers = []
            current_prefix = None

        for part in parts:
            formatted = part.format_for_chunk()
            while True:
                if current_prefix is None:
                    current_prefix = self._build_prefix(ticket, prev_conv_body)

                if self.max_turns_per_chunk is not None and current_turns:
                    distinct_turns = len(set(current_turns))
                    if distinct_turns >= self.max_turns_per_chunk and part.original_id not in current_turns:
                        finalize_current_chunk()
                        continue

                candidate_text = current_prefix + current_body + formatted
                if self._num_tokens(candidate_text) <= self.chunk_size_tokens:
                    current_body += formatted
                    if not current_turns or current_turns[-1] != part.original_id:
                        current_turns.append(part.original_id)
                    current_speakers.append(part.role)
                    break

                if current_body:
                    finalize_current_chunk()
                    continue

                current_body += formatted
                if not current_turns or current_turns[-1] != part.original_id:
                    current_turns.append(part.original_id)
                current_speakers.append(part.role)
                break

                # raise ValueError(
                #     "A single message part is too large to fit into an empty chunk; "
                #     "consider increasing chunk_size_tokens."
                # )

        if current_body:
            finalize_current_chunk()

        for i in range(len(chunks) - 1):
            chunks[i].next_id = chunks[i + 1].id

        return chunks


def get_chunks_from_jira_ticket(
        ticket: Document,
        *,
        token_counter: TokenCounter,
        chunk_size: int = 800,
        overlap: int = 80,
    ) -> list[DocumentChunk]:
    """Chunk a single Jira ticket using an explicit token counter.

    Parameters
    ----------
    ticket : Document
        Jira ticket document to chunk.
    token_counter : TokenCounter
        Token counter used to measure and bound chunk size.
    chunk_size : int, optional
        Maximum token budget per chunk. Defaults to ``800``.
    overlap : int, optional
        Overlap context size in tokens. Defaults to ``80``.

    Returns
    -------
    list[DocumentChunk]
        Chunked ticket.
    """
    chunker = JIRATicketChunker(
        token_counter=token_counter,
        chunk_size_tokens=chunk_size,
        overlap_tokens=overlap,
    )
    return chunker.chunk(ticket)

def get_chunks_from_jira_tickets(
        tickets: list[Document],
        *,
        token_counter: TokenCounter,
        chunk_size: int = 800,
        overlap: int = 80,
    ) -> list[DocumentChunk]:
    """Chunk multiple Jira tickets using an explicit token counter.

    Parameters
    ----------
    tickets : list[Document]
        Jira ticket documents to chunk.
    token_counter : TokenCounter
        Token counter used to measure and bound chunk size.
    chunk_size : int, optional
        Maximum token budget per chunk. Defaults to ``800``.
    overlap : int, optional
        Overlap context size in tokens. Defaults to ``80``.

    Returns
    -------
    list[DocumentChunk]
        All chunks across all tickets, in input order.
    """
    chunker = JIRATicketChunker(
        token_counter=token_counter,
        chunk_size_tokens=chunk_size,
        overlap_tokens=overlap,
    )

    all_chunks: list[DocumentChunk] = []
    total_tickets = len(tickets)
    for i, ticket in enumerate(tickets, start=1):
        print(f"Chunking ticket {ticket.id} ({i}/{total_tickets})...")
        all_chunks.extend(chunker.chunk(ticket))

    return all_chunks
