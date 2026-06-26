"""Shared helpers for recursive internal-link crawling."""

from __future__ import annotations

from collections import deque
from typing import Callable, Sequence
from urllib.parse import urlsplit

from polaris_rag.retrieval.document_loader import canonicalize_url

MAX_INTERNAL_LINKS = 512
STATIC_ASSET_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".pdf",
    ".ico",
    ".css",
    ".js",
    ".xml",
    ".txt",
)


def is_allowed_docs_subtree_url(homepage: str, url: str) -> bool:
    """Return whether a URL is allowed within a docs subtree crawl."""

    normalized_homepage = canonicalize_url(homepage)
    normalized_url = canonicalize_url(url)

    homepage_parts = urlsplit(normalized_homepage)
    candidate_parts = urlsplit(normalized_url)
    if not candidate_parts.scheme or not candidate_parts.netloc:
        return False
    if candidate_parts.scheme != homepage_parts.scheme or candidate_parts.netloc != homepage_parts.netloc:
        return False

    homepage_path = homepage_parts.path or "/"
    allowed_prefix = homepage_path.rsplit("/", 1)[0] + "/"
    candidate_path = candidate_parts.path or "/"
    candidate_path_lower = candidate_path.lower()

    if not candidate_path.startswith(allowed_prefix):
        return False
    if "/_sources/" in candidate_path_lower or "/_images/" in candidate_path_lower or "/images/" in candidate_path_lower:
        return False
    if "/storage/" in candidate_path_lower:
        return False
    if candidate_path_lower.endswith(STATIC_ASSET_EXTENSIONS):
        return False
    return True


def crawl_internal_links(
    homepage: str,
    *,
    get_internal_links_one_hop: Callable[[str], Sequence[str]],
    is_allowed_url: Callable[[str, str], bool],
    max_depth: int | None = None,
    max_links: int = MAX_INTERNAL_LINKS,
) -> list[str]:
    """Recursively crawl internal links from a homepage with bounded BFS."""

    normalized_homepage = canonicalize_url(homepage)
    queue = deque([(normalized_homepage, 0)])
    queued: set[str] = {normalized_homepage}
    seen: set[str] = set()
    ordered: list[str] = []

    while queue and len(seen) < max_links:
        current, depth = queue.popleft()
        queued.discard(current)
        if current in seen:
            continue
        seen.add(current)

        discovered = list(get_internal_links_one_hop(current) or [])
        if not discovered:
            discovered = [current]

        for link in discovered:
            normalized_link = canonicalize_url(link)
            if not is_allowed_url(normalized_homepage, normalized_link):
                continue
            if normalized_link not in ordered:
                ordered.append(normalized_link)
            if max_depth is not None and depth + 1 > max_depth:
                continue
            if normalized_link in seen or normalized_link in queued:
                continue
            if len(seen) + len(queue) >= max_links:
                break
            queue.append((normalized_link, depth + 1))
            queued.add(normalized_link)

    if normalized_homepage not in ordered and is_allowed_url(normalized_homepage, normalized_homepage):
        ordered.insert(0, normalized_homepage)
    return ordered
