"""Offline authority-registry builder for local official docs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

MAX_INTERNAL_LINKS = 512
_STATIC_ASSET_EXTENSIONS = (
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


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return Path.cwd()


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.authority import (
    RegistrySourceDocument,
    SOURCE_SCOPE_LOCAL_OFFICIAL,
    SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES,
    build_registry_artifact,
    persist_registry_artifact,
    persist_review_rows,
)
from polaris_rag.authority.service_catalog import (
    SERVICE_CATALOG_DEFAULT_HOMEPAGE,
    SERVICE_CATALOG_MAX_DEPTH,
    extract_service_catalog_candidates,
    is_allowed_service_catalog_url,
)
from polaris_rag.config import GlobalConfig
from polaris_rag.retrieval.document_preprocessor import preprocess_html_documents
from polaris_rag.retrieval.ingestion_settings import resolve_conversion_settings
from polaris_rag.retrieval.markdown_converter import convert_documents_to_markdown


def _get_internal_links_one_hop(homepage: str) -> list[str]:
    from polaris_rag.retrieval.document_loader import get_internal_links as _get_internal_links

    return _get_internal_links(homepage)


def _is_allowed_authority_url(homepage: str, url: str) -> bool:
    homepage_parts = urlsplit(homepage)
    candidate_parts = urlsplit(url)
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
    if candidate_path_lower.endswith(_STATIC_ASSET_EXTENSIONS):
        return False
    return True


def _crawl_internal_links(
    homepage: str,
    *,
    is_allowed_url,
    max_depth: int | None = None,
) -> list[str]:
    queue: list[tuple[str, int]] = [(homepage, 0)]
    seen: set[str] = set()
    ordered: list[str] = []

    while queue and len(seen) < MAX_INTERNAL_LINKS:
        current, depth = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)

        discovered = _get_internal_links_one_hop(current)
        if not discovered:
            discovered = [current]

        for link in discovered:
            if not is_allowed_url(homepage, link):
                continue
            if link not in ordered:
                ordered.append(link)
            if max_depth is not None and depth + 1 > max_depth:
                continue
            if link not in seen and link not in [candidate for candidate, _ in queue] and len(seen) + len(queue) < MAX_INTERNAL_LINKS:
                queue.append((link, depth + 1))

    if homepage not in ordered and is_allowed_url(homepage, homepage):
        ordered.insert(0, homepage)
    return ordered


def get_internal_links(homepage: str) -> list[str]:
    return _crawl_internal_links(
        homepage,
        is_allowed_url=_is_allowed_authority_url,
    )


def get_service_catalog_links(homepage: str) -> list[str]:
    return _crawl_internal_links(
        homepage,
        is_allowed_url=is_allowed_service_catalog_url,
        max_depth=SERVICE_CATALOG_MAX_DEPTH,
    )


def load_website_docs(links: list[str]):
    from polaris_rag.retrieval.document_loader import load_website_docs as _load_website_docs

    return _load_website_docs(links)


def _load_markdown_documents(cfg: GlobalConfig, links: list[str]) -> list:
    conditions = cfg.document_preprocess_html_conditions
    tags = cfg.document_preprocess_html_tags
    documents = load_website_docs(links)
    processed_documents = preprocess_html_documents(
        documents,
        tags=tags,
        conditions=conditions,
    )
    conversion_settings = resolve_conversion_settings(cfg, source="docs")
    return convert_documents_to_markdown(
        processed_documents,
        engine=conversion_settings.engine,
        options=conversion_settings.options,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic authority registry from local official docs")
    parser.add_argument(
        "--config-file",
        "-c",
        required=True,
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--homepage",
        "-p",
        required=True,
        type=str,
        help="Homepage URL for the local official docs set.",
    )
    parser.add_argument(
        "--ingest-internal-links",
        "-i",
        action="store_true",
        help="If set, discover and include internal links from the homepage.",
    )
    parser.add_argument(
        "--output-file",
        required=False,
        type=str,
        default="data/authority/registry.local_official.v1.json",
        help="Destination JSON artifact path.",
    )
    parser.add_argument(
        "--review-file",
        required=False,
        type=str,
        default="data/authority/review_queue.local_official.v1.csv",
        help="Destination CSV review-queue path.",
    )
    parser.add_argument(
        "--source-scope",
        required=False,
        choices=[SOURCE_SCOPE_LOCAL_OFFICIAL],
        default=SOURCE_SCOPE_LOCAL_OFFICIAL,
        help="Authority source scope for this build. Stage 1 only supports local_official.",
    )
    parser.add_argument(
        "--services-homepage",
        required=False,
        type=str,
        default=SERVICE_CATALOG_DEFAULT_HOMEPAGE,
        help="Homepage URL for the official RCS services catalog.",
    )
    parser.add_argument(
        "--skip-services-catalog",
        action="store_true",
        help="If set, skip crawling and merging the official RCS services catalog.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GlobalConfig.load(args.config_file)

    discovered_links = get_internal_links(args.homepage) if args.ingest_internal_links else [args.homepage]
    links = [link for link in discovered_links if _is_allowed_authority_url(args.homepage, link)]

    service_links: list[str] = []
    service_candidates = []
    service_review_rows = []
    if not args.skip_services_catalog:
        discovered_service_links = (
            get_service_catalog_links(args.services_homepage)
            if args.ingest_internal_links
            else [args.services_homepage]
        )
        service_links = [
            link
            for link in discovered_service_links
            if is_allowed_service_catalog_url(args.services_homepage, link)
        ]

    all_link_count = len(links) + len(service_links)
    print(f"Loading authority source pages: {all_link_count} URL(s)")

    markdown_documents = _load_markdown_documents(cfg, links)
    if service_links:
        service_markdown_documents = _load_markdown_documents(cfg, service_links)
        service_candidates, service_review_rows = extract_service_catalog_candidates(service_markdown_documents)

    artifact, review_rows = build_registry_artifact(
        markdown_documents,
        homepage=args.homepage,
        source_urls=links,
        source_scope=args.source_scope,
        additional_candidates=service_candidates,
        additional_review_rows=service_review_rows,
        additional_source_urls=service_links,
        source_documents=[
            *[
                RegistrySourceDocument(url=link, source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL, source_id="local_docs")
                for link in links
            ],
            *[
                RegistrySourceDocument(
                    url=link,
                    source_scope=SOURCE_SCOPE_LOCAL_OFFICIAL_SERVICES,
                    source_id="local_services_catalog",
                )
                for link in service_links
            ],
        ],
        build_metadata={
            "docs_homepage": str(args.homepage),
            "services_homepage": str(args.services_homepage),
            "service_catalog_included": not args.skip_services_catalog,
        },
    )
    output_path = persist_registry_artifact(artifact, args.output_file)
    review_path = persist_review_rows(review_rows, args.review_file)

    print(
        f"Authority registry complete: {len(artifact.entities)} entities, "
        f"{len(review_rows)} review row(s)."
    )
    print(f"Registry artifact: {output_path}")
    print(f"Review queue: {review_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
