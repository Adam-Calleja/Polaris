"""Offline authority-registry builder for seeded external official docs.

This module exposes public helper functions used by the surrounding Polaris subsystem.

Functions
---------
get_internal_links
    Return internal Links.
load_website_docs
    Load website Docs.
parse_args
    Parse args.
main
    Run the command-line entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Find Repo Root.
    
    Parameters
    ----------
    start : Path
        Value for start.
    
    Returns
    -------
    Path
        Result of the operation.
    """
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
    SOURCE_SCOPE_EXTERNAL_OFFICIAL,
    build_registry_artifact,
    load_registry_artifact,
    merge_registry_artifacts,
    persist_registry_artifact,
    persist_review_rows,
)
from polaris_rag.authority.source_register import (
    attach_source_register_metadata,
    discover_all_external_source_urls,
    load_external_source_register,
)
from polaris_rag.config import GlobalConfig
from polaris_rag.retrieval.document_preprocessor import preprocess_html_documents
from polaris_rag.retrieval.ingestion_settings import resolve_conversion_settings
from polaris_rag.retrieval.markdown_converter import convert_documents_to_markdown


def get_internal_links(homepage: str) -> list[str]:
    """Return internal Links.
    
    Parameters
    ----------
    homepage : str
        Value for homepage.
    
    Returns
    -------
    list[str]
        Requested internal Links.
    """
    from polaris_rag.retrieval.document_loader import get_internal_links as _get_internal_links

    return _get_internal_links(homepage)


def load_website_docs(links: list[str]):
    """Load website Docs.
    
    Parameters
    ----------
    links : list[str]
        Value for links.
    """
    from polaris_rag.retrieval.document_loader import load_website_docs as _load_website_docs

    return _load_website_docs(links)


def parse_args() -> argparse.Namespace:
    """Parse args.
    
    Returns
    -------
    argparse.Namespace
        Parsed args.
    """
    parser = argparse.ArgumentParser(description="Build a deterministic authority registry from seeded external official docs")
    parser.add_argument(
        "--config-file",
        "-c",
        required=True,
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--source-register-file",
        required=False,
        type=str,
        default="data/authority/source_register.external_v1.yaml",
        help="Path to the external official source register YAML.",
    )
    parser.add_argument(
        "--local-registry-file",
        required=False,
        type=str,
        default="data/authority/registry.local_official.v1.json",
        help="Existing local official registry artifact used when building the combined runtime registry.",
    )
    parser.add_argument(
        "--external-output-file",
        required=False,
        type=str,
        default="data/authority/registry.external_official.v1.json",
        help="Destination JSON artifact path for the external-only registry.",
    )
    parser.add_argument(
        "--external-review-file",
        required=False,
        type=str,
        default="data/authority/review_queue.external_official.v1.csv",
        help="Destination CSV review path for the external-only registry.",
    )
    parser.add_argument(
        "--combined-output-file",
        required=False,
        type=str,
        default="data/authority/registry.official_combined.v1.json",
        help="Destination JSON artifact path for the combined runtime registry.",
    )
    parser.add_argument(
        "--combined-review-file",
        required=False,
        type=str,
        default="data/authority/review_queue.official_combined.v1.csv",
        help="Destination CSV review path for the combined runtime registry.",
    )
    return parser.parse_args()


def _load_external_markdown_documents(cfg: GlobalConfig, register_path: str | Path):
    """Load external Markdown Documents.
    
    Parameters
    ----------
    cfg : GlobalConfig
        Configuration object or mapping used to resolve runtime settings.
    register_path : str or Path
        Filesystem path used by the operation.
    """
    register = load_external_source_register(register_path)
    discovered_urls = discover_all_external_source_urls(register, get_internal_links=get_internal_links)

    conditions = cfg.document_preprocess_html_conditions
    tags = cfg.document_preprocess_html_tags
    conversion_settings = resolve_conversion_settings(cfg, source="external_docs")

    markdown_documents = []
    source_documents: list[RegistrySourceDocument] = []
    for source in register.sources:
        links = discovered_urls[source.source_id]
        if not links:
            continue
        html_documents = load_website_docs(links)
        html_documents = attach_source_register_metadata(html_documents, source=source)
        processed_documents = preprocess_html_documents(
            html_documents,
            tags=tags,
            conditions=conditions,
        )
        markdown_batch = convert_documents_to_markdown(
            processed_documents,
            engine=conversion_settings.engine,
            options=conversion_settings.options,
        )
        markdown_batch = attach_source_register_metadata(markdown_batch, source=source)
        markdown_documents.extend(markdown_batch)
        source_documents.extend(
            RegistrySourceDocument(
                url=str(document.id),
                source_scope=SOURCE_SCOPE_EXTERNAL_OFFICIAL,
                source_id=source.source_id,
            )
            for document in markdown_batch
            if getattr(document, "id", None)
        )
    return register, markdown_documents, source_documents


def main() -> None:
    """Run the command-line entrypoint.
    
    Raises
    ------
    RuntimeError
        If `RuntimeError` is raised while executing the operation.
    """
    args = parse_args()
    cfg = GlobalConfig.load(args.config_file)

    register, markdown_documents, source_documents = _load_external_markdown_documents(
        cfg,
        args.source_register_file,
    )
    if not markdown_documents:
        raise RuntimeError("No external markdown documents were loaded from the source register.")

    external_artifact, external_review_rows = build_registry_artifact(
        markdown_documents,
        homepage=f"source-register:{Path(args.source_register_file).name}",
        source_urls=[item.url for item in source_documents],
        source_scope=SOURCE_SCOPE_EXTERNAL_OFFICIAL,
        source_documents=source_documents,
        build_metadata={
            "source_register_path": str(Path(args.source_register_file).expanduser().resolve()),
            "source_count": len(register.sources),
            "source_ids": [source.source_id for source in register.sources],
            "source_homepages": [source.homepage for source in register.sources],
        },
    )
    external_output_path = persist_registry_artifact(external_artifact, args.external_output_file)
    external_review_path = persist_review_rows(external_review_rows, args.external_review_file)

    local_artifact = load_registry_artifact(args.local_registry_file)
    combined_artifact, combined_review_rows = merge_registry_artifacts(
        [local_artifact, external_artifact],
        build_metadata={
            "source_register_path": str(Path(args.source_register_file).expanduser().resolve()),
            "local_registry_path": str(Path(args.local_registry_file).expanduser().resolve()),
            "external_registry_path": str(Path(args.external_output_file).expanduser().resolve()),
        },
    )
    combined_output_path = persist_registry_artifact(combined_artifact, args.combined_output_file)
    combined_review_path = persist_review_rows(combined_review_rows, args.combined_review_file)

    print(
        "External authority registry complete: "
        f"{external_artifact.summary.get('entity_count', 0)} external entities, "
        f"{combined_artifact.summary.get('entity_count', 0)} combined entities.\n"
        f"External registry: {external_output_path}\n"
        f"External review queue: {external_review_path}\n"
        f"Combined registry: {combined_output_path}\n"
        f"Combined review queue: {combined_review_path}"
    )


if __name__ == "__main__":
    main()
