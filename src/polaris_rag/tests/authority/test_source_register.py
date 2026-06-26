from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polaris_rag.authority.source_register import (
    EXTERNAL_SOURCE_REGISTER_VERSION,
    discover_external_source_urls,
    load_external_source_register,
)


def _write_register(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "source_register.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_external_source_register_rejects_overlapping_prefixes(tmp_path: Path) -> None:
    path = _write_register(
        tmp_path,
        {
            "version": EXTERNAL_SOURCE_REGISTER_VERSION,
            "sources": [
                {
                    "source_id": "one",
                    "canonical_name": "One",
                    "entity_type": "software",
                    "homepage": "https://docs.example.org/product/one/index.html",
                    "allowed_domains": ["docs.example.org"],
                    "include_url_prefixes": ["https://docs.example.org/product/"],
                    "exclude_url_patterns": [],
                    "aliases": ["one"],
                    "relevance_tags": ["software:one"],
                    "crawl": {"max_depth": 1, "max_pages": 10},
                },
                {
                    "source_id": "two",
                    "canonical_name": "Two",
                    "entity_type": "software",
                    "homepage": "https://docs.example.org/product/two/index.html",
                    "allowed_domains": ["docs.example.org"],
                    "include_url_prefixes": ["https://docs.example.org/product/two/"],
                    "exclude_url_patterns": [],
                    "aliases": ["two"],
                    "relevance_tags": ["software:two"],
                    "crawl": {"max_depth": 1, "max_pages": 10},
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="overlapping include_url_prefixes"):
        load_external_source_register(path)


def test_discover_external_source_urls_stays_within_allowed_prefixes_and_exclusions(tmp_path: Path) -> None:
    path = _write_register(
        tmp_path,
        {
            "version": EXTERNAL_SOURCE_REGISTER_VERSION,
            "sources": [
                {
                    "source_id": "gromacs",
                    "canonical_name": "GROMACS",
                    "entity_type": "software",
                    "homepage": "https://docs.example.org/gromacs/index.html",
                    "allowed_domains": ["docs.example.org"],
                    "include_url_prefixes": ["https://docs.example.org/gromacs/"],
                    "exclude_url_patterns": ["/release-notes/"],
                    "aliases": ["gromacs"],
                    "relevance_tags": ["software:gromacs"],
                    "crawl": {"max_depth": 2, "max_pages": 10},
                }
            ],
        },
    )
    register = load_external_source_register(path)

    graph = {
        "https://docs.example.org/gromacs/index.html": [
            "https://docs.example.org/gromacs/index.html",
            "https://docs.example.org/gromacs/install.html",
            "https://docs.example.org/gromacs/release-notes/latest.html",
            "https://community.example.org/gromacs/thread.html",
        ],
        "https://docs.example.org/gromacs/install.html": [
            "https://docs.example.org/gromacs/install.html",
            "https://docs.example.org/gromacs/reference.html",
        ],
        "https://docs.example.org/gromacs/reference.html": [
            "https://docs.example.org/gromacs/reference.html",
        ],
    }

    urls = discover_external_source_urls(
        register.sources[0],
        get_internal_links=lambda url: graph.get(url, []),
    )

    assert urls == [
        "https://docs.example.org/gromacs/index.html",
        "https://docs.example.org/gromacs/install.html",
        "https://docs.example.org/gromacs/reference.html",
    ]
