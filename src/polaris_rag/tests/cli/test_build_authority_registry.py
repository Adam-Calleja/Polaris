from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import types

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.modules.setdefault("atlassian", types.SimpleNamespace(Jira=object))

from polaris_rag.cli import build_authority_registry
from polaris_rag.common import Document, MarkdownDocument


def test_parse_args_supports_stage_one_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_authority_registry.py",
            "-c",
            "config/config.yaml",
            "-p",
            "https://docs.example.org/hpc/index.html",
            "-i",
            "--output-file",
            "data/authority/registry.json",
            "--review-file",
            "data/authority/review.csv",
            "--source-scope",
            "local_official",
            "--services-homepage",
            "https://www.example.org/services",
            "--skip-services-catalog",
        ],
    )

    args = build_authority_registry.parse_args()

    assert args.config_file == "config/config.yaml"
    assert args.homepage == "https://docs.example.org/hpc/index.html"
    assert args.ingest_internal_links is True
    assert args.output_file == "data/authority/registry.json"
    assert args.review_file == "data/authority/review.csv"
    assert args.source_scope == "local_official"
    assert args.services_homepage == "https://www.example.org/services"
    assert args.skip_services_catalog is True


def test_main_writes_registry_and_review_outputs(tmp_path, monkeypatch, capsys):
    fake_cfg = SimpleNamespace(
        ingestion={"conversion": {"sources": {"docs": {"engine": "markitdown"}}}},
        document_preprocess_html_conditions=[],
        document_preprocess_html_tags=[],
    )
    html_documents = [
        Document(
            id="https://docs.example.org/hpc/software-packages/tensorflow",
            document_type="html",
            text="<html><head><title>TensorFlow - CSD3 1.0 documentation</title></head><body><h1>TensorFlow</h1></body></html>",
            metadata={"source": "https://docs.example.org/hpc/software-packages/tensorflow"},
        )
    ]
    service_html_documents = [
        Document(
            id="https://www.example.org/secure-research-computing",
            document_type="html",
            text="<html><head><title>Secure Research Computing</title></head><body><h1>Secure Research Computing</h1></body></html>",
            metadata={"source": "https://www.example.org/secure-research-computing"},
        )
    ]
    markdown_documents = [
        MarkdownDocument(
            id="https://docs.example.org/hpc/software-packages/tensorflow",
            document_type="html",
            text="# TensorFlow\n\nSupported version 2.16.1.\n\n```bash\nmodule load TensorFlow/2.16.1-foss-2023a-CUDA-12.1.1\n```",
            metadata={
                "source": "https://docs.example.org/hpc/software-packages/tensorflow",
                "title": "TensorFlow - CSD3 1.0 documentation",
            },
        )
    ]
    service_markdown_documents = [
        MarkdownDocument(
            id="https://www.example.org/secure-research-computing",
            document_type="html",
            text="# Secure Research Computing\n\nSecure Research Computing provides secure services.",
            metadata={
                "source": "https://www.example.org/secure-research-computing",
                "title": "Secure Research Computing",
            },
        )
    ]
    output_file = tmp_path / "registry.json"
    review_file = tmp_path / "review.csv"

    monkeypatch.setattr(build_authority_registry.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(
        build_authority_registry,
        "get_internal_links",
        lambda homepage: [
            "https://docs.example.org/hpc/index.html",
            "https://docs.example.org/hpc/software-packages/tensorflow",
        ],
    )
    monkeypatch.setattr(
        build_authority_registry,
        "get_service_catalog_links",
        lambda homepage: [
            "https://www.example.org/services",
            "https://www.example.org/secure-research-computing",
        ],
    )
    monkeypatch.setattr(
        build_authority_registry,
        "load_website_docs",
        lambda links: service_html_documents if links and links[0].startswith("https://www.example.org/") else html_documents,
    )
    monkeypatch.setattr(
        build_authority_registry,
        "preprocess_html_documents",
        lambda documents, tags, conditions: documents,
    )
    monkeypatch.setattr(
        build_authority_registry,
        "convert_documents_to_markdown",
        lambda documents, engine, options: (
            service_markdown_documents
            if documents and documents[0].id.startswith("https://www.example.org/")
            else markdown_documents
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_authority_registry.py",
            "-c",
            "config/config.yaml",
            "-p",
            "https://docs.example.org/hpc/index.html",
            "-i",
            "--output-file",
            str(output_file),
            "--review-file",
            str(review_file),
            "--services-homepage",
            "https://www.example.org/services",
        ],
    )

    build_authority_registry.main()

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["build"]["homepage"] == "https://docs.example.org/hpc/index.html"
    assert payload["build"]["docs_homepage"] == "https://docs.example.org/hpc/index.html"
    assert payload["build"]["services_homepage"] == "https://www.example.org/services"
    assert payload["build"]["service_catalog_included"] is True
    assert payload["summary"]["entity_count"] >= 1
    assert payload["summary"]["counts_by_source_scope"]["local_official"] >= 1
    assert payload["summary"]["counts_by_source_scope"]["local_official_services"] >= 1
    assert review_file.exists()

    captured = capsys.readouterr()
    assert "Authority registry complete" in captured.out


def test_main_skip_services_catalog_produces_docs_only_registry(tmp_path, monkeypatch):
    fake_cfg = SimpleNamespace(
        ingestion={"conversion": {"sources": {"docs": {"engine": "markitdown"}}}},
        document_preprocess_html_conditions=[],
        document_preprocess_html_tags=[],
    )
    html_documents = [
        Document(
            id="https://docs.example.org/hpc/software-packages/tensorflow",
            document_type="html",
            text="<html><head><title>TensorFlow</title></head><body><h1>TensorFlow</h1></body></html>",
            metadata={"source": "https://docs.example.org/hpc/software-packages/tensorflow"},
        )
    ]
    markdown_documents = [
        MarkdownDocument(
            id="https://docs.example.org/hpc/software-packages/tensorflow",
            document_type="html",
            text="# TensorFlow\n\nSupported version 2.16.1.",
            metadata={"source": "https://docs.example.org/hpc/software-packages/tensorflow", "title": "TensorFlow"},
        )
    ]
    output_file = tmp_path / "registry.json"
    review_file = tmp_path / "review.csv"

    monkeypatch.setattr(build_authority_registry.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(
        build_authority_registry,
        "get_internal_links",
        lambda homepage: [
            "https://docs.example.org/hpc/index.html",
            "https://docs.example.org/hpc/software-packages/tensorflow",
        ],
    )
    monkeypatch.setattr(
        build_authority_registry,
        "get_service_catalog_links",
        lambda homepage: (_ for _ in ()).throw(AssertionError("services crawl should be skipped")),
    )
    monkeypatch.setattr(build_authority_registry, "load_website_docs", lambda links: html_documents)
    monkeypatch.setattr(build_authority_registry, "preprocess_html_documents", lambda documents, tags, conditions: documents)
    monkeypatch.setattr(build_authority_registry, "convert_documents_to_markdown", lambda documents, engine, options: markdown_documents)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_authority_registry.py",
            "-c",
            "config/config.yaml",
            "-p",
            "https://docs.example.org/hpc/index.html",
            "-i",
            "--skip-services-catalog",
            "--output-file",
            str(output_file),
            "--review-file",
            str(review_file),
        ],
    )

    build_authority_registry.main()

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["build"]["service_catalog_included"] is False
    assert payload["summary"]["counts_by_source_scope"] == {"local_official": payload["summary"]["entity_count"]}


def test_get_internal_links_recurses_over_discovered_pages(monkeypatch):
    graph = {
        "https://docs.example.org/hpc/index.html": [
            "https://docs.example.org/hpc/index.html",
            "https://docs.example.org/hpc/software-packages/index.html",
        ],
        "https://docs.example.org/hpc/software-packages/index.html": [
            "https://docs.example.org/hpc/software-packages/index.html",
            "https://docs.example.org/hpc/software-packages/castep.html",
            "https://docs.example.org/hpc/software-packages/gaussian.html",
        ],
        "https://docs.example.org/hpc/software-packages/castep.html": [
            "https://docs.example.org/hpc/software-packages/castep.html",
        ],
        "https://docs.example.org/hpc/software-packages/gaussian.html": [
            "https://docs.example.org/hpc/software-packages/gaussian.html",
        ],
    }

    monkeypatch.setattr(
        build_authority_registry,
        "_get_internal_links_one_hop",
        lambda url: graph.get(url, [url]),
    )

    links = build_authority_registry.get_internal_links("https://docs.example.org/hpc/index.html")

    assert "https://docs.example.org/hpc/software-packages/castep.html" in links
    assert "https://docs.example.org/hpc/software-packages/gaussian.html" in links


def test_get_internal_links_filters_out_sources_storage_and_static_assets(monkeypatch):
    graph = {
        "https://docs.example.org/hpc/index.html": [
            "https://docs.example.org/hpc/index.html",
            "https://docs.example.org/hpc/software-packages/index.html",
            "https://docs.example.org/hpc/_sources/software-packages/gromacs.rst.txt",
            "https://docs.example.org/storage/rcs/ssh.html",
            "https://docs.example.org/hpc/_images/plot.png",
        ],
        "https://docs.example.org/hpc/software-packages/index.html": [
            "https://docs.example.org/hpc/software-packages/index.html",
            "https://docs.example.org/hpc/software-packages/gromacs.html",
        ],
    }

    monkeypatch.setattr(
        build_authority_registry,
        "_get_internal_links_one_hop",
        lambda url: graph.get(url, [url]),
    )

    links = build_authority_registry.get_internal_links("https://docs.example.org/hpc/index.html")

    assert "https://docs.example.org/hpc/index.html" in links
    assert "https://docs.example.org/hpc/software-packages/gromacs.html" in links
    assert "https://docs.example.org/hpc/_sources/software-packages/gromacs.rst.txt" not in links
    assert "https://docs.example.org/storage/rcs/ssh.html" not in links
    assert "https://docs.example.org/hpc/_images/plot.png" not in links


def test_get_service_catalog_links_filters_navigation_pages(monkeypatch):
    graph = {
        "https://www.example.org/services": [
            "https://www.example.org/services",
            "https://www.example.org/high-performance-computing",
            "https://www.example.org/data-storage",
            "https://www.example.org/service-charges",
            "https://www.example.org/about",
            "https://www.example.org/documentation",
        ],
        "https://www.example.org/high-performance-computing": [
            "https://www.example.org/high-performance-computing",
            "https://www.example.org/d-w-n",
        ],
        "https://www.example.org/data-storage": [
            "https://www.example.org/data-storage",
            "https://www.example.org/research-file-share",
            "https://www.example.org/research-data-store",
        ],
    }

    monkeypatch.setattr(
        build_authority_registry,
        "_get_internal_links_one_hop",
        lambda url: graph.get(url, [url]),
    )

    links = build_authority_registry.get_service_catalog_links("https://www.example.org/services")

    assert "https://www.example.org/services" in links
    assert "https://www.example.org/high-performance-computing" in links
    assert "https://www.example.org/data-storage" in links
    assert "https://www.example.org/research-file-share" in links
    assert "https://www.example.org/research-data-store" in links
    assert "https://www.example.org/service-charges" not in links
    assert "https://www.example.org/about" not in links
    assert "https://www.example.org/documentation" not in links
