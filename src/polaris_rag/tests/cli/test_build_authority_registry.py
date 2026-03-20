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
        ],
    )

    args = build_authority_registry.parse_args()

    assert args.config_file == "config/config.yaml"
    assert args.homepage == "https://docs.example.org/hpc/index.html"
    assert args.ingest_internal_links is True
    assert args.output_file == "data/authority/registry.json"
    assert args.review_file == "data/authority/review.csv"
    assert args.source_scope == "local_official"


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
    monkeypatch.setattr(build_authority_registry, "load_website_docs", lambda links: html_documents)
    monkeypatch.setattr(
        build_authority_registry,
        "preprocess_html_documents",
        lambda documents, tags, conditions: documents,
    )
    monkeypatch.setattr(
        build_authority_registry,
        "convert_documents_to_markdown",
        lambda documents, engine, options: markdown_documents,
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
        ],
    )

    build_authority_registry.main()

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["build"]["homepage"] == "https://docs.example.org/hpc/index.html"
    assert payload["summary"]["entity_count"] >= 1
    assert review_file.exists()

    captured = capsys.readouterr()
    assert "Authority registry complete" in captured.out


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
