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

from polaris_rag.cli import build_external_authority_registry
from polaris_rag.common import MarkdownDocument


def test_parse_args_supports_external_registry_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_external_authority_registry.py",
            "-c",
            "config/config.yaml",
            "--source-register-file",
            "data/authority/source_register.external_v1.yaml",
            "--local-registry-file",
            "data/authority/registry.local_official.v1.json",
            "--external-output-file",
            "data/authority/registry.external.json",
            "--external-review-file",
            "data/authority/review.external.csv",
            "--combined-output-file",
            "data/authority/registry.combined.json",
            "--combined-review-file",
            "data/authority/review.combined.csv",
        ],
    )

    args = build_external_authority_registry.parse_args()

    assert args.source_register_file.endswith("source_register.external_v1.yaml")
    assert args.local_registry_file.endswith("registry.local_official.v1.json")
    assert args.external_output_file.endswith("registry.external.json")
    assert args.combined_review_file.endswith("review.combined.csv")


def test_main_writes_external_and_combined_registry_outputs(tmp_path, monkeypatch, capsys):
    fake_cfg = SimpleNamespace(
        ingestion={"conversion": {"sources": {"external_docs": {"engine": "markitdown"}}}},
        document_preprocess_html_conditions=[],
        document_preprocess_html_tags=[],
    )
    register = SimpleNamespace(
        sources=[
            SimpleNamespace(
                source_id="gromacs",
                homepage="https://manual.example.org/gromacs/index.html",
            )
        ]
    )
    markdown_documents = [
        MarkdownDocument(
            id="https://manual.example.org/gromacs/install.html",
            document_type="html",
            text="# Installing GROMACS\n\nSupported version 2025.1.\n\n```bash\nmodule load gromacs/2025.1\n```",
            metadata={
                "source": "https://manual.example.org/gromacs/install.html",
                "title": "Installing GROMACS",
                "source_register_entity_type": "software",
                "source_register_canonical_name": "GROMACS",
                "source_register_aliases": ["gromacs", "GROMACS"],
            },
        )
    ]
    source_documents = [
        build_external_authority_registry.RegistrySourceDocument(
            url="https://manual.example.org/gromacs/install.html",
            source_scope="external_official",
            source_id="gromacs",
        )
    ]
    local_registry_path = tmp_path / "registry.local.json"
    local_registry_path.write_text(
        json.dumps(
            {
                "build": {
                    "homepage": "https://docs.example.org/hpc/index.html",
                    "source_scope": "local_official",
                    "extraction_version": "authority_registry_v1",
                },
                "source_urls": ["https://docs.example.org/hpc/gromacs.html"],
                "source_documents": [
                    {
                        "url": "https://docs.example.org/hpc/gromacs.html",
                        "source_scope": "local_official",
                        "source_id": "local_docs",
                    }
                ],
                "entities": [
                    {
                        "entity_id": "software-gromacs-local",
                        "entity_type": "software",
                        "canonical_name": "GROMACS",
                        "aliases": ["GROMACS", "gromacs"],
                        "source_scope": "local_official",
                        "status": "current",
                        "known_versions": ["2024.4"],
                        "doc_id": "https://docs.example.org/hpc/gromacs.html",
                        "doc_title": "GROMACS",
                        "heading_path": ["GROMACS"],
                        "evidence_spans": [],
                        "extraction_method": "test_fixture",
                        "review_state": "auto_verified",
                    }
                ],
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    external_output_file = tmp_path / "registry.external.json"
    external_review_file = tmp_path / "review.external.csv"
    combined_output_file = tmp_path / "registry.combined.json"
    combined_review_file = tmp_path / "review.combined.csv"

    monkeypatch.setattr(build_external_authority_registry.GlobalConfig, "load", lambda path: fake_cfg)
    monkeypatch.setattr(
        build_external_authority_registry,
        "_load_external_markdown_documents",
        lambda cfg, register_path: (register, markdown_documents, source_documents),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_external_authority_registry.py",
            "-c",
            "config/config.yaml",
            "--source-register-file",
            str(tmp_path / "source_register.yaml"),
            "--local-registry-file",
            str(local_registry_path),
            "--external-output-file",
            str(external_output_file),
            "--external-review-file",
            str(external_review_file),
            "--combined-output-file",
            str(combined_output_file),
            "--combined-review-file",
            str(combined_review_file),
        ],
    )

    build_external_authority_registry.main()

    external_payload = json.loads(external_output_file.read_text(encoding="utf-8"))
    combined_payload = json.loads(combined_output_file.read_text(encoding="utf-8"))

    assert external_payload["build"]["source_count"] == 1
    assert external_payload["summary"]["counts_by_source_scope"]["external_official"] >= 1
    assert combined_payload["summary"]["counts_by_source_scope"]["external_official"] >= 1
    assert combined_payload["summary"]["counts_by_source_scope"]["local_official"] >= 1
    assert external_review_file.exists()
    assert combined_review_file.exists()

    captured = capsys.readouterr()
    assert "External authority registry complete" in captured.out
