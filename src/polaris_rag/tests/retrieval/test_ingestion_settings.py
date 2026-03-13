from types import SimpleNamespace

from polaris_rag.retrieval.ingestion_settings import (
    MARKDOWN_TOKEN_CHUNKING_STRATEGY,
    MARKITDOWN_CONVERSION_ENGINE,
    NATIVE_JIRA_CONVERSION_ENGINE,
    resolve_chunking_settings,
    resolve_conversion_settings,
)


def test_resolve_chunking_settings_prefers_cli_overrides():
    cfg = SimpleNamespace(
        ingestion={
            "chunking": {
                "default": {"strategy": "markdown_token", "chunk_size_tokens": 400, "overlap_tokens": 40},
                "sources": {"docs": {"chunk_size_tokens": 600}},
            }
        }
    )

    settings = resolve_chunking_settings(
        cfg,
        source="docs",
        strategy_override="html_hierarchical",
        chunk_size_override=256,
        overlap_override=16,
    )

    assert settings.strategy == "html_hierarchical"
    assert settings.chunk_size_tokens == 256
    assert settings.overlap_tokens == 16


def test_resolve_chunking_settings_uses_markdown_defaults_when_unset():
    cfg = SimpleNamespace(ingestion={})

    settings = resolve_chunking_settings(cfg, source="tickets")

    assert settings.strategy == MARKDOWN_TOKEN_CHUNKING_STRATEGY
    assert settings.chunk_size_tokens == 800
    assert settings.overlap_tokens == 80


def test_resolve_conversion_settings_defaults_by_source():
    cfg = SimpleNamespace(ingestion={})

    docs = resolve_conversion_settings(cfg, source="docs")
    tickets = resolve_conversion_settings(cfg, source="tickets")

    assert docs.engine == MARKITDOWN_CONVERSION_ENGINE
    assert tickets.engine == NATIVE_JIRA_CONVERSION_ENGINE


def test_resolve_conversion_settings_merges_default_and_source_options():
    cfg = SimpleNamespace(
        ingestion={
            "conversion": {
                "default": {"engine": "markitdown", "options": {"foo": "bar", "shared": "default"}},
                "sources": {"docs": {"options": {"shared": "docs"}}},
            }
        }
    )

    settings = resolve_conversion_settings(cfg, source="docs")

    assert settings.engine == "markitdown"
    assert settings.options == {"foo": "bar", "shared": "docs"}
