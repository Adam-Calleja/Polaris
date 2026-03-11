from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest
import yaml


_MODULE_PATH = Path(__file__).resolve().parents[2] / "config" / "global_config.py"
_SPEC = spec_from_file_location("test_global_config_module", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
GlobalConfig = _MODULE.GlobalConfig


def test_vector_stores_uses_explicit_multi_store_mapping() -> None:
    cfg = GlobalConfig(
        {
            "generator_llm": {},
            "evaluator_llm": {},
            "embedder": {},
            "vector_stores": {
                "docs": {"collection_name": "support_docs"},
                "tickets": {"collection_name": "support_tickets"},
            },
        }
    )

    assert cfg.vector_stores == {
        "docs": {"collection_name": "support_docs"},
        "tickets": {"collection_name": "support_tickets"},
    }
    assert cfg.vector_store == {"collection_name": "support_docs"}


def test_vector_stores_requires_explicit_mapping() -> None:
    cfg = GlobalConfig(
        {
            "generator_llm": {},
            "evaluator_llm": {},
            "embedder": {},
        }
    )

    with pytest.raises(KeyError, match="vector_stores"):
        _ = cfg.vector_stores


def test_vector_store_requires_docs_source() -> None:
    cfg = GlobalConfig(
        {
            "generator_llm": {},
            "evaluator_llm": {},
            "embedder": {},
            "vector_stores": {
                "tickets": {"collection_name": "support_tickets"},
            },
        }
    )

    with pytest.raises(KeyError, match="vector_stores.docs"):
        _ = cfg.vector_store


def test_load_supports_extends_and_deep_merges(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("POLARIS_LLM_API_KEY", "test-token")

    base_cfg = tmp_path / "config.base.yaml"
    base_cfg.write_text(
        yaml.safe_dump(
            {
                "generator_llm": {
                    "type": "openai_like",
                    "model_name": "base-generator",
                },
                "evaluator_llm": {
                    "type": "OpenAILike",
                    "model_name": "base-evaluator",
                },
                "embedder": {
                    "type": "OpenAILike",
                    "api_base": "${POLARIS_LLM_API_KEY}",
                    "timeout": 300,
                },
                "vector_stores": {
                    "docs": {
                        "type": "qdrant",
                        "host": "qdrant",
                        "port": 6333,
                        "collection_name": "support_docs",
                    },
                    "tickets": {
                        "type": "qdrant",
                        "host": "qdrant",
                        "port": 6333,
                        "collection_name": "support_tickets",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    local_cfg = tmp_path / "config.local.yaml"
    local_cfg.write_text(
        yaml.safe_dump(
            {
                "extends": "config.base.yaml",
                "generator_llm": {
                    "model_name": "local-generator",
                },
                "vector_stores": {
                    "docs": {
                        "collection_name": "support_docs_local",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = GlobalConfig.load(local_cfg)

    assert cfg.generator_llm == {
        "type": "openai_like",
        "model_name": "local-generator",
    }
    assert cfg.embedder["api_base"] == "test-token"
    assert cfg.vector_stores["docs"] == {
        "type": "qdrant",
        "host": "qdrant",
        "port": 6333,
        "collection_name": "support_docs_local",
    }
    assert cfg.vector_stores["tickets"]["collection_name"] == "support_tickets"
